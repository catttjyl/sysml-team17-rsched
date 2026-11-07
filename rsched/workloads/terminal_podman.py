"""Container sandbox for Terminal-Bench 1 tasks, using podman-hpc.

Set RSCHED_SANDBOX_BACKEND=podman to use it. A sandbox's env_id is either:
  * a live container, named rsched-<id>
  * a snapshot image, localhost/rsched-snap:<id>, made by fork()

Images must be in the node's writable storage. On Perlmutter, `commit` fails for
containers started from migrated (squashed) images, so jobs load saved images
with scripts/tbench_images.py instead.

Forks keep files, the working directory and exported variables, but not running
processes, because `commit` only saves the filesystem.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from typing import Optional

from tvclient import Sandbox, TestToolCall, ToolCall

from rsched.workloads.terminal import (
    BashToolCall,
    CMD_TIMEOUT_S,
    format_result,
    get_task,
    run_argv,
    truncate_output,
)

PODMAN = "podman-hpc"
CONTAINER_PREFIX = "rsched-"
SNAPSHOT_REPO = "localhost/rsched-snap"
TEST_DIR = "/tests"  # CONTAINER_TEST_DIR in the Terminal-Bench 1 harness
STATE_FILE = "/var/tmp/.rsched_shell_state"
PODMAN_TIMEOUT_S = 300.0
# Concurrent `run` calls can fail with "pasta failed ... netlink: Unexpected
# sequence number", so starts on a node go one at a time, across all workers.
# Always under /tmp: a shared lock can't depend on each process's TMPDIR.
START_LOCK = f"/tmp/rsched-podman-start-{os.getuid()}.lock"
START_ATTEMPTS = 3
MAX_EXEC_OUTPUT_BYTES = 10_000_000  # far above what's kept: see MAX_OUTPUT_CHARS in terminal.py
# Extra time for podman itself, on top of the timeout enforced inside the container.
EXEC_GRACE_S = 30.0
TIMED_OUT = "Failed to execute: command timed out"

# Each call is a new bash process, so restore the previous call's working
# directory and exports, and save them again on exit. The file is inside the
# container, so `commit` carries it into forks.
#
# Output goes through `head -c`, so a command that prints forever (like `yes`)
# gets SIGPIPE at the cap. Otherwise it fills run_argv's temp file, which is in
# memory-backed /tmp, at about 0.8 GB/s. On exit, give head a moment to pass on
# the last output, without waiting for background jobs that still hold the pipe.
SHELL_WRAPPER = f"""
if [ -f {STATE_FILE} ]; then . {STATE_FILE} >/dev/null 2>&1; fi
exec > >(head -c {MAX_EXEC_OUTPUT_BYTES}) 2>&1
_rsched_head=$!
trap '_rsched_rc=$?; {{ export -p; printf "cd %q\\n" "$PWD"; }} > {STATE_FILE} 2>/dev/null; exec >&- 2>&-; for _rsched_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do kill -0 "$_rsched_head" 2>/dev/null || break; sleep 0.01; done; exit $_rsched_rc' EXIT
_rsched_cmd=$1
shift
eval "$_rsched_cmd"
"""

_PYTEST_PASSED = {"PASSED", "XFAIL", "SKIPPED"}
_PYTEST_FAILED = {"FAILED", "XPASS", "ERROR"}


def pytest_passed(output: str) -> bool:
    """Same rule as the Terminal-Bench 1 harness: read pytest's short summary,
    and pass only if no listed test failed. No summary counts as a failure."""
    parts = re.split(r"=+\s*short test summary info\s*=+", output, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) < 2:
        return False
    for line in parts[1].splitlines():
        words = line.split(maxsplit=1)
        if len(words) < 2:
            continue
        status = words[0].strip().strip(":")
        if status in _PYTEST_FAILED:
            return False
    return True


async def podman(*args: str, timeout: float = PODMAN_TIMEOUT_S, prefix: tuple[str, ...] = ()) -> str:
    code, out = await run_argv([*prefix, PODMAN, *args], timeout)
    if code != 0:
        raise RuntimeError(f"{PODMAN} {args[0]} failed (exit {code}): {out[-1000:]}")
    return out


class PodmanSandbox(Sandbox):
    async def start(self) -> None:
        image = self.env_id or get_task(self.task_name).image
        t0 = time.time()
        for attempt in range(1, START_ATTEMPTS + 1):
            name = CONTAINER_PREFIX + uuid.uuid4().hex[:16]
            try:
                await podman(
                    "run", "-d", "--name", name, "--entrypoint=", "--pull=never", "--stop-timeout=0",
                    "--env", f"TEST_DIR={TEST_DIR}", "--env", "TERM=dumb",
                    image, "sleep", "infinity",
                    # -o: don't pass the lock to podman's detached helpers, which would
                    # keep it held for the container's whole life.
                    prefix=("flock", "-o", START_LOCK),
                )  # fmt: skip
                break
            except RuntimeError as e:
                # The lock doesn't prevent every pasta failure. They're quick and
                # transient, but leave a container behind that holds the name.
                if "pasta failed" not in str(e) or attempt == START_ATTEMPTS:
                    raise
                await run_argv([PODMAN, "rm", "-f", name], PODMAN_TIMEOUT_S)
                await asyncio.sleep(attempt)
        self.start_seconds = time.time() - t0  # includes waiting for the lock
        self.env_id = name

    async def fork(self) -> "PodmanSandbox":
        snapshot = f"{SNAPSHOT_REPO}:{uuid.uuid4().hex[:16]}"
        await podman("commit", self.env_id, snapshot)
        return PodmanSandbox(task_name=self.task_name, env_id=snapshot)

    async def run_command(self, command: str, timeout: float = CMD_TIMEOUT_S) -> tuple[int, str]:
        """Run in the shared shell state. Returns (exit code, output); a timeout returns code -9."""
        code, out = await self._exec(timeout, "bash", "-c", SHELL_WRAPPER, "rsched", command)
        if code in (-9, 124, 137):
            return -9, TIMED_OUT
        return code, out

    async def copy_in(self, source: str, dest: str) -> None:
        await podman("cp", source, f"{self.env_id}:{dest}")

    async def execute(self, tool_call: ToolCall) -> str:
        if isinstance(tool_call, TestToolCall):
            return await self._run_tests()
        assert isinstance(tool_call, BashToolCall)
        code, out = await self.run_command(tool_call.command)
        if code == -9:
            return out
        return format_result(code, truncate_output(out))

    async def stop(self) -> None:
        if not self.env_id:
            return
        if self.env_id.startswith(SNAPSHOT_REPO):
            await podman("rmi", self.env_id)
        else:
            await podman("rm", "-f", self.env_id)

    async def _exec(self, timeout: float, *argv: str, user: Optional[str] = None) -> tuple[int, str]:
        # `timeout` runs inside the container, because killing podman exec on
        # the host would leave the command running.
        user_args = ["--user", user] if user else []
        code, out = await run_argv(
            [PODMAN, "exec", *user_args, self.env_id, "timeout", "-k", "5", str(timeout), *argv],
            timeout + EXEC_GRACE_S,
        )
        if code == 125:
            raise RuntimeError(f"{PODMAN} exec failed in {self.env_id}: {out[-1000:]}")
        return code, out

    async def _run_tests(self) -> str:
        task = get_task(self.task_name)
        await podman("exec", "--user", "root", self.env_id, "mkdir", "-p", TEST_DIR)
        tests = os.path.join(task.task_dir, "tests")
        if os.path.isdir(tests):
            await self.copy_in(tests + "/.", TEST_DIR + "/")
        await self.copy_in(os.path.join(task.task_dir, "run-tests.sh"), f"{TEST_DIR}/run-tests.sh")
        # Not in the shared shell state: the harness also runs tests in a new shell.
        code, out = await self._exec(task.test_timeout, "bash", f"{TEST_DIR}/run-tests.sh", user="root")
        if code in (-9, 124, 137):
            return TIMED_OUT
        return "PASS" if pytest_passed(out) else f"FAIL (exit {code})\n{out[-2000:]}"
