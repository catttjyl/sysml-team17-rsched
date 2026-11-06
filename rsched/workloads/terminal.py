"""Terminal (shell) workload.

The agent calls ``bash(command)``. Every command is treated as state-mutating
(the safe default for shell tools, as in the TVCache paper), so the cache key
is the full command history -- the TCG is a real tree here and forking matters.

Sandbox backend for bring-up: a plain directory per sandbox.
  * fresh start  -> new dir, write task files, run task setup
  * fork()       -> copy live dir to a snapshot dir (cold handle)
  * start(handle)-> copy snapshot to a new live dir
  * execute      -> ``bash -c`` with cwd = live dir, timeout, scrubbed env
This is NOT a security boundary (commands can touch the rest of the
filesystem). Set ``RSCHED_SANDBOX_ISOLATION=bwrap`` to run commands under
bubblewrap with the sandbox bound at /workspace, if bwrap is available.
Terminal-Bench 1 tasks run in containers instead: see terminal_podman.py.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import uuid
from typing import Any, List, Optional

from tvclient import Sandbox, TestToolCall, ToolCall

from rsched.workloads.base import Workload
from rsched.workloads.terminal_tasks import TerminalTask, load_tasks

SANDBOX_ROOT = os.environ.get(
    "RSCHED_SANDBOX_ROOT", os.path.join(os.environ.get("TMPDIR", "/tmp"), f"rsched-sbx-{os.getuid()}")
)
CMD_TIMEOUT_S = float(os.environ.get("RSCHED_CMD_TIMEOUT", "30"))
MAX_OUTPUT_CHARS = 8000
ISOLATION = os.environ.get("RSCHED_SANDBOX_ISOLATION", "none")

_TASKS: Optional[dict[str, TerminalTask]] = None


def get_task(task_name: str) -> TerminalTask:
    global _TASKS
    if _TASKS is None:
        _TASKS = load_tasks()
    tid = task_name.split("|", 1)[1] if task_name.startswith("term|") else task_name
    return _TASKS[tid]


def _new_dir(kind: str) -> str:
    path = os.path.join(SANDBOX_ROOT, kind, uuid.uuid4().hex)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _copy(src: str, dst: str) -> None:
    shutil.copytree(src, dst, symlinks=True)


async def run_argv(argv: List[str], timeout: float, **kwargs: Any) -> tuple[int, str]:
    """Run a command and return (exit code, full output). A timeout returns code -9."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
        **kwargs,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        try:
            os.killpg(proc.pid, 9)
        except ProcessLookupError:
            pass
        await proc.wait()
        return -9, "Failed to execute: command timed out"
    return proc.returncode, out.decode("utf-8", errors="replace")


def truncate_output(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        half = MAX_OUTPUT_CHARS // 2
        return text[:half] + "\n...(output truncated)...\n" + text[-half:]
    return text


def format_result(code: int, out: str) -> str:
    return f"{out}\n[exit code: {code}]" if out else f"[exit code: {code}]"


async def run_bash(cwd: str, command: str, timeout: float = CMD_TIMEOUT_S) -> tuple[int, str]:
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": cwd,
        "LANG": "C.UTF-8",
        "TERM": "dumb",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if ISOLATION == "bwrap" and shutil.which("bwrap"):
        argv = [
            "bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
            "--bind", cwd, "/workspace", "--chdir", "/workspace", "--unshare-all", "--die-with-parent",
            "--setenv", "HOME", "/workspace", "bash", "-c", command,
        ]  # fmt: skip
        cwd_arg = None
    else:
        argv = ["bash", "-c", command]
        cwd_arg = cwd
    code, text = await run_argv(argv, timeout, cwd=cwd_arg, env=env)
    return code, truncate_output(text)


class BashToolCall(ToolCall):
    def __init__(self, command: str):
        self.command = command.strip()

    def to_dict(self) -> dict:
        return {"tool": "bash", "command": self.command}

    def will_mutate_state(self) -> bool:
        return True


class LocalDirSandbox(Sandbox):
    async def start(self) -> None:
        live = _new_dir("live")
        if self.env_id is None:
            task = get_task(self.task_name)
            if task.image:
                # Its default test is `true`, so a folder sandbox would score every rollout as a pass.
                raise RuntimeError(f"{task.task_id} needs RSCHED_SANDBOX_BACKEND=podman")
            os.makedirs(live)
            for rel, content in task.files.items():
                p = os.path.join(live, rel)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w") as fh:
                    fh.write(content)
            for rel in task.executable:
                os.chmod(os.path.join(live, rel), 0o755)
            if task.setup:
                code, out = await run_bash(live, task.setup, timeout=120)
                if code != 0:
                    raise RuntimeError(f"setup failed for {task.task_id}: {out}")
        else:
            await asyncio.to_thread(_copy, self.env_id, live)
        self.env_id = live

    async def fork(self) -> "LocalDirSandbox":
        snap = _new_dir("snap")
        await asyncio.to_thread(_copy, self.env_id, snap)
        return LocalDirSandbox(task_name=self.task_name, env_id=snap)

    async def execute(self, tool_call: ToolCall) -> str:
        if isinstance(tool_call, TestToolCall):
            code, out = await run_bash(self.env_id, get_task(self.task_name).test, timeout=120)
            return "PASS" if code == 0 else f"FAIL (exit {code})\n{out[-2000:]}"
        assert isinstance(tool_call, BashToolCall)
        code, out = await run_bash(self.env_id, tool_call.command)
        if code == -9:
            return out
        return format_result(code, out)

    async def stop(self) -> None:
        if self.env_id and self.env_id.startswith(SANDBOX_ROOT):
            await asyncio.to_thread(shutil.rmtree, self.env_id, True)


class TerminalWorkload(Workload):
    name = "terminal"
    tool_names = ("bash",)
    sandbox_cls = LocalDirSandbox
    tool_call_cls = BashToolCall

    def __init__(self, **config: Any):
        super().__init__(**config)
        if os.environ.get("RSCHED_SANDBOX_BACKEND") == "podman":
            from rsched.workloads.terminal_podman import PodmanSandbox

            self.sandbox_cls = PodmanSandbox

    def task_id(self, row_kwargs: dict) -> str:
        info = row_kwargs.get("extra_info") or {}
        return info["tvcache_task_id"]

    def make_tool_call(self, tool_name: str, args: dict) -> ToolCall:
        return BashToolCall(str(args.get("command", "")))

    async def score(
        self, task_id: str, executor: Any, history: List[ToolCall], messages: list, row_kwargs: dict
    ) -> float:
        # Goes through the cache: the test result is keyed by the stateful chain.
        result = await executor.test(history)
        return 1.0 if result.startswith("PASS") else 0.0
