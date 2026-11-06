"""Check that each task works on this system, using its reference solution.

Run on a compute node, after `tbench_images.py load --allow-missing`:

    python scripts/check_tbench_oracle.py --tasks $RSCHED_DATA/terminal_easy/tasks.json

A task is "ok" if its tests fail in an untouched container and pass after the
reference solution. Results go to oracle_status.json next to tasks.json, and
re-running skips finished tasks. Pass that file to prepare_terminal_bench --only
to keep the tasks that work.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

os.environ.setdefault("RSCHED_SANDBOX_BACKEND", "podman")


async def run_solution(sandbox, task) -> str | None:
    """Run the reference solution. Returns an error message, or None."""
    import yaml

    script = os.path.join(task.task_dir, "solution.sh")
    if os.path.exists(script):
        await sandbox.run_command("mkdir -p /oracle")
        await sandbox.copy_in(script, "/oracle/solution.sh")
        commands = ["bash /oracle/solution.sh"]
    else:
        with open(os.path.join(task.task_dir, "solution.yaml")) as fh:
            commands = [step["command"] for step in yaml.safe_load(fh)]
    for command in commands:
        code, out = await sandbox.run_command(command, timeout=task.agent_timeout)
        if code == -9:
            return f"timed out: {command[:200]}"
    return None


async def check_task(task, sandbox_cls) -> dict:
    from tvclient import TestToolCall

    t0 = time.time()
    name = f"term|{task.task_id}"
    untouched = sandbox_cls(task_name=name)
    try:
        await untouched.start()
        before = await untouched.execute(TestToolCall(""))
    finally:
        await untouched.stop()
    solved = sandbox_cls(task_name=name)
    try:
        await solved.start()
        error = await run_solution(solved, task)
        after = await solved.execute(TestToolCall("")) if error is None else error
    finally:
        await solved.stop()
    if before == "PASS":
        status = "passes_untouched"
    elif after == "PASS":
        status = "ok"
    else:
        status = "solution_failed"
    return {"status": status, "seconds": time.time() - t0, "detail": "" if status == "ok" else after[-1500:]}


async def main_async(args: argparse.Namespace) -> None:
    os.environ["RSCHED_TERMINAL_TASKS"] = args.tasks
    from rsched.workloads.terminal import TerminalWorkload
    from rsched.workloads.terminal_tasks import load_tasks

    with open(args.tasks) as fh:
        names = [t["task_id"] for t in json.load(fh)]
    all_tasks = load_tasks()
    sandbox_cls = TerminalWorkload().sandbox_cls
    status_path = os.path.join(os.path.dirname(os.path.abspath(args.tasks)), "oracle_status.json")
    status = {}
    if os.path.exists(status_path):
        with open(status_path) as fh:
            status = json.load(fh)

    images_dir = args.images_dir
    if not images_dir:
        images_dir = os.path.join(os.environ["RSCHED_DATA"], "terminal_bench1", "images")
    limit = asyncio.Semaphore(args.jobs)
    lock = asyncio.Lock()

    async def one(name: str) -> None:
        if name in status:
            return
        if not os.path.exists(os.path.join(images_dir, f"{name}.tar")):
            result = {"status": "no_image", "seconds": 0.0, "detail": ""}
        else:
            async with limit:
                try:
                    result = await check_task(all_tasks[name], sandbox_cls)
                except Exception as e:
                    result = {"status": "error", "seconds": 0.0, "detail": f"{type(e).__name__}: {e}"[-1500:]}
        async with lock:
            status[name] = result
            with open(status_path + ".tmp", "w") as fh:
                json.dump(status, fh, indent=1)
            os.replace(status_path + ".tmp", status_path)
            print(f"{name}: {result['status']} ({result['seconds']:.0f} s)", flush=True)

    await asyncio.gather(*(one(name) for name in names))
    counts: dict[str, int] = {}
    for name in names:
        counts[status[name]["status"]] = counts.get(status[name]["status"], 0) + 1
    print(f"{len(names)} tasks: {counts}. Details: {status_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--images-dir")
    ap.add_argument("--jobs", type=int, default=8, help="tasks checked at once")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
