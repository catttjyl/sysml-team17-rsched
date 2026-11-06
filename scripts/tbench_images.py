"""Build, load and clean up Terminal-Bench 1 images with podman-hpc.

    # login node: build each task and save it as a tar. Safe to re-run: finished
    # tasks are skipped and failed ones are retried.
    python scripts/tbench_images.py build --tasks $RSCHED_DATA/terminal_easy/tasks.json

    # compute node: load saved images into the node's writable storage
    python scripts/tbench_images.py load --tasks $RSCHED_DATA/terminal_easy/tasks.json

    # compute node: remove rollout containers and snapshots
    python scripts/tbench_images.py clean

Images are saved as tars, not migrated, because `commit` fails for containers
started from migrated images on Perlmutter.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

PODMAN = "podman-hpc"


def run(*args: str, cwd: str | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PODMAN, *args], cwd=cwd, timeout=timeout, text=True, errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )  # fmt: skip


def load_tasks(path: str) -> list[dict]:
    with open(path) as fh:
        return json.load(fh)


def default_images_dir() -> str:
    data = os.environ.get("RSCHED_DATA")
    if not data:
        raise SystemExit("set --images-dir, or run `source launch/env.sh` first")
    return os.path.join(data, "terminal_bench1", "images")


def build_one(task: dict, tar: str, timeout: float) -> dict:
    t0 = time.time()
    image = task["image"]
    context = task["build_context"]
    dockerfile = os.path.join(context, task["dockerfile"])
    steps = [
        (("build", "-t", image, "-f", dockerfile, context), task["task_dir"]),
        (("save", "-o", tar + ".tmp", image), None),
    ]
    try:
        for args, cwd in steps:
            result = run(*args, cwd=cwd, timeout=timeout)
            if result.returncode != 0:
                return {"status": "failed", "step": args[0], "seconds": time.time() - t0, "error": result.stdout[-2000:]}
    except subprocess.TimeoutExpired:
        return {"status": "failed", "step": args[0], "seconds": time.time() - t0, "error": f"timed out after {timeout} s"}
    os.replace(tar + ".tmp", tar)
    run("rmi", image)  # keep the login node's storage small; the shared base image stays
    return {"status": "ok", "seconds": time.time() - t0, "tar_mb": os.path.getsize(tar) / 1e6}


def cmd_build(args: argparse.Namespace) -> None:
    images_dir = args.images_dir or default_images_dir()
    os.makedirs(images_dir, exist_ok=True)
    status_path = os.path.join(images_dir, "build_status.json")
    status = {}
    if os.path.exists(status_path):
        with open(status_path) as fh:
            status = json.load(fh)

    for i, task in enumerate(load_tasks(args.tasks)):
        name = task["task_id"]
        tar = os.path.join(images_dir, f"{name}.tar")
        if status.get(name, {}).get("status") == "ok" and os.path.exists(tar):
            continue
        status[name] = build_one(task, tar, args.timeout)
        with open(status_path + ".tmp", "w") as fh:
            json.dump(status, fh, indent=1)
        os.replace(status_path + ".tmp", status_path)
        print(f"[{i + 1}] {name}: {status[name]['status']} ({status[name]['seconds']:.0f} s)", flush=True)

    names = [t["task_id"] for t in load_tasks(args.tasks)]
    done = [status[n] for n in names if status.get(n, {}).get("status") == "ok"]
    failed = [n for n in names if status.get(n, {}).get("status") != "ok"]
    print(f"built {len(done)}/{len(names)}, {sum(s['tar_mb'] for s in done) / 1e3:.1f} GB of tars in {images_dir}")
    if failed:
        print(f"failed: {', '.join(failed)}  (errors are in {status_path})")


def cmd_load(args: argparse.Namespace) -> None:
    images_dir = args.images_dir or default_images_dir()
    missing, loaded = [], 0
    t0 = time.time()
    for task in load_tasks(args.tasks):
        tar = os.path.join(images_dir, f"{task['task_id']}.tar")
        if not os.path.exists(tar):
            missing.append(task["task_id"])
        elif run("image", "exists", task["image"]).returncode != 0:
            result = run("load", "-i", tar)
            if result.returncode != 0:
                raise SystemExit(f"loading {tar} failed: {result.stdout[-2000:]}")
            loaded += 1
    print(f"loaded {loaded} images in {time.time() - t0:.0f} s")
    if missing:
        print(f"no saved image for: {', '.join(missing)}")
        if not args.allow_missing:
            raise SystemExit(1)


def cmd_clean(args: argparse.Namespace) -> None:
    containers = run("ps", "-a", "--filter", "name=^rsched-", "--format", "{{.Names}}").stdout.split()
    if containers:
        run("rm", "-f", *containers)
    listed = run("images", "--format", "{{.Repository}}:{{.Tag}}").stdout.split()
    snapshots = [image for image in listed if image.startswith("localhost/rsched-snap:")]
    if snapshots:
        run("rmi", "-f", *snapshots)
    print(f"removed {len(containers)} containers and {len(snapshots)} snapshots", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--tasks", required=True, help="tasks.json from prepare_terminal_bench")
    build.add_argument("--images-dir")
    build.add_argument("--timeout", type=float, default=3600, help="seconds per step")
    build.set_defaults(func=cmd_build)
    load = sub.add_parser("load")
    load.add_argument("--tasks", required=True)
    load.add_argument("--images-dir")
    load.add_argument("--allow-missing", action="store_true", help="skip tasks with no saved image")
    load.set_defaults(func=cmd_load)
    clean = sub.add_parser("clean")
    clean.set_defaults(func=cmd_clean)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
