"""Build veRL rows and a task list from Terminal-Bench 1 (original-tasks/).

    python -m rsched.data.prepare_terminal_bench --tasks-dir $TB1/original-tasks \
        --difficulty easy --out $RSCHED_DATA/terminal_easy

Run once per difficulty, so each group is a separate dataset. Writes tasks.json
(load it with RSCHED_TERMINAL_TASKS) and train/validation parquet files. Only
single-service tasks are kept for now; the rest are reported as skipped.
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import yaml

from rsched.data.prepare_terminal import make_row

# Keys every generated compose file has. Anything else means the task expects
# container settings that the sandbox has to reproduce, so it gets reported.
STANDARD_COMPOSE_KEYS = {"build", "image", "container_name", "command", "environment", "volumes", "platform"}


def load_task(task_dir: str, image_tag: str) -> tuple[dict | None, str | None, list[str]]:
    """Return (task or None, skip reason or None, non-standard compose keys)."""
    with open(os.path.join(task_dir, "task.yaml")) as fh:
        meta = yaml.safe_load(fh)
    with open(os.path.join(task_dir, "docker-compose.yaml")) as fh:
        services = yaml.safe_load(fh)["services"]
    if len(services) != 1:
        return None, "multi_service", []
    service = next(iter(services.values()))
    build = service.get("build") or {}
    context = build.get("context", ".")
    dockerfile = build.get("dockerfile", "Dockerfile")
    if not os.path.exists(os.path.join(task_dir, context, dockerfile)):
        return None, "no_dockerfile", []
    task_id = os.path.basename(task_dir)
    task = {
        "task_id": task_id,
        "instruction": meta["instruction"],
        "image": f"tb1/{task_id}:{image_tag}",
        "task_dir": os.path.abspath(task_dir),
        "build_context": context,
        "dockerfile": dockerfile,
        "difficulty": meta["difficulty"],
    }
    return task, None, sorted(set(service) - STANDARD_COMPOSE_KEYS)


def collect(tasks_dir: str, difficulty: str, image_tag: str) -> tuple[list[dict], dict[str, list[str]], dict]:
    tasks, skipped, extras = [], {}, {}
    for name in sorted(os.listdir(tasks_dir)):
        task_dir = os.path.join(tasks_dir, name)
        yaml_path = os.path.join(task_dir, "task.yaml")
        if not os.path.exists(yaml_path):
            continue
        with open(yaml_path) as fh:
            if yaml.safe_load(fh).get("difficulty") != difficulty:
                continue
        task, reason, extra = load_task(task_dir, image_tag)
        if task is None:
            skipped.setdefault(reason, []).append(name)
            continue
        tasks.append(task)
        if extra:
            extras[name] = extra
    return tasks, skipped, extras


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", required=True, help="the original-tasks directory of Terminal-Bench 1")
    ap.add_argument("--difficulty", required=True, choices=["easy", "medium"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-tag", default="d28711d", help="tag of the images built from these tasks")
    ap.add_argument("--val-tasks", type=int, default=2, help="last N tasks go to validation")
    ap.add_argument("--prompt-len", type=int, default=4096, help="PROMPT_LEN in launch/run.sh, in tokens")
    args = ap.parse_args()

    tasks, skipped, extras = collect(args.tasks_dir, args.difficulty, args.image_tag)
    if len(tasks) <= args.val_tasks:
        raise SystemExit(f"only {len(tasks)} {args.difficulty} tasks under {args.tasks_dir}")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "tasks.json"), "w") as fh:
        json.dump(tasks, fh, indent=1)

    train, val = tasks[: -args.val_tasks], tasks[-args.val_tasks :]
    for split, ts in (("train", train), ("validation", val)):
        recs = [
            make_row(t["task_id"], t["instruction"], split, i, difficulty=t["difficulty"]) for i, t in enumerate(ts)
        ]
        pd.DataFrame(recs).to_parquet(os.path.join(args.out, f"{split}.parquet"))

    total = len(tasks) + sum(len(names) for names in skipped.values())
    print(f"{args.difficulty}: {total} tasks, kept {len(tasks)} ({len(train)} train, {len(val)} validation)")
    for reason, names in sorted(skipped.items()):
        print(f"  skipped {len(names)} ({reason}): {', '.join(names)}")
    for name, keys in sorted(extras.items()):
        print(f"  non-standard compose settings: {name}: {', '.join(keys)}")
    # veRL drops overlong prompts, so flag likely ones. About 4 characters per token.
    long_tasks = [t["task_id"] for t in tasks if len(t["instruction"]) / 4 > args.prompt_len]
    if long_tasks:
        print(f"  may exceed PROMPT_LEN={args.prompt_len}: {', '.join(long_tasks)}")


if __name__ == "__main__":
    main()
