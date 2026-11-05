"""Build veRL parquet rows for the terminal workload.

    python -m rsched.data.prepare_terminal --out $RSCHED_DATA/terminal [--repeat 4]

``--repeat`` duplicates the (small) built-in task list so a training batch is
not one task; each copy keeps the same TVCache task id, so cache entries are
shared across copies exactly as across epochs.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from rsched.workloads.terminal_tasks import load_tasks

SYSTEM = (
    "You are an expert Linux user working in a terminal. Use the `bash` tool to inspect and change files "
    "in the current working directory. Work step by step and verify your result. When the task is complete, "
    "reply with a short summary and do not call any more tools."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--val-tasks", type=int, default=2, help="last N tasks go to validation")
    args = ap.parse_args()

    tasks = list(load_tasks().values())
    val, train = tasks[-args.val_tasks :], tasks[: -args.val_tasks]
    os.makedirs(args.out, exist_ok=True)
    for split, ts, rep in (("train", train, args.repeat), ("validation", val, 1)):
        recs = []
        for r in range(rep):
            for t in ts:
                recs.append(
                    {
                        "data_source": "rsched_terminal",
                        "agent_name": "tvcache_terminal_agent",
                        "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": t.instruction}],
                        "ability": "terminal",
                        "reward_model": {"style": "rule", "ground_truth": ""},
                        "extra_info": {
                            "index": len(recs),
                            "split": split,
                            "task": t.task_id,
                            "tvcache_task_id": f"term|{t.task_id}",
                        },
                    }
                )
        pd.DataFrame(recs).to_parquet(os.path.join(args.out, f"{split}.parquet"))
        print(f"{split}: {len(recs)} rows ({len(ts)} tasks) -> {args.out}/{split}.parquet")


if __name__ == "__main__":
    main()
