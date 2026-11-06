"""Make a one-task Terminal-Bench dataset: $RSCHED_DATA/terminal_one from terminal_easy."""
import json, os, sys
import pandas as pd
d = os.environ["RSCHED_DATA"]
task = sys.argv[1] if len(sys.argv) > 1 else "hello-world"
src, out = f"{d}/terminal_easy", f"{d}/terminal_one"
os.makedirs(out, exist_ok=True)
all_tasks = json.load(open(f"{src}/tasks.json"))
tasks = [t for t in all_tasks if t["task_id"] == task]
if not tasks:
    sys.exit(f"{task} not in terminal_easy. Available: {[t['task_id'] for t in all_tasks]}")
json.dump(tasks, open(f"{out}/tasks.json", "w"), indent=1)
rows = pd.concat([pd.read_parquet(f"{src}/{s}.parquet") for s in ("train", "validation")])
row = rows[rows["extra_info"].map(lambda e: e["tvcache_task_id"].split("|", 1)[1] == task)].head(1)
row.to_parquet(f"{out}/train.parquet")
row.to_parquet(f"{out}/validation.parquet")
print("terminal_one:", task, len(row), "row")
