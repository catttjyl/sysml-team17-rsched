"""Convert SkyRL-SQL-653 parquet files into veRL rows for TVCacheToolAgentLoop.

    hf download NovaSky-AI/SkyRL-SQL-653-data-newfmt --repo-type dataset --local-dir $RSCHED_DATA/skyrl_sql
    python -m rsched.data.prepare_sql --src $RSCHED_DATA/skyrl_sql --db-root $RSCHED_DATA/sql_db \
        --out $RSCHED_DATA/sql

The converter is defensive about column names because SkyRL has shipped a few
formats: it needs db_id, the source ("data": synsql/spider/bird), the gold SQL
(reward_spec.ground_truth or similar) and a user prompt it can parse the
schema + question out of. Rows whose .sqlite file is missing are dropped with
a count, so you can run it on a partial DB download.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

import pandas as pd

from rsched.data.sql_prompt import build_row, parse_skyrl_prompt
from rsched.workloads import sql as sqlw


def _get(row: dict, *names: str) -> Any:
    for n in names:
        cur: Any = row
        ok = True
        for part in n.split("."):
            if isinstance(cur, str):
                try:
                    cur = json.loads(cur)
                except Exception:
                    ok = False
                    break
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur is not None:
            return cur
    return None


def _user_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    msgs = list(prompt)
    for m in msgs:
        if m.get("role") == "user":
            return m["content"]
    return msgs[-1]["content"]


def convert(df: pd.DataFrame, split: str, max_turns: int) -> tuple[list[dict], dict]:
    out, stats = [], {"rows": len(df), "kept": 0, "no_db": 0, "parse_fail": 0}
    for i, row in enumerate(df.to_dict("records")):
        db_id = _get(row, "db_id", "extra_info.db_id", "env_extras.db_id")
        source = _get(row, "data", "extra_info.data", "env_extras.data") or "synsql"
        gold = _get(row, "reward_spec.ground_truth", "reward_model.ground_truth", "extra_info.gold_sql", "sql")
        # Newer SkyRL files have schema / question / external_knowledge as columns.
        # Older ones only have the prompt, so fall back to parsing it.
        schema, question = row.get("schema"), row.get("question")
        knowledge = row.get("external_knowledge") or ""
        if not schema or not question:
            try:
                schema, knowledge, question = parse_skyrl_prompt(_user_text(row["prompt"]))
            except Exception:
                stats["parse_fail"] += 1
                continue
        if not os.path.exists(sqlw.db_file_for(source, db_id)):
            stats["no_db"] += 1
            continue
        out.append(
            build_row(source=source, db_id=db_id, idx=i, schema=schema, question=question, gold_sql=gold,
                      knowledge=knowledge, split=split, max_turns=max_turns)
        )
        stats["kept"] += 1
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir with SkyRL parquet files")
    ap.add_argument("--db-root", required=True, help="dir containing SynSQL-2.5M/, spider/, bird/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-turns", type=int, default=6)
    args = ap.parse_args()
    sqlw.set_db_root(args.db_root)
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.src, "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no parquet files under {args.src}")
    for f in files:
        split = "validation" if ("val" in os.path.basename(f) or "test" in os.path.basename(f)) else "train"
        df = pd.read_parquet(f)
        print(f"{f}: columns={list(df.columns)}")
        recs, stats = convert(df, split, args.max_turns)
        print(f"  {split}: {stats}")
        if recs:
            pd.DataFrame(recs).to_parquet(os.path.join(args.out, f"{split}.parquet"))


if __name__ == "__main__":
    main()
