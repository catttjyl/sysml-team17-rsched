"""Summarize a timeline directory: hit rate and *why* calls missed.

    python -m rsched.analysis.timeline_report <timeline_dir> [--csv out.csv]

Every tool call has a cache key (task + stateful prefix + call). For a call
issued at time t that did not hit, look at earlier calls with the same key:

  cold    no earlier call with this key was issued              -> unavoidable
  timing  an earlier call with this key was issued before t but
          had not finished by t (the "filler" was in flight)    -> scheduling-induced
  late    an earlier call with this key had already finished,
          yet this call still missed (uncacheable result,
          eviction, failed insert, ...)                         -> cache-side

``upper bound`` = hit rate if every timing miss had waited for its filler.
For cache-off runs ("direct"), the same classification gives the hit rate the
cache *would* have had: calls whose key had finished earlier count as
would-be hits.

Also reports the breakdown by call position within a rollout (H1: are timing
misses concentrated in the first one or two calls?).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from collections import Counter, defaultdict

from rsched.timeline import load_events


def classify(tool_events: list[dict]) -> list[dict]:
    by_key: dict[str, list[tuple[float, float]]] = defaultdict(list)  # key -> [(t_start, t_end)] sorted by start
    for e in sorted(tool_events, key=lambda e: e["t_start"]):
        by_key[e["key"]].append((e["t_start"], e["t_end"]))
    # earliest completion among calls that *started* strictly before each start time
    out = []
    for e in tool_events:
        prior = by_key[e["key"]]
        i = bisect.bisect_left(prior, (e["t_start"], float("-inf")))
        earlier = prior[:i]
        if e.get("hit"):
            cls = "hit"
        elif not earlier:
            cls = "cold"
        elif min(te for _, te in earlier) <= e["t_start"]:
            cls = "would_hit" if e.get("source") == "direct" else "late"
        else:
            cls = "timing"
        out.append({**e, "class": cls, "n_earlier": len(earlier)})
    return out


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    c = Counter(r["class"] for r in rows)
    hits = c["hit"] + c["would_hit"]
    dur = [r["t_end"] - r["t_start"] for r in rows]
    by_class_time = defaultdict(float)
    for r, d in zip(rows, dur):
        by_class_time[r["class"]] += d
    pos = defaultdict(Counter)
    for r in rows:
        pos[min(r["seq"], 6)][r["class"]] += 1
    return {
        "tool_calls": n,
        "classes": dict(c),
        "hit_rate": hits / n if n else 0.0,
        "timing_miss_rate": c["timing"] / n if n else 0.0,
        "upper_bound_hit_rate": (hits + c["timing"]) / n if n else 0.0,
        "tool_seconds_total": sum(dur),
        "tool_seconds_by_class": dict(by_class_time),
        "by_call_position": {f"{k}{'+' if k == 6 else ''}": dict(v) for k, v in sorted(pos.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("timeline_dir")
    ap.add_argument("--csv", help="write per-call rows with their class")
    args = ap.parse_args()

    events = load_events(args.timeline_dir)
    tools = [e for e in events if e["ev"] == "tool" and e.get("error") is None]
    rollouts = [e for e in events if e["ev"] == "rollout_end"]
    gens = [e for e in events if e["ev"] == "gen_end"]
    rows = classify(tools)

    report = {"rollouts": len(rollouts), "gen_turns": len(gens)}
    for wl in sorted({r["workload"] for r in rows}):
        report[wl] = summarize([r for r in rows if r["workload"] == wl])
    rewards = [r["reward"] for r in rollouts if r.get("reward") is not None]
    if rewards:
        report["mean_reward"] = sum(rewards) / len(rewards)
    if gens:
        report["gen_seconds_total"] = sum(g["t"] - g["t_start"] for g in gens)
    print(json.dumps(report, indent=2))

    if args.csv and rows:
        keys = ["t_start", "t_end", "workload", "task", "uid", "rollout", "seq", "turn", "key", "depth",
                "hit", "source", "executed", "class", "n_earlier"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
