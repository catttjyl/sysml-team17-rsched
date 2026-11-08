"""Replay recorded runs under several scheduling policies and compare them.

    python scripts/simulate.py results/tbench_full_b4n8/tb-easy_cache-on \
        --policies default hold:5 hold:30 hold:inf scout:30 scout:inf+hold:inf \
        [--json out.json]

Prints, per run, the trace's own numbers (``trace``: what actually happened)
and a markdown table per policy. Tool seconds exclude sandbox start.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rsched.sim import Policy, load_run, simulate, summarize  # noqa: E402

DEFAULT_POLICIES = ["default", "hold:2", "hold:10", "hold:30", "hold:inf", "scout:30", "scout:inf", "scout:30+hold:inf"]


def trace_summary(run) -> dict:
    """The recorded run's own numbers, computed with the simulator's conventions."""
    tools = [s for st in run.steps for r in st.rollouts for s in r.segs if s.kind == "tool"]
    tests = [s for st in run.steps for r in st.rollouts for s in r.segs if s.kind == "score" and s.cacheable]
    mk = [st.t_end - st.t0 for st in run.steps]
    return {
        "tool_hits": sum(s.trace_hit for s in tools),
        "tool_calls": len(tools),
        "test_hits": sum(s.trace_hit for s in tests),
        "test_calls": len(tests),
        "makespan_mean": sum(mk) / len(mk),
    }


def fmt_row(s: dict) -> str:
    t, x = s["tool"], s["test"]
    cells = [
        s["policy"],
        f"{t['hit_rate']:.1%} ({t['hits']}/{t['calls']})",
        str(t["timing"]),
        f"{x['hit_rate']:.1%} ({x['hits']}/{x['calls']})" if x["calls"] else "-",
        f"{t['cmd_s'] + x['cmd_s']:,.0f}",
        f"{t['cmd_s_no_timeout'] + x['cmd_s_no_timeout']:,.0f}",
        f"{s['sandbox_s']:,.0f}",
        f"{t['wait_s'] + x['wait_s']:,.0f}",
        f"{s['scout_delay_s']:,.0f}",
        f"{s['makespan_mean']:.1f}",
        f"{s['straggler_tail_mean']:.1f}",
        f"{s['gen_busy_frac']:.1%}",
    ]
    return "| " + " | ".join(cells) + " |"


HEADER = (
    "| policy | tool hit rate | timing misses | test hit rate | exec s | exec s (no timeouts) | sandbox s "
    "| wait s | scout delay s | step makespan s | straggler tail s | gen-busy |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run folders (each with timeline/)")
    ap.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    ap.add_argument("--step-gap", type=float, default=5.0, help="rollout-start gap (s) that separates steps")
    ap.add_argument("--cmd-timeout", type=float, default=600.0)
    ap.add_argument("--hit-latency", type=float, default=None, help="default: median trace hit, else 5 ms")
    ap.add_argument("--json", help="write all summaries here")
    args = ap.parse_args()

    out = []
    for path in args.runs:
        run = load_run(path, step_gap=args.step_gap, cmd_timeout=args.cmd_timeout, hit_latency=args.hit_latency)
        tr = trace_summary(run)
        print(f"\n### {path}  ({run.workload}, cache {run.cache}, {len(run.steps)} steps, "
              f"hit latency {run.hit_latency * 1e3:.1f} ms)")
        print(f"trace: tool hits {tr['tool_hits']}/{tr['tool_calls']}, test hits {tr['test_hits']}/{tr['test_calls']}, "
              f"step makespan {tr['makespan_mean']:.1f} s\n")
        print(HEADER)
        for p in args.policies:
            s = summarize(simulate(run, Policy.parse(p)))
            s["trace"] = tr
            out.append(s)
            print(fmt_row(s))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
