"""Poster figure: simulated scheduling policies on the full Terminal-Bench runs.

    python scripts/plot_sim.py [--out results/poster_figures]

Replays the cache-on traces in results/tbench_full_b4n8 with rsched.sim under
each policy and writes fig6_sim_policies.{png,pdf}. Same colors and fonts as
scripts/plot_poster.py: blue is a hit, orange a timing miss.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from plot_poster import GRID, HIT, INK2  # noqa: E402  (also applies rcParams)
from rsched.sim import Policy, load_run, simulate, summarize  # noqa: E402

RUNS = [("tb-easy_cache-on", "TB easy"), ("tb-medium_cache-on", "TB medium")]
POLICIES = [
    ("default", "default\n(veRL today)"),
    ("hold:10", "hold\n≤ 10 s"),
    ("hold:inf", "hold until\nfilled"),
    ("scout:inf", "scout\nfirst"),
]
LIGHT = "#9cc3ee"  # second workload: lighter hit blue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "poster_figures"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = {}
    for folder, label in RUNS:
        run = load_run(str(ROOT / "results" / "tbench_full_b4n8" / folder))
        data[label] = [summarize(simulate(run, Policy.parse(p))) for p, _ in POLICIES]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw={"wspace": 0.28})
    x = range(len(POLICIES))
    w = 0.38
    colors = [HIT, LIGHT]

    def bars(ax, getter, title, ylabel, pct=True, ub=None, fmt="{:.0%}"):
        for j, (label, rows) in enumerate(data.items()):
            vals = [getter(r) for r in rows]
            xs = [i + (j - 0.5) * w for i in x]
            ax.bar(xs, vals, w, color=colors[j], label=label)
            for xi, v in zip(xs, vals):
                ax.text(xi, v, fmt.format(v), ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=11, color=INK2)
            if ub:
                u = ub(rows)
                ax.hlines(u, -0.5 + j * 0.02, len(POLICIES) - 0.5, colors=colors[j], linestyles=(0, (4, 3)), lw=1.6)
        ax.set_xticks(list(x), [p for _, p in POLICIES], fontsize=12)
        ax.margins(y=0.15)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)

    bars(axes[0], lambda r: r["tool"]["hit_rate"], "Tool-call hit rate", "share of tool calls",
         ub=lambda rows: rows[0]["tool"]["upper_bound"], fmt="{:.1%}")
    bars(axes[1], lambda r: r["test"]["hit_rate"], "Reward-test hit rate", "share of rollouts")
    base = {k: v[0]["makespan_mean"] for k, v in data.items()}
    for k, rows in data.items():
        for r in rows:
            r["_dmk"] = r["makespan_mean"] / base[k] - 1
    bars(axes[2], lambda r: r["_dmk"], "Step time vs. default", "change in mean step time", pct=False, fmt="{:+.0%}")
    axes[2].axhline(0, color=INK2, lw=0.8)
    for ax in axes[:2]:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[2].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(color=HIT), Patch(color=LIGHT), Line2D([], [], color=INK2, ls=(0, (4, 3)))]
    fig.legend(handles, ["TB easy", "TB medium", "upper bound (hits + timing misses), from the trace"],
               ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.1), fontsize=13)
    fig.suptitle("Holding a sibling until the in-flight call is filled reaches the hit-rate upper bound at ~1% step time",
                 x=0.01, ha="left", fontsize=17, fontweight="bold", y=1.06)
    fig.text(0.01, 0.98, "Simulated on the full Terminal-Bench traces (Qwen2.5-1.5B, 4 prompts × 8 rollouts × 20 steps, "
             "cache on); tool time excludes sandbox start", ha="left", fontsize=12.5, color=INK2)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig6_sim_policies.{ext}", bbox_inches="tight")
    print(f"wrote {out}/fig6_sim_policies.png")


if __name__ == "__main__":
    main()
