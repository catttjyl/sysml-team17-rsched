"""Poster figure: simulated scheduling policies on the full Terminal-Bench runs.

    python scripts/plot_sim.py [--out results/poster_figures]

Replays the cache-on traces in results/tbench_full_b4n8 with rsched.sim under
each policy and writes fig6_sim_policies.{png,pdf} and the one-column
fig6_sim_policies_column.{png,pdf}. Same colors and fonts as
scripts/plot_poster.py.

* Hit-rate panels: the dashed line is the reuse bound (every repeat of a key
  hits); waiting with no bound reaches it by construction.
* Time panel: total rollout time over the 20 steps against default, a paired
  comparison on the same trace. Bar = sandbox-lock queue model, diamond =
  recorded sandbox start times. Neither is validated for non-default
  schedules. A 10 s hold stays within about +-2% under both; holding with no
  limit costs +12-16% on TB medium under both (siblings wait out 600 s
  timeouts); scout's sign depends on the model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

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
    ("scout:30", "scout\n≤ 30 s"),
]
COLUMN_POLICIES = [("default", "default"), ("hold:10", "hold\n≤ 10 s"), ("hold:inf", "hold\nno limit")]
LIGHT = "#9cc3ee"  # second workload: lighter hit blue
COLORS = [HIT, LIGHT]


def load(policies, sandbox="queue") -> dict:
    data = {}
    for folder, label in RUNS:
        run = load_run(str(ROOT / "results" / "tbench_full_b4n8" / folder))
        data[label] = [summarize(simulate(run, Policy.parse(p), sandbox=sandbox)) for p in policies]
    return data


def total_change(rows: list[dict]) -> list[float]:
    return [r["makespan_total"] / rows[0]["makespan_total"] - 1 for r in rows]


def draw(axes, pols, data, fixed, label_size, ub_lw):
    """axes: hit-rate panels (tool[, test]) then the time panel."""
    n = len(pols)
    w = 0.38
    hit_panels = [("tool", "Tool-call hit rate"), ("test", "Reward-test hit rate")][: len(axes) - 1]
    for ax, (kind, title) in zip(axes, hit_panels):
        for j, (label, rows) in enumerate(data.items()):
            xs = [i + (j - 0.5) * w for i in range(n)]
            vals = [r[kind]["hit_rate"] for r in rows]
            ax.bar(xs, vals, w, color=COLORS[j])
            ax.hlines(rows[0][kind]["reuse_bound"], -0.5, n - 0.5, colors=COLORS[j], linestyles=(0, (4, 3)), lw=ub_lw)
            for xi, v in zip(xs, vals):
                if v >= 0.005:
                    ax.text(xi, v, f"{v:.0%}", ha="center", va="bottom", fontsize=label_size, color=INK2)
        ax.set_title(title)
    ax = axes[-1]
    for j, (label, rows) in enumerate(data.items()):
        xs = [i + (j - 0.5) * w for i in range(n)]
        ax.bar(xs, total_change(rows), w, color=COLORS[j])
        ax.scatter(xs, total_change(fixed[label]), marker="D", s=label_size * 3, color=INK2, zorder=5)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title("Total time vs. default")
    for ax in axes:
        ax.set_xticks(range(n), [p for _, p in pols])
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        ax.margins(y=0.16)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def legend_handles(lw):
    return ([Patch(color=HIT), Patch(color=LIGHT), Line2D([], [], color=INK2, ls=(0, (4, 3)), lw=lw),
             Line2D([], [], color=INK2, marker="D", ls="")])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "poster_figures"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pols = [p for p, _ in POLICIES]
    data, fixed = load(pols), load(pols, "fixed")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw={"wspace": 0.28})
    draw(axes, POLICIES, data, fixed, label_size=11, ub_lw=1.6)
    axes[0].set_ylabel("share of tool calls")
    axes[1].set_ylabel("share of rollouts")
    axes[2].set_ylabel("change in summed step time")
    fig.legend(legend_handles(1.6), ["TB easy", "TB medium", "reuse bound (every repeated key hits)",
                                     "recorded sandbox start times"],
               ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.1), fontsize=13)
    fig.suptitle("A short bounded hold recovers most sibling reuse; waiting without a limit can wait out a timeout",
                 x=0.01, ha="left", fontsize=17, fontweight="bold", y=1.06)
    fig.text(0.01, 0.98, "Simulated on the full Terminal-Bench traces (Qwen2.5-1.5B, 4 prompts × 8 rollouts × 20 steps, "
             "cache on). Bars: sandbox starts queue on the node's start lock", ha="left", fontsize=12.5, color=INK2)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig6_sim_policies.{ext}", bbox_inches="tight")
    print(f"wrote {out}/fig6_sim_policies.png")
    column_figure(out)


def column_figure(out: Path) -> None:
    """Narrow version for one 8.2-inch poster column, drawn at print size (fonts are real points)."""
    pols = [p for p, _ in COLUMN_POLICIES]
    data, fixed = load(pols), load(pols, "fixed")
    with plt.rc_context({"font.size": 17, "axes.titlesize": 19, "xtick.labelsize": 15, "ytick.labelsize": 15,
                         "axes.labelsize": 16, "legend.fontsize": 16}):
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.5), gridspec_kw={"wspace": 0.42})
        draw(axes, COLUMN_POLICIES, data, fixed, label_size=13, ub_lw=2)
        fig.legend(legend_handles(2), ["TB easy", "TB medium", "reuse bound", "recorded sandbox times"],
                   ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.27), handlelength=1.6, columnspacing=1.2)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"fig6_sim_policies_column.{ext}", bbox_inches="tight")
    print(f"wrote {out}/fig6_sim_policies_column.png")


if __name__ == "__main__":
    main()
