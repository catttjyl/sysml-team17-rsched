"""Poster figures from the runs in results/.

    python scripts/plot_poster.py            # reads results/toy_n4 and results/sql_full_n16
    python scripts/plot_poster.py --out results/poster_figures

Every figure is written as PNG (300 dpi) and PDF. Sizes are in inches for a
poster panel; fonts are large enough to read from about 1.5 m.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "results"

# Color carries the miss class; text stays in ink colors.
HIT, TIMING, COLD, GEN = "#2a78d6", "#eb6834", "#b5b3ad", "#e4e2dc"
FILLER = "#6f6e69"  # the first (cold) call that the timing misses waited on
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e2dc"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 14,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.frameon": False,
    "legend.fontsize": 13,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
})

# (folder, run, label). Order is the order on every chart.
RUNS = [
    ("toy_n4", "sql_cache-on", "Toy SQL · cache on"),
    ("toy_n4", "terminal_cache-on", "Toy terminal · cache on"),
    ("toy_n4", "terminal_cache-off", "Toy terminal · cache off"),
    ("sql_full_n16", "sql_cache-on", "SkyRL-SQL · cache on (pre-fix)"),
    ("sql_full_n16", "sql_cache-on-nodelay", "SkyRL-SQL · cache on (fixed)"),
    ("sql_full_n16", "sql_cache-off", "SkyRL-SQL · cache off"),
]


def read_report(path: Path) -> dict:
    # Some committed report.json files have a trailing "wrote N rows" line.
    return json.JSONDecoder().raw_decode(path.read_text())[0]


def load():
    runs = []
    for folder, run, label in RUNS:
        d = ROOT / folder / run
        df = pd.read_csv(d / "tool_calls.csv")
        df["cls"] = df["class"].replace({"would_hit": "hit"})
        df["seconds"] = df["t_end"] - df["t_start"]
        rep = read_report(d / "report.json")
        runs.append({"dir": d, "label": label, "df": df, "report": rep})
    return runs


def save(fig, out: Path, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def bar_ends(ax):
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- 1. miss classes
def fig_miss_classes(runs, out):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    order = ["hit", "timing", "cold"]
    colors = {"hit": HIT, "timing": TIMING, "cold": COLD}
    for i, r in enumerate(runs):
        df = r["df"]
        n = len(df)
        share = df["cls"].value_counts(normalize=True) * 100
        left = 0.0
        for c in order:
            w = share.get(c, 0.0)
            ax.barh(i, w, left=left, height=0.62, color=colors[c], edgecolor="white", linewidth=2)
            left += w
        hit, tim = share.get("hit", 0), share.get("timing", 0)
        ax.text(101.5, i, f"hit {hit:.1f}%  ·  timing {tim:.1f}%   (n={n})", va="center", color=INK2, fontsize=12.5)
    ax.axhline(2.5, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_yticks(range(len(runs)), [r["label"] for r in runs])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of tool calls (%)")
    bar_ends(ax)
    ax.set_title("Most tool calls miss cold; timing misses are a thin slice", pad=14)
    handles = [Patch(color=colors[c]) for c in order]
    ax.legend(handles, ["Hit (cache off: would hit)", "Timing miss", "Cold miss"],
              ncol=3, loc="upper left", bbox_to_anchor=(0, -0.16))
    save(fig, out, "fig1_miss_classes")


# ---------------------------------------------------------------- 2. H1: position
def fig_position(runs, out):
    all_calls = pd.concat([r["df"] for r in runs])
    pos = all_calls["seq"].clip(upper=4)
    labels = ["1st", "2nd", "3rd", "4th+"]
    total = pos.value_counts().reindex([1, 2, 3, 4], fill_value=0)
    timing = pos[all_calls["cls"] == "timing"].value_counts().reindex([1, 2, 3, 4], fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, vals, color, title in [
        (axes[0], total, COLD, "All tool calls"),
        (axes[1], timing, TIMING, "Timing misses"),
    ]:
        ax.bar(labels, vals.values, width=0.6, color=color, edgecolor="white", linewidth=2)
        for x, v in zip(labels, vals.values):
            ax.text(x, v, f"{v}", ha="center", va="bottom", color=INK, fontsize=13)
        ax.set_title(title, fontsize=15)
        ax.set_xlabel("Position of the call in its rollout")
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
        ax.set_ylim(0, vals.max() * 1.15 + 1)
    axes[0].set_ylabel("Tool calls (6 runs pooled)")
    share_first = total[1] / total.sum() * 100
    fig.suptitle(f"H1: all {timing.sum()} timing misses are on the 1st call "
                 f"(which is {share_first:.0f}% of calls)", x=0.01, ha="left",
                 fontsize=17, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig2_h1_position")


# ---------------------------------------------------------------- 3. H2: time budget
def fig_time_budget(runs, out):
    keep = [r for r in runs if "pre-fix" not in r["label"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    for i, r in enumerate(keep):
        rep = r["report"]
        wl = r["df"]["workload"].iloc[0]
        pct = rep[wl]["tool_seconds_total"] / rep["gen_seconds_total"] * 100
        ax.barh(i, pct, height=0.6, color=FILLER)
        ax.text(pct + 0.6, i, f"{pct:.2g}%", va="center", color=INK, fontsize=13)
    ax.set_yticks(range(len(keep)), [r["label"] for r in keep])
    ax.invert_yaxis()
    ax.set_xlim(0, 40)
    ax.set_xlabel("Tool time as % of generation time")
    bar_ends(ax)
    ax.set_title("Tool time vs. LLM generation time", fontsize=15)

    ax = axes[1]
    data = [r["df"].loc[(r["df"]["cls"] == "cold") & (r["df"]["seconds"] < 25), "seconds"] * 1000 for r in keep]
    ax.boxplot(data, orientation="horizontal", widths=0.5, showfliers=False,
               medianprops={"color": INK, "linewidth": 2.5},
               boxprops={"color": INK2}, whiskerprops={"color": INK2}, capprops={"color": INK2})
    for i, d in enumerate(data, start=1):
        ax.text(2500, i, f"median {d.median():.3g} ms", va="center", color=INK2, fontsize=12)
    ax.set_xscale("log")
    ax.set_yticks(range(1, len(keep) + 1), [""] * len(keep))
    ax.invert_yaxis()
    ax.set_xlim(0.3, 3e4)
    ax.set_xlabel("Time per cold tool call (ms, log scale)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.set_title("Per-call latency", fontsize=15)

    fig.suptitle("H2: SQL tools are too fast for scheduling to matter; slow tools are where the gain is",
                 x=0.01, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.01, -0.03, "Toy terminal includes 3 calls that hit the 30 s timeout (left panel); they are left out of the right panel. "
             "Toy runs predate the TCP_NODELAY fix (~85 ms extra per cached call); SkyRL-SQL cache-on is the fixed run.", fontsize=11, color=INK2)
    fig.tight_layout()
    save(fig, out, "fig3_h2_time_budget")


# ---------------------------------------------------------------- 4. TCP_NODELAY fix
def fig_nodelay(runs, out):
    sel = [r for r in runs if r["label"].startswith("SkyRL")]
    names = ["Cache on\n(before fix)", "Cache on\n(TCP_NODELAY)", "Cache off"]
    med = [r["df"].loc[r["df"]["cls"] == "cold", "seconds"].median() * 1000 for r in sel]
    tim = [int((r["df"]["cls"] == "timing").sum()) for r in sel]
    colors = [FILLER, FILLER, COLD]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, vals, fmt, title in [
        (axes[0], med, "{:.3g} ms", "Median cold call (ms)"),
        (axes[1], tim, "{}", "Timing misses (count)"),
    ]:
        ax.bar(names, vals, width=0.58, color=colors, edgecolor="white", linewidth=2)
        for x, v in zip(names, vals):
            ax.text(x, v, fmt.format(v), ha="center", va="bottom", color=INK, fontsize=14)
        ax.set_title(title, fontsize=15)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
    fig.suptitle("A one-line TVCache fix: 88 ms → 2 ms per call, and fewer timing misses",
                 x=0.01, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.01, -0.04, "SkyRL-SQL, Qwen2.5-1.5B, 20 GRPO steps × 8 prompts × 16 rollouts. "
             "Nagle + delayed ACK added ~40 ms per reply until TCP_NODELAY was set.", fontsize=11, color=INK2)
    fig.tight_layout()
    save(fig, out, "fig4_tcp_nodelay_fix")


# ---------------------------------------------------------------- 5. sibling timeline
def load_events(d: Path):
    ev = []
    for f in sorted((d / "timeline").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                ev.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return ev


def fig_siblings(run, out, name, window=None):
    df = run["df"]
    uid = df[df["cls"] == "timing"]["uid"].value_counts().index[0]
    calls = df[df["uid"] == uid].copy()
    shared = set(calls.loc[calls["cls"] == "timing", "key"])
    filler_idx = calls[calls["key"].isin(shared)].sort_values("t_start").index[0]
    ev = [e for e in load_events(run["dir"]) if e.get("uid") == uid]
    t0 = min(e["t"] for e in ev if e["ev"] == "rollout_start")
    rollouts = sorted({e["rollout"] for e in ev if e["ev"] == "rollout_start"},
                      key=lambda r: min([c for c in calls.loc[calls["rollout"] == r, "t_start"]] or [1e30]))

    h = 0.42 * len(rollouts) + 2.0
    fig, ax = plt.subplots(figsize=(12, h))
    for i, r in enumerate(rollouts):
        for e in ev:
            if e["ev"] == "gen_end" and e["rollout"] == r:
                ax.barh(i, e["t"] - e["t_start"], left=e["t_start"] - t0, height=0.55, color=GEN,
                        edgecolor="white", linewidth=1)
        for idx, c in calls[calls["rollout"] == r].iterrows():
            x = c["t_start"] - t0
            if c["cls"] == "hit":
                ax.plot(x, i, marker="D", markersize=9, color=HIT, markeredgecolor="white", markeredgewidth=1.5)
                continue
            color = FILLER if idx == filler_idx else (TIMING if c["cls"] == "timing" else COLD)
            ax.barh(i, max(c["seconds"], 0.02), left=x, height=0.55, color=color, edgecolor="white", linewidth=1)
    ax.set_yticks(range(len(rollouts)), [f"Rollout {i + 1}" for i in range(len(rollouts))], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Seconds since the prompt's rollouts started")
    if window:
        ax.set_xlim(*window)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    task = calls["task"].iloc[0].split("|")[-1]
    ax.set_title("Siblings send the same first call while its first copy is still running → timing misses",
                 pad=48, fontsize=16)
    handles = [Patch(color=GEN), Patch(color=FILLER), Patch(color=TIMING), Patch(color=COLD),
               plt.Line2D([], [], marker="D", color=HIT, linestyle="", markersize=9)]
    labels = ["LLM generation", "First copy of the shared call", "Timing miss", "Other cold miss", "Cache hit"]
    ax.legend(handles, labels, ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=12,
              handlelength=1.2, columnspacing=1.2)
    fig.text(0.01, 0.0, f"{run['label']} · task {task} · one GRPO group", fontsize=11, color=INK2)
    fig.tight_layout()
    save(fig, out, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "poster_figures"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = load()
    fig_miss_classes(runs, out)
    fig_position(runs, out)
    fig_time_budget(runs, out)
    fig_nodelay(runs, out)
    by = {r["label"]: r for r in runs}
    fig_siblings(by["SkyRL-SQL · cache on (pre-fix)"], out, "fig5_siblings_sql")
    fig_siblings(by["Toy terminal · cache on"], out, "fig5_siblings_terminal")
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
