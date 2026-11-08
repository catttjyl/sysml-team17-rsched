"""Figures for the reuse-sched runs.

    python scripts/plot.py toy              # results/toy_n4
    python scripts/plot.py full             # results/*full*b4n8*
    python scripts/plot.py concept          # timing-miss concept sketch (no data)
    python scripts/plot.py toy full concept # several at once

Figures go to results/figures/<name>/ as PNG (300 dpi).

Data figures (toy, full):
  time_breakdown  LLM generation vs tool time, one bar per rollout, one panel per workload
  tool_latency    time per cold tool call, every run
  miss_classes    hit / timing miss / cold miss share, every run
  by_turn         tool calls and timing misses by turn, cache-on runs

A run is any subfolder with a tool_calls.csv, named <workload>_cache-<on|off>.
To add a dataset, add an entry to DATASETS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "results"

# Glob patterns (relative to results/) whose subfolders are the runs of a dataset.
DATASETS = {
    "toy": ["toy_n4"],
    "full": ["*full*b4n8*"],
}

# Order on every chart; anything not listed goes last.
WORKLOADS = {
    "sql": "SQL",
    "terminal": "Terminal",
    "tb-easy": "Terminal-Bench easy",
    "tb-medium": "Terminal-Bench medium",
}

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e2dc"
HIT, TIMING, COLD = "#2a78d6", "#eb6834", "#b5b3ad"
GEN, TOOL = "#c9c7c0", "#4f81bd"
# Workload identity in by_turn: teal / lavender / mustard / brown. Hues stay away from HIT (blue) and
# TIMING (orange) so a workload is never read as a miss class; all pairs pass the colorblind check.
WL_COLORS = ["#2a9d8f", "#8e6bbf", "#c9a227", "#7a5c48"]


def grayer(color, amount=0.5):
    """Blend a color toward COLD gray; amount 0 keeps it, 1 is plain gray."""
    c, g = np.array(to_rgb(color)), np.array(to_rgb(COLD))
    return tuple((1 - amount) * c + amount * g)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 300,
})


# ---------------------------------------------------------------- loading
def load_events(run_dir: Path) -> list[dict]:
    ev = []
    for f in sorted((run_dir / "timeline").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                ev.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return ev


def command_seconds(total, sandbox_start):
    """Tool time without starting the sandbox (container), which is not what scheduling can change.
    A scheduler evaluation should leave sandbox start out the same way."""
    return (total - sandbox_start).clip(lower=0) if hasattr(total, "clip") else max(total - sandbox_start, 0.0)


def load_run(d: Path) -> dict:
    workload, cache = d.name.rsplit("_cache-", 1)
    df = pd.read_csv(d / "tool_calls.csv")
    # With the cache off, "would_hit" means the call would have hit.
    df["cls"] = df["class"].replace({"would_hit": "hit", "late": "cold"})
    df["seconds"] = df["t_end"] - df["t_start"]
    ev = load_events(d)
    # A "Failed to execute" result is not cached; for these tools it means the command timed out.
    tools = pd.DataFrame([{"rollout": e["rollout"], "seq": e["seq"], "timed_out": bool(e.get("uncacheable")),
                           "sandbox_s": e.get("sandbox_start_s") or 0.0}
                          for e in ev if e["ev"] == "tool"])
    if len(tools):
        df = df.merge(tools.drop_duplicates(["rollout", "seq"]), on=["rollout", "seq"], how="left")
    df["timed_out"] = df.get("timed_out", False)
    df["timed_out"] = df["timed_out"].fillna(False).astype(bool)
    df["sandbox_s"] = df.get("sandbox_s", 0.0)
    df["sandbox_s"] = df["sandbox_s"].fillna(0.0)
    df["exec_seconds"] = command_seconds(df["seconds"], df["sandbox_s"])
    label = WORKLOADS.get(workload, workload)  # with_cache() adds "cache on/off" where a chart mixes both
    return {"dir": d, "workload": workload, "cache": cache, "label": label, "df": df, "events": ev}


def load_dataset(name: str) -> list[dict]:
    dirs = sorted({d for pat in DATASETS[name] for folder in ROOT.glob(pat)
                   for d in folder.iterdir() if (d / "tool_calls.csv").exists()})
    runs = [load_run(d) for d in dirs]
    order = list(WORKLOADS)
    runs.sort(key=lambda r: (order.index(r["workload"]) if r["workload"] in order else len(order),
                             r["workload"], r["cache"] != "on"))
    if not runs:
        raise SystemExit(f"no runs found for {name}: {DATASETS[name]}")
    return runs


def with_cache(run) -> str:
    return f"{run['label']} · cache {run['cache']}"


def by_workload(runs):
    out = {}
    for r in runs:
        out.setdefault(r["workload"], {})[r["cache"]] = r
    return out


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def fmt_seconds(s: float) -> str:
    return f"{s * 1000:.3g} ms" if s < 1 else f"{s:.3g} s"


def workload_separators(ax, runs, offset=0):
    """Dashed lines between workloads on a chart with one row per run (row i at y = i + offset)."""
    for i in range(1, len(runs)):
        if runs[i]["workload"] != runs[i - 1]["workload"]:
            ax.axhline(i - 0.5 + offset, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))


# ---------------------------------------------------------------- time breakdown
def rollout_segments(run) -> pd.DataFrame:
    """One row per rollout, in step/group order, with its segments in the order they happened:
    segs = [("gen" | "tool", seconds), ...]. Tool seconds leave out sandbox start;
    gaps between segments (queueing) are left out too."""
    ev = run["events"]
    starts = {e["rollout"]: (e["t"], e["uid"]) for e in ev if e["ev"] == "rollout_start"}
    segs = {r: [] for r in starts}
    for e in ev:
        if e.get("rollout") not in segs:
            continue
        if e["ev"] == "gen_end":
            segs[e["rollout"]].append((e["t_start"], "gen", e["t"] - e["t_start"]))
        elif e["ev"] == "tool":
            secs = command_seconds(e["t_end"] - e["t_start"], e.get("sandbox_start_s") or 0.0)
            segs[e["rollout"]].append((e["t_start"], "tool", secs))
    rows = []
    for r, (t, u) in starts.items():
        s = [(k, w) for _, k, w in sorted(segs[r])]
        rows.append({"rollout": r, "t0": t, "uid": u, "segs": s,
                     "gen": sum(w for k, w in s if k == "gen"), "tool": sum(w for k, w in s if k == "tool")})
    df = pd.DataFrame(rows)
    # Groups in the order they started; siblings of one prompt stay adjacent.
    df["group_t0"] = df.groupby("uid")["t0"].transform("min")
    return df.sort_values(["group_t0", "uid", "t0"]).reset_index(drop=True)


def prefer_cache_off(runs):
    """One run per workload: cache off if there is one (every call runs its command in full), else cache on."""
    return [caches.get("off") or caches["on"] for caches in by_workload(runs).values()]


def fig_time_breakdown(runs, out):
    # Cache OFF. Toy SQL has no cache-off run, so it falls back to cache ON.
    picked = prefer_cache_off(runs)
    fig, axes = plt.subplots(len(picked), 1, figsize=(13, 2.9 * len(picked) + 0.6), squeeze=False)
    for ax, run in zip(axes[:, 0], picked):
        t = rollout_segments(run)
        x = np.arange(len(t))
        # Stack segments in the order they happened: gen, tool, gen, tool, ..., gen.
        bottom = np.zeros(len(t))
        for k in range(max(len(s) for s in t["segs"])):
            w = np.array([s[k][1] if k < len(s) else 0.0 for s in t["segs"]])
            kind = [s[k][0] if k < len(s) else "gen" for s in t["segs"]]
            ax.bar(x, w, bottom=bottom, width=1.0, linewidth=0,
                   color=[GEN if c == "gen" else TOOL for c in kind])
            bottom += w
        share = t["tool"].sum() / (t["gen"].sum() + t["tool"].sum())
        n_sib = int(t.groupby("uid").size().median())
        ax.set_title(f"{run['label']}: tool executions are {share:.1%} of the time", fontsize=13)
        ax.set_xlim(-0.5, len(t) - 0.5)
        # A few rollouts with timed-out commands would flatten the rest: clip them and print their value.
        total = t["gen"] + t["tool"]
        cap = total.quantile(0.99) * 1.3
        clipped = ""
        if total.max() > cap * 1.5:
            ax.set_ylim(0, cap)
            over = total > cap
            ax.plot(x[over], [cap] * over.sum(), marker="^", linestyle="", color=INK, markersize=6, clip_on=False)
            clipped = f" · ▲ {over.sum()} clipped (up to {total.max():.0f} s)"
        ax.set_ylabel("Seconds")
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
        ax.text(1.0, 1.02, f"{len(t)} rollouts, {n_sib} siblings per prompt{clipped}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=11, color=INK2)
    axes[-1, 0].set_xlabel("Rollout index (training order; siblings of one prompt are adjacent)")
    fig.legend([Patch(color=GEN), Patch(color=TOOL)],
               ["LLM generation", "Tool call execution (sandbox start left out)"],
               ncol=2, loc="lower left", bbox_to_anchor=(0.01, -0.02), fontsize=12)
    fig.suptitle(f"Where each rollout's time goes?",
                 x=0.01, ha="left", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out, "time_breakdown")


# ---------------------------------------------------------------- latency
def fig_latency(runs, out):
    # Cache OFF. Toy SQL has no cache-off run, so it falls back to cache ON with hits left out.
    runs = prefer_cache_off(runs)
    fig, ax = plt.subplots(figsize=(12, 0.7 * len(runs) + 1.8))
    # Cache off: every call ran its command. Cache on (fallback only): leave out hits, which ran nothing.
    tool = [r["df"] if r["cache"] == "off" else r["df"][r["df"]["executed"] >= 1] for r in runs]
    data = [t["exec_seconds"].clip(lower=1e-4) for t in tool]
    ax.boxplot(data, orientation="horizontal", widths=0.5, showfliers=False,
               medianprops={"color": INK, "linewidth": 2},
               boxprops={"color": INK2}, whiskerprops={"color": INK2}, capprops={"color": INK2})
    hi = max(d.max() for d in data)
    any_timeout = False
    for i, (c, d) in enumerate(zip(tool, data), start=1):
        jitter = (np.arange(len(d)) % 7 - 3) * 0.04
        to = c["timed_out"].to_numpy()
        ax.scatter(d[~to], i + jitter[~to], s=10, color=TOOL, alpha=0.4, zorder=0)
        if to.any():
            any_timeout = True
            ax.scatter(d[to], i + jitter[to], s=40, marker="x", color=INK, linewidths=1.5, zorder=3)
        txt = f"median {fmt_seconds(d.median())}"
        if to.any():
            txt += f" · {to.sum()} timed out"
        ax.text(hi * 3, i, txt, va="center", color=INK2, fontsize=11)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(range(1, len(runs) + 1), [r["label"] for r in runs])
    ax.set_ylim(len(runs) + 0.5, 0.5)
    ax.set_xlabel("Time per tool call (s, log scale; sandbox start left out)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.set_title(f"How long a tool call takes", pad=10)
    fig.tight_layout()
    handles = [plt.Line2D([], [], marker="o", color=TOOL, linestyle="", markersize=5)]
    labels = ["One tool call"]
    if any_timeout:
        handles.append(plt.Line2D([], [], marker="x", color=INK, linestyle="", markersize=7))
        labels.append("Tool call that hit the sandbox timeout")
    fig.legend(handles, labels, ncol=len(handles), loc="upper left", bbox_to_anchor=(0.01, 0.0))
    save(fig, out, "tool_latency")


# ---------------------------------------------------------------- miss classes
def fig_miss_classes(runs, out):
    # Cache ON and OFF, one row each, so rows are labeled with the cache setting.
    order = ["hit", "timing", "cold"]
    colors = {"hit": HIT, "timing": TIMING, "cold": COLD}
    fig, ax = plt.subplots(figsize=(11, 0.7 * len(runs) + 1.8))
    for i, r in enumerate(runs):
        share = r["df"]["cls"].value_counts(normalize=True) * 100
        left = 0.0
        for c in order:
            w = share.get(c, 0.0)
            ax.barh(i, w, left=left, height=0.62, color=colors[c], edgecolor="white", linewidth=2)
            left += w
        ax.text(101.5, i, f"hit {share.get('hit', 0):.1f}%  ·  timing {share.get('timing', 0):.1f}%",
                va="center", color=INK2, fontsize=12)
    workload_separators(ax, runs)
    ax.set_yticks(range(len(runs)), [with_cache(r) for r in runs])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of tool calls (%)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.set_title(f"Most tool calls miss cold; timing misses are a thin slice", pad=12)
    fig.tight_layout()
    fig.legend([Patch(color=colors[c]) for c in order],
               ["Hit (cache off: would have hit)", "Timing miss", "Cold miss"],
               ncol=3, loc="upper left", bbox_to_anchor=(0.01, 0.0))
    save(fig, out, "miss_classes")


# ---------------------------------------------------------------- by turn
def fig_by_turn(runs, out):
    # Cache ON only: timing misses are real only when the cache is on.
    on =[r for r in runs if r["cache"] == "on"]
    max_turn = int(max(r["df"]["turn"].max() for r in on))
    turns = np.arange(1, max_turn + 1)
    width = 0.8 / len(on)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    total_tim = total_early = 0
    for k, r in enumerate(on):
        df = r["df"]
        n = len(df)
        all_n = df["turn"].value_counts().reindex(turns, fill_value=0)
        tim = df.loc[df["cls"] == "timing", "turn"].value_counts().reindex(turns, fill_value=0)
        total_tim += tim.sum()
        total_early += tim[tim.index == 1].sum()
        x = turns + (k - (len(on) - 1) / 2) * width
        color = WL_COLORS[k % len(WL_COLORS)]
        name = WORKLOADS.get(r["workload"], r["workload"])
        axes[0].bar(x, all_n, width=width, color=grayer(color, 0.35), edgecolor="white", linewidth=2, label=name)
        axes[1].bar(x, tim / n * 100, width=width, color=grayer(color, 0.35), edgecolor="white", linewidth=2)
        for xi, c in zip(x, tim):
            if c:
                axes[1].text(xi, c / n * 100, f"{c / n:.1%}\n({c})", ha="center", va="bottom",
                             fontsize=10.5, color=INK)
    for ax, title in [(axes[0], "All tool calls"), (axes[1], "Timing misses")]:
        ax.set_xticks(turns, [str(t) for t in turns])
        ax.set_xlabel("Turn of the tool call in its rollout")
        ax.set_title(title, fontsize=14)
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Tool calls")
    axes[1].set_ylabel("Timing misses, % of the run's tool calls")
    top = axes[1].get_ylim()[1]
    axes[1].set_ylim(0, top * 1.25)
    axes[0].legend(loc="upper right")
    early = total_early / total_tim if total_tim else 0
    fig.suptitle(f"{early:.0%} of timing misses are on turn 1 (a prompt's siblings start together)", x=0.01, ha="left", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "by_turn")


# ---------------------------------------------------------------- concept
def fig_concept(out):
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True)
    rows = ["Rollout 1", "Rollout 2", "Rollout 3"]
    cases = [
        ("Today: siblings run independently", [
            [("gen", 0, 2.0), ("cold", 2.0, 2.0)],
            [("gen", 0, 2.4), ("timing", 2.4, 2.0)],
            [("gen", 0, 4.6), ("hit", 4.6, 0.15)],
        ], "2 executions of the same call; rollout 2 pays the full tool time again"),
        ("Reuse-aware: rollout 2 waits for K", [
            [("gen", 0, 2.0), ("cold", 2.0, 2.0)],
            [("gen", 0, 2.4), ("wait", 2.4, 1.6), ("hit", 4.0, 0.15)],
            [("gen", 0, 4.6), ("hit", 4.6, 0.15)],
        ], "1 execution; the timing miss becomes a hit (saves sandbox work)"),
    ]
    colors = {"gen": "#e4e2dc", "cold": "#6f6e69", "timing": TIMING, "hit": HIT, "wait": "white"}
    for ax, (title, bars, note) in zip(axes, cases):
        for i, row in enumerate(bars):
            for kind, x, w in row:
                ax.barh(i, w, left=x, height=0.55, color=colors[kind],
                        edgecolor=TIMING if kind == "wait" else "white",
                        hatch="///" if kind == "wait" else None, linewidth=1.5)
        ax.axvline(2.0, color=INK2, lw=0.8, ls=(0, (2, 3)))
        ax.axvline(4.0, color=INK2, lw=0.8, ls=(0, (2, 3)))
        ax.set_yticks(range(3), rows)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=14)
        ax.text(1.0, 1.02, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=11.5,
                color=INK2, style="italic")
        ax.set_xlim(0, 7)
    axes[1].text(2.05, 2.75, "call K issued", fontsize=10, color=INK2, va="top")
    axes[1].text(4.05, 2.75, "K's result cached", fontsize=10, color=INK2, va="top")
    axes[1].text(7, 2.75, "time →", fontsize=11, color=INK2, ha="right", va="top")
    axes[1].set_xticks([])
    handles = [Patch(color="#e4e2dc"), Patch(color="#6f6e69"), Patch(color=TIMING), Patch(color=HIT),
               Patch(facecolor="white", edgecolor=TIMING, hatch="///")]
    fig.legend(handles, ["LLM generation", "Cold miss (first copy of K)", "Timing miss (K still in flight)",
                         "Cache hit", "Waiting (scheduler)"], ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), fontsize=11)
    fig.suptitle("Timing miss: an identical call that misses only because its first copy hasn't finished yet",
                 x=0.01, ha="left", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "timing_miss_concept")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="+", choices=[*DATASETS, "concept"])
    ap.add_argument("--out", default=str(ROOT / "figures"), help="parent folder; one subfolder per dataset")
    args = ap.parse_args()
    for name in args.what:
        out = Path(args.out) / name
        out.mkdir(parents=True, exist_ok=True)
        if name == "concept":
            fig_concept(out)
        else:
            runs = load_dataset(name)
            fig_time_breakdown(runs, out)
            fig_latency(runs, out)
            fig_miss_classes(runs, out)
            fig_by_turn(runs, out)
        print(f"{name}: written to {out}")


if __name__ == "__main__":
    main()
