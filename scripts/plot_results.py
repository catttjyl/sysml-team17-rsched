"""Plot the results of a batch of runs.

    python scripts/plot_results.py results/toy_n4

Each subfolder with a tool_calls.csv is one run. Figures go to <dir>/figures/.
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

# Colors by class. Cold is gray because it cannot be avoided.
COLORS = {"hit": "#2a78d6", "timing": "#eb6834", "cold": "#b5b3ad", "gen": "#e4e2dc"}
TEXT = "#52514e"
CLASSES = ["hit", "timing", "cold"]
LABELS = {"hit": "Hit", "timing": "Timing miss", "cold": "Cold miss"}

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 150,
})


def run_label(name):
    workload, cache = name.split("_", 1)
    workload = "SQL" if workload == "sql" else workload.capitalize()
    cache = cache.replace("-nodelay", ", fixed").replace("-", " ")
    return f"{workload}, {cache}"


def load_runs(root):
    runs = {}
    order = {"cache-on": 0, "cache-on-nodelay": 1, "cache-off": 2}
    dirs = sorted(root.iterdir(), key=lambda d: (not d.name.startswith("sql"), order.get(d.name.split("_")[-1], 2), d.name))
    for d in dirs:
        csv = d / "tool_calls.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        # With the cache off, "would_hit" means the call would have hit.
        df["class"] = df["class"].replace({"would_hit": "hit", "late": "cold"})
        df["seconds"] = df["t_end"] - df["t_start"]
        runs[d.name] = df
    return runs


def plot_classes(runs, out):
    """Share of tool calls in each class, one bar per run."""
    names = list(runs)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    for i, name in enumerate(names):
        share = runs[name]["class"].value_counts(normalize=True)
        left = 0
        for c in CLASSES:
            w = share.get(c, 0) * 100
            ax.barh(i, w, left=left, height=0.6, color=COLORS[c],
                    edgecolor="white", linewidth=2, label=LABELS[c] if i == 0 else None)
            if c == "timing":
                ax.text(102, i, f"timing {w:.1f}%", va="center", color=TEXT, fontsize=9)
            left += w
    ax.set_yticks(range(len(names)), [run_label(n) for n in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of tool calls (%)")
    ax.set_title("Most misses are cold. Timing misses are a small slice.", loc="left", fontsize=12)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.28))
    fig.tight_layout()
    fig.savefig(out / "miss_classes.png", bbox_inches="tight")
    plt.close(fig)


def plot_position(runs, out):
    """Calls by position in the rollout, split by class (H1)."""
    names = list(runs)
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3), sharey=True)
    for ax, name in zip(axes, names):
        df = runs[name]
        pos = df["seq"].clip(upper=3).map({1: "1st", 2: "2nd", 3: "3rd+"})
        counts = pd.crosstab(pos, df["class"]).reindex(["1st", "2nd", "3rd+"], fill_value=0)
        bottom = None
        for c in CLASSES:
            vals = counts[c] if c in counts else pd.Series(0, index=counts.index)
            ax.bar(counts.index, vals, bottom=bottom, width=0.6, color=COLORS[c],
                   edgecolor="white", linewidth=2, label=LABELS[c])
            bottom = vals if bottom is None else bottom + vals
        ax.set_title(run_label(name), fontsize=11, loc="left")
        ax.set_xlabel("Tool call in rollout")
    axes[0].set_ylabel("Tool calls")
    axes[0].legend(loc="upper right")
    timing = pd.concat([runs[n].loc[runs[n]["class"] == "timing", "seq"] for n in names])
    if len(timing) and (timing == 1).all():
        title = "All timing misses happen on the first call (H1)"
    else:
        title = "Tool calls by position in the rollout (H1)"
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "by_position.png", bbox_inches="tight")
    plt.close(fig)


def plot_latency(runs, out, timeout=5.0):
    """Time per cold call. Compares workloads (H2) and cache on vs off.
    Calls that hit the command timeout are left out."""
    names = list(runs)
    cold = [runs[n].loc[runs[n]["class"] == "cold", "seconds"] for n in names]
    data = [d[d < timeout] for d in cold]
    dropped = sum(int((d >= timeout).sum()) for d in cold)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.boxplot(data, vert=False, widths=0.5, showfliers=False,
               medianprops={"color": TEXT, "linewidth": 2},
               boxprops={"color": TEXT}, whiskerprops={"color": TEXT}, capprops={"color": TEXT})
    for i, d in enumerate(data, start=1):
        jitter = (pd.Series(range(len(d))) % 7 - 3) * 0.04
        ax.scatter(d, i + jitter.values, s=10, color=COLORS["cold"], zorder=0)
        ax.text(d.max() * 1.2, i, f"median {d.median():.2g} s", va="center", color=TEXT, fontsize=9)
    ax.set_xscale("log")
    lo = min(d.min() for d in data)
    hi = max(d.max() for d in data)
    ticks = [t for t in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
             if lo * 0.8 <= t <= hi * 1.5]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_yticks(range(1, len(names) + 1), [run_label(n) for n in names])
    ax.invert_yaxis()
    label = "Seconds per cold call, log scale"
    if dropped:
        label += f" ({dropped} calls that timed out are not shown)"
    ax.set_xlabel(label)
    ax.set_title("Time per cold call", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "tool_latency.png", bbox_inches="tight")
    plt.close(fig)


def load_events(timeline_dir):
    events = []
    for f in sorted(timeline_dir.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def plot_siblings(run_dir, df, out):
    """Timeline of the sibling rollouts of one prompt that had timing misses."""
    uid = df[df["class"] == "timing"]["uid"].value_counts().index[0]
    events = [e for e in load_events(run_dir / "timeline") if e.get("uid") == uid]
    calls = df[df["uid"] == uid]
    t0 = min(e["t"] for e in events if e["ev"] == "rollout_start")
    rollouts = sorted({e["rollout"] for e in events if e["ev"] == "rollout_start"})

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(rollouts) + 1.4))
    for i, r in enumerate(rollouts):
        for e in events:
            if e["ev"] == "gen_end" and e["rollout"] == r:
                ax.barh(i, e["t"] - e["t_start"], left=e["t_start"] - t0, height=0.5,
                        color=COLORS["gen"], edgecolor="white", linewidth=1)
        for _, c in calls[calls["rollout"] == r].iterrows():
            ax.barh(i, c["seconds"], left=c["t_start"] - t0, height=0.5,
                    color=COLORS[c["class"]], edgecolor="white", linewidth=1)
    keys = ["gen"] + [k for k in CLASSES if (calls["class"] == k).any()]
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in keys]
    names = ["Generation" if k == "gen" else LABELS[k] for k in keys]
    height = 0.45 * len(rollouts) + 1.6
    fig.set_size_inches(8, height)
    fig.legend(handles, names, ncol=len(keys), loc="lower left", bbox_to_anchor=(0.02, 0))
    ax.set_yticks(range(len(rollouts)), [f"Rollout {i + 1}" for i in range(len(rollouts))])
    ax.invert_yaxis()
    ax.set_xlabel("Seconds since the prompt's rollouts started")
    task = calls["task"].iloc[0].split("|")[-1]
    ax.set_title(f"Siblings send the same call at once, and all of them miss (task: {task})",
                 loc="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.45 / height, 1, 1))
    fig.savefig(out / "sibling_timeline.png", bbox_inches="tight")
    plt.close(fig)


def main():
    root = Path(sys.argv[1])
    out = root / "figures"
    out.mkdir(exist_ok=True)
    runs = load_runs(root)
    plot_classes(runs, out)
    plot_position(runs, out)
    plot_latency(runs, out)
    with_timing = [n for n in runs if "cache-on" in n and (runs[n]["class"] == "timing").any()]
    if with_timing:
        best = max(with_timing, key=lambda n: (runs[n]["class"] == "timing").sum())
        plot_siblings(root / best, runs[best], out)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
