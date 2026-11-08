"""Midpoint figures.

    python scripts/plot_midpoint.py   # writes results/midpoint_figures/

F1  pipeline architecture (no data)
F2  what a timing miss is, and what reuse-aware scheduling would change (no data)
F3  rollout timelines, one GRPO group of 8 siblings per workload
F4  where a rollout's time goes (generation / sandbox start / tool calls / reward test)
F5  per-call tool latency, cache on and off, log scale

F3-F5 use the full runs at 4 prompts x 8 rollouts x 20 steps
(results/sql_full_b4n8 and results/tbench_full_b4n8). Tool time counts command
execution only; sandbox (container) start is shown on its own.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
OUT = HERE.parent / "results" / "midpoint_figures"
OUT.mkdir(exist_ok=True)

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e2dc", "#fcfcfb"
GEN = "#c9c7c0"        # LLM generation
TOOL = "#6250d6"       # tool call (violet: not reused from hit/timing colors)
TEST = "#1baf7a"       # reward test
SANDBOX = "#cfcbe8"    # sandbox (container) start
TIMEOUT = "#9a99a3"    # tool calls that hit the command timeout
HIT, TIMING, COLD = "#2a78d6", "#eb6834", "#b5b3ad"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 300,
})

# (run folder without the _cache-on/off suffix, label)
WORKLOADS = [
    ("sql_full_b4n8/sql", "SkyRL-SQL"),
    ("tbench_full_b4n8/tb-easy", "Terminal-Bench easy"),
    ("tbench_full_b4n8/tb-medium", "Terminal-Bench medium"),
]
SETUP = "Qwen2.5-1.5B, 20 GRPO steps × 4 prompts × 8 rollouts"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def events(run):
    ev = []
    for f in sorted((RES / run / "timeline").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                ev.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return ev


# ---------------------------------------------------------------- F1 architecture
def box(ax, x, y, w, h, title, sub="", fc="#f3f2ee", ec=INK2, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2 + (0.13 if sub else 0), title, ha="center", va="center",
            fontsize=13, fontweight="bold" if bold else "normal", color=INK)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.2, sub, ha="center", va="center", fontsize=10.5, color=INK2)


def arrow(ax, x1, y1, x2, y2, text="", color=INK2, style="-|>", ls="-", tx=0, ty=0.12):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=1.6, linestyle=ls, shrinkA=2, shrinkB=2))
    if text:
        ax.text((x1 + x2) / 2 + tx, (y1 + y2) / 2 + ty, text, ha="center", va="bottom", fontsize=10.5, color=INK2)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(15, 7.4))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    box(ax, 0.3, 5.8, 3.2, 1.1, "veRL GRPO trainer", "8 prompts / step, N siblings each")
    box(ax, 4.6, 5.8, 3.2, 1.1, "vLLM rollout engine", "Qwen2.5-1.5B · 4×A100")
    ax.add_patch(FancyBboxPatch((0.3, 2.35), 7.5, 2.7, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="none", ec=INK2, lw=1.2, ls=(0, (4, 3))))
    ax.text(0.45, 4.8, "One GRPO group = N sibling rollouts of the same prompt", fontsize=11, color=INK2)
    for i in range(4):
        box(ax, 0.55 + i * 1.8, 2.6, 1.6, 1.85, f"Rollout {i + 1}", "TVCacheTool-\nAgentLoop", fc="white")
    box(ax, 10.3, 3.75, 4.4, 1.55, "TVCache server (Rust)", "Tool Call Graph per task\nkey = task + stateful prefix + call",
        fc="#e8f0fb", ec=HIT)
    box(ax, 10.3, 0.9, 4.4, 1.6, "Sandboxes", "SQL: read-only SQLite file\nTerminal: dir copy / podman container",
        fc="#eeecfb", ec=TOOL)
    box(ax, 0.3, 0.25, 7.5, 1.2, "Timeline log → analysis",
        "every gen turn & tool call: t_start, t_end, key, hit\n→ hit / cold / timing miss, by call position",
        fc="#e8f6f0", ec=TEST)
    box(ax, 9.0, 5.85, 5.7, 1.2, "Our proposal: reuse-aware scheduler",
        "hold a sibling until the identical in-flight call finishes", fc="#fdeee6", ec=TIMING)

    arrow(ax, 3.5, 6.35, 4.6, 6.35, "prompts")
    arrow(ax, 6.2, 5.8, 6.2, 5.05, "tokens", tx=0.45, ty=-0.12)
    arrow(ax, 1.9, 5.05, 1.9, 5.8, "rewards", tx=-0.55, ty=-0.12)
    arrow(ax, 7.85, 4.35, 10.3, 4.75, "lookup(history)", ty=0.12)
    arrow(ax, 10.3, 4.15, 7.85, 3.75, "hit → cached output", ty=-0.5)
    arrow(ax, 7.85, 2.95, 10.3, 1.9, "miss → fork deepest cached\nsandbox, run the suffix", ty=-0.95, tx=-0.55)
    arrow(ax, 12.5, 2.5, 12.5, 3.75, "insert result", tx=0.85, ty=-0.12)
    arrow(ax, 4.05, 2.35, 4.05, 1.45)
    arrow(ax, 9.0, 6.1, 7.85, 4.95, "", color=TIMING, ls=(0, (4, 3)))
    ax.set_title("Pipeline: veRL + TVCache, instrumented to measure reuse between siblings", fontsize=17, pad=8)
    save(fig, "F1_pipeline")


# ---------------------------------------------------------------- F2 concept
def fig_concept():
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
        ax.text(1.0, 1.02, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=11.5, color=INK2, style="italic")
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
    save(fig, "F2_timing_miss_concept")


# ---------------------------------------------------------------- F3 timelines
def tool_parts(e):
    """(sandbox start, command time, timed out) for one tool event."""
    sb = e.get("sandbox_start_s") or 0.0
    return sb, max(e["t_end"] - e["t_start"] - sb, 0.0), bool(e.get("uncacheable"))


def rollout_segments(ev):
    t0 = min(e["t"] for e in ev if e["ev"] == "rollout_start")
    rolls = sorted({e["rollout"] for e in ev if e["ev"] == "rollout_start"})
    seg = defaultdict(list)
    for e in ev:
        if e["ev"] == "gen_end":
            seg[e["rollout"]].append(("gen", e["t_start"] - t0, e["t"] - e["t_start"]))
        elif e["ev"] == "tool" and e.get("error") is None:
            sb, cmd, to = tool_parts(e)
            x = e["t_start"] - t0
            if sb:
                seg[e["rollout"]].append(("sandbox", x, sb))
            seg[e["rollout"]].append(("timeout" if to else "tool", x + sb, cmd))
        elif e["ev"] == "score":
            seg[e["rollout"]].append(("test", e["t_start"] - t0, e["t_end"] - e["t_start"]))
    reward = {e["rollout"]: e.get("reward") for e in ev if e["ev"] == "rollout_end"}
    return rolls, seg, reward


def pick_group(ev):
    """The group with the most timing-miss candidates: the most siblings that call a tool on turn 1."""
    first = defaultdict(set)
    for e in ev:
        if e["ev"] == "tool" and e.get("seq") == 1:
            first[e["uid"]].add(e["rollout"])
    return max(first, key=lambda u: len(first[u]))


COLORS = {"gen": GEN, "sandbox": SANDBOX, "tool": TOOL, "timeout": TIMEOUT, "test": TEST}
NAMES = {"gen": "LLM generation", "sandbox": "Sandbox start", "tool": "Tool call",
         "timeout": "Tool call that timed out", "test": "Reward test"}


def fig_timelines():
    fig, axes = plt.subplots(len(WORKLOADS), 1, figsize=(13, 11))
    for ax, (run, label) in zip(axes, WORKLOADS):
        ev = events(f"{run}_cache-on")
        uid = pick_group(ev)
        rolls, seg, reward = rollout_segments([e for e in ev if e.get("uid") == uid])
        for i, r in enumerate(rolls):
            for kind, x, w in seg[r]:
                ax.barh(i, max(w, 0.02), left=x, height=0.6, color=COLORS[kind], edgecolor="white", linewidth=0.8)
            end = max(x + w for _, x, w in seg[r])
            rw = reward.get(r)
            ax.text(end, i, f"  reward {rw:+.0f}" if rw is not None else "", va="center", fontsize=9, color=INK2)
        ax.set_yticks(range(len(rolls)), [f"R{i + 1}" for i in range(len(rolls))], fontsize=10)
        ax.invert_yaxis()
        task = next(e["task"] for e in ev if e.get("uid") == uid).split("|")[-1]
        ax.set_title(f"{label} · task {task}", fontsize=13)
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)
        ax.margins(x=0.08)
    axes[-1].set_xlabel("Seconds since the group's rollouts started (cache on, 1 prompt × 8 siblings)")
    kinds = list(COLORS)
    fig.legend([Patch(color=COLORS[k]) for k in kinds], [NAMES[k] for k in kinds],
               ncol=len(kinds), loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=11)
    fig.suptitle("One GRPO group per workload: generation dominates SQL; the sandbox dominates Terminal-Bench",
                 x=0.01, ha="left", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "F3_rollout_timelines")


# ---------------------------------------------------------------- F4 time budget
def fig_budget():
    data = []
    for run, label in WORKLOADS:
        ev = events(f"{run}_cache-on")
        rolls, seg, _ = rollout_segments(ev)
        n = len(rolls)
        tot = {k: sum(w for r in rolls for kk, _, w in seg[r] if kk == k) / n for k in COLORS}
        data.append((label, tot))
    fig, ax = plt.subplots(figsize=(13, 4.2))
    for i, (label, tot) in enumerate(data):
        left = 0
        total = sum(tot.values())
        for k in COLORS:
            ax.barh(i, tot[k], left=left, height=0.6, color=COLORS[k], edgecolor="white", linewidth=2)
            if tot[k] / total > 0.06 and tot[k] > 3:
                ax.text(left + tot[k] / 2, i, f"{tot[k] / total:.0%}", ha="center", va="center",
                        fontsize=11, color="white" if k in ("tool", "test") else INK)
            left += tot[k]
        ax.text(left + 0.4, i, f"{total:.1f} s", va="center", fontsize=11, color=INK2)
    ax.set_yticks(range(len(data)), [d[0] for d in data])
    ax.invert_yaxis()
    ax.set_xlabel("Seconds per rollout (mean over 640 rollouts, cache on); segments are summed, not wall-clock")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.legend([Patch(color=COLORS[k]) for k in COLORS], [NAMES[k] for k in COLORS], ncol=len(COLORS),
              loc="upper left", bbox_to_anchor=(0, -0.3), fontsize=11)
    ax.text(0, -0.62, SETUP + ". The reward test also starts a sandbox when the rollout made no tool call.",
            transform=ax.transAxes, fontsize=10.5, color=INK2)
    ax.set_title("Where a rollout's time goes: environment work is ~0% for SQL and most of it for Terminal-Bench",
                 pad=10)
    fig.tight_layout()
    save(fig, "F4_time_budget")


# ---------------------------------------------------------------- F5 latency
def fig_latency():
    fig, ax = plt.subplots(figsize=(13, 4.4))
    for i, (run, label) in enumerate(WORKLOADS):
        for j, cache in enumerate(["on", "off"]):
            ev = [e for e in events(f"{run}_cache-{cache}") if e["ev"] == "tool" and e.get("error") is None
                  and not e.get("hit")]
            parts = [tool_parts(e) for e in ev]
            xs = [cmd * 1000 for _, cmd, to in parts if not to]
            tos = [cmd * 1000 for _, cmd, to in parts if to]
            y = i + (j - 0.5) * 0.3
            ax.scatter(xs, [y] * len(xs), s=22, marker="o" if cache == "on" else "s", alpha=0.45,
                       color=TOOL if cache == "on" else INK2, linewidth=0, zorder=3,
                       label=f"cache {cache}" if i == 0 else None)
            if tos:
                ax.scatter(tos, [y] * len(tos), s=60, marker="x", color=INK, zorder=4,
                           label="timed out (600 s)" if i == 1 and j == 0 else None)
            if cache == "on":
                ax.text(4e6, i, f"median {statistics.median(xs):,.3g} ms", va="center", fontsize=11, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 3e7)
    ax.set_yticks(range(len(WORKLOADS)), [l for _, l in WORKLOADS])
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.set_xlabel("Time per cold tool call (ms, log scale; sandbox start left out); each dot is one call")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=3)
    ax.set_title("A Terminal-Bench call is ~300× slower than a SQL call", pad=10)
    fig.tight_layout()
    save(fig, "F5_tool_latency")


if __name__ == "__main__":
    fig_architecture()
    fig_concept()
    fig_timelines()
    fig_budget()
    fig_latency()
    print("written to", OUT)
