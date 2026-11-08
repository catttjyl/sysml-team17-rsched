"""Turn a recorded timeline directory into replayable rollout scripts.

Each rollout becomes an ordered list of segments, taken from its own trace:

  gen    one LLM generation turn (duration fixed from the trace)
  gap    framework overhead between two recorded events
  tool   a tool call: cache key, command seconds, whether it timed out
  score  the reward computation. On ``terminal`` it is a cached test call
         (``executor.test(history)``), keyed by the rollout's full history

Sandbox start is *not* part of a tool segment's cost. In the trace it is folded
into ``t_end - t_start`` of the rollout's first executed call (``sandbox_start_s``),
and if that call hits, the next executed call (or the test) pays it instead.
So it is a per-rollout cost (``Rollout.sandbox_s``) that the engine charges
on whichever call first needs a live sandbox. Scheduling moves it, it does not
remove it.

Rollouts are grouped into training steps by gaps in ``rollout_start`` times
(veRL launches a step's rollouts together, then trains).
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from rsched.timeline import load_events

TRACE_HIT_S = 1.0  # a recorded test call faster than this was served from the cache


@dataclass
class Seg:
    kind: str  # gen | gap | tool | score
    dur: float = 0.0  # gen / gap / non-cacheable score: fixed duration
    key: Optional[str] = None  # tool / cacheable score
    cmd_s: float = 0.0  # tool / score: execution time, sandbox start excluded
    cacheable: bool = False
    timed_out: bool = False
    seq: int = 0  # tool: position in rollout (1-based); score: 0
    trace_hit: bool = False


@dataclass
class Rollout:
    id: str
    uid: str  # GRPO group (same prompt, same step)
    task: str
    workload: str
    start: float  # offset from its step's start
    segs: list[Seg]
    sandbox_s: float  # cost of starting this rollout's sandbox


@dataclass
class Step:
    idx: int
    t0: float  # trace wall-clock start
    t_end: float  # trace wall-clock end of the last rollout
    rollouts: list[Rollout]


@dataclass
class Run:
    path: str
    workload: str
    cache: str
    steps: list[Step]
    hit_latency: float
    train_gaps: list[float] = field(default_factory=list)  # trace time between step i end and step i+1 start


def _median(xs, default):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else default


def load_run(
    path: str,
    step_gap: float = 5.0,
    cmd_timeout: float = 600.0,
    hit_latency: Optional[float] = None,
) -> Run:
    events = load_events(f"{path}/timeline" if not path.rstrip("/").endswith("timeline") else path)
    by_r: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        if "rollout" in e:
            by_r[e["rollout"]].append(e)

    starts = {r: next((e for e in evs if e["ev"] == "rollout_start"), None) for r, evs in by_r.items()}
    ends = {r: next((e for e in evs if e["ev"] == "rollout_end"), None) for r, evs in by_r.items()}
    rids = [r for r in by_r if starts[r] and ends[r]]  # drop rollouts cut off by a killed job
    if not rids:
        raise ValueError(f"no complete rollouts in {path}")
    workload = starts[rids[0]]["workload"]
    cache = starts[rids[0]]["cache"]

    tools = [e for r in rids for e in by_r[r] if e["ev"] == "tool"]
    sb = [e.get("sandbox_start_s") or 0.0 for e in tools]

    # --- per-key / per-task estimates for things a trace hit did not measure
    cmd_by_key: dict[str, list[float]] = defaultdict(list)
    sandbox_by_task: dict[str, list[float]] = defaultdict(list)
    for e, s in zip(tools, sb):
        if not e["hit"]:
            cmd_by_key[e["key"]].append(max(0.0, e["t_end"] - e["t_start"] - s))
        if s:
            sandbox_by_task[e["task"]].append(s)
    run_sandbox = _median([x for v in sandbox_by_task.values() for x in v], 0.0)
    run_cmd = _median([x for v in cmd_by_key.values() for x in v], 0.0)
    if hit_latency is None:
        hit_latency = _median([e["t_end"] - e["t_start"] for e in tools if e["hit"]], 0.005)

    def rollout_sandbox(r: str, task: str) -> float:
        own = [e.get("sandbox_start_s") for e in by_r[r] if e["ev"] == "tool" and e.get("sandbox_start_s")]
        return own[0] if own else _median(sandbox_by_task.get(task, []), run_sandbox)

    # --- test (score) calls: key and execution time without sandbox start
    test_s_by_key: dict[str, list[float]] = defaultdict(list)
    test_s_by_task: dict[str, list[float]] = defaultdict(list)
    test_info = {}
    for r in rids:
        if workload != "terminal":
            break
        evs = by_r[r]
        task = starts[r]["task"]
        ok = sorted((e for e in evs if e["ev"] == "tool" and e.get("error") is None), key=lambda e: e["seq"])
        key = "test|" + (ok[-1]["key"] if ok else task)
        sc = next((e for e in evs if e["ev"] == "score"), None)
        if sc is None:
            continue
        dur = sc["t_end"] - sc["t_start"]
        hit = cache != "off" and dur < TRACE_HIT_S
        had_sandbox = any(e["ev"] == "tool" and not e["hit"] for e in evs)
        test_info[r] = (key, hit)
        if not hit:
            t = max(0.0, dur - (0.0 if had_sandbox else rollout_sandbox(r, task)))
            test_s_by_key[key].append(t)
            test_s_by_task[task].append(t)
    run_test = _median([x for v in test_s_by_task.values() for x in v], 0.0)

    # --- rollouts -> segments
    rollouts = []
    for r in rids:
        st, en = starts[r], ends[r]
        task = st["task"]
        spans = []
        for e in by_r[r]:
            if e["ev"] == "gen_end":
                spans.append((e["t_start"], e["t"], "gen", e))
            elif e["ev"] in ("tool", "score"):
                spans.append((e["t_start"], e["t_end"], e["ev"], e))
        spans.sort(key=lambda x: x[0])
        segs: list[Seg] = []
        cur = st["t"]
        for a, b, kind, e in spans:
            if a > cur:
                segs.append(Seg("gap", dur=a - cur))
            if kind == "gen":
                segs.append(Seg("gen", dur=b - a))
            elif kind == "tool":
                s = e.get("sandbox_start_s") or 0.0
                if e["hit"]:
                    cmd = _median(cmd_by_key.get(e["key"], []), run_cmd)
                else:
                    cmd = max(0.0, b - a - s)
                segs.append(
                    Seg(
                        "tool",
                        key=e["key"],
                        cmd_s=cmd,
                        cacheable=e.get("error") is None and not e.get("uncacheable"),
                        timed_out=cmd >= cmd_timeout - 1.0,
                        seq=e["seq"],
                        trace_hit=bool(e["hit"]),
                    )
                )
            else:  # score
                if r in test_info:
                    key, hit = test_info[r]
                    if hit:
                        cmd = _median(test_s_by_key.get(key, []), _median(test_s_by_task.get(task, []), run_test))
                    else:
                        had = any(x["ev"] == "tool" and not x["hit"] for x in by_r[r])
                        cmd = max(0.0, b - a - (0.0 if had else rollout_sandbox(r, task)))
                    segs.append(Seg("score", key=key, cmd_s=cmd, cacheable=True, trace_hit=hit))
                else:
                    segs.append(Seg("score", dur=b - a))
            cur = max(cur, b)
        if en["t"] > cur:
            segs.append(Seg("gap", dur=en["t"] - cur))
        rollouts.append(
            (
                st["t"],
                en["t"],
                Rollout(r, st["uid"], task, workload, 0.0, segs, rollout_sandbox(r, task)),
            )
        )

    # --- group into steps
    rollouts.sort(key=lambda x: x[0])
    steps: list[Step] = []
    prev = None
    for t0, t1, ro in rollouts:
        if prev is None or t0 - prev > step_gap:
            steps.append(Step(len(steps), t0, t1, []))
        s = steps[-1]
        ro.start = t0 - s.t0
        s.rollouts.append(ro)
        s.t_end = max(s.t_end, t1)
        prev = t0
    gaps = [max(0.0, b.t0 - a.t_end) for a, b in zip(steps, steps[1:])]
    return Run(path, workload, cache, steps, hit_latency, gaps)
