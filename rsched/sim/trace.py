"""Turn a recorded timeline directory into replayable rollout scripts.

Each rollout becomes an ordered list of segments, taken from its own trace:

  gen    one LLM generation turn (duration fixed from the trace)
  gap    framework overhead between two recorded events
  tool   a tool call: cache key, command seconds, whether it timed out
  score  the reward computation. On ``terminal`` it is a cached test call
         (``executor.test(history)``), keyed by the rollout's full history

Sandbox start is *not* part of a tool segment's cost. In the trace it is folded
into ``t_end - t_start`` of the rollout's first executed call (``sandbox_start_s``),
or into the test of a rollout with no executed call. Starts are serialised by a
node-wide ``flock`` (``terminal_podman.py``), so most of that time is queueing:
each start holds the lock for ``Run.sandbox_svc`` (about 1.2 s) and the rest is
waiting for the other rollouts' starts. The engine models that queue; the
recorded per-rollout time (``Rollout.sandbox_s``) is kept for the ``fixed``
sensitivity mode. A rollout whose calls and test all hit never starts one.

For a rollout with no executed tool call the trace does not log its sandbox
start, so it is reconstructed: replay the lock as a FIFO over all recorded
start requests, anchored on the logged ready times, with ``sandbox_svc`` per start.

TVCache keeps no snapshots in these runs (no ``fork`` events), so an execution
after a hit re-runs the hit calls on the sandbox first (``source=fresh,
executed>1``). Those replay seconds are taken out of the segment's ``cmd_s`` here
and charged again by the engine whenever the simulated rollout is behind.

Assumption: after an uncacheable call (a timeout, not inserted), the rest of the
rollout, its test included, is uncacheable too. One trace case (tb-medium,
rollouts 4479511a / 3b254b9e) shows a repeated test below a timeout missing.

Rollouts are grouped into training steps by gaps in ``rollout_start`` times
(veRL launches a step's rollouts together, then trains).
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from rsched.timeline import load_events

TRACE_HIT_S = 1.0  # a recorded test call faster than this was served from the cache
TEST_FILL_LEAD_S = 0.5  # a test result is inserted before the sandbox is stopped (trace: up to 0.43 s before t_end)


@dataclass
class Seg:
    kind: str  # gen | gap | tool | score
    dur: float = 0.0  # gen / gap / non-cacheable score: fixed duration
    key: Optional[str] = None  # tool / cacheable score
    cmd_s: float = 0.0  # tool / score: execution time, sandbox start excluded
    cacheable: bool = False
    timed_out: bool = False
    fill_lead: float = 0.0  # the result is visible to others this long before the call ends
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
    sandbox_s: float  # recorded (queued) sandbox start of this rollout, for the fixed model


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
    test_hit_latency: float = 0.5
    sandbox_svc: float = 1.2  # seconds one sandbox start holds the start lock
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

    tool_evs = {r: sorted((e for e in by_r[r] if e["ev"] == "tool"), key=lambda e: e["seq"]) for r in rids}
    score_ev = {r: next((e for e in by_r[r] if e["ev"] == "score"), None) for r in rids}

    # --- the start lock: service time from uncontended starts, then a FIFO replay
    known = sorted((e["t_start"], e["t_start"] + e["sandbox_start_s"]) for e in tools if e.get("sandbox_start_s"))
    lone = [b - a for a, b in known if not any(a2 < a < b2 for a2, b2 in known)]
    svc = _median(lone, 1.2)
    reqs = []  # (request time, logged ready time or None, rollout)
    for r in rids:
        first = next((e for e in tool_evs[r] if e.get("sandbox_start_s")), None)
        sc = score_ev[r]
        if first is not None:
            reqs.append((first["t_start"], first["t_start"] + first["sandbox_start_s"], r))
        elif (workload == "terminal" and sc is not None and not any(not e["hit"] for e in tool_evs[r])
              and not (cache != "off" and sc["t_end"] - sc["t_start"] < TRACE_HIT_S)):
            reqs.append((sc["t_start"], None, r))
    recon: dict[str, float] = {}
    lock_free = -math.inf
    for a, b, r in sorted(reqs, key=lambda x: x[0]):
        ready = b if b is not None else max(a, lock_free) + svc
        lock_free = max(lock_free, ready)
        recon[r] = ready - a

    def rollout_sandbox(r: str, task: str) -> float:
        if r in recon:
            return recon[r]
        return _median(sandbox_by_task.get(task, []), run_sandbox)

    def replay_before(r: str, upto_seq: float) -> float:
        """Trace seconds spent re-running hit calls before the execution at ``upto_seq``."""
        tot = 0.0
        for e in tool_evs[r]:
            if e["seq"] >= upto_seq:
                break
            # hits since the last execution are re-run; an execution brings the sandbox up to date
            tot = tot + _median(cmd_by_key.get(e["key"], []), run_cmd) if e["hit"] else 0.0
        return tot

    def broken_from(r: str) -> float:
        """seq of the first uncacheable call; later calls and the test are not cached."""
        return next((e["seq"] for e in tool_evs[r] if e.get("error") is not None or e.get("uncacheable")), math.inf)

    # --- test (score) calls: key and execution time without sandbox start
    test_s_by_key: dict[str, list[float]] = defaultdict(list)
    test_s_by_task: dict[str, list[float]] = defaultdict(list)
    test_info = {}
    test_cold: dict[str, float] = {}
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
        test_info[r] = (key, hit, broken_from(r) == math.inf)
        if not hit:
            t = max(0.0, dur - (0.0 if had_sandbox else rollout_sandbox(r, task)) - replay_before(r, math.inf))
            test_cold[r] = t
            test_s_by_key[key].append(t)
            test_s_by_task[task].append(t)
    run_test = _median([x for v in test_s_by_task.values() for x in v], 0.0)
    test_hits = [score_ev[r]["t_end"] - score_ev[r]["t_start"] for r in test_info if test_info[r][1]]
    test_hit_latency = _median(test_hits, 0.5)

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
                    cmd = max(0.0, b - a - s - replay_before(r, e["seq"]))
                segs.append(
                    Seg(
                        "tool",
                        key=e["key"],
                        cmd_s=cmd,
                        cacheable=e["seq"] < broken_from(r) and e.get("error") is None and not e.get("uncacheable"),
                        timed_out=cmd >= cmd_timeout - 1.0,
                        seq=e["seq"],
                        trace_hit=bool(e["hit"]),
                    )
                )
            else:  # score
                if r in test_info:
                    key, hit, ok = test_info[r]
                    if hit:
                        cmd = _median(test_s_by_key.get(key, []), _median(test_s_by_task.get(task, []), run_test))
                    else:
                        cmd = test_cold[r]
                    segs.append(Seg("score", key=key, cmd_s=cmd, cacheable=ok, trace_hit=hit,
                                    fill_lead=min(TEST_FILL_LEAD_S, cmd)))
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
    return Run(path, workload, cache, steps, hit_latency, test_hit_latency, svc, gaps)
