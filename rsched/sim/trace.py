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
an uncontended start takes about 1.2 s and the rest is waiting for the other
rollouts' starts. ``Run.sandbox_svc`` is calibrated by the engine. The engine models that queue; the
recorded per-rollout time (``Rollout.sandbox_s``) is kept for the ``fixed``
sensitivity mode. A rollout whose calls and test all hit never starts one.

For a rollout with no executed tool call the trace does not log its sandbox
start, so it is reconstructed: replay the lock as a FIFO over all recorded
start requests, anchored on the logged ready times, with the uncontended start
time per start. flock is not FIFO, so the result is capped to leave the test at
least the 10th-percentile cold test time.

TVCache snapshots adaptively (``adaptive_snapshots=True`` in the agent loop):
only expensive calls get one. ``tvcache_dump.json`` lists them (nodes with a
``sandbox_id``); without a dump (cache-off traces), a call is assumed
snapshotted if its command takes >= ``SNAPSHOT_MIN_S``, which misses none of
the dumped snapshots in the TB runs but over-includes a few. After a hit on a
snapshotted key, the next execution forks from the snapshot (a new sandbox
start through the lock) and re-runs only the hits after it. After a hit on a
key without one, the next execution re-runs the hit calls on the sandbox first
(``source=fresh, executed>1``; two such calls in the traces). Those replay
seconds are taken out of the segment's ``cmd_s`` here and charged again by the
engine whenever the simulated rollout is behind. A snapshotted call's own
``cmd_s`` still includes taking the snapshot (about 5-8 s on the few keys seen
in both cache-on and cache-off runs).

Assumption: after an uncacheable call (a timeout, not inserted), the rest of the
rollout, its test included, is uncacheable too. One trace case (tb-medium,
rollouts 4479511a / 3b254b9e) shows a repeated test below a timeout missing.

Rollouts are grouped into training steps by gaps in ``rollout_start`` times
(veRL launches a step's rollouts together, then trains).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from rsched.sim.gpu import FLAT, LoadProfile, SpeedCurve, assign_replicas, fit_curve
from rsched.timeline import load_events

SNAPSHOT_MIN_S = 2.5  # cache-off fallback: commands at least this long are assumed snapshotted
TRACE_HIT_S = 1.0  # a recorded test call faster than this was served from the cache
TEST_FILL_LEAD_S = 0.5  # a test result is inserted before the sandbox is stopped (trace: up to 0.43 s before t_end)


@dataclass
class Seg:
    kind: str  # gen | gap | tool | score
    work: float = 0.0  # gen: recorded duration x relative speed at the recorded load (see gpu.py)
    dur: float = 0.0  # gen / gap / non-cacheable score: fixed duration
    key: Optional[str] = None  # tool / cacheable score
    cmd_s: float = 0.0  # tool / score: execution time, sandbox start excluded
    cacheable: bool = False
    timed_out: bool = False
    error: bool = False  # the call raised: the executor closes the sandbox
    snap: bool = False  # tool: TVCache keeps a snapshot of the sandbox after this call
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
    replica: int = -1  # vLLM replica reconstructed from the trace (veRL routing), for checks


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
    test_hit_latency: float = 0.5  # test hit in a rollout with a live sandbox (it is stopped too)
    test_hit_latency_cold: float = 0.0  # test hit in a rollout that never started one
    sandbox_svc: Optional[float] = None  # lock seconds per start; None: calibrate (engine.lock_service)
    has_lock: bool = False  # the trace has podman sandbox starts behind the start lock
    lone_svc: float = 1.2  # median uncontended start, for reference
    gpu: SpeedCurve = FLAT  # per-sequence generation speed vs. per-replica load, fitted on this run
    replicas: int = 4  # vLLM replicas (GPUs / tensor-parallel size), from train.log
    train_gaps: list[float] = field(default_factory=list)  # trace time between step i end and step i+1 start


def _median(xs, default):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else default


def _replicas(path: str, default: int = 4) -> int:
    """vLLM replicas = rollout GPUs / tensor-parallel size, read from the run's train.log."""
    import re

    log = os.path.join(path.rstrip("/").removesuffix("/timeline"), "train.log")
    if not os.path.exists(log):
        return default
    with open(log, errors="replace") as fh:
        text = fh.read(2_000_000)
    g = re.search(r"'n_gpus_per_node': (\d+)", text)
    tp = re.search(r"'tensor_model_parallel_size': (\d+)", text)
    if not (g and tp) or int(tp.group(1)) < 1:
        return default
    return max(1, int(g.group(1)) // int(tp.group(1)))


def _snapshot_keys(path: str) -> Optional[set]:
    """Timeline keys of calls TVCache kept a snapshot for, from the end-of-run cache dump."""
    dump = os.path.join(path.rstrip("/").removesuffix("/timeline"), "tvcache_dump.json")
    if not os.path.exists(dump):
        return None
    with open(dump) as fh:
        tree = json.load(fh)
    keys: set = set()

    def walk(task: str, node: dict, prefix: list) -> None:
        for ser, child in node.get("children", {}).items():
            p = prefix + [ser]  # terminal calls all mutate state, so the key path is the full chain
            if child.get("sandbox_id"):
                keys.add(hashlib.sha1(json.dumps([task, p]).encode()).hexdigest()[:16])
            walk(task, child, p)

    for task, t in tree.items():
        walk(task, t.get("root", {}), [])
    return keys


def load_run(
    path: str,
    step_gap: float = 5.0,
    cmd_timeout: float = 600.0,
    hit_latency: Optional[float] = None,
) -> Run:
    events = load_events(f"{path}/timeline" if not path.rstrip("/").endswith("timeline") else path)
    snap_keys = _snapshot_keys(path)
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
        # executed>1: the call also re-ran earlier hits; keep only clean single executions here
        if (not e["hit"] and (e.get("executed") or 1) <= 1 and e.get("error") is None
                and not e.get("uncacheable")):
            cmd_by_key[e["key"]].append(max(0.0, e["t_end"] - e["t_start"] - s))
        if s:
            sandbox_by_task[e["task"]].append(s)
    run_sandbox = _median([x for v in sandbox_by_task.values() for x in v], 0.0)
    run_cmd = _median([x for v in cmd_by_key.values() for x in v], 0.0)
    if hit_latency is None:
        hit_latency = _median([e["t_end"] - e["t_start"] for e in tools if e["hit"]], 0.005)

    # an errored call is logged with the previous call's seq (it is popped first), so break ties by time
    tool_evs = {r: sorted((e for e in by_r[r] if e["ev"] == "tool"), key=lambda e: (e["seq"], e["t_start"]))
                for r in rids}
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
    # flock is not FIFO, so a FIFO replay can give a start longer than the whole test. Cap it so the
    # test keeps at least the 10th-percentile cold test time of rollouts that had a live sandbox.
    live_tests = [score_ev[r]["t_end"] - score_ev[r]["t_start"] for r in rids
                  if score_ev[r] is not None and any(not e["hit"] for e in tool_evs[r])
                  and not (cache != "off" and score_ev[r]["t_end"] - score_ev[r]["t_start"] < TRACE_HIT_S)]
    test_floor = statistics.quantiles(live_tests, n=10)[0] if len(live_tests) > 1 else 0.0
    recon: dict[str, float] = {}
    lock_free = -math.inf
    for a, b, r in sorted(reqs, key=lambda x: x[0]):
        if b is not None:
            recon[r] = b - a
            lock_free = max(lock_free, b)
            continue
        dur = score_ev[r]["t_end"] - score_ev[r]["t_start"]
        recon[r] = min(max(a, lock_free) + svc - a, max(svc, dur - test_floor))
        lock_free = max(lock_free, a + recon[r])

    def rollout_sandbox(r: str, task: str) -> float:
        if r in recon:
            return recon[r]
        return _median(sandbox_by_task.get(task, []), run_sandbox)

    def replay_before(r: str, upto_seq: float) -> float:
        """Trace seconds spent re-running hit calls before the execution at ``upto_seq``."""
        tot = hist = 0.0  # owed replay; successful history since the last snapshot
        for e in tool_evs[r]:
            if e["seq"] >= upto_seq:
                break
            m = _median(cmd_by_key.get(e["key"], []), run_cmd)
            if e.get("error") is not None:  # the sandbox is closed: a new one replays everything
                tot = hist
            elif is_snap(e["key"]):  # forked from (or took) a snapshot: nothing before it is re-run
                tot = hist = 0.0
            elif e["hit"]:  # hits since the last execution are re-run
                tot += m
                hist += m
            else:  # an execution brings the sandbox up to date
                tot = 0.0
                hist += m
        return tot

    def is_snap(key: str) -> bool:
        if snap_keys is not None:
            return key in snap_keys
        return _median(cmd_by_key.get(key, []), 0.0) >= SNAPSHOT_MIN_S

    def broken_from(r: str) -> float:
        """seq of the first uncacheable call (a timeout); later calls and the test are not cached.
        A call that raised is dropped from the history instead, so it doesn't break the chain."""
        return next((e["seq"] for e in tool_evs[r] if e.get("uncacheable")), math.inf)

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
        execd = [e for e in tool_evs[r] if not e["hit"]]
        had_sandbox = bool(execd) and execd[-1].get("error") is None  # an error closes it
        test_info[r] = (key, hit, broken_from(r) == math.inf)
        if not hit:
            t = max(0.0, dur - (0.0 if had_sandbox else rollout_sandbox(r, task)) - replay_before(r, math.inf))
            test_cold[r] = t
            test_s_by_key[key].append(t)
            test_s_by_task[task].append(t)
    run_test = _median([x for v in test_s_by_task.values() for x in v], 0.0)
    # a test hit in a rollout with a live sandbox also stops it (~0.5 s); without one it is ~0
    live = {r for r in test_info if any(not e["hit"] for e in tool_evs[r])}
    test_hits = {r: score_ev[r]["t_end"] - score_ev[r]["t_start"] for r in test_info if test_info[r][1]}
    test_hit_latency = _median([v for r, v in test_hits.items() if r in live], 0.5)
    test_hit_latency_cold = _median([v for r, v in test_hits.items() if r not in live], 0.0)
    if cache == "off":  # a cache-off trace has no hits to measure; use what cache-on runs show
        test_hit_latency, test_hit_latency_cold = 0.5, 0.0

    # --- generation speed under load (gpu="shared"): per replica, routed as veRL does
    replicas = _replicas(path)
    gen_iv = {r: [(e["t_start"], e["t"]) for e in by_r[r] if e["ev"] == "gen_end"] for r in rids}
    replica_of = assign_replicas(gen_iv, replicas)
    profiles = [LoadProfile([iv for r in rids if replica_of.get(r) == k for iv in gen_iv[r]]) for k in range(replicas)]
    curve = fit_curve([(e["t"] - e["t_start"], e.get("new_tokens") or 0,
                        profiles[replica_of[r]].mean_load(e["t_start"], e["t"]))
                       for r in rids for e in by_r[r] if e["ev"] == "gen_end"])

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
                segs.append(Seg("gen", dur=b - a, work=profiles[replica_of[r]].work(a, b, curve)))
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
                        timed_out=bool(e.get("uncacheable")) or cmd >= cmd_timeout - 1.0,
                        error=e.get("error") is not None,
                        seq=e["seq"],
                        trace_hit=bool(e["hit"]),
                        snap=is_snap(e["key"]),
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
                Rollout(r, st["uid"], task, workload, 0.0, segs, rollout_sandbox(r, task), replica_of.get(r, -1)),
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
    return Run(path, workload, cache, steps, hit_latency, test_hit_latency, test_hit_latency_cold,
               None if known else 0.0, bool(known), svc, curve, replicas, gaps)
