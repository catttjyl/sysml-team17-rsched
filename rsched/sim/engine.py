"""Discrete-event replay of the rollout loop under a scheduling policy.

Model
-----
* Steps run one after another; between steps the trace's training time is
  replayed unchanged. The cache persists across steps (one TVCache per run).
* Every rollout replays its own segments. Generation and overhead take their
  recorded time: holding a rollout does not change its generations (no GPU
  contention model yet; see ``gen_busy`` for the utilisation proxy).
* A cacheable call with key K at time t:
    - an execution of K already finished  -> hit (``hit_latency``)
    - an execution of K is still running  -> the policy may wait for it, up to
      ``hold_tau`` seconds. If it finishes in time: hit. If not, the rollout
      gives up after ``hold_tau`` and executes (the wait is wasted)
    - otherwise                           -> execute
* An execution costs ``cmd_s`` plus the rollout's ``sandbox_s`` if the rollout
  has no live sandbox yet. Hits never start a sandbox, so a rollout whose
  first call hits pays sandbox start on its next executed call or its test.
* Hits return the recorded output, so later turns are unchanged.

Policies
--------
``default``          never wait (what veRL does today)
``hold:<tau>``       bounded delay: wait up to tau s for an in-flight filler
``scout:<tau>``      one scout per group starts first; its siblings start when the
                     scout's first cacheable call finishes, or after tau s
``scout:<a>+hold:<b>`` both
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import Optional

from rsched.sim.trace import Rollout, Run


@dataclass(frozen=True)
class Policy:
    name: str = "default"
    hold_tau: float = 0.0
    scout_tau: Optional[float] = None

    @classmethod
    def parse(cls, spec: str) -> "Policy":
        hold, scout = 0.0, None
        for part in spec.split("+"):
            kind, _, arg = part.partition(":")
            val = math.inf if arg in ("", "inf") else float(arg)
            if kind == "default":
                continue
            if kind == "hold":
                hold = val
            elif kind == "scout":
                scout = val
            else:
                raise ValueError(f"unknown policy {part!r}")
        return cls(spec, hold, scout)


@dataclass
class CallRec:
    step: int
    rollout: str
    uid: str
    kind: str  # tool | score
    seq: int
    key: str
    t_issue: float
    t_end: float
    outcome: str  # hit | held_hit | cold | timing (filler in flight, did not wait long enough)
    wait_s: float
    cmd_s: float  # executed command seconds (0 on hits)
    sandbox_s: float  # sandbox start paid by this call
    timed_out: bool
    trace_hit: bool


@dataclass
class StepRec:
    idx: int
    t0: float
    makespan: float  # first rollout start -> last rollout end
    gen_busy: float  # seconds with >= 1 generation running
    gen_seconds: float  # sum of generation time
    rollout_ends: list[float]
    gated_s: float  # sum of scout start delays


@dataclass
class SimResult:
    run: Run
    policy: Policy
    calls: list[CallRec] = field(default_factory=list)
    steps: list[StepRec] = field(default_factory=list)


class _R:
    __slots__ = ("ro", "i", "has_sandbox", "waited_i", "wait_acc", "started", "t_start", "gens", "done", "t_ready")

    def __init__(self, ro: Rollout):
        self.ro = ro
        self.i = 0
        self.has_sandbox = False
        self.waited_i = -1  # segment index we already gave up waiting on
        self.wait_acc = 0.0
        self.started = False
        self.t_start = 0.0
        self.gens: list[tuple[float, float]] = []
        self.done = 0.0
        self.t_ready = 0.0  # when the rollout would have started without gating


def simulate(run: Run, policy: Policy) -> SimResult:
    res = SimResult(run, policy)
    fill_end: dict[str, float] = {}  # key -> earliest completion among started executions
    lat = run.hit_latency
    t = 0.0
    for si, step in enumerate(run.steps):
        t0 = t
        heap: list = []
        ctr = itertools.count()
        rs = [_R(ro) for ro in step.rollouts]

        # scout gating: per group, the earliest-starting rollout runs ahead
        scout_of: dict[str, _R] = {}
        gate_open: dict[str, Optional[float]] = {}
        waiters: dict[str, list[_R]] = {}
        if policy.scout_tau is not None:
            for r in sorted(rs, key=lambda r: r.ro.start):
                scout_of.setdefault(r.ro.uid, r)
            gate_open = {u: None for u in scout_of}
            waiters = {u: [] for u in scout_of}

        def push(when: float, r: _R, tag: str = "run") -> None:
            heapq.heappush(heap, (when, next(ctr), tag, r))

        for r in rs:
            r.t_ready = t0 + r.ro.start
            push(r.t_ready, r, "start")

        def open_gate(uid: str, now: float) -> None:
            if uid in gate_open and gate_open[uid] is None:
                gate_open[uid] = now
                for w in waiters[uid]:
                    if not w.started:
                        begin(w, max(now, w.t_ready))
                waiters[uid] = []

        def begin(r: _R, now: float) -> None:
            r.started = True
            r.t_start = now
            push(now, r)

        while heap:
            now, _, tag, r = heapq.heappop(heap)
            if tag in ("start", "gate_timeout"):
                if r.started:
                    continue
                uid = r.ro.uid
                gated = uid in scout_of and scout_of[uid] is not r and gate_open[uid] is None
                if gated and tag == "start":
                    waiters[uid].append(r)
                    if math.isfinite(policy.scout_tau):
                        push(now + policy.scout_tau, r, "gate_timeout")
                    continue
                begin(r, now)
                continue

            segs = r.ro.segs
            if r.i >= len(segs):
                r.done = now
                if scout_of.get(r.ro.uid) is r:
                    open_gate(r.ro.uid, now)
                continue
            s = segs[r.i]
            if s.kind in ("gen", "gap") or (s.kind == "score" and not s.cacheable):
                if s.kind == "gen":
                    r.gens.append((now, now + s.dur))
                r.i += 1
                push(now + s.dur, r)
                continue

            # cacheable tool call or test
            fe = fill_end.get(s.key) if s.cacheable else None
            if fe is not None and fe <= now:
                outcome = "hit" if r.wait_acc == 0.0 else "held_hit"
                _record(res, si, r, s, now - r.wait_acc, now + lat, outcome, r.wait_acc, 0.0, 0.0)
                r.wait_acc = 0.0
                r.i += 1
                push(now + lat, r)
                _after_cacheable(r, now + lat, scout_of, open_gate)
                continue
            if fe is not None and r.waited_i != r.i:
                remaining = fe - now
                if remaining <= policy.hold_tau:
                    r.wait_acc = remaining
                    push(fe, r)
                    continue
                if policy.hold_tau > 0:
                    r.wait_acc = policy.hold_tau
                    r.waited_i = r.i  # give up after tau
                    push(now + policy.hold_tau, r)
                    continue
            # execute
            outcome = "timing" if fe is not None else "cold"
            sbx = 0.0 if r.has_sandbox else r.ro.sandbox_s
            r.has_sandbox = True
            end = now + sbx + s.cmd_s
            if s.cacheable:
                fill_end[s.key] = min(fill_end.get(s.key, math.inf), end)
            _record(res, si, r, s, now - r.wait_acc, end, outcome, r.wait_acc, s.cmd_s, sbx)
            r.wait_acc = 0.0
            r.i += 1
            push(end, r)
            _after_cacheable(r, end, scout_of, open_gate)

        ends = [r.done for r in rs]
        t_last = max(ends)
        gated = sum(max(0.0, r.t_start - r.t_ready) for r in rs)
        res.steps.append(
            StepRec(si, t0, t_last - t0, _union([g for r in rs for g in r.gens]),
                    sum(b - a for r in rs for a, b in r.gens), [e - t0 for e in ends], gated)
        )
        t = t_last + (run.train_gaps[si] if si < len(run.train_gaps) else 0.0)
    return res


def _after_cacheable(r: _R, when: float, scout_of, open_gate) -> None:
    if scout_of.get(r.ro.uid) is r:
        open_gate(r.ro.uid, when)


def _record(res, si, r, s, t_issue, t_end, outcome, wait, cmd, sbx) -> None:
    res.calls.append(
        CallRec(si, r.ro.id, r.ro.uid, s.kind, s.seq, s.key, t_issue, t_end, outcome, wait,
                cmd, sbx, s.timed_out and cmd > 0, s.trace_hit)
    )


def _union(iv: list[tuple[float, float]]) -> float:
    tot, cur_a, cur_b = 0.0, None, None
    for a, b in sorted(iv):
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                tot += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        tot += cur_b - cur_a
    return tot
