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
* An execution costs ``cmd_s``, plus a sandbox start if the rollout has none
  yet, plus re-running any calls it hit since its sandbox was last brought up
  to date (no snapshots in these runs, so a miss after a hit replays them).
* Sandbox starts (``sandbox="queue"``, the default) go through one FIFO lock
  shared by the whole run, as ``flock`` does on the node: each start holds it
  for ``run.sandbox_svc`` seconds, calibrated so that the default replay's
  total sandbox time equals the trace's (``lock_service``). ``sandbox="fixed"``
  instead charges each rollout its recorded start time wherever it starts
  (exact trace replay). Runs without podman starts (SQL) have no lock.
  Hits never start a sandbox; a rollout whose calls and test all hit has none.
  A call that raises closes the sandbox (the executor does), so the next
  execution starts a new one.
* A finished execution is visible ``fill_lead`` seconds before its end (a test
  result is inserted before the sandbox is stopped).
* Hits return the recorded output, so later turns are unchanged.

Policies
--------
``default``          never wait (what veRL does today)
``hold:<tau>``       bounded delay: wait up to tau s for an in-flight filler
``hold:cost``        wait only if the fill will be visible before this rollout could
                     finish the call itself (sandbox start + replay + command).
                     An oracle: it uses the call's true command time
``scout:<tau>``      one scout per group starts first; its siblings start when the
                     scout's first cacheable call (tool or test) finishes, or tau s
                     after they were ready, whichever is first
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
    hold_cost: bool = False

    @classmethod
    def parse(cls, spec: str) -> "Policy":
        hold, scout, cost = 0.0, None, False
        for part in spec.split("+"):
            kind, _, arg = part.partition(":")
            if kind == "hold" and arg == "cost":
                cost = True
                continue
            val = math.inf if arg in ("", "inf") else float(arg)
            if kind == "default":
                continue
            if kind == "hold":
                hold = val
            elif kind == "scout":
                scout = val
            else:
                raise ValueError(f"unknown policy {part!r}")
        return cls(spec, hold, scout, cost)


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
    outcome: str  # hit | held_hit | cold | timing (filler in flight, did not wait long enough) | uncacheable
    wait_s: float
    cmd_s: float  # executed command seconds (0 on hits)
    sandbox_s: float  # sandbox start paid by this call (queue wait + start)
    replay_s: float  # re-running earlier hit calls before this one
    timed_out: bool
    trace_hit: bool
    cacheable: bool


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
    __slots__ = ("ro", "i", "has_sandbox", "waited_i", "wait_acc", "started", "t_start", "gens", "done", "t_ready",
                 "stale", "hist")

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
        self.stale = 0.0  # command seconds of hit calls not yet applied to the sandbox
        self.hist = 0.0  # command seconds of all successful calls (replayed after an error)


def lock_service(run: Run) -> float:
    """Seconds each start holds the lock, so that the default replay's sandbox total matches the trace.

    The median uncontended start (~1.2 s) under-predicts the total by 10-23%: some
    starts are slow (pasta retries, up to ~35 s) and podman slows under load.
    """
    if run.sandbox_svc is None:
        target = sum(c.sandbox_s for c in simulate(run, Policy(), sandbox="fixed").calls)
        lo, hi = 0.0, 120.0
        for _ in range(30):
            run.sandbox_svc = (lo + hi) / 2
            got = sum(c.sandbox_s for c in simulate(run, Policy()).calls)
            lo, hi = (run.sandbox_svc, hi) if got < target else (lo, run.sandbox_svc)
        run.sandbox_svc = (lo + hi) / 2
    return run.sandbox_svc


def simulate(run: Run, policy: Policy, sandbox: str = "queue") -> SimResult:
    if sandbox not in ("queue", "fixed"):
        raise ValueError(f"sandbox must be 'queue' or 'fixed', not {sandbox!r}")
    if not run.has_lock:
        sandbox = "fixed"  # no podman starts in this trace: nothing to queue
    svc = lock_service(run) if sandbox == "queue" else 0.0
    res = SimResult(run, policy)
    fill_end: dict[str, float] = {}  # key -> earliest time a started execution's result is visible
    lock_free = -math.inf  # sandbox start lock, shared across steps
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
            if gate_open.get(uid, 0.0) is None:
                gate_open[uid] = now
                for w in waiters[uid]:
                    if not w.started:
                        begin(w, max(now, w.t_ready))
                waiters[uid] = []

        def gate_after(r: _R, s, when: float) -> None:
            """The scout's first cacheable call opens its group's gate when it finishes."""
            if s.cacheable and scout_of.get(r.ro.uid) is r and gate_open[r.ro.uid] is None:
                push(when, r, "gate_open")

        def begin(r: _R, now: float) -> None:
            r.started = True
            r.t_start = now
            push(now, r)

        while heap:
            now, _, tag, r = heapq.heappop(heap)
            if tag == "gate_open":
                open_gate(r.ro.uid, now)
                continue
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
            if s.kind in ("gen", "gap") or (s.kind == "score" and s.key is None):  # SQL scoring: not a cache call
                if s.kind == "gen":
                    r.gens.append((now, now + s.dur))
                r.i += 1
                push(now + s.dur, r)
                continue

            # tool call or test
            fe = fill_end.get(s.key) if s.cacheable else None
            if fe is not None and fe <= now:
                if s.kind == "score":
                    lat = run.test_hit_latency if r.has_sandbox else run.test_hit_latency_cold
                else:
                    lat = run.hit_latency
                outcome = "hit" if r.wait_acc == 0.0 else "held_hit"
                _record(res, si, r, s, now - r.wait_acc, now + lat, outcome, r.wait_acc, 0.0, 0.0, 0.0)
                r.wait_acc = 0.0
                r.stale += s.cmd_s
                r.hist += s.cmd_s
                r.i += 1
                push(now + lat, r)
                gate_after(r, s, now + lat)
                continue
            if fe is not None and policy.hold_cost:
                limit = _local_cost(r, s, now, sandbox, svc, lock_free)
                if policy.hold_tau > 0:  # hold:cost+hold:<tau>: both limits apply
                    limit = min(limit, policy.hold_tau)
                if fe - now <= limit:
                    r.wait_acc = fe - now
                    push(fe, r)
                    continue
            elif fe is not None and r.waited_i != r.i:
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
            outcome = "timing" if fe is not None else ("cold" if s.cacheable else "uncacheable")
            sbx = 0.0
            if not r.has_sandbox:
                if sandbox == "queue":
                    ready = max(now, lock_free) + svc
                    lock_free = ready
                    sbx = ready - now
                else:
                    sbx = r.ro.sandbox_s
                r.has_sandbox = True
            replay, r.stale = r.stale, 0.0
            end = now + sbx + replay + s.cmd_s
            if s.cacheable:
                fill_end[s.key] = min(fill_end.get(s.key, math.inf), end - s.fill_lead)
            _record(res, si, r, s, now - r.wait_acc, end, outcome, r.wait_acc, s.cmd_s, sbx, replay)
            if s.error:  # the executor closes the sandbox; a new one must replay the history
                r.has_sandbox = False
                r.stale = r.hist
            else:
                r.hist += s.cmd_s
            r.wait_acc = 0.0
            r.i += 1
            push(end, r)
            gate_after(r, s, end)

        ends = [r.done for r in rs]
        t_last = max(ends)
        gated = sum(max(0.0, r.t_start - r.t_ready) for r in rs)
        res.steps.append(
            StepRec(si, t0, t_last - t0, _union([g for r in rs for g in r.gens]),
                    sum(b - a for r in rs for a, b in r.gens), [e - t0 for e in ends], gated)
        )
        t = t_last + (run.train_gaps[si] if si < len(run.train_gaps) else 0.0)
    return res


def _local_cost(r: _R, s, now: float, sandbox: str, svc: float, lock_free: float) -> float:
    """Seconds until this rollout's own execution of the call would finish."""
    sbx = 0.0
    if not r.has_sandbox:
        sbx = max(now, lock_free) + svc - now if sandbox == "queue" else r.ro.sandbox_s
    return sbx + r.stale + s.cmd_s


def _record(res, si, r, s, t_issue, t_end, outcome, wait, cmd, sbx, replay) -> None:
    res.calls.append(
        CallRec(si, r.ro.id, r.ro.uid, s.kind, s.seq, s.key, t_issue, t_end, outcome, wait,
                cmd, sbx, replay, s.timed_out and cmd > 0, s.trace_hit, s.cacheable)
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
