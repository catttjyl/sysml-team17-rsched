"""Generation speed under load: the accelerator model for ``gpu="shared"``.

The recorded runs serve rollouts from vLLM, one replica per GPU (4 GPUs,
TP=1). A generation's decode speed falls when many generations run on its
replica at once.

Routing follows veRL v0.9.1 (``verl/workers/rollout/router.py``,
``GlobalRequestLoadBalancer``): each rollout has one ``request_id`` for its
whole conversation (``ToolAgentLoop._make_request_id``, a uuid). That id sticks
to the replica that had the fewest generation requests in flight when the
rollout's first generation arrived. Ties are broken at random in veRL and by
lowest index here. The traces don't record the replica, so ``assign_replicas``
replays the same rule on the recorded generation intervals. Load is the
number of generations running on the rollout's replica.

Model
-----
* ``SpeedCurve``: per-sequence speed as a function of per-replica load, fitted
  per run from (tokens / duration, time-averaged load) of each recorded generation,
  forced non-increasing (pool-adjacent-violators) and checked positive.
* Each recorded generation becomes an amount of work: the integral of
  ``speed(load_recorded(t))`` over its recorded interval. In the simulation it
  progresses at ``speed(load_simulated(t))``, so under the recorded schedule
  every generation ends exactly when it did. The curve's shape only matters
  for schedules that change the load.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

MIN_DUR_S = 0.05  # shorter generations give noisy speeds; left out of the fit
MIN_TOKENS = 20
MIN_BIN = 5  # loads with fewer generations are pooled with their neighbours


@dataclass(frozen=True)
class SpeedCurve:
    """Per-sequence decode speed (tok/s) at integer loads 1..len(speeds); capped beyond."""

    speeds: tuple[float, ...]

    def __post_init__(self):
        if not self.speeds:
            raise ValueError("empty speed curve")
        for a, b in zip(self.speeds, self.speeds[1:]):
            if b > a + 1e-9:
                raise ValueError(f"speed curve must not increase with load: {self.speeds}")
        if not all(math.isfinite(s) and s > 0 for s in self.speeds):
            raise ValueError(f"speed curve must be positive and finite: {self.speeds}")

    def __call__(self, load: float) -> float:
        """Speed at a (possibly fractional) load, linear between integer loads."""
        if load <= 1:
            return self.speeds[0]
        if load >= len(self.speeds):
            return self.speeds[-1]
        lo = int(load)
        f = load - lo
        return self.speeds[lo - 1] * (1 - f) + self.speeds[lo] * f

    def relative(self, load: float) -> float:
        """Speed relative to running alone."""
        return self(load) / self.speeds[0]


FLAT = SpeedCurve((1.0,))  # no contention: every generation keeps its recorded duration


class LoadProfile:
    """Number of generations running at each moment of a recorded run."""

    def __init__(self, intervals: list[tuple[float, float]]):
        pts = sorted([(a, 1) for a, b in intervals] + [(b, -1) for a, b in intervals])
        self.t: list[float] = []
        self.n: list[int] = []  # load from t[i] until t[i+1]
        load = 0
        for t, d in pts:
            load += d
            if self.t and self.t[-1] == t:
                self.n[-1] = load
            else:
                self.t.append(t)
                self.n.append(load)

    def pieces(self, a: float, b: float):
        """(duration, load) pieces covering [a, b]."""
        i = max(0, bisect.bisect_right(self.t, a) - 1)
        out = []
        while i < len(self.t) and self.t[i] < b:
            lo = max(a, self.t[i])
            hi = min(b, self.t[i + 1]) if i + 1 < len(self.t) else b
            if hi > lo:
                out.append((hi - lo, self.n[i]))
            i += 1
        return out

    def mean_load(self, a: float, b: float) -> float:
        ps = self.pieces(a, b)
        tot = sum(d for d, _ in ps)
        return sum(d * n for d, n in ps) / tot if tot > 0 else 1.0

    def work(self, a: float, b: float, curve: SpeedCurve) -> float:
        """Work of a generation recorded on [a, b], in units of relative-speed seconds."""
        return sum(d * curve.relative(n) for d, n in self.pieces(a, b))


def assign_replicas(gens: dict[str, list[tuple[float, float]]], replicas: int) -> dict[str, int]:
    """veRL's sticky least-in-flight routing, replayed on recorded generation intervals per rollout."""
    if replicas < 1:
        raise ValueError(f"replicas must be >= 1, got {replicas}")
    placed: list[list[tuple[float, float]]] = [[] for _ in range(replicas)]
    out: dict[str, int] = {}
    for rid in sorted((r for r in gens if gens[r]), key=lambda r: min(a for a, _ in gens[r])):
        t = min(a for a, _ in gens[rid])
        loads = [sum(1 for a, b in iv if a <= t < b) for iv in placed]
        k = loads.index(min(loads))
        out[rid] = k
        placed[k].extend(gens[rid])
    return out


def fit_curve(samples: list[tuple[float, int, float]]) -> SpeedCurve:
    """Fit per-sequence speed vs. load from (duration, new_tokens, mean load) of recorded generations."""
    by_load: dict[int, list[float]] = {}
    for dur, n, load in samples:
        if dur < MIN_DUR_S or not n or n < MIN_TOKENS:
            continue
        by_load.setdefault(max(1, round(load)), []).append(n / dur)
    if not by_load:
        return FLAT
    # pool sparse loads into their lower neighbour, then median per bin
    bins: list[tuple[int, list[float]]] = []
    for load in sorted(by_load):
        if bins and len(bins[-1][1]) < MIN_BIN:
            bins[-1] = (load, bins[-1][1] + by_load[load])
        else:
            bins.append((load, list(by_load[load])))
    if len(bins) > 1 and len(bins[-1][1]) < MIN_BIN:
        last = bins.pop()
        bins[-1] = (last[0], bins[-1][1] + last[1])
    pts = [(load, sorted(v)[len(v) // 2], len(v)) for load, v in bins]
    # pool adjacent violators: speed must not rise with load (weighted by count)
    blocks: list[list[float]] = []  # [speed, weight, last_load]
    for load, s, w in pts:
        blocks.append([s, w, load])
        while len(blocks) > 1 and blocks[-1][0] > blocks[-2][0]:
            s2, w2, l2 = blocks.pop()
            s1, w1, _ = blocks[-1]
            blocks[-1] = [(s1 * w1 + s2 * w2) / (w1 + w2), w1 + w2, l2]
    # piecewise-constant per block, then expand to integer loads 1..max
    speeds = []
    for s, _, last in blocks:
        while len(speeds) < last:
            speeds.append(s)
    return SpeedCurve(tuple(speeds))
