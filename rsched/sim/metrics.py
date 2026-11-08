"""Summary numbers for a simulated (or replayed) run.

Tool time is command execution only: sandbox start is reported separately,
the same convention as ``scripts/plot.py`` and the tbench README. Sandbox
start is a per-rollout cost that scheduling can move but not remove.
"""

from __future__ import annotations

import statistics

from rsched.sim.engine import SimResult


def _rate(n, d):
    return n / d if d else 0.0


def summarize(res: SimResult) -> dict:
    tools = [c for c in res.calls if c.kind == "tool"]
    tests = [c for c in res.calls if c.kind == "score"]

    def block(cs):
        hits = sum(c.outcome in ("hit", "held_hit") for c in cs)
        timing = sum(c.outcome == "timing" for c in cs)
        return {
            "calls": len(cs),
            "hits": hits,
            "held_hits": sum(c.outcome == "held_hit" for c in cs),
            "timing": timing,
            "cold": sum(c.outcome == "cold" for c in cs),
            "hit_rate": _rate(hits, len(cs)),
            "upper_bound": _rate(hits + timing, len(cs)),
            "cmd_s": sum(c.cmd_s for c in cs),
            "cmd_s_no_timeout": sum(c.cmd_s for c in cs if not c.timed_out),
            "wait_s": sum(c.wait_s for c in cs),
        }

    steps = res.steps
    mk = [s.makespan for s in steps]
    tail = [s.makespan - statistics.median(s.rollout_ends) for s in steps if s.rollout_ends]
    return {
        "policy": res.policy.name,
        "run": res.run.path,
        "steps": len(steps),
        "tool": block(tools),
        "test": block(tests),
        "sandbox_s": sum(c.sandbox_s for c in res.calls),
        "timed_out_calls": sum(c.timed_out for c in tools),
        "makespan_mean": statistics.mean(mk) if mk else 0.0,
        "makespan_total": sum(mk),
        "gen_busy_frac": _rate(sum(s.gen_busy for s in steps), sum(mk)),
        "gen_seconds": sum(s.gen_seconds for s in steps),
        "straggler_tail_mean": statistics.mean(tail) if tail else 0.0,
        "scout_delay_s": sum(s.gated_s for s in steps),
    }
