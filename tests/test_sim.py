"""Simulator tests on a hand-written timeline (no GPU, no TVCache).

    pytest -q tests/test_sim.py
"""

from __future__ import annotations

import glob
import json
import os

import pytest

from rsched.sim import Policy, load_run, simulate, summarize

HERE = os.path.dirname(os.path.abspath(__file__))
SB = 16.0  # sandbox start
CMD = 2.0
TEST = 5.0


def _write(tmp_path, rollouts):
    """rollouts: list of (rid, [(kind, t_start, t_end, extra)]), all in group g, task T."""
    d = tmp_path / "run" / "timeline"
    d.mkdir(parents=True)
    ctx = lambda r: {"rollout": r, "task": "term|T", "uid": "g", "workload": "terminal", "cache": "tvcache"}
    lines = []
    for rid, evs in rollouts:
        t0 = evs[0][1]
        lines.append({"ev": "rollout_start", "t": t0, **ctx(rid)})
        for kind, a, b, extra in evs:
            if kind == "gen":
                lines.append({"ev": "gen_end", "t_start": a, "t": b, **ctx(rid)})
            elif kind == "tool":
                lines.append({"ev": "tool", "t": b, "t_start": a, "t_end": b, "error": None, **extra, **ctx(rid)})
            elif kind == "score":
                lines.append({"ev": "score", "t": b, "t_start": a, "t_end": b, **ctx(rid)})
        lines.append({"ev": "rollout_end", "t_start": t0, "t": evs[-1][2], **ctx(rid)})
    with open(d / "w.jsonl", "w") as fh:
        for e in sorted(lines, key=lambda e: e.get("t_start", e.get("t"))):
            fh.write(json.dumps(e) + "\n")
    return str(tmp_path / "run")


def _tool(key, seq, hit=False, sandbox=SB):
    return {"key": key, "seq": seq, "hit": hit, "sandbox_start_s": None if hit else sandbox}


@pytest.fixture
def siblings(tmp_path):
    """A fills k1 (1 -> 19, incl. sandbox start). B issues k1 at 2 and misses (timing);
    B then makes a second call k2. Both run the test."""
    a = [("gen", 0, 1, {}), ("tool", 1, 1 + SB + CMD, _tool("k1", 1)), ("gen", 19, 20, {}),
         ("score", 20, 20 + TEST, {})]
    b = [("gen", 0, 2, {}), ("tool", 2, 2 + SB + CMD, _tool("k1", 1)), ("gen", 20, 21, {}),
         ("tool", 21, 21 + CMD, _tool("k2", 2, sandbox=None)), ("gen", 23, 24, {}),
         ("score", 24, 24 + TEST, {})]
    return _write(tmp_path, [("A", a), ("B", b)])


def test_default_replays_trace(siblings):
    run = load_run(siblings)
    s = summarize(simulate(run, Policy.parse("default"), sandbox="fixed"))
    assert s["tool"]["hits"] == 0 and s["tool"]["timing"] == 1
    assert s["sandbox_s"] == pytest.approx(2 * SB)
    assert s["tool"]["cmd_s"] == pytest.approx(3 * CMD)  # sandbox start not in tool time
    assert s["makespan_mean"] == pytest.approx(29.0)  # same as the trace


def test_hold_turns_timing_miss_into_hit_and_defers_sandbox(siblings):
    res = simulate(load_run(siblings), Policy.parse("hold:inf"), sandbox="fixed")
    s = summarize(res)
    assert s["tool"]["held_hits"] == 1 and s["tool"]["timing"] == 0
    b = [c for c in res.calls if c.rollout == "B"]
    k1, k2 = b[0], b[1]
    assert k1.t_end == pytest.approx(19.0, abs=0.01)  # waited for A's fill
    assert k1.sandbox_s == 0.0
    assert k2.sandbox_s == pytest.approx(SB)  # next executed call pays the deferred start
    assert k2.replay_s == pytest.approx(CMD)  # and re-runs the k1 it hit (no snapshot)


def test_bounded_hold_gives_up(siblings):
    s = summarize(simulate(load_run(siblings), Policy.parse("hold:5")))
    # B would need to wait 17 s for A's fill: gives up after 5 s and executes
    assert s["tool"]["timing"] == 1
    assert s["tool"]["wait_s"] == pytest.approx(5.0)


def test_scout_delays_siblings(siblings):
    res = simulate(load_run(siblings), Policy.parse("scout:inf"))
    b0 = next(c for c in res.calls if c.rollout == "B")
    assert b0.outcome == "hit"  # started after A's first call finished
    assert summarize(res)["scout_delay_s"] == pytest.approx(19.0, abs=0.01)


def test_test_call_is_cached(tmp_path):
    """Two rollouts with no tool calls share the empty-history test key."""
    a = [("gen", 0, 1, {}), ("score", 1, 1 + SB + TEST, {})]
    b = [("gen", 0, 3, {}), ("score", 3, 3 + SB + TEST, {})]
    # C has a tool call, so the trace records this task's sandbox start
    c = [("gen", 0, 1, {}), ("tool", 1, 1 + SB + CMD, _tool("k9", 1)), ("score", 19, 19 + TEST, {})]
    run = load_run(_write(tmp_path, [("A", a), ("B", b), ("C", c)]))
    s = summarize(simulate(run, Policy.parse("default"), sandbox="fixed"))
    assert s["test"]["timing"] == 1
    s = summarize(simulate(run, Policy.parse("hold:inf"), sandbox="fixed"))
    assert s["test"]["hits"] == 1
    assert s["sandbox_s"] == pytest.approx(2 * SB)  # A and C; B's test hit A's, no sandbox


RUN = os.path.join(HERE, "..", "results", "tbench_full_b4n8", "tb-easy_cache-on")


@pytest.mark.skipif(not glob.glob(os.path.join(RUN, "timeline", "*.jsonl")), reason="trace not present")
def test_default_matches_recorded_run():
    run = load_run(RUN)
    s = summarize(simulate(run, Policy.parse("default"), sandbox="fixed"))
    segs = [seg for st in run.steps for r in st.rollouts for seg in r.segs]
    assert s["tool"]["hits"] == sum(seg.trace_hit for seg in segs if seg.kind == "tool")
    assert s["test"]["hits"] == sum(seg.trace_hit for seg in segs if seg.kind == "score")
    trace_mk = sum(st.t_end - st.t0 for st in run.steps) / len(run.steps)
    assert s["makespan_mean"] == pytest.approx(trace_mk, rel=0.01)
    # the lock-queue model is not exact, but should stay close on the default schedule
    q = summarize(simulate(run, Policy.parse("default")))
    assert q["makespan_mean"] == pytest.approx(trace_mk, rel=0.05)


def _slow_scout(tmp_path, first_cmd, uncacheable=False):
    """A (the scout) starts first; its first call takes ``first_cmd`` s. B is ready 0.1 s later."""
    a = [("gen", 0, 1, {}), ("tool", 1, 1 + SB + first_cmd, {**_tool("k1", 1), "uncacheable": uncacheable}),
         ("gen", 1 + SB + first_cmd, 2 + SB + first_cmd, {}),
         ("tool", 2 + SB + first_cmd, 2 + SB + first_cmd + CMD, _tool("k2", 2, sandbox=None)),
         ("score", 2 + SB + first_cmd + CMD, 2 + SB + first_cmd + CMD + TEST, {})]
    b = [("gen", 0.1, 1, {}), ("tool", 1, 1 + SB + CMD, _tool("k3", 1)), ("score", 1 + SB + CMD, 1 + SB + CMD + TEST, {})]
    return load_run(_write(tmp_path, [("A", a), ("B", b)]))


def test_scout_timeout_releases_siblings(tmp_path):
    """The scout's first call outlasts tau: siblings start after tau, not when it ends."""
    run = _slow_scout(tmp_path, first_cmd=100.0)
    s = summarize(simulate(run, Policy.parse("scout:5"), sandbox="fixed"))
    assert s["scout_delay_s"] == pytest.approx(5.0, abs=0.01)
    s = summarize(simulate(run, Policy.parse("scout:inf"), sandbox="fixed"))
    assert s["scout_delay_s"] == pytest.approx(SB + 100.0 + 1.0 - 0.1, abs=0.01)


def test_scout_gate_ignores_uncacheable_call(tmp_path):
    """A timed-out first call can't be hit, so it doesn't open the gate. Everything below it is
    uncacheable too, so here the gate only opens when the scout's rollout ends."""
    run = _slow_scout(tmp_path, first_cmd=CMD, uncacheable=True)
    res = simulate(run, Policy.parse("scout:inf"), sandbox="fixed")
    a_test = next(c for c in res.calls if c.rollout == "A" and c.kind == "score")
    b_start = min(c.t_issue for c in res.calls if c.rollout == "B")
    assert b_start >= a_test.t_end


def test_calls_below_uncacheable_are_not_cached(tmp_path):
    """Both rollouts time out on the same first call; their tests share a key but must not hit."""
    a = [("gen", 0, 1, {}), ("tool", 1, 1 + SB + CMD, {**_tool("k1", 1), "uncacheable": True}), ("score", 19, 19 + TEST, {})]
    b = [("gen", 0, 30, {}), ("tool", 30, 30 + SB + CMD, {**_tool("k1", 1), "uncacheable": True}), ("score", 48, 48 + TEST, {})]
    run = load_run(_write(tmp_path, [("A", a), ("B", b)]))
    s = summarize(simulate(run, Policy.parse("hold:inf")))
    assert s["tool"]["hits"] == 0 and s["test"]["hits"] == 0
    assert s["test"]["uncacheable"] == 2


def test_sandbox_lock_queue(tmp_path):
    """Starts go through one lock: the second start waits for the first."""
    a = [("gen", 0, 1, {}), ("tool", 1, 1 + SB + CMD, _tool("k1", 1)), ("score", 19, 19 + TEST, {})]
    b = [("gen", 0, 1, {}), ("tool", 1, 1 + SB + CMD, _tool("k2", 1)), ("score", 19, 19 + TEST, {})]
    run = load_run(_write(tmp_path, [("A", a), ("B", b)]))
    run.sandbox_svc = 3.0
    res = simulate(run, Policy.parse("default"))
    waits = sorted(c.sandbox_s for c in res.calls if c.sandbox_s)
    assert waits == pytest.approx([3.0, 6.0])


def test_reuse_bound_is_policy_independent(siblings):
    run = load_run(siblings)
    bounds = [summarize(simulate(run, Policy.parse(p)))["tool"]["reuse_bound"] for p in ("default", "hold:inf", "scout:inf")]
    assert bounds == pytest.approx([1 / 3] * 3)
