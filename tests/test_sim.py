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
    s = summarize(simulate(run, Policy.parse("default")))
    assert s["tool"]["hits"] == 0 and s["tool"]["timing"] == 1
    assert s["sandbox_s"] == pytest.approx(2 * SB)
    assert s["tool"]["cmd_s"] == pytest.approx(3 * CMD)  # sandbox start not in tool time
    assert s["makespan_mean"] == pytest.approx(29.0)  # same as the trace


def test_hold_turns_timing_miss_into_hit_and_defers_sandbox(siblings):
    res = simulate(load_run(siblings), Policy.parse("hold:inf"))
    s = summarize(res)
    assert s["tool"]["held_hits"] == 1 and s["tool"]["timing"] == 0
    b = [c for c in res.calls if c.rollout == "B"]
    k1, k2 = b[0], b[1]
    assert k1.t_end == pytest.approx(19.0, abs=0.01)  # waited for A's fill
    assert k1.sandbox_s == 0.0
    assert k2.sandbox_s == pytest.approx(SB)  # next executed call pays the deferred start
    assert s["sandbox_s"] == pytest.approx(2 * SB)  # moved, not removed


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
    s = summarize(simulate(run, Policy.parse("default")))
    assert s["test"]["timing"] == 1
    s = summarize(simulate(run, Policy.parse("hold:inf")))
    assert s["test"]["hits"] == 1
    assert s["sandbox_s"] == pytest.approx(2 * SB)  # A and C; B's test hit A's, no sandbox


RUN = os.path.join(HERE, "..", "results", "tbench_full_b4n8", "tb-easy_cache-on")


@pytest.mark.skipif(not glob.glob(os.path.join(RUN, "timeline", "*.jsonl")), reason="trace not present")
def test_default_matches_recorded_run():
    run = load_run(RUN)
    s = summarize(simulate(run, Policy.parse("default")))
    trace_hits = sum(seg.trace_hit for st in run.steps for r in st.rollouts for seg in r.segs if seg.kind == "tool")
    assert s["tool"]["hits"] == trace_hits
    trace_mk = sum(st.t_end - st.t0 for st in run.steps) / len(run.steps)
    assert s["makespan_mean"] == pytest.approx(trace_mk, rel=0.01)
