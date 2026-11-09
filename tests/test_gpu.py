"""The accelerator model (gpu="shared") on its own and inside the engine.

    pytest -q tests/test_gpu.py
"""

from __future__ import annotations

import pytest

from rsched.sim import Policy, load_run, simulate, summarize
from rsched.sim.gpu import LoadProfile, SpeedCurve, fit_curve
from test_sim import _write

# --- the curve: invalid parameters are rejected when it is built (a speed of 0 would never finish)


def test_curve_rejects_bad_parameters():
    with pytest.raises(ValueError):
        SpeedCurve(())
    with pytest.raises(ValueError):
        SpeedCurve((200.0, 0.0))
    with pytest.raises(ValueError):
        SpeedCurve((100.0, 150.0))  # faster under more load
    with pytest.raises(ValueError):
        SpeedCurve((float("nan"),))


def test_curve_interpolates_and_caps():
    c = SpeedCurve((200.0, 100.0))
    assert c(0.5) == 200.0 and c(1.5) == pytest.approx(150.0) and c(10) == 100.0
    assert c.relative(2) == pytest.approx(0.5)


def test_fit_is_non_increasing():
    # load 1: fast; load 2 (two overlapping gens): slower; a noisy fast pair at load 3 gets pooled down
    gens = [(0, 1, 200)] * 6 + [(10, 12, 200), (10, 12, 200)] * 3 + [(20, 21, 300)] * 9
    prof = LoadProfile([(a, b) for a, b, _ in gens])
    c = fit_curve(gens, prof)
    assert all(b <= a for a, b in zip(c.speeds, c.speeds[1:]))


def test_profile_work_matches_recorded_load():
    prof = LoadProfile([(0, 10), (5, 10)])
    c = SpeedCurve((100.0, 50.0))
    # first gen: 5 s alone (rel 1) + 5 s with two running (rel 0.5)
    assert prof.work(0, 10, c) == pytest.approx(7.5)
    assert prof.mean_load(0, 10) == pytest.approx(1.5)


# --- inside the engine


def _two_siblings(tmp_path):
    """A and B generate together for 10 s (load 2), then each makes a call and runs its test."""
    tool = {"key": None, "seq": 1, "hit": False, "sandbox_start_s": 1.0}
    a = [("gen", 0, 10, {}), ("tool", 10, 12, {**tool, "key": "k1"}), ("score", 12, 14, {})]
    b = [("gen", 0, 10, {}), ("tool", 10, 12, {**tool, "key": "k2"}), ("score", 12, 14, {})]
    run = load_run(_write(tmp_path, [("A", a), ("B", b)]))
    run.gpu = SpeedCurve((100.0, 50.0))  # alone: twice as fast
    for st in run.steps:
        for ro in st.rollouts:
            for s in ro.segs:
                if s.kind == "gen":
                    s.work = 5.0  # 10 s at relative speed 0.5
    return run


def test_shared_replays_recorded_schedule_exactly(tmp_path):
    run = _two_siblings(tmp_path)
    f = summarize(simulate(run, Policy.parse("default"), sandbox="fixed", gpu="fixed"))
    s = summarize(simulate(run, Policy.parse("default"), sandbox="fixed", gpu="shared"))
    assert s["makespan_total"] == pytest.approx(f["makespan_total"])
    assert s["gen_seconds"] == pytest.approx(20.0)


def test_shared_scout_generates_alone_at_full_speed(tmp_path):
    run = _two_siblings(tmp_path)
    res = simulate(run, Policy.parse("scout:inf"), sandbox="fixed", gpu="shared")
    s = summarize(res)
    # scout A runs alone: 5 s. Its call is 2 s in the trace, 1 s of it sandbox start, so B starts at
    # 5 + 1 + 1 = 7 and also generates alone: 5 s
    assert s["gen_seconds"] == pytest.approx(10.0)
    assert s["scout_delay_s"] == pytest.approx(7.0)


def test_shared_reschedules_without_double_ending(tmp_path):
    """B starts while A is generating: A's end is rescheduled; each generation ends exactly once."""
    run = _two_siblings(tmp_path)
    run.steps[0].rollouts[1].start = 2.0  # B arrives 2 s later
    res = simulate(run, Policy.parse("default"), sandbox="fixed", gpu="shared")
    s = summarize(res)
    # A alone 2 s (2 work), then both at 0.5: A needs 3 more work = 6 s -> ends at 8; B did 3 work, 2 left alone = 2 s -> 10
    assert s["gen_seconds"] == pytest.approx(8.0 + 8.0)
    assert len([c for c in res.calls if c.kind == "tool"]) == 2
