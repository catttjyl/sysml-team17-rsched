"""The accelerator model (gpu="shared") on its own and inside the engine.

    pytest -q tests/test_gpu.py
"""

from __future__ import annotations

import pytest

from rsched.sim import Policy, load_run, simulate, summarize
from rsched.sim.gpu import LoadProfile, SpeedCurve, assign_replicas, fit_curve
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
    c = fit_curve([(b - a, n, prof.mean_load(a, b)) for a, b, n in gens])
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
    run.replicas = 1  # both on one vLLM replica
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


# --- routing (veRL v0.9.1 GlobalRequestLoadBalancer: sticky per rollout, least in flight)


def test_assign_replicas_least_loaded_and_sticky():
    gens = {"A": [(0, 10), (20, 30)], "B": [(1, 5)], "C": [(6, 8)], "D": [(12, 14)]}
    out = assign_replicas(gens, 2)
    assert out["A"] == 0 and out["B"] == 1  # B arrives while A runs on 0
    assert out["C"] == 1  # at t=6: A on 0, B done: tie -> lowest index... A still running, so replica 1
    assert out["D"] == 0  # at t=12 nothing runs: tie -> 0
    with pytest.raises(ValueError):
        assign_replicas(gens, 0)


def test_engine_spreads_rollouts_over_replicas(tmp_path):
    """Two replicas: A and B generate at the same time on different GPUs, so neither slows down."""
    run = _two_siblings(tmp_path)
    run.replicas = 2
    for st in run.steps:
        for ro in st.rollouts:
            for seg in ro.segs:
                if seg.kind == "gen":
                    seg.work = 10.0  # 10 s alone
    s = summarize(simulate(run, Policy.parse("default"), sandbox="fixed", gpu="shared"))
    assert s["gen_seconds"] == pytest.approx(20.0)
    run.replicas = 1
    s = summarize(simulate(run, Policy.parse("default"), sandbox="fixed", gpu="shared"))
    assert s["gen_seconds"] == pytest.approx(40.0)  # sharing one GPU at half speed


def test_real_trace_replays_exactly_with_shared_gpu():
    import glob
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "tbench_full_b4n8",
                        "tb-easy_cache-on")
    if not glob.glob(os.path.join(path, "timeline", "*.jsonl")):
        pytest.skip("trace not present")
    run = load_run(path)
    assert run.replicas == 4
    f = summarize(simulate(run, Policy.parse("default"), sandbox="fixed", gpu="fixed"))
    g = summarize(simulate(run, Policy.parse("default"), sandbox="fixed", gpu="shared"))
    assert g["makespan_total"] == pytest.approx(f["makespan_total"], rel=1e-6)
    # replayed trace hits take the median hit latency, which shifts a few later generations by
    # milliseconds and so their load pieces: generation seconds agree to 0.1 s of 1146 s
    assert g["gen_seconds"] == pytest.approx(f["gen_seconds"], rel=1e-3)
    routed = simulate(run, Policy.parse("default"), sandbox="fixed", gpu="shared").replica_of
    assert all(routed[ro.id] == ro.replica for st in run.steps for ro in st.rollouts)
