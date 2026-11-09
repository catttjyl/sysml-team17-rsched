"""Out-of-sample check: predict a cache-on run from its cache-off twin.

    python scripts/validate_sim.py [results/tbench_full_b4n8]

The cache-off and cache-on runs of a workload used the same tasks in every
step, so simulating the cache-off trace with the cache predicts the cache-on
run. Nothing from the cache-on trace goes into the prediction except three
constants measured on cache-on traces: the 1.0 s test-hit threshold, the 0.5 s
test fill lead, and the test-hit latencies (a cache-off trace has no hits). Also prints how
close the default replay of each trace is to the trace itself (in-sample).

Step makespan is shown as median and mean. The two runs sampled different
trajectories, and a step's makespan is set by its slowest rollout: a step with
a 600 s command timeout takes 10x longer than the rest, so the mean mostly
reflects how many timeouts the policy happened to sample, not the cache.
"""

from __future__ import annotations

import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rsched.sim import Policy, load_run, simulate, summarize  # noqa: E402


def trace_numbers(run) -> dict:
    segs = [s for st in run.steps for r in st.rollouts for s in r.segs]
    tools = [s for s in segs if s.kind == "tool"]
    tests = [s for s in segs if s.kind == "score" and s.key]
    mk = [st.t_end - st.t0 for st in run.steps]
    return {
        "tool": sum(s.trace_hit for s in tools) / max(1, len(tools)),
        "test": sum(s.trace_hit for s in tests) / max(1, len(tests)),
        "mk": statistics.mean(mk),
        "mk_med": statistics.median(mk),
    }


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "results/tbench_full_b4n8"
    names = sorted(os.listdir(root))
    print("| check | sandbox model | tool hit rate (sim / real) | test hit rate (sim / real) | median step s (sim / real) | mean step s (sim / real) |")
    print("|---|---|---|---|---|---|")
    for name in names:
        if not name.endswith("_cache-on"):
            continue
        twin = name.replace("_cache-on", "_cache-off")
        on = load_run(os.path.join(root, name))
        real = trace_numbers(on)
        rows = [(f"{name} replay (in-sample)", on)]
        if twin in names:
            rows.append((f"{twin} → {name} (out-of-sample)", load_run(os.path.join(root, twin))))
        for label, run in rows:
            for model in ("fixed", "queue"):
                s = summarize(simulate(run, Policy.parse("default"), sandbox=model))
                print(f"| {label} | {model} | {s['tool']['hit_rate']:.1%} / {real['tool']:.1%} "
                      f"| {s['test']['hit_rate']:.1%} / {real['test']:.1%} "
                      f"| {s['makespan_median']:.1f} / {real['mk_med']:.1f} | {s['makespan_mean']:.1f} / {real['mk']:.1f} |")


def curve_check(root: str, names: list[str]) -> None:
    """Fit the speed-vs-load curve on one run and predict another run's generation times."""
    import json

    from rsched.sim.gpu import LoadProfile, fit_curve
    from rsched.timeline import load_events

    def gens_of(name):
        evs = load_events(os.path.join(root, name, "timeline"))
        g = [(e["t_start"], e["t"], e.get("new_tokens") or 0) for e in evs if e["ev"] == "gen_end"]
        return g, LoadProfile([(a, b) for a, b, _ in g])

    data = {n: gens_of(n) for n in names}
    curves = {n: fit_curve(*data[n]) for n in names}
    print("\n| curve fitted on | predicts | median abs. error, with load | without load | same, load > 16: with | without |")
    print("|---|---|---|---|---|---|")
    for src in names:
        for dst in names:
            if src == dst:
                continue
            gens, prof = data[dst]
            c = curves[src]
            rows = []  # (load, error with load, error without)
            for a, b, n in gens:
                if b - a < 0.05 or n < 20:
                    continue
                dur, load = b - a, prof.mean_load(a, b)
                rows.append((load, abs(n / c(load) - dur) / dur, abs(n / c(1) - dur) / dur))
            hi = [x for x in rows if x[0] > 16]
            med = lambda xs, i: f"{statistics.median([x[i] for x in xs]):.1%}" if xs else "-"
            print(f"| {src} | {dst} | {med(rows, 1)} | {med(rows, 2)} | {med(hi, 1)} | {med(hi, 2)} |")


if __name__ == "__main__":
    main()
    _root = sys.argv[1] if len(sys.argv) > 1 else "results/tbench_full_b4n8"
    curve_check(_root, sorted(n for n in os.listdir(_root) if os.path.isdir(os.path.join(_root, n, "timeline"))))
