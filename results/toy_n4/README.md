# Toy runs

First traces from Perlmutter. Goal: check that the pipeline works end to end.

## Setup

- 1 node, 4x A100 40GB
- Qwen2.5-1.5B-Instruct, GRPO, async rollout
- 5 steps, 8 prompts per step, 4 rollouts per prompt (160 rollouts per run)
- Toy data: SQL has 12 train tasks, terminal has 10 train tasks
- All other settings are the `launch/run.sh` defaults

| Folder | Command |
|---|---|
| `sql_cache-on` | `WORKLOAD=sql bash launch/run.sh` |
| `terminal_cache-on` | `WORKLOAD=terminal bash launch/run.sh` |
| `terminal_cache-off` | `WORKLOAD=terminal CACHE=off bash launch/run.sh` |

## Results

| | SQL, cache on | Terminal, cache on | Terminal, cache off |
|---|---|---|---|
| Tool calls | 166 | 151 | 139 |
| Cold / timing / hit | 121 / 2 / 43 | 130 / 6 / 15 | 113 / 7 / 19 |
| Hit rate | 25.9% | 9.9% | 13.7% |
| Upper bound | 27.1% | 13.9% | 18.7% |
| Tool time (s) | 10.5 | 83.3 | 49.3 |
| Generation time (s) | 241.6 | 271.1 | 259.5 |

With cache off, "hit" means "would have hit": the same key had already finished.

## Notes

- SQL calls are very fast (about 0.06 s each), so timing misses are rare.
- Terminal calls are slower (about 0.5 s each) and have more timing misses.
- All timing misses are on the first tool call. This fits H1.
- Cold calls are slower with the cache on (0.63 s) than off (0.40 s). The extra time is likely cache overhead (lookup, sandbox copy, insert). Small sample, so this needs a proper check.
- Toy SQL has only 12 tasks, so prompts repeat across steps. Many SQL hits come from earlier steps, not from siblings.
- Mean reward is low on SQL (-0.89). The model often does not output `<solution>`.
- "DataLoader worker is killed" shows up at the end of each run. It happens after step 5 and does not affect the results.

## Files in each folder

- `report.json`: summary numbers
- `tool_calls.csv`: one row per tool call, with its miss class
- `timeline/`: raw events, one file per worker
- `gpu.csv`: GPU use over time
- `tvcache_dump.json`, `tvcache.log`: cache state and server log (cache-on only)
- `train.log`: full training log
