# Full SkyRL-SQL runs

Same pipeline as `toy_n4`, now with the real SQL data and more siblings per prompt.

## Setup

- 1 node, 4x A100 40GB, submitted with `train.sbatch`
- Qwen2.5-1.5B-Instruct, GRPO, async rollout
- SkyRL-SQL train set (653 tasks)
- 20 steps, 8 prompts per step, 16 rollouts per prompt (2560 rollouts per run)
- All other settings are the `launch/run.sh` defaults

| Folder | Command |
|---|---|
| `sql_cache-on` | `sbatch -A m4999 --export=ALL,WORKLOAD=sql,SQL_DATA=sql,N=16,STEPS=20 launch/train.sbatch` |
| `sql_cache-off` | same, plus `CACHE=off` |

## Results

| | Cache on | Cache off |
|---|---|---|
| Tool calls | 580 | 601 |
| Cold / timing / hit | 544 / 7 / 29 | 568 / 4 / 29 |
| Hit rate | 5.0% | 4.8% |
| Upper bound | 6.2% | 5.5% |
| Median cold call (s) | 0.088 | 0.002 |
| Tool time (s) | 45.2 | 1.2 |
| Generation time (s) | 3277 | 3289 |
| Mean step time (s) | 19.7 | 21.1 |
| Mean reward | -0.03 | -0.06 |

With cache off, "hit" means "would have hit": the same key had already finished.

## Notes

- Timing misses are about 1% of tool calls. All of them are on the first tool call. This fits H1.
- The hit rate is much lower than on toy SQL (5% vs 26%). Toy SQL repeats its 12 tasks, so most toy hits came from earlier steps.
- Most rollouts do not call a tool. 2560 rollouts made about 600 tool calls.
- A SQL query takes about 2 ms. With the cache on, a cold call takes about 88 ms. The cache adds about 85 ms per call, so it makes SQL tools slower, not faster.
- Even with that overhead, tool time is about 1.4% of generation time. On SQL, better scheduling can save almost nothing. This fits H2: we need slow tools, like real Terminal-Bench tasks, to see a gain.
- The step time difference between the two runs is small and within normal noise.

## Figures

Made with `python scripts/plot_results.py results/sql_full_n16`. They are in `figures/`.

## Files in each folder

- `report.json`: summary numbers
- `tool_calls.csv`: one row per tool call, with its miss class
- `timeline/`: raw events, one file per worker
- `gpu.csv`: GPU use over time
- `tvcache_dump.json`, `tvcache.log`: cache state and server log (cache-on only)
- `train.log`: full training log
