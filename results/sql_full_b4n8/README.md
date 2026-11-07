# Full SkyRL-SQL runs, 4 x 8 x 20

Same setup as the Terminal-Bench runs, so the two workloads can be compared directly. This is also the setup used in the TVCache paper.

## Setup

- 1 node, 4x A100 40GB, submitted with `train.sbatch`
- Qwen2.5-1.5B-Instruct, GRPO, async rollout
- SkyRL-SQL train set (653 tasks)
- 20 steps, 4 prompts per step, 8 rollouts per prompt (640 rollouts per run)
- TVCache with the TCP_NODELAY fix
- All other settings are the `launch/run.sh` defaults

| Folder | Command |
|---|---|
| `sql_cache-on` | `sbatch -A m4999 --export=ALL,WORKLOAD=sql,SQL_DATA=sql,BATCH=4,N=8,STEPS=20 launch/train.sbatch` |
| `sql_cache-off` | same, plus `CACHE=off` |

## Results

| | Cache on | Cache off |
|---|---|---|
| Tool calls | 262 | 224 |
| Cold / timing / hit | 250 / 1 / 11 | 213 / 1 / 10 |
| Hit rate | 4.2% | 4.5% |
| Upper bound | 4.6% | 4.9% |
| Median cold call (ms) | 2.0 | 1.5 |
| Tool time (s) | 0.6 | 0.4 |
| Generation time (s) | 780 | 760 |
| Mean step time (s) | 10.5 | 9.9 |
| Mean reward | -0.21 | -0.15 |

With cache off, "hit" means "would have hit": the same key had already finished.

## Notes

- The results match the larger `sql_full_n16` runs after the fix.
- There is 1 timing miss in each run, both on the first tool call. This fits H1.
- Tool time is under 0.1% of generation time, so scheduling can save almost nothing on SQL. This fits H2.
- Most rollouts do not call a tool. 640 rollouts made about 250 tool calls.
- Step time differences between the two runs are small and within normal noise.

## Figures

Made with `python scripts/plot_results.py results/sql_full_b4n8`. They are in `figures/`.

## Files in each folder

- `report.json`: summary numbers
- `tool_calls.csv`: one row per tool call, with its miss class
- `timeline/`: raw events, one file per worker
- `gpu.csv`: GPU use over time
- `tvcache_dump.json`, `tvcache.log`: cache state and server log (cache-on only)
- `train.log`: full training log
