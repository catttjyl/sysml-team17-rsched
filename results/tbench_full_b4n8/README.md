# Full Terminal-Bench runs, 4 x 8 x 20

Terminal-Bench 1 tasks, each run in its own podman container. The setup matches the `sql_full_b4n8` SkyRL-SQL runs, so the two workloads can be compared directly. This is also the setup used in the TVCache paper.

## Setup

- 1 node, 4x A100 40GB, submitted with `train.sbatch`
- Qwen2.5-1.5B-Instruct, GRPO, async rollout
- Terminal-Bench easy (49 tasks) and medium (88 tasks after filtering, 89 before)
- 20 steps, 4 prompts per step, 8 rollouts per prompt (640 rollouts per run)
- Podman sandbox backend, 600 s command timeout (`RSCHED_CMD_TIMEOUT`)
- TVCache with the TCP_NODELAY fix, and the worker memory fix (c2223ab)
- All other settings are the `launch/run.sh` defaults

| Folder                | Command                                                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `tb-easy_cache-on`    | `sbatch -A m4999 --export=ALL,WORKLOAD=terminal,TERMINAL_DATA=terminal_easy,BATCH=4,N=8,STEPS=20 launch/train.sbatch` |
| `tb-easy_cache-off`   | same, plus `CACHE=off`                                                                                                |
| `tb-medium_cache-on`  | same as easy, with `TERMINAL_DATA=terminal_medium`                                                                    |
| `tb-medium_cache-off` | same, plus `CACHE=off`                                                                                                |

## Results

|                      | Easy, cache on | Easy, cache off | Medium, cache on | Medium, cache off |
| -------------------- | -------------- | --------------- | ---------------- | ----------------- |
| Tasks seen           | 48             | 48              | 80               | 80                |
| Tool calls           | 567            | 546             | 610              | 608               |
| Cold / timing / hit  | 502 / 37 / 28  | 501 / 22 / 23   | 559 / 51 / 0     | 560 / 47 / 1      |
| Hit rate             | 4.9%           | 4.2%            | 0.0%             | 0.2%              |
| Upper bound          | 11.5%          | 8.2%            | 8.4%             | 7.9%              |
| Median cold call (s) | 0.58           | 0.55            | 0.57             | 0.56              |
| Tool time (s)        | 3,097          | 2,211           | 5,930            | 3,720             |
| Sandbox start (s)    | 7,909          | 7,863           | 11,370           | 9,568             |
| Timed-out calls      | 4              | 3               | 6                | 5                 |
| Generation time (s)  | 1,146          | 1,205           | 1,312            | 1,237             |
| Mean step time (s)   | 203            | 134             | 202              | 151               |
| Mean reward          | 0.014          | 0.011           | 0.005            | 0.000             |

With cache off, "hit" means "would have hit": the same key had already finished.

Tool time and median cold call count command execution only: `t_end - t_start - sandbox_start_s`, the same as `command_seconds` in `scripts/plot.py`. Sandbox start is shown in its own row. The `tool_seconds_*` fields in `report.json` still include sandbox start.

## Notes

- **Tool time leaves out sandbox (container) start.** Sandbox start is 66-78% of the raw tool time and is not something scheduling can change. Any scheduler evaluation on these runs must subtract it the same way, or the comparison with this baseline is not fair.
- All hits and all timing misses are on the first tool call of a rollout. Later calls depend on earlier outputs and never repeat. This fits H1.
- Timing misses are as common as hits on easy (22-37 vs 23-28) and are almost all of the reusable calls on medium. A reuse-aware scheduler could at best about double the hit rate on easy and take medium from about 0% to about 8%.
- Timed-out calls (600 s each) make up most of the tool time: 1,802-3,614 s per run. Without them, tool time is 409-2,317 s. Results that depend on tool time should be checked with and without timeouts.
- On medium, timing misses are a large share of command time: 26% with cache on and 18% with cache off, mostly from 1-2 timed-out calls. On easy they are about 1%.
- Unlike SQL, tool time is larger than generation time here, even without sandbox start. This fits H2: scheduling has more room to save time on Terminal-Bench.
- Cache-on runs have a longer mean step time than cache-off runs (about 200 s vs 134-151 s). This gap is larger than normal noise and needs a closer look before drawing conclusions.
- Rewards are near zero on both splits, which is expected for a 1.5B model on Terminal-Bench.

## Files in each folder

- `report.json`: summary numbers (tool seconds here include sandbox start)
- `tool_calls.csv`: one row per tool call, with its miss class
- `timeline/`: raw events, one file per worker (`tool` events carry `sandbox_start_s`)
- `gpu.csv`: GPU use over time
- `tvcache_dump.json`, `tvcache.log`: cache state and server log (cache-on only)
- `train.log`: full training log
