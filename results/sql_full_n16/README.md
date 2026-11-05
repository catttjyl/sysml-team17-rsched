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
| `sql_cache-on-nodelay` | same, after the TVCache fix below |
| `sql_cache-off` | same, plus `CACHE=off` |

## Results

| | Cache on | Cache on, fixed | Cache off |
|---|---|---|---|
| Tool calls | 580 | 587 | 601 |
| Cold / timing / hit | 544 / 7 / 29 | 555 / 1 / 31 | 568 / 4 / 29 |
| Hit rate | 5.0% | 5.3% | 4.8% |
| Upper bound | 6.2% | 5.5% | 5.5% |
| Median cold call (ms) | 88 | 2.1 | 1.5 |
| Tool time (s) | 45.2 | 1.7 | 1.2 |
| Generation time (s) | 3277 | 3226 | 3289 |
| Mean step time (s) | 19.7 | 19.0 | 21.1 |
| Mean reward | -0.03 | -0.04 | -0.06 |

With cache off, "hit" means "would have hit": the same key had already finished.

## TVCache fix

### What was wrong

Each tool call talks to the TVCache server once or twice: one request to look up the call, and one to store the result. The server sends each reply in several small pieces, for example 1 byte for the status, then 4 bytes for the length, then the value.

By default, a TCP connection uses Nagle's algorithm: it holds back a small piece until the other side confirms the previous one. The other side uses delayed ACK: it waits about 40 ms before it confirms. Together, each reply waits about 40 ms for nothing. That is why cold calls took about 45 ms (one request) or 90 ms (two requests), while the SQL query itself takes about 2 ms.

### The fix

One line in `tvcache-rust/src/main.rs`, on each new connection:

```rust
let _ = tcp_stream.set_nodelay(true);
```

This turns off Nagle's algorithm, so each piece is sent right away. `launch/install.sh` adds this line after it clones TVCache. If you installed before this change, run `bash launch/install.sh` again. It only rebuilds TVCache.

### Why it matters

- Without the fix, the cache made SQL tool calls about 40 times slower (88 ms vs 2 ms), so it could never save time.
- It also skews our measurements. A timing miss happens while a matching call is still running, so calls that take longer cause more timing misses. With the extra 85 ms, SQL had 7 timing misses. After the fix it had 1. Runs from before the fix overstate timing misses.

After the fix, the cache adds about 1 ms per call.

## Notes

- Timing misses are rare: 7, then 1 after the fix. All of them are on the first tool call. This fits H1.
- Faster tool calls mean fewer timing misses, because a call is in flight for less time. This is the idea behind H2.
- A SQL query takes about 2 ms, and all tool time is under 0.1% of generation time. On SQL, better scheduling can save almost nothing. We need slow tools, like real Terminal-Bench tasks, to see a gain.
- The hit rate is much lower than on toy SQL (5% vs 26%). Toy SQL repeats its 12 tasks, so most toy hits came from earlier steps.
- Most rollouts do not call a tool. 2560 rollouts made about 600 tool calls.
- Step time differences between runs are small and within normal noise.

## Figures

Made with `python scripts/plot_results.py results/sql_full_n16`. They are in `figures/`.

## Files in each folder

- `report.json`: summary numbers
- `tool_calls.csv`: one row per tool call, with its miss class
- `timeline/`: raw events, one file per worker
- `gpu.csv`: GPU use over time
- `tvcache_dump.json`, `tvcache.log`: cache state and server log (cache-on only)
- `train.log`: full training log
