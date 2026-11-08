# Midpoint figures

F1 and F2 are diagrams with no data. F3 to F5 use the full runs in `results/sql_full_b4n8` and `results/tbench_full_b4n8`: 20 GRPO steps, 4 prompts × 8 rollouts, with the cache on and off. Tool time counts command execution only; sandbox (container) start is shown on its own.
Regenerate them with `python scripts/plot_midpoint.py`. Each figure is saved as PNG (300 dpi) and PDF.

| Figure | What it shows | How to read it |
|---|---|---|
| `F1_pipeline` | The system: veRL GRPO, vLLM, N sibling rollouts, TVCache, sandboxes and the timeline analysis. The proposed reuse-aware scheduler is marked in orange. | Every generation turn and tool call is timestamped and classified as a hit, a cold miss or a timing miss. The scheduler would sit between the rollouts and the cache. |
| `F2_timing_miss_concept` | A schematic of today's behavior next to reuse-aware scheduling. | A timing miss is an identical call that misses only because its first copy is still running. Waiting for that copy turns the miss into a hit and saves one sandbox execution. |
| `F3_rollout_timelines` | Real timelines for the 8 siblings of one group per workload (cache on): generation, sandbox start, tool calls and the reward test. | A SQL rollout is almost all generation. A Terminal-Bench rollout is mostly sandbox start and the reward test. |
| `F4_time_budget` | The mean time per rollout over 640 rollouts, split into generation, sandbox start, tool calls, timed-out tool calls and the reward test. | Environment work is about 0% of a SQL rollout and over 90% of a Terminal-Bench rollout. The reward test is about half of it, because `run-tests.sh` installs curl, uv and pytest on every run. |
| `F5_tool_latency` | Every cold tool call on a log scale, cache on (circles) and off (squares), with timed-out calls marked. | A Terminal-Bench call takes about 0.6 s, roughly 300× a SQL call (about 2 ms). |

