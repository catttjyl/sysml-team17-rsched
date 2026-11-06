# Figures for examples A–D

These figures use the Perlmutter runs in `results/examples_perlmutter`: one GRPO step, 1 prompt × 4 rollouts, with the cache on and off.
Regenerate them with `python scripts/plot_midpoint.py`. Each figure is saved as PNG (300 dpi) and PDF.

| Figure | What it shows | How to read it |
|---|---|---|
| `F1_pipeline` | The system: veRL GRPO, vLLM, N sibling rollouts, TVCache, sandboxes and the timeline analysis. The proposed reuse-aware scheduler is marked in orange. | Every generation turn and tool call is timestamped and classified as a hit, a cold miss or a timing miss. The scheduler would sit between the rollouts and the cache. |
| `F2_timing_miss_concept` | A schematic of today's behavior next to reuse-aware scheduling. | A timing miss is an identical call that misses only because its first copy is still running. Waiting for that copy turns the miss into a hit and saves one sandbox execution. |
| `F3_rollout_timelines` | Real timelines for the 4 siblings of one group in each of A–D (cache on): generation, tool calls and the reward test. | A SQL rollout is almost all generation. A Terminal-Bench rollout is dominated by sandbox work, and mostly by the reward test. |
| `F4_time_budget` | The mean time per rollout, split into generation, tool calls and reward test. | Environment work is about 0% of a SQL rollout and about 73% of a Terminal-Bench rollout (9% tool calls and 64% reward test). The test is slow because `run-tests.sh` runs `apt-get` and installs curl, uv and pytest on every run. |
| `F5_tool_latency` | Every tool call in A–D on a log scale, with cache on (circles) and off (squares). | A Terminal-Bench call takes about 1.8 s, roughly 1,000× a SQL call (about 2 ms). |

These are smoke tests with only 4 rollouts per run, and every call was a cold miss. They show that each workload runs end to end and how large each cost is. They do not measure hit rates or speedups.
