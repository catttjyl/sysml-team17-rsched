# Poster figures

fig1, fig2, fig3 and fig5 use the full runs in `results/sql_full_b4n8` and `results/tbench_full_b4n8` (20 GRPO steps, 4 prompts × 8 rollouts). fig4 uses `results/sql_full_n16`, the only runs from before the TVCache fix. Tool time counts command execution only; sandbox (container) start is left out.
Regenerate them with `python scripts/plot_poster.py`. Each figure is saved as PNG (300 dpi) and PDF.
The colors mean the same thing in every figure: blue is a hit, orange is a timing miss, and gray is a cold miss.

| Figure | What it shows | How to read it |
|---|---|---|
| `fig1_miss_classes` | The share of hit, timing-miss and cold-miss calls in each of the 6 runs. | Timing misses are under 0.5% on SQL, but 4 to 8% on Terminal-Bench, as common as hits on easy and almost all reusable calls on medium. |
| `fig2_h1_position` | All tool calls and all timing misses, grouped by the call's position in its rollout, pooled over the 6 runs. | **H1:** all 159 timing misses are on the 1st call, where siblings issue the same call at about the same time. |
| `fig3_h2_time_budget` | Left: tool time as a % of generation time, with timed-out calls shown separately. Right: time per cold call, on a log scale. | **H2:** SQL tool calls take about 2 ms, under 0.1% of generation time. Terminal-Bench calls take about 0.6 s, and tool time is 34 to 177% of generation time even without timeouts. |
| `fig4_tcp_nodelay_fix` | Median cold-call time and the number of timing misses on SkyRL-SQL: before the fix, after it, and with the cache off. | Adding TCP_NODELAY to TVCache cut each call from 88 ms to 2 ms. The slow calls had also inflated timing misses (7 before the fix, 1 after). |
| `fig5_siblings_sql` / `fig5_siblings_terminal` | One real GRPO group over time: generation, sandbox start, the first copy of a shared call, timing misses, other cold misses, and hits. | On Terminal-Bench, 7 of 8 siblings issue the same first call while its first copy is still running, so they all miss. This is what a reuse-aware scheduler would turn into hits. |
