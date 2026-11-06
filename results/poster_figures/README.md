# Poster figures

These figures use the existing runs in `results/toy_n4` and `results/sql_full_n16`.
Regenerate them with `python scripts/plot_poster.py`. Each figure is saved as PNG (300 dpi) and PDF.
The colors mean the same thing in every figure: blue is a hit, orange is a timing miss, and gray is a cold miss.

| Figure | What it shows | How to read it |
|---|---|---|
| `fig1_miss_classes` | The share of hit, timing-miss and cold-miss calls in each of the 6 runs. | Most calls are cold misses, and timing misses are a thin slice: 0.2% on SkyRL-SQL after the TVCache fix. The toy hit rates come mostly from earlier steps, not from siblings. |
| `fig2_h1_position` | All tool calls and all timing misses, grouped by the call's position in its rollout, pooled over the 6 runs. | **H1:** all 27 timing misses are on the 1st call, where siblings issue the same query at the same moment. |
| `fig3_h2_time_budget` | Left: tool time as a % of generation time. Right: time per cold call, on a log scale. | **H2:** SQL tool calls take about 2 ms, which is under 0.1% of generation time, so scheduling has almost nothing to save there. Slower tools are where it can help. |
| `fig4_tcp_nodelay_fix` | Median cold-call time and the number of timing misses on SkyRL-SQL: before the fix, after it, and with the cache off. | Adding TCP_NODELAY to TVCache cut each call from 88 ms to 2 ms. The slow calls had also inflated timing misses (7 before the fix, 1 after). |
| `fig5_siblings_sql` / `fig5_siblings_terminal` | One real GRPO group over time: generation, the first copy of a shared call, timing misses, other cold misses, and hits. | Several siblings issue the same first call while its first copy is still running, so they all miss. This is what a reuse-aware scheduler would turn into hits. Use either the SQL or the terminal version. |
