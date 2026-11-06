# Examples A–D on Perlmutter (2026-11-05)

One GRPO step, 1 prompt × 4 rollouts, Qwen2.5-1.5B-Instruct, 1 node 4×A100, cache on and off.
Run with `scripts/examples/run_ab.sh`, `run_c.sh`, `run_d.sh` (data prep: `prep_data.sh`).
Example D used the `terminal.py` fix in this branch.

| Folder | Workload | Data |
|---|---|---|
| `exA_sql-toy_cache-{on,off}` | SQL | toy SQLite DBs |
| `exB_term-toy_cache-{on,off}` | terminal, folder sandbox | built-in toy tasks |
| `exC_sql-full_cache-{on,off}` | SQL | SkyRL-SQL-653 |
| `exD_tb-one_cache-{on,off}` | terminal, podman-hpc containers | Terminal-Bench 1 `hello-world` |

`SUMMARY_TABLE.md` has the numbers. Figures: `python scripts/plot_midpoint.py`.
