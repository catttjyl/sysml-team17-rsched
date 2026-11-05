# reuse-sched: veRL + TVCache pipeline (Team 17)

This is the minimum runnable pipeline for _Reuse-aware rollout scheduling for cached agentic RL_.
It runs veRL GRPO with tool calls served through **TVCache** and logs a per-call timeline.
Two workloads are included:

| workload                              | tool               | state                      | sandbox                                      | cache key                |
| ------------------------------------- | ------------------ | -------------------------- | -------------------------------------------- | ------------------------ |
| `sql` (SkyRL-SQL-653, plus a toy set) | `execute_sql(sql)` | read-only DB, so stateless | handle to the `.sqlite` file                 | the current query only   |
| `terminal` (Terminal-Bench style)     | `bash(command)`    | mutating                   | one directory per sandbox, forked by copying | the full command history |

Pinned versions: **veRL v0.9.1** (its own `uv.lock`: torch 2.11+cu130, vLLM 0.24), tvcache-rust `main`, tvclient `main`.

## Progress

Done
- Tested locally: TVCache build, tvclient, both workloads, rewards, logging and analysis, and the veRL settings in `run.sh`.
- Runs end to end on Perlmutter (1 node, 4x A100) for both workloads, with the cache on and off.
- `install.sh` works on Perlmutter. It builds TVCache with gcc and patches a broken flash-attn URL in veRL's lock file.
- `prepare_sql.py` loads the full SkyRL-SQL data (653 train, 1034 validation tasks).
- Found and fixed a TVCache bug that added about 85 ms to every call (TCP_NODELAY not set). `install.sh` applies the fix. Details in `results/sql_full_n16/README.md`.
- Toy runs: `results/toy_n4/`.
- Full SkyRL-SQL runs, 16 rollouts per prompt, 20 steps, cache on (before and after the fix) and off: `results/sql_full_n16/`.

First findings
- Timing misses are rare: 1 to 5% of tool calls, and under 1% on full SQL after the fix.
- All timing misses are on the first tool call. This fits H1.
- Faster tool calls mean fewer timing misses. After the fix, timing misses on SQL dropped from 7 to 1.
- SQL tool calls are very fast (about 2 ms) and take under 0.1% of generation time. Scheduling can save almost nothing on SQL. This fits H2.
- After the fix, the cache adds about 1 ms per call.
- Toy terminal calls are also fast (median about 0.1 s), so they cannot show a scheduling gain either.

Next
- A podman-hpc sandbox for real Terminal-Bench tasks, where tool calls are slow. Converted tasks can already be loaded with `RSCHED_TERMINAL_TASKS=tasks.json` (same fields as `terminal_tasks.py`).
- Trace analysis on the full SQL runs.
- Simulator v0 that replays the traces.
- Report the TCP_NODELAY fix upstream, and find out how to run Terminal-Bench sandboxes on Perlmutter.

Known limits
- The TVCache port is fixed at 7878, so only one cache run fits on a node.
- The terminal sandbox is a plain folder, not an isolated container.
- SQL uses native function calling, while SkyRL uses `<sql>` tags. The reward follows SkyRL: -1 with no `<solution>`, 0 for a wrong result, 1 for a correct one. The `<think>` format check is not applied.
- SQL queries are whitespace-normalized before they become cache keys. This does not change their meaning.
- The global step is not logged. Use `uid` (one per prompt per step) with wall-clock time.
- "DataLoader worker is killed" appears at the end of each run. It happens after the last step and does not affect the results.

## How it fits together

```
veRL main_ppo (GRPO) ── AgentLoopWorker (Ray) ── TVCacheToolAgentLoop   (one per rollout)
                                                  │ subclass of veRL ToolAgentLoop
                                                  ├─ generation turns → vLLM            (logged: gen_start/gen_end)
                                                  ├─ tool call → tvclient.TvcacheToolExecutor.lookup_or_execute(history)
                                                  │              ├─ hit  → cached value
                                                  │              └─ miss → fork deepest cached sandbox, run suffix, insert
                                                  │                                     (logged: tool  t_start/t_end/hit/key)
                                                  └─ end → workload.score() → output.reward_score
                                                          (terminal: test also goes through the cache)
TVCache server (Rust, :7878, same node)    nvidia-smi sampler → gpu.csv
```

- `rsched/agent_loop.py`: the agent loop. Registered through `configs/agent_loops.yaml`. Dataset rows pick it with `agent_name`.
- `rsched/workloads/{sql,terminal}.py`: the `ToolCall` and `Sandbox` implementations for tvclient, plus the reward for each workload.
- `rsched/timeline.py`: JSONL events, one file per worker process.
- `rsched/analysis/timeline_report.py`: computes hit rate and splits misses into **cold / timing / late**, then gives the timing-miss upper bound and a breakdown by call position (for H1).
- `RSCHED_CACHE=off` runs the same thing without the cache. Keys are still logged, so you can compute the hit rate the cache would have had.

## Perlmutter quick start

```bash
ssh -i ~/.ssh/nersc <user>@perlmutter.nersc.gov
git clone https://github.com/catttjyl/sysml-team17-rsched.git
cd sysml-team17-rsched

bash launch/install.sh        # login node, one time: uv, rust, veRL v0.9.1 env, TVCache build (~20 min)
bash launch/data.sh           # login node: Qwen2.5-1.5B-Instruct, toy SQL DBs, terminal tasks

# interactive bring-up (debug QOS, 4×A100)
salloc -N 1 -C gpu -q interactive -t 01:00:00 --gpus-per-node=4 -A <account>
WORKLOAD=sql      bash launch/run.sh
WORKLOAD=terminal bash launch/run.sh
WORKLOAD=terminal CACHE=off bash launch/run.sh   # baseline

# or batch
mkdir -p logs && sbatch -A <account> launch/train.sbatch
sbatch -A <account> --export=ALL,WORKLOAD=terminal launch/train.sbatch
```

Each run writes `$PSCRATCH/rsched/runs/<name>/`:
`timeline/*.jsonl`, `tool_calls.csv` (every call with its miss class), `report.json`, `gpu.csv`, `tvcache.log`, `tvcache_dump.json` (the Tool Call Graphs), and `train.log`.

You can tune the run with environment variables on `run.sh`: `MODEL`, `STEPS` (5), `BATCH` (8 prompts), `N` (4 rollouts per prompt), `MAX_TURNS` (6), `PROMPT_LEN`/`RESP_LEN` (4096), `TP`, `NGPUS`. Anything after the script name is passed to veRL as a Hydra override.

Real SkyRL-SQL data needs about 50 GB. Run `FULL_SQL=1 bash launch/data.sh` on a login node. The unzip prints nothing and takes a while. Then run with `SQL_DATA=sql`, for example:

```bash
sbatch -A <account> --export=ALL,WORKLOAD=sql,SQL_DATA=sql,N=16,STEPS=20,NAME=sql-full_n16_cache-on launch/train.sbatch
```

`NAME` sets the run folder name. `train.sbatch` uses the debug queue, which has a 30 minute limit.

To plot a batch of runs, put them in one folder under `results/` and run `python scripts/plot_results.py results/<folder>`.

For a shared team install, set `RSCHED_ROOT` (and `RSCHED_DATA`) to a project CFS or common-software directory before running the scripts. Read `launch/env.sh` for the details. `$PSCRATCH` is purged after 8 weeks without access.

## Miss classes (draft definition from the proposal)

Take a call that missed at time _t_ with key _K_:

- **cold**: no earlier call with key _K_ exists.
- **timing**: an earlier call with key _K_ was issued before _t_ but had **not finished** by _t_. Its filler was still in flight, so this is the miss that scheduling can recover.
- **late**: an earlier call with key _K_ had already finished, yet this call still missed. Causes include an uncacheable result, eviction or a lost insert race.

The **upper bound** is (hits + timing) / calls.

## Tests (CPU, no GPU needed)

```bash
TVCACHE_BIN=$TVCACHE_BIN pytest -q tests/
```

This starts a real TVCache server and drives the real `TVCacheToolAgentLoop` tool path with scripted LLM turns. It checks rewards, cache hits on a second wave, the timeline, and the cache-off baseline. The script also shows the effect the project studies: 4 concurrent siblings that share their first call all record **timing** misses.
