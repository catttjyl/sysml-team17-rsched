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

- Build the Terminal-Bench 1 images and run the first container runs. See "Running the real Terminal-Bench tasks" below.
- Trace analysis on the full SQL runs.
- Simulator v0 that replays the traces.
- Report the TCP_NODELAY fix upstream, and find out how to run Terminal-Bench sandboxes on Perlmutter.

Known limits

- The TVCache port is fixed at 7878, so only one cache run fits on a node.
- The built-in terminal tasks run in a plain folder, not an isolated container.
- Container forks keep files, the working directory and exported variables, but not running processes.
- Multi-service Terminal-Bench tasks are skipped, and settings like `tmpfs` or `extra_hosts` in a task's compose file are not applied. The converter lists the affected tasks.
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

This sets up the environment and runs the toy data, which needs no large downloads.

```bash
ssh -i ~/.ssh/nersc <user>@perlmutter.nersc.gov
git clone https://github.com/catttjyl/sysml-team17-rsched.git
cd sysml-team17-rsched

bash launch/install.sh        # login node, one time: uv, rust, veRL v0.9.1 env, TVCache build (~20 min)
bash launch/data.sh           # login node: Qwen2.5-1.5B-Instruct, toy SQL DBs, built-in terminal tasks

# interactive bring-up (4×A100)
salloc -N 1 -C gpu -q interactive -t 01:00:00 --gpus-per-node=4 -A <account>
WORKLOAD=sql      bash launch/run.sh
WORKLOAD=terminal bash launch/run.sh
WORKLOAD=terminal CACHE=off bash launch/run.sh   # baseline

# or batch
mkdir -p logs && sbatch -A <account> launch/train.sbatch
sbatch -A <account> --export=ALL,WORKLOAD=terminal launch/train.sbatch
```

Each run writes `$PSCRATCH/rsched/runs/<name>/`:

- `timeline/*.jsonl`: the raw events
- `tool_calls.csv`: every call with its miss class
- `report.json`: summary numbers
- `gpu.csv`: GPU use over time
- `tvcache.log`, `tvcache_dump.json`: the cache server log and the Tool Call Graphs (cache-on only)
- `train.log`: the full training log

You can tune the run with environment variables on `run.sh`: `MODEL`, `STEPS` (5), `BATCH` (8 prompts), `N` (4 rollouts per prompt), `MAX_TURNS` (6), `PROMPT_LEN`/`RESP_LEN` (4096), `TP`, `NGPUS`. `NAME` sets the run folder name. Anything after the script name is passed to veRL as a Hydra override.

`train.sbatch` uses the debug queue, which has a 30 minute limit. For longer runs, add for example `-q regular -t 04:00:00` to `sbatch`.

## Running the real SkyRL-SQL data

SkyRL-SQL is a text-to-SQL dataset with 653 train and 1034 validation tasks. The agent answers questions by querying read-only SQLite databases, which come from OmniSQL. Each call is a fast read with no state to fork.

Fetch the data on a login node. It needs about 50 GB, and the unzip prints nothing and takes a while.

```bash
FULL_SQL=1 bash launch/data.sh
```

Run with `SQL_DATA=sql`, once for each setup you want to compare:

```bash
sbatch -A <account> --export=ALL,WORKLOAD=sql,SQL_DATA=sql,N=16,STEPS=20,NAME=sql-full_n16_cache-on launch/train.sbatch
sbatch -A <account> --export=ALL,WORKLOAD=sql,SQL_DATA=sql,N=16,STEPS=20,CACHE=off,NAME=sql-full_n16_cache-off launch/train.sbatch
```

To plot a batch of runs, copy them into one folder under `results/`, named `<workload>_cache-on` or `<workload>_cache-off`. The figures go to `results/<folder>/figures/`.

```bash
mkdir -p results/sql_full_n16
cp -r $PSCRATCH/rsched/runs/sql-full_n16_cache-on  results/sql_full_n16/sql_cache-on
cp -r $PSCRATCH/rsched/runs/sql-full_n16_cache-off results/sql_full_n16/sql_cache-off
python scripts/plot_results.py results/sql_full_n16
```

## Running the real Terminal-Bench tasks

Terminal-Bench 1 has 241 shell tasks, each with its own Docker image and tests. Each task's `task.yaml` labels it easy (64), medium (120) or hard (57). We use the easy and medium tasks at a fixed commit, with one dataset for each. Tasks with several services are skipped for now, which leaves 59 easy and 113 medium. These counts differ from the TVCache paper's (51 and 95), which gives no task list.

Rewards on these tasks are training signals only. Do not present a trained model's score as a Terminal-Bench result, or publish trained checkpoints as a Terminal-Bench model.

### 1. Set up containers (one time)

On Perlmutter, image names with no registry resolve to `registry.suse.com`, so make Docker Hub the default:

```bash
mkdir -p ~/.config/containers
echo 'unqualified-search-registries = ["docker.io"]' > ~/.config/containers/registries.conf
```

### 2. Prepare the tasks (one time per group)

These commands use the easy group. For the medium group, replace `easy` with `medium`.

```bash
source launch/env.sh               # needed in each new shell, to set $RSCHED_DATA

# login node: fetch and convert the tasks
FULL_TBENCH=1 bash launch/data.sh

# login node, in tmux: build each task's image and save it as a tar. Safe to re-run.
python scripts/tbench_images.py build --tasks $RSCHED_DATA/terminal_easy/tasks.json
```

Errors are saved in `$RSCHED_DATA/terminal_bench1/images/build_status.json`. `accelerate-maximal-square` is expected to fail, because its base image has a file owner that rootless podman can't map.

Next, keep only the tasks that work on Perlmutter. A task passes if its tests fail in an untouched container and pass after its reference solution.

```bash
# compute node. Safe to re-run.
source launch/env.sh
python scripts/tbench_images.py load --tasks $RSCHED_DATA/terminal_easy/tasks.json --allow-missing
python scripts/check_tbench_oracle.py --tasks $RSCHED_DATA/terminal_easy/tasks.json
python -m rsched.data.prepare_terminal_bench --tasks-dir $RSCHED_DATA/terminal_bench1/src/original-tasks \
  --difficulty easy --out $RSCHED_DATA/terminal_easy --only $RSCHED_DATA/terminal_easy/oracle_status.json
python scripts/tbench_images.py clean
```

### 3. Run

`run.sh` loads the images, and removes containers and snapshots at the end. Results and tuning options are the same as in the quick start, and `RSCHED_CMD_TIMEOUT` sets the limit per command (600 s).

```bash
# interactive
WORKLOAD=terminal TERMINAL_DATA=terminal_easy BATCH=4 N=4 bash launch/run.sh

# batch
sbatch -A <account> --export=ALL,WORKLOAD=terminal,TERMINAL_DATA=terminal_easy,BATCH=4,N=4 launch/train.sbatch
```

Add `CACHE=off` for the baseline. Plot the results the same way as for SQL, with folders named `terminal_cache-on` and `terminal_cache-off`.

### Notes

- Use tars, not `podman-hpc migrate`. On Perlmutter, `commit` fails for containers started from migrated images.
- Loaded images and snapshots live in `/tmp`, which is memory-backed on compute nodes and cleared between jobs. Snapshots are not evicted during a run, so long runs can use a lot of memory.

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
