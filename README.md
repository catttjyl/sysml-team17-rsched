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
- Full SkyRL-SQL runs with 4 prompts, 8 rollouts, 20 steps (same setup with TVCache paper), cache on and off: `results/sql_full_b4n8/`.
- Full Terminal-Bench runs, easy and medium, same setup above, cache on and off: `results/tbench_full_b4n8/`.
- Plotting script `scripts/plot.py`, with figures for the toy and full runs in `results/figures/`. Tool time counts command execution only and leaves out sandbox start.

First findings

- On SQL task, timing misses are rare: 1 to 5% of tool calls, and under 1% on full SQL after the fix.
- Faster tool calls mean fewer timing misses. After the fix, timing misses on SQL dropped from 7 to 1.
- On Terminal-Bench, tool time is larger than generation time, even without sandbox start. Scheduling has room to save time here. This fits H2.
- On Terminal-Bench, timing misses are as common as hits on easy and are almost all reusable calls on medium. Hit rate is 4.2-4.9% on easy (upper bound 8-12%) and 0-0.2% on medium (upper bound about 8%). A reuse-aware scheduler could at best about double the hit rate on easy and take medium from about 0% to about 8%.
- Sandbox start is 66-78% of raw tool time on Terminal-Bench. Scheduling cannot change it, so it is left out of tool time.
- Timed-out calls (600 s) make up most of Terminal-Bench tool time. Results should be checked with and without them.
- All hits and all timing misses are on the first tool call. This fits H1.
- SQL tool calls are very fast (about 2 ms) and take under 0.1% of generation time. Scheduling can save almost nothing on SQL. This fits H2.

Next

- Simulator: done for Terminal-Bench (see [Simulator](#simulator)). Next, an accelerator model, so batch-by-prompt and priority policies can be compared too.
- Reuse-aware scheduler that turns timing misses into hits: hold or delay sibling rollouts until the first call with the same key finishes. Test it in the simulator first, then in veRL.
- Measure the scheduler by hit rate (against the upper bound) and by step time. Tool time must leave out sandbox start, the same as the baseline.
- Look into why cache-on Terminal-Bench runs have longer step times than cache-off (about 200 s vs 134-151 s).
- Report the TCP_NODELAY fix upstream.

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

For a full run, use `BATCH=4 N=8 STEPS=20` (640 rollouts). The batch size and rollouts per prompt match the TVCache paper's setting.

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

- Loaded images and snapshots live in `/tmp`, which is memory-backed on compute nodes and cleared between jobs. Snapshots are not evicted during a run, so long runs can use a lot of memory.

## Simulator

`rsched/sim/` replays recorded timelines under other scheduling policies, with no GPU needed. Each rollout replays its own generation turns, overheads and tool calls. A cached call hits once the same key has finished somewhere in the run.

```bash
python scripts/simulate.py results/tbench_full_b4n8/tb-easy_cache-on \
  --policies default hold:10 hold:inf scout:30 scout:30+hold:inf [--sandbox queue|fixed] [--json out.json]
python scripts/validate_sim.py   # in-sample replay and cache-off -> cache-on prediction
```

- `default`: never wait (veRL today).
- `hold:<tau>`: bounded delay. A call whose key is being filled by another rollout waits up to `tau` s. If the fill is not done by then, the call executes anyway, and the wait is wasted.
- `hold:cost`: wait only if the fill will be visible before this rollout could finish the call itself (sandbox start + replay + command). It is an oracle: it knows the call's true command time.
- `scout:<tau>`: one scout per prompt starts first. Its siblings start when the scout's first cacheable call (tool or test) finishes, or `tau` s after they were ready, whichever comes first.

Model notes:

- **Sandbox start is not tool time, and most of it is queueing.** In the trace it is inside `t_end - t_start` of a rollout's first executed call. `podman run` runs under one node-wide `flock` (`terminal_podman.py`). An uncontended start takes about 1.2 s, so the recorded 16–20 s median is mostly waiting for other rollouts' starts. By default (`--sandbox queue`) the simulator models that lock as one FIFO queue shared by the whole run. Each start holds it for a time calibrated so that the default replay's total sandbox time equals the trace's: 1.37 s (easy) and 1.53 s (medium), above 1.2 s because some starts are slow. About 28% of that target is reconstructed rather than logged (see below). `--sandbox fixed` instead charges each rollout its recorded start time wherever it starts. Runs without podman starts (SQL) have no lock. A rollout whose calls and test all hit never starts a sandbox. A call that raises closes the sandbox, so the next execution starts a new one.
- **The Terminal-Bench test is a cached call too.** `executor.test(history)` is keyed by the full history. Siblings with the same history share it, including all rollouts that made no tool call; those make up most of the test hits that holding adds. Cold tests take about 12 s (median); p90 is 20 s on easy and 41 s on medium.
- **Below an uncacheable call (a timeout), nothing is cached**, the test included. This is an assumption backed by one trace case. A call that raised is dropped from the history instead and doesn't break the chain.
- A test hit costs about 0.5 s in a rollout with a live sandbox, which is stopped too, and about 0 s otherwise. Both values are measured from the trace.
- A test result becomes visible to siblings 0.5 s before the call ends (the sandbox is stopped after the insert).
- **Generation and the GPU** (`--gpu fixed|shared`, `rsched/sim/gpu.py`). With `fixed` (the default), generations take their recorded time. With `shared`, they slow down under load.
  - **The evidence:** within the same turn, Terminal-Bench generations run at about 225 tok/s with ≤ 8 running, about 205–215 with 9–16, and about 150 with 17–32 (the burst at the start of a step).
  - **Routing as in veRL v0.9.1** (`verl/workers/rollout/router.py`, `GlobalRequestLoadBalancer`). There are 4 vLLM replicas, one per GPU (TP=1), read from `train.log`. Each rollout has one `request_id` for its whole conversation and sticks to the replica with the fewest generations in flight when its first generation arrives. veRL breaks ties at random; the simulator uses the lowest index. The traces don't record the replica, so the same rule is replayed on the recorded intervals. The engine's routing of the recorded schedule matches that reconstruction for all 640 rollouts.
  - **The curve:** a speed-vs-load curve is fitted per run against per-replica load, forced non-increasing and checked positive. It is flat up to 3 generations per replica, then falls to about 135 tok/s at 8. Each generation becomes an amount of work that progresses at the speed for its replica's simulated load. Under the recorded schedule this reproduces every generation.
  - **Events can be cancelled** (as in a classic discrete-event queue). When a replica's load changes, each running generation's end event is cancelled and rescheduled. A generation-end event for a rollout that isn't generating raises an error.
  - **Cross-run check** (`validate_sim.py`): a curve fitted on one run predicts another run's generation times under load (more than 4 per replica) with 9–15% median error, vs. 21–31% if load is ignored.
  - **Limits of the curve:** it is fitted on our own runs only (median speed per load level), for one model (Qwen2.5-1.5B), one GPU type (A100), one veRL configuration and one benchmark. It is untested beyond 8 concurrent generations per replica (capped there), and it does not model prefill vs. decode, KV-cache pressure or preemption. It predicts generation times; it does not validate the simulator's scout predictions, which need a real run.
  - **What it changes:** scout lets the scout generate alone, and cuts total generation time by about 11% (28–31% if the GPU saturated at 8 or 4 running). But generation is a small part of a Terminal-Bench step: the GPU averages about 5% utilization (`gpu.csv`), and tools and tests dominate. So total time barely moves (−0.3% to −0.6% with the queue sandbox model). Scout's time cost is set by the sandbox model, not the GPU: about 0% with queue, +11–15% with fixed. Hold gets the same cache benefit without delaying anyone's generation. On SQL, where generation is the whole step, the GPU model halves scout's cost (+22.6% → +13%), but there is almost nothing to reuse.
  - **Not modelled:** prefill vs. decode, KV-cache memory pressure and preemption, and which replica serves a request. "gen-busy" is still only the share of time with at least one generation running.
- A hit returns the recorded output, so later turns stay the same.
- **Bounds.** `reuse bound` = every repeat of a cacheable key hits (calls − distinct keys). It doesn't depend on the policy. In these traces every repeat is either a hit or a timing miss, so `hold:inf` reaches the bound **by construction**. The result worth reading is what the waiting costs. `upper_bound` in the JSON is hits + timing misses of one replay.
- **Snapshots are adaptive.** The runs use `adaptive_snapshots=True`, and `tvcache_dump.json` lists the calls TVCache snapshotted: 26 keys on easy and 50 on medium, all expensive (median command about 10 s vs 0.6 s). Without a dump (cache-off traces), a call of 2.5 s or more is assumed snapshotted.
  - After a hit on a snapshotted key, the next execution forks: one more sandbox start through the lock.
  - After a hit on any other key, the next execution first re-runs the hit calls (`replay s`). Only two trace calls show this (`executed=2`).
  - A snapshotted call's command time still includes taking the snapshot.
- **Holding can wait on a call that will time out.** The scheduler can't know in advance, so a held rollout waits for any execution of the key in flight. If that execution times out, nothing is inserted, and the rollout runs the call itself after the wait.
- **Sandbox start of rollouts with no executed call** isn't logged. It is reconstructed by replaying the lock in request order. `flock` isn't strictly FIFO, so it is capped to leave the test at least the 10th-percentile cold test time.
- **What the timing misses are.** Every repeat inside a step of a key not already cached is a timing miss in these runs: 37 on easy, 49 on medium. Almost all are siblings issuing the same first call within about 3 s of the filler. Cross-step repeats (28 on easy) are already hits. Holding saves duplicated sandboxes and commands. It doesn't save latency, because the median held wait is about 20 s.
- **Step time:** the CLI gives the per-step change (median, 10th–90th percentile) and the total change. Total time is a paired comparison on the same trace, but a few steps with 600 s timeouts dominate it, and the median hides them. Report both.
- **Small step-time effects are inside the model's error.** The queue model's in-sample median step is 5–8% high.
  - `hold:10` changes total time by −1.0% to +4.4% across the four runs, and the two sandbox models disagree on the sign.
  - `hold:inf` changes total time by −3.8% to +1.4% on three runs. On TB medium cache-off it adds +18% (queue) / +22% (fixed): a held sibling waits out a call that then times out after 600 s, and has to run it itself. That is one event, but it shows the risk of waiting with no limit.
  - A call waits at most once. If the execution it waited for inserts nothing, it runs the call itself rather than queueing behind another sibling's retry.
  - A 10 s bound recovers most of the reuse anyway: tool hit rate 10.2% of a possible 11.5% on easy, 5.1% of 8.0% on medium.
- **Where the waiting goes.** Most held seconds are on tests, not tool calls. Held tool calls wait a median of about 5–8 s, held tests about 30–35 s. Most of the sandbox starts holding saves come from no-tool rollouts whose test now hits. On TB medium, the starts saved don't cover the waiting.
- **Containers free (`--sandbox free`).** Every container start costs nothing, as with a warm pool of pre-started containers (standard practice elsewhere, not in our setup). This isolates what a schedule does by itself. It matters because some time gains under the queue model come from *container-start contention*, not reuse:

  | Total time vs default | hold ≤ 5 s | ≤ 10 s | ≤ 30 s | no limit | scout ≤ 30 s |
  |---|---|---|---|---|---|
  | TB easy, queue (as run) | −0.6% | −1.0% | −2.1% | −2.5% | −0.1% |
  | TB easy, containers free | +0.5% | +1.7% | −0.1% | −0.1% | +2.6% |
  | TB medium, queue | −0.3% | −0.8% | −0.6% | −2.4% | −0.2% |
  | TB medium, containers free | +0.6% | +1.4% | +3.2% | −0.0% | +6.3% |

  - With free starts, a hit saves only the command or the test, and waiting costs time.
  - **Evaluate policies across scenarios (queue and free) and across runs.** Keep a policy only if it never costs much anywhere, or say which scenario a claim is for.
  - By that rule, short holds are the most robust: `hold:5` stays within ±0.8% everywhere.
  - Hit counts and work saved don't depend on the scenario.
- **Neither sandbox model is validated for non-default schedules.** The queue model's step-time gains come from fewer sandbox starts. The fixed model charges a delayed rollout its recorded queued start, which overstates the cost of delaying it. The figures show both.
- **No accelerator model.** Batch-by-prompt and priority policies, and the hit-rate vs. accelerator-idle trade-off, need one. Scout frees no GPU time here; in the queue model it only spreads out sandbox starts.

Validation (`scripts/validate_sim.py`):

| Check | Tool hits, sim / real | Test hits, sim / real | Median step s, sim / real |
|---|---|---|---|
| TB easy cache-on, replay, fixed | 4.9% / 4.9% | 15.6% / 15.6% | 56.2 / 56.2 |
| TB easy cache-on, replay, queue | 4.9% / 4.9% | 15.3% / 15.6% | 59.2 / 56.2 |
| TB medium cache-on, replay, queue | 0.0% / 0.0% | 2.7% / 3.3% | 85.8 / 79.3 |
| TB easy cache-off → cache-on, queue | 3.8% / 4.9% | 11.9% / 15.6% | 52.5 / 56.2 |
| TB medium cache-off → cache-on, queue | 0.2% / 0.0% | 3.4% / 3.3% | 69.0 / 79.3 |

The queue model matches the mean step to within 1% in-sample (199.2 vs 198.6 s, 198.4 vs 197.2 s), but its median step runs 5–8% high. The cache-off → cache-on rows predict the cache-on run from a different run with the same tasks, and only for `default`. Hit rates come out the right size but 20–25% low on easy. The median step is 7–13% low and the mean 35% / 25% low (127 vs 199 s, 147 vs 197 s). That run sampled different trajectories, including more 600 s timeouts.

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
