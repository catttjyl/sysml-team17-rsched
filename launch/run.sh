#!/usr/bin/env bash
# Launch one veRL GRPO run with TVCache on the CURRENT GPU node.
# Use inside `salloc` (interactive) or via launch/train.sbatch.
#
#   WORKLOAD=sql      CACHE=tvcache bash launch/run.sh
#   WORKLOAD=terminal CACHE=off     bash launch/run.sh
#   WORKLOAD=sql SQL_DATA=sql STEPS=50 bash launch/run.sh     # real SkyRL-SQL data
#   bash launch/run.sh trainer.total_training_steps=2         # extra hydra overrides pass through
#
# Outputs in $RSCHED_RUNS/<name>/: timeline/*.jsonl (per-call trace), gpu.csv
# (nvidia-smi samples), tvcache.log, tvcache_dump.json, report.json, train.log.
set -euo pipefail
source "$(dirname "$0")/env.sh"

WORKLOAD="${WORKLOAD:-sql}"            # sql | terminal
CACHE="${CACHE:-tvcache}"              # tvcache | off
MODEL="${MODEL:-$RSCHED_DATA/models/Qwen2.5-1.5B-Instruct}"
SQL_DATA="${SQL_DATA:-sql_toy}"        # sql_toy (bring-up) | sql (SkyRL-SQL-653)
TERMINAL_DATA="${TERMINAL_DATA:-terminal}"  # terminal (built-in) | terminal_easy | terminal_medium
NGPUS="${NGPUS:-4}"
STEPS="${STEPS:-5}"
BATCH="${BATCH:-8}"                    # prompts per step
N="${N:-4}"                            # rollouts per prompt (GRPO group size)
MAX_TURNS="${MAX_TURNS:-6}"
PROMPT_LEN="${PROMPT_LEN:-4096}"
RESP_LEN="${RESP_LEN:-4096}"
TP="${TP:-1}"
NAME="${NAME:-${WORKLOAD}-${CACHE}-$(date +%m%d-%H%M%S)-${SLURM_JOB_ID:-local}}"

RUN_DIR="$RSCHED_RUNS/$NAME"
mkdir -p "$RUN_DIR"
echo "run dir: $RUN_DIR"

case "$WORKLOAD" in
  sql)
    DATA_DIR="$RSCHED_DATA/$SQL_DATA"
    TOOLS="$RSCHED_HOME/configs/sql_tools.yaml" ;;
  terminal)
    DATA_DIR="$RSCHED_DATA/$TERMINAL_DATA"
    if [ -f "$DATA_DIR/tasks.json" ]; then export RSCHED_TERMINAL_TASKS="$DATA_DIR/tasks.json"; fi
    TOOLS="$RSCHED_HOME/configs/terminal_tools.yaml" ;;
  *) echo "unknown WORKLOAD=$WORKLOAD"; exit 1 ;;
esac

# --- env read by rsched inside Ray workers (inherited from this shell) -------
export RSCHED_CACHE="$CACHE"
export RSCHED_TIMELINE_DIR="$RUN_DIR/timeline"
export RSCHED_SQL_DB_ROOT="$RSCHED_DATA/sql_db"
export RSCHED_SANDBOX_ROOT="${TMPDIR:-/tmp}/rsched-sbx-${SLURM_JOB_ID:-$$}"
export TVCACHE_HOST=127.0.0.1 TVCACHE_PORT=7878
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-WARN}"

PIDS=()
cleanup() {
  if [ "$CACHE" = tvcache ] && [ -n "${TVC_PID:-}" ]; then
    python "$RSCHED_HOME/scripts/dump_tvcache.py" "$RUN_DIR/tvcache_dump.json" || true
  fi
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
  rm -rf "$RSCHED_SANDBOX_ROOT"
  python -m rsched.analysis.timeline_report "$RSCHED_TIMELINE_DIR" --csv "$RUN_DIR/tool_calls.csv" \
    > "$RUN_DIR/report.json" 2>/dev/null && cat "$RUN_DIR/report.json" || true
}
trap cleanup EXIT

# --- TVCache server (same node; single port, so one run per node) ------------
if [ "$CACHE" = tvcache ]; then
  if (exec 3<>/dev/tcp/127.0.0.1/7878) 2>/dev/null; then
    echo "port 7878 already in use on $(hostname) -- another run?"; exit 1
  fi
  RUST_LOG=info "$TVCACHE_BIN" --eviction-policy=none > "$RUN_DIR/tvcache.log" 2>&1 &
  TVC_PID=$!; PIDS+=("$TVC_PID")
  for _ in $(seq 50); do (exec 3<>/dev/tcp/127.0.0.1/7878) 2>/dev/null && break; sleep 0.2; done
  echo "tvcache up (pid $TVC_PID)"
fi

# --- GPU utilization sampler (for idle-time accounting) ----------------------
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits \
  -lms 500 > "$RUN_DIR/gpu.csv" &
PIDS+=("$!")

# --- veRL --------------------------------------------------------------------
cd "$RSCHED_SRC/verl"
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/validation.parquet" \
  data.train_batch_size="$BATCH" \
  data.max_prompt_length="$PROMPT_LEN" \
  data.max_response_length="$RESP_LEN" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(( (PROMPT_LEN + RESP_LEN) * 2 )) \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TP" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.n="$N" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="$MAX_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOLS" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$RSCHED_HOME/configs/agent_loops.yaml" \
  actor_rollout_ref.rollout.agent.num_workers=4 \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name=rsched \
  trainer.experiment_name="$NAME" \
  trainer.default_local_dir="$RUN_DIR/ckpt" \
  trainer.n_gpus_per_node="$NGPUS" \
  trainer.nnodes=1 \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.total_training_steps="$STEPS" \
  trainer.total_epochs=100 \
  ray_kwargs.ray_init.num_cpus=64 \
  ray_kwargs.ray_init.runtime_env.py_executable="$RSCHED_VENV/bin/python3" \
  "$@" 2>&1 | tee "$RUN_DIR/train.log"
