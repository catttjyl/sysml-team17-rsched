#!/usr/bin/env bash
# One-time install. Run on a LOGIN node (needs internet; ~15-25 min, ~20 GB):
#   bash install.sh

set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$RSCHED_SRC" "$RSCHED_DATA" "$RSCHED_RUNS" "$UV_CACHE_DIR"

# --- uv ---------------------------------------------------------------------
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# --- rust (for the TVCache server) ------------------------------------------
if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
fi

clone() {  # clone <url> <dir> <ref>
  local url=$1 dir=$2 ref=$3
  if [ ! -d "$dir/.git" ]; then git clone "$url" "$dir"; fi
  git -C "$dir" fetch --tags origin
  git -C "$dir" checkout -q "$ref"
  git -C "$dir" pull -q --ff-only 2>/dev/null || true   # no-op on tags
  echo "$(basename "$dir") @ $(git -C "$dir" rev-parse --short HEAD)"
}
clone https://github.com/volcengine/verl.git          "$RSCHED_SRC/verl"         "$VERL_REF"
clone https://github.com/abhivijay96/tvcache-rust.git "$RSCHED_SRC/tvcache-rust" "$TVCACHE_REF"
clone https://github.com/abhivijay96/tvclient.git     "$RSCHED_SRC/tvclient"     "$TVCLIENT_REF"
# veRL v0.9.1's uv.lock points at a flash-attn wheel URL that now 404s; the wheel moved to a py/torch-tagged release.
sed -i 's|releases/download/flash-attention-v2.8.3/flash_attn-2.8.3-cp312|releases/download/flash-attention-v2.8.3-py3.12-torch2.11.0/flash_attn-2.8.3-cp312|' "$RSCHED_SRC/verl/uv.lock"

# --- TVCache server ---------------------------------------------------------
( cd "$RSCHED_SRC/tvcache-rust" && CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=gcc cargo build --release )
echo "tvcache binary: $TVCACHE_BIN"

# --- Python env: veRL's own lockfile (vLLM rollout + FSDP training) -----------
# UV_PROJECT_ENVIRONMENT puts the venv outside the checkout.
( cd "$RSCHED_SRC/verl" && \
  UV_PROJECT_ENVIRONMENT="$RSCHED_VENV" uv sync --frozen --python 3.12 --extra vllm --extra fsdp )

# Our code + tvclient as editable installs on top. --no-deps so the locked env
# is not touched. (Do NOT launch with `uv run` afterwards: it would re-sync and
# drop these.)
VIRTUAL_ENV="$RSCHED_VENV" uv pip install --no-deps -e "$RSCHED_SRC/tvclient" -e "$RSCHED_HOME"
VIRTUAL_ENV="$RSCHED_VENV" uv pip install "huggingface_hub[cli]" pytest

# --- sanity -------------------------------------------------------------------
"$RSCHED_VENV/bin/python" - <<'EOF'
import torch, vllm, verl, tvclient, rsched
print("torch", torch.__version__, "| cuda build", torch.version.cuda)
print("vllm", vllm.__version__, "| verl", verl.__version__)
from rsched.agent_loop import TVCacheToolAgentLoop  # import check
print("rsched agent loop import OK")
EOF
echo "Install done. Next: bash launch/data.sh"
