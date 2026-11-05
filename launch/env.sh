# Source this on Perlmutter (login or compute node):  source launch/env.sh
# Everything big (venv, model weights, datasets, caches) goes under $RSCHED_ROOT,
# which defaults to your scratch.
# NOTE: $PSCRATCH is purged for files not accessed in 8 weeks.

export RSCHED_HOME="${RSCHED_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export RSCHED_ROOT="${RSCHED_ROOT:-$PSCRATCH/rsched}"
export RSCHED_DATA="${RSCHED_DATA:-$RSCHED_ROOT/data}"
export RSCHED_SRC="$RSCHED_ROOT/src"            # verl, tvcache-rust, tvclient checkouts
export RSCHED_VENV="$RSCHED_ROOT/venv"
export RSCHED_RUNS="${RSCHED_RUNS:-$RSCHED_ROOT/runs}"

# Pinned upstream versions (bump deliberately, together).
export VERL_REF="${VERL_REF:-v0.9.1}"
export TVCACHE_REF="${TVCACHE_REF:-main}"
export TVCLIENT_REF="${TVCLIENT_REF:-main}"

# Caches off $HOME (40 GB quota)
export UV_CACHE_DIR="$RSCHED_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$RSCHED_ROOT/.cache/uv-python"
export HF_HOME="$RSCHED_ROOT/.cache/huggingface"
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
export PATH="$HOME/.local/bin:$CARGO_HOME/bin:$PATH"

export TVCACHE_BIN="$RSCHED_SRC/tvcache-rust/target/release/tvcache"

# veRL brings its own CUDA 13 wheels (torch 2.11+cu130, vLLM 0.24); Perlmutter's
# driver (580.x) supports CUDA 13. Don't mix in the cudatoolkit / python modules.
module unload cudatoolkit 2>/dev/null || true

if [ -f "$RSCHED_VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$RSCHED_VENV/bin/activate"
fi
