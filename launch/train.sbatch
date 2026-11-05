#!/bin/bash
# Batch version of launch/run.sh. Submit from the repo root:
#   sbatch -A <account> launch/train.sbatch                          # sql + tvcache
#   sbatch -A <account> --export=ALL,WORKLOAD=terminal,CACHE=off launch/train.sbatch
# Use -q debug (<=30 min) for bring-up, -q regular for real runs.
# 80 GB A100s: add  -C "gpu&hbm80g"
#SBATCH -J rsched
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
bash launch/run.sh "$@"
