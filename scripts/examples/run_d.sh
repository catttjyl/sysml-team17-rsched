#!/usr/bin/env bash
# GPU node: example D, one Terminal-Bench easy task in podman-hpc containers, cache on and off.
set -uo pipefail
cd ~/sysml-team17-rsched && source launch/env.sh
T=$RSCHED_DATA/terminal_one/tasks.json
echo "=== oracle check  $(date +%T)"
python scripts/tbench_images.py load --tasks $T && python scripts/check_tbench_oracle.py --tasks $T
python scripts/tbench_images.py clean
grep -q '"ok"' $RSCHED_DATA/terminal_one/oracle_status.json || { echo "ORACLE FAILED"; echo "=== D DONE"; exit 1; }
for cache in tvcache off; do
  tag=$([ $cache = off ] && echo off || echo on)
  echo "=== exD_tb-one cache=$cache  $(date +%T)"
  WORKLOAD=terminal TERMINAL_DATA=terminal_one CACHE=$cache BATCH=1 N=4 STEPS=1 NAME=exD_tb-one_cache-$tag \
    bash launch/run.sh > logs/exD_$cache.out 2>&1
  echo "    exit $?  $(date +%T)"; tail -c 500 $RSCHED_RUNS/exD_tb-one_cache-$tag/report.json 2>/dev/null; echo
done
echo "=== D DONE"
