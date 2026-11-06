#!/usr/bin/env bash
# GPU node: examples A (SQL toy) and B (terminal toy), cache on and off. 1 prompt x 4 rollouts x 1 step.
set -uo pipefail
cd ~/sysml-team17-rsched && source launch/env.sh
for spec in "sql exA_sql-toy" "terminal exB_term-toy"; do
  set -- $spec
  for cache in tvcache off; do
    echo "=== $2 cache=$cache  $(date +%T)"
    WORKLOAD=$1 CACHE=$cache BATCH=1 N=4 STEPS=1 NAME=$2_cache-$([ $cache = off ] && echo off || echo on) \
      bash launch/run.sh > logs/$2_$cache.out 2>&1
    echo "    exit $?  $(date +%T)"; tail -c 400 $RSCHED_RUNS/$2_cache-$([ $cache = off ] && echo off || echo on)/report.json 2>/dev/null; echo
  done
done
echo "=== AB DONE"
