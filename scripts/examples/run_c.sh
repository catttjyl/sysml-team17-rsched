#!/usr/bin/env bash
# GPU node: example C, SkyRL-SQL real data, cache on and off. 1 prompt x 4 rollouts x 1 step.
set -uo pipefail
cd ~/sysml-team17-rsched && source launch/env.sh
for cache in tvcache off; do
  tag=$([ $cache = off ] && echo off || echo on)
  echo "=== exC_sql-full cache=$cache  $(date +%T)"
  WORKLOAD=sql SQL_DATA=sql CACHE=$cache BATCH=1 N=4 STEPS=1 NAME=exC_sql-full_cache-$tag \
    bash launch/run.sh > logs/exC_$cache.out 2>&1
  echo "    exit $?  $(date +%T)"; tail -c 500 $RSCHED_RUNS/exC_sql-full_cache-$tag/report.json 2>/dev/null; echo
done
echo "=== C DONE"
