#!/usr/bin/env bash
# Data + model download. Run on a LOGIN node:
#   bash launch/data.sh            # toy SQL + terminal tasks + model (minutes)
#   FULL_SQL=1 bash launch/data.sh # also SkyRL-SQL-653 + OmniSQL DBs (~50 GB unzipped)
set -euo pipefail
source "$(dirname "$0")/env.sh"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

cd "$RSCHED_HOME"

# 1) model weights
hf download "$MODEL" --local-dir "$RSCHED_DATA/models/$(basename "$MODEL")"

# 2) toy SQL (3 small SQLite DBs, 16 questions) -- runs the SQL workflow without the big download
python -m rsched.data.make_toy_sql --db-root "$RSCHED_DATA/sql_db" --out "$RSCHED_DATA/sql_toy"

# 3) terminal tasks (built-in Terminal-Bench-style set)
python -m rsched.data.prepare_terminal --out "$RSCHED_DATA/terminal" --repeat 4

# 4) optional: the real SkyRL-SQL-653 workload
if [ "${FULL_SQL:-0}" = 1 ]; then
  hf download NovaSky-AI/SkyRL-SQL-653-data-newfmt --repo-type dataset --local-dir "$RSCHED_DATA/skyrl_sql"
  if [ ! -d "$RSCHED_DATA/sql_db/SynSQL-2.5M" ]; then
    hf download seeklhy/OmniSQL-datasets data.zip --repo-type dataset --local-dir "$RSCHED_DATA/omnisql_zip"
    # unzip into a tmp dir, then move the three DB trees next to toy/
    unzip -q "$RSCHED_DATA/omnisql_zip/data.zip" -d "$RSCHED_DATA/omnisql_unzip"
    src=$(dirname "$(find "$RSCHED_DATA/omnisql_unzip" -maxdepth 3 -type d -name 'SynSQL-2.5M' | head -1)")
    for d in SynSQL-2.5M spider bird; do [ -e "$src/$d" ] && mv "$src/$d" "$RSCHED_DATA/sql_db/"; done
  fi
  python -m rsched.data.prepare_sql --src "$RSCHED_DATA/skyrl_sql" --db-root "$RSCHED_DATA/sql_db" \
    --out "$RSCHED_DATA/sql"
fi
ls -R "$RSCHED_DATA" | head -40
