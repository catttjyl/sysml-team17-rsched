#!/usr/bin/env bash
# Login node: toy data + Terminal-Bench tasks + one TB image. Run from repo root.
set -euo pipefail
source launch/env.sh
echo "=== [1/4] model + toy data"; bash launch/data.sh || [ $? = 141 ]
echo "=== [2/4] Terminal-Bench tasks"; FULL_TBENCH=1 bash launch/data.sh || [ $? = 141 ]
echo "=== [3/4] one-task dataset"; python scripts/examples/make_terminal_one.py "${TASK:-hello-world}"
echo "=== [4/4] build its image"; python scripts/tbench_images.py build --tasks "$RSCHED_DATA/terminal_one/tasks.json"
echo "=== PREP DONE"
