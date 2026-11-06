#!/usr/bin/env bash
# GPU node: does podman-hpc run hang when 4 containers start at once?
cd ~/sysml-team17-rsched && source launch/env.sh
T=$RSCHED_DATA/terminal_one/tasks.json
IMG=$(python -c "import json;print(json.load(open('$T'))[0]['image'])")
echo "image: $IMG"
python scripts/tbench_images.py load --tasks $T
podman-hpc images | head -5
echo "=== 1 container  $(date +%T)"
time timeout 120 podman-hpc run -d --name dbg-1 --entrypoint= --pull=never --stop-timeout=0 "$IMG" sleep infinity; echo "rc=$?"
echo "=== 4 at once  $(date +%T)"
for i in 2 3 4 5; do ( s=$(date +%s); timeout 180 podman-hpc run -d --name dbg-$i --entrypoint= --pull=never --stop-timeout=0 "$IMG" sleep infinity >/dev/null 2>/tmp/dbg-$i.err; echo "dbg-$i rc=$? $(( $(date +%s)-s ))s $(tail -c 200 /tmp/dbg-$i.err)" ) & done; wait
podman-hpc ps -a --format '{{.Names}} {{.Status}}' | head
python scripts/tbench_images.py clean; podman-hpc rm -f dbg-1 dbg-2 dbg-3 dbg-4 dbg-5 >/dev/null 2>&1
echo "=== DEBUG DONE  $(date +%T)"
