#!/usr/bin/env bash
# Pre-launch checks and detached launch of the ablation pipeline on physical GPU 1.
set -eu
P=/home/bigking/anaconda3/envs/lcg/bin/python
W=/home/bigking/scfeng/LCG-diagnostic-gaussian-h10-replayvalue-ablation
A=$W/outputs/ablation_20261027
L=$A/launch
UUID=GPU-923e4182-b64b-82be-7913-2bcf8993a3dd
mkdir -p $L
[ -d $A/arms ] && { echo "arms/ already exists; refusing to relaunch over it"; exit 1; }
nvidia-smi > $L/gpu_before_launch.txt
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader > $L/gpu_apps_before_launch.txt
if grep -q $UUID $L/gpu_apps_before_launch.txt; then echo "GPU 1 busy"; cat $L/gpu_apps_before_launch.txt; exit 1; fi
df -h /home | tail -1 > $L/disk_before_launch.txt
pgrep -af "run_arm.py|eval_arm.py|behavior_arm.py|src/main.py" > $L/conflicting_processes.txt || true
[ -s $L/conflicting_processes.txt ] && { echo "conflicting processes:"; cat $L/conflicting_processes.txt; exit 1; }
$P - <<PY
import json, secrets, wandb
api = wandb.Api()
project = "models-national-tsing-hua-university7947/lcg-gaussian-h10-ema-replayvalue-ablation"
names = dict(A="replayvalue-s1-ep190-A-independent-no-replay", B="replayvalue-s1-ep190-B-grouped-no-replay",
             C="replayvalue-s1-ep190-C-grouped-replay03")
ids = {}
for arm, name in names.items():
    wid = f"rvabl{arm.lower()}{secrets.token_hex(5)}"
    try:
        api.run(f"{project}/{wid}")
    except Exception as e:  # not found: the id is free
        absent = f"{type(e).__name__}: {str(e)[:120]}"
    else:
        raise SystemExit(f"W&B id {wid} already exists")
    ids[arm] = dict(id=wid, name=name, project=project.split("/")[1], entity=project.split("/")[0], absence_check=absent)
json.dump(ids, open("$L/wandb_ids.json", "w"), indent=1)
print(json.dumps(ids, indent=1))
PY
cp $W/diagnostic_ablation/pipeline.sh $L/pipeline_at_launch.sh
(cd $W && git rev-parse HEAD && git branch --show-current && git status --short) > $L/worktree_status_at_launch.txt
(cd $W && sha256sum src/models/pmpo_beta.py src/models/replay_value.py config/agent/pmpo_beta.yaml diagnostic_ablation/*.py diagnostic_ablation/*.sh) > $L/source_sha256_at_launch.txt
date -u +%FT%TZ > $L/start_utc.txt
cd $A
setsid nohup bash $W/diagnostic_ablation/pipeline.sh > $L/pipeline.nohup.log 2>&1 < /dev/null &
PID=$!
echo $PID > $L/pipeline.pid
setsid nohup $P -B $L/gpu_monitor.py $PID 1 $UUID $L 60 > $L/monitor.nohup.log 2>&1 < /dev/null &
echo $! > $L/monitor.pid
echo "pipeline pid $PID, monitor pid $(cat $L/monitor.pid)"
