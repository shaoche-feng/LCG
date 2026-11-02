#!/usr/bin/env bash
# Pre-launch checks (idle GPU 1, no conflicting jobs, fresh W&B ids) and detached launch of the swap pipeline.
set -eu
P=/home/bigking/anaconda3/envs/lcg/bin/python
W=/home/bigking/scfeng/LCG-diagnostic-gaussian-h10-ema-component-swap
O=$W/outputs/swap_20261027
L=$O/launch
UUID=GPU-923e4182-b64b-82be-7913-2bcf8993a3dd
mkdir -p $L
[ -d $O/arms ] && { echo "arms/ exists; refusing to relaunch over it"; exit 1; }
nvidia-smi > $L/gpu_before_launch.txt
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader > $L/gpu_apps_before_launch.txt
grep -q $UUID $L/gpu_apps_before_launch.txt && { echo "GPU 1 busy"; cat $L/gpu_apps_before_launch.txt; exit 1; }
pgrep -af "swap_arm.py|run_arm.py|eval_arm.py|behavior_arm.py|src/main.py" > $L/conflicting_processes.txt || true
[ -s $L/conflicting_processes.txt ] && { cat $L/conflicting_processes.txt; exit 1; }
df -h /home | tail -1 > $L/disk_before_launch.txt
$P - <<PY
import json, secrets, wandb
api = wandb.Api()
project = "models-national-tsing-hua-university7947/lcg-gaussian-h10-ema-controller-modeldata-swap"
names = dict(S00="swap-50k-ep90-S00-ctrl-seed0-modeldata-seed0", S01="swap-50k-ep90-S01-ctrl-seed0-modeldata-seed1",
             S10="swap-50k-ep90-S10-ctrl-seed1-modeldata-seed0", S11="swap-50k-ep90-S11-ctrl-seed1-modeldata-seed1")
ids = {}
for arm, name in names.items():
    wid = f"swp{arm.lower()}{secrets.token_hex(5)}"
    try:
        api.run(f"{project}/{wid}")
    except Exception as e:
        absent = f"{type(e).__name__}: {str(e)[:120]}"
    else:
        raise SystemExit(f"W&B id {wid} already exists")
    ids[arm] = dict(id=wid, name=name, project=project.split("/")[1], entity=project.split("/")[0], absence_check=absent)
json.dump(ids, open("$L/wandb_ids.json", "w"), indent=1)
print(json.dumps({a: (v["id"], v["name"]) for a, v in ids.items()}, indent=1))
PY
cp $W/diagnostic_swap/pipeline_swap.sh $L/pipeline_at_launch.sh
cp /home/bigking/scfeng/LCG-diagnostic-gaussian-h10-replayvalue-ablation/outputs/ablation_20261027/launch/gpu_monitor.py $L/
(cd $W && git rev-parse HEAD && git branch --show-current && git status --short) > $L/worktree_status_at_launch.txt
(cd $W && sha256sum src/models/pmpo_beta.py config/agent/pmpo_beta.yaml diagnostic_swap/*.py diagnostic_swap/*.sh) > $L/source_sha256_at_launch.txt
date -u +%FT%TZ > $L/start_utc.txt
cd $O
setsid nohup bash $W/diagnostic_swap/pipeline_swap.sh > $L/pipeline.nohup.log 2>&1 < /dev/null &
PID=$!; echo $PID > $L/pipeline.pid
setsid nohup $P -B $L/gpu_monitor.py $PID 1 $UUID $L 60 > $L/monitor.nohup.log 2>&1 < /dev/null &
echo $! > $L/monitor.pid
echo "pipeline pid $PID"
