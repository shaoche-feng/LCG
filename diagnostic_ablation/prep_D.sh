#!/usr/bin/env bash
# Baseline follow-up D: preparation, smoke and detached launch (physical GPU 1). Smoke outputs under baseline_D/smoke only.
set -eu
P=/home/bigking/anaconda3/envs/lcg/bin/python
W=/home/bigking/scfeng/LCG-diagnostic-gaussian-h10-replayvalue-ablation
AB=$W/outputs/ablation_20261027
B=$AB/baseline_D
D=$W/diagnostic_ablation
SR=/home/bigking/scfeng/LCG-stage3-gaussian-h10-emacritic/outputs/k4-walker-100k-entropy-stage3-seed1-gaussian-h10-emacritic-20261021
SNAP=$SR/pmpo_diagnostic/agent_pmpo_update_080600_epoch.pt
UUID=GPU-923e4182-b64b-82be-7913-2bcf8993a3dd
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 OMP_NUM_THREADS=1
[ -d $B/D ] && { echo "baseline_D/D exists; refusing to relaunch over it"; exit 1; }
mkdir -p $B/records $B/launch $B/smoke
cd $W
# provenance and source records
(cd $SR && find dataset/train -type f ! -name info.pt | sort | xargs sha256sum) > $B/records/source_dataset_episodes_sha256_before.txt
sha256sum $SR/dataset/train/info.pt > $B/records/source_info_sha256_before.txt
find $SR/dataset/train -type f ! -name info.pt -newer $SNAP > $B/records/episode_files_newer_than_snapshot.txt
sha256sum $SNAP > $B/records/source_checkpoint_sha256_before.txt
cp $SR/launch/run_metadata.json $B/records/baseline_run_metadata.json
cp $SR/config/trainer.yaml $B/records/baseline_trainer.yaml
diff -ruN --exclude=__pycache__ $SR/src $W/src > $B/records/diagnostic_src_vs_baseline_run_src.patch || true
sha256sum $D/*.py $D/*.sh > $B/records/diagnostic_scripts_sha256.txt
# fixed inputs: arm A's 600 states + the baseline's own epoch-190 states
$P -B $D/build_fixed_inputs.py $B/fixed_inputs_D.pt with-base190 > $B/records/build_fixed_inputs.log
# the snapshot's controller in diagnostic-checkpoint form, for the pre-learning evaluation
$P - <<PY
import torch
s = torch.load("$SNAP", map_location="cpu", weights_only=False)
ac = {k[len("actor_critic."):]: v for k, v in s["trainer_state"]["agent"].items() if k.startswith("actor_critic.")}
torch.save(dict(actor_critic=ac, pmpo_updates_counter=int(ac["updates"]), relative_update=0, source="$SNAP"), "$B/update0_from_snapshot.pt")
print("update0_from_snapshot.pt written; updates counter", int(ac["updates"]))
PY
# smoke: 30 updates (every start frame checked), stochastic eval on one environment seed, checks
$P -B $D/run_arm.py --source-run $SR --snapshot $SNAP --arm D --out $B/smoke/D --updates 30 --checkpoints 0,10,30 \
  --prompt-stream dedicated --prompt-seed 190000000 --check-every 1 --fixed-inputs $B/fixed_inputs_D.pt > $B/smoke/D.log 2>&1
echo "smoke train exit $?"
$P -u -B $D/eval_stochastic.py --source-run $SR --out-dir $B/smoke/stochastic --spec 0:$B/smoke/D/checkpoints/update_00000.pt --env-seeds 3001 > $B/smoke/stochastic.log 2>&1
echo "smoke stochastic exit $?"; grep "stochastic seed" $B/smoke/stochastic.log
$P -B $D/check_D.py $B/smoke/D $AB/arms/A > $B/smoke/check_D_smoke.json; echo "smoke check exit $?"
# GPU and process checks, W&B id
nvidia-smi > $B/launch/gpu_before_launch.txt
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader > $B/launch/gpu_apps_before_launch.txt
grep -q $UUID $B/launch/gpu_apps_before_launch.txt && { echo "GPU 1 busy"; exit 1; }
pgrep -af "run_arm.py|eval_arm.py|eval_stochastic.py|behavior_arm.py|src/main.py" > $B/launch/conflicting_processes.txt || true
[ -s $B/launch/conflicting_processes.txt ] && { cat $B/launch/conflicting_processes.txt; exit 1; }
df -h /home | tail -1 > $B/launch/disk_before_launch.txt
$P - <<PY
import json, secrets, wandb
api = wandb.Api()
project = "models-national-tsing-hua-university7947/lcg-gaussian-h10-ema-replayvalue-ablation"
wid = f"rvabld{secrets.token_hex(5)}"
try:
    api.run(f"{project}/{wid}")
except Exception as e:
    absent = f"{type(e).__name__}: {str(e)[:120]}"
else:
    raise SystemExit(f"W&B id {wid} already exists")
d = dict(id=wid, name="baseline-s1-ep190-D-independent-no-replay", project=project.split("/")[1], entity=project.split("/")[0], absence_check=absent)
json.dump(d, open("$B/launch/wandb_id.json", "w"), indent=1); print(json.dumps(d, indent=1))
PY
cp $D/pipeline_D.sh $B/launch/pipeline_at_launch.sh
cp $AB/launch/gpu_monitor.py $B/launch/
(cd $W && git rev-parse HEAD && git branch --show-current && git status --short) > $B/launch/worktree_status_at_launch.txt
date -u +%FT%TZ > $B/launch/start_utc.txt
cd $B
setsid nohup bash $D/pipeline_D.sh > $B/launch/pipeline.nohup.log 2>&1 < /dev/null &
PID=$!; echo $PID > $B/launch/pipeline.pid
setsid nohup $P -B $B/launch/gpu_monitor.py $PID 1 $UUID $B/launch 60 > $B/launch/monitor.nohup.log 2>&1 < /dev/null &
echo $! > $B/launch/monitor.pid
echo "pipeline pid $PID"
