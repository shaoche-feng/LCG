#!/usr/bin/env bash
# Baseline follow-up D: arm-A settings (32 independent starts, replay loss off, frozen world/reward models, fixed dataset)
# on the H10 scalar-EMA seed-1 baseline h10ema5bc11033 from its own epoch-190 checkpoint. 2,000 updates, then
# evaluations, stochastic evaluation, behaviour traces, data checks. Nothing else. Detached, resumable; physical GPU 1.
set -u
P=/home/bigking/anaconda3/envs/lcg/bin/python
W=/home/bigking/scfeng/LCG-diagnostic-gaussian-h10-replayvalue-ablation
AB=$W/outputs/ablation_20261027
B=$AB/baseline_D
D=$W/diagnostic_ablation
SR=/home/bigking/scfeng/LCG-stage3-gaussian-h10-emacritic/outputs/k4-walker-100k-entropy-stage3-seed1-gaussian-h10-emacritic-20261021
SNAP=$SR/pmpo_diagnostic/agent_pmpo_update_080600_epoch.pt
PROJECT=lcg-gaussian-h10-ema-replayvalue-ablation
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 OMP_NUM_THREADS=1
cd $W
mkdir -p $B/logs
WID=$($P -c "import json; print(json.load(open('$B/launch/wandb_id.json'))['id'])")
WNAME=$($P -c "import json; print(json.load(open('$B/launch/wandb_id.json'))['name'])")
stage() {
  local name=$1; shift
  if [ -f $B/logs/$name.exit ] && [ "$(cat $B/logs/$name.exit)" = 0 ]; then echo "$(date -u +%FT%TZ) skip $name (done)" >> $B/progress.log; return 0; fi
  printf '%q ' "$@" > $B/logs/$name.command.txt; echo >> $B/logs/$name.command.txt
  echo "$(date -u +%FT%TZ) start $name" >> $B/progress.log
  "$@" > $B/logs/$name.log 2>&1
  local rc=$?
  echo $rc > $B/logs/$name.exit
  echo "$(date -u +%FT%TZ) done $name exit=$rc" >> $B/progress.log
  return $rc
}
fail() { echo "$(date -u +%FT%TZ) STOP: $1" >> $B/progress.log; echo 1 > $B/pipeline_exit_code.txt; exit 1; }

# 0. reproduce the saved epoch-190 deterministic eval30 BEFORE learning (snapshot controller = update-0 state)
stage eval_update0_prelearning $P -u -B $D/eval_arm.py --source-run $SR --out-dir $B/eval_update0 \
  --spec 0:eval30:$B/update0_from_snapshot.pt --saved-reference $SR/eval30/eval30_100k_collection_end.json || fail "pre-learning eval failed"
$P -c "import json,sys; sys.exit(0 if json.load(open('$B/eval_update0/eval_eval30_update_00000.json'))['reproduces_saved_reference'] else 1)" || fail "update-0 eval30 does not reproduce the saved epoch-190 returns"

# 1. training
stage train_D $P -u -B $D/run_arm.py --source-run $SR --snapshot $SNAP --arm D --out $B/D --updates 2000 \
  --checkpoints 0,500,1000,2000 --prompt-stream dedicated --prompt-seed 190000000 --check-every 50 --fixed-inputs $B/fixed_inputs_D.pt \
  --wandb-id $WID --wandb-name $WNAME --wandb-project $PROJECT || fail "train_D failed"

# 2. checks (incl. update-0 checkpoint == the evaluated snapshot controller, and agreement with the smoke)
stage checks_D $P -B $D/check_D.py $B/D $AB/arms/A $B/smoke/D || fail "checks failed"
$P - <<PY || fail "update-0 checkpoint differs from the pre-learning evaluated controller"
import sys, torch
sys.path.insert(0, "$D"); from run_arm import canon_hash
a = torch.load("$B/update0_from_snapshot.pt", map_location="cpu", weights_only=False)["actor_critic"]
b = torch.load("$B/D/checkpoints/update_00000.pt", map_location="cpu", weights_only=False)["actor_critic"]
sys.exit(0 if canon_hash(a) == canon_hash(b) else 1)
PY
echo "update-0 checkpoint equals the pre-learning evaluated snapshot controller" > $B/update0_identity.txt
$P -c "
import json; d = json.load(open('$B/eval_update0/eval_eval30_update_00000.json'))
import wandb; r = wandb.init(project='$PROJECT', entity='models-national-tsing-hua-university7947', id='$WID', resume='must')
r.define_metric('diagnostic_update'); r.define_metric('*', step_metric='diagnostic_update')
r.log({'diagnostic_update': 0, **{f'eval30/return_{m}': d[m] for m in ('mean','median','std','min','max')}, **{f'eval30/{m}': d[m] for m in ('frac_gt_200','frac_gt_400','frac_gt_500','frac_gt_600')}})
r.finish()" > $B/logs/wandb_log_update0.log 2>&1 || fail "W&B update-0 log failed"

# 3. deterministic evaluations: four fixed seeds at 500 and 1,000; 30 seeds at 2,000
stage eval_D $P -u -B $D/eval_arm.py --source-run $SR --out-dir $B/D/eval \
  --spec 500:eval4:$B/D/checkpoints/update_00500.pt --spec 1000:eval4:$B/D/checkpoints/update_01000.pt \
  --spec 2000:eval30:$B/D/checkpoints/update_02000.pt --wandb-ids $WID --wandb-project $PROJECT || fail "eval_D failed"

# 4. stochastic evaluation at 0 and 2,000 (30 environment seeds x 3 action seeds)
stage eval_stochastic_D $P -u -B $D/eval_stochastic.py --source-run $SR --out-dir $B/D/eval \
  --spec 0:$B/D/checkpoints/update_00000.pt --spec 2000:$B/D/checkpoints/update_02000.pt --wandb-id $WID --wandb-project $PROJECT || fail "stochastic eval failed"

# 5. behaviour traces at 2,000
stage behavior_D $P -u -B $D/behavior_arm.py --source-run $SR --checkpoint $B/D/checkpoints/update_02000.pt \
  --eval30 $B/D/eval/eval_eval30_update_02000.json --out $B/D/behavior --label ep2000 || fail "behavior_D failed"

# 6. source data unchanged
(cd $SR && find dataset/train -type f ! -name info.pt | sort | xargs sha256sum) > $B/records/source_dataset_episodes_sha256_after.txt
cmp -s $B/records/source_dataset_episodes_sha256_before.txt $B/records/source_dataset_episodes_sha256_after.txt \
  && echo "source dataset episodes unchanged ($(wc -l < $B/records/source_dataset_episodes_sha256_after.txt) files); info.pt sha256 before/after: $(cut -c1-16 $B/records/source_info_sha256_before.txt) / $(sha256sum $SR/dataset/train/info.pt | cut -c1-16)" > $B/records/source_dataset_check.txt \
  || { echo "SOURCE DATASET CHANGED" > $B/records/source_dataset_check.txt; fail "source dataset changed"; }
sha256sum $SNAP > $B/records/source_checkpoint_sha256_after.txt
cmp -s <(cut -d" " -f1 $B/records/source_checkpoint_sha256_before.txt) <(cut -d" " -f1 $B/records/source_checkpoint_sha256_after.txt) || fail "source checkpoint changed"
echo "$(date -u +%FT%TZ) pipeline complete" >> $B/progress.log
echo 0 > $B/pipeline_exit_code.txt
