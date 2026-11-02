#!/usr/bin/env bash
# Replay-value ablation: the three bounded 2,000-update arms and their evaluations, nothing else. Detached, resumable
# (stages whose logs/<name>.exit holds 0 are skipped on rerun). Physical GPU 1 for CUDA and EGL.
set -u
P=/home/bigking/anaconda3/envs/lcg/bin/python
W=/home/bigking/scfeng/LCG-diagnostic-gaussian-h10-replayvalue-ablation
A=$W/outputs/ablation_20261027
D=$W/diagnostic_ablation
SR=/home/bigking/scfeng/LCG-stage3-gaussian-h10-ema-replayvalue/outputs/k4-walker-100k-entropy-stage3-seed1-gaussian-h10-ema-replayvalue-20261026
SNAP=$SR/pmpo_diagnostic/agent_pmpo_update_080600_epoch.pt
PROJECT=lcg-gaussian-h10-ema-replayvalue-ablation
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 OMP_NUM_THREADS=1
cd $W
mkdir -p $A/logs
id_of() { $P -c "import json; print(json.load(open('$A/launch/wandb_ids.json'))['$1']['id'])"; }
name_of() { $P -c "import json; print(json.load(open('$A/launch/wandb_ids.json'))['$1']['name'])"; }
stage() {  # stage NAME COMMAND...
  local name=$1; shift
  if [ -f $A/logs/$name.exit ] && [ "$(cat $A/logs/$name.exit)" = 0 ]; then echo "$(date -u +%FT%TZ) skip $name (done)" >> $A/progress.log; return 0; fi
  printf '%q ' "$@" > $A/logs/$name.command.txt; echo >> $A/logs/$name.command.txt
  echo "$(date -u +%FT%TZ) start $name" >> $A/progress.log
  "$@" > $A/logs/$name.log 2>&1
  local rc=$?
  echo $rc > $A/logs/$name.exit
  echo "$(date -u +%FT%TZ) done $name exit=$rc" >> $A/progress.log
  return $rc
}
fail() { echo "$(date -u +%FT%TZ) STOP: $1" >> $A/progress.log; echo 1 > $A/pipeline_exit_code.txt; exit 1; }

# 1. three arms in parallel
for arm in A B C; do
  stage train_$arm $P -u -B $D/run_arm.py --source-run $SR --snapshot $SNAP --arm $arm --out $A/arms/$arm --updates 2000 \
    --checkpoints 0,500,1000,2000 --prompt-stream dedicated --prompt-seed 190000000 --check-every 50 --fixed-inputs $A/fixed_inputs.pt \
    --wandb-id $(id_of $arm) --wandb-name $(name_of $arm) --wandb-project $PROJECT &
done
wait
for arm in A B C; do [ "$(cat $A/logs/train_$arm.exit)" = 0 ] || fail "train_$arm failed"; done

# 2. pre-evaluation checks on the main arms (identical start, layouts, B/C indices, frozen models, smoke agreement)
stage checks $P -B $D/check_arms.py $A/arms || fail "check_arms failed"
cp $A/logs/checks.log $A/main_checks.json
$P -c "import json,sys; r=json.load(open('$A/main_checks.json')); sys.exit(0 if r['identical_update0_checkpoint'] and r['all_boolean_checks_pass'] else 1)" || fail "main checks did not pass"
stage smoke_agreement $P -B $D/compare_smoke_main.py $A/smoke/arms $A/arms || fail "smoke/main disagreement"

# 3. shared update-0 eval30 (identical checkpoint in all arms), against the saved epoch-190 eval30
stage eval_shared_update0 $P -u -B $D/eval_arm.py --source-run $SR --out-dir $A/eval_shared \
  --spec 0:eval30:$A/arms/A/checkpoints/update_00000.pt --saved-reference $SR/eval30/eval30_100k_collection_end.json \
  --wandb-ids $(id_of A),$(id_of B),$(id_of C) --wandb-project $PROJECT || fail "shared eval failed"
$P -c "import json,sys; sys.exit(0 if json.load(open('$A/eval_shared/eval_eval30_update_00000.json'))['reproduces_saved_reference'] else 1)" || fail "update-0 eval30 does not reproduce the saved epoch-190 returns"

# 4. per-arm evaluations: four fixed seeds at 500 and 1,000; 30 seeds at 2,000
for arm in A B C; do
  stage eval_$arm $P -u -B $D/eval_arm.py --source-run $SR --out-dir $A/arms/$arm/eval \
    --spec 500:eval4:$A/arms/$arm/checkpoints/update_00500.pt --spec 1000:eval4:$A/arms/$arm/checkpoints/update_01000.pt \
    --spec 2000:eval30:$A/arms/$arm/checkpoints/update_02000.pt --wandb-ids $(id_of $arm) --wandb-project $PROJECT &
done
wait
for arm in A B C; do [ "$(cat $A/logs/eval_$arm.exit)" = 0 ] || fail "eval_$arm failed"; done

# 5. behaviour traces at update 2,000 (fixed seeds)
for arm in A B C; do
  stage behavior_$arm $P -u -B $D/behavior_arm.py --source-run $SR --checkpoint $A/arms/$arm/checkpoints/update_02000.pt \
    --eval30 $A/arms/$arm/eval/eval_eval30_update_02000.json --out $A/arms/$arm/behavior --label ep2000 &
done
wait
for arm in A B C; do [ "$(cat $A/logs/behavior_$arm.exit)" = 0 ] || fail "behavior_$arm failed"; done

# 6. source dataset unchanged
(cd $SR && find dataset/train -type f | sort | xargs sha256sum) > $A/records/source_dataset_sha256_after.txt
if cmp -s $A/records/source_dataset_sha256_before.txt $A/records/source_dataset_sha256_after.txt; then
  echo "source dataset unchanged ($(wc -l < $A/records/source_dataset_sha256_after.txt) files)" > $A/records/source_dataset_check.txt
else
  echo "SOURCE DATASET CHANGED" > $A/records/source_dataset_check.txt; fail "source dataset changed"
fi
sha256sum $SNAP >> $A/records/source_checkpoint_sha256_after.txt
echo "$(date -u +%FT%TZ) pipeline complete" >> $A/progress.log
echo 0 > $A/pipeline_exit_code.txt
