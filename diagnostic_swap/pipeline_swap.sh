#!/usr/bin/env bash
# Component swap: the four bounded 2,000-update arms (S00, S01, S10, S11) and their evaluations, nothing else.
# Detached, resumable (stages with logs/<name>.exit = 0 are skipped). Physical GPU 1 for CUDA and EGL.
set -u
P=/home/bigking/anaconda3/envs/lcg/bin/python
W=/home/bigking/scfeng/LCG-diagnostic-gaussian-h10-ema-component-swap
O=$W/outputs/swap_20261027
D=$W/diagnostic_swap
E=/home/bigking/scfeng/LCG-stage3-gaussian-h10-emacritic/outputs
R0=$E/k4-walker-100k-entropy-stage3-seed0-gaussian-h10-emacritic-20261022
R1=$E/k4-walker-100k-entropy-stage3-seed1-gaussian-h10-emacritic-20261021
SN=pmpo_diagnostic/agent_pmpo_update_040600_epoch.pt
PROJECT=lcg-gaussian-h10-ema-controller-modeldata-swap
ARMS="S00 S01 S10 S11"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 OMP_NUM_THREADS=1
cd $W
mkdir -p $O/logs
wid() { $P -c "import json; print(json.load(open('$O/launch/wandb_ids.json'))['$1']['$2'])"; }
crun() { eval echo \$R${1:1:1}; }   # controller source run of arm
mrun() { eval echo \$R${1:2:1}; }   # model/data source run of arm
stage() {
  local name=$1; shift
  if [ -f $O/logs/$name.exit ] && [ "$(cat $O/logs/$name.exit)" = 0 ]; then echo "$(date -u +%FT%TZ) skip $name (done)" >> $O/progress.log; return 0; fi
  printf '%q ' "$@" > $O/logs/$name.command.txt; echo >> $O/logs/$name.command.txt
  echo "$(date -u +%FT%TZ) start $name" >> $O/progress.log
  "$@" > $O/logs/$name.log 2>&1
  local rc=$?
  echo $rc > $O/logs/$name.exit
  echo "$(date -u +%FT%TZ) done $name exit=$rc" >> $O/progress.log
  return $rc
}
fail() { echo "$(date -u +%FT%TZ) STOP: $1" >> $O/progress.log; echo 1 > $O/pipeline_exit_code.txt; exit 1; }

# 1. four arms in parallel
for arm in $ARMS; do
  RC=$(crun $arm); RM=$(mrun $arm)
  stage train_$arm $P -u -B $D/swap_arm.py --source-run $RC --snapshot $RC/$SN --model-source-run $RM --model-snapshot $RM/$SN --arm-name $arm \
    --out $O/arms/$arm --updates 2000 --checkpoints 0,500,1000,2000 --prompt-stream dedicated --prompt-seed 90000000 --check-every 50 \
    --fixed-inputs $O/fixed_inputs_swap.pt --wandb-id $(wid $arm id) --wandb-name $(wid $arm name) --wandb-project $PROJECT &
done
wait
for arm in $ARMS; do [ "$(cat $O/logs/train_$arm.exit)" = 0 ] || fail "train_$arm failed"; done

# 2. wiring checks on the main arms
stage checks $P -B $D/check_swap.py $O/arms || fail "wiring checks failed"
cp $O/logs/checks.log $O/main_checks.json

# 3. update-0 eval30 of every arm's own update-0 checkpoint (swap must not change real-environment returns), against
#    the controller's saved 50k eval30; then four fixed seeds at 500 / 1,000 and 30 seeds at 2,000
for arm in $ARMS; do
  RC=$(crun $arm)
  stage eval_$arm $P -u -B $D/eval_arm.py --source-run $RC --out-dir $O/arms/$arm/eval \
    --spec 0:eval30:$O/arms/$arm/checkpoints/update_00000.pt --saved-reference $RC/eval30/eval30_50k.json \
    --spec 500:eval4:$O/arms/$arm/checkpoints/update_00500.pt --spec 1000:eval4:$O/arms/$arm/checkpoints/update_01000.pt \
    --spec 2000:eval30:$O/arms/$arm/checkpoints/update_02000.pt --wandb-ids $(wid $arm id) --wandb-project $PROJECT &
done
wait
for arm in $ARMS; do [ "$(cat $O/logs/eval_$arm.exit)" = 0 ] || fail "eval_$arm failed"; done
$P - <<PY || fail "update-0 evaluation check failed"
import json, sys
O = "$O"
e = {a: json.load(open(f"{O}/arms/{a}/eval/eval_eval30_update_00000.json")) for a in "S00 S01 S10 S11".split()}
ok = all(e[a]["reproduces_saved_reference"] for a in e) and e["S00"]["returns"] == e["S01"]["returns"] and e["S10"]["returns"] == e["S11"]["returns"]
json.dump(dict(all_reproduce_saved_50k={a: e[a]["reproduces_saved_reference"] for a in e},
               swap_leaves_update0_returns_unchanged=dict(ctrl0=e["S00"]["returns"] == e["S01"]["returns"], ctrl1=e["S10"]["returns"] == e["S11"]["returns"]), ok=ok),
          open(f"{O}/update0_eval_check.json", "w"), indent=1)
sys.exit(0 if ok else 1)
PY

# 4. behaviour traces at 2,000 (fixed seeds)
for arm in $ARMS; do
  stage behavior_$arm $P -u -B $D/behavior_arm.py --source-run $(crun $arm) --checkpoint $O/arms/$arm/checkpoints/update_02000.pt \
    --eval30 $O/arms/$arm/eval/eval_eval30_update_02000.json --out $O/arms/$arm/behavior --label ep2000 &
done
wait
for arm in $ARMS; do [ "$(cat $O/logs/behavior_$arm.exit)" = 0 ] || fail "behavior_$arm failed"; done

# 5. source data and checkpoints unchanged
status=ok
for s in 0 1; do R=$(eval echo \$R$s)
  (cd $R && find dataset/train -type f ! -name info.pt | sort | xargs sha256sum) > $O/records/seed${s}_dataset_episodes_sha256_after.txt
  sha256sum $R/dataset/train/info.pt > $O/records/seed${s}_info_sha256_after.txt
  sha256sum $R/$SN > $O/records/seed${s}_snapshot_sha256_after.txt
  for f in dataset_episodes info snapshot; do cmp -s <(cut -d" " -f1 $O/records/seed${s}_${f}_sha256_before.txt) <(cut -d" " -f1 $O/records/seed${s}_${f}_sha256_after.txt) || status="changed:seed${s}_$f"; done
done
echo "source datasets, info.pt and epoch-90 snapshots: $status" > $O/records/source_data_check.txt
[ $status = ok ] || fail "source data changed"
echo "$(date -u +%FT%TZ) pipeline complete" >> $O/progress.log
echo 0 > $O/pipeline_exit_code.txt
