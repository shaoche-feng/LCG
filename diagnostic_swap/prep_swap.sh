#!/usr/bin/env bash
# Component swap: preparation, pre-learning reproductions, epoch-90 behaviour traces, fixed inputs and smoke (no main
# arms). Physical GPU 1. Outputs under outputs/swap_20261027 only; source runs are read-only.
set -eu
P=/home/bigking/anaconda3/envs/lcg/bin/python
W=/home/bigking/scfeng/LCG-diagnostic-gaussian-h10-ema-component-swap
O=$W/outputs/swap_20261027
D=$W/diagnostic_swap
E=/home/bigking/scfeng/LCG-stage3-gaussian-h10-emacritic/outputs
R0=$E/k4-walker-100k-entropy-stage3-seed0-gaussian-h10-emacritic-20261022
R1=$E/k4-walker-100k-entropy-stage3-seed1-gaussian-h10-emacritic-20261021
SN=pmpo_diagnostic/agent_pmpo_update_040600_epoch.pt
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 OMP_NUM_THREADS=1
mkdir -p $O/records $O/smoke $O/logs
cd $W
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $O/prep_progress.log; }
# records: source checkpoints, configs, dataset episode hashes (ids < 100 at epoch 90; all files for the after-check)
for s in 0 1; do R=$(eval echo \$R$s)
  sha256sum $R/$SN > $O/records/seed${s}_snapshot_sha256_before.txt
  (cd $R && find dataset/train -type f ! -name info.pt | sort | xargs sha256sum) > $O/records/seed${s}_dataset_episodes_sha256_before.txt
  sha256sum $R/dataset/train/info.pt > $O/records/seed${s}_info_sha256_before.txt
  cp $R/launch/run_metadata.json $O/records/seed${s}_run_metadata.json; cp $R/config/trainer.yaml $O/records/seed${s}_trainer.yaml
  cp $R/eval30/eval30_50k.json $O/records/seed${s}_saved_eval30_50k.json
done
diff $O/records/seed0_trainer.yaml $O/records/seed1_trainer.yaml > $O/records/config_diff_seed0_vs_seed1.txt || true
# monitoring patch + unit tests
$P $D/patch_monitor.py $W > $O/records/patch_monitor.log
diff -u <(tr -d "\r" < $O/records/carried_source_snapshot/src/models/pmpo_beta.py) <(tr -d "\r" < $W/src/models/pmpo_beta.py) > $O/records/monitor_diff_vs_carried_pmpo_beta.patch || true
$P -m pytest -q -p no:cacheprovider tests/models/test_ema_critic.py > $O/smoke/unit_tests_ema_critic.txt 2>&1; log "unit tests: $(tail -1 $O/smoke/unit_tests_ema_critic.txt)"
# Trainer-path reproduction with the swap driver (scalar-EMA smoke: update-15 snapshot -> saved update-24 epoch snapshot);
# once with the default single snapshot and once with the model/data bundle passed explicitly from the same snapshot
SM=$E/smoke_ema_20261021
$P -B $D/swap_arm.py --source-run $SM --snapshot $SM/pmpo_diagnostic/agent_pmpo_update_000015.pt --compare-to $SM/pmpo_diagnostic/agent_pmpo_update_000024_epoch.pt \
  --prompt-stream shared --check-every 1 --out $O/smoke/repro_trainer_path > $O/smoke/repro_trainer_path.log 2>&1
$P -B $D/swap_arm.py --source-run $SM --snapshot $SM/pmpo_diagnostic/agent_pmpo_update_000015.pt --model-source-run $SM --model-snapshot $SM/pmpo_diagnostic/agent_pmpo_update_000015.pt \
  --compare-to $SM/pmpo_diagnostic/agent_pmpo_update_000024_epoch.pt --prompt-stream shared --check-every 1 --out $O/smoke/repro_trainer_path_explicit_bundle > $O/smoke/repro_trainer_path_explicit_bundle.log 2>&1
for d in repro_trainer_path repro_trainer_path_explicit_bundle; do log "$d: $($P -c "import json; f=json.load(open('$O/smoke/$d/final.json')); print(f['actor_critic_identical'], f['pmpo_optimizers_identical'], f['max_abs_metric_diff'])")"; done
# update-0 controllers in diagnostic-checkpoint form; pre-learning eval30 reproductions
for s in 0 1; do R=$(eval echo \$R$s)
  $P - <<PY
import torch
s = torch.load("$R/$SN", map_location="cpu", weights_only=False)
ac = {k[len("actor_critic."):]: v for k, v in s["trainer_state"]["agent"].items() if k.startswith("actor_critic.")}
torch.save(dict(actor_critic=ac, pmpo_updates_counter=int(ac["updates"]), relative_update=0, source="$R/$SN"), "$O/update0_ctrl$s.pt")
PY
  $P -u -B $D/eval_arm.py --source-run $R --out-dir $O/eval_prelearning/ctrl$s --spec 0:eval30:$O/update0_ctrl$s.pt --saved-reference $R/eval30/eval30_50k.json > $O/logs/eval_prelearning_ctrl$s.log 2>&1
  log "ctrl$s pre-learning eval30 reproduces saved 50k: $($P -c "import json; d=json.load(open('$O/eval_prelearning/ctrl$s/eval_eval30_update_00000.json')); print(d['reproduces_saved_reference'], d['max_abs_diff_vs_saved_reference'], round(d['mean'],1))")"
  $P -u -B $D/behavior_arm.py --source-run $R --checkpoint $O/update0_ctrl$s.pt --eval30 $R/eval30/eval30_50k.json --out $O/behavior_update0/ctrl$s --label ep90 > $O/logs/behavior_update0_ctrl$s.log 2>&1
  log "ctrl$s epoch-90 traces: $(grep -c 'reproduces True' $O/logs/behavior_update0_ctrl$s.log)/4 reproduce"
done
$P -B $D/build_fixed_inputs_swap.py $O/fixed_inputs_swap.pt ctrl0_ep90=$O/behavior_update0/ctrl0:ep90 ctrl1_ep90=$O/behavior_update0/ctrl1:ep90 > $O/records/build_fixed_inputs.log
env CUDA_VISIBLE_DEVICES= $P -B $D/suitability.py $O > $O/logs/suitability.log 2>&1; log "suitability written"
# smoke: four arms x 20 updates (every start frame checked), checks, resume test
S=$O/smoke/arms; rm -rf $S $O/smoke/resume_S01; mkdir -p $S $O/smoke/resume_S01
COMMON="--updates 20 --checkpoints 0,10,20 --prompt-stream dedicated --prompt-seed 90000000 --check-every 1 --fixed-inputs $O/fixed_inputs_swap.pt"
for arm in S00 S01 S10 S11; do c=${arm:1:1}; m=${arm:2:1}; RC=$(eval echo \$R$c); RM=$(eval echo \$R$m)
  $P -B $D/swap_arm.py --source-run $RC --snapshot $RC/$SN --model-source-run $RM --model-snapshot $RM/$SN --arm-name $arm $COMMON --out $S/$arm > $S/$arm.log 2>&1
  log "smoke $arm exit $?"
done
$P -B $D/check_swap.py $S > $O/smoke/check_swap_smoke.json || { log "SMOKE CHECKS FAILED"; exit 1; }
log "smoke checks pass"
$P -B $D/swap_arm.py --source-run $R0 --snapshot $R0/$SN --model-source-run $R1 --model-snapshot $R1/$SN --arm-name S01 $COMMON --out $O/smoke/resume_S01/S01 --stop-after 10 > $O/smoke/resume_S01/part1.log 2>&1
$P -B $D/swap_arm.py --source-run $R0 --snapshot $R0/$SN --model-source-run $R1 --model-snapshot $R1/$SN --arm-name S01 $COMMON --out $O/smoke/resume_S01/S01 > $O/smoke/resume_S01/part2.log 2>&1
log "resume S01 identical: $($P -c "import json; a=json.load(open('$O/smoke/resume_S01/S01/final.json')); b=json.load(open('$S/S01/final.json')); print(a['actor_critic_hash']==b['actor_critic_hash'] and a['pmpo_optimizers_hash']==b['pmpo_optimizers_hash'] and open('$O/smoke/resume_S01/S01/updates.jsonl').read()==open('$S/S01/updates.jsonl').read())")"
log "prep complete"
