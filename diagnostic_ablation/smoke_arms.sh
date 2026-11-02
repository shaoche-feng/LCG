#!/usr/bin/env bash
# Ablation smoke from the real epoch-190 snapshot: arms A, B, C for 30 updates (checkpoints 0, 10, 30), and a resume
# test of C (stop after 10, resume to 30) against the uninterrupted C. Outputs under smoke/ only.
set -u
P=/home/bigking/anaconda3/envs/lcg/bin/python
W=/home/bigking/scfeng/LCG-diagnostic-gaussian-h10-replayvalue-ablation
A=$W/outputs/ablation_20261027
SR=/home/bigking/scfeng/LCG-stage3-gaussian-h10-ema-replayvalue/outputs/k4-walker-100k-entropy-stage3-seed1-gaussian-h10-ema-replayvalue-20261026
SNAP=$SR/pmpo_diagnostic/agent_pmpo_update_080600_epoch.pt
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 OMP_NUM_THREADS=1
cd $W
S=$A/smoke/arms; rm -rf $S $A/smoke/resume_C; mkdir -p $S $A/smoke/resume_C
COMMON="--source-run $SR --snapshot $SNAP --updates 30 --checkpoints 0,10,30 --prompt-stream dedicated --check-every 1 --fixed-inputs $A/fixed_inputs.pt"
for arm in A B C; do
  $P -B diagnostic_ablation/run_arm.py $COMMON --arm $arm --out $S/$arm > $S/$arm.log 2>&1; echo "smoke arm $arm exit $?"
done
$P -B diagnostic_ablation/run_arm.py $COMMON --arm C --out $A/smoke/resume_C/C --stop-after 10 > $A/smoke/resume_C/part1.log 2>&1; echo "resume part1 exit $?"
$P -B diagnostic_ablation/run_arm.py $COMMON --arm C --out $A/smoke/resume_C/C > $A/smoke/resume_C/part2.log 2>&1; echo "resume part2 exit $?"
$P -B diagnostic_ablation/check_arms.py $S $A/smoke/resume_C/C $S/C > $A/smoke/check_arms_smoke.json; echo "check exit $?"
