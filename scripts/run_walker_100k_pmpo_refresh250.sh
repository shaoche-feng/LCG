#!/usr/bin/env bash
# Prior-refresh-interval ablation: identical to Stage 3 (entropy_coef=0.01,
# scripts/run_walker_100k_pmpo_entropy_stage3.sh) with exactly ONE experimental
# change -- agent.actor_critic.prior_refresh_interval=250 (was 500). Everything
# else is identical to Stage 3's launch: entropy_coef=0.01, beta_kl=0.3,
# alpha_pmpo=0.5, K=4, imagination_horizon=15, steps_per_epoch=500,
# num_steps_total=100000, seed=0, fresh initialization, same world-model/
# optimizer/logging config. Run on device 0 (free at launch time; the ongoing
# zero-shot adaptation run occupies device 1) so it does not interfere with
# that run.
#
# Motivation: the prior-refresh-interval diagnostic on Stage 3's own per-update
# logs (outputs/prior_refresh_diagnostic/) found that KL(current||prior) reaches
# ~80% of its within-block total by update 100 of 500 and ~95% by update 250,
# then stays essentially flat through update 500 -- consistent across early/mid/
# late training. This run tests directly whether halving the refresh interval to
# 250 (refreshing the prior right around where the policy has already
# effectively settled) changes learning-curve/final-performance outcomes at all,
# as a strict controlled ablation against Stage 3.
#
# Checkpoint policy (bookkeeping-only difference, NOT experimental): pmpo_diagnostic
# saves one untagged full checkpoint every prior_refresh_interval updates
# (unconditional, see src/pmpo_diagnostic.py check_update) -- at interval=250 this
# would double the untagged-checkpoint count vs Stage 3/3B (~402 vs ~201) on top of
# the existing per-epoch "_epoch" checkpoints, recreating a large dense checkpoint
# history. Per the agreed archival policy, non-milestone checkpoints are pruned
# periodically during monitoring (external housekeeping, same procedure used for
# the Stage 3 / Stage 3B post-hoc cleanup) rather than by changing training code.
# Trainer's own epoch-boundary resume checkpoint (checkpoints/state.pt) is
# untouched by this pruning and remains available for resumability.
set -euo pipefail

OUTPUT_DIR="outputs/k4-walker-100k-entropy-refresh250"
WANDB_PROJECT="lcg-controller-baselines"
WANDB_RUN_NAME="walker-pmpo-100k-eta001-refresh250-seed0"
WANDB_GROUP="prior-refresh-ablation-eta001"
DEVICE=0
# Override with PYTHON=/path/to/python if the target env's interpreter isn't on PATH
# as a bare `python` (it usually isn't outside an activated conda env).
PYTHON="${PYTHON:-python}"

mkdir -p "${OUTPUT_DIR}"

LAUNCH_CMD=(
  env MUJOCO_GL=egl "MUJOCO_EGL_DEVICE_ID=${DEVICE}" OMP_NUM_THREADS=1
  "${PYTHON}" -u src/main.py
  agent=pmpo_beta
  env=dm_control env.train.domain_name=walker env.train.task_name=walk
  "common.devices=${DEVICE}" common.seed=0
  training.compile_wm=false
  collection.train.steps_per_epoch=500
  collection.train.num_steps_total=100000
  rew_end_model.training.batch_size=8
  intrinsic_reward.enabled=false
  agent.actor_critic.prior_refresh_interval=250
  agent.actor_critic.entropy_coef=0.01
  +pmpo_diagnostic=bounded_full_loop
  pmpo_diagnostic.max_epochs=260
  "wandb.mode=online"
  "wandb.project=${WANDB_PROJECT}"
  "wandb.name=${WANDB_RUN_NAME}"
  "wandb.group=${WANDB_GROUP}"
  "hydra.run.dir=${OUTPUT_DIR}"
)

# --- Static provenance, written before launch; wandb run id/url appended after
# the process starts writing its own local run directory. ---
"${PYTHON}" - "$0" "${OUTPUT_DIR}" "${WANDB_PROJECT}" "${WANDB_RUN_NAME}" "${LAUNCH_CMD[@]}" <<'PYEOF'
import json, os, subprocess, sys, platform
from datetime import datetime, timezone

script_path, output_dir, wandb_project, wandb_run_name, *launch_cmd = sys.argv[1:]

def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception:
        return None

branch = sh("git rev-parse --abbrev-ref HEAD")
commit = sh("git rev-parse HEAD")
dirty = sh("git status --porcelain")

import torch
meta = dict(
    experiment_name="walker-pmpo-100k-refresh250-eta001",
    baseline_experiment="walker-pmpo-100k-entropy-stage3 (outputs/k4-walker-100k-entropy-stage3, NOT modified by this run)",
    single_change="agent.actor_critic.prior_refresh_interval: 500 -> 250 (all else identical to the Stage 3 launch)",
    git_branch=branch, git_commit=commit,
    working_tree_status="dirty" if dirty else "clean",
    launch_command=" ".join(launch_cmd),
    canonical_script=script_path,
    hostname=platform.node(),
    gpu_model=sh("nvidia-smi --query-gpu=name --format=csv,noheader -i 0"),
    cuda_device_assignment="common.devices=0 (MUJOCO_EGL_DEVICE_ID=0)",
    start_timestamp_utc=datetime.now(timezone.utc).isoformat(),
    python_version=platform.python_version(),
    torch_version=torch.__version__,
    cuda_version=torch.version.cuda,
    env_domain="walker", env_task="walk",
    seed=0,
    total_real_step_budget=100000,
    steps_per_epoch=500,
    prior_refresh_interval=250,
    beta_kl=0.3,
    alpha_pmpo=0.5,
    entropy_coef=0.01,
    pmpo_controller="PMPOBeta",
    intrinsic_reward_mode="disabled",
    output_dir=output_dir,
    updates_jsonl_path=f"{output_dir}/pmpo_diagnostic/updates.jsonl",
    checkpoints_jsonl_path=f"{output_dir}/pmpo_diagnostic/checkpoints.jsonl",
    checkpoint_archival_policy="pruned periodically to ~10k-real-step milestones + final checkpoint (external housekeeping, not a training-code change)",
    wandb_mode="online", wandb_project=wandb_project, wandb_run_name=wandb_run_name,
    wandb_run_id=None, wandb_run_url=None,
)
with open(f"{output_dir}/run_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"Wrote initial {output_dir}/run_metadata.json")
PYEOF

# --- Launch (backgrounded, nohup, never depends on this shell staying attached) ---
nohup "${LAUNCH_CMD[@]}" > "${OUTPUT_DIR}/stdout.log" 2>&1 &
TRAIN_PID=$!
disown
echo "LAUNCHED_PID=${TRAIN_PID}"
echo "${TRAIN_PID}" > "${OUTPUT_DIR}/train.pid"

# --- Poll briefly for wandb's own local run directory to learn the real run id/url,
# then fold it into run_metadata.json. Purely observational -- never blocks training. ---
for _ in $(seq 1 30); do
  RUN_LINK="${OUTPUT_DIR}/wandb/latest-run"
  if [ -L "${RUN_LINK}" ] || [ -d "${RUN_LINK}" ]; then
    RUN_DIR="$(readlink -f "${RUN_LINK}" 2>/dev/null || true)"
    RUN_ID="$(basename "${RUN_DIR}" | sed -E 's/^run-[0-9_]+-//')"
    if [ -n "${RUN_ID}" ]; then
      "${PYTHON}" - "${OUTPUT_DIR}" "${WANDB_PROJECT}" "${RUN_ID}" <<'PYEOF2'
import json, sys
output_dir, project, run_id = sys.argv[1:]
path = f"{output_dir}/run_metadata.json"
with open(path) as f:
    meta = json.load(f)
meta["wandb_run_id"] = run_id
meta["wandb_run_url"] = f"https://wandb.ai/models-national-tsing-hua-university7947/{project}/runs/{run_id}"
with open(path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Updated {path} with wandb_run_id={run_id}")
PYEOF2
      break
    fi
  fi
  sleep 2
done

echo "Launch complete. Training PID: ${TRAIN_PID}"
