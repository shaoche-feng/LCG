#!/usr/bin/env bash
# Canonical launch recipe for the frozen PMPO-Beta controller's 100k-real-step
# DIAMOND-derived Walker/Walk baseline. Run from the repository root.
#
# Deliberate, frozen protocol for this experiment family (LCG / ensemble / random
# controls should later reuse everything here except the intrinsic-reward setting):
#   - agent=pmpo_beta, prior_refresh_interval=500 (validated controller config)
#   - collection.train.steps_per_epoch=500 (an intentional continuous-control choice,
#     NOT a reproduction of original DIAMOND's steps_per_epoch=100 cadence)
#   - collection.train.num_steps_total=100000
#   - fresh initialization: no initialization.path_to_ckpt, no warm-start
#   - wandb.mode=online (relies on an ALREADY-AUTHENTICATED environment, e.g. via
#     `wandb login` / ~/.netrc -- this script never contains or requires an API key)
#
# pmpo_diagnostic.max_epochs is overridden (not edited in the committed YAML) to a
# safe margin above the resolved schedule (241 expected epochs at this cadence), so
# the run naturally reaches 100000 real steps instead of stopping at the file's own
# shorter default (meant for bounded diagnostics, not this baseline).
set -euo pipefail

OUTPUT_DIR="outputs/k4-walker-100k"
WANDB_PROJECT="lcg-controller-baselines"
WANDB_RUN_NAME="walker-pmpo-100k-seed0"
DEVICE=1

mkdir -p "${OUTPUT_DIR}"

LAUNCH_CMD=(
  env MUJOCO_GL=egl "MUJOCO_EGL_DEVICE_ID=${DEVICE}" OMP_NUM_THREADS=1
  python -u src/main.py
  agent=pmpo_beta
  env=dm_control env.train.domain_name=walker env.train.task_name=walk
  "common.devices=${DEVICE}" common.seed=0
  training.compile_wm=false
  collection.train.steps_per_epoch=500
  collection.train.num_steps_total=100000
  rew_end_model.training.batch_size=8
  intrinsic_reward.enabled=false
  agent.actor_critic.prior_refresh_interval=500
  +pmpo_diagnostic=bounded_full_loop
  pmpo_diagnostic.max_epochs=260
  "wandb.mode=online"
  "wandb.project=${WANDB_PROJECT}"
  "wandb.name=${WANDB_RUN_NAME}"
  "hydra.run.dir=${OUTPUT_DIR}"
)

# --- Static provenance, written before launch; wandb run id/url appended after
# the process starts writing its own local run directory (never blocks/depends on
# training itself -- if wandb is slow to initialize, this just polls briefly and
# records whatever is available). ---
python - "$0" "${OUTPUT_DIR}" "${WANDB_PROJECT}" "${WANDB_RUN_NAME}" "${LAUNCH_CMD[@]}" <<'PYEOF'
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
    experiment_name="walker-pmpo-100k-baseline",
    git_branch=branch, git_commit=commit,
    working_tree_status="dirty" if dirty else "clean",
    launch_command=" ".join(launch_cmd),
    canonical_script=script_path,
    hostname=platform.node(),
    gpu_model=sh("nvidia-smi --query-gpu=name --format=csv,noheader -i 1"),
    cuda_device_assignment="common.devices=1 (MUJOCO_EGL_DEVICE_ID=1)",
    start_timestamp_utc=datetime.now(timezone.utc).isoformat(),
    python_version=platform.python_version(),
    torch_version=torch.__version__,
    cuda_version=torch.version.cuda,
    env_domain="walker", env_task="walk",
    seed=0,
    total_real_step_budget=100000,
    steps_per_epoch=500,
    prior_refresh_interval=500,
    pmpo_controller="PMPOBeta",
    intrinsic_reward_mode="disabled",
    output_dir=output_dir,
    updates_jsonl_path=f"{output_dir}/pmpo_diagnostic/updates.jsonl",
    checkpoints_jsonl_path=f"{output_dir}/pmpo_diagnostic/checkpoints.jsonl",
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
# then fold it into run_metadata.json. Purely observational -- never blocks training,
# which is already running in the background regardless of this loop's outcome. ---
for _ in $(seq 1 30); do
  RUN_LINK="${OUTPUT_DIR}/wandb/latest-run"
  if [ -L "${RUN_LINK}" ] || [ -d "${RUN_LINK}" ]; then
    RUN_DIR="$(readlink -f "${RUN_LINK}" 2>/dev/null || true)"
    RUN_ID="$(basename "${RUN_DIR}" | sed -E 's/^run-[0-9_]+-//')"
    if [ -n "${RUN_ID}" ]; then
      python - "${OUTPUT_DIR}" "${WANDB_PROJECT}" "${RUN_ID}" <<'PYEOF2'
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
