#!/usr/bin/env bash
set -euo pipefail

# Run inside tmux. Replace GPU=0 with GPU=1 to use the second GPU.
#
# Full ensemble:
#   GPU=0 PRESET=ensemble_100k LABEL=full-ensemble bash scripts/run_100k_pilot.sh
# LCG:
#   GPU=0 PRESET=lcg_100k LABEL=lcg bash scripts/run_100k_pilot.sh
# Laplace ensemble:
#   GPU=0 PRESET=laplace_100k LABEL=laplace-ensemble bash scripts/run_100k_pilot.sh
# Plan2Explore:
#   GPU=0 PRESET=plan2explore_100k LABEL=plan2explore bash scripts/run_100k_pilot.sh
# Uniform:
#   GPU=0 PRESET=uniform_100k LABEL=uniform bash scripts/run_100k_pilot.sh
#
# Phase 1: explore all six listed task groups.
# Phase 2: adapt only 100k milestones.
# Seeds default to 0; override with EXPLORATION_SEED and ADAPTATION_SEED.
# Resume interrupted exploration: bash scripts/resume.sh <run folder>

# Run from the repository root.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# -------------------------------------------------------------------
# Editable launch settings
# -------------------------------------------------------------------
EXPLORATION_SEED="${EXPLORATION_SEED:-0}"
ADAPTATION_SEED="${ADAPTATION_SEED:-0}"

GPU="${GPU:?Set GPU, e.g. 0 or 1}"
PYTHON="${PYTHON:-python}"

EXPLORATION_PROJECT="${EXPLORATION_PROJECT:-reward-free-exploration}"
# One downstream W&B project per task group and task, shared by methods and seeds:
# <DOWNSTREAM_PROJECT_PREFIX>-<group>-<task>.
DOWNSTREAM_PROJECT_PREFIX="${DOWNSTREAM_PROJECT_PREFIX:-reward-free-downstream}"

PRESET="${PRESET:?Set PRESET, e.g. lcg_100k}"
LABEL="${LABEL:?Set LABEL, e.g. lcg}" # output label of the method (data/milestone.py OUTPUT_LABELS)

OUTPUT_ROOT="$REPO_ROOT/outputs/$LABEL"
LOG_ROOT="$REPO_ROOT/logs/$LABEL"

# Computation and EGL rendering on the selected physical GPU.
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="$GPU"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Group label | actual dm_control domain | available tasks
#
# Separate Cartpole labels prevent the two groups sharing an output.
TASK_GROUPS=(
    'walker|walker|stand,walk,run'
    'hopper|hopper|stand,hop'
    'acrobot|acrobot|swingup,swingup_sparse'
    'cartpole-balance|cartpole|balance,balance_sparse'
    'cartpole-swingup|cartpole|swingup,swingup_sparse'
    'humanoid|humanoid|stand,walk,run,run_pure_state'
    'humanoid_CMU|humanoid_CMU|stand,walk,run'
)

# -------------------------------------------------------------------
# Exploration: one task group
# -------------------------------------------------------------------
run_exploration() {
    local group="$1"
    local domain="$2"
    local tasks="$3"

    local run_dir="$OUTPUT_ROOT/$group/seed_$EXPLORATION_SEED"
    local log_name="${group}--exploration-es${EXPLORATION_SEED}"
    local log_file="$ATTEMPT_DIR/${log_name}.log"

    printf '\n=== Exploration: %s ===\n' "$group"
    printf 'Output: %s\nLog: %s\n' "$run_dir" "$log_file"

    "$PYTHON" -u src/exploration_main.py \
        "+experiment=$PRESET" \
        "env.train.domain_name=$domain" \
        "env.train.task_names=[$tasks]" \
        "common.seed=$EXPLORATION_SEED" \
        "common.devices=$GPU" \
        wandb.mode=online \
        "wandb.project=$EXPLORATION_PROJECT-${LABEL}" \
        "wandb.group=${LABEL}-${log_name}" \
        "wandb.name=${LABEL}-${log_name}" \
        "hydra.run.dir=$run_dir" \
        > >(tee "$log_file") \
        2> >(tee "${log_file%.log}.stderr.log" >&2)
}

# -------------------------------------------------------------------
# Preflight: refuse existing exploration outputs before starting work.
# Resume interrupted runs separately with scripts/resume.sh.
# -------------------------------------------------------------------
for entry in "${TASK_GROUPS[@]}"; do
    IFS='|' read -r group domain tasks <<< "$entry"
    run_dir="$OUTPUT_ROOT/$group/seed_$EXPLORATION_SEED"

    if [[ -e "$run_dir" ]]; then
        printf 'Output already exists: %s\n' "$run_dir" >&2
        printf 'Use scripts/resume.sh for interrupted exploration.\n' >&2
        exit 1
    fi
done

mkdir -p "$LOG_ROOT"

# Unique directory for this invocation's terminal logs.
ATTEMPT_DIR="$(mktemp -d \
    "$LOG_ROOT/$(date +%Y%m%d-%H%M%S)-XXXXXX")"

# Combined launcher logs: every message and every phase's output, stdout and stderr kept apart (both
# still on the terminal); the per-phase logs below are kept as well.
LAUNCH_LOG="$ATTEMPT_DIR/launcher.stdout.log"
ERROR_LOG="$ATTEMPT_DIR/launcher.stderr.log"

exec > >(tee -a "$LAUNCH_LOG") \
     2> >(tee -a "$ERROR_LOG" >&2)

printf '%s experiment | GPU %s | exploration seed %s | adaptation seed %s\n' \
    "$LABEL" "$GPU" "$EXPLORATION_SEED" "$ADAPTATION_SEED"
printf 'Outputs: %s\nTerminal logs: %s\n' "$OUTPUT_ROOT" "$ATTEMPT_DIR"

# -------------------------------------------------------------------
# Phase 1: finish exploration for every task group.
# -------------------------------------------------------------------
for entry in "${TASK_GROUPS[@]}"; do
    IFS='|' read -r group domain tasks <<< "$entry"
    run_exploration "$group" "$domain" "$tasks"
done

printf '\nAll %s exploration and downstream runs completed.\n' "$LABEL"
printf 'Terminal logs: %s\n' "$ATTEMPT_DIR"
