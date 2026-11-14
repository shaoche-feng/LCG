#!/usr/bin/env bash
set -euo pipefail

# Run inside tmux. Replace GPU=0 with GPU=1 to use the second GPU.
#
# Full ensemble:
#   GPU=0 LABEL=full-ensemble bash scripts/run_100k_pilot_adapt.sh
# LCG:
#   GPU=0 LABEL=lcg bash scripts/run_100k_pilot_adapt.sh
# Laplace ensemble:
#   GPU=0 LABEL=laplace-ensemble bash scripts/run_100k_pilot_adapt.sh
# Plan2Explore:
#   GPU=0 LABEL=plan2explore bash scripts/run_100k_pilot_adapt.sh
# Uniform:
#   GPU=0 LABEL=uniform bash scripts/run_100k_pilot_adapt.sh
#
# Adapt each task at ten milestones, sequentially.
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

# One downstream W&B project per task group and task, shared by methods and seeds:
# <DOWNSTREAM_PROJECT_PREFIX>-<group>-<task>.
DOWNSTREAM_PROJECT_PREFIX="${DOWNSTREAM_PROJECT_PREFIX:-reward-free-downstream}"

LABEL="${LABEL:?Set LABEL, e.g. lcg}"   # output label of the method (data/milestone.py OUTPUT_LABELS)

OUTPUT_ROOT="$REPO_ROOT/outputs/$LABEL"
LOG_ROOT="$REPO_ROOT/logs/$LABEL"

# Computation and EGL rendering on the selected physical GPU.
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="$GPU"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Group label | actual dm_control domain | adapted tasks
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
# Downstream: one milestones folder and one task
# The Python launcher processes the ten selected milestones sequentially.
# -------------------------------------------------------------------
run_downstream() {
    local group="$1"
    local task="$2"

    local run_dir="$OUTPUT_ROOT/$group/seed_$EXPLORATION_SEED"
    local milestones_dir="$run_dir/milestones"
    local log_name="${group}--downstream-${task}-es${EXPLORATION_SEED}-ds${ADAPTATION_SEED}"
    local log_file="$ATTEMPT_DIR/${log_name}.log"

    if [[ ! -d "$milestones_dir" ]]; then
        printf 'Missing milestones folder: %s\n' "$milestones_dir" >&2
        return 1
    fi

    printf '\n=== Downstream: %s / %s ===\n' "$group" "$task"
    printf 'Milestones: %s\nLog: %s\n' "$milestones_dir" "$log_file"

    # One W&B run, named and grouped ${LABEL}-${log_name}, shared by the ten milestones; each milestone
    # is still adapted independently, in its own child process and output folder.
    "$PYTHON" -u src/downstream_main.py \
        +experiment=downstream_10_milestones \
        "initialization.milestones_dir=$milestones_dir" \
        "downstream.task_name=$task" \
        "downstream.seed=$ADAPTATION_SEED" \
        downstream.mode=train \
        "common.devices=$GPU" \
        wandb.mode=online \
        "wandb.project=${DOWNSTREAM_PROJECT_PREFIX}-${group}-${task}" \
        "wandb.group=${LABEL}-${log_name}" \
        "wandb.name=${LABEL}-${log_name}" \
        > >(tee "$log_file") \
        2> >(tee "${log_file%.log}.stderr.log" >&2)
}

# -------------------------------------------------------------------
# Preflight: the milestones must exist; refuse existing downstream outputs and an existing shared W&B
# run record of these tasks (a fresh adaptation, never an earlier run id) before starting work.
# -------------------------------------------------------------------
for entry in "${TASK_GROUPS[@]}"; do
    IFS='|' read -r group domain tasks <<< "$entry"
    IFS=',' read -r -a task_list <<< "$tasks"
    run_dir="$OUTPUT_ROOT/$group/seed_$EXPLORATION_SEED"

    if [[ ! -d "$run_dir/milestones" ]]; then
        printf 'Missing milestones folder: %s/milestones\n' "$run_dir" >&2
        exit 1
    fi
    for task in "${task_list[@]}"; do
        for existing in "$run_dir"/downstream/*/"$task"/seed_"$ADAPTATION_SEED" \
                        "$run_dir"/downstream/*/"$task"/.seed_"$ADAPTATION_SEED".tmp \
                        "$run_dir/downstream/wandb_${task}_seed_${ADAPTATION_SEED}.json"; do
            if [[ -e "$existing" ]]; then
                printf 'Output already exists: %s\n' "$existing" >&2
                exit 1
            fi
        done
    done
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

printf '%s adaptation | GPU %s | exploration seed %s | adaptation seed %s\n' \
    "$LABEL" "$GPU" "$EXPLORATION_SEED" "$ADAPTATION_SEED"
printf 'Outputs: %s\nTerminal logs: %s\n' "$OUTPUT_ROOT" "$ATTEMPT_DIR"

# -------------------------------------------------------------------
# Adapt every task, one task at a time.
# Each task adapts ten milestones before the next task starts.
# -------------------------------------------------------------------
for entry in "${TASK_GROUPS[@]}"; do
    IFS='|' read -r group domain tasks <<< "$entry"
    IFS=',' read -r -a task_list <<< "$tasks"

    for task in "${task_list[@]}"; do
        run_downstream "$group" "$task"
    done
done

printf '\nAll %s adaptations completed.\n' "$LABEL"
printf 'Terminal logs: %s\n' "$ATTEMPT_DIR"
