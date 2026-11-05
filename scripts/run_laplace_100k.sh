#!/usr/bin/env bash
set -euo pipefail

# Laplace-ensemble, exploration seed 0 and adaptation seed 0, on one GPU, in the foreground (run it inside tmux):
#   GPU=1 PYTHON=/path/to/python bash scripts/run_laplace_100k.sh
# Phase 1 explores all seven task groups (+experiment=laplace_100k); phase 2 then adapts every task of every
# group on the ten selected milestones (+experiment=downstream_10_milestones), one task at a time.
# Training and evaluation budgets come from config/exploration.yaml and config/downstream.yaml.
# Resume an interrupted exploration run with scripts/resume.sh <run folder>.

# Run from the repository root.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# -------------------------------------------------------------------
# Editable launch settings
# -------------------------------------------------------------------
EXPLORATION_SEED=0
ADAPTATION_SEED=0

GPU="${GPU:-0}"
PYTHON="${PYTHON:-python}"

EXPLORATION_PROJECT="${EXPLORATION_PROJECT:-reward-free-exploration}"
# One downstream W&B project per task group and task, shared by methods and seeds:
# <DOWNSTREAM_PROJECT_PREFIX>-<group>-<task>.
DOWNSTREAM_PROJECT_PREFIX="${DOWNSTREAM_PROJECT_PREFIX:-reward-free-downstream}"

PRESET=laplace_100k
LABEL=laplace-ensemble  # output label of the method (data/milestone.py OUTPUT_LABELS)

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

printf 'Laplace-ensemble experiment | GPU %s | exploration seed %s | adaptation seed %s\n' \
    "$GPU" "$EXPLORATION_SEED" "$ADAPTATION_SEED"
printf 'Outputs: %s\nTerminal logs: %s\n' "$OUTPUT_ROOT" "$ATTEMPT_DIR"

# -------------------------------------------------------------------
# Phase 1: finish exploration for every task group.
# -------------------------------------------------------------------
for entry in "${TASK_GROUPS[@]}"; do
    IFS='|' read -r group domain tasks <<< "$entry"
    run_exploration "$group" "$domain" "$tasks"
done

# -------------------------------------------------------------------
# Phase 2: adapt every task, one task at a time.
# Each task adapts ten milestones before the next task starts.
# -------------------------------------------------------------------
for entry in "${TASK_GROUPS[@]}"; do
    IFS='|' read -r group domain tasks <<< "$entry"
    IFS=',' read -r -a task_list <<< "$tasks"

    for task in "${task_list[@]}"; do
        run_downstream "$group" "$task"
    done
done

printf '\nAll Laplace-ensemble exploration and downstream runs completed.\n'
printf 'Terminal logs: %s\n' "$ATTEMPT_DIR"
