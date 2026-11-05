#!/usr/bin/env bash
set -euo pipefail

# Plan2Explore Walker pilot, exploration seed 0 and adaptation seed 0, on one GPU, in the foreground (run it
# inside tmux):
#   GPU=1 PYTHON=/path/to/python bash scripts/walker_pilot/run_plan2explore_100k_walker.sh
# Phase 1 explores Walker only (+experiment=plan2explore_100k, production budgets, all 20 milestones saved);
# phase 2 adapts stand, walk and run on the 100k milestone only (+experiment=downstream_10_milestones,
# production budgets), into the '-pilot' downstream projects.
# After all three adaptations succeed, the run's downstream/ folder is archived as downstream_pilot_<attempt>,
# freeing the standard path for scripts/walker_pilot_adapt/adapt_plan2explore_100k_walker.sh. A failure stops
# the launcher before archiving, so the recovery files stay in downstream/.
# Resume an interrupted exploration run with scripts/resume.sh <run folder>.

# Run from the repository root.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# -------------------------------------------------------------------
# Editable launch settings
# -------------------------------------------------------------------
EXPLORATION_SEED=0
ADAPTATION_SEED=0

GPU="${GPU:-0}"
PYTHON="${PYTHON:-python}"

EXPLORATION_PROJECT="${EXPLORATION_PROJECT:-reward-free-exploration}"
# One pilot downstream W&B project per task group and task, shared by methods and seeds:
# <DOWNSTREAM_PROJECT_PREFIX>-<group>-<task>-pilot.
DOWNSTREAM_PROJECT_PREFIX="${DOWNSTREAM_PROJECT_PREFIX:-reward-free-downstream}"

PRESET=plan2explore_100k
LABEL=plan2explore  # output label of the method (data/milestone.py OUTPUT_LABELS)

OUTPUT_ROOT="$REPO_ROOT/outputs/$LABEL"
LOG_ROOT="$REPO_ROOT/logs/$LABEL"

# Computation and EGL rendering on the selected physical GPU.
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="$GPU"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Group label | actual dm_control domain | available tasks
TASK_GROUPS=(
    'walker|walker|stand,walk,run'
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
        "wandb.project=$EXPLORATION_PROJECT" \
        "wandb.group=${LABEL}-${log_name}" \
        "wandb.name=${LABEL}-${log_name}" \
        "hydra.run.dir=$run_dir" \
        > >(tee "$log_file") \
        2> >(tee "${log_file%.log}.stderr.log" >&2)
}

# -------------------------------------------------------------------
# Downstream: one milestones folder and one task
# The Python launcher adapts the selected milestone (100k only).
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

    # One W&B run, named and grouped ${LABEL}-${log_name}, holding the 100k milestone's adaptation.
    "$PYTHON" -u src/downstream_main.py \
        +experiment=downstream_10_milestones \
        'initialization.milestone_steps=[100000]' \
        "initialization.milestones_dir=$milestones_dir" \
        "downstream.task_name=$task" \
        "downstream.seed=$ADAPTATION_SEED" \
        downstream.mode=train \
        "common.devices=$GPU" \
        wandb.mode=online \
        "wandb.project=${DOWNSTREAM_PROJECT_PREFIX}-${group}-${task}-pilot" \
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

printf 'Plan2Explore Walker pilot | GPU %s | exploration seed %s | adaptation seed %s\n' \
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
# Each task adapts the 100k milestone before the next task starts.
# -------------------------------------------------------------------
for entry in "${TASK_GROUPS[@]}"; do
    IFS='|' read -r group domain tasks <<< "$entry"
    IFS=',' read -r -a task_list <<< "$tasks"

    for task in "${task_list[@]}"; do
        run_downstream "$group" "$task"
    done
done

# Reached only after all three adaptations succeeded (set -e stops on any failure).
# Preserve pilot outputs and W&B records; free the standard downstream path.
WALKER_RUN="$OUTPUT_ROOT/walker/seed_$EXPLORATION_SEED"
PILOT_ARCHIVE="$WALKER_RUN/downstream_pilot_$(basename "$ATTEMPT_DIR")"

if [[ -e "$PILOT_ARCHIVE" ]]; then
    printf 'Pilot archive already exists: %s\n' "$PILOT_ARCHIVE" >&2
    exit 1
fi

mv -- "$WALKER_RUN/downstream" "$PILOT_ARCHIVE"
printf 'Pilot outputs archived: %s\n' "$PILOT_ARCHIVE"

printf '\nAll Plan2Explore Walker pilot exploration and downstream runs completed.\n'
printf 'Terminal logs: %s\n' "$ATTEMPT_DIR"
