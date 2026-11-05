#!/usr/bin/env bash
set -euo pipefail

# Storage/integration smoke using the production Plan2Explore launch structure.
#   GPU=1 bash scripts/smoke.sh
# Only TASK_GROUPS and training/evaluation command-line settings differ.
# Production output paths, logs, W&B identity, preflight and phase loops are retained.
# WARNING: these smoke runs occupy outputs/plan2explore/{walker,hopper}/seed_0.
# 100k steps, 20 saved milestones and 10 downstream milestones come from production YAML.
# Each training phase uses one update; evaluation uses one episode (seed 3001).
# Plan2Explore predictor fitting also uses one update per predictor per refresh.
# No YAML files are edited. This run is not a learning-quality experiment.

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
# One downstream W&B project per task group and task, shared by methods and seeds:
# <DOWNSTREAM_PROJECT_PREFIX>-<group>-<task>.
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
#
# Separate Cartpole labels prevent the two groups sharing an output.
TASK_GROUPS=(
    'walker|walker|walk'
    'hopper|hopper|stand'
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
        denoiser.training.steps_first_epoch=1 \
        denoiser.training.steps_per_epoch=1 \
        denoiser.training.lr_warmup_steps=0 \
        actor_critic.training.steps_first_epoch=1 \
        actor_critic.training.steps_per_epoch=1 \
        actor_critic.training.lr_warmup_steps=0 \
        intrinsic_reward.fit_updates=1 \
        collection.test.num_episodes=1 \
        'evaluation.reset_seeds=[3001]' \
        collection.test.num_final_episodes=1 \
        'evaluation.final_reset_seeds=[3001]' \
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
        rew_end_model.training.steps=1 \
        rew_end_model.training.lr_warmup_steps=0 \
        actor_critic.training.steps=1 \
        actor_critic.training.lr_warmup_steps=0 \
        'evaluation.seeds=[3001]' \
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

printf 'Plan2Explore experiment | GPU %s | exploration seed %s | adaptation seed %s\n' \
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

printf '\nAll Plan2Explore exploration and downstream runs completed.\n'
printf 'Terminal logs: %s\n' "$ATTEMPT_DIR"
