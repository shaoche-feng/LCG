#!/usr/bin/env bash
set -euo pipefail

# Downstream controller tuning: adapt one method's milestone (default: 100k) with a controller configuration
# from config/control_tuning/<TUNING>.yaml. Same structure as pilot_downstream.sh. Run inside tmux.
#
#   GPU=0 LABEL=plan2explore RUN_GROUPS=hopper TASKS=stand TUNING=advantage_weighted bash scripts/control_downstream.sh
#   GPU=1 LABEL=plan2explore RUN_GROUPS=hopper TASKS=stand TUNING=reward_div20 bash scripts/control_downstream.sh
#
# LABEL       method whose exploration run is adapted (outputs/<LABEL>/<group>/seed_<EXPLORATION_SEED>), e.g.
#             uniform | plan2explore | full-ensemble | laplace-ensemble | lcg | lcg-data-scaled | lcg-predictor |
#             lcg-predictor-div5000
# TUNING      a file name in config/control_tuning/ without .yaml (baseline, advantage_weighted, reward_div20, ...);
#             passed as +control_tuning=<TUNING>, so each run's saved config.yaml records its settings
# RUN_GROUPS  comma-separated task groups (below), e.g. hopper or hopper,acrobot (required)
# TASKS       optional comma-separated task filter within those groups (default: every task of the group)
# MILESTONE_STEPS  the milestone(s) to adapt, by collected steps (default 100000; e.g. "50000,100000")
# Seeds default to 0; override with EXPLORATION_SEED and ADAPTATION_SEED.
#
# Outputs: a tuning run folder per method, configuration and group, never the original run folder:
#   outputs/tuning/<LABEL>/<TUNING>/control_tuning.yaml           copy of the configuration (must stay the same)
#   outputs/tuning/<LABEL>/<TUNING>/<group>/seed_<N>/
#       config   -> outputs/<LABEL>/<group>/seed_<N>/config        (symlink: the run's exploration identity)
#       dataset  -> outputs/<LABEL>/<group>/seed_<N>/dataset       (symlink: the replay the milestones refer to)
#       milestones/steps_<S>/                                      (hard links of the selected milestones' files)
#       downstream/steps_<S>/<task>/seed_<M>/                      (the adaptation outputs)
# The milestones are hard-linked, not symlinked: downstream_main.py resolves milestone paths, and the trainer writes
# next to the (resolved) milestone, so a symlinked milestone would put the outputs into the original run. A
# milestone reaches its replay by the relative path ../../dataset/train, through the dataset symlink. Each
# configuration has its own folder, so different configurations can run in parallel (e.g. one per GPU).

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

LABEL="${LABEL:?Set LABEL, e.g. plan2explore}"   # output label of the method (data/milestone.py OUTPUT_LABELS)
TUNING="${TUNING:?Set TUNING to a file name in config/control_tuning (without .yaml)}"
RUN_GROUPS="${RUN_GROUPS:?Set RUN_GROUPS, e.g. hopper}"
TASKS="${TASKS:-}"
MILESTONE_STEPS="${MILESTONE_STEPS:-100000}"

# One W&B project per task group and task for the tuning runs, shared by methods, configurations and seeds:
# <TUNING_PROJECT_PREFIX>-<group>-<task>.
TUNING_PROJECT_PREFIX="${TUNING_PROJECT_PREFIX:-control-tuning}"

TUNING_FILE="$REPO_ROOT/config/control_tuning/$TUNING.yaml"
SOURCE_ROOT="$REPO_ROOT/outputs/$LABEL"                 # the exploration runs (read only)
OUTPUT_ROOT="$REPO_ROOT/outputs/tuning/$LABEL/$TUNING"  # this configuration's tuning run folders
LOG_ROOT="$REPO_ROOT/logs/control_tuning/$LABEL"

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

contains() {  # contains <comma-separated list> <item>
    [[ ",$1," == *",$2,"* ]]
}

milestone_name() {  # milestone_name <collected steps>  (data/milestone.py milestone_name)
    printf 'steps_%06d' "$1"
}

# -------------------------------------------------------------------
# Selection: the chosen groups and their (filtered) tasks
# -------------------------------------------------------------------
if [[ ! -f "$TUNING_FILE" ]]; then
    printf 'No tuning configuration %s; available:\n' "$TUNING_FILE" >&2
    ls "$REPO_ROOT/config/control_tuning" >&2
    exit 1
fi

SELECTED=()      # "group|tasks" entries
KNOWN_GROUPS=""
for entry in "${TASK_GROUPS[@]}"; do
    IFS='|' read -r group domain tasks <<< "$entry"
    KNOWN_GROUPS="$KNOWN_GROUPS,$group"
    contains "$RUN_GROUPS" "$group" || continue
    IFS=',' read -r -a task_list <<< "$tasks"
    chosen=()
    for task in "${task_list[@]}"; do
        if [[ -z "$TASKS" ]] || contains "$TASKS" "$task"; then
            chosen+=("$task")
        fi
    done
    if (( ${#chosen[@]} == 0 )); then
        printf 'Group %s has none of the tasks %s (its tasks: %s)\n' "$group" "$TASKS" "$tasks" >&2
        exit 1
    fi
    SELECTED+=("$group|$(IFS=','; echo "${chosen[*]}")")
done
IFS=',' read -r -a requested <<< "$RUN_GROUPS"
for group in "${requested[@]}"; do
    if ! contains "$KNOWN_GROUPS" "$group"; then
        printf 'Unknown task group %s (known: %s)\n' "$group" "${KNOWN_GROUPS#,}" >&2
        exit 1
    fi
done
IFS=',' read -r -a step_list <<< "$MILESTONE_STEPS"

# -------------------------------------------------------------------
# Downstream: one tuning run folder and one task
# The Python launcher adapts the selected milestone(s) with the tuning configuration.
# -------------------------------------------------------------------
run_downstream() {
    local group="$1"
    local task="$2"

    local run_dir="$OUTPUT_ROOT/$group/seed_$EXPLORATION_SEED"
    local milestones_dir="$run_dir/milestones"
    local log_name="${group}-${task}-es${EXPLORATION_SEED}-ds${ADAPTATION_SEED}"
    local log_file="$ATTEMPT_DIR/${log_name}.log"

    printf '\n=== Downstream (%s): %s / %s ===\n' "$TUNING" "$group" "$task"
    printf 'Milestones: %s (steps %s)\nLog: %s\n' "$milestones_dir" "$MILESTONE_STEPS" "$log_file"

    # One W&B run, named and grouped ${LABEL}-${TUNING}-${log_name}, shared by the selected milestones.
    "$PYTHON" -u src/downstream_main.py \
        "+control_tuning=$TUNING" \
        "initialization.milestones_dir=$milestones_dir" \
        "initialization.milestone_steps=[$MILESTONE_STEPS]" \
        "downstream.task_name=$task" \
        "downstream.seed=$ADAPTATION_SEED" \
        downstream.mode=train \
        "common.devices=$GPU" \
        wandb.mode=online \
        "wandb.project=${TUNING_PROJECT_PREFIX}-${group}-${task}" \
        "wandb.group=${LABEL}-${TUNING}-${log_name}" \
        "wandb.name=${LABEL}-${TUNING}-${log_name}" \
        > >(tee "$log_file") \
        2> >(tee "${log_file%.log}.stderr.log" >&2)
}

# -------------------------------------------------------------------
# Preflight, before anything is created: the source runs and milestones exist; the configuration matches the copy
# recorded by an earlier invocation; an existing tuning run folder links to the same source run; no output of the
# selected tasks exists yet (as pilot_downstream.sh).
# -------------------------------------------------------------------
if [[ -e "$OUTPUT_ROOT/control_tuning.yaml" ]] && ! cmp -s "$TUNING_FILE" "$OUTPUT_ROOT/control_tuning.yaml"; then
    printf '%s differs from the configuration recorded for this folder (%s).\n' "$TUNING_FILE" \
        "$OUTPUT_ROOT/control_tuning.yaml" >&2
    printf 'A changed configuration needs a new name (a new file in config/control_tuning).\n' >&2
    exit 1
fi
for entry in "${SELECTED[@]}"; do
    IFS='|' read -r group tasks <<< "$entry"
    IFS=',' read -r -a task_list <<< "$tasks"
    source_dir="$SOURCE_ROOT/$group/seed_$EXPLORATION_SEED"
    run_dir="$OUTPUT_ROOT/$group/seed_$EXPLORATION_SEED"

    for item in milestones config/exploration.yaml dataset; do
        if [[ ! -e "$source_dir/$item" ]]; then
            printf 'Missing in the exploration run: %s/%s\n' "$source_dir" "$item" >&2
            exit 1
        fi
    done
    for steps in "${step_list[@]}"; do
        if [[ ! -f "$source_dir/milestones/$(milestone_name "$steps")/manifest.json" ]]; then
            printf 'Missing milestone: %s/milestones/%s\n' "$source_dir" "$(milestone_name "$steps")" >&2
            exit 1
        fi
    done
    for link in config dataset; do
        if [[ -e "$run_dir/$link" || -L "$run_dir/$link" ]] && \
                [[ "$(readlink -f "$run_dir/$link")" != "$(readlink -f "$source_dir/$link")" ]]; then
            printf '%s/%s does not link to %s/%s\n' "$run_dir" "$link" "$source_dir" "$link" >&2
            exit 1
        fi
    done
    for steps in "${step_list[@]}"; do
        name="$(milestone_name "$steps")"
        if [[ -e "$run_dir/milestones/$name" ]] && \
                ! cmp -s "$run_dir/milestones/$name/manifest.json" "$source_dir/milestones/$name/manifest.json"; then
            printf '%s/milestones/%s is not the exploration run'"'"'s milestone\n' "$run_dir" "$name" >&2
            exit 1
        fi
        for task in "${task_list[@]}"; do
            for existing in "$run_dir/downstream/$name/$task/seed_$ADAPTATION_SEED" \
                            "$run_dir/downstream/$name/$task/.seed_$ADAPTATION_SEED.tmp"; do
                if [[ -e "$existing" ]]; then
                    printf 'Output already exists: %s\n' "$existing" >&2
                    exit 1
                fi
            done
        done
    done
    for task in "${task_list[@]}"; do
        if [[ -e "$run_dir/downstream/wandb_${task}_seed_${ADAPTATION_SEED}.json" ]]; then
            printf 'Output already exists: %s\n' "$run_dir/downstream/wandb_${task}_seed_${ADAPTATION_SEED}.json" >&2
            exit 1
        fi
    done
done

# -------------------------------------------------------------------
# Tuning run folders: links to the exploration run, hard-linked milestones (no copies of the data).
# -------------------------------------------------------------------
mkdir -p "$OUTPUT_ROOT"
[[ -e "$OUTPUT_ROOT/control_tuning.yaml" ]] || cp -- "$TUNING_FILE" "$OUTPUT_ROOT/control_tuning.yaml"
for entry in "${SELECTED[@]}"; do
    IFS='|' read -r group tasks <<< "$entry"
    source_dir="$SOURCE_ROOT/$group/seed_$EXPLORATION_SEED"
    run_dir="$OUTPUT_ROOT/$group/seed_$EXPLORATION_SEED"
    mkdir -p "$run_dir/milestones"
    for link in config dataset; do
        [[ -L "$run_dir/$link" ]] || ln -s -- "$(readlink -f "$source_dir/$link")" "$run_dir/$link"
    done
    for steps in "${step_list[@]}"; do
        name="$(milestone_name "$steps")"
        [[ -e "$run_dir/milestones/$name" ]] || cp -al -- "$source_dir/milestones/$name" "$run_dir/milestones/$name"
    done
done

mkdir -p "$LOG_ROOT"

# Unique directory for this invocation's terminal logs.
ATTEMPT_DIR="$(mktemp -d \
    "$LOG_ROOT/$(date +%Y%m%d-%H%M%S)-${TUNING}-XXXXXX")"
cp -- "$TUNING_FILE" "$ATTEMPT_DIR/control_tuning.yaml"

# Combined launcher logs: every message and every phase's output, stdout and stderr kept apart (both
# still on the terminal); the per-phase logs below are kept as well.
LAUNCH_LOG="$ATTEMPT_DIR/launcher.stdout.log"
ERROR_LOG="$ATTEMPT_DIR/launcher.stderr.log"

exec > >(tee -a "$LAUNCH_LOG") \
     2> >(tee -a "$ERROR_LOG" >&2)

printf '%s control tuning %s | GPU %s | exploration seed %s | adaptation seed %s | milestone steps %s\n' \
    "$LABEL" "$TUNING" "$GPU" "$EXPLORATION_SEED" "$ADAPTATION_SEED" "$MILESTONE_STEPS"
printf 'Groups and tasks: %s\n' "${SELECTED[*]}"
printf 'Tuning configuration (%s):\n' "$TUNING_FILE"
grep -v '^\s*#' "$TUNING_FILE" | sed '/^\s*$/d; s/^/  /'
printf 'Exploration runs: %s\nOutputs: %s\nTerminal logs: %s\n' "$SOURCE_ROOT" "$OUTPUT_ROOT" "$ATTEMPT_DIR"

# -------------------------------------------------------------------
# Adapt every selected task, one task at a time.
# -------------------------------------------------------------------
for entry in "${SELECTED[@]}"; do
    IFS='|' read -r group tasks <<< "$entry"
    IFS=',' read -r -a task_list <<< "$tasks"

    for task in "${task_list[@]}"; do
        run_downstream "$group" "$task"
    done
done

printf '\nAll %s control-tuning (%s) adaptations completed.\n' "$LABEL" "$TUNING"
printf 'Outputs: %s\nTerminal logs: %s\n' "$OUTPUT_ROOT" "$ATTEMPT_DIR"
