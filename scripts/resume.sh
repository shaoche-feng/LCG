#!/usr/bin/env bash
# Resume a training run from its own saved state.
#
# Usage:
#   [PYTHON=/path/to/python] [DEVICE=N|cpu] bash scripts/resume.sh [RUN_DIR] [overrides...]
#
# RUN_DIR (default: the current directory) is a run folder created by src/main.py. It is
# resumed with that folder's own copied src/main.py, saved config/trainer.yaml and
# checkpoints/state.pt -- never with this checkout's source or the repository defaults.
# Saved settings are kept; Hydra overrides given after RUN_DIR are applied on top, e.g.
# pmpo_diagnostic.max_epochs=300. Overrides in any form (key=, +key=, ++key=, ~key) that
# change or delete common.devices, common.resume, hydra.run.dir, hydra.job.chdir or
# hydra.output_subdir, or a parent mapping of them, are rejected, as are Hydra options
# such as --config-name or --multirun. Diagnostic snapshots (pmpo_diagnostic/*.pt) are
# not supported resume inputs.
#
# Environment:
#   PYTHON           Interpreter (default: python on PATH). Paths with spaces are fine.
#   DEVICE           GPU index or "cpu" (default: the saved common.devices). src/main.py
#                    exports it as CUDA_VISIBLE_DEVICES, so it names a physical GPU; a
#                    different pre-set CUDA_VISIBLE_DEVICES is rejected, not overridden.
#   MUJOCO_GL        Linux: defaults to egl, and MUJOCO_EGL_DEVICE_ID to the GPU index (a
#                    different pre-set MUJOCO_EGL_DEVICE_ID is rejected). Windows/macOS:
#                    MuJoCo's platform default unless MUJOCO_GL is set.
#   OMP_NUM_THREADS  Defaults to 1, as in the launch scripts.
#
# A finished run has a .run_is_over sentinel (written whenever src/main.py returns
# normally, including at a pmpo_diagnostic.max_epochs cap) and is refused. To extend it
# deliberately, rename the sentinel (mv .run_is_over .run_is_over.<date>) and pass the
# override that raises the bound.
set -euo pipefail

die() { echo "resume.sh: error: $*" >&2; exit 1; }

if [ "${1-}" = -h ] || [ "${1-}" = --help ]; then
  sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'
  exit 0
fi

RUN_DIR=.
if [ $# -gt 0 ] && [[ "$1" != *=* && "$1" != [-+~]* ]]; then RUN_DIR=$1; shift; fi
# Only job overrides are accepted: Hydra options (--config-name, --multirun, ...) could
# select another config or execution mode. Overrides that set or delete a script-owned
# key -- including through a parent mapping such as common or hydra.job -- are rejected.
for arg in "$@"; do
  case $arg in -*) die "$arg: Hydra options are not accepted; the run's saved config is always used" ;; esac
  key=${arg#\~}; key=${key#++}; key=${key#+}; key=${key%%=*}; key=${key%%@*}; key=${key//\//.}
  for owned in common.devices common.resume hydra.run.dir hydra.job.chdir hydra.output_subdir; do
    if [ "$key" = "$owned" ] || [[ "$owned" == "$key".* || "$key" == "$owned".* ]]; then
      [ "$owned" = common.devices ] && die "$arg would change common.devices; set the device with DEVICE=..."
      die "$arg would change $owned, which this script sets"
    fi
  done
done
[ -e "$RUN_DIR" ] || die "run directory not found: $RUN_DIR"
[ -d "$RUN_DIR" ] || die "$RUN_DIR is not a directory. Pass the run folder containing checkpoints/state.pt; diagnostic snapshots are not supported resume inputs."
cd "$RUN_DIR"
RUN_DIR=$(pwd -W 2>/dev/null || pwd -P)
for f in src/main.py config/trainer.yaml checkpoints/state.pt; do
  [ -f "$f" ] || die "$RUN_DIR/$f is missing: not a run folder created by src/main.py"
done
[ ! -e .run_is_over ] || die "$RUN_DIR is marked finished (.run_is_over). To extend it deliberately, rename the sentinel and pass the override that raises the bound (see --help)."

PYTHON=${PYTHON:-python}
command -v "$PYTHON" >/dev/null 2>&1 || die "interpreter not found: $PYTHON (set PYTHON=/full/path/to/python)"
saved=$("$PYTHON" - <<'EOF' | tr -d '\r'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load("config/trainer.yaml")
for key in ("common.devices", "agent", "env", "static_dataset"):
    if OmegaConf.select(cfg, key, default="<missing>") == "<missing>":
        sys.exit(f"resume.sh: error: config/trainer.yaml has no {key}")
print(sys.executable)
print(OmegaConf.select(cfg, "common.devices"))
print(OmegaConf.select(cfg, "static_dataset.path"))
EOF
) || die "could not read config/trainer.yaml with $PYTHON"
{ read -r PYTHON_EXE; read -r SAVED_DEVICE; read -r STATIC_DATASET; } <<< "$saved"
if [ "$STATIC_DATASET" = None ]; then
  [ -f dataset/train/info.pt ] || die "$RUN_DIR/dataset/train/info.pt is missing; the replay dataset is required to resume"
fi

overrides=(common.resume=true hydra.run.dir=. hydra.job.chdir=true hydra.output_subdir=null)
if [ -n "${DEVICE-}" ]; then
  overrides+=("common.devices=$DEVICE")
  device_source="DEVICE"
else
  DEVICE=$SAVED_DEVICE
  device_source="saved common.devices"
fi
case $DEVICE in
  cpu) ;;
  ''|*[!0-9]*) die "device '$DEVICE' ($device_source) is not one GPU index or cpu; PMPO trains on one device (set DEVICE=N or DEVICE=cpu)" ;;
  *)
    if [ -n "${CUDA_VISIBLE_DEVICES+x}" ] && [ "$CUDA_VISIBLE_DEVICES" != "$DEVICE" ]; then
      die "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES conflicts with GPU $DEVICE: src/main.py would export CUDA_VISIBLE_DEVICES=$DEVICE (a physical index). Unset it, or set DEVICE to the physical GPU to use."
    fi ;;
esac

case $(uname -s) in
  Linux)
    export MUJOCO_GL=${MUJOCO_GL:-egl}
    if [ "$MUJOCO_GL" = egl ] && [ "$DEVICE" != cpu ]; then
      if [ -n "${MUJOCO_EGL_DEVICE_ID+x}" ] && [ "$MUJOCO_EGL_DEVICE_ID" != "$DEVICE" ]; then
        die "MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID conflicts with GPU $DEVICE; unset it or make them match"
      fi
      export MUJOCO_EGL_DEVICE_ID=$DEVICE
    fi ;;
  *) ;;  # Windows and macOS: MuJoCo's platform default unless MUJOCO_GL is set.
esac
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

echo "resume.sh: interpreter:   $PYTHON_EXE"
echo "resume.sh: run directory: $RUN_DIR"
echo "resume.sh: device:        $DEVICE ($device_source)"
echo "resume.sh: rendering:     MUJOCO_GL=${MUJOCO_GL-<platform default>} MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID-<unset>}"
echo "resume.sh: command:       $PYTHON src/main.py ${overrides[*]} $*"
exec "$PYTHON" src/main.py "${overrides[@]}" "$@"
