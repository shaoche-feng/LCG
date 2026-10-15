# DM Control + PMPO-Beta

This project trains a PMPO-Beta controller through a DIAMOND-derived diffusion world model on DM Control RGB observations. The DM Control configuration retains its original Cheetah Run default; the experiment launch scripts explicitly select Walker Walk. The only supported controller is feedforward PMPO, with four-frame stacks, sign-only advantage grouping, separate actor/value optimizers, and a frozen prior within each refresh block. The reward/end model retains its LSTM and predicts raw scalar rewards and termination probabilities.

## Environment and configuration

Use the established `lcg` conda environment. On kingplus0 its interpreter is `/home/bigking/anaconda3/envs/lcg/bin/python`. Dependency versions are in [requirements.txt](requirements.txt).

- [config/trainer.yaml](config/trainer.yaml): collection budgets, world-model training, evaluation, checkpoints, and optional LCG.
- [config/agent/pmpo_beta.yaml](config/agent/pmpo_beta.yaml): PMPO architecture and learning settings; includes the shared world-model architecture in [default.yaml](config/agent/default.yaml).
- [config/env/dm_control.yaml](config/env/dm_control.yaml): task, RGB size, camera, action repeat, and time limit.
- [config/pmpo_diagnostic/](config/pmpo_diagnostic/): optional bounded diagnostics and checkpoint recording.

Action dimensions and bounds come from the task's action specification. RGB pixels are normalized to [-1, 1]. Each agent action executes `action_repeat` physics steps, summing their raw rewards. True termination suppresses bootstrapping; time-limit truncation uses the pre-reset final observation. Real collection continues across epoch boundaries.

The base collection cadence remains 100 steps per epoch. The historical 100k Walker launch scripts explicitly use 500; use those scripts to reproduce the experiments instead of assuming the defaults encode that protocol. No experiment learning settings, budgets, or evaluation seeds were changed by the cleanup.

## Running and validating

For the existing baseline recipe:

```bash
PYTHON=/home/bigking/anaconda3/envs/lcg/bin/python bash scripts/run_walker_100k_pmpo_baseline.sh
```

The Stage 3, Stage 3B, and refresh=250 scripts remain in `scripts/`. They record launch provenance. Headless rendering needs both `MUJOCO_GL=egl` and `MUJOCO_EGL_DEVICE_ID` matching the selected GPU; the recipes set these. PMPO supports one device, `grad_acc_steps=1`, and zero data-loader workers for checkpointed imagination RNG behavior.

For tests without training:

```bash
OMP_NUM_THREADS=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 /home/bigking/anaconda3/envs/lcg/bin/python -m pytest -q tests
```

Runs save the effective Hydra configuration, source/scripts, datasets, agent versions, and full resume state under their output directory. `static_dataset.path` retains training from an existing dataset. `intrinsic_reward.enabled` controls the optional LCG lifecycle; it is disabled by default.

To resume a run, pass its folder to the resume script, from the repository or anywhere else:

```bash
PYTHON=/home/bigking/anaconda3/envs/lcg/bin/python bash scripts/resume.sh outputs/<run> [overrides...]
```

The script resumes with the run folder's own copied `src/`, saved `config/trainer.yaml` and `checkpoints/state.pt`, never this checkout's source or defaults. The folder defaults to the current directory. Saved settings are kept, and Hydra overrides after the folder apply on top. Two kinds of argument are rejected:

- An override in any form (`key=`, `+key=`, `++key=`, `~key`) that changes or deletes a key the script sets. That covers `common.devices`, `common.resume`, `hydra.run.dir`, `hydra.job.chdir` and `hydra.output_subdir`, and parent mappings such as `common` or `hydra.job`.
- A Hydra option such as `--config-name` or `--multirun`. The script refuses to start, with a nonzero exit, when any of these is missing. Diagnostic snapshots under `pmpo_diagnostic/` are not resume inputs. It prints the resolved interpreter, run folder, device and rendering settings before launching, and returns the training process's exit status.

- `PYTHON`: interpreter (default `python`; paths with spaces are fine).
- `DEVICE`: GPU index or `cpu` (default: the saved `common.devices`). `src/main.py` exports it as `CUDA_VISIBLE_DEVICES`, so it is a physical GPU index; a conflicting pre-set `CUDA_VISIBLE_DEVICES` is an error.
- `MUJOCO_GL` / `MUJOCO_EGL_DEVICE_ID`: on Linux they default to `egl` and the same GPU index; a conflicting EGL device is an error. On Windows and macOS, MuJoCo's platform default is used unless `MUJOCO_GL` is set.
- `OMP_NUM_THREADS`: defaults to 1, as in the launch scripts.

A completed run has a `.run_is_over` sentinel, written whenever training returns normally, including at a `pmpo_diagnostic.max_epochs` cap. The script refuses such a run. To extend it deliberately, rename the sentinel (for example `mv .run_is_over .run_is_over.<date>`) and pass the override that raises the bound, such as `pmpo_diagnostic.max_epochs=300`. `bash scripts/resume.sh --help` lists these details. Run folders created before this script changed keep their own older copy in `<run>/scripts/`; use this repository's script for them.

`python src/play.py` shows the PMPO policy in the world model and real environments; arrow keys switch environments and adjust imagination horizon. `-r` records policy episodes and `-d` browses datasets, displaying continuous action vectors. The viewer uses the same four-frame state and action sampling as collection.

## Compatibility and history

DM Control/PMPO state-dictionary parameter names and tensor shapes are unchanged, including nested encoder keys, action bounds, update count, private imagination RNG, world-model weights, optimizers, and schedulers. Existing PMPO diagnostic payloads keep their `models.pmpo_beta.PMPOBetaConfig` pickle path. The reward/end model's LSTM is unchanged.

`normalize_saved_config` in [utils.py](src/utils.py) migrates old saved PMPO configurations at Trainer/viewer entry points: it removes the unused policy `lstm_dim`, discrete/reward flags, old controller loss/optimizer sections, initial reward-class thresholds, and `training.model_free=false`. Only the known legacy schemas and valid values are removed; unknown targets/fields in discarded sections and invalid flags are rejected. To instantiate an old agent config directly, first call `normalize_saved_config(cfg.agent)` before Hydra `instantiate`. Historical YAML files are never rewritten. Dataset snapshots containing the obsolete reward-sign counter load after that counter is dropped; episode tensors and sampling metadata are preserved. Reward-sign count log entries are no longer emitted.

Atari, the original recurrent/Gaussian controller, reward-sign classification, and model-free training are unsupported. Their checkpoints/configurations cannot resume as PMPO controllers. Continuous world-model weights from older controllers can still be loaded selectively with `load_actor_critic=false`; a fresh PMPO controller is required. The internal policy prediction API now returns `(logits_act, val)` without a dummy recurrent state; external callers must update accordingly. Old run folders retain their copied source, and therefore their original code unless deliberately updated.

Historical experiment scripts, checkpoint diagnostics, and evidence in [docs/pmpo_beta/](docs/pmpo_beta/) remain available. [SYNC_MANIFEST_kingplus0.txt](SYNC_MANIFEST_kingplus0.txt) records the original source checkout and hashes; it is historical provenance, not a manifest of the cleaned files. The preservation HEAD for this cleanup is `36382563ee8534fb7f58012ccc7782fbe698dc0f`. Compare resolved overrides before drawing experimental conclusions. Use 30-seed deterministic evaluation (3001–3030) for comparisons; in-training eval4 is a noisy proxy.

Dense diagnostic checkpoints can be large. Preserve 10k real-step milestones, the true final checkpoint, and `fixed_real_stacks.pt`. Retain JSONL logs, Hydra configs, datasets, and `checkpoints/state.pt`.

The inherited upstream Atari result data in `results/data/DIAMOND.json` is historical evidence from DIAMOND, rather than results for this PMPO implementation. The original project and paper are available at [diamond-wm.github.io](https://diamond-wm.github.io) and [arXiv](https://arxiv.org/abs/2405.12399).

## Upstream citation

```text
@inproceedings{alonso2024diffusionworldmodelingvisual,
      title={Diffusion for World Modeling: Visual Details Matter in Atari},
      author={Eloi Alonso and Adam Jelley and Vincent Micheli and Anssi Kanervisto and Amos Storkey and Tim Pearce and François Fleuret},
      booktitle={Thirty-eighth Conference on Neural Information Processing Systems}}
      year={2024},
      url={https://arxiv.org/abs/2405.12399},
}
```
