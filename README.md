# Reward-free exploration with DIAMOND world models on DM Control

This project explores DM Control tasks without task reward, using a DIAMOND-derived diffusion world model trained on RGB observations and a PMPO controller trained in imagination on an intrinsic reward (LCG, a full world-model ensemble, Laplace-sampled ensemble, Plan2Explore-style predictors, or uniform random as the baseline). Task rewards are recorded in the replay only for later adaptation: from saved exploration milestones, a fresh reward predictor and sign-only PMPO controller are trained on the frozen denoiser and replay, then evaluated on the real task.

## Configuration

- [config/exploration.yaml](config/exploration.yaml) (entry point `src/exploration_main.py`): the shared exploration settings, i.e. the selected 100k schedule (5,000 initial uniform steps, then 190 epochs of 500 steps; denoiser 10,000 then 400 updates; 200 PMPO updates per epoch; batch 32; no training-only epochs), the 20 saved milestones, physical-state recording, test diagnostics (every 10 epochs: 4 episodes, reset seeds 3001–3004; final: 30 episodes, reset seeds 3001–3030), checkpoints and the default run folder.
- [config/experiment/](config/experiment/): reusable method presets `uniform_100k`, `lcg_100k`, `ensemble_100k`, `laplace_100k` and `plan2explore_100k` (method, controller objective, world-model compilation), and `downstream_10_milestones` (downstream folder mode over the ten selected milestones).
- [config/intrinsic_reward/](config/intrinsic_reward/): each method's scoring settings (`lcg.yaml` is the default group, with the selected R1 selector and reference reward scaling).
- [config/downstream.yaml](config/downstream.yaml) (entry points `src/downstream_main.py` and `src/downstream_analysis.py`): downstream adaptation (5,000 reward-predictor and 10,000 policy updates), evaluation (seeds 3001–3030, at most 500 steps), W&B, result collection and comparison.
- [config/agent/pmpo_beta.yaml](config/agent/pmpo_beta.yaml): the Gaussian H10 EMA-critic PMPO controller; the shared world-model architecture is in [default.yaml](config/agent/default.yaml).
- [config/env/dm_control.yaml](config/env/dm_control.yaml): RGB size, camera, action repeat and time limit. Its Walker `[stand, walk, run]` task group is only a default: launch scripts set `env.train.domain_name` and `env.train.task_names` (accepted groups: `src/envs/dm_control_env.py`, `REWARD_LABEL_TASK_GROUPS`).
- Launch scripts choose the domain and task group, `common.seed`, `common.devices`, the run folder (`hydra.run.dir`) and the W&B identity (`wandb.mode`, `project`, `name`, `group`). The presets set none of these; W&B is disabled unless a launcher enables it.
- Source: `src/exploration_main.py`/`exploration_trainer.py` (exploration), `src/downstream_main.py`/`downstream_trainer.py` (downstream), `src/downstream_analysis.py` (results), `src/intrinsic_reward/` (LCG in `lcg/`, full ensemble, Plan2Explore, Laplace and the real-episode intrinsic evaluation), `src/models/` (denoiser, reward predictor, PMPO), `src/data/` (replay, sampling, milestones), `src/pmpo_diagnostic.py` (opt-in PMPO controller diagnostics, configured inline as `+pmpo_diagnostic.*`) and `src/utils.py` (shared helpers, including compute/rendering device reporting).

Action dimensions and bounds come from the task's action specification. RGB pixels are normalized to [-1, 1]. Each agent action executes `action_repeat` physics steps, summing their raw rewards. True termination suppresses bootstrapping; time-limit truncation uses the pre-reset final observation. Real collection continues across epoch boundaries.

## Setup

Use the established `lcg` conda environment (Python 3.10). On kingplus0 its interpreter is `/home/bigking/anaconda3/envs/lcg/bin/python`. Dependency versions are pinned in [requirements.txt](requirements.txt).

- **PyTorch:** install `torch==2.7.1` from the PyTorch wheel index matching the CUDA driver before the other requirements. For the RTX 5090 (Blackwell, driver 570 with CUDA 12.8) this is `pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128`; builds for CUDA older than 12.8 lack the RTX 5090's (sm_120) kernels.
- **Headless rendering (Linux servers):** MuJoCo renders through EGL. The NVIDIA driver's EGL library (`libEGL_nvidia`, with `libEGL.so.1` from libglvnd) must be installed. Set `MUJOCO_GL=egl` and `MUJOCO_EGL_DEVICE_ID=<GPU>` for the GPU used for compute (the resume script sets both). Windows and macOS use MuJoCo's platform default.
- **Analysis plots:** `python src/downstream_analysis.py downstream.mode=compare` needs `matplotlib` (pinned); training does not import it.
- **Tests** need `pytest` (not pinned in `requirements.txt`).

## Running and validating

### Launch scripts

Each method has one launcher: `scripts/run_lcg_100k.sh`, `run_uniform_100k.sh`, `run_ensemble_100k.sh`, `run_laplace_100k.sh` and `run_plan2explore_100k.sh`. Run one in the foreground inside tmux:

```bash
GPU=1 PYTHON=/home/bigking/anaconda3/envs/lcg/bin/python bash scripts/run_lcg_100k.sh
```

- **Order:** a launcher first explores all seven task groups (walker, hopper, acrobot, cartpole-balance, cartpole-swingup, humanoid, humanoid_CMU) with its `<method>_100k` preset. It then adapts every task of every group, one task at a time, on the ten selected milestones (`downstream_10_milestones`).
- **Seeds and budgets:** exploration seed 0 and adaptation seed 0. Training and evaluation budgets come from the YAML.
- **Settings at the top of each script:** the GPU, the Python interpreter, the seeds and the W&B projects (`EXPLORATION_PROJECT`, `DOWNSTREAM_PROJECT`).
- **GPU and output:** the script binds CUDA and EGL rendering to `GPU` and logs to W&B online.
- **Exploration outputs:** `outputs/<method>/<group>/seed_0/`, where `<method>` is `lcg`, `uniform`, `full-ensemble`, `laplace-ensemble` or `plan2explore`. The two Cartpole groups get separate folders (`cartpole-balance`, `cartpole-swingup`), but both run the `cartpole` domain.
- **Downstream outputs:** inside each run folder, under `downstream/`.
- **Terminal logs:** one per run, in a new folder `logs/<method>/<date>-<id>/`.
- **Existing output:** a launcher refuses to start if any of its seven run folders exists. Resume an interrupted exploration with `scripts/resume.sh <run folder>`.
- **Failures:** any failure stops the script with a nonzero exit status.
- **W&B identities:**
  - exploration run and group: `<method>-<group>--exploration-es0`;
  - downstream runs: `<method>-<group>--downstream-es0-steps_NNNNNN-<task>-adaptation-seed0`, one per milestone (the launcher sets the prefix `<method>-<group>--downstream-es0`; the downstream code appends the rest), grouped by `<method>-<group>--downstream-<task>-es0-ds0`.
- **Terminal log names:** `<group>--exploration-es0.log` and `<group>--downstream-<task>-es0-ds0.log`. `es`/`ds` are the exploration and adaptation seeds; `--` separates the group label, which may itself contain `-`.

`scripts/smoke.sh` is a storage and integration smoke built on `run_plan2explore_100k.sh`.
- **Scope:** it keeps that launcher's structure, paths, logs and W&B identities. Only its task groups and inline training/evaluation overrides differ.
- **Groups:** Walker `[walk]`, then Hopper `[stand]`, each with the full 100k training steps.
- **Training:**
  - one update per epoch for the denoiser, the policy and the Plan2Explore predictors, with no warmup;
  - one downstream reward-model and one policy update.
- **Milestones:** the production 20 are saved; downstream adapts the selected ten.
- **Evaluation:** one episode, reset seed 3001.
- **YAML:** unchanged.
- **Outputs:** it writes `outputs/plan2explore/<group>/seed_0` and `logs/plan2explore/`, exactly like the production launcher. Run it only in a separate checkout; in the production checkout those folders would make the real launcher refuse to start.
- **W&B:** separate projects, set on the command line:

```bash
GPU=1 PYTHON=/home/bigking/anaconda3/envs/lcg/bin/python \
    EXPLORATION_PROJECT=lcg-reward-free-exploration-smoke DOWNSTREAM_PROJECT=lcg-reward-free-downstream-smoke \
    bash scripts/smoke.sh
```

Exploration (one method, one domain/task group, one seed):

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 OMP_NUM_THREADS=1 /home/bigking/anaconda3/envs/lcg/bin/python src/exploration_main.py     +experiment=lcg_100k env.train.domain_name=walker "env.train.task_names=[stand,walk,run]"     common.seed=0 common.devices=1 wandb.mode=online wandb.project=<project> wandb.name=<name>
```

Two groups of one domain (Cartpole's `[balance, balance_sparse]` and `[swingup, swingup_sparse]`) would share the default run folder `outputs/<method>/cartpole/seed_<N>`; the launch scripts therefore give each group its own `hydra.run.dir`.

Downstream adaptation of the ten selected milestones, for one task and adaptation seed:

```bash
python src/downstream_main.py +experiment=downstream_10_milestones     initialization.milestones_dir=<run>/milestones downstream.task_name=walk downstream.seed=0
```

PMPO supports one device, `grad_acc_steps=1`, and zero data-loader workers for checkpointed imagination RNG behavior.

For tests without training:

```bash
OMP_NUM_THREADS=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 /home/bigking/anaconda3/envs/lcg/bin/python -m pytest -q tests
```

Runs save the effective Hydra configuration (`config/exploration.yaml`), source/scripts, datasets, agent versions, milestones and full resume state under their output directory. By default an exploration run's folder is `outputs/<method>/<domain>/seed_<exploration seed>/` (method: `uniform`, `lcg`, `full-ensemble`, `laplace-ensemble` or `plan2explore`; a fixed `common.seed` is required), and a new run refuses a folder that already holds run outputs; resume it instead. Downstream outputs go to `<run>/downstream/steps_<N>/<task>/seed_<s>/`; `python src/downstream_analysis.py downstream.mode=collect` writes `<run>/result/<task>/seed_<s>/result.csv` and `downstream.mode=compare` writes `outputs/comparisons/<domain>/exploration_seed_<N>/<task>/adaptation_seed_<s>/` (CSV, analysis.json and plots; matplotlib needed). Custom output roots are pending. `static_dataset.path` retains training from an existing dataset. `intrinsic_reward.enabled` switches the intrinsic scorer; the uniform preset disables it.

To resume a run, pass its folder to the resume script, from the repository or anywhere else:

```bash
PYTHON=/home/bigking/anaconda3/envs/lcg/bin/python bash scripts/resume.sh outputs/<run> [overrides...]
```

The script resumes with the run folder's own copied `src/`, saved `config/exploration.yaml` (`config/trainer.yaml` in run folders created before 2026-10-30) and `checkpoints/state.pt`, never this checkout's source or defaults. The folder defaults to the current directory. Saved settings are kept, and Hydra overrides after the folder apply on top. Two kinds of argument are rejected:

- An override in any form (`key=`, `+key=`, `++key=`, `~key`) that changes or deletes a key the script sets. That covers `common.devices`, `common.resume`, `hydra.run.dir`, `hydra.job.chdir` and `hydra.output_subdir`, and parent mappings such as `common` or `hydra.job`.
- A Hydra option such as `--config-name` or `--multirun`. The script refuses to start, with a nonzero exit, when any of these is missing. Diagnostic snapshots under `pmpo_diagnostic/` are not resume inputs. It prints the resolved interpreter, run folder, device and rendering settings before launching, and returns the training process's exit status.

- `PYTHON`: interpreter (default `python`; paths with spaces are fine).
- `DEVICE`: GPU index or `cpu` (default: the saved `common.devices`). `src/exploration_main.py` exports it as `CUDA_VISIBLE_DEVICES`, so it is a physical GPU index; a conflicting pre-set `CUDA_VISIBLE_DEVICES` is an error.
- `MUJOCO_GL` / `MUJOCO_EGL_DEVICE_ID`: on Linux they default to `egl` and the same GPU index; a conflicting EGL device is an error. On Windows and macOS, MuJoCo's platform default is used unless `MUJOCO_GL` is set.
- `OMP_NUM_THREADS`: defaults to 1, as in the launch scripts.

A completed run has a `.run_is_over` sentinel, written whenever training returns normally, including at a `pmpo_diagnostic.max_epochs` cap. The script refuses such a run. To extend it deliberately, rename the sentinel (for example `mv .run_is_over .run_is_over.<date>`) and pass the override that raises the bound, such as `pmpo_diagnostic.max_epochs=300` or a higher `collection.train.num_steps_total` (whole collection epochs; see `scripts/resume.sh --help`). `bash scripts/resume.sh --help` lists these details. Run folders created before this script changed keep their own older copy in `<run>/scripts/`; use this repository's script for them.

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
