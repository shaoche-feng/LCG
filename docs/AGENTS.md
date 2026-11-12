# AGENTS.md

Context for any AI coding agent (Claude Code, Codex, etc.) working in this repository. Keep this file in sync with
reality: if a fact here turns out stale, fix it rather than leaving it wrong.

Controller settings, results and findings are in [CONTROLLER.md](CONTROLLER.md). Read it before proposing or running a
controller experiment.

## Purpose

The goal of this repository is to find a **stable controller for a DIAMOND-style diffusion world model on DM Control**,
in a **continuous action space**.

- **World model:** a DIAMOND-derived diffusion world model, with a reward/end model (LSTM) trained on real RGB
  observations. The controller never trains on real transitions directly; it learns from rollouts imagined by the world
  model, starting from replay states.
- **Controller:** an actor–critic trained in imagination. The current family is PMPO (sign-only advantage groups, KL to a
  frozen prior that is refreshed every block), with a Beta or tanh-Gaussian actor and a scalar or two-hot critic.
- **Benchmark task:** `walker_walk` (DM Control), 100k real environment steps. Its reward is
  `stand_reward × move_reward`, which permits non-walking gaits (see CONTROLLER.md).
- **What "stable" means here:**
  - good final performance on the authoritative 30-seed evaluation;
  - reproducible across training seeds, not one lucky seed;
  - no collapse after real data collection ends (the last 50 training-only epochs);
  - no escape into absorbing low-reward modes (sitting, lying).
- **Success target** recorded for controller interventions: final mean ≥ 400 **and** ≥ 50% of evaluation seeds above 500.
  No controller has met it reliably yet.

## Repository layout

- `src/`: training (`main.py`, `trainer.py`, `agent.py`), environments (`envs/`), world model and controllers
  (`models/`: `pmpo_beta.py`, diffusion, `rew_end_model.py`), PMPO diagnostics (`pmpo_diagnostic.py`), optional LCG
  intrinsic reward (`lcg/`), viewer (`play.py`).
- `config/`: Hydra configs. `trainer.yaml` (budgets, world-model training, evaluation, checkpoints), `agent/pmpo_beta.yaml`
  (controller), `env/dm_control.yaml`, `pmpo_diagnostic/` (opt-in diagnostics).
- `scripts/`: launch recipes (`run_walker_100k_pmpo_*.sh`), `resume.sh`, diagnostic tooling.
- `tests/`: pytest suites (`models/`, `checkpoint/`, `lcg/`).
- `docs/`: this file, [CONTROLLER.md](CONTROLLER.md) (controller results and findings) and [PMPO.md](PMPO.md) (PMPO
  controller design, defaults and the history-initialization experiment).
- `outputs/` (git-ignored): run outputs. The local review packages and reports of all experiments up to 2026-11-12
  were deleted on 2026-11-12; the findings survive only in CONTROLLER.md and W&B.

**Branches.** The baseline is `dm-control-pmpo-cleanup` (`b976a6e`). Each controller variant lives on its own branch on
GitHub `shaoche-feng/LCG` (list in CONTROLLER.md). Most variants are based on `a13b57c` (Gaussian controller); the Beta
EMA variant is based on `1fb7be6` (`b976a6e` plus the reviewed repeated-final-frame `env_loop.py` patch).

## Code-writing rules

- **Match the surrounding code:** naming, comment density, idiom. Read neighbouring code before writing.
- **New controller features go behind config flags whose default is off.** With the default, the learning path,
  `state_dict` keys and checkpoints must stay exactly as before (all EMA, two-hot and replay-value changes followed this).
  Add the flag to `config/agent/pmpo_beta.yaml` with a short comment, and validate it in the config checks of the
  controller.
- **Keep checkpoint compatibility.** Do not rename or reshape existing PMPO parameters, optimizer or RNG state. Legacy
  saved configs are migrated by `normalize_saved_config` in `src/utils.py`; never rewrite historical YAML.
- **Every new feature needs unit tests** in `tests/models/` (or `tests/checkpoint/` for resume behaviour), runnable
  without training. Run the suite in the `lcg` environment:
  `OMP_NUM_THREADS=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=<gpu> <python> -m pytest -q tests`.
- **Shell scripts keep LF line endings** (`.gitattributes`: `*.sh text eol=lf`). Scripts edited on Windows must not
  gain CRLF.
- **PMPO constraints:** one device, `grad_acc_steps=1`, zero data-loader workers (imagination RNG is checkpointed).
  `PMPOBeta` accepts only the four-frame stack (`frame_stack=4`).
- **Do not change baseline learning settings, budgets or evaluation seeds** as a side effect of another change. An
  experiment changes exactly the intended setting(s).
- **Launch through the scripts** in `scripts/` (they record provenance: resolved config, source copy, launch command).
  Pass the interpreter with `PYTHON=`.
- **Git:** commit or push only when asked. One branch per experiment. On the server, `git push` has no credentials;
  move commits as a git bundle and push from the local machine (see Access rules).

## Experiment and evaluation rules

- **Authoritative metric:** 30-seed deterministic evaluation (eval30, seeds 3001–3030, 500 steps). The in-training
  4-seed "eval4" (seeds 1001–1004) is a cheap, noisy proxy; don't use it for real comparisons.
- **Report mean, median and fraction above 500** (fractions above 200/400/600 when useful), plus paired per-seed
  differences and wins/losses for comparisons. Medians are unreliable for bimodal outcomes.
- **Before comparing two runs,** confirm the resolved configs differ in *exactly* the intended fields (`diff` the Hydra
  `overrides.yaml`, resolved `config/trainer.yaml` or launch `LAUNCH_CMD`) and report that diff before drawing
  conclusions.
- **One training seed is not a result.** Outcomes vary strongly between seeds; state the number of seeds and treat
  single-seed effects as suggestive.
- **Resume is not bit-exact** for world-model training (denoiser and reward-model losses diverge from the first resumed
  epoch). PMPO actor, prior and PMPO RNG round-trip exactly. Matters only for resumed runs and continuation experiments,
  which need a control arm resumed the same way.
- **Diagnostics are read-only against source runs:** copy or hash inputs, write outputs to a separate folder, verify
  hashes afterwards.

## Checkpoint and disk policy

Trainer always writes its resumable checkpoint: `checkpoints/state.pt`, each dataset's `info.pt`, and
`checkpoints/agent_versions/agent_epoch_NNNNN.pt` (latest epoch plus every `checkpointing.save_agent_every=5`th epoch, up
to `num_to_keep=11`).

The dense `pmpo_diagnostic/` history is opt-in (`+pmpo_diagnostic=<name>`; the 100k Walker launch scripts pass
`+pmpo_diagnostic=bounded_full_loop`). It writes full Trainer-state snapshots at each prior-refresh boundary
(`agent_pmpo_update_NNNNNN.pt`) and at every epoch end (`agent_pmpo_update_NNNNNN_epoch.pt`, NNNNNN = completed updates;
epoch e ↔ `5000 + 400·(e − 1)`), plus `fixed_real_stacks.pt` at the first evaluation. These snapshots are not resume
points; Trainer resumes only from `checkpoints/state.pt`.

This history grows to about 60 GB per 100k-step run. **Do not let it accumulate.** Prune to the checkpoint nearest each
10k-real-step milestone, the true final checkpoint and `fixed_real_stacks.pt`. Never touch `updates.jsonl`,
`epochs.jsonl`, `checkpoints.jsonl`, `.hydra/`, the dataset directory, `checkpoints/state.pt`, or the run folder's
`config/trainer.yaml`, `src/` and `scripts/`. `scripts/resume.sh` resumes with the run's own copied `src/` and
`config/trainer.yaml`.

**Run outputs are git-ignored.** Deleting an output folder, on a server or locally, deletes its checkpoints, datasets and
reports for good. Before deleting, keep what must survive (results in CONTROLLER.md, eval JSONs, reports) and record a
file list.

## Access rules

### kingplus0 (training server)

- 2× RTX 5090 (32 GB). Connect with `ssh kingplus0` (the user's SSH config handles the connection).
- **All work stays under `~/scfeng/`**, never at the home-directory top level. Other accounts share the machine
  (`temp0` holds most of the disk and has run a Jupyter kernel on GPU 0); never touch other users' processes or files.
- **Check GPUs before launching;** use only idle capacity and verify placement during the run.
- Conda env `lcg`: `/home/bigking/anaconda3/envs/lcg/bin/python`. Bare `python` is not on PATH over non-interactive
  SSH; use the full path or `PYTHON=`.
- **GPU isolation:** `common.devices=N` sets `CUDA_VISIBLE_DEVICES=N`, but `MUJOCO_GL=egl` enumerates GPUs itself.
  Always also set `MUJOCO_EGL_DEVICE_ID=N` (and `CUDA_DEVICE_ORDER=PCI_BUS_ID`).
- Python scripts that spawn environment workers must be files, not stdin.
- **No GitHub credentials** on the server: read (clone, fetch, `ls-remote`) works, push does not.
- **State (2026-11-02):** all checkouts and run outputs under `~/scfeng` were deleted after their code was pushed to
  GitHub; only `~/scfeng/LCG-deleted-manifests/` remains. Start new work from a fresh clone. Several non-git folders in
  `~/scfeng` disappeared on 2026-11-02 without an identified cause.

### datalab-king1

- Connect with `ssh datalab-king1` (the user's SSH config handles the connection).
- Conda env `lcg` (Python 3.11): `/data/scfeng/conda/envs/lcg/bin/python`. Use the full path or `PYTHON=`.
- **Only put data or files under `/data/scfeng`.** Nothing in the home directory, `/tmp` or elsewhere in `/data`. That
  includes checkouts, outputs, logs, bundles, caches and environments you control.

### Local machine and GitHub

- Local preservation checkout: `C:/Users/jerry/Project_LCG/LCG-pmpo-beta-k4-20260923`. Its `origin` points at an old
  server bundle, not GitHub; push with the explicit URL `https://github.com/shaoche-feng/LCG.git` (credential manager is
  configured locally).
- No archive of past runs remains: the local server archive (`LCG-server-archive-20261020/`, git bundles and
  manifests) and the local `outputs/` were deleted by 2026-11-12. Code survives on GitHub, metrics on W&B.

### Weights & Biases

- Entity `models-national-tsing-hua-university7947`.
- Projects: `lcg-controller-baselines` (Beta baselines), `lcg-stage3-seed1-interventions` (controller interventions and
  most later runs), `LCG-zero-shot-adaptation`, and one project per bounded diagnostic
  (`lcg-gaussian-h10-ema-replayvalue-ablation`, `lcg-gaussian-h10-ema-controller-modeldata-swap`).
- W&B is the only remaining record of several runs whose server outputs were deleted.
