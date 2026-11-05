# Reward-free experiment: launch checklist

Updated: 2026-11-05 (launch scripts and Plan2Explore storage smoke). Complete exploration seed 0 and adaptation seed 0 first.

## 1. Freeze the codebase

**Complete this category only after categories 2, 3 and 4 are complete and `requirements.txt` is verified in category 5.** Install the frozen version on both servers; production launch also requires the remaining server checks to pass.

- [x] Finish and commit the reviewed physical-state recording and milestone-selection changes, including the compact evaluation progress-bar adjustment. (`6cf8a07`, `073774c`)
- [ ] Complete configurations, launch scripts and requirements verification; resolve required fixes with focused checks.
- [ ] Record one final commit, confirm a clean working tree and make that version available to both servers. Freeze source, configurations, scripts and requirements.

After freezing, change only blocking bugs with a documented fix. Dataset-diversity analysis, additional seeds and ablations are deferred.

## 2. Configuration: exploration and downstream

### Exploration

| Setting | Target |
| --- | --- |
| Collection | 100k env steps: 5k initial uniform + 190 epochs × 500 steps |
| Denoiser training | 10k updates first epoch, then 400 per epoch; batch 32 |
| Exploration-policy training | 200 updates per epoch, batch 32; none for uniform |
| Training-only epochs | 0 |
| Controller / objective | Gaussian H10, repeat-frame initialization, EMA critic regularization; advantage-weighted PMPO, cap 5 |
| Saved milestones | 20: 5,500, then 10k, 15k, 20k, …, 100k; retain all |
| Physical-state recording | Training on; evaluation off |
| Seed / debug / logging | Seed 0; debug off; W&B online |

- [x] Apply and verify the 20-snapshot schedule. (`0874335`: `milestones.at_steps`, epochs 1 and 10–190, all retained; verified for all presets and groups and in the smoke: 20 milestones per run)
- [x] Finalize exploration evaluation: frequency, four fixed periodic seeds, and final episode count/seeds. Every 10 epochs, four periodic episodes with seeds 3001–3004; one separate final evaluation of 30 episodes with seeds 3001–3030. All 35 production compositions pass the trainer's episode-count/reset-seed check.

### Downstream

| Setting | Target |
| --- | --- |
| Initial milestone selection | Ten: 10k, 20k, …, 100k, using the folder-mode step filter |
| Reward-predictor training | 5k updates, batch 32 |
| Policy training | 10k updates, batch 32 |
| Controller / objective | Gaussian H10, repeat-frame initialization, EMA critic regularization; sign-only PMPO |
| Initialization | Fresh reward predictor/controller each adaptation; frozen source replay and denoiser; member 0 |
| Evaluation | 30 seeds, 3001–3030; maximum 500 steps per episode |
| Adaptation seed / debug / logging | Seed 0; debug off; W&B online |

- [x] Put these settings into maintained configurations and check resolved values. Each downstream launch takes one task and processes selected milestones sequentially. (`downstream.yaml`: 5,000 + 10,000 updates; `+experiment=downstream_10_milestones initialization.milestones_dir=<run>/milestones`; `0874335`)
- [x] Select the first task group/task and execution order; set W&B projects, groups and run names for both phases. (Launchers: all seven groups explored first, then every task adapted in group order. Projects `lcg-reward-free` / `lcg-reward-free-downstream`; exploration runs `<method>-<group>-exploration-seed0` in group `<method>-<group>-exploration`; downstream runs `<method>-<group>-exploration-seed0-steps_N-<task>-adaptation-seed0` in group `<method>-<group>-exploration-seed0-<task>-adaptation-seed0`.)

## 3. Configuration for each method

| Method | Method-specific settings | Server |
| --- | --- | --- |
| LCG | Reference scaling, 32 calibration prompts; candidate bank 5, chunk 32; historical reference 320 × 3 samples | 1 |
| Laplace ensemble | 5 samples; fixed reward divisor 1.0 | 1 |
| Full ensemble | 5 denoisers, each following the common denoiser schedule; fixed reward divisor 1.0 | 2 |
| Plan2Explore-style | 5 predictors; fixed reward divisor 1.0 | 2 |
| Uniform | Uniform actions; no exploration-policy training or intrinsic scoring | 2 |

- [x] Prepare/check each method's YAML in the existing configuration structure. Verify category 2 settings and compilation settings; remove diagnostic overrides. (presets renamed to `<method>_100k`, domain-free; production scoring in `intrinsic_reward/`; `config/pmpo_diagnostic/` removed; `0874335`)
- [x] Cover the seven accepted groups (all 35 preset × group compositions verified; R1 guards hold for every domain's action size): Walker (stand/walk/run), Hopper (stand/hop), Acrobot (swingup/swingup_sparse), Cartpole balance (balance/balance_sparse), Cartpole swingup (swingup/swingup_sparse), Humanoid (stand/walk/run/run_pure_state), Humanoid CMU (stand/walk/run).
- [x] Keep environment settings identical across methods within each group. Give the two Cartpole groups distinct output locations that downstream analysis can distinguish. (Environment settings are shared YAML. The launchers write `outputs/<method>/cartpole-balance/seed_0` and `outputs/<method>/cartpole-swingup/seed_0`, both domain `cartpole`. Note: `downstream_analysis.py compare` locates runs as `outputs/<method>/<domain>/seed_N` and checks the recorded domain, so it does not yet handle the Cartpole group folders.)
- [x] Compose/check intended launch configurations and update the existing settings document, including 20 saved versus ten initially adapted milestones. Correct stale comments describing exploration as sign-only. (settings document Section 0; `outputs/step32-config-cleanup/evidence/config_resolution.json`)

## 4. Shell scripts for each method

- [x] Prepare/check five `.sh` launchers: LCG, Laplace, full ensemble, Plan2Explore and uniform, using existing script locations. (`scripts/run_{lcg,laplace,ensemble,plan2explore,uniform}_100k.sh`, from the user's `reference.sh`; `bash -n` clean; obsolete diagnostic folders and `run_walker_100k_pmpo_*.sh` removed.)
- [x] Include the intended YAML, task group, seed, output path, GPU/EGL mapping and online W&B settings. Keep shared experiment settings in YAML rather than duplicating conflicting overrides. (Checked by composing every command: 35 explorations, 90 downstream commands; budgets come from YAML only.)
- [x] Capture logs and exit codes, preserve failures through `tee`, use a persistent session and provide a resume command. (Unique `logs/<method>/<date>-<id>/` per invocation; `set -euo pipefail`; failures stop the script with their exit code; run in tmux; `scripts/resume.sh <run folder>`.)
- [x] Prepare downstream commands with the ten-step filter and one task per command. Verify shell syntax, resolved settings and output paths. (Verified in the Plan2Explore smoke: each task adapted exactly 10k–100k from its own run's milestones.)
- [ ] Record physical GPU assignments: Server 1 runs LCG/Laplace; Server 2 runs full ensemble and sequential uniform/Plan2Explore. Keep one of our training processes per GPU.

## 5. Server environment setup and requirements

- [ ] Verify `requirements.txt` is complete and consistent with the tested environment, including W&B 0.30.0. Record the appropriate PyTorch/CUDA installation choice for both RTX 5090 servers. **Requirements verification is necessary to complete category 1.** (Laptop done 2026-11-05: `pip check` clean, all pins installed, imports declared; README records the cu128 wheel index and EGL needs. Remaining: the same checks on both servers; decide on `h5py`/`opencv-python`, unused by maintained code.)
- [ ] Set up SSH keys/VS Code access; clone/fetch the frozen commit; create environments and install requirements on both servers.
- [ ] Run `python -m pip check`, import checks, CUDA/compilation checks, EGL rendering and W&B login on each server.
- [ ] Confirm writable data paths and capacity. Planning targets: Server 1 ~150 GB free; Server 2 ~200 GB free, for one seed across all seven groups and ten downstream adaptations per task. These are estimates; Server 1 currently reports 204 GB available on shared `/data`.
- [ ] Confirm quotas, temporary-server deadlines and backup destinations. Preserve shared episode files along with milestones.
- [ ] Run bounded preflight smokes separately from production; record server/GPU, commit, commands and results in an existing project document. Launch after all five categories are complete. (Done on kingplus0 GPU 1, 2026-11-05: `scripts/smoke.sh`, Plan2Explore, Walker [walk] and Hopper [stand], separate checkout, 120.8 min, all checks passed; storage ~2.0 GB exploration + 0.32 GB downstream per group; recorded in the notes. Remaining: the second server.)
