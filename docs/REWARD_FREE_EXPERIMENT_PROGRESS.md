# Reward-Free Experiment — To-Do List

Updated: 2026-11-04

Branch: `lcg-reward-scaling` (from `reward-free-experiment`)\
Starting commit: `b976a6e`

## Overall structure

**Exploration trainer:** collect data → train denoiser → update exploration signal → train exploration policy → log → save.

**Downstream trainer:** load a saved milestone → freeze denoiser → train a fresh task-reward predictor and policy → evaluate.

Task rewards are recorded separately during exploration and used only for downstream training. Downstream training uses no additional real interactions; evaluation is counted separately.

**Agreed design:** one exploration run per verified compatible task group: one dataset and one shared denoiser, with each task's reward recorded separately. At each selected milestone, the frozen denoiser and matching dataset train a fresh reward predictor and policy per task. Domain and task selection stay configurable. Start with Walker Stand/Walk/Run. Groups and evidence: `docs/DM_CONTROL_TASK_COMPATIBILITY.md`.

## Checklist

### 0. Preparation

- [x] Create the new worktree and branch.
- [x] Read the current code and identify task-reward dependencies.

### 1. Experiment settings

- [x] Confirm the first task group: Walker Stand/Walk/Run.
- [x] Agree the collection design: one exploration run per verified compatible group
  (see above). The main experiment covers verified groups only; the earlier 16-task
  reference list is not required.
- [ ] Choose budgets, saved milestones, training schedules, and evaluation seeds.
  Selected (2026-11-04): the 100k exploration schedule for all five presets: 5,000 initial + 190 epochs x 500 steps,
  no training-only epochs, denoiser 10,000 then 400 updates (85,600), 200 PMPO updates per epoch for every learned
  method (38,000), and ten saved milestones at 10k, 20k, ..., 100k (epochs 10, 30, ..., 190; all kept). For LCG,
  reference reward scaling with 32 prompts (`e5b1e61`). Downstream budgets and evaluation seeds remain open.
  Notes: "Selected 100k exploration schedule".
  Downstream throughput measured on the RTX 5090 (2026-11-02, measurement only; one uniform milestone, Walker walk):
  reward predictor 0.093 s/update, policy 0.380 s/update, evaluation 45 s. Working plan 5,000 + 10,000 updates:
  projected 72 min per adaptation, 362 GPU-hours for 5 methods x 3 seeds x 20 milestones x 1 task. Budgets remain
  undecided. Notes: "Downstream adaptation throughput"; settings document: Section 9A.
  Exploration PMPO updates per epoch, 200 vs 400 (2026-11-03, one frozen LCG model, fresh controller, kingplus0 GPU 1):
  neither improved imagined LCG return meaningfully (200 -> 400: -0.24 +- 0.42 training bank, -0.91 +- 0.51
  independent bank; policy below uniform throughout); 400 costs 17.7 more PMPO GPU-hours per LCG run. Not
  decided. Notes: "PMPO updates per epoch: 200 vs 400".
  PMPO actor objective, sign-only vs advantage-weighted cap 5 (2026-11-03, same frozen model, matched fresh
  controllers): inconclusive; differences under 1% of the return, changing sign between snapshots and banks
  (at 400: +0.20 +- 0.35 training bank, -1.05 +- 0.42 independent bank); both below uniform. Presets unchanged.
  Notes: "PMPO actor objective: sign-only vs advantage-weighted".
  Training batch-size throughput on the RTX 5090 (2026-11-03, uniform milestone, 16 trials): exploration PMPO is
  flat from 16 to 128 (LCG scoring 80-83%), so 32 stays; reward predictor best at 32. Repeatable gains at 64: denoiser
  1.15x, downstream PMPO 1.08x; proposed for review only, learning effect unknown, presets unchanged.
  Notes: "Training batch-size throughput on the RTX 5090".
  Online pilot, 200 vs 400 PMPO updates per epoch with a uniform baseline (2026-11-03; Walker, one exploration
  seed and one adaptation seed, 10,000 env steps, shared initial replay, 37/37 integrity checks; 6.75 GPU-hours).
  Imagined LCG return kept rising past 400 cumulative updates in both arms (at 10,000 steps over uniform: LCG-200
  +4.2 +- 1.5, LCG-400 +23.7 +- 1.6 training bank). Downstream walk: no arm learned to walk (means 46-57, none above
  200). LCG-200 - uniform +11.6 +- 3.2, LCG-400 - uniform +6.1 +- 2.7, LCG-400 - LCG-200 -5.6 +- 3.9 (9/30 seeds).
  400's doubled PMPO time is not justified on this evidence; not decided. Notes: "Online pilot: 200 vs 400".
  Follow-up, full ensemble at 200 (2026-11-04, user request after the pilot; same settings and shared replay,
  16/16 checks, 3.26 GPU-hours): downstream walk mean 79.0 (none above 200); ensemble - uniform +33.4 +- 7.2,
  - LCG-200 +21.7 +- 6.5 (25/30 seeds). One seed pair, and 5x the world-model training. Notes: "Follow-up to the
  online pilot: full world-model ensemble".
  LCG policy-reward scaling (2026-11-04, branch lcg-reward-scaling; new reward_scaling modes data_scaled
  and reference, run with launch overrides, 713 tests pass at the time; LCG-200 arms from the pilot's shared replay, 40/40 checks).
  Both keep the reward scale steady and prevent the critic's late overestimation (positive-advantage fraction at
  10k steps 0.70 / 0.79 vs 0.32). Downstream walk: data-scaled 60.8, reference 65.4 vs LCG-200 57.3 (+3.5 +- 4.0,
  +8.2 +- 5.3); within noise on one seed pair. Notes: "LCG policy-reward scaling". Later committed with reference
  scaling selected in `walker_lcg_100k` (`e5b1e61`; `docs/REWARD_SCALING.md`).
  Continuation to 15,000 steps (2026-11-04; both scaling arms resumed from epoch 10 for 10 more epochs with the new
  ExplorationTrainer.extend_collection_budget, committed as `1c557b3`, 716 tests pass, 57/57 checks, 2.1 GPU-hours per arm). Downstream walk at
  15k: data-scaled 40.7, reference 50.6 (10k: 60.8, 65.4), both lower than at 10k (-20.1 +- 4.1, -14.8 +- 5.9);
  reference - data-scaled +9.9 +- 5.0 (19/30 seeds), inconclusive on one seed pair. Imagined objective over uniform
  rose to +26 / +23. Notes: "Reward-scaling continuation".

### 2. Data collection

- [x] Support initial uniform-random collection and exploration-policy collection.
- [x] Record rewards for every compatible downstream task separately.

Done (`3ba08f3`). Notes: `REWARD_FREE_EXPERIMENT_NOTES.md`, Step 2.

### 3. Exploration trainer

- [x] Keep task rewards inaccessible to the exploration learner.
- [x] Implement the exploration loop described above.
- [x] Remove task-reward-model training and prediction from exploration.
- [x] Handle imagined endings explicitly; do not use an untrained end predictor.

Done (`6b7964a`). Notes: `REWARD_FREE_EXPERIMENT_NOTES.md`, Step 3.

### 4. Saving and resume

- [x] Save matching models, dataset, task labels, configuration, and random states.
- [x] Keep earlier milestones unchanged as collection continues.
- [x] Support collection-boundary resume, including ongoing environment and collector state.
- [x] Generated metadata is JSON only (milestone `manifest.json`, downstream `summary.json`, evaluation reports
  `eval_*.json`); historical generated YAML metadata is no longer read (left as it is). Tested, and an end-to-end
  exploration -> downstream smoke on kingplus0 GPU 1 verified (`14db41b`).
- [x] Default output structure: `outputs/<method>/<domain>/seed_<exploration seed>/` from the configuration (entry point
  and config renamed `src/exploration_main.py`, `config/exploration.yaml`); a new run refuses a folder holding run
  outputs. 10-milestone smokes (uniform, Plan2Explore) on kingplus0 GPU 1 (`2c95eb0`).
- [ ] Custom output roots. Pending.
- [x] Source organization: intrinsic-reward implementations grouped under `src/intrinsic_reward/`, device
  reporting folded into `utils.py`, `pmpo_diagnostic.py` kept in place; before/after evidence identical.
  Committed (`88bc7a2`).

Done (`8a372b7`). Notes: `REWARD_FREE_EXPERIMENT_NOTES.md`, Step 4.

### 5. Downstream trainer

- [x] Load a milestone and freeze its denoiser.
- [x] Train a fresh task-reward predictor and PMPO policy.
- [x] Choose replay sampling explicitly and save outputs separately.
- [x] Evaluate without collecting additional task-training data.
- [x] Downstream checkpoint/resume.
- [x] Launch every milestone in a folder (`initialization.milestones_dir`): sequential independent runs, one
  child process per milestone, train mode only (`94408bb`).

Done (`22bae57`, `0678a6d`, `94408bb`). Notes: `REWARD_FREE_EXPERIMENT_NOTES.md`, Step 5.

### 6. Exploration methods

Start with random and LCG, then add the remaining methods.

- [x] Uniform random. Implemented and smoke-verified; the full 100k run has not been run.
- [x] LCG. Preset with R1 and M5, verified by smoke and resume (`be8a38d`), originally with a fixed reward scale
  (raw / 100); now reference reward scaling with 200 PMPO updates per epoch (`e5b1e61`). The advantage-weighted PMPO
  objective, a 10k-step online pilot and a pilot downstream run are done; no full 100k run.
- [x] Full world-model ensemble (5 bootstrapped denoisers, preset `walker_ensemble_100k`).
  Implemented and verified, settings in `intrinsic_reward` (`4a82dde`, `ea3356d`); no pilot or full run.
- [x] Plan2Explore-style predictor ensemble (one denoiser plus 5 bootstrapped feature heads, preset
  `walker_plan2explore_100k`). Implemented and verified (`cd78477`); settings are placeholders, no pilot or full run.
- [x] Laplace-sampled ensemble (5 samples of LCG's R1 parameters from its historical precision, T = 1, preset
  `walker_laplace_100k`). Implemented and verified as software (`9176c13`); T = 1 kept provisionally, no pilot or full run.

Details and status history: `REWARD_FREE_EXPERIMENT_NOTES.md`, Step 6.

### 7. Logging and analysis

- [x] Terminal logging for the exploration phase. Per-epoch phase output for collection, denoiser, scorer refresh,
  PMPO and evaluation, readable when stdout is captured; final format committed (`62ecded`).
- [x] W&B logging for the exploration phase. Progress, timing, denoiser losses, the 43 actor_critic keys and the
  evaluation returns, with same-run resume and a local mirror; cleanup verified in ten-epoch smokes of all five methods (`2aaec3b`).
- [ ] Analysis for the exploration phase. Test collection with denoiser test loss and a four-seed intrinsic evaluation
  are in place (`84c119d`); comparing exploration across methods and seeds is not done.
- [x] Terminal logging for the downstream phase. One banner and batch plan per command (single milestone or
  folder), then one block per milestone: progress bars for reward-predictor training, task-policy training and
  evaluation, the evaluation summary (max, median, mean, min, std) and the saved paths. Evaluate-only works from
  the completed output without the source milestone. Smoke-verified on kingplus0 GPU 1 (`12f2829`).
- [x] W&B logging for the downstream phase. One run per milestone x task x adaptation seed x member (folder children
  each start one), reward_model / actor_critic / eval groups, provenance, same-run resume and evaluate-only.
  Online smokes on kingplus0 GPU 1 verified against the synced runs (`9c08542`).
- [x] Analysis for the downstream phase. Compare downstream returns across methods, seeds and milestones, and save the
  fixed analysis candidates and artifacts needed for later comparisons. Done so far (`2c95eb0`): per-run
  `result.csv` (`downstream_main.py downstream.mode=collect`) and a cross-method comparison for one exploration seed and
  adaptation seed (`downstream.mode=compare`: CSV, analysis.json, return and threshold plots), smoke-verified on
  uniform and Plan2Explore; review fixes (environment-aware compatibility, ambiguous-policy refusal, plot gaps,
  W&B names identifying the exploration run) tested. Not yet: several exploration/adaptation seeds, and fixed
  analysis candidates.
  Analysis moved to its own entry point, `src/downstream_analysis.py` (2026-11-02, `7c1a383`): the commands above now
  use it, `downstream_main.py` rejects `collect`/`compare`, and old and new results match on tiny uniform and
  Plan2Explore fixtures. Notes: "Standalone downstream analysis entry point".

Details and status history: `REWARD_FREE_EXPERIMENT_NOTES.md`, Step 7.

### 8. Validation

- [x] Test reward isolation, task labels, milestone consistency, and resume.
- [x] Verify that downstream training keeps the denoiser and source dataset unchanged.
- [x] Run a small end-to-end experiment before the full comparison.

## Working rules

- Discuss each step before implementation; settings not listed as selected under item 1 are still pending.
- Check items only after completion and verification.
- Resolve Claude's reported issues within the relevant steps.
- Keep the separate running Stage 3 experiment untouched.
- File formats: YAML for configuration, JSON (only) for generated metadata, JSONL for append-only training
  metrics (notes: "File formats: generated metadata is JSON only").
