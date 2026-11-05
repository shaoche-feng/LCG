# Reward-Free Experiment — Step Notes

Implementation notes for each completed step. The checklist and next step are in
`docs/REWARD_FREE_EXPERIMENT_PROGRESS.md`; task-group evidence is in
`docs/DM_CONTROL_TASK_COMPATIBILITY.md`.

## Step 2 — Data collection

Commit `3ba08f3`; verified locally 2026-10-15, `lcg` env, 253 tests passed.

- Collection mode is per collect call: `collection.train.first_epoch.mode` (default
  `uniform`) and `collection.train.mode` (default `policy`). Uniform mode never evaluates
  the policy; one shared environment loop keeps episodes in progress across mode switches.
- `env.*.task_names` (Walker: `[stand, walk, run]`). The first task drives the collection
  environment; every other task is a separate dm_control `Task` whose `get_reward` is
  evaluated on the collection physics after every executed action-repeat substep and
  summed per task. Labels add no physics steps and draw no randomness.
- Several tasks are accepted only when all belong to one verified group
  (`REWARD_LABEL_TASK_GROUPS`): acrobot, cartpole (two groups), hopper, humanoid,
  humanoid_CMU, walker. Structure (task/physics class, MuJoCo model, timestep, time limit,
  action spec) is also checked at construction. Evidence, exclusions and source references
  for all 19 installed domains: `docs/DM_CONTROL_TASK_COMPATIBILITY.md`.
- Stored as `Episode.task_rew` (T x tasks) and `Episode.task_names`, sliced with segments
  (`Segment/Batch.task_rew`); `Dataset.task_names` rejects mixed labels.
- Not yet reward-free: `Episode.rew` (first task) still trains the reward/end model and
  reaches the PMPO learner. Isolation is Step 3.

## Step 3 — Exploration trainer

Commit `6b7964a`; verified locally 2026-10-15, `lcg` env, 267 tests passed, Walker smoke run.

- `src/trainer.py` is now `src/exploration_trainer.py` (`ExplorationTrainer`); `src/main.py`
  uses it. Loop: initial collection, then per epoch collect, train denoiser, refresh LCG,
  train PMPO on intrinsic reward, log, save.
- Method comes from existing fields: `collection.train.mode=policy` is learned (LCG)
  exploration and needs `intrinsic_reward.enabled=true` (now the default);
  `collection.train.mode=uniform` with `intrinsic_reward.enabled=false` is the uniform
  baseline (denoiser only, no PMPO, no imagination env). Other combinations are rejected.
- Reward isolation: every exploration loader uses `collate_segments_without_rewards`
  (task labels dropped, `rew` zero, `info` metadata replaced by empty dicts; the dataset
  and saved files keep everything). LCG reads only obs/act windows. Changing stored task
  rewards leaves weights, LCG precision and all logged metrics bit-identical (test).
- No reward/end model in exploration: the agent is built without one, so there is no
  optimizer, training, test loss, prediction or burn-in. `WorldModelEnv` without one
  returns a zero reward placeholder and no true endings; episodes truncate at the horizon.
  PMPO refuses to update without an intrinsic hook in that case.
- Endings: only time-limit-only domains are accepted (`TIME_LIMIT_ONLY_DOMAINS`: every
  suite domain except lqr); a real true termination stops the run.
- Resume: superseded by Step 4 below.
- Note: run folders copy `scripts/`; on Windows keep run paths short (MAX_PATH).

## Step 4 — Saving and resume

Commit `8a372b7`; verified locally 2026-10-15, `lcg` env, 286 tests passed, Walker resume smoke run.

- Replay files are immutable versions `<id>_<length>.pt`; extending an unfinished episode
  writes a new version. Indexes (checkpoint, milestones) read exactly what they recorded.
  Unreferenced versions are pruned after each committed checkpoint.
- `checkpoints/state.pt` is the atomic commit point: models, optimizers, schedulers,
  counters, replay index, global and PMPO RNG, collector state (unfinished episode ids,
  observation, frame history) and each real env's full MuJoCo/task state, plus source and
  package identity. Faithful resume at completed epoch boundaries only; older checkpoints,
  changed packages or a different `env.train` are refused. LCG is rebuilt each epoch.
- Milestones `milestones/steps_NNNNNN/` (`denoiser.pt`, `dataset.pt`, `manifest.yaml`),
  published complete after the checkpoint commit; settings `milestones.at_steps`,
  `every_steps`, `keep_last` (null or >= 1; values undecided). Retention is recorded in the
  checkpoint before the commit and applied after it, so it survives interruptions.
  `data.milestone.load_milestone(path, task)` verifies SHA-256 of every file (episode
  checksums are taken once, when each version is written) and resolves relative paths;
  `downstream/` is reserved. Run identity and RNG helpers live in `utils.py`.
- Uninterrupted vs resumed runs are bit-identical (CPU tests, learned and uniform; GPU
  Walker smoke with default model sizes and LCG).
- Fixed: `Episode.save` truncated frames to uint8, lowering some levels on every save; it
  now rounds, so save/load is exact.
- Resumed runs log `resume/from_epoch`; epochs replayed after a crash reuse their epoch
  numbers (keep the last row per epoch).

## Step 5 — Downstream trainer

### Part 1: training

Commit `22bae57`; verified locally 2026-10-17, `lcg` env, 294 tests passed, real Walker
downstream smoke run on milestone `steps_000700`.

- `src/downstream_trainer.py` (`DownstreamTrainer`), entry point `src/downstream_main.py`
  with `config/downstream.yaml`: `initialization.milestone_path`, `downstream.task_name`,
  `downstream.seed`, budgets `rew_end_model.training.steps` and
  `actor_critic.training.steps` (placeholders).
- Loads the exact milestone (`load_milestone`), uses its saved env and denoiser config;
  denoiser frozen (no optimizer, eval). Fresh reward/end model and PMPO; nothing restored
  from exploration; randomness from `downstream.seed`.
- One dm_control environment is built only to read the action spec, then closed; no real
  resets, actions, collection or evaluation. All task-training data is the milestone's.
- Two phases: reward model on the selected task's reward column only, then frozen; PMPO in
  imagination on its predicted reward. No LCG; imagined episodes never end, only truncate
  at the horizon (`WorldModelEnv(sample_ends=False)`).
- Sampling (`sample_weights: null`): episode by length, anchor step uniform, so anchors are
  uniform over stored transitions; windows are placed around the anchor (reward model may
  pad both sides, prompts only before the start), so per-step coverage is not exactly
  uniform. The last stored step's next frame is the episode's final observation or the
  milestone's successor observation at the cut.
- Outputs: `<run>/downstream/steps_N/<task>/seed_<seed>/` (config, metrics, weights,
  summary), written aside and renamed; never overwritten; sources untouched.

### Part 2: evaluation and resume

Commit `0678a6d`; verified locally 2026-10-17; review corrections verified 2026-10-18,
`lcg` env, 307 tests passed, GPU Walker smoke: train, kill, resume, evaluate through
`src/downstream_main.py`.

- `downstream.mode`: `train` (then evaluate), `resume`, `evaluate` (completed outputs,
  including earlier weights-only ones).
- Evaluation: the requested task built explicitly with the milestone's env settings,
  deterministic actions, new frame history per seed, `evaluation.seeds` (default 3001-3030)
  and `evaluation.max_steps` (default 500). Return = sum of task rewards per environment
  step, each summing its action-repeat substeps. Results in `evaluations/`, named by policy
  and protocol hashes, never overwritten. No data, training, counters or RNG changes; the
  next update is identical with or without evaluation.
- Checkpoints `.seed_<s>.tmp/checkpoints/state.pt` (atomic) every `checkpointing.every`
  updates, at the phase transition and at the end: models, optimizers, schedulers,
  counters, phase, all RNG, config, source identity and the milestone's identity and
  relative path, plus the checksum of its `dataset.pt` (pins the replay contents; format 2,
  format-1 checkpoints refused). Denoiser and replay are referenced, not copied. Resume is bit-identical
  to uninterrupted training (tests on CPU in both phases and at the transition; GPU smoke).
  Changed settings, other milestones, changed packages and weights-only outputs are refused.
- `cudnn.deterministic=True` is now set for downstream runs (GPU resume equality).
- Evaluation restores every submodule's own train/eval mode (also on errors). Resume first
  repairs an interrupted last metrics row (kept verbatim in `metrics.jsonl.interrupted-<n>`)
  and refuses corruption inside the log.
- Smoke runs: `outputs/step5-part2-smoke/`, `outputs/step5-part2-fix-smoke/` (gitignored).

### Part 3: folder-based milestone launching (2026-10-29)

Committed as `94408bb`. Review: `docs/changes/downstream_folder_launch_review.zip`. Evidence: `outputs/step14-downstream-folder-launch/`. Terminal output was revised later (Step 7, "Downstream terminal format"): the representative output below is the earlier format. Verified on CPU with tiny models only; no exploration retraining and no full downstream adaptation.

- **Scope:** `src/downstream_main.py` launches the milestones; each `DownstreamTrainer` still trains one milestone × one task × one adaptation seed. No W&B logging, and the trainer's terminal output is unchanged. Budgets, sampling, the controller, the evaluation protocol and reward scaling are unchanged.
- **Configuration:** give exactly one of `initialization.milestone_path` (one milestone; unchanged behavior) and `initialization.milestones_dir` (every milestone in a folder; default `null`). The folder field is launcher-level: `downstream_main.pop_milestones_dir` removes it, and `DownstreamTrainer` drops it too (and refuses a non-null one), so every saved `config.yaml` and checkpoint config is a normal single-milestone config. Existing checkpoints and weights-only outputs resume and evaluate as before.
- **Discovery** (`data.milestone.discover_milestones`): every immediate child of the folder must be a published milestone (a `manifest.yaml` with the current schema). They are ordered by the manifest's `collected_steps`, not by folder name. Rejected: a missing or empty folder; any other entry, including an interrupted publish `.<name>.tmp`; a manifest without an integer step count; two milestones recording the same step count.
- **Preflight** (before anything is launched; all problems reported together, "Nothing was started"):
  - every milestone is loaded with `load_milestone` on CPU for the requested task and `initialization.member_index`: file presence, checksums of every listed file and episode, and the task's reward labels;
  - output conflicts for the whole selection: an existing output `<run>/downstream/<milestone>/<task>/seed_<seed>/` or an incomplete attempt `.seed_<seed>.tmp/`. Nothing is overwritten, deleted or skipped. Each child keeps the trainer's own checks as well.
  - Cost: each milestone's replay files are hashed once here and again in its child run.
- **Runs:** sequential, one child process per milestone (`child_command_base`, `milestone_command`): `sys.executable src/downstream_main.py --config-path=<parent's primary config path> [--config-dir=<each of the parent's>] --config-name=<same> <the parent's hydra.* overrides> <the parent's overrides, minus milestones_dir, milestone_path and downstream.mode> ++initialization.milestones_dir=null initialization.milestone_path=<milestone> downstream.mode=train`. The folder field is cleared explicitly, because it may come from a config file rather than the command line (revised 2026-10-29 after review: the first version only dropped the command-line override, so a YAML-configured folder reached every child as a second milestone source, and custom `--config-path`/`--config-dir` were not passed on). Each child loads its milestone's frozen denoiser and replay and trains a fresh reward model and sign-only PMPO controller from `downstream.seed`; no state, GPU memory or RNG carries over from the previous milestone. Outputs use the existing layout, one folder per milestone. The launcher itself does no GPU work; since the downstream terminal format (Step 7) it sets `CUDA_VISIBLE_DEVICES` from `common.devices` for its banner's device lines, and each child sets the same value.
- **Modes:** folder mode supports `downstream.mode=train` only. `resume` and `evaluate` with `milestones_dir` are rejected with a message that they take one milestone (`initialization.milestone_path`). No batch resume and no automatic skipping.
- **Failure:** the launcher stops at the first child with a nonzero exit, launches nothing further, prints the failed milestone and the milestones not started, and exits with status 1. It deletes nothing. For the failed milestone it reports, by what is on disk:
  - training outputs published (`summary.yaml` in the output folder; the failure came after training, e.g. in the evaluation): their path and the single-milestone evaluate command (added 2026-10-29 after review; such a failure was reported as "no work folder" before);
  - a checkpoint in the work folder: the folder and the resume command;
  - a work folder without a checkpoint: delete it to rerun;
  - otherwise: no work folder was created.

  Recovery commands are the child's own command with only `downstream.mode` changed, so they keep the configuration source, task, seed, member and every other setting.

Commands:

```
# One milestone (unchanged)
python src/downstream_main.py initialization.milestone_path=<run>/milestones/steps_010000 \
    downstream.task_name=walk downstream.seed=0
# Every milestone in a folder (train only), in collection order
python src/downstream_main.py initialization.milestones_dir=<run>/milestones \
    downstream.task_name=walk downstream.seed=0
# A failed or interrupted milestone is resumed on its own
python src/downstream_main.py initialization.milestone_path=<run>/milestones/steps_020000 \
    downstream.task_name=walk downstream.seed=0 downstream.mode=resume
```

Representative output (the tests' tiny two-milestone run, `demo_1_batch_train.log`; paths shortened):

```
Downstream batch: 2 milestones from <run>/milestones
  task walk | adaptation seed 0 | member 0 | mode train | sequential, one process per milestone
  1. steps_000020 (20 collected steps) -> <run>/downstream/steps_000020/walk/seed_0
  2. steps_000026 (26 collected steps) -> <run>/downstream/steps_000026/walk/seed_0

=== Downstream 1/2: steps_000020 (20 collected steps) ===
Starting on cpu
Downstream outputs: <run>/downstream/steps_000020/walk/seed_0

=== Downstream 2/2: steps_000026 (26 collected steps) ===
Starting on cpu
Downstream outputs: <run>/downstream/steps_000026/walk/seed_0

Downstream batch summary (task walk, adaptation seed 0):
  steps_000020: completed in 0.1 min -> <run>/downstream/steps_000020/walk/seed_0
  steps_000026: completed in 0.1 min -> <run>/downstream/steps_000026/walk/seed_0
All 2 completed.
```

On a failure (`demo_4_child_failure.log`, exit status 1):

```
Downstream batch summary (task walk, adaptation seed 1):
  steps_000020: FAILED (exit code 1) after 0.1 min; no work folder was created
  steps_000026: not started
Stopped at the failure; later milestones were not started.
```

After a crash past a checkpoint, the failed line reads `resumable work folder <...>/.seed_1.tmp kept; resume it with:` followed by the full command; after a failure in the evaluation it reads `training outputs were published at <output> (...); evaluate them with:` followed by the full command. Re-running a completed batch is rejected before anything starts, listing every existing output (`demo_2_rerun_conflicts.log`).

**Tests** (`tests/downstream/test_downstream_trainer.py`; `build_milestone(early=...)` also publishes earlier milestones of the same tiny run):
- single-milestone entry point unchanged, and its `config.yaml` has no folder field; exactly one milestone source; the trainer drops or refuses the field;
- discovery: order by recorded steps against reversed folder names; missing, empty, malformed (no manifest, interrupted publish, stray file, non-integer steps) and duplicate-step folders rejected;
- preflight: missing task, a corrupted file, and output conflicts at the last milestone (existing output and incomplete attempt) all rejected before any child starts, with every file unchanged; resume/evaluate rejected;
- children (recorded commands): one per milestone in recorded-step order, each with its own milestone and otherwise identical arguments (executable, script, config path, config name, overrides, the cleared folder field); a failure stops later launches, keeps completed and resumable outputs, prints the full resume command, and returns 1;
- child command construction: `--config-path`, every `--config-dir`, the config name and `hydra.*` overrides are kept; `milestones_dir` (also as `+...`), `milestone_path` and `downstream.mode` are replaced;
- folder set in a config file, loaded through `--config-dir` and through `--config-path` (two real children each): every child has exactly one milestone source and the file's non-default task (`run`), seed (3), learning rate and tiny budgets;
- failure after publication: a child that published its training outputs and then failed; the batch stops, nothing is deleted, the summary names the outputs, and the printed evaluate command (same task, seed, member, evaluation settings) evaluates them when run;
- integration (two real child processes): separate outputs with their own milestone, identical configs apart from `milestone_path`, and the later milestone's weights and metrics bit-identical to the single-milestone command on a copy of the run folder (no warm start). The reference is a fresh process too: inside the full test suite an in-process run differed slightly, through process-global state left by earlier tests, which is what child-process isolation avoids;
- a real failing child gives exit status 1, nothing later is started.

## Step 6 — Exploration methods

### Status history (moved from the progress file, 2026-10-29)

- [x] Uniform random. Configuration and workflow verified (smoke); the full 100k run has
  not been run. Notes: `REWARD_FREE_EXPERIMENT_NOTES.md`, Step 6.
- [x] LCG. Preset with R1, M5, fixed reward scale (raw / 100); configuration and workflow
  verified (smoke, resume). Done (`be8a38d`). The full 100k run has not been run; the
  bounded smoke showed no controller improvement. Notes: `REWARD_FREE_EXPERIMENT_NOTES.md`, Step 6.
  Optional advantage-weighted PMPO objective implemented (`eee48cd`; sign-only stays the
  default). One-seed frozen-LCG check: higher imagined LCG return than sign-only, consistent
  with uniform within its uncertainty (neither superiority nor equivalence shown); controller and full-run schedule not validated. Notes: same file, Step 6.
  Bounded 10k-step online pilot (weighted LCG vs uniform, one seed): the loop works end to end;
  the policy beats uniform on its own imagined LCG objective by epoch 10; held-out denoiser
  loss not better than uniform. Not a validation of LCG exploration or the full schedule.
  Pilot downstream (walk, one seed, 5k reward-model + 2k PMPO updates): LCG milestone 87.5 vs
  uniform 59.2 mean return (paired +28.3 +- 7.1 over eval seeds only); both weak/bimodal near a
  floor; inconclusive beyond this seed.
- [x] Full world-model ensemble (MAX-inspired diffusion ensemble disagreement; 5 bootstrapped
  denoisers, uniform per-transition member selection, preset `walker_ensemble_100k`).
  Implemented and verified (tests; local-GPU smoke with bit-identical resume); done (`4a82dde`);
  no pilot or full run. Notes: `REWARD_FREE_EXPERIMENT_NOTES.md`, Step 6.
  Review corrections (2026-10-21): older downstream checkpoint resume and refusal of partial
  ensemble initialization via `Agent.load()`; 412 tests passed; local-GPU smoke re-run on the
  corrected code (bit-identical resume).
  RTX 5090 timing (PMPO update): K = 4 at or below LCG (0.82x), K = 5 within 10% (1.02x);
  preset unchanged pending review.
  Ensemble settings moved to `intrinsic_reward.num_members` / `bootstrap` (older `ensemble.*`
  configs read in memory); done (`ea3356d`).
- [x] Plan2Explore-style predictor ensemble (one denoiser plus 5 bootstrapped next-feature
  heads on `unet.d_blocks.2` + `unet.d_blocks.3` features, 4x4 pooling, pixel sigma 0.1 with the
  denoiser's 0.3 offset noise (effective conditioning sigma ~0.316); preset
  `walker_plan2explore_100k`). Implemented and verified (44 tests, full suite 462 passed after the review corrections;
  local-GPU smoke with bit-identical resume, exact refit reproduction, downstream milestone
  loading); done (`cd78477`). Settings are placeholders; no feature diagnostic, pilot or full run.
  Notes: `REWARD_FREE_EXPERIMENT_NOTES.md`, Step 6; design: `docs/PLAN2EXPLORE_DESIGN.md`.
- [x] Laplace-sampled ensemble (one denoiser; 5 Gaussian samples of the LCG-selected R1
  parameters from LCG's own historical precision, T = 1; full-ensemble pixel disagreement and
  member selection; preset `walker_laplace_100k`). Implemented and verified as software (tests
  pass; local-GPU smoke: training state matched across uninterrupted and resumed runs, bank
  reproduced exactly, shared-prefix path exact, downstream milestone loading); not committed.
  The smoke driver exited 1: the resumed run's final test episode diverged from step 260 (cause
  not isolated). Sampling-scale concern (frozen-model diagnostic, 2026-10-27): at T = 1 every
  member predicts a blank grey frame (perturbations ~1,000-2,000x the selected weights, far
  outside the local linear regime). Reported, not tuned. Trained-model scale check (uniform
  pilot `steps_010000`, 10k steps / 13,600 denoiser updates, verified source): at T = 1 the
  data term dominates the precision, perturbations are 1.4x the selected weights, one-step and
  6-step predictions keep the scene, and disagreement is robust to TF32 and quantization; below
  T ~ 1e-4 rewards are arithmetic/quantization-dominated. Calibration (does disagreement track
  error?) and exploration benefit untested. First-refresh check (pilot `steps_005500`: 10,000
  denoiser updates, the actual first-refresh state): T = 1 preserves usable predictions
  (perturbation 1.8x, coherent 6-step rollouts); T = 1 retained provisionally for a bounded online
  pilot. No pilot or full run. Notes:
  `REWARD_FREE_EXPERIMENT_NOTES.md`, Step 6.

### Uniform random

Commit `a837015`; verified locally 2026-10-18, `lcg` env, 314 tests passed, GPU Walker
smoke runs through `src/main.py` and `scripts/resume.sh`. The full 100k run has not been run.

- Preset `config/experiment/walker_uniform_100k.yaml` (`+experiment=walker_uniform_100k`):
  Walker `[stand, walk, run]`, uniform initial and later collection, LCG off, seed 0,
  W&B online (`lcg-reward-free`). Uses the existing `ExplorationTrainer`; the policy object
  exists only for its frame stack and never acts or updates.
- Budget: 5,000 + 190 x 500 = 100,000 training environment steps (200,000 simulator
  control steps at action_repeat 2), 240 epochs (190 + 50 final), 105,600 denoiser updates.
  `check_collection_budget` rejects budgets that would not end exactly, at startup.
- Separate test collection with the defaults (uniform actions, test dataset only):
  48,000 + 50,000 = 98,000 environment steps. Not counted in the 100k; a decision for the
  final schedule.
- Denoiser schedule and milestones stay at the repository defaults, to be shared by all
  methods once agreed.
- Smoke (`outputs/step6-uniform-smoke/`): all actions uniform, no policy evaluation, no
  PMPO/LCG, finite denoiser losses, exact replay counts and labels, milestones, retention
  and pruning; resumed runs bit-identical to a plain uninterrupted run. One harness-run
  uninterrupted run differed in 3-4 pixel values of 2 frames from every other run
  (cross-process GPU rendering variation, not resume).

### Exploration test collection and diagnostics

Commit `84c119d`; verified locally 2026-10-19, `lcg` env, 329 tests passed, GPU Walker smoke
runs through `src/main.py` and `scripts/resume.sh` (`outputs/exploration-test-diagnostics/`).

- Flags (`config/trainer.yaml`, `evaluation`):
  - `should` (default true): false skips every periodic and final test collection and
    diagnostic.
  - `every` (default 10): epochs between diagnostics; must be a positive integer when
    `should` is true. A final diagnostic follows the last epoch (not with `pmpo_diagnostic`).
  - `collect_test` (default true): true collects fresh test episodes
    (`collection.test.num_episodes`, final `num_final_episodes`) right before each
    diagnostic; false reuses the existing test dataset and collects nothing. Configs without
    the key behave as true (the earlier behaviour). A new run has no test data, so
    reuse-only is rejected at startup (before any file is written); on resume or with a
    static dataset an empty test dataset or a missing episode file is rejected before
    training. Training replay is never substituted. Static datasets never collect (as
    before) and their diagnostics always reuse the static test split.
  - The uniform preset keeps diagnostics on explicitly (`should: true`, `every: 10`,
    `collect_test: true`).
- Diagnostics (`ExplorationTrainer.test_diagnostics`, logged under `test/` or `final_test/`):
  - Denoiser test loss (`denoiser/test/*`, unchanged) on reward-free batches
    (`collate_segments_without_rewards`).
  - Per-task return summaries `return_<task>/{mean,std,min,max}`: for each test episode, the
    sum of each recorded task-reward column (each step already sums its action-repeat
    substeps). Read only for logging; never a denoiser, LCG or PMPO input. Exploration still
    has no task-reward predictor; these are not reward-prediction metrics (those belong to
    downstream work).
  - `num_episodes`, `num_steps` (test dataset used), `fresh` (1 collected now, 0 reused).
- Accounting (environment steps; x action_repeat for simulator control steps):
  `new_env_steps` (this point; 0 when reused) and `total_new_env_steps` (all fresh test
  collection so far, final included; committed as `num_test_steps`). Separate from the
  training budget (`train_dataset/num_steps`). The collector now also logs
  `<dataset>/new_env_steps`, the env steps of each collect including discarded unfinished
  episodes. Uniform preset defaults: 24 x 4 + 100 test episodes = 98,000 env steps
  (196,000 control steps) besides the 100,000 training steps.
- Isolation: test collection and diagnostics run on their own RNG stream (seeded from
  `SeedSequence([seed, 1])`, saved in checkpoints as `test_rng`); the training RNG is saved
  before and restored after, and every submodule's train/eval mode is restored. Training
  (weights, optimizers, PMPO prior and updates, schedules, replay, actions) is bit-identical
  with diagnostics on or off. Test episodes stay in `dataset/test`, never in training replay
  or milestones.
- Committed test files: each fresh collection writes a new generation folder
  (`dataset/test/generation_N/`, `Dataset.clear(keep_files=True)`); the previous
  collection's files are pruned only after the next checkpoint commit. The final diagnostic
  is committed (`final_test_done`), so resuming a finished run does not repeat it.
- Resume: checkpoint format 2 (test RNG and counters); format-1 checkpoints are refused.
  Resume continues with fresh or reused test data (`scripts/resume.sh <run>
  evaluation.collect_test=false`); with fresh data it is bit-identical to an uninterrupted
  run, including test episodes and diagnostics.
- Behaviour change: the final test collection used to run even with `evaluation.should=false`
  and only logged collection returns; it now obeys the flags and runs diagnostics.
- Smoke (uniform preset, 500 + 3 x 100 steps, test every epoch, 1 + final 2 episodes): off
  (no test files, no diagnostics); fresh (500/1000/1500 then final +1000 = 2500 test env
  steps); reuse-only on a new run rejected before any checkpoint; killed after epoch 1 and
  resumed with `evaluation.collect_test=false` (0 new steps, same committed episode); killed
  between epoch 2's new test files and its commit, resumed fresh. Training state is exactly
  identical across off, fresh and both resumed runs; the fresh resume also matches test RNG,
  counters and returns; one test frame differs in 2 values by one level (the cross-process
  GPU rendering variation seen in Step 6).

### LCG

Commit `be8a38d`; verified locally 2026-10-19, `lcg` env, 355 tests passed, GPU Walker
smoke runs through `src/main.py` and `scripts/resume.sh` (`outputs/step6-lcg-smoke/`). The
full 100k run has not been run.

- Preset `config/experiment/walker_lcg_100k.yaml` (`+experiment=walker_lcg_100k`): Walker
  `[stand, walk, run]`, uniform initial collection then policy collection, LCG on. Budget,
  collection and denoiser schedules, milestones, test diagnostics, models and PMPO schedule
  are the uniform preset's / repository defaults (test-checked); only the method differs.
- LCG settings: theta_S = R1 `unet.u_blocks.3.resblocks.0.*` (10 tensors, 217,664 scalars,
  enforced by `theta_s.expected_num_tensors/expected_dim`, checked when the trainer builds
  the model, before collection or checkpoints); candidate MC 5 (Full CRN, the existing and
  only estimator), chunk 16; historical reference 320, MC 3; damping 1e-4; beta 1.0; scorer
  refreshed once per epoch (unchanged).
- Reward scaling (`intrinsic_reward.reward_scaling`): `fixed` = raw LCG / `fixed_divisor`
  (finite, > 0; preset 100), nothing else -- no running statistics, centering, clipping or
  novelty compensation, same divisor every batch and epoch. `rms` (RunningRMS, reset at each
  refresh as before) and `none` remain. Must agree with `rms_enabled` (true only for rms);
  configs without `reward_scaling` keep their meaning (rms if `rms_enabled` else none). The
  resolved scaling is saved in the checkpoint (resume format 3) and a resume with different
  scaling is refused.
- Logging per PMPO update: `actor_critic/train/lcg_raw_reward_*` and `lcg_policy_reward_*`
  (mean, std, min, p10, median, p90, p99, max). Per epoch: `timing/lcg_refresh_seconds`,
  `lcg_precision_seconds`, `lcg_scoring_seconds`, `pmpo_update_seconds`, `pmpo_updates`,
  `gpu_peak_allocated_gb`, `gpu_peak_reserved_gb` (wall clock; excluded from run
  comparisons).
- Smoke (preset + 500 + 3 x 100 steps, denoiser 20/10/10 and PMPO 20/10/10 updates,
  `compile_wm=false`, test every epoch; full R1/M5/320/3 settings): 500 uniform then 300
  policy actions; LCG refresh each epoch; losses and reward summaries finite; milestones,
  retention, pruning and test diagnostics as before. On the laptop GPU: precision refresh
  ~18 s, candidate scoring ~10 s per PMPO update (480 candidates, M=5, chunk 16), ~13 s per
  update in total, peak 4.4 GB allocated / 5.7 GB reserved. At this rate the preset's
  ~100,600 PMPO updates would take roughly 15 days on this GPU.
- Raw LCG scores were ~1.4k-6.6k on average per epoch (max ~11k); the policy reward
  (raw / 100) ~14-66. The raw scale shifts strongly between epochs as the denoiser and h_D
  change; fixed scaling passes this through by design.
- Resume: a run killed after epoch 1 and resumed with `scripts/resume.sh` is bit-identical
  (training and test state) to an uninterrupted run. A second uninterrupted run diverged
  from both from env step 788: one rendered frame differed by one level with identical
  physics (cross-process GPU rendering variation, as in Step 6); the policy reads pixels,
  so the difference propagated into its actions, replay and training.
- Controller check (pipeline evidence only, not learning quality): 128 paired imagined
  rollouts (32 fixed prompts x 4 repeats, horizon 15, final denoiser, one fixed scorer,
  same world-model noise), raw LCG return: trained - initial = -9.8 +- 1.7 (SE; mean
  ~38.4k), trained - uniform = -714 +- 202. After 40 bounded PMPO updates the policy did not
  improve raw LCG return and uniform actions scored higher. No controller tuning was done.

### Optional advantage-weighted PMPO objective (pending review)

Commit `eee48cd`; implemented and verified locally 2026-10-20 (`lcg` env, 381 tests
passed); bounded comparison on kingplus0 GPU 1. Sign-only remains the default; neither the
controller nor the full-run schedule is validated by this.

- `agent.actor_critic.actor_objective`: `sign` (default, numerically unchanged) or
  `advantage_weighted`; `advantage_weight_cap` (default 5, an engineering choice, not a
  validated optimum). Weighted: within each sign group g (A >= 0 / A < 0), detached
  `w_i = min(cap, |A_i| / max(mean_g |A_j|, 1e-8))`; policy loss
  `(1-alpha) mean_neg(w log pi) - alpha mean_pos(w log pi)` (sample-count means, empty group
  -> 0, all-zero advantages -> 0, no renormalization after clipping). KL, entropy, critic
  targets, raw LCG and the /100 reward scaling are unchanged.
- The objective (mode, and cap when weighted) is saved in PMPO's checkpointed extra state;
  loading or resuming with a different objective or cap is refused. States saved before the
  setting count as sign-only (no format bump).
- New per-update metrics: per sign group |A| mean/max, weight mean/max/std and fraction at
  the cap, and actor relative distance from its initial snapshot (alongside the existing
  loss components, grad norms, value bias/correlation, entropy and KL).
- An earlier `abs_advantage` experiment (`stage3-weighted-beta-seed1`, archived) multiplies
  by raw |A| without group normalization or cap; it does not match this definition and was
  not reused beyond its group-sum structure.
- Bounded comparison (one seed, frozen model; `docs/changes/step6_lcg_weighted_pmpo_review.zip`):
  the canonical `k4-walker-100k` source was deleted from the server (its checkpoint was
  discarded by the archive policy), so both arms used the finished task-reward run
  `k4-walker-100k-entropy-stage3-seed1-gaussian-20261019` (seed 1, tanh-Gaussian policy;
  state.pt `7a1db37d...`, replay info.pt `c9475c60...`). Matched starts verified. Raw LCG
  return, 128 paired rollouts (mean +- SE), training scorer / separate bank:
  sign +1.55 / +1.74 at 100 and +2.66 / +2.72 at 200 vs update 0; weighted +3.03 / +3.55 and
  +4.38 / +4.94; weighted - sign at 200: +1.72 +- 0.21 / +2.22 +- 0.26. Versus uniform at 200:
  sign -1.26 +- 0.42 / -2.30 +- 0.52, weighted +0.45 +- 0.45 / -0.08 +- 0.60 (consistent with
  uniform within the uncertainty of this check; superiority and equivalence were not
  established). The weighted actor moved further (KL to prior 0.30 vs 0.12 at
  191-200) and one action dimension's entropy fell to 0.56; part of the gain may be a larger
  effective step rather than a better direction. 200 updates do not reach the update-500
  prior refresh; equal entropy settings do not show entropy is irrelevant; better imagined
  LCG return is not yet better real exploration.

### Bounded online pilot: weighted-PMPO LCG vs uniform (2026-10-20)

kingplus0 GPU 1; uncommitted code (base `be8a38d` + weighted objective with the review
fixes: duplicate `@torch.no_grad()` removed from `objective_diagnostics`, restored on
`diagnostics`). Review: `docs/changes/step6_lcg_online_pilot_review.zip`.

- Seed 0, fresh weights, no pretrained model; 5,000 uniform initial steps then 10 epochs x 500
  steps (uniform vs policy) = 10,000 environment steps each (20,000 control steps); 13,600
  denoiser updates each; LCG arm 2,000 weighted PMPO updates (200/epoch, prior refreshes at
  500/1,000/1,500) and 10 LCG refreshes; no test collection; milestones 5,500/7,500/10,000.
  Epoch 1's policy steps are collected before the first PMPO phase (collect-then-train).
- Initial weights and the initial 5,000-step replay (dynamics and frames) identical across
  arms. All losses/scores finite; placement verified on GPU 1 (compute and EGL rendering).
- Imagined LCG return, policy - uniform (that epoch's denoiser and scorer): epoch 1 -5.4 +- 2.8,
  epoch 5 +11.7 +- 3.6, epoch 10 +16.4 +- 1.6 (+6%): the policy follows its LCG objective.
- Common held-out denoiser loss (2,000 uniform steps, matched corruption): LCG arm worse by
  1.8% / 4.8% / 1.3% at 5.5k / 7.5k / 10k (the last within ~1 SE); held-out data favour the
  uniform arm. Diagnostic only.
- Runtime (arms concurrent): uniform 30 min, LCG 97 min; LCG epochs 2-10 7.25 min each
  (PMPO 2.0 s/update, 82% scoring).
- Functioning online exploration, not demonstrated learning quality; one seed. The existing
  agent-copy clean-up deletes `agent_epoch_00000.pt` when `save_agent_every=1` (initial
  weights were verified via observer digests instead).

### Pilot downstream adaptation (Walker walk, seed 0; 2026-10-20)

One-seed check on the online pilot's 10,000-step milestones (`uniform-pilot-run1-20261020`,
`lcg-pilot-run1-20261020`, copied; sources verified unchanged). Existing downstream trainer,
sign-only PMPO (repository default), identical fresh models (verified digests) and
settings: 5,000 reward-model updates, 2,000 PMPO updates, frozen milestone denoisers
(unchanged), milestone replay only. kingplus0 GPU 1, sequential arms, 27.5 min each.
Review: `docs/changes/step6_pilot_downstream_review.zip`.

- 30-seed deterministic evaluation (3001-3030, 500 steps): uniform 59.2 +- 43.4 (median 30.6),
  LCG 87.5 +- 55.7 (median 66.7); paired LCG - uniform +28.3 +- 7.1 (SE over evaluation
  seeds only), LCG higher on 23/30.
- Returns are bimodal (about 23-27 vs 115-158); many episodes near the floor (uniform 15/30,
  LCG 8/30 below 30); both far from a good policy. One exploration and one adaptation seed:
  suggestive, not conclusive.

### Full diffusion-world-model ensemble (MAX-inspired ensemble disagreement)

Commit `4a82dde`; implemented and verified locally 2026-10-20/21 (`lcg` env, 412 tests passed;
real Walker smoke on the local GPU because kingplus0 GPU 1 was busy). Review:
`docs/changes/step6_full_ensemble_review.zip`. Not an exact MAX reproduction or information gain.

- `intrinsic_reward=ensemble_disagreement` (scoring: `num_noise_samples`, `member_selection:
  uniform_per_transition`, `reward_scaling`/`fixed_divisor`, provisional 1.0) and `ensemble:`
  in trainer.yaml (`members: 5`, `bootstrap: true`; used only by this method, >= 2 required).
  Uniform/LCG keep one denoiser; sign-only stays the global default; preset
  `walker_ensemble_100k` uses the LCG pilot's weighted PMPO and `compile_wm=false`.
- Members: independent initializations (Agent.ensemble_members after member 0 = `denoiser`),
  own optimizer/scheduler (member 0 uses the existing ones) and own checkpointed RNG stream
  for sampling and corruption; same reward-free replay. Per-member schedule (K x updates).
- Bootstrap: episode-level Poisson(1) multiplicity = f(seed, member, episode id), fixed as the
  replay grows; BatchSampler episode probabilities x multiplicity, anchor uniform within the
  episode; all-zero -> ordinary weights for that batch (counted).
- Reward: shared initial noise across members (s_churn=0), on the sampler's outputs (float [-1, 1], values on the uint8 grid):
  mean over pixels/channels of the population variance across members; several noise samples
  are scored separately and averaged. Next observation: one member drawn uniformly per
  trajectory per transition (PMPO's imagination RNG), from the first noise sample; never the
  ensemble mean. Members frozen (eval) during PMPO.
- Checkpoint: members, ensemble training state (optimizers, schedulers, counters, multiplicities,
  member RNG), intrinsic settings; mismatched ensemble/intrinsic settings refused before any
  load; single-denoiser format-3 checkpoints still load (no format bump). Milestones:
  `denoiser.pt` = member 0, `denoiser_member_k.pt`, `identity.member_sha256`; downstream
  `initialization.member_index` (default 0) -> output `seed_<s>[_member_<k>]`, index and
  identity recorded.
- Smoke (500 + 3 x 100 steps, 5 members): two uninterrupted runs and a crash/resume run all
  bit-identical; 40 updates per member (200 total), 40 PMPO; raw disagreement 7.7e-5 to 2.3e-4
  per transition (max 3.6e-4) -> policy rewards ~1e-4 at divisor 1 (provisional, untuned);
  PMPO ~12.6 s/update on the laptop GPU (5 members, uncompiled), peak 4.5 GB; downstream
  members 0 and 4 loaded.
- Corrections 2026-10-21 (412 tests passed): downstream resume compares against the saved
  config with a missing `initialization.member_index` read as 0 (older checkpoints resume; a
  real member change is still refused; saved files untouched); `Agent.load()` refuses denoiser
  initialization when the snapshot or the agent has more than one denoiser (checked before any
  load; ensembles start fresh or resume via the full-state checkpoint). Timing: only the laptop
  measurement exists; K members multiply forward-prediction cost relative to one denoiser,
  LCG pays derivative-scoring costs, and their relative runtime is unmeasured on matched
  hardware. The local-GPU smoke was re-run with the corrected code (2026-10-21): exit 0, all
  three exact comparisons identical, same member weights/metrics as before, downstream members
  0 and 4 load; PMPO ~13.6-14.1 s/update on the laptop GPU.

### Ensemble vs LCG PMPO-update throughput on the RTX 5090 (2026-10-21)

Timing only, kingplus0 GPU 1, commit `4a82dde`, LCG online-pilot controller settings
(weighted PMPO cap 5, batch 32, horizon 15, 3 denoising steps, compile_wm off), prompts and
LCG precision from the weighted-LCG pilot's `steps_010000` milestone (read-only, unchanged).
Review: `docs/changes/step6_ensemble_5090_timing_review.zip`.

- Complete PMPO update (median of 20 after 5 warm-up): LCG 2.137 s (2.136 / 2.139); ensemble
  K = 2: 0.93 s, 3: 1.34, 4: 1.76 (0.82x), 5: 2.18 (1.02x), 6: 2.61 (1.22x), 8: 3.44 (1.61x);
  about 0.09 s + 0.42 s per member; p90 within 0.3% of the median; peak memory ~4.5 GB for
  every K (LCG 4.7 GB).
- Largest K with update time <= LCG: 4; within 10%: 5 (preset default unchanged, pending review).
- LCG refresh (N = 10,000, B = 320 x 3 probes): 8.2 s / 7.9 s per epoch, about 1% of an epoch's
  PMPO time at 400 updates.
- Equal update time is not equal total compute: K members train K x the denoiser updates and
  store K denoisers per checkpoint/milestone. Members 1..K-1 were freshly initialized (timing only).

### Ensemble settings moved into intrinsic_reward (2026-10-21)

Commit `ea3356d`; configuration refactor only (no change to reward scaling, bootstrap
behaviour, schedules or algorithms). Review: `docs/changes/step6_ensemble_config_review.zip`.

- `config/intrinsic_reward/ensemble_disagreement.yaml` now holds all of the method's
  settings: `num_members: 5`, `bootstrap: true` (fields of `EnsembleDisagreementConfig`,
  validated there) plus the unchanged scoring settings. The top-level `ensemble:` section was
  removed from `config/trainer.yaml` and the preset (no duplicated defaults). Override:
  `python src/main.py +experiment=walker_ensemble_100k intrinsic_reward.num_members=5`.
  Uniform and LCG still build one denoiser.
- Older saved configs: `ExplorationTrainer` reads `ensemble.members` / `ensemble.bootstrap` as
  `intrinsic_reward.num_members` / `bootstrap` in memory (`migrate_ensemble_layout`, right
  after `normalize_saved_config`), keeping the saved values; disagreeing layouts are refused;
  for LCG/uniform the unused old section is dropped; files are never rewritten. Checkpoint
  contents are unchanged, so equivalent old/new configs resume identically; real changes to
  members, bootstrap or scoring are still refused.
- Resume workflow: `scripts/resume.sh` runs the run folder's own copied `src/main.py` and
  saved `config/trainer.yaml`, so runs created before the refactor resume with their own code
  and older layout (pass overrides in that layout, e.g. `ensemble.members=...`); runs created
  now save and resume the new layout.
- Historical commands: the 2026-10-20/21 ensemble smokes and the RTX 5090 timing benchmark
  used the older layout (`ensemble.members=K`); their evidence is unchanged. The equivalent
  now is `intrinsic_reward.num_members=K`.

### Plan2Explore-style lightweight ensemble disagreement (2026-10-26)

Commit `cd78477` (base `ea3356d`); implemented and verified locally (`lcg` env). Review: `docs/changes/step6_plan2explore_review.zip`. Design and research basis: `docs/PLAN2EXPLORE_DESIGN.md` (Sections 1-12). An adaptation of Plan2Explore, not a reproduction; every numeric setting is a placeholder.

#### Files and commands

- `src/plan2explore.py`: `Plan2ExploreConfig`, `FeatureExtractor`, replay feature cache, `PredictorHead`, `Plan2Explore` (refresh, PMPO hook, checkpoint state), `predictor_disagreement`.
- `src/envs/world_model_env.py`: optional `feature_encoder`; tracks the frame and action preceding the sampler window; reports `plan2explore_feature` and `plan2explore_action` per step. Unused by every other method, whose behaviour is unchanged.
- `src/exploration_trainer.py`: method `plan2explore_disagreement`, refresh after the denoiser phase, logging, checkpoint validation, milestone provenance.
- `config/intrinsic_reward/plan2explore_disagreement.yaml`, `config/experiment/walker_plan2explore_100k.yaml`.

```bash
python src/main.py +experiment=walker_plan2explore_100k common.devices=0 common.seed=0 hydra.run.dir=outputs/<run>
python src/main.py +experiment=walker_plan2explore_100k intrinsic_reward.feature_layers=[unet.u_blocks.2]   # alternative
bash <run>/scripts/resume.sh <run>
```

The preset copies `walker_ensemble_100k` except for the method: same budget, collection, denoiser schedule, milestones, test diagnostics, models, and controller (advantage-weighted PMPO, cap 5, as in the LCG online pilot). `walker_lcg_100k` and the repository default remain sign-only. This mismatch with the LCG preset file is deliberate, inherited from the ensemble preset. `training.compile_wm=true` is rejected for this method.

#### Layers and shapes

Default UNet: depths [2,2,2,2], channels [64]*4, 64x64 frames.

| Selector | Output | Role |
| --- | --- | --- |
| `unet.d_blocks.2` | 64 x 16 x 16 | default (last two downsampling resolutions) |
| `unet.d_blocks.3` | 64 x 8 x 8 | default |
| `unet.mid_blocks` | 64 x 8 x 8 | deepest/middle alternative |
| `unet.u_blocks.2` | 64 x 32 x 32 | middle upsampling alternative (`unet.u_blocks.1`: 64 x 16 x 16) |

- Each map is adaptive-average-pooled to 4 x 4 and concatenated: 2048 features by default.
- Any `InnerModel` module name is accepted.
- A model-only shape check runs before the run folder is written, on a throwaway CPU denoiser built under a forked RNG. Unknown modules, and maps narrower than the pool size, are rejected.
- Activations are read through forward hooks registered for one call only and always removed.
- The pass stops once the last selected block has run (early exit, verified equal to a full pass).
- Every submodule's train/eval mode is restored afterwards. Ordinary denoising is unchanged (tested).

#### Extraction and context convention

- **Input:** `o_t + 0.1 * eps + 0.3 * eps_offset`, the denoiser's own corruption formula including its offset noise.
  - Pixel-noise sigma: `feature_sigma` = 0.1. `eps` has the frame's shape.
  - Offset-noise scale: the denoiser's `sigma_offset_noise` = 0.3. `eps_offset` is one value per channel, constant over pixels.
  - `(eps, eps_offset)` is one fixed probe drawn from `probe_seed` with a private generator, and its SHA-256 is recorded. The probe is shared by every frame and head and draws nothing from the global RNG.
- **Conditioning:** `compute_conditioners(0.1)`, the normal EDM preconditioning. Its effective noise-conditioning sigma is sqrt(0.1² + 0.3²) = sqrt(0.1) ≈ 0.316 (this sets `c_in`, `c_noise`, `c_skip` and `c_out`).
- **Not a near-clean probe:** the offset term (0.3) dominates. Lowering `feature_sigma` alone, even towards 0, leaves the effective sigma at ≥ 0.3 and the input shifted per channel by 0.3 * `eps_offset`. A near-clean extraction would need a different convention for the offset noise; that is a decision for the layer/noise diagnostic and is not implemented.
- **Context:** real frames `o_{t-4..t-1}` and actions `a_{t-4..t-1}` (n = 4). The candidate action `a_t` goes only to the heads.
- **Episode starts:** positions before the start are zero frames and zero actions. This is the left padding the denoiser itself trains on, applied identically in replay encoding and imagination. No context crosses episodes.
- **Replay transitions** `(o_t, a_t) -> o_{t+1}`:
  - `t+1` lies inside the episode; or
  - the last stored step, using either the episode's `final_observation` (time limit) or, for an unfinished episode at the cut, the collector's current observation (`ExplorationTrainer.successor_observations()`, also used by milestones).
  - Otherwise the last step is excluded and counted.
  - Eligible, excluded, successor and padded-context transition counts are logged.
  - The successor frame's context includes `a_t`: the shifted convention of the design's Section 5.
- **Imagination:**
  - The prompt's preceding step is read from the replay episode by segment id; it is zero at an episode start or for a padded window.
  - After each step, the frame and action leaving the sampler window become the new preceding pair.
  - The feature of `o_t` is computed before the next frame is sampled and draws no random numbers. Imagined frames, actions and RNG are identical with or without the encoder (tested).
- **Cached vs online features:** for the same replay frame they agree to 1e-6. The only difference is batch composition (CPU reduction order).

#### Refresh, fitting and scoring

After each denoiser phase, inside the PMPO branch of `train_agent`:

1. **Cache identity:** three hashes.
   - The denoiser weights.
   - The replay content: per episode in order, its length and a digest of its observations, actions, end/truncation boundaries and final observation; plus the collector's successor observations. Task rewards and other metadata are excluded.
     - Episode digests are memoized in memory, keyed by the episode version file's path and its write-time SHA-256 (immutable versions). Unchanged versions are not reloaded.
     - A rewritten file has a new checksum and is rehashed. Versions without a checksum are always rehashed.
   - The extraction settings.
2. **Encoding:** if no cache has exactly that identity, all frames are re-encoded in batches of `encode_batch_size` into a CPU tensor (never kept on the GPU). Each frame is encoded once and transitions index into it. A cache is installed only once complete, so a failed build leaves none.
3. **Standardization:** coordinate-wise mean and population std over the frames used by eligible training transitions, std clamped at `feature_std_floor` (1e-3). The number of coordinates below the floor is logged. The statistics stay fixed until the next refresh.
4. **Heads:** K = 5 freshly initialized MLPs.
   - Input `[z, a]`: the standardized feature and the action rescaled to [-1, 1].
   - Architecture 256-256 SiLU with a linear output.
   - Training: Adam at 1e-3, 500 updates of 256 transitions, MSE in standardized coordinates.
   - Only head parameters are optimized; the denoiser is only read, under `no_grad`.
5. **Seeds and bootstrap:**
   - Initialization and sampling: `SeedSequence([seed, 11, epoch, member])` (CPU generator only).
   - Bootstrap: the existing `episode_multiplicity` seeded by `SeedSequence([seed, 12, epoch])`, i.e. Poisson(1) per epoch, member and episode, fixed for the refresh.
   - Sampling probability is proportional to the episode's multiplicity and uniform within the episode.
   - Empty support falls back to uniform sampling, counted and logged.
   - With bootstrap off, members share uniform support but keep separate initializations and samples.
6. **Freezing:** heads go to eval mode with `requires_grad=False`, and the denoiser to eval mode for PMPO. PMPO's hook is `Plan2Explore.__call__`.

**Reward:**
- Raw: for each imagined step, the mean over features of `sum_k (pred_k - mean)^2 / (K - 1)` from the heads' predictions for `(z_t, a_t)`.
- Policy reward: raw / `fixed_divisor` (1.0, provisional); `none` also available.
- PMPO's loss, horizon, bootstrap values and history are unchanged.

#### Checkpoints, resume and milestones

- **Saved** in `checkpoints/state.pt["plan2explore"]`:
  - settings: config, extractor settings with layer shapes, probe hash and feature dimension, action bounds;
  - seed, last heads, mean/std;
  - refresh and fit counters, empty-bootstrap counts;
  - last cache identity and the seed specification.
- **Not saved:** optimizer state and the cache. Fitting restarts from scratch at each refresh and never spans a checkpoint. The cache is regenerated deterministically; a refit from a final checkpoint reproduced the saved heads exactly on the GPU.
- **Resume refusal:** any changed setting is refused before any weights load, including `encode_batch_size`, which changes features at the 1e-7 level. Reward scaling is checked first.
- **Compatibility:** checkpoints without the key (all older ones) load with `plan2explore=None`. Other methods refuse a Plan2Explore checkpoint, and vice versa. No format bump.
- **Milestones:** a single `denoiser.pt`, the replay, and a manifest with `plan2explore` provenance (members, layers, feature dimension, sigma, pool, probe hash, last refresh). The heads are not included, and the downstream trainer loads these milestones unchanged.

#### Logging

- **Per refresh:**
  - `plan2explore/*`: transition counts, feature dimension, head parameters, cache reuse and size, feature std min/mean/max, near-constant coordinates, per-member first/last losses, fit updates, bootstrap zero fraction and fallbacks.
  - `timing/p2e_identity_seconds`, `p2e_feature_preparation_seconds`, `p2e_fit_seconds`, `p2e_refresh_gpu_peak_allocated_gb`.
- **Per PMPO update:** `actor_critic/train/p2e_raw_reward_*` and `p2e_policy_reward_*`.
- **Per epoch:** `timing/p2e_refresh_seconds`, `pmpo_update_seconds`, `p2e_imagination_feature_seconds`, `p2e_scoring_seconds`, GPU peaks.

#### Verification and cost (local RTX 5060 Laptop GPU, 8 GB)

- **Tests:** see "Corrections (2026-10-26)" below for the current counts (originally 42 focused tests; full suite 460 passed).
- **Smoke** (`outputs/step6-p2e/smoke/`; pre-correction evidence, run on the code before the replay-identity fix below): preset defaults, 500 + 3 x 100 steps, 20/10/10 denoiser and PMPO updates, test every epoch.
  - Totals: 800 training and 2,000 test environment steps; 40 denoiser and 40 PMPO updates; 3 refreshes of 5 x 500 head updates. All logged values finite.
  - A separate-process refit from the final checkpoint reproduced the saved heads and standardization exactly. It also reproduced the cache identity under that code's (pre-correction) identity function.
  - `full2` versus the run killed after epoch 1 and resumed with `scripts/resume.sh`: the exact comparison passed.
  - Both comparisons involving `full` (`full` vs resumed, `full` vs `full2`) failed.
  - The smoke driver's overall exit status was 1, because of those two failed comparisons.
  - The first difference between `full` and `full2` is at environment step 789 (episode 1, step 289). The frame (4 values, 1 level), the action and the task-reward labels first differ at that same step; everything earlier is identical, as are the agent snapshots after epochs 1 and 2. Consistent with previously observed GPU rendering variation; the cause was not conclusively isolated. The log does not establish identical simulator state immediately before that frame was rendered.
  - The downstream trainer loaded `steps_000700` (single `denoiser.pt`).
  - Untuned magnitudes: raw disagreement 0.35-1.04 mean per update (max 3.8). Head loss fell from ~0.7-1.5 to ~0.03-0.07 on 600-800 transitions, which is memorization at this size.
- **Cost on the laptop:**
  - Complete PMPO update: ~3.1-3.4 s, of which imagination feature extraction is ~0.25 s and head scoring ~2 ms.
  - Refresh at 600-800 transitions: ~0.5 s encoding plus ~6 s fitting (2,500 head updates).
  - At 10,000 transitions: encoding 6.5 s (extrapolated to ~65 s and a ~780 MB CPU cache at 100k), fitting 9.5 s.
  - Early-exit extraction costs ~0.31 of a full denoiser pass.
  - Peak GPU allocation: 4.4 GB during PMPO, 2.3 GB during a refresh.
- **Not validated:** the layer and sigma choice, predictor capacity and budget, the divisor, and whether this exploration improves downstream learning. The design's Section 8 feature diagnostic, an online pilot and the full run have not been done.

#### Bug found and fixed during testing

`torch.manual_seed` inside `fork_rng(devices=[])` also reseeded the global CUDA generator, which that context does not restore: head initialization would have reset the denoiser's CUDA noise stream on GPU runs. Head initialization now seeds only the CPU generator (test: refresh leaves every global RNG state unchanged).

#### Corrections (2026-10-26, after review)

- **Replay cache identity.** The original `replay_identity()` hashed episode filenames and lengths plus the successors, so different replay contents could share an identity and reuse stale features.
  - It now hashes each episode's reward-free content: observations, actions, end/truncation boundaries and final observation. Collector successors are included as before.
  - Episode digests are memoized by version file and write-time checksum. Rewards, task rewards and other metadata are excluded.
- **Regression tests:**
  - same filenames and lengths with different observations, actions or final observation, and changed cut successors, all rebuild;
  - unchanged content reuses the cache;
  - a replay rewritten in the same directory is rehashed;
  - unchanged versions are not reloaded;
  - a task-reward-only change leaves identity, features, head updates and intrinsic rewards unchanged.
  - The content test fails against the original function.
- **Divergence wording.** The smoke's `full` divergence is "consistent with previously observed GPU rendering variation; the cause was not conclusively isolated" (see Verification above). The earlier wording, "identical physics before it", overstated what the first-divergence log shows.
- **Noise convention.** Documented above: pixel sigma 0.1, offset scale 0.3, effective conditioning sigma ≈ 0.316. Extraction behaviour is unchanged.
- **Validation (corrected code):** focused cache/isolation/resume tests 9 passed; all Plan2Explore tests 44 passed; affected regressions 167 passed; full suite 462 passed (all exit 0).
- **Smoke evidence.** The smoke was not re-run. Its evidence predates the identity fix, which changes only when a cache may be reused within a process. In a run where every refresh follows a denoiser update, the denoiser hash already forces a rebuild, so features, fitting and rewards are the same. The `replay_sha256` value recorded in checkpoints (`cache_identity`, metadata only, not compared on resume) is now computed differently.

### Laplace-sampled diffusion ensemble (2026-10-26)

Not yet committed (base `cd78477`); implemented and verified locally (`lcg` env).
- Review: `docs/changes/step6_laplace_ensemble_review.zip`.
- Design: `docs/LAPLACE_ENSEMBLE_DESIGN.md`.
- Precision audit: `docs/LAPLACE_PRECISION_AUDIT.md`.

This is a Laplace-inspired, curvature-based subnetwork ensemble: not a calibrated posterior, not independently trained models, and not an information-gain reward.

#### Files and commands

- `src/laplace.py`:
  - `LaplaceConfig`;
  - `LaplaceEnsemble` (refresh, sample bank, member predictions, checkpoint state);
  - `shared_prefix_first_evaluation`;
  - `perturbation_scale`;
  - `LaplaceScorer` (the full ensemble's scorer, prefix `laplace`).
- `src/lcg/lifecycle.py`: new `lcg_historical_precision`, the exact precision call LCG makes. LCG now uses it too, unchanged.
- `src/models/diffusion/denoiser.py`, `diffusion_sampler.py`: optional `params`, a functional substitution of InnerModel parameter values for one call or one whole chain. The sampler also gains an optional precomputed `first_denoised`. Defaults are unchanged.
- `src/envs/world_model_env.py`: optional `member_predictor`. It uses the same shared noise, population-variance disagreement and uniform member selection as the full ensemble, whose code path and RNG draws are unchanged.
- `src/exploration_trainer.py`: method `laplace_disagreement`, refresh, logging, checkpoint validation, milestone provenance.
- `config/intrinsic_reward/laplace_disagreement.yaml`, `config/experiment/walker_laplace_100k.yaml`.

```bash
python src/main.py +experiment=walker_laplace_100k common.devices=0 common.seed=0 hydra.run.dir=outputs/<run>
python src/main.py +experiment=walker_laplace_100k intrinsic_reward.num_members=5 intrinsic_reward.posterior_temperature=1.0
bash <run>/scripts/resume.sh <run>
```

The preset matches `walker_ensemble_100k` in budget, schedules, milestones, diagnostics, models and controller (advantage-weighted PMPO, cap 5). It takes R1 with its guards (10 tensors, 217,664 scalars) and the historical settings (320 / 3 / 1e-4, beta 1) from `walker_lcg_100k`. The maintained LCG preset is still sign-only, so controlled LCG comparisons must align the controller explicitly. `compile_wm=true` and `s_churn != 0` are rejected.

#### Precision and sampling

- **Precision:** see the audit. LCG's diagonal `h = damping + beta * (N / B) * sum_i mean_m v_im^2`, with `v = grad sqrt(2) xi^T F_theta`, i.e. a Monte Carlo diagonal of the summed-pixel squared-error Gauss-Newton matrix.
  - The replay-size factor N/B is already inside it.
  - Damping 1e-4 is applied once.
  - It is reused once and unchanged, with no candidate scoring.
- **Gaussian:** `theta_S[k] = theta_hat_S + sqrt(T) * eps_k / sqrt(h)`, with T = 1 and K = 5 independent samples; the central model is not a member.
  - `eps` comes from a private CPU generator seeded by `SeedSequence([seed, 21, epoch])`. No global or CUDA RNG is touched (tested).
  - Invalid precision raises an error, with no fallback covariance.
  - No clamp or extra scale.
- **Storage:** one base denoiser plus a (K, d_S) perturbation bank (R1: 5 x 217,664 floats). Member values are base + perturbation, built at prediction time and passed through `torch.func.functional_call`, so the module's Parameters are never written. Weights, buffers, Parameter identity and modes survive an exception (tested).
- **Lifetime:** drawn after each denoiser phase, frozen for that epoch's PMPO phase. The base denoiser is in eval mode during PMPO and is only read.

#### Prediction, reward and reuse

- **Per imagined transition:** one initial noise tensor shared by all members. Each member runs the full Euler chain (3 steps) with its sampled parameters fixed at every evaluation (tested against an independent copy holding those weights).
- **Reward:** `ensemble_disagreement`, the population variance (divisor K) averaged over pixels, applied to the sampler outputs.
- **Next observation:** a member drawn uniformly per trajectory and transition, from the first noise sample.
- **Policy reward:** raw / 1.0.
- **Scored stage:** the same as the full ensemble's, the sampler's final output: float tensors whose values lie on the uint8 grid (within 7.6e-6 levels), because the last Euler step returns the denoised value that `wrap_model_output` quantizes. The earlier "unquantized" / "before uint8 quantization" wording in the full ensemble's docstring and config comment and in the design was corrected on 2026-10-27; neither method's behaviour changed.
- **Shared prefix (R1 and any selection inside the last decoder level, norm_out or conv_out):**
  - At the first evaluation the input is identical for all members. Conditioning, conv_in, the down path, the middle blocks, decoder levels 0..L-1 and the last upsampling run once, with every skip tensor kept. The last decoder level, norm_out and conv_out run per member.
  - Later evaluations are full per-member forwards, since member images differ after the first update. A test shows that reusing the prefix there would give a different (wrong) result.
  - Other selections use the full-forward path, with a log message.
- **Agreement:** the reference and optimized paths agree exactly on CPU (tests) and on the GPU (benchmark: max difference 0.0 for predictions and rewards).

#### Checkpoints and milestones

- **Saved** in `state.pt["laplace"]`: settings (config, selected names, order, shapes and dim, the precision source and representation, the sampling formula), seed, precision, the actual perturbation bank, refresh counters and epoch, and the bank seed.
- **Resume:** changed members, temperature, selection, precision settings, `shared_prefix`, noise samples or scaling are refused before any weights load. Older checkpoints load with no Laplace state. Cross-method checkpoints are refused.
- **Milestones:** a single unperturbed `denoiser.pt` plus `laplace` provenance. The downstream trainer is unchanged and loads them (tested; smoke).

#### Verification

- **Tests:** 27 focused tests, covering:
  - precision bit-identical to LCG's;
  - damping applied once and N/B scale;
  - covariance factor and the empirical covariance of 4,000 draws;
  - selected-only changes, in order;
  - fixed-parameter chains with shared noise;
  - population variance;
  - prefix vs reference;
  - the unsupported-selection fallback;
  - exception safety;
  - imagination scoring and selection;
  - task-reward independence;
  - crash/resume identity plus 10 refused setting changes;
  - checkpoint compatibility;
  - milestones and downstream loading.
- **Mutation check:** using covariance `T/h` instead of `sqrt(T/h)` fails the sampling test.
- **Regressions:** affected suites (exploration, downstream, resume, checkpoint, lcg, models) 449 passed; full suite 489 passed; all exit 0.
- **Smoke** (`outputs/step6-laplace/smoke/`): local RTX 5060 Laptop GPU; preset defaults; 500 uniform + 3 x 100 policy steps; denoiser and PMPO 20/10/10 updates; test every epoch.
  - Two uninterrupted runs, and one hard-killed after epoch 1 and resumed with `scripts/resume.sh`.
  - Counts: 800 training and 2,000 test environment steps; 40 denoiser and 40 PMPO updates; 3 refreshes. All logged values finite.
  - Comparisons as re-run with the strengthened script on 2026-10-27; the smoke itself was not re-run. They now also cover episode metadata (incl. `final_observation`), task names and order, the complete task RNG state, and list lengths.
  - `full` vs `full2`: the exact comparison passed (identical, including test data).
  - Both comparisons involving the resumed run failed, and the smoke driver's overall exit status was 1.
    - The only differing entries are the final test episode's (`test_dataset[0]`) frames, actions, rewards, task-reward labels and `final_observation`. The last of these was not checked by the original comparison.
    - Agent, optimizers, schedulers, the Laplace precision and bank, counters, RNG and test RNG, the training replay including episode metadata, and the collector and env state (physics fields and the complete task RNG state) are identical.
    - First difference: test step 260, where the frame (4 values, 1 level) and the action differ; everything earlier is identical.
    - Consistent with previously observed GPU rendering variation; the cause was not conclusively isolated.
  - A separate-process refresh from the final checkpoint reproduced the precision and the sample bank exactly (both runs).
  - Downstream loaded `steps_000700` (the single base denoiser), exit 0.
- **Magnitudes at T = 1** (reported, not tuned):
  - precision min ~1e-4 (damping only), median 2.1e-4 / 1.5e-3 / 2.1e-2, max 12.7 / 63 / 462 (epochs 1-3);
  - per-coordinate std median 69 / 26 / 7.0, max ~100;
  - perturbation norm ~2,520 / 1,940 / 1,160x the selected weights' norm (13.1).
  - **Members' predictions are blank.** At T = 1 every member's completed prediction is a nearly uniform grey frame with the walker gone (image sheets, frozen-model diagnostic below). The members agree with each other but not with the base: RMSE 0.14-0.18 against the base, within-ensemble spread RMS 0.017. The earlier note that the images were "not degenerate" was wrong: it was inferred from RMSE and saturation without looking at images. Corrected 2026-10-27.
  - In the smoke's PMPO phase every imagined next observation came from such a member.
  - Raw reward per PMPO update: mean 1.3e-5 to 2.2e-4, max up to 6.5e-4; policy reward identical (divisor 1).
- **Cost on the laptop:**
  - Complete PMPO update ~11.2-11.5 s, of which member predictions are ~95% (0.72 s per imagined step at batch 32 vs 0.156 s for one base chain).
  - The shared prefix saves 12.6% of member prediction time (0.713 vs 0.815 s; matched inputs).
  - Refresh 15-20 s, almost all precision (320 transitions x 3 backward probes). Sampling takes 5 ms.
  - Peak GPU 4.3 GB during PMPO, 0.18-0.24 GB during a refresh.
  - The earlier laptop 5-member full-ensemble smoke measured ~13.6-14.1 s per update (not matched).

#### Not established

- Calibration of the uncertainty.
- Whether the damping-dominated perturbation scale is appropriate.
- Reward calibration (raw values ~1e-4).
- Exploration or downstream benefit.
- A temperature, damping or subset choice.
- No pilot or full run has been done.

#### Review corrections and frozen-model diagnostic (2026-10-27)

No training, data collection, preset or hyperparameter change; the smoke outputs were only read.

**Validation (2026-10-27).** Focused tests (Laplace, ensemble, config layout) 58 passed; affected (exploration, models, lcg) 400 passed; full suite 489 passed; all exit 0. No new training smoke (training behaviour unchanged).

**Scoring wording.** Corrected in the `ensemble_disagreement` docstring, the full ensemble's config comment, the design (Section 7) and these notes; behaviour unchanged.

**Strengthened comparison.**
- What it adds: episode metadata (every info entry, including `final_observation`); dataset and episode task names and their order; every task attribute, including the RNG's complete state (algorithm, key array, position, cached-Gaussian flag and value); checks of env-list and episode-list lengths before pairing. MuJoCo state is still compared through explicit physics fields, never raw bytes, and there are no tolerances.
- Environment seed: none is recorded in episodes or checkpoints. Reset seeds come from the global Python RNG, which is compared.
- Self-check on copies of `full` (the original files were only read):
  - an unchanged copy compares as IDENTICAL;
  - each mutant (one `final_observation` value, the task RNG position, its cached Gaussian, an extra env entry) is flagged at exactly that entry by the strengthened script;
  - the original script reported all four mutants as IDENTICAL.
- My first strengthened draft also compared dict key order and reported spurious "keys differ" entries. StateDictMixin builds its dicts from sets, whose order varies between processes, so the script compares key sets instead. Those logs were replaced.
- Results on the smoke outputs: see Verification above.

**Frozen-model diagnostic.**
- Script and outputs: `outputs/step6-laplace/diagnostic/`, 50 s on the laptop GPU.
- Models: the final checkpoint (epoch 3, saved bank) and the epoch-2 milestone `steps_000700` (bank recomputed by the production refresh, which is deterministic). These are very lightly trained smoke models (40 / 30 denoiser updates).
- Design: 5 members, 32 fixed replay prompts, 2 fixed noise draws. The same precision and sample directions are used at every temperature: `perturbation(T) = sqrt(T) * bank(T = 1)`, for T = 1, 1e-4, 1e-6, 1e-8, 1e-10, plus a zero-perturbation control.

Results, final checkpoint / epoch-2 milestone (both noise draws agree to 3 digits):

| T | relative perturbation | mean shift RMSE | spread RMS | common share of member-to-base MSD | raw reward mean |
| --- | --- | --- | --- | --- | --- |
| 1 | 1,160 / 1,930 | 0.162 / 0.079 | 0.0166 / 0.0084 | 99.0% / 98.9% | 2.7e-4 / 7.1e-5 |
| 1e-4 | 11.6 / 19.3 | 0.0046 / 0.012 | 0.0027 / 0.0054 | 74% / 84% | 7.2e-6 / 2.9e-5 |
| 1e-6 | 1.16 / 1.93 | 4.1e-4 / 5.9e-4 | 5.1e-4 / 5.2e-4 | 40% / 57% | 2.6e-7 / 2.7e-7 |
| 1e-8 | 0.12 / 0.19 | 1.5e-4 / 1.3e-4 | 1.7e-4 / 1.6e-4 | 43% / 41% | 2.9e-8 / 2.5e-8 |
| 1e-10 | 0.012 / 0.019 | 9.8e-5 / 6.7e-5 | 8.5e-5 / 6.4e-5 | 57% / 53% | 7.1e-9 / 4.1e-9 |

- **Reviewer's figures reproduced.** At T = 1 on the final checkpoint, with bench.json's prompts and noise: mean shift 0.162, spread 0.0166, common shift 99%. The decomposition MSD = shift² + spread² holds to ≤ 3e-9.
- **Image sheets.** At T = 1, all members are blank grey frames on both models. At T = 1e-4 they are visually indistinguishable from the base.
- **Common shift alone** is not evidence that the reward is unusable. Here, though, it reflects members that have all lost the image content.
- **Precision near the damping floor:**
  - h < 2·damping (data curvature below the damping): 6.8% / 29.8% of coordinates;
  - h < 1.1·damping: 0.7% / 8.3%.
- **Per selected tensor at T = 1:** the largest perturbations hit the AdaGroupNorm projections (`norm1.linear.weight` ~12,600 / 20,700, `norm2.linear.weight` ~8,100 / 13,300) and `conv1.weight` (~1,600 / 5,300). Relative to each tensor's norm the ratios are ~9-6,700; `conv2.weight` has the largest ratio, since its own norm is only 0.07-0.12.
- **First-order check** (raw network output at the first denoising input, 8 prompts):
  - At T = 1 and 1e-4, the actual change F(θ+δ)−F(θ) is ~0.9-1.2x and 0.03-0.26x the output's norm, but bears little relation to the JVP J·δ: relative error ~1.0-1.5, norm ratio actual/JVP up to 24. The samples lie far outside the local linear regime the curvature describes.
  - At T = 1e-8 the error falls to ~0.2-0.5 (closest to linear). At T = 1e-6 it is 0.3-0.5 on the final checkpoint and 0.4-1.2 on the milestone.
  - At T = 1e-10 it rises again (~0.8-1.0), where the changes approach float32/TF32 numerical resolution.
  - TF32 is on for matmuls (trainer setting) and cuDNN convolutions (PyTorch default), so small-T values are near the numerical floor.
- **Rewards.** Exactly-zero rewards: none at any T > 0. Rewards still occur at T = 1e-10, mainly because members' outputs fall on different uint8 levels near rounding boundaries. The pre-quantization spread (diagnostic only, not the reward) is ~2/3 of the quantized one at small T and about equal at T = 1.
- **Finiteness and saturation.** All predictions finite, no saturated values at any temperature.
- **Control.** Zero-perturbation members are exactly identical to the base, on GPU and CPU (`control_floor_*.json`). The control's residual statistics (~3e-9 RMS, rewards ~1e-17) are float32 rounding from averaging five identical values; the production function gives the same for identical members.
- **Instrumentation.** The instrumented trajectories equal the production predictions exactly.

**Interpretation.**
- Software: correct as specified, tests pass, and training state matched across the resumed and uninterrupted runs.
- Sampling scale: at the production temperature T = 1, the literal Gaussian from LCG's precision destroys the prediction, and the reward measures disagreement between blank frames. This is a sampling-scale problem, not a software defect.
- Exploration performance: not assessed.
- Nothing was tuned. Temperature, damping, subset, divisor and presets are unchanged. These are smoke models, and the precision grows with replay and training, so the scale at realistic maturity is unknown.

#### Trained-model scale check (2026-10-27): blocked, source not available locally

*Historical: the source was fetched and the diagnostic run later the same day (next section).*

Review: `docs/changes/step6_laplace_mature_scale_review.zip`. No training, data collection, preset or hyperparameter change. kingplus0 was not accessed.

**Comparison fix.**
- The env-entry key check in `compare_laplace.py` now compares key sets, not key order.
- Self-check (copies of the smoke `full` run):
  - reordered env keys pass; the previous script wrongly flagged them;
  - a missing key and an extra key are flagged with their names;
  - the earlier mutants (final observation, RNG position, cached Gaussian, env count) are still flagged.
- Re-run on the existing smoke outputs, with unchanged results:
  - `full` vs `full2` IDENTICAL;
  - both comparisons with the resumed run differ only in the final test episode (act, rew, task_rew, obs, `info.final_observation`);
  - the original smoke driver's exit status stays 1.

**Source not available.**
- The intended model is the final milestone `steps_010000` of `uniform-pilot-run1-20261020`. It was selected by its collection budget, not task performance: 10,000 steps, 13,600 denoiser updates, 0 PMPO updates, uniform collection.
- No local copy exists. Searched: `outputs/`, `LCG-server-archive-20261020` (task-reward runs only, not substituted), and the user folder.
- Only the pilot's downstream outputs were fetched earlier (`outputs/step6-pilot-downstream/server/`). They record:
  - the milestone identity: denoiser `11c4da27…`, dataset `986f7c97…`, epoch 10;
  - SHA-256 checksums of the exact 25 files the diagnostic needs: `config/trainer.yaml`, `milestones/steps_010000/{denoiser.pt,dataset.pt,manifest.yaml}`, 20 episode files of 500 steps, and `dataset/train/info.pt`.
  - The copied run's checksums equal the source's.
  - The server path was `/home/bigking/scfeng/LCG-rf-pilot-downstream-20261020/runs/uniform` (a byte-identical copy of the pilot run).

**Prepared, not run on the trained model:** `diagnose_mature.py`.
- Gate: it refuses to start unless all 25 files match their SHA-256 and the manifest matches the expected identity, uniform collection and 0 PMPO updates.
- It uses the run's saved model, corruption, conditioning and sampler settings, and leaves the source read-only (checksummed before and after).
- It computes the precision once and keeps directions paired across T = 1…1e-10 plus the control. It produces image sheets with the replay context, 6-step fixed-action imagined sequences per member, and a matched strict-FP32 (TF32 off) subset with rank comparisons.
- Script check on the smoke milestone `steps_000700`, labelled as not a result: it ran in 44 s, reproduced the earlier smoke diagnostic exactly, and left the milestone files unchanged.
- On the smoke model even the base prediction is a faint grey ghost of the scene shown in the replay context. These models are too immature to answer the scale question.
- The gate refused the absent pilot source (25 of 25 files missing), as intended.
- Fix: `np.corrcoef` loads a second OpenMP runtime next to torch's here (fatal OMP error #15), so ranks are correlated with torch; `KMP_DUPLICATE_LIB_OK` was not used.

**Questions 1-5** (does T = 1 still destroy predictions on a trained model; is there a usable range; first-order agreement there; quantization/arithmetic dominance; difference from the smoke models): **not answered**. That requires the trained milestone.

**Next step.** Fetch the 25 files (needs authorization for kingplus0) into `outputs/step6-laplace-mature/source/uniform-pilot-run1-20261020/` and run the prepared command (well under 30 GPU minutes). If the scale is still unusable, the next principled change to investigate is the prior: coordinates the data barely constrain currently get only the 1e-4 damping as prior precision (std 100). An explicit prior precision justified independently of the outcome (e.g. from the parameter initialization scale or the optimizer's weight decay) would bound them. Not implemented; no search.

#### Trained-model scale check: results (2026-10-27)

Review: `docs/changes/step6_laplace_mature_scale_review.zip`. Diagnostic only: no training, data collection, policy or pilot, and no change to temperature, damping, prior, subset, divisor, presets or training.

**Source.**
- kingplus0 access was authorized for read-only fetching. The original run, `/home/bigking/scfeng/LCG-rf-lcg-online-pilot-20261020/outputs/uniform-pilot-run1-20261020`, and its downstream copy both matched the 25-file manifest exactly (server-side `sha256sum`).
- The 25 files were streamed from the original (tar over ssh; nothing written on the server) into `outputs/step6-laplace-mature/source/uniform-pilot-run1-20261020/`, verified (25/25), and made read-only.
- The diagnostic's gate verified, before computing:
  - checksums;
  - identity: denoiser `11c4da27…`, dataset `986f7c97…`, epoch 10;
  - 10,000 steps, 13,600 denoiser updates, 0 PMPO updates, uniform collection.
- Saved settings: the default denoiser (depths [2,2,2,2], channels [64]×4, sigma_data 0.5, offset noise 0.3, 4 conditioning frames); the sampler (3 Euler steps, s_churn 0); the training sigma distribution (log-normal −0.4 / 1.2, clipped to 2e-3 to 20).
- Afterwards the local files (25/25) and the server source were unchanged.
- Run: the prepared `diagnose_mature.py`, unchanged, 67 s on the RTX 5060 Laptop GPU (precision 29 s).

**Precision (R1, 320 / 3 / 1e-4 / 1).**
- Min 4.3, p01 46, median 7,884, p99 7.6e5, max 4.7e6.
- No coordinate is near the damping floor (0% below 2·damping): the data term dominates everywhere.
- On the smoke models the median was 1.5e-3 to 2.1e-2, with 7-30% of coordinates near the floor.

**Paired temperatures** (32 prompts × 2 noise draws; values per draw agree to ~1%):

| T | relative perturbation | shift RMSE | spread RMS | common share | raw reward mean | linearization error (TF32 / strict FP32, 8 prompts) |
| --- | --- | --- | --- | --- | --- | --- |
| control | 0 | 2e-8 | 2e-8 | – | 4e-16 | – |
| 1 | 1.43 | 0.0108 | 0.0171 | 28% | 2.9e-4 | 0.39-0.49 / 0.39-0.49 |
| 1e-4 | 0.0143 | 6.9e-4 | 7.1e-4 | 49% | 5.0e-7 | 0.30-0.35 / 0.004-0.005 |
| 1e-6 | 1.4e-3 | 4.7e-4 | 4.0e-4 | 57% | 1.6e-7 | 0.91-0.95 / 0.02 |
| 1e-8 | 1.4e-4 | 4.4e-4 | 3.6e-4 | 59% | 1.3e-7 | ~1.0 / 0.15-0.18 |
| 1e-10 | 1.4e-5 | 4.4e-4 | 3.6e-4 | 60% | 1.3e-7 | 1.0 / 0.66-0.73 |

- **Decomposition:** MSD = shift² + spread² holds to ≤ 3e-11 at every temperature.
- **Rewards:** none exactly zero. Predictions finite. 0.11-0.16% saturated values, the same in the base (bright pixels of real frames).
- **T = 1, raw output:** actual change ~2.6-3.2% of the raw output norm, actual/JVP norm ratio 1.00-1.04, i.e. moderately nonlinear in direction but matching in magnitude.
- **T = 1, per tensor:** the AdaGroupNorm scale projections receive perturbations larger than their weights (`norm1.linear.weight` 1.74x, `norm2.linear.weight` 1.49x). The others are 0.03-0.46x.
- **T = 1, pre-quantization (diagnostic only):** reward 2.86e-4 vs 2.90e-4 quantized; shift and spread within ~2%.
- **T = 1e-4 and below, quantization:** the quantized reward is ~7x (1e-4), ~14x (1e-6) and ~19x (≤ 1e-8) the pre-quantization one.

**Arithmetic** (matched 8-prompt subset; same weights, precision, directions, prompts and noise; TF32 on vs off for matmuls and cuDNN):
- Control: identical structure in both (reward ~4e-16).
- T = 1: same reward ranking (Spearman 1.0), rewards within 0.35%, identical linearization errors.
- T = 1e-4: Spearman 0.69, rewards up to 12% apart.
- T = 1e-6: Spearman 0.40, rewards up to 70% apart.
- T ≤ 1e-8: rankings unrelated (Spearman ≤ 0); TF32 rewards ~20-90x the FP32 ones.
- Below T ≈ 1e-4 the TF32 production arithmetic, not the sampled parameters, sets the reward.

**Images** (inspected directly).
- One-step predictions at every temperature, T = 1 included, preserve the scene: walker, floor, lighting, continuing the replay context. Member differences are not visible at sheet resolution.
- Six-step fixed-action sequences: the base and all members stay coherent at every temperature, with no collapse.
- At T = 1 the members show small pose differences, visible from about step 3. Member-to-base RMSE grows from 0.019 to 0.087 over 6 steps, with spread 0.016 to 0.063, while the base frame changes 0.08-0.16 per step. At T ≤ 1e-4 member divergence stays ≤ 0.003.
- Content preservation is established. Whether the T = 1 variation is *useful* uncertainty (larger where the model is wrong) is not.

**Answers.**
1. **Does T = 1 destroy predictions on the trained model?** No. Scene content is preserved in one-step predictions and in 6-step sequences.
2. **Is there a range with useful dynamics and measurable disagreement?** Of the tested values, only T = 1 gives disagreement that is clearly above the arithmetic and quantization floors while preserving dynamics. T = 1e-4 preserves dynamics, but its reward is quantization-inflated and arithmetic-sensitive; T ≤ 1e-6 is arithmetic-dominated. Values between 1e-4 and 1 were not tested.
3. **First-order agreement at T = 1?** Moderate: direction error 0.39-0.49, magnitude ratio ~1.0, identical in strict FP32. Strict FP32 shows near-linear behaviour at 1e-4 (0.005). This is a diagnostic, not calibration.
4. **Are T = 1 rewards dominated by quantization or arithmetic?** No: ranking unchanged without TF32, and pre- vs post-quantization within ~1-2%. At T ≤ 1e-4, yes.
5. **Different from the smoke models?** Materially.
   - The smoke models' precision was damping-dominated: perturbations 1,160-1,930x the weights, and blank frames at T = 1.
   - On the trained model the data term dominates, perturbations are 1.4x the weights, and predictions are preserved.
   - The scale problem is a property of barely trained models with small replay (the smoke's 20-40 denoiser updates).
   - *Corrected 2026-10-27:* this note originally claimed the collapse would recur in the first epochs of a 100k run. It does not follow. Under the actual schedule the first refresh comes after 10,000 denoiser updates on 5,500 steps, and the first-refresh check below found T = 1 usable there.

**Separation.**
- Implementation correctness: unchanged (tests pass; comparison corrected).
- Uncertainty calibration: open. T = 1 on a mature model gives non-degenerate, arithmetic-robust disagreement, but whether it tracks model error is untested.
- Exploration performance: not assessed.

#### First-refresh check (2026-10-27)

Review: `docs/changes/step6_laplace_first_refresh_review.zip`. A configuration check on a frozen model, with T = 1 and the unchanged-base control only. No temperature sweep, error-correlation study, training, data collection or pilot. The disagreement–prediction-error association is deferred to `lcg_undersample_diagnostic`.

**Where the first refresh happens.**
- The pilot's saved schedule: 5,000 uniform steps, then epochs of 500 steps; the denoiser trains 10,000 updates in epoch 1 (`steps_first_epoch`); PMPO starts in epoch 1.
- The first intrinsic-scorer refresh therefore follows 10,000 denoiser updates on 5,500 replay steps.
- `uniform-pilot-run1-20261020/milestones/steps_005500` is exactly that state: epoch 1, 10,000 denoiser updates, 0 PMPO updates, 11 episodes. Denoiser `64bcd19c…`, dataset `3eb1232e…`.
- Difference from a real Laplace run: there, epoch 1's 500 steps would be collected by the initial policy, not uniformly.

**Placement and integrity.**
- kingplus0 GPU 1 (UUID `GPU-923e4182…`), at the user's direction after GPU 0 was found occupied by another user's idle Jupyter kernel. GPU 1 had no processes before the run.
- The process saw one device. `nvidia-smi` listed its PID only on GPU 1. GPU 0 and its kernel were untouched.
- Code: an isolated snapshot (HEAD `cd78477` plus the 11 reviewed Laplace working-tree files; 378 files, hashes verified on the server) in a new directory `/home/bigking/scfeng/LCG-laplace-first-refresh-20261027`.
- `load_milestone` verified all checksums. All 15 source files (milestone, run config, 11 episode versions) were hashed before and after: unchanged.
- Run time: 11 s, exit 0.

**Settings.** 5 members; T = 1; R1 with guards; 320 / 3 / 1e-4 / 1; the milestone's denoiser, the run's corruption and sampler settings; production arithmetic (TF32); 32 prompts; 2 noise draws.

**Results.**
- **Precision:** min 3.3, p01 28, median 4,708, p99 4.6e5, max 3.3e6; 0% of coordinates near the damping floor.
- **Perturbations:** 1.84x the selected weights' norm (10k milestone: 1.43x; smoke: 1,160-1,930x).
- **Predictions:** finite; saturation 0.17% vs 0.14% for the base. Ensemble-mean shift 0.017, member spread 0.024, member-to-base RMSE 0.029.
- **Raw rewards:** mean 5.6e-4, p10 4.2e-4, median 5.2e-4, p90 7.1e-4, max 9.3e-4; none zero. The control gives float-rounding values only (~1e-16).
- **Cost on the RTX 5090:**
  - precision refresh 7.8 s, peak 0.15 GB;
  - member prediction 0.125 s per imagined step at batch 32, vs 0.028 s for one base chain (4.5x); peak 0.75 GB.
- **Images (inspected).**
  - One-step predictions from all members keep the scene and continue the replay context.
  - Six-step fixed-action sequences (3 prompts) stay coherent. Members follow the base's motion and diverge in pose from about step 4. Member-to-base RMSE grows 0.030→0.132, spread 0.024→0.094, against base frame changes of 0.14-0.19 per step.
  - Occasional artifacts appear in the base as well (e.g. a detached limb at step 6), in some members and not others.

**Established.** At the first refresh under the pilot's actual schedule, T = 1 preserves usable predictions, with non-degenerate member disagreement above the arithmetic floor.

**Not established.** Whether that disagreement tracks prediction error (deferred), behaviour on a run whose epoch-1 data come from the initial policy, other seeds, and exploration benefit.

**Recommendation.** Retain T = 1 provisionally for a later bounded online pilot. Temperature, prior, damping, subset and divisor are unchanged.


## Step 7 — Logging and analysis

### Status history (moved from the progress file, 2026-10-29)

- [ ] Track collection, model losses, intrinsic rewards, and policy training.
  Exploration stage done and verified (2026-10-27, not committed):
  - terminal phase output (captured-stdout friendly);
  - W&B epoch summaries, per-update and epoch axes, env-step axis, counters, timing, provenance and
    milestone identities;
  - same-run W&B resume, with a local row mirror that backfills committed rows a crash kept from W&B;
  - four-seed real-environment intrinsic evaluation of the frozen scorer
    (`evaluation.reset_seeds`, `evaluation/*.json`).
  Verified by 16 new tests (full suite passing) and five short online W&B smokes (project
  `lcg-reward-free-exploration-logging-smoke`, group `exploration-logging-smoke-20261027-final`), the LCG one
  hard-killed mid-epoch and resumed via
  `scripts/resume.sh`.
  Notes: `REWARD_FREE_EXPERIMENT_NOTES.md`, Step 7.
  Terminal format finalized and committed (`62ecded`, 2026-10-28).
  Resume fixes (2026-10-28, committed `b1e10c2`):
  - an interrupted `wandb_history.jsonl` record is repaired before appending or backfilling;
  - backfill uses only checkpoint-committed rows; offline W&B resumes no longer re-log them;
  - the frozen intrinsic scorer (LCG round, full-ensemble RunningRMS scale) is checkpointed and
    restored, so a crash before the final evaluation no longer loses it.
  Verified by tests (all five methods) and an LCG crash/resume smoke on kingplus0 GPU 1 via
  `scripts/resume.sh`. Notes: Step 7, "Resume fixes".
  W&B metric inventory (2026-10-28, documentation only; logging unchanged; not committed):
  - every exploration-stage key by section, with meaning, frequency, axis, methods, source and evidence status;
  - verified exact duplicates and a Keep/Merge/Remove/Debug-only recommendation;
  - traceability checked against all local run evidence.
  Found, not fixed: a resume after the last epoch drops the `resume/*` row.
  Notes: Step 7, "W&B metric inventory"; review `docs/changes/step7_wandb_inventory_review.zip`.
  Controller migration (2026-10-28, committed `3ee4c28`): both stages now use the Stage 3 Gaussian H10
  EMA-critic controller (reference kingplus0:/home/bigking/scfeng/LCG-stage3-gaussian-h10-emacritic):
  - tanh-Gaussian actor; slow EMA critic regularization (coef 1.0, rate 0.02); horizon 10;
    repeat-anchor policy frame stacks at imagined resets;
  - advantage-weighted PMPO in all four exploration presets; sign-only, fresh controller downstream;
  - controller settings recorded in checkpoints, mismatches refused before loading.
  Verified by tests (576 passed) and an exploration -> milestone -> downstream smoke with
  crash/resume in both stages on kingplus0 GPU 1 (48 checks). Notes: "Controller migration";
  review `docs/changes/gaussian_h10_controller_migration_review.zip`.
  actor_critic metric organization (2026-10-28, committed `4b21ea5`): grouped W&B keys (core, EMA,
  gaussian, weights, intrinsic; 43 for Walker) and `agent.actor_critic.debug_metrics` (default false;
  105 more under debug/, computed only when on; training identical). Verified by tests (586 passed)
  and an online two-epoch LCG smoke (project lcg-reward-free-actor-critic-logging, run 8jyqydhf;
  29 checks). Notes: "actor_critic metric organization and debug switch".
  Baseline two-epoch smokes (2026-10-28): uniform, full ensemble, Plan2Explore and Laplace with the
  accepted source, online in the same project (group baseline-gaussian-h10-two-epoch-debug-off);
  136 checks passed (43 actor_critic keys per learned method, uniform none). Notes: "Baseline
  two-epoch smokes".
  W&B logging cleanup (2026-10-28, not committed): the remaining categories reduced to
  progress, timing, denoiser losses and the 12 evaluation returns per block. Retired keys and
  status moved to replay, evaluation JSON, provenance and the W&B summary. System GPU metrics
  are limited to the run's GPU. LCG goes from 192 to 85 keys, the actor_critic 43 unchanged.
  Full suite 592 passed; online smoke run nx4gnm77, 44 checks. Notes: "W&B logging cleanup".
  Workspace panels: manual removal list in the notes.
  Still open: the downstream trainer has no W&B logging (only `metrics.jsonl` and evaluation
  YAMLs); the pilot downstream runs are not on W&B yet.
- [x] Configurable exploration test collection and diagnostics (`evaluation.should`,
  `every`, `collect_test`): denoiser test loss, per-task test returns, test counts, fresh or
  reused test data, test interactions counted separately. Done (`84c119d`).
  Notes: `REWARD_FREE_EXPERIMENT_NOTES.md`, Step 6, "Exploration test collection and diagnostics".

### Exploration-stage terminal and W&B logging (2026-10-27)

Review: `docs/changes/step7_exploration_logging_review.zip`. Exploration trainer only; the downstream trainer's logging is unchanged (still no W&B).

#### Files

- `src/exploration_trainer.py`: startup summary, per-phase messages, epoch summary, W&B axes and provenance, the local W&B row mirror and resume backfill, seeded test collection, intrinsic evaluation, milestone messages.
- `src/intrinsic_eval.py` (new): `IntrinsicEvaluator`, which scores recorded test episodes with a method's frozen scorer; `sampler_windows`.
- `src/utils.py`: `Progress` (a standard tqdm bar since the terminal-logging correction), `say`, `phase`, `configure_progress`.
- Read-only scoring hooks: `LCGLifecycle.ready / score_candidates / scale_frozen`, `RunningRMS.apply_frozen`, `EnsembleScorer.scale_frozen`, `Plan2Explore.disagreement / scale`.
- `src/coroutines/collector.py`, `env_loop.py`: optional `NumToCollect.reset_seed` (test collection only) and progress output.
- `src/lcg/precision.py`, `src/plan2explore.py`: progress for historical precision, replay encoding and head fitting.
- `src/data/milestone.py`: the milestone folder rename is retried on Windows `PermissionError`.
- `config/trainer.yaml`: `logging.*`, `evaluation.reset_seeds / final_reset_seeds / intrinsic`, `wandb.id`. All optional: configs without them behave as before.
- Tests: `tests/exploration/test_exploration_logging.py` (new, 16). Three existing tests were updated for the new resume row and attributes.
- Environment: `wandb` upgraded 0.17.0 -> 0.30.0 in the local `lcg` env at the user's request (pip also raised protobuf 4.25.9 -> 7.36.2 and added opentelemetry/pydantic; `pip check` clean). kingplus0's env was not changed.

#### Terminal output

Revised three times (see "Terminal-logging revision", "Terminal-logging correction" and "Final terminal format" below); this is the current behaviour.

- Messages are flushed stdout lines written through `tqdm.write`, so a message issued while a bar is active gets its own line and the bar is redrawn below it. Progress uses tqdm bars written to stdout: phase description, percentage, completed/total, elapsed and remaining time, and rate with a space before the unit (`steps/s`, `updates/s`, `trans/s`, `ep/s`; inverted below one per second like tqdm, e.g. `3.85 s/update`). No postfix. The bars are display only: the `timing/*` phase timers in W&B are measured separately (tqdm overhead included, not subtracted). Each bar is closed complete on its own line when its phase ends.
- One blank line before each epoch heading and before the blocks outside epochs (initial collection, final evaluation); no blank lines between the phases of an epoch (`utils.phase()`).
- `logging.progress`: `bar` (default) or `none`.
- No losses, timing, GPU memory, PMPO diagnostics or task returns on the terminal; they are in W&B (task returns nowhere, see W&B). Exception: the opt-in `actor_critic.log_diagnostics=true` (default false) still prints `PMPO_DIAGNOSTICS` JSON lines.
- **Startup:** domain and downstream tasks, method, seed, compute device, rendering device, budget (initial + epochs x steps, training-only epochs), output directory; on resume the resume point; for uniform random one line saying there is no scorer refresh, no exploration-policy training and no intrinsic scores.
- **Each epoch:**
  - device lines (`src/device_report.py`): `Compute device: <name> | physical GPU <i> | process cuda:<k>`, where the physical index is nvidia-smi's, matched by the device UUID (never inferred from `cuda:<k>`, which `CUDA_VISIBLE_DEVICES` renumbers); without a match `UUID <uuid> (physical index not verified)`; CPU runs `Compute device: CPU`. `Rendering device: EGL | physical GPU <i>` when verified: a subprocess with the run's environment asks the EGL device dm_control selects (`MUJOCO_EGL_DEVICE_ID`) for its CUDA device (`EGL_NV_device_cuda`) and that device for its UUID (CUDA driver API; no GL or CUDA context). Otherwise the configured value is shown and marked not verified (`EGL device <n> (configured; …)`, MUJOCO_GL unset, GLFW); OSMesa is CPU software rendering;
  - header `=== Epoch k/N | exploration phase | training env steps s/B ===` (`training-only phase (no new collection)` for epochs without collection);
  - bars: `Initial collection (<mode> actions)` (before epoch 1), `Collection (<mode> actions)`, `Denoiser training` (`Denoiser training (member k/K)` for the full ensemble), `Historical precision` (LCG, Laplace), `P2E replay encoding` and `P2E head fitting (K heads)`, `Exploration-policy training`, `Evaluation episodes (seeds a-b)` (one bar for all seeded episodes; `Final evaluation episodes …` at the end);
  - `Scorer refresh (<method>): <what it does>` before a refresh (learned methods only);
  - `Imagined intrinsic reward/step: raw mean …, max … | scaled mean …, max …` right after the PMPO bar. Definition: the intrinsic reward per imagined step during this epoch's PMPO imagination, from the scorer refreshed this epoch; mean of the updates' means, max of their maxima; raw = the scorer's output, scaled = the policy reward after the method's reward scaling (same as `epoch_summary/intrinsic_*`);
  - `Intrinsic evaluation:` (each scheduled evaluation) or `Final intrinsic evaluation:` (the one after training), then one line per seed: `seed s | L steps | raw return R | scaled return S`. Definition: episode totals over the real test episode (one per fixed reset seed), scored by the method's frozen scorer of the latest refresh, nothing updated; units are each method's own (`evaluation_protocol.intrinsic_note`). Uniform: `seed s | L steps | intrinsic N/A`. A learned method that cannot be scored shows `Intrinsic evaluation: N/A (<reason>)`. Whether the episodes are fresh or reused is in `evaluation/*.json` (`kind`, `fresh`) and the `test/fresh` W&B key. No four-seed aggregates (W&B only);
  - `Milestone steps_NNNNNN published: > <absolute path>`.
- Uniform random has no refresh, PMPO or historical-precision bars and no intrinsic numbers.
- The scorer-internal stdout lines (`[LCG] theta_S/round`, `assert_setup_valid`, `[P2E]`, `[LAPLACE]` refresh and setup lines) are off in the exploration trainer (`LCGLifecycle.verbose = False`; the P2E/Laplace refresh prints were removed). Their values are in W&B; LCG's refresh identity is new there as `lcg/*`. Genuine warnings (e.g. Laplace's shared-prefix fallback), errors, Python warnings and resume/backfill messages stay.

#### W&B

- Existing rows are unchanged. Additions:
  - every row carries `progress/train_env_steps`, the training env steps when it is logged (collection precedes training within an epoch);
  - one `epoch_summary/*` row per epoch:
    - means of the epoch's existing metrics (denoiser train/test loss, per member; PMPO `loss_actor`, `loss_value`, `policy_entropy`, `kl_prior`, sign fractions, grad norms, `return_mean`);
    - raw and scaled intrinsic reward (mean, raw max);
    - `progress/*` counters: train and separate test env steps, new train steps, denoiser updates, PMPO updates, ensemble total;
    - `timing/epoch_*_seconds` for collection, denoiser, scorer refresh, PMPO and test, plus the epoch total and GPU peak;
  - `timing/scorer_refresh_seconds` and `timing/pmpo_seconds_per_update` next to the existing PMPO timing;
  - `timing/test_diagnostic_seconds`.
- **Axes (`define_metric`):**
  - `actor_critic/train/*` uses its update counter `num_batch_train_actor_critic`;
  - `denoiser/train/*` uses `num_batch_train_denoiser`, and members use their own counters;
  - every other prefix uses `epoch`. The globs do not overlap, so W&B's glob order is irrelevant.
  - An env-step axis is available on every row. Training-only epochs show on the epoch axis.
- **Provenance:** `config` (resolved, as before) plus `config.provenance`: actual seed, method, method settings (reward scaling, Laplace/P2E/ensemble settings), source identity, evaluation protocol, milestones.
  - Each milestone's identity (denoiser and dataset sha256, counters, path) goes to `summary["milestones/<name>"]`, with a `milestone/collected_steps` row (no longer emitted by the current code; see "W&B metric inventory").
- **Run identity and resume:**
  - The run id is written to `wandb_run.json` in the run folder.
  - A resume passes it back explicitly (`resume="allow"`), so the same W&B run continues. Folders without the file resume as before.
  - `resume/from_epoch` and `resume/count` are logged; `resume_count` is in the summary and checkpointed.
  - **Replayed updates:** epochs after the checkpoint are replayed and logged again with identical values (same epoch and counters, bit-identical training). Rows an interrupted epoch had already logged (only if the crash fell between that epoch's `wandb_log` and its checkpoint) therefore appear twice at the same x.
  - **Local mirror and backfill:** W&B uploads in batches about every 15 s. A hard crash that also kills its upload process loses the rows still queued.
    - Each epoch's rows are logged together at its end, so a crash shortly after an epoch can lose that whole epoch.
    - This happened in the first smoke attempt: the LCG run lost all of epoch 1.
    - Fix:
      - Every row is appended to `wandb_history.jsonl` in the run folder (W&B step, epoch, resume segment, scalar values) before the checkpoint that commits it.
      - The checkpoint stores the committed row count (`wandb_steps`) and, since the resume fixes, the mirror's byte length at the commit (`wandb_history_bytes`); every append is fsynced first.
      - On resume the mirror is repaired before anything is appended or backfilled (see "Resume fixes" below).
      - On resume, W&B's `starting_step` (rows the server holds) is compared with it:
        - **missing committed rows:** re-logged from the mirror's committed records (before `wandb_history_bytes`) with `resume/backfilled=1`, and counted in `resume/backfilled_rows`;
        - **extra rows (an uncommitted epoch about to be replayed):** counted in `resume/server_rows_beyond_checkpoint`.
      - The resume row also records `resume/wandb_server_rows` and `resume/wandb_committed_rows`.
      - Offline W&B (`wandb.mode=offline`) reports no server rows and starts a new offline file per segment: nothing is backfilled (the mirror keeps every committed row); the resume row records only `resume/wandb_committed_rows`.
    - Milestone identities are re-set in the summary on resume.
    - `logging.local_history: false` turns the mirror off (and with it the backfill).
    - The mirror holds every scalar of every row: about 8.7 kB per PMPO row and 0.3 kB per denoiser row, an estimated ~0.7 GB for a 100k-step run (more for the full ensemble's member rows).
  - Rows of an epoch that a crash interrupts before its checkpoint are not backfilled (that epoch is replayed). Data from a final segment that never resumes is only in the mirror.
- Logging never consumes training random numbers and never changes losses, updates or collection: tests compare training state and learning rows with logging on/off and with an active (fake) W&B run.

#### Real-environment intrinsic evaluation

- **Collection (existing test collector):** `evaluation.reset_seeds` collects one episode per seed (the env reset uses that seed). `num_episodes` must equal the number of seeds, and `collection.test.num_envs` must be 1. `final_reset_seeds` does the same for the final collection.
  - `null` keeps the earlier random reset seed.
  - Test interactions are counted as before (`test/total_new_env_steps`, never training replay).
  - Seed labels are checkpointed (`test_episode_seeds`).
  - The episode limit uses the existing `env.test.time_limit` (seconds of simulation; Walker 2.5 s = 50 env steps at action_repeat 2).
- **Scoring:** `evaluation.intrinsic` (default true) scores every test episode with the current frozen scorer, each method using its own definition:
  - **sampler window:** frames o_{t-n+1..t} and actions a_{t-n+1..t}, zero padded before the episode start like replay segments; this is the WorldModelEnv buffer at an imagined step;
  - **LCG:** candidate (window, real o_{t+1}), the final observation completing the last step;
  - **full ensemble / Laplace:** shared-noise member predictions, population pixel variance averaged over `num_noise_samples`; noise from the diagnostic RNG stream;
  - **Plan2Explore:** frozen heads on the feature of o_t (context o_{t-n..t-1}, a_{t-n..t-1}) and a_t.
  - Tests check each against the imagination path on the same real windows: the LCG PMPO hook, `WorldModelEnv.predict_next_obs_ensemble` (ensemble and Laplace) and the P2E hook.
  - Scaled reward uses the current scale without updating it (fixed divisor, or RunningRMS's current `s2`).
- **Isolation:** nothing is refreshed, redrawn or refitted. No RunningRMS state, last-batch PMPO metrics, timing or call counters change (restored where a call would touch them). The global RNG, weights and optimizers are unchanged; it runs inside the existing test-RNG workflow.
  - Before LCG has refreshed in a process (e.g. a resume's final diagnostic), it reports "not refreshed" instead of a score.
- **Outputs:** W&B block `test_eval/` (`final_test_eval/`) for fresh episodes, or `test_dataset_diagnostic/` (`final_…`) for reused ones.
  - Per seed (`seed_<s>/`, or `episode_<i>/` without a label): `length`, `intrinsic_raw_return`, `intrinsic_scaled_return`, `intrinsic_raw_per_step`, `intrinsic_scaled_per_step`, `intrinsic_scored_steps`.
  - Mean/std over episodes, `intrinsic_applicable`, `intrinsic_scored`.
  - Locally: `evaluation/epoch_NNNNN.json` / `final_epoch_NNNNN.json` with the protocol, kind (fresh/reused status), scorer settings, per-episode records and aggregates.
  - Downstream task returns are not logged anywhere (terminal, W&B, mirror, evaluation JSON): the per-episode `return` of collection rows, `final_return_mean/std`, `test/return_<task>/*` and the per-seed `return_<task>` were removed in the terminal-logging revision. Every replay episode (training and test) still stores all task-reward columns (`task_rew`, `task_names`) for downstream adaptation.
- **Labels:**
  - Fresh episodes from a learned method are `current_policy_evaluation`.
  - Uniform's are `fresh_episodes (uniform-random actions, no policy)`, with intrinsic marked not applicable and no intrinsic values.
  - Reused episodes are `dataset_diagnostic`, keeping their saved seed labels. Episodes from checkpoints without labels stay unlabeled.
- **Units:** intrinsic returns are in each method's own units (LCG score, pixel variance, feature variance). They compare checkpoints within a method and cannot establish that one method is better than another.
- **Bug found by the GPU dry run and fixed:** a test episode's final observation could still be on the GPU; `sampler_windows` now moves it to the frames' device. CPU tests could not catch this.

#### Other fixes found by the smokes

- **Milestone publication on Windows:** `src/data/milestone.py` renames the finished temporary folder into place. On Windows this failed once with `PermissionError` (WinError 5) for the ensemble's 700-step milestone, with 5 member files, most likely because a scanner held a fresh file.
  - The rename is now retried (20 x 0.5 s) on `PermissionError`.
  - That run's checkpoint was already committed, so a resume would have republished the milestone.
- **LCG on this laptop:** `walker_lcg_100k` keeps `training.compile_wm=true`. torch.compile needs Triton, which is not available on this Windows install, so the LCG smoke passes `training.compile_wm=false`. This is execution only; the other presets already use false.

#### Verification

- **Tests:** 16 new tests in `tests/exploration/test_exploration_logging.py`. Full suite: 505 passed (local `lcg` env, wandb 0.30.0). They cover:
  - seeded collection, labels, counts and saving;
  - alignment of every method with its imagination scoring path;
  - isolation (scorer, RunningRMS, banks/heads, timing counters, weights, optimizers, RNG);
  - invalid seeds;
  - uniform omissions;
  - rows, counters, axes and training-only epochs;
  - logging on/off and active-W&B training equivalence;
  - resume: same W&B id, replayed rows identical, reuse labels, old checkpoints;
  - backfill when the server lost rows, and duplicates counted when it holds extra rows.
- **Smokes** (`outputs/step7-exploration-logging/`, local RTX 5060 Laptop GPU; `run_smokes.sh`, `crash_resume_lcg.sh`, `verify_smokes.py`):
  - Real entry point; W&B project `lcg-reward-free-exploration-logging-smoke`, group `exploration-logging-smoke-20261027-final`; seed 0.
  - Budget: 500 uniform steps, then 2 epochs x 100 steps (700 training steps), no training-only epochs.
  - Updates: 5 denoiser updates per epoch (per member for the ensemble), 2 PMPO updates per epoch, P2E `fit_updates=20`.
  - Test diagnostics every epoch plus a final one: 4 episodes, reset seeds 3001-3004, `env.test.time_limit=2.5` (50 env steps). That is 200 test env steps per point and 600 in total, counted separately.
  - Milestones at 600 and 700 steps. Reviewed method settings otherwise (5 members, Laplace T = 1).
  - LCG adds `training.compile_wm=false` (no Triton on this laptop).
- **Crash and resume:**
  - The LCG run was force-killed (`taskkill /F /T`, which also killed W&B's upload process) during epoch 2's scorer refresh, after epoch 1's checkpoint and milestone.
  - It was resumed with the run folder's `scripts/resume.sh`.
  - The W&B server had received none of epoch 1's 26 rows. The resume backfilled all 26 from the mirror into the same run (`dtjq4f7k`, `resume_count` 1), and the run finished.
- `verify_smokes.py`: 303 checks passed, for all five runs. They cover:
  - every phase in stdout, the uniform omission explained once, no refresh/PMPO lines for uniform;
  - checkpoint counters (700 train / 600 test env steps, 10 denoiser updates, 4 or 0 PMPO updates), final diagnostic committed, seed labels, both milestones;
  - evaluation JSONs (4 seeded 50-step episodes; intrinsic scored only for learned methods);
  - W&B run finished, in the right group; summary counters agree with the checkpoint; milestone identities;
  - provenance (seed, method, source); history keys for every block, one epoch-summary row per epoch with matching counters;
  - W&B history rows equal the mirror's steps; LCG resume and backfill as reported.

| Method | W&B run | Final test (4 seeds): intrinsic raw return, mean +- std | Scaled | Walk return |
| --- | --- | --- | --- | --- |
| Uniform random | `n9elzkx1` | not applicable | – | 4.72 |
| LCG | `dtjq4f7k` | 4573 +- 220 | 45.73 (divisor 100) | 4.92 |
| Full diffusion ensemble | `1ol34obz` | 0.01417 +- 4.8e-6 | 0.01417 | 6.21 |
| Plan2Explore-style ensemble | `901wgx2n` | 8.24 +- 0.61 | 8.24 | 5.00 |
| Laplace ensemble | `cogc7git` | 1.68e-4 +- 6.6e-6 | 1.68e-4 | 6.47 |

- These numbers validate logging only. The units differ by method, and 10 denoiser updates are far from any real schedule.
  - **Laplace (known limitation):** with this shortened denoiser schedule its members are in the barely trained regime seen in the earlier smoke diagnostics, where predictions can collapse. Disagreement is ~3e-6 per step. Images were not inspected; this is not a calibration result.
  - **Full ensemble:** 5 barely trained members also give near-constant disagreement (~2.8e-4 per step).
- **Epoch-2 phase times (s), laptop:**

| Method | Collection | Denoiser | Refresh | PMPO (2 updates) | Test | Total | GPU peak |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Uniform | 0.6 | 1.7 | – | – | 37.8 | 40.2 | 5.13 GB |
| LCG (resumed) | 2.3 | 2.2 | 30.2 | 43.8 | 25.0 | 103.5 | 5.13 GB |
| Ensemble | 1.3 | 8.7 | 0.0 | 31.8 | 24.2 | 66.1 | 5.37 GB |
| P2E | 1.4 | 1.8 | 3.8 | 23.1 | 43.6 | 73.7 | 5.22 GB |
| Laplace | 1.2 | 1.6 | 28.4 | 26.9 | 25.4 | 83.7 | 5.17 GB |

  Test time is dominated by rebuilding the dm_control task on each seeded reset and by rendering.
- **GPU resume identity was not shown:** an uninterrupted offline dry run with the same seed and configuration already differs from the online LCG run at epoch 1 (~0.1% in intrinsic returns), before any crash, i.e. cross-process GPU nondeterminism. Exact resume identity rests on the CPU tests.
- **First attempt (group `exploration-logging-smoke-20261027`; runs tagged `superseded`, with notes):**
  - It found the W&B row loss: the LCG run kept only epoch 2's 40 rows.
  - It found the Windows milestone rename failure: the ensemble run exited 1.
  - Both were fixed before the final group. Its local outputs are in `smoke_attempt1/`.
- An offline dry run before that (`dryrun_offline/`) found the GPU device bug and the Triton limitation.

### Terminal-logging revision (2026-10-27)

Review: `docs/changes/step7_terminal_logging_review.zip`. Terminal output to the user's preferences; W&B keeps losses, timing, memory, counters and PMPO diagnostics; downstream task returns are no longer logged. No training, reward-scaling, schedule or method change. The terminal and W&B descriptions above are updated to this behaviour.

#### Changes

- `src/utils.py`: `Progress` is now a tqdm bar (stdout, `{desc}: {pct}%|{bar}| n/total unit`, no time, rate or postfix); `logging.progress` is `bar` or `none` (the `auto`/`lines` modes and `progress_lines` were removed).
- `src/coroutines/collector.py`: `NumToCollect.desc` (bar text) and `NumToCollect.progress` (False: no bar; the seeded evaluation shows one bar for all seeds). The collector's postfix is gone.
- `src/exploration_trainer.py`:
  - compact banner (no W&B URL, test protocol, parameter counts, dataset sizes or `Starting on`);
  - removed: per-phase start/done messages, `[TIMING]` lines, `Scorer refresh done in`, `Test diagnostics …` messages, episode returns, `final_return_*`, the `test:` dict line, the `test_eval: <kind>` line and the epoch-summary line (the `epoch_summary/*` W&B row stays);
  - new labelled lines `Intrinsic reward [training …]` and `Intrinsic return [evaluation …]` (per seed: length, raw and scaled episode totals; uniform: N/A);
  - milestone message `Milestone <name> published: > <absolute path>`;
  - task returns removed from W&B rows (`without_task_returns`, no `task_return_summary`, no per-seed `return_<task>`);
  - LCG refresh identity logged as `lcg/{round, replay_steps, theta_s_dim, h_D_fingerprint, banks_fingerprint}` (epoch axis).
- `src/lcg/lifecycle.py` (`verbose`), `src/lcg/precision.py` (`assert_setup_valid(verbose=)`): their stdout lines stay on by default for the diagnostic scripts; the exploration trainer turns them off.
- `src/plan2explore.py`, `src/laplace.py`: refresh prints removed (all values were already in `plan2explore/*`, `laplace/*`); P2E head-fitting bar without loss postfix.
- `config/trainer.yaml`: `logging.progress: bar`; `epoch_summary` now controls only the W&B row.
- Tests: `test_exploration_logging.py` checks the terminal through what stays on screen (text after each line's last carriage return): banner, one closed bar per phase with its counts, no duplicates, forbidden content (losses, timing, memory, PMPO, task returns, tqdm time/rate, old messages), labels and values of the intrinsic lines, uniform N/A, milestone format, resume messages; a new parametrized test covers the ensemble member bars, P2E encoding/fitting and Laplace precision. Task-return tests in `test_test_diagnostics.py` and `test_exploration_trainer.py` now check that no task return is logged while every replay episode keeps its task-reward columns. Logging on/off still gives identical weights, optimizers, replay and final RNG state.

#### Verification

- Tests: full suite 507 passed (local `lcg` env); focused 47 passed.
- Smokes on kingplus0 GPU 1 (`outputs/step7-terminal-logging/`):
  - Code: isolated snapshot (HEAD `9176c13` + working tree via a temporary index, 380 files, hashes verified on the server) in a new directory `/home/bigking/scfeng/LCG-step7-terminal-logging-20261027`.
  - Placement: `CUDA_VISIBLE_DEVICES=GPU-923e4182-…` (UUID of GPU 1; `common.devices=all` keeps it), `MUJOCO_GL=egl`, `MUJOCO_EGL_DEVICE_ID=1`. A probe with the same settings showed its only `nvidia-smi` row on GPU 1 as `C+G` (CUDA and EGL). During the smokes (2 s sampling) the trainer process appeared only on GPU 1; GPU 0's memory stayed at the other user's 6848 MiB in all 50 samples. The env workers (gymnasium `AsyncVectorEnv` subprocesses, which render) inherit the same variables; they were not sampled by PID.
  - W&B: `WANDB_MODE=offline` and `wandb.mode=offline`; nothing was synced.
  - Budget (output checks only): 300 initial uniform steps (LCG/Laplace precision needs 320 distinct transitions at the first refresh), one epoch of 100, 2 denoiser updates (per member for the ensemble), 2 PMPO updates, P2E `fit_updates=20`; evaluation at epoch 1 and final, seeds 3001-3004, 50-step episodes; milestone at 400. Method settings unchanged (5 members, Laplace T = 1).
  - All five exited 0 (13-32 s each). `verify_smokes.py`: 495 checks passed (bars, completeness, no duplicates, forbidden content, labels, per-seed values against W&B, milestone format, offline `.wandb` history keys present and task-return keys absent, offline rows = mirror rows, replay `task_rew` (T, 3) for stand/walk/run, placement).
  - First attempt (`smoke_attempt1/`): identical outputs, 494/495 checks; uniform finished before the first 15 s placement sample, so the set was rerun with 2 s sampling.
- Smoke numbers are not meaningful (2 denoiser updates). Laplace raw reward ~6e-8 per imagined step: the known barely-trained regime.

#### Still pending (not fixed here; both fixed on 2026-10-28, see "Resume fixes")

1. A partially written last line of `wandb_history.jsonl` (crash during the append) makes `reconcile_wandb_history` raise `JSONDecodeError`, so the resume fails.
2. A resume after the final epoch's checkpoint but before the final evaluation does not refresh the scorer (no epoch runs). LCG's lifecycle (precision, candidate bank, RunningRMS) is not checkpointed, so the final intrinsic evaluation has no frozen scorer or normalization state; LCG reports "not refreshed". The other methods' state in this window was not checked.

### Terminal-logging correction (2026-10-28)

Review: `docs/changes/step7_terminal_logging_correction_review.zip`. Terminal changes only, from the user's review of the revision above; no training, evaluation, reward-scaling, schedule or method change, and no W&B change.

Evaluation behaviour, confirmed and unchanged: a periodic evaluation at every scheduled epoch (`evaluation.every`), and one final evaluation after all training, also when the last epoch already had a periodic one.

#### Changes

- `src/utils.py`:
  - `Progress` uses tqdm's standard format: phase, percentage, completed/total, `[elapsed<remaining, rate]`. "No timing" means no separate timing summaries; they stay W&B-only (`timing/*`).
  - `say()` writes through `tqdm.write` (an active bar is cleared and redrawn below the message).
  - New `phase()`: one blank line before a major phase, never two in a row.
- `src/exploration_trainer.py`:
  - `phase()` before the epoch header, initial collection, denoiser training (once for all ensemble members), scorer refresh, exploration-policy training, evaluation and the milestone message.
  - Training line simplified to `Imagined intrinsic reward/step: raw mean …, max … | scaled mean …, max …`; same computation as before.
  - Evaluation headings simplified to `Intrinsic evaluation:` / `Final intrinsic evaluation:`, per seed `seed s | L steps | raw return R | scaled return S` (uniform: `| intrinsic N/A`). The removed bracketed explanations are kept in "Terminal output" above and in `evaluation/*.json` (`protocol.intrinsic_note`, `intrinsic.scorer`, `kind`).
- `config/trainer.yaml`: comment of `logging.progress`.
- Tests (`tests/exploration/test_exploration_logging.py`): bars must be closed at 100% with `[elapsed<remaining, rate]`; exact training lines computed from the per-update W&B rows; per-seed lines of every evaluation round against W&B; blank line before each phase (the scorer-refresh line comes after the denoiser bar and one blank line); no double blank lines; old long labels forbidden; new unit test for a message during an active bar.

#### Verification

- Tests: full suite 508 passed; focused (logging, test diagnostics, exploration trainer, resume/milestones) 67 passed. Logging on/off still gives identical weights, optimizers, replay, training rows and final RNG state.
- Smokes on kingplus0 GPU 1 (`outputs/step7-terminal-logging-v2/`; they precede the final format adjustment below, so their terminal output shows the earlier bar units, headings and blank lines), new directory `/home/bigking/scfeng/LCG-step7-terminal-logging-v2-20261028`, isolated snapshot (HEAD `9176c13` + working tree, 380 files, hashes verified on the server):
  - GPU 1 idle before (19 MiB, Xorg only); GPU 0 had another user's process (pid 1247789, untouched).
  - Placement: `CUDA_VISIBLE_DEVICES=<GPU 1 UUID>`, `MUJOCO_GL=egl`, `MUJOCO_EGL_DEVICE_ID=1`. Sampled every 2 s with the trainer's child PIDs: the trainer (`C`) and its env-worker subprocesses (`G`, EGL rendering) appeared only on GPU 1, in every run; GPU 0 stayed at 6848 MiB in all 72 samples.
  - W&B `offline` (config and `WANDB_MODE`); nothing synced.
  - Schedule: 300 initial uniform steps + 2 epochs x 100 = 500 training env steps; 2 denoiser updates per epoch (per member for the ensemble); 2 PMPO updates; P2E `fit_updates=20`; evaluation every epoch plus the final one, seeds 3001-3004, 50-step episodes: 3 rounds, 12 episodes, 600 test env steps; milestones at 400 and 500. Method settings unchanged.
  - All five exited 0 (15 / 47 / 31 / 19 / 41 s: uniform, LCG, ensemble, P2E, Laplace). `verify_smokes.py`: 770 checks passed (bar fields, completeness, counts per epoch, phase spacing, simplified labels, training lines and all three evaluation rounds against the offline W&B records, milestone messages, forbidden content, expected W&B keys present and task-return keys absent, offline rows = mirror rows, env-step accounting, replay `task_rew` (T, 3), placement). The first verifier pass had two checker bugs (`fresh` matched inside "refresh"; the ensemble's member bars were each expected after a blank line instead of the group); fixed in the checker only, kept as `verify_smokes_attempt1_checker_bugs.log`.
- The two resume issues under "Terminal-logging revision → Still pending" remain pending.
- Found later (final format check): the v2 verifier's "no 'fresh'" check was vacuous (its regex contained backspace characters from shell quoting, so it matched nothing). A direct search of the v2 output finds "fresh" only in P2E's "Scorer refresh …: replay feature encoding and fresh predictor heads", not as an episode status, so the conclusion holds; the final verifier checks the status phrases instead.

### Final terminal format (2026-10-28)

Review: `docs/changes/step7_terminal_logging_final_review.zip`. Final formatting from the user's review; supersedes the earlier formatting. Terminal only: training, reward calculation and scaling, evaluation protocol (periodic at scheduled epochs plus one final evaluation, also after a periodic one in the last epoch) and W&B values are unchanged.

#### Changes

- Banner: `Compute device: …` and `Rendering device: …` after the seed (new `src/device_report.py`; see "Terminal output" for the rules). Physical GPU numbers come from nvidia-smi matched by UUID, never from `cuda:<k>`; EGL rendering is verified by a subprocess probe (EGL device → CUDA device → UUID, no context), else shown as configured and marked not verified; CPU runs show `Compute device: CPU`.
- Epoch headings: `exploration phase` (`training-only phase (no new collection)` for epochs without collection).
- Spacing: one blank line before each epoch heading, before the initial collection and before the final evaluation; none between the phases of an epoch (scorer refresh, PMPO, evaluation, milestone follow directly).
- Bars: rate with a space before the unit, units `steps/s`, `updates/s`, `trans/s` (historical precision), `ep/s` (evaluation, P2E encoding), inverted below one per second as `s/update`, `s/step`, `s/ep` (`utils.rate_text`); the PMPO bar is `Exploration-policy training`. Elapsed and remaining time unchanged. Phase timers in W&B (`timing/*`) remain separate wall-clock measurements; tqdm overhead is included, not subtracted.
- Labels kept: `Imagined intrinsic reward/step: …`, `Intrinsic evaluation:`, `Final intrinsic evaluation:`, per seed `seed s | L steps | raw return R | scaled return S` (uniform `intrinsic N/A`), milestone `Milestone <name> published: > <path>`.

#### Verification

- Tests: new `tests/exploration/test_device_report.py` (physical vs process-local mapping by UUID, unverified fallbacks, EGL verified/configured/unset/OSMesa/GLFW, no subprocess on CPU runs, failed probe or nvidia-smi → unverified, rate text with spacing and inversion); `test_exploration_logging.py` updated (new bar format and units, `exploration phase` headings, blank lines only before epochs and the blocks outside them, the exact epoch-1 layout, CPU device line). Full suite 521 passed; focused 80 passed.
- Local GPU (Windows, RTX 5060 Laptop, `MUJOCO_GL` unset): `Compute device: NVIDIA GeForce RTX 5060 Laptop GPU | physical GPU 0 | process cuda:0`, `Rendering device: MuJoCo default backend (MUJOCO_GL not set; device not verified)`.
- Server check, uniform + LCG only (cosmetic change; the five-method v2 smokes were not repeated), kingplus0 GPU 1, W&B offline, same schedule as v2, new directory `/home/bigking/scfeng/LCG-step7-terminal-logging-final-20261028`, isolated snapshot (382 files, hashes verified): both exited 0 (15 s, 48 s); banner `Compute device: NVIDIA GeForce RTX 5090 | physical GPU 1 | process cuda:0` and `Rendering device: EGL | physical GPU 1`; trainer (`C`) and env workers (`G`) only on GPU 1, GPU 0 at the other user's 6848 MiB in all 30 samples. `verify_smokes.py`: 297 checks passed (first pass: one checker bug, the forbidden word "GPU" also matched the new device lines; kept as `verify_smokes_attempt1_checker_bug.log`). Every non-timing W&B value equals the v2 run's (uniform 51 rows, LCG 59 rows, 0 differences).
- The two resume issues under "Terminal-logging revision → Still pending" remain pending.

### Resume fixes (2026-10-28)

Committed `b1e10c2`. Review: `docs/changes/step7_resume_fixes_review.zip`. Fixes the two pending resume issues of the terminal-logging revision, on top of the accepted terminal format (`62ecded`). Unchanged: the terminal format, W&B metric names, training, evaluation schedules and method settings.

- Code: `src/utils.py`, `src/exploration_trainer.py`, `src/lcg/lifecycle.py`, `src/ensemble.py`.
- Tests: `tests/exploration/test_exploration_logging.py`.
- No new source files.

#### 1. Interrupted `wandb_history.jsonl` records

**Root cause.** A hard kill can cut the mirror's last record.
- `reconcile_wandb_history` parsed every line (`read_text` + `json.loads`), so one cut record made the resume fail even when every committed row was intact.
- A run that appended after the cut would have turned it into corruption inside the file.
- The whole file was also read into memory.

**Fix.** `utils.repair_history` and `utils.history_records` run from `ExplorationTrainer.repair_wandb_history`, right after the checkpoint is loaded and before anything is appended or reconciled.
- Appends are flushed and fsynced. The checkpoint stores the mirror's byte length at its commit (`wandb_history_bytes`): records before it are committed, records after it come from an interrupted segment.
- An incomplete last record is copied byte-for-byte to `wandb_history.jsonl.interrupted-<n>` (exclusive create, never overwritten), then truncated away.
- A whole last record that only lost its newline gets the newline.
- A malformed record before the last line raises, giving its record number and byte offset. Committed bytes that are missing, cut or do not end on a record boundary also raise ("not resuming over them"). Nothing is dropped silently.
- The repair is idempotent. It streams line by line: the same approach as `downstream_trainer.recover_metrics_log`, but without loading the whole file.
- Backfill reads only the committed records, taking the latest write per step within them. Rows of an interrupted segment stay in the file and are never backfilled.

**Older checkpoints** (no `wandb_history_bytes`): the tail is repaired the same way. Backfill uses all complete records with step < `wandb_steps`, as before.

**Offline W&B (found while testing the server smoke).** An offline resume reports `starting_step` 0 and starts a new offline file per segment (W&B: "`resume` will be ignored since W&B syncing is set to offline"). The resume therefore re-logged every committed row as "backfill", duplicating them when the files are synced. Offline resumes now backfill nothing and say so.

#### 2. Frozen intrinsic scorer across a resume

**Root cause.** After a crash between the last epoch's checkpoint and the final evaluation, the resume runs no epoch, so there is no refresh. Audit of that window:

| Method | State the final evaluation needs | Before the fix |
| --- | --- | --- |
| Uniform random | none | nothing to restore |
| LCG | precision h_D, candidate JVP bank, round, RunningRMS | lifecycle not checkpointed: final intrinsic evaluation N/A |
| Full ensemble | members (checkpointed); `EnsembleScorer` RunningRMS scale | scale not checkpointed: scaled returns lost with `reward_scaling=rms` (fixed/none: nothing lost) |
| Laplace | precision and perturbation bank (already in `laplace`); fixed/none scaling | no loss |
| Plan2Explore | heads and standardization (already in `plan2explore`); fixed/none scaling | no loss |

**Fix.**
- `LCGLifecycle.state_dict/load_state_dict` save, on CPU:
  - settings, round, theta_S names and dimension;
  - h_D;
  - the bank's sigmas, epsilons, offsets and etas;
  - RunningRMS `s2`.
- That is about 5.5 MB for theta_S = 217,664 and the preset's 5 bank samples.
- The restore re-selects theta_S on the loaded denoiser and rebuilds h_D^-1/2 and the PMPO hook; nothing is recomputed or redrawn. `refresh` was refactored into `_select_theta_s`, `_new_rms` and `_make_hook`, with the same behavior.
- `EnsembleScorer.state_dict/load_state_dict` (also used by `LaplaceScorer`): reward-scaling settings and RunningRMS `s2`.
- `ExplorationTrainer.state_dict` adds `lcg_scorer` and `ensemble_scorer`. `restore_scorers` runs after the weights are loaded:
  - restores both states;
  - rejects a different method or different settings;
  - wires the PMPO hook as the refresh left it.
- The next epoch's refresh replaces everything as before, so mid-run resumes are unchanged.
- Evaluation applies the frozen scale without updating RunningRMS. `test_diagnostics` restores the training RNG and module modes, as before. `final_test_done` is set only after the final rows are logged.

**Older checkpoints** (no `lcg_scorer` / `ensemble_scorer`):
- Mid-run resumes are unaffected: the next refresh rebuilds the scorer before any evaluation.
- In the final-only window the missing state is reported, not re-created.
  - LCG's final evaluation shows `Final intrinsic evaluation: N/A (the checkpoint predates saved LCG scorer state ...)`, with the reason in `evaluation/*.json`.
  - An RMS ensemble keeps its raw returns and prints a warning that scaled returns are N/A.

**Size.** `state.pt` grows by the LCG bank and h_D: measured +5.48 MB (86.9 to 92.4 MB, +6.3%) at the 100k configuration. The other methods add a few bytes.

#### Verification

**Tests** (`test_exploration_logging.py`):
- Mirror repair:
  - committed rows, an interrupted complete row and a cut record: recovery bytes, idempotence, a second interruption going to `interrupted-2`, safe appends;
  - a record missing only its newline;
  - interior corruption, reported with its location;
  - missing, short or non-boundary committed bytes;
  - a cut record inside the committed bytes.
- Trainer-level reconciliation:
  - a cut tail plus an interrupted-segment row that rewrites a committed step;
  - backfill against a fake W&B server that lost rows re-logs exactly the committed values;
  - a second crash and resume keep the mirror whole;
  - offline W&B backfills nothing.
- Final-only resume, all five methods (crash after the last checkpoint):
  - no training, refresh, posterior redraw or head fitting (those functions raise if called);
  - per-seed raw and scaled returns equal the uninterrupted run's;
  - LCG h_D, bank and round are equal;
  - RunningRMS `s2` is restored and not updated (LCG, RMS ensemble);
  - weights, optimizers, counters, replay and training RNG are equal;
  - another resume does not repeat the final evaluation.
- Older checkpoints: final-only LCG reports N/A with the reason; the RMS ensemble gives raw returns only, plus a warning; mid-run resumes equal the uninterrupted run.
- Negative control: with the restore disabled, the LCG and RMS-ensemble final-only tests fail. Laplace and Plan2Explore pass, because their state was already checkpointed.

**Server smoke** (kingplus0 GPU 1, W&B offline):
- Setup:
  - new directory `/home/bigking/scfeng/LCG-step7-resume-fixes-20261028`, isolated snapshot;
  - LCG, 300 + 2 x 100 steps;
  - evaluation every epoch plus the final one (seeds 3001-3004, 50 steps); milestones at 400 and 500;
  - preset scaling `fixed` (divisor 100).
- **Phase 1** (`src/main.py`, exit 137): the process group was stopped and killed within about 1 ms of the third commit of `checkpoints/state.pt` (epoch 2), before its milestone and the final evaluation.
  - The first attempt waited for the milestone message with 50 ms polling and was too late: the final-evaluation bar had appeared. It is kept as `attempt1_kill_late/`.
- **Phase 2** (`scripts/resume.sh`, `DEVICE=1`, exit 0):
  - banner `Compute device: ... physical GPU 1 | process cuda:0`, `Rendering device: EGL | physical GPU 1`, `Resumed at epoch 2 (resume #1)`;
  - the pending milestone was published;
  - no collection, denoiser, refresh, precision or PMPO output;
  - `Final intrinsic evaluation:` with raw and scaled returns for all four seeds;
  - `final_test_done` set.
- `verify_resume.py` passed all 60 checks, including:
  - the restored round 2: the h_D sum and bank sigma sum equal epoch 2's `lcg/h_D_fingerprint` and `lcg/banks_fingerprint`;
  - scaled/raw = 1/100 for every seed;
  - the LCG state, weights, optimizers and counters unchanged by the resume;
  - only 4 x 50 new test steps;
  - no backfilled rows;
  - trainer and env workers only on GPU 1.
- The resumed final returns (1857 / 1755 / 1789 / 1790) equal those of the uninterrupted final-format run. Equality on GPU is not required; the exact checks are the CPU tests.

**Limitations.**
- The final evaluation's rows are logged before `final_test_done` is checkpointed. A crash between the two repeats the final evaluation on resume; its first rows then count as rows beyond the checkpoint (online) or remain in the earlier offline file.
- An offline run's segments are separate offline files, so after a crash a synced history can contain replayed epochs.

### W&B metric inventory (2026-10-28)

**Superseded for the current logging** by "W&B logging cleanup (2026-10-28)" below; this inventory describes the state before it. Downstream W&B keys: "Downstream W&B logging (2026-10-29)" below.

Review: `docs/changes/step7_wandb_inventory_review.zip`. This section covers the exploration stage at `b1e10c2`. It records what is logged and recommends a cleanup; logging is unchanged. The downstream trainer logs nothing to W&B. Evidence and the extraction script are in `outputs/step7-wandb-inventory/`.

#### Evidence and how to read the tables

**Evidence sets** (W&B run files `run-<id>.wandb` and the local mirror `wandb_history.jsonl`, read by `extract_keys.py`):

| Set | Runs | Code | W&B |
| --- | --- | --- | --- |
| `online_attempt1_20261027` | 5 methods | Step 7 logging before the terminal revisions | online, project `lcg-reward-free-exploration-logging-smoke`, group `exploration-logging-smoke-20261027` |
| `online_20261027` | 5 methods (LCG killed and resumed) | same | online, same project, group `exploration-logging-smoke-20261027-final` |
| `terminal_v1_20261027` | 5 methods | first terminal revision | offline (never synced) |
| `v2_20261028` | 5 methods | terminal v2; the W&B keys are identical to `62ecded` and `b1e10c2` (the source diff since then changes no key) | offline |
| `final_20261028` | uniform, LCG | `62ecded` (snapshot tree = commit tree) | offline |
| `resume_fixes_20261028` | LCG, killed before the final evaluation, then resumed | `b1e10c2` (HEAD) | offline |

- **Reviewed configuration:** the smoke schedule.
  - Walker Stand/Walk/Run; 300 initial plus 2 × 100 training env steps; 2 epochs.
  - 2 denoiser and 2 PMPO updates per epoch.
  - A test diagnostic every epoch plus the final one: 4 seeded episodes (3001–3004) of 50 steps each.
  - Action dimension 6; K = 5 members or heads; presets' scaling: LCG fixed/100, the others fixed/1.
  - Key counts below are for this configuration unless a row says otherwise.
- **Evidence tags** (in the "Logged when" column):
  - **[C]** confirmed in the current-code run history (`v2`, `final` or `resume_fixes` sets);
  - **[O]** present only in older evidence (the online dashboard runs or `terminal_v1`); the current code no longer emits it;
  - **[U]** supported by the current code but not exercised in any available run;
  - **[A]** collected automatically by W&B.
- **Record types:**
  - **History** is a row of `wandb.log`. Every history row also carries `epoch` (from `utils.wandb_log`) and `progress/train_env_steps` (from `ExplorationTrainer.with_env_steps`), and is mirrored to `wandb_history.jsonl`.
  - **Summary:** W&B keeps the last value of every history key there automatically. Explicit summary keys are listed under "Summary and configuration".
  - **Configuration:** `wandb.config`.
  - **System:** W&B's own sampler; not logged by our code and not mirrored.
- **X-axes** (`setup_wandb`, `src/exploration_trainer.py:859-881`; globs in `EPOCH_AXIS_METRICS`, `:1377-1380`):
  - `actor_critic/train/*` is plotted against `actor_critic/train/num_batch_train_actor_critic`, the cumulative PMPO update.
  - `denoiser/train/*` is plotted against `denoiser/train/num_batch_train_denoiser`.
  - `denoiser/member_k/train/*` is plotted against `denoiser/member_k/train/num_batch_train_member`.
  - Everything else is plotted against `epoch`.
  - `progress/train_env_steps`, the real training env steps, is on every row and can be chosen as an alternative x-axis.
- **Granularity:**
  - per update: one row per denoiser or PMPO update;
  - per collect: one row per collector call, or per completed episode;
  - per refresh: once per epoch, before PMPO;
  - per epoch: one row;
  - per diagnostic: each test diagnostic (every `evaluation.every` epochs) and once for the final diagnostic.
  - In the reviewed configuration, the update-level rows of one epoch (2 denoiser and 2 PMPO updates) all share one `epoch` value. On the epoch axis they appear as several points at the same x.
- **Abbreviations:**

  | Abbreviation | Meaning |
  | --- | --- |
  | PMPO | the exploration policy's actor-critic update |
  | KL | Kullback–Leibler divergence, in nats |
  | λ-return | the discounted return target of imagined rollouts (γ, λ) |
  | p10/p90/p99 | 10th/90th/99th percentile |
  | std | population standard deviation |
  | GiB | 2^30 bytes |
  | `ens` | full diffusion ensemble |
  | `p2e` | Plan2Explore-style heads |
  | `h_D` | LCG's diagonal historical precision |
  | θ_S (`theta_s`) | the selected parameter subset |
  | RMS | RunningRMS reward normalisation |
  | end_0 / end_1 | dataset steps whose `end` flag is 0 or 1; 1 is a true termination, which exploration rejects |
- **Intrinsic quantities:**
  - **"raw"** is the scorer's own value, in method units: LCG score, pixel variance, feature variance. These units are not comparable across methods.
  - **"scaled"** (`policy_reward`) is what PMPO optimises: raw / `fixed_divisor` (LCG preset 100; others 1.0), or RunningRMS-normalised when `reward_scaling=rms`.
  - Under `actor_critic/train/` the reward values are **per imagined step**, summarised over the last PMPO update's imagined batch.
  - Under `test_eval/` and `final_test_eval/`:
    - **`*_return`** is the **episode total over real test steps**;
    - **`*_per_step`** is that total divided by the scored steps.
- **Source locations** are `file:line` at `b1e10c2`, relative to `src/`:
  - ET = `exploration_trainer.py`;
  - PB = `models/pmpo_beta.py`;
  - P2E = `plan2explore.py`;
  - LAP = `laplace.py`;
  - ENS = `ensemble.py`;
  - LC = `lcg/lifecycle.py`;
  - IE = `intrinsic_eval.py`;
  - COL = `coroutines/collector.py`.
- **Brace notation:** `a_{x,y}` means `a_x` and `a_y`; `{0..4}` means 0 to 4. `extract_keys.py --check-notes` expands every key written in this section and checks it against the evidence and the source (see "Traceability check").

#### Dashboard sections, "Charts" and key counts

- **Which dashboard this is (inferred):**
  - The listed sections match the only synced project, `lcg-reward-free-exploration-logging-smoke`: its 10 online runs (2026-10-27) used **older code**. Every later run is offline and has not been synced.
  - That explains:
    - why the dashboard has `final_test`/`test` with task returns but no `lcg` section;
    - why `lcg/*` (added later) is missing from it.
- **What W&B counts (W&B's documented behaviour; not checkable from local files, so unverified here):**
  - A workspace shows the **union of keys over the project's visible runs**. A section's count therefore combines keys from all five methods and includes keys that only the older code emitted.
  - W&B groups panels by the text before the first `/`. Keys without `/` go to the default section **"Charts"**, which is a dashboard section, not a metric namespace.
- **"Charts" in the evidence:**
  - In our runs the keys without `/` are `epoch`, `duration` and `length`, plus `return`, `final_return_mean` and `final_return_std` in the older runs.
  - W&B's internal `_step`, `_runtime` and `_timestamp` are hidden.
  - Whether "Charts" also shows a panel for `epoch` itself is unverified.
- **Mismatch with the listed sections:** `milestone/collected_steps` appears in the first online attempt (older code), yet no "milestone" section is listed. Either those runs are hidden or deleted in the workspace, or the section is collapsed. This is unverified.
- **"System"** holds W&B's automatic system metrics (see the System table).

#### Category overview

Current = union over the five methods, reviewed configuration, current code. Dashboard = expected union over the 10 online runs (older code).

| Section | Current keys | Per method (uni / LCG / ens / Lap / P2E) | Online dashboard | Main purpose | Largest redundancy |
| --- | --- | --- | --- | --- | --- |
| actor_critic | 188 | 0 / 140 / 140 / 140 / 140 | 188 | per-update PMPO diagnostics and per-step intrinsic reward | 48 per-action-dimension keys; 64 reward stats under four method prefixes; scaled = raw for ens/Lap/P2E (fixed/1); `loss_actor_total` = `loss_actor`; 12 weight stats constant under the sign objective |
| Charts | 3 (`epoch`, `duration`, `length`) | 3 each | 6 (+`return`, `final_return_mean`, `final_return_std`) | axis, epoch duration, episode length | `duration` = `timing/epoch_seconds`/3600; `length` mixes train and test episodes |
| denoiser | 27 | 6 / 6 / 23 / 6 / 6 | 27 | denoiser losses per update; test loss | member-0 rows carry two equal counters; test loss logged per batch (no final aggregate) |
| ensemble | 16 | ens only | 16 | member bootstrap health and counters | `total_denoiser_updates` = `progress/member_denoiser_updates_total`; member update counters equal by schedule |
| epoch_summary | 19 | 2 / 14 / 19 / 14 / 14 | 19 | epoch means of existing metrics | aggregates (not duplicates); scaled = raw mean for fixed/1 |
| final_test | 5 | 5 each | 17 (+12 task returns, removed) | final test-collection counters | overlaps `final_test_eval/num_episodes` and progress counters |
| final_test_eval | 39 | 9 / 39 / 39 / 39 / 39 | 57 (+18 task returns, removed) | per-seed and four-seed intrinsic returns | per-step and scored-steps derivable; scaled = raw for fixed/1 |
| laplace | 23 | Lap only | 23 | precision and perturbation scale per refresh | constants (`selected_dim`, `posterior_temperature`) |
| lcg | 5 | LCG only | 0 (not in older code) | refresh identity | `round` = epoch; `replay_steps` = `progress/train_env_steps` |
| plan2explore | 46 | P2E only | 46 | replay encoding and head fitting | constants (`feature_dim`, `head_parameters`, `fit_updates`) |
| progress | 6 | 5 / 5 / 6 / 5 / 5 | 6 | cumulative counters (env steps, updates) | `test_env_steps` = `test/total_new_env_steps` |
| resume | 0 in current runs (7 in code) | — | 6 | resume and backfill accounting | none; the current final-only resume drops the row (finding 1) |
| test | 5 | 5 each | 17 (+12 task returns, removed) | test-collection counters | `num_episodes` = `test_eval/num_episodes`; `total_new_env_steps` = `progress/test_env_steps` |
| test_dataset | 5 | 5 each | 5 | per-collect test dataset counters | `counts/end_0` = `num_steps`; `counts/end_1` always 0 |
| test_eval | 39 | 9 / 39 / 39 / 39 / 39 | 57 (+18 task returns, removed) | per-seed and four-seed intrinsic returns of each diagnostic | as `final_test_eval` |
| timing | 37 | 7 / 18 / 20 / 22 / 22 | 37 | wall-clock and GPU memory | refresh time logged under up to 3 names; PMPO and test time twice |
| train_dataset | 5 | 5 each | 5 | per-collect replay counters | `num_steps` = `progress/train_env_steps`; `counts/end_*` as test |
| System | 34 (laptop) / 42 (kingplus0) | automatic | 34 | host and GPU telemetry | samples GPUs the run does not use |
| **Total history keys** | **471** (uniform 64, LCG 292, ens 328, Lap 314, P2E 337) | | **536** | | |

#### actor_critic (PMPO updates; history; one row per update; x-axis `actor_critic/train/num_batch_train_actor_critic`; learned methods only)

**Superseded (2026-10-28):** this table lists the earlier flat keys. The implemented schema (groups `core/`, `EMA/`, `gaussian/`, `weights/`, `intrinsic/` and `debug/`) is in "actor_critic metric organization and debug switch" below.

Common keys per method: 124. The method-specific reward statistics add 16 per method (64 in the union). In the reviewed configuration each key has 4 rows: 2 epochs × 2 updates. Source: `PB.forward` `PB:311-353`, `PB.objective_diagnostics` `PB:359-384`, `PB.diagnostics` `PB:387-407`, `PB.temporal_diagnostics` `PB:410-422`, `PMPOOptimizers.step` `PB:438-451`, prefixing `ET:1030-1043`, reward stats `ET:699-711`.

| Exact metric key | Meaning and units | Logged when/how often | X-axis | Applicable methods | Source location | Possible redundancy |
| --- | --- | --- | --- | --- | --- | --- |
| `actor_critic/train/num_batch_train_actor_critic` | Cumulative PMPO update count (1-based, after the update) | History [C]; every update | itself (axis) | learned | ET:1041-1042 | Same count as `progress/pmpo_updates` (the model's `updates` buffer) at epoch end; different purpose (axis vs epoch counter) |
| `actor_critic/train/loss_actor` | Total actor loss = `loss_pmpo_raw` + `loss_kl` + `loss_entropy` | History [C]; every update | PMPO update | learned | PB:308, 312 | **Exact duplicate** of `loss_actor_total` (same tensor) |
| `actor_critic/train/loss_actor_total` | Same tensor as `loss_actor` | History [C] | PMPO update | learned | PB:315 | **Exact duplicate** of `loss_actor` |
| `actor_critic/train/loss_pmpo_raw` | PMPO policy term (sign objective, or advantage-weighted) | History [C] | PMPO update | learned | PB:305, 314 | Sum of `loss_positive` + `loss_negative` under the sign objective (derived) |
| `actor_critic/train/loss_{positive,negative}` | −α·`positive_log_probability`; (1−α)·`negative_log_probability` | History [C] | PMPO update | learned | PB:331-332 | Constant multiples of the log-probability keys (derived) |
| `actor_critic/train/loss_kl` | β_KL·`kl_prior` | History [C] | PMPO update | learned | PB:333 | Constant multiple of `kl_prior` (derived) |
| `actor_critic/train/loss_entropy` | −entropy_coef·`policy_entropy` | History [C] | PMPO update | learned | PB:307, 314 | Constant multiple of `policy_entropy` (derived) |
| `actor_critic/train/loss_value` | Value MSE to the λ-return targets | History [C] | PMPO update | learned | PB:309, 312 | — |
| `actor_critic/train/kl_prior` | KL(current actor ‖ prior snapshot), mean over states, summed over action dims (nats) | History [C] | PMPO update | learned | PB:304, 312 | Equal to `kl_initial` in the smoke only: within the first prior block both references are the initial actor. Not a duplicate in general |
| `actor_critic/train/kl_initial` | KL(current ‖ construction-time actor), diagnostic only (nats) | History [C] | PMPO update | learned | PB:303, 313 | see `kl_prior` |
| `actor_critic/train/{log_probability,positive_log_probability,negative_log_probability}` | Mean log-likelihood of the imagined actions: all; A ≥ 0; A < 0 | History [C] | PMPO update | learned | PB:324-330, 405 | — |
| `actor_critic/train/{policy_entropy,policy_entropy_std}` | Entropy of the Beta policy in environment units (nats): mean and std over states | History [C] | PMPO update | learned | PB:405-406 | — |
| `actor_critic/train/policy_entropy_dim_{0..5}` | Per-action-dimension entropy (6 keys; one per action dimension) | History [C] | PMPO update | learned | PB:340-342 | Per-dimension breakdown of `policy_entropy` |
| `actor_critic/train/{alpha,beta}_{mean,std,min,max,p10,median,p90}` | Beta concentration parameters over the batch (14 keys) | History [C] | PMPO update | learned | PB:391-392, 67-72 | Distribution detail |
| `actor_critic/train/{alpha,beta}_near_floor_fraction` | Fraction within 0.01 of the concentration floor | History [C] | PMPO update | learned | PB:393 | — |
| `actor_critic/train/{action_mean,action_std}_{0..5}` | Mean and std of the imagined actions taken, per dimension (env units; 12 keys) | History [C] | PMPO update | learned | PB:396-397 | Per-dimension detail |
| `actor_critic/train/{policy_mean,policy_mean_state_std,policy_mean_min,policy_mean_max}_{0..5}` | The policy's mean action over states: mean, std, min and max, per dimension (24 keys) | History [C] | PMPO update | learned | PB:398-401 | Per-dimension detail |
| `actor_critic/train/{near_lower_fraction,near_upper_fraction}` | Fraction of the imagined actions within 1% of the bounds | History [C] | PMPO update | learned | PB:402-403 | — |
| `actor_critic/train/fixed_policy_mean_state_std_{0..5}` | Std over 32 fixed probe states of the deterministic action (6 keys) | History [C] | PMPO update | learned | PB:350-352 | Related to `policy_mean_state_std_*` (different states) |
| `actor_critic/train/fixed_{temporal_difference_mean,temporal_variation_fraction,ordered_repeated_action_difference,ordered_reversed_action_difference}` | Frame-stack sensitivity probes on the fixed states | History [C] | PMPO update | learned | PB:410-419 | Debug probes; `fixed_temporal_*` and `stack_temporal_*` measure the same quantity on different states |
| `actor_critic/train/stack_temporal_{difference_mean,variation_fraction}` | Mean frame-to-frame difference in the imagined frame stacks; fraction that change | History [C] | PMPO update | learned | PB:343-346 | see above |
| `actor_critic/train/{value_mean,value_std,value_bias,value_abs_max,value_target_correlation}` | Critic predictions: mean, std, mean(value − target), max\|value\|, correlation with the targets | History [C] | PMPO update | learned | PB:316, 326-338 | — |
| `actor_critic/train/{return_mean,return_std,return_abs_max}` | λ-return targets of the **scaled** intrinsic reward (a discounted sum over the horizon, not an episode total) | History [C] | PMPO update | learned | PB:317, 338 | `epoch_summary/pmpo_return_mean_mean` is its epoch mean |
| `actor_critic/train/advantage_{mean,std,min,max,p10,median,p90}` | Advantage = λ-return − old value (7 keys) | History [C] | PMPO update | learned | PB:321, 67-72 | — |
| `actor_critic/train/{advantage_zero_fraction,advantage_near_zero_fraction}` | Fraction with A = 0; fraction with \|A\| ≤ 1e-6 | History [C] | PMPO update | learned | PB:334-335 | — |
| `actor_critic/train/{positive_fraction,negative_fraction}` | Fraction with A ≥ 0; with A < 0 | History [C] | PMPO update | learned | PB:318 | `negative_fraction` = 1 − `positive_fraction` exactly |
| `actor_critic/train/advantage_abs_{mean,max}_{positive,negative}` | \|A\| mean and max per sign group (4 keys) | History [C] | PMPO update | learned | PB:374, 379-380 | Equal to `advantage_mean`/`max` in the smoke only (no negative advantages) |
| `actor_critic/train/weight_{mean,max,std,clipped_fraction}_{positive,negative}` | Actor sample weights per sign group (8 keys; with the per-group \|A\| keys, 12 objective keys) | History [C] | PMPO update | learned | PB:364-380 | **Constant** under `actor_objective=sign`, the default and preset setting: mean = max = 1, std = 0, clipped = 0. Informative only for `advantage_weighted` |
| `actor_critic/train/actor_rel_distance_from_init` | ‖θ_actor − θ_init‖ / ‖θ_init‖ | History [C] | PMPO update | learned | PB:381-383 | — |
| `actor_critic/train/{actor_grad_norm,value_grad_norm}` | Gradient norm before clipping | History [C] | PMPO update | learned | PB:440-443 | — |
| `actor_critic/train/rollout_length` | Imagination horizon (steps) | History [C] | PMPO update | learned | PB:319 | **Constant** (`imagination_horizon`, a config value) |
| `actor_critic/train/all_finite` | Always 1.0: a non-finite value raises before logging | History [C] | PMPO update | learned | PB:319, 354 | **Constant** |
| `actor_critic/train/lcg_{raw,policy}_reward_{mean,std,min,max,p10,median,p90,p99}` | LCG **per-imagined-step** reward of the last update's batch: raw score and scaled (raw/100 in the preset) (16 keys) | History [C]; every update | PMPO update | LCG | ET:699-711; hook LC:189-196 | scaled = raw/100 exactly (a constant rescale); `epoch_summary/intrinsic_*` aggregates |
| `actor_critic/train/ens_{raw,policy}_reward_{mean,std,min,max,p10,median,p90,p99}` | Full ensemble pixel-variance disagreement per step (16 keys) | History [C] | PMPO update | ens | ET:699-711; ENS:181, 224-229 | **policy = raw exactly** with fixed/1.0 (verified in the evidence) |
| `actor_critic/train/laplace_{raw,policy}_reward_{mean,std,min,max,p10,median,p90,p99}` | Laplace-ensemble disagreement per step (16 keys) | History [C] | PMPO update | Lap | ET:699-711; LAP:117 | **policy = raw exactly** (fixed/1.0) |
| `actor_critic/train/p2e_{raw,policy}_reward_{mean,std,min,max,p10,median,p90,p99}` | Plan2Explore head disagreement (feature variance) per step (16 keys) | History [C] | PMPO update | P2E | ET:699-711; P2E:395, 629-645 | **policy = raw exactly** (fixed/1.0) |

#### denoiser (history)

| Exact metric key | Meaning and units | Logged when/how often | X-axis | Applicable methods | Source location | Possible redundancy |
| --- | --- | --- | --- | --- | --- | --- |
| `denoiser/train/loss_denoising` | EDM denoising loss of one training batch | History [C]; every denoiser update | `denoiser/train/num_batch_train_denoiser` | uniform, LCG, Lap, P2E | `models/diffusion/denoiser.py:164`; ET:1053-1087 | `epoch_summary/denoiser_loss_mean` is the epoch mean |
| `denoiser/train/grad_norm_before_clip` | Gradient norm before clipping | History [C]; every update | same | same | ET:1070-1073 | — |
| `denoiser/train/lr` | Learning rate used by the update (scheduler value) | History [C]; every update | same | same | ET:1078-1080 | Deterministic from the schedule |
| `denoiser/train/num_batch_train_denoiser` | Cumulative denoiser update count (0-based, before the update) | History [C]; every update | itself (axis) | same | ET:1061-1064 | Final value + 1 = `progress/denoiser_updates` |
| `denoiser/member_{0..4}/train/{loss_denoising,grad_norm_before_clip,lr}` | As above, per ensemble member (15 keys) | History [C]; every member update | `denoiser/member_k/train/num_batch_train_member` | ens | ET:1065-1087 | `epoch_summary/denoiser_member_k_loss_mean` is the epoch mean |
| `denoiser/member_{0..4}/train/num_batch_train_member` | Member k's cumulative update count (axis; 5 keys) | History [C] | itself (axis) | ens | ET:1065-1067 | Equal across members by schedule |
| `denoiser/member_0/train/num_batch_train_denoiser` | The global denoiser counter, advanced by member 0 only | History [C] | member-0 axis | ens | ET:1061-1064, 1085 | **Exact duplicate** of `denoiser/member_0/train/num_batch_train_member` (both start at 0 and advance together) |
| `denoiser/test/loss_denoising` | Denoiser loss on one **test batch** (`agent.denoiser`; member 0 for the ensemble) | History [C]; each test diagnostic, **one row per test batch** (2 per diagnostic here) | `epoch` | all | ET:1006-1011, 1090-1105 | `epoch_summary/denoiser_test_loss_mean` averages the periodic ones; the final diagnostic has no aggregate |
| `denoiser/test/num_batch_test_denoiser` | Cumulative test batch counter | History [C]; per test batch | `epoch` | all | ET:1098-1100 | Bookkeeping only |

#### ensemble (history; one row per epoch after member training; `epoch` axis; full ensemble only)

| Exact metric key | Meaning and units | Logged when/how often | X-axis | Applicable methods | Source location | Possible redundancy |
| --- | --- | --- | --- | --- | --- | --- |
| `ensemble/total_denoiser_updates` | Sum of the members' cumulative updates | History [C]; per epoch | `epoch` | ens | ET:689 | **Exact duplicate** of `progress/member_denoiser_updates_total` (same sum; verified per epoch) |
| `ensemble/member_{0..4}_updates` | Member k's cumulative updates (5 keys) | History [C]; per epoch | `epoch` | ens | ET:690-691 | Equal across members by schedule; member 0 = `progress/denoiser_updates` (verified) |
| `ensemble/member_{0..4}_empty_bootstrap_batches` | Cumulative batches whose bootstrap weights were all zero, so the ordinary weights were used (5 keys) | History [C]; per epoch | `epoch` | ens | ET:686-692; `data/batch_sampler.py:33-66` | — |
| `ensemble/member_{0..4}_bootstrap_zero_episode_fraction` | Fraction of replay episodes with Poisson(1) multiplicity 0 for member k (5 keys) | History [C]; per epoch; only if `bootstrap=true` | `epoch` | ens | ET:693-695 | — |

#### epoch_summary (history; one row per epoch; `epoch` axis; switched off by `logging.epoch_summary=false`)

All are **epoch aggregates** of existing per-update or per-batch keys (`ET:803-836`). They are not duplicates: the values are means over an epoch on the epoch axis, while the sources are individual updates on update axes.

| Exact metric key | Meaning and units | Logged when/how often | X-axis | Applicable methods | Source location | Possible redundancy |
| --- | --- | --- | --- | --- | --- | --- |
| `epoch_summary/denoiser_loss_mean` | Mean `loss_denoising` over the epoch's denoiser updates (all members pooled) | History [C]; per epoch | `epoch` | all | ET:814-815 | Aggregate of `denoiser/train/loss_denoising` |
| `epoch_summary/denoiser_member_{0..4}_loss_mean` | Per-member epoch mean (5 keys) | History [C] | `epoch` | ens | ET:816-818 | Aggregate |
| `epoch_summary/denoiser_test_loss_mean` | Mean of this epoch's test-batch losses | History [C]; epochs with a diagnostic | `epoch` | all | ET:819 | Aggregate of `denoiser/test/loss_denoising` |
| `epoch_summary/pmpo_{loss_actor,loss_value,policy_entropy,kl_prior,positive_fraction,negative_fraction,actor_grad_norm,value_grad_norm,return_mean}_mean` | Epoch means of 9 PMPO keys | History [C] | `epoch` | learned | ET:820-821, 1373-1374 | Aggregates; `pmpo_negative_fraction_mean` = 1 − `pmpo_positive_fraction_mean` |
| `epoch_summary/intrinsic_raw_reward_mean` | Mean of the per-update raw per-step means (equal-sized batches, so the mean over all imagined steps) | History [C] | `epoch` | learned | ET:822 | Aggregate of `*_raw_reward_mean` |
| `epoch_summary/intrinsic_raw_reward_max` | Max of the per-update maxima | History [C] | `epoch` | learned | ET:823-824 | Aggregate |
| `epoch_summary/intrinsic_scaled_reward_mean` | Mean of the per-update scaled means | History [C] | `epoch` | learned | ET:825 | **= raw mean** for ens/Lap/P2E (fixed/1, verified); raw/100 for LCG |

#### final_test and test (history; one row per diagnostic; `epoch` axis)

`test/*` is the periodic diagnostic (every `evaluation.every` epochs); `final_test/*` is the single final diagnostic after training. Source `ET:981-984`.

| Exact metric key | Meaning and units | Logged when/how often | X-axis | Applicable methods | Source location | Possible redundancy |
| --- | --- | --- | --- | --- | --- | --- |
| `{test,final_test}/fresh` | 1 if fresh test episodes were collected (`evaluation.collect_test`), else 0 | History [C]; per diagnostic | `epoch` | all | ET:981 | **Constant per run** (configuration; also in `config.provenance.evaluation.collect_test`) |
| `{test,final_test}/num_episodes` | Test episodes in the diagnostic | History [C] | `epoch` | all | ET:981 | **Exact duplicate** of `{test_eval,final_test_eval}/num_episodes` |
| `{test,final_test}/num_steps` | Test-dataset steps (episodes × length) | History [C] | `epoch` | all | ET:981 | = the last `test_dataset/num_steps` of the diagnostic; = `new_env_steps` when fresh (verified) |
| `{test,final_test}/new_env_steps` | Env steps this diagnostic collected (0 when reused) | History [C] | `epoch` | all | ET:976, 982 | Sum of the diagnostic's `test_dataset/new_env_steps` |
| `{test,final_test}/total_new_env_steps` | Cumulative test env steps | History [C] | `epoch` | all | ET:982 | **= `progress/test_env_steps`** in the same epoch (verified, different rows) |
| `{test,final_test}/return_{stand,walk,run}/{mean,std,min,max}` | Downstream task returns (24 keys) | History **[O]**; removed by the terminal revisions (task returns are not logged during exploration) | `epoch` | all (older) | older code only | — |

#### test_eval and final_test_eval (history; one row per diagnostic; `epoch` axis)

The frozen scorer's evaluation on the real test episodes (`ET:906-957`, `IE:149-170`). The prefix is `test_eval` for fresh episodes; with `evaluation.collect_test=false` it becomes `test_dataset_diagnostic` **[U]**, and `final_` is prepended for the final diagnostic. Per-episode labels are `seed_<reset seed>`. Without `evaluation.reset_seeds` they are `episode_<i>` **[U]**.

**Configuration-dependent key count.** Per block: 3 + 2 + 10 + 6·N keys for learned methods, or 3 + 2 + N for uniform, with N episodes.
- The reviewed configuration has N = 4: 39 keys (uniform 9).
- The 100k presets set no reset seeds and `num_final_episodes: 100`. Their final diagnostic would log `final_test_eval/episode_{0..99}/*`: 615 keys (uniform 105). The periodic one would log `test_eval/episode_{0..3}/*`. Neither is exercised **[U]**.

| Exact metric key | Meaning and units | Logged when/how often | X-axis | Applicable methods | Source location | Possible redundancy |
| --- | --- | --- | --- | --- | --- | --- |
| `{test_eval,final_test_eval}/intrinsic_applicable` | 1 for learned methods, 0 for uniform | History [C]; per diagnostic | `epoch` | all | ET:931 | **Constant per method** (configuration) |
| `{test_eval,final_test_eval}/intrinsic_scored` | 1 if the frozen scorer scored the episodes | History [C] | `epoch` | all | ET:931 | Equals `intrinsic_applicable` whenever a scorer exists (all evidence) |
| `{test_eval,final_test_eval}/num_episodes` | Episodes evaluated | History [C] | `epoch` | all | ET:932 | = `{test,final_test}/num_episodes` |
| `{test_eval,final_test_eval}/seed_{3001..3004}/length` | Episode length (steps), **per seed** | History [C] | `epoch` | all | ET:927, 933-935 | Mixed into the ungrouped `length` as well |
| `{test_eval,final_test_eval}/seed_{3001..3004}/intrinsic_raw_return` | **Episode total** of the raw per-step score, per seed | History [C] | `epoch` | learned | IE:153 | — |
| `{test_eval,final_test_eval}/seed_{3001..3004}/intrinsic_scaled_return` | Episode total of the scaled score (frozen scale), per seed | History [C]; absent if no frozen scale | `epoch` | learned | IE:155-157 | **= raw** for ens/Lap/P2E (fixed/1; verified); raw/100 for LCG |
| `{test_eval,final_test_eval}/seed_{3001..3004}/intrinsic_{raw,scaled}_per_step` | Episode total / scored steps (mean per step), per seed | History [C] | `epoch` | learned | IE:154, 157 | **Derived** (`*_return` / `intrinsic_scored_steps`) |
| `{test_eval,final_test_eval}/seed_{3001..3004}/intrinsic_scored_steps` | Steps scored | History [C] | `epoch` | learned | IE:152-153 | **= length** whenever the episode has its final observation (all evidence; `IE:28-36`) |
| `{test_eval,final_test_eval}/{length,intrinsic_scored_steps}_{mean,std}` | **Four-seed** mean and std (population) | History [C] | `epoch` | length: all; scored steps: learned | IE:161-170 | `length_mean` = `test/num_steps` / `num_episodes`; std is 0 under a fixed time limit |
| `{test_eval,final_test_eval}/intrinsic_{raw,scaled}_{return,per_step}_{mean,std}` | **Four-seed** mean and std of the per-seed values (8 keys) | History [C] | `epoch` | learned | IE:161-170 | Scaled = raw for fixed/1; per-step aggregates derivable |
| `{test_eval,final_test_eval}/return_{stand,walk,run}_{mean,std}`, `{test_eval,final_test_eval}/seed_{3001..3004}/return_{stand,walk,run}` | Task returns (36 keys) | History **[O]**; removed | `epoch` | all (older) | older code only | — |
| `{test_dataset_diagnostic,final_test_dataset_diagnostic}/…`, `…/episode_{i}/…` | Same keys for reused test episodes or unseeded labels | History **[U]** | `epoch` | all | ET:911, 934 | — |

#### test_dataset and train_dataset (history; per collect or completed episode; `epoch` axis)

Source `COL:136-156`: a row per completed episode (`episode_id`, `length`) and a row at the end of each collector call. The episode's task `return` is stripped (`ET:1388-1391`). Test episode ids are offset by the earlier test episodes (`ET:571-572`).

| Exact metric key | Meaning and units | Logged when/how often | X-axis | Applicable methods | Source location | Possible redundancy |
| --- | --- | --- | --- | --- | --- | --- |
| `{train_dataset,test_dataset}/episode_id` | Id of a completed episode | History [C]; per completed episode | `epoch` | all | COL:136-139 | Bookkeeping |
| `{train_dataset,test_dataset}/num_steps` | Dataset steps after the collect | History [C]; per collect | `epoch` | all | COL:151 | train: = `progress/train_env_steps` in the same row; test: see `test/num_steps` |
| `{train_dataset,test_dataset}/new_env_steps` | Env steps of this collect (including discarded unfinished episodes) | History [C]; per collect | `epoch` | all | COL:152 | train: the epoch's sum is `progress/new_train_env_steps` (the initial collection is outside epoch 1) |
| `{train_dataset,test_dataset}/counts/end_0` | Steps with end = 0 | History [C]; per collect | `epoch` | all | COL:153 | **= `num_steps`** (exploration raises on true terminations, `ET:544-546`) |
| `{train_dataset,test_dataset}/counts/end_1` | Steps with end = 1 (true terminations) | History [C]; per collect | `epoch` | all | COL:154 | **Always 0** (otherwise the run stops) |

#### Charts (keys without "/"; history)

| Exact metric key | Meaning and units | Logged when/how often | X-axis | Applicable methods | Source location | Possible redundancy |
| --- | --- | --- | --- | --- | --- | --- |
| `epoch` | Epoch number; the step metric of every epoch-axis key | History [C]; **on every row** | — (axis) | all | `utils.py:269-271`; ET:863 | — |
| `duration` | Epoch wall time in **hours** | History [C]; per epoch | `epoch` | all | ET:502 | **= `timing/epoch_seconds`/3600** to the logged precision (measured microseconds apart; verified) |
| `length` | Length of each completed **train or test** episode | History [C]; per completed episode | `epoch` | all | `data/episode.py:52`; COL:139 | Mixes train and test episodes in one chart; the test values repeat `test_eval/seed_*/length` |
| `return` | Env-reward total of each completed episode (a task return) | History **[O]**; stripped since the terminal revisions | `epoch` | all (older) | `data/episode.py:52` (stripped at ET:1388-1391) | — |
| `final_return_{mean,std}` | DIAMOND's original final-test task return | History **[O]**; removed in `62ecded` | `epoch` | all (older) | older code only | — |
| `_step`, `_runtime`, `_timestamp` | W&B's row index, seconds since start, and wall time | History [A]; every row | — | all | W&B | Hidden internals |

#### laplace (history; per refresh, i.e. per epoch with PMPO; `epoch` axis; Laplace only)

Source `LAP:160-189`.

| Exact metric key | Meaning and units | Logged when/how often | X-axis | Applicable methods | Source location | Possible redundancy |
| --- | --- | --- | --- | --- | --- | --- |
| `laplace/selected_dim` | Selected parameters (scalars) | History [C]; per refresh | `epoch` | Lap | LAP:180 | **Constant** per run (derivable from the shapes in `config.provenance.method_settings.laplace`) |
| `laplace/posterior_temperature` | T | History [C] | `epoch` | Lap | LAP:180 | **Constant** (configuration) |
| `laplace/refreshes` | Cumulative refreshes | History [C] | `epoch` | Lap | LAP:181 | = epochs with PMPO so far |
| `laplace/precision_{mean,min,p01,median,p99,max}` | LCG historical precision h over the selected parameters (6 keys) | History [C] | `epoch` | Lap | LAP:181-182 | — |
| `laplace/perturbation_std_{min,median,max}` | sqrt(T/h) | History [C] | `epoch` | Lap | LAP:183 | — |
| `laplace/selected_weight_norm` | ‖θ̂_S‖ | History [C] | `epoch` | Lap | LAP:184 | — |
| `laplace/member_{0..4}_perturbation_norm` | ‖ε_k·scale‖ per member (5 keys) | History [C] | `epoch` | Lap | LAP:187 | `relative_perturbation` × `selected_weight_norm` |
| `laplace/member_{0..4}_relative_perturbation` | perturbation_norm / selected_weight_norm (5 keys) | History [C] | `epoch` | Lap | LAP:188 | **Derived** from the two keys above |

#### lcg (history; per refresh; `epoch` axis; LCG only; current code only, so not on the dashboard)

| Exact metric key | Meaning and units | Logged when/how often | X-axis | Applicable methods | Source location | Possible redundancy |
| --- | --- | --- | --- | --- | --- | --- |
| `lcg/round` | Refresh round id | History [C]; per refresh | `epoch` | LCG | ET:597, 1398 | **= `epoch`** (the round id is the epoch; verified) |
| `lcg/replay_steps` | Replay steps the precision was computed on | History [C] | `epoch` | LCG | ET:1398 | **= `progress/train_env_steps`** in the same row (verified) |
| `lcg/theta_s_dim` | θ_S size (scalars) | History [C] | `epoch` | LCG | ET:1399 | **Constant** (configuration guard `expected_dim`) |
| `lcg/{h_D_fingerprint,banks_fingerprint}` | Sum of h_D; sum of the bank's sigmas (identity checks, used by resume verification) | History [C] | `epoch` | LCG | ET:1399-1400 | — |

#### plan2explore (history; per refresh; `epoch` axis; P2E only)

Source `P2E:336-369` (counts), `P2E:499-529` (refresh), `P2E:571-603` (head fitting).

| Exact metric key | Meaning and units | Logged when/how often | X-axis | Applicable methods | Source location | Possible redundancy |
| --- | --- | --- | --- | --- | --- | --- |
| `plan2explore/{episodes,encoded_frames,eligible_transitions,excluded_no_successor,padded_context_transitions,successor_transitions}` | Replay encoding counts (6 keys) | History [C]; per refresh | `epoch` | P2E | P2E:336-369, 520 | Bookkeeping; `eligible_transitions` is the fit-set size |
| `plan2explore/{feature_dim,head_parameters}` | Feature size; parameters per head | History [C] | `epoch` | P2E | P2E:521 | **Constant** (configuration and architecture) |
| `plan2explore/{cache_reused,cache_mb,refreshes}` | Whether the feature cache was reused (0/1); its size in MiB; cumulative refreshes | History [C] | `epoch` | P2E | P2E:522-523 | Bookkeeping |
| `plan2explore/feature_std_{min,mean,max}`, `plan2explore/feature_mean_abs_mean`, `plan2explore/near_constant_coordinates` | Feature standardisation statistics | History [C] | `epoch` | P2E | P2E:524-527 | — |
| `plan2explore/member_{0..4}_{loss_first,loss_last,loss_last10_mean}` | Head k's fit loss: first batch, last batch, mean of the last 10 (15 keys) | History [C] | `epoch` | P2E | P2E:599-601 | `loss_last` is one noisy batch; `loss_last10_mean` smooths it |
| `plan2explore/member_{0..4}_fit_updates` | Cumulative fit updates of head k (5 keys) | History [C] | `epoch` | P2E | P2E:602 | Equal across heads by schedule |
| `plan2explore/member_{0..4}_{empty_bootstrap_fallbacks,bootstrap_zero_episode_fraction}` | Bootstrap health per head (10 keys) | History [C] | `epoch` | P2E | P2E:580, 603 | — |

#### progress (history; `epoch` axis)

| Exact metric key | Meaning and units | Logged when/how often | X-axis | Applicable methods | Source location | Possible redundancy |
| --- | --- | --- | --- | --- | --- | --- |
| `progress/train_env_steps` | Real training env steps in replay when the row is logged | History [C]; **every row** (alternative x-axis), and in the epoch-summary row | `epoch` | all | ET:720, 728-732 | = `train_dataset/num_steps` in collection rows (an intended axis copy) |
| `progress/test_env_steps` | Cumulative test env steps | History [C]; per epoch | `epoch` | all | ET:721 | = `test/total_new_env_steps` (verified) |
| `progress/denoiser_updates` | Global denoiser update count (member 0's for the ensemble) | History [C]; per epoch | `epoch` | all | ET:722 | = `ensemble/member_0_updates` (verified) |
| `progress/pmpo_updates` | PMPO update count | History [C]; per epoch | `epoch` | all (0 for uniform) | ET:723 | Last `actor_critic/train/num_batch_train_actor_critic` |
| `progress/member_denoiser_updates_total` | Sum over members | History [C]; per epoch | `epoch` | ens | ET:724-725 | **= `ensemble/total_denoiser_updates`** |
| `progress/new_train_env_steps` | Env steps collected in this epoch | History [C]; per epoch | `epoch` | all | ET:828 | Epoch sum of `train_dataset/new_env_steps` |

#### resume (history; one row at the first epoch after a resume, plus backfilled rows; `epoch` axis)

| Exact metric key | Meaning and units | Logged when/how often | X-axis | Applicable methods | Source location | Possible redundancy |
| --- | --- | --- | --- | --- | --- | --- |
| `resume/from_epoch`, `resume/count` | Epoch resumed from; resumes of this run folder | History **[O]** (online LCG run); current code [U] (see finding 1) | `epoch` | all | ET:418 | `resume_count` is also in the summary |
| `resume/wandb_committed_rows` | W&B rows committed by the checkpoint | History [O]; current [U] | `epoch` | all | ET:773, 778 | — |
| `resume/wandb_server_rows` | Rows the server holds (`starting_step`); online only | History [O]; current [U] | `epoch` | all | ET:774 | — |
| `resume/backfilled_rows` | Committed rows re-logged from the mirror | History [O]; current [U] | `epoch` | all | ET:793 | — |
| `resume/backfilled` | 1 on every re-logged row | History [O]; current [U] | `epoch` | all | ET:792 | — |
| `resume/server_rows_beyond_checkpoint` | Server rows past the checkpoint (an uncommitted epoch about to be replayed) | History [U] | `epoch` | all | ET:796 | — |

#### timing (history; `epoch` axis; wall-clock seconds unless the name says otherwise; monitoring only)

| Exact metric key | Meaning and units | Logged when/how often | X-axis | Applicable methods | Source location | Possible redundancy |
| --- | --- | --- | --- | --- | --- | --- |
| `timing/epoch_{collection,denoiser,scorer_refresh,pmpo,test}_seconds` | Seconds of each phase of the epoch (5 keys; refresh and PMPO for learned methods only) | History [C]; per epoch (epoch-summary row) | `epoch` | all / learned | ET:486, 629, 643, 992, 830-831 | `epoch_pmpo_seconds` **= `pmpo_update_seconds`**; `epoch_scorer_refresh_seconds` **= `scorer_refresh_seconds`**; `epoch_test_seconds` **= `test_diagnostic_seconds`** (all exact, verified) |
| `timing/epoch_seconds` | Whole epoch | History [C]; per epoch | `epoch` | all | ET:832 | ≈ `duration`×3600 |
| `timing/epoch_gpu_peak_allocated_gb` | Peak allocated GPU memory over the epoch (GiB) | History [C]; CUDA only | `epoch` | all | ET:829, 833-834 | ≥ `gpu_peak_allocated_gb` |
| `timing/pmpo_update_seconds` | Whole PMPO phase, including imagination and scoring | History [C]; per epoch | `epoch` | learned | ET:647 | **Exact duplicate** of `epoch_pmpo_seconds` |
| `timing/pmpo_updates` | PMPO updates this epoch (a count, not a time) | History [C] | `epoch` | learned | ET:647 | **Constant** = `actor_critic.training.steps_per_epoch` (or `steps_first_epoch`) |
| `timing/pmpo_seconds_per_update` | `pmpo_update_seconds` / `pmpo_updates` | History [C] | `epoch` | learned | ET:648 | Derived |
| `timing/scorer_refresh_seconds` | Scorer refresh, method-neutral name | History [C] | `epoch` | learned | ET:649 | **Exact duplicate** of `epoch_scorer_refresh_seconds` and of the method key below |
| `timing/{lcg,p2e,laplace}_refresh_seconds` | The same refresh time under a method name | History [C] | `epoch` | LCG / P2E / Lap | ET:652, 658, 664 | **Exact duplicates** of `scorer_refresh_seconds` (same variable `t_refresh`; verified) |
| `timing/lcg_{precision,scoring}_seconds` | Precision part of the refresh; LCG hook time inside PMPO | History [C] | `epoch` | LCG | ET:653-654; LC:131, 193 | Parts of the refresh and PMPO times |
| `timing/p2e_{identity,feature_preparation,fit}_seconds` | Parts of the refresh | History [C] | `epoch` | P2E | P2E:528-529 | Their sum ≈ `p2e_refresh_seconds` |
| `timing/p2e_{imagination_feature,scoring}_seconds` | Feature extraction and head scoring inside PMPO | History [C] | `epoch` | P2E | ET:659-660 | Parts of `pmpo_update_seconds` |
| `timing/laplace_{precision,sampling}_seconds` | Parts of the refresh | History [C] | `epoch` | Lap | LAP:185 | Their sum ≈ `laplace_refresh_seconds` |
| `timing/laplace_prediction_seconds` | Member predictions inside PMPO | History [C] | `epoch` | Lap | ET:665; LAP:230 | Part of `pmpo_update_seconds` |
| `timing/laplace_{prediction_calls,shared_prefix_evaluations}` | **Counts** of prediction calls and shared first-step evaluations | History [C] | `epoch` | Lap | ET:666-667; LAP:227-231 | Counts filed under timing |
| `timing/gpu_peak_{allocated,reserved}_gb` | Peak GPU memory over the refresh + PMPO window (GiB) | History [C]; CUDA only | `epoch` | learned | ET:669-670 | Window of the epoch peak |
| `timing/{p2e,laplace}_refresh_gpu_peak_allocated_gb` | Peak over the refresh alone (GiB) | History [C]; CUDA only | `epoch` | P2E / Lap | ET:605-607, 614-616 | Window of the above |
| `timing/denoiser_member_{0..4}_seconds` | Each member's training phase (5 keys) | History [C]; per epoch | `epoch` | ens | ET:685 | Their sum ≈ `epoch_denoiser_seconds` |
| `timing/test_diagnostic_seconds` | Periodic test diagnostic | History [C] | `epoch` | all | ET:993 | **Exact duplicate** of `epoch_test_seconds` |
| `timing/final_test_diagnostic_seconds` | Final diagnostic | History [C]; once | `epoch` | all | ET:993 | — |

#### milestone (history; older evidence only; not a section in the listed dashboard)

| Exact metric key | Meaning and units | Logged when/how often | X-axis | Applicable methods | Source location | Possible redundancy |
| --- | --- | --- | --- | --- | --- | --- |
| `milestone/collected_steps` | Real steps of a published milestone | History **[O]** (first online attempt only); the current code emits no `milestone/` key | `epoch` | all (older) | older code only; the `milestone/*` axis glob remains (ET:1377) | `summary["milestones/<name>"]` holds the counters |

#### Summary and configuration

| Exact metric key | Meaning and units | Logged when/how often | X-axis | Applicable methods | Source location | Possible redundancy |
| --- | --- | --- | --- | --- | --- | --- |
| last value of every history key | Automatic W&B run summary | Summary [A] | — | all | W&B | Mirrors history |
| `resume_count` | Resumes of the run folder | Summary [C]; at start | — | all | ET:876 | `resume/count` (history) |
| `milestones/steps_<n>` (reviewed configuration: `milestones/steps_{000400,000500}`; the online runs: 000600, 000700) | A published milestone's path and identity (denoiser/dataset sha256, counters) as a dict | Summary [C]; at publication and re-set on resume | — | all | ET:1346-1350 | — |
| `_wandb/runtime` | Run time (s) | Summary [A] | — | all | W&B | — |
| config top level: `actor_critic`, `agent`, `checkpointing`, `collection`, `common`, `denoiser`, `env`, `evaluation`, `initialization`, `intrinsic_reward`, `logging`, `milestones`, `rew_end_model`, `static_dataset`, `training`, `wandb`, `world_model_env` | The resolved trainer configuration | Configuration [C]; at `wandb.init` | — | all | ET:116-118 | — |
| config `provenance` | seed, method, method name, method settings (reward scaling; Laplace/P2E/ensemble settings), source identity, evaluation protocol, milestones | Configuration [C]; at start and on resume | — | all | ET:875, 883-904 | Holds the constants also logged as history (above) |
| config `_wandb` | W&B's own metadata | Configuration [A] | — | all | W&B | — |

#### System (automatic; W&B 0.30 system monitor; not logged by our code; not mirrored)

W&B samples these every few seconds against wall time. Key names come from the evidence. The sampler and its interval are W&B defaults; they are not set in our config.

| Exact metric key | Meaning and units | Logged when/how often | X-axis | Applicable methods | Source location | Possible redundancy |
| --- | --- | --- | --- | --- | --- | --- |
| `cpu`, `proc.cpu.threads` | Process CPU percent; thread count | System [A]; periodic | wall time | all | W&B | — |
| `memory_percent`, `proc.memory.{availableMB,percent,rssMB}` | Host memory; the process's memory | System [A] | wall time | all | W&B | — |
| `disk./.{usageGB,usagePercent}`, `disk.<device>.{in,out}` | Disk use; I/O in MB (`<device>` = `C:` on the laptop, `dm-0` on kingplus0) | System [A] | wall time | all | W&B | — |
| `network.{recv,sent}` | Bytes | System [A] | wall time | all | W&B | — |
| `gpu.<i>.{gpu,memory,memoryAllocated,memoryAllocatedBytes,memoryClock,smClock,temp,powerWatts,powerPercent,enforcedPowerLimitWatts}` | Per-GPU utilisation, memory, clocks, temperature, power, **for every visible GPU** | System [A] | wall time | all | W&B | On kingplus0, `gpu.0` is another job's GPU |
| `gpu.<i>.fanSpeed` | Fan (kingplus0 only) | System [A] | wall time | all | W&B | — |
| `gpu.0.{dramActive,smActive,smOccupancy,pipeFp16Active,pipeFp32Active,pipeFp64Active,pipeTensorActive,pipeTensorHmmaActive,pcieRxBytes,pcieTxBytes,nvlinkRxBytes,nvlinkTxBytes}` | Profiling counters (the laptop only: the online dashboard runs) | System [A] | wall time | all | W&B | — |
| `gpu.process.<i>.{gpu,memory,memoryAllocated,memoryAllocatedBytes,powerWatts,powerPercent,enforcedPowerLimitWatts,temp}` | The same, for the GPU the process uses | System [A]; on kingplus0 for the learned methods; absent in the uniform runs (cause unverified) | wall time | all | W&B | Duplicates `gpu.<i>.*` of that GPU |

Counts: 34 keys in the online (laptop) runs and 42 in the kingplus0 learned-method runs (34 in the kingplus0 uniform runs). The sets differ by host: GPU count, fan, the `gpu.process.<i>.*` keys, and profiling counters on the laptop only. Exact lists per run are in `keys/<set>/<method>.json` (`system_keys`).

#### Not in these sections

- **`pmpo/*` and `eval/*`:** logged only in the separate PMPO-diagnostic mode (`pmpo_diagnostic` config; `pmpo_diagnostic.py:248-249, 377-381`), directly through `wandb.log`, so not mirrored **[U]**. They are absent from all evidence.
- **`classification_metrics/*`:** `utils.py:208-219` produces keys only for models that return a confusion matrix; no exploration model does. The `classification_metrics/*` axis glob is therefore dead.
- **The `milestone/*` axis glob:** stale (see the milestone table). The Step 7 "W&B" section above still mentions a `milestone/collected_steps` row; this inventory supersedes it.

#### Findings beyond redundancy (not fixed: this task changes no logging)

1. **The resume row is dropped by a final-only resume.**
   - The `resume/*` start row is logged with the first epoch's rows (`ET:446`). A resume after the last epoch runs no epoch and logs only the final diagnostic, so the row is lost.
   - In the resume-fix smoke this lost `resume/from_epoch`, `resume/count` and `resume/wandb_committed_rows`. The mirror of that run has no `resume/` key.
   - `resume_count` still reaches the summary.
2. **The final test loss has no aggregate.** `denoiser/test/loss_denoising` is per batch, and `epoch_summary/denoiser_test_loss_mean` covers only periodic diagnostics.
3. **Per-episode keys grow with the episode count.** Without reset seeds, every final episode gets its own keys (615 for the 100-episode presets).
4. **`length` mixes train and test episodes** in one chart.
5. **On shared hosts, System panels include GPUs the run does not use.**

#### Recommendations (not implemented)

| Metric or group | Keep / Merge / Remove / Debug-only | Reason | Proposed canonical key, if applicable |
| --- | --- | --- | --- |
| `actor_critic/train/loss_actor_total` | Remove | Same tensor as `loss_actor` | `actor_critic/train/loss_actor` |
| `actor_critic/train/{lcg,ens,laplace,p2e}_{raw,policy}_reward_{mean,max}` | Merge | Four method prefixes for one quantity force separate panels. The method is already in `config.provenance`; units stay method-specific | `actor_critic/train/intrinsic_{raw,scaled}_reward_{mean,max}` |
| `…_{raw,policy}_reward_{std,min,p10,median,p90,p99}` | Debug-only | Distribution detail per update; mean/max carry the training signal, and the epoch summary aggregates them | — |
| `…_policy_reward_*`, `epoch_summary/intrinsic_scaled_reward_mean`, `*_eval/…intrinsic_scaled_*` when scaling is the identity (fixed/1 or none) | Remove (configuration-dependent) | Exactly equal to raw, verified for ens/Lap/P2E. Keep them for LCG (/100) and RMS, where they differ | the raw key |
| `actor_critic/train/{loss_kl,loss_entropy,loss_positive,loss_negative}` | Debug-only | Constant multiples of `kl_prior`, `policy_entropy` and the log-probabilities | — |
| `actor_critic/train/negative_fraction`, `epoch_summary/pmpo_negative_fraction_mean` | Remove | 1 − `positive_fraction` | `…positive_fraction` |
| `actor_critic/train/weight_*_{positive,negative}`, `advantage_abs_{mean,max}_{positive,negative}` | Debug-only (log only for `actor_objective=advantage_weighted`) | Constant under the sign objective | — |
| `actor_critic/train/{action_mean,action_std,policy_mean,policy_mean_state_std,policy_mean_min,policy_mean_max,policy_entropy_dim,fixed_policy_mean_state_std}_{0..5}` (48) | Debug-only | Per-dimension detail; the largest single source of keys | — |
| `actor_critic/train/{alpha,beta}_{std,min,max,p10,median,p90}` | Debug-only | Keep `{alpha,beta}_mean` and `*_near_floor_fraction`, which flag collapse | — |
| `actor_critic/train/fixed_*`, `stack_temporal_*`, `kl_initial`, `actor_rel_distance_from_init`, `advantage_{min,max,p10,median,p90,zero_fraction,near_zero_fraction}`, `value_{std,abs_max}`, `return_{std,abs_max}`, `{log,positive_log,negative_log}_probability`, `policy_entropy_std`, `near_{lower,upper}_fraction` | Debug-only | PMPO-diagnostic study probes; the PMPO-diagnostic mode already covers this | — |
| `actor_critic/train/{rollout_length,all_finite}` | Remove | Constants: the horizon is in config; non-finite values raise | — |
| `actor_critic/train/{loss_actor,loss_value,loss_pmpo_raw,kl_prior,policy_entropy,positive_fraction,actor_grad_norm,value_grad_norm,return_mean,value_mean,value_bias,value_target_correlation,advantage_mean,advantage_std}`, `…/num_batch_train_actor_critic` | Keep | Core PMPO health; the axis | — |
| `denoiser/train/{loss_denoising,grad_norm_before_clip,num_batch_train_denoiser}`, member equivalents | Keep | Core model training; axes | — |
| `denoiser/train/lr`, `denoiser/member_*/train/lr` | Debug-only | Deterministic schedule | — |
| `denoiser/member_0/train/num_batch_train_denoiser` | Remove | Equal to the member-0 axis | `denoiser/member_0/train/num_batch_train_member` |
| `denoiser/test/loss_denoising` (per batch), `denoiser/test/num_batch_test_denoiser`, `epoch_summary/denoiser_test_loss_mean` | Merge | One value per diagnostic, including the final one | `denoiser/test/loss_denoising` (one mean row per diagnostic) and `final_test/denoiser_loss` |
| `ensemble/total_denoiser_updates` | Remove | Equal to `progress/member_denoiser_updates_total` | `progress/member_denoiser_updates_total` |
| `ensemble/member_{k}_updates` | Debug-only | Equal by schedule; member 0 = `progress/denoiser_updates` | — |
| `ensemble/member_{k}_{empty_bootstrap_batches,bootstrap_zero_episode_fraction}` | Keep | Bootstrap health | — |
| `epoch_summary/*` (others) | Keep | Compact epoch view; aggregates, not duplicates | — |
| `{test,final_test}/fresh`, `*_eval/intrinsic_applicable` | Remove | Run constants, already in `config.provenance.evaluation` | — |
| `{test,final_test}/num_episodes` | Remove | Equal to `*_eval/num_episodes` | `{test_eval,final_test_eval}/num_episodes` |
| `{test,final_test}/total_new_env_steps` | Remove | Equal to `progress/test_env_steps` | `progress/test_env_steps` |
| `{test,final_test}/{num_steps,new_env_steps}` | Keep | Evaluation cost; they differ when test data are reused | — |
| `*_eval/seed_*/{intrinsic_raw_return,intrinsic_scaled_return,length}`, `*_eval/intrinsic_{raw,scaled}_return_{mean,std}`, `*_eval/{intrinsic_scored,num_episodes}` | Keep | The per-seed and four-seed results | — |
| `*_eval/seed_*/{intrinsic_*_per_step,intrinsic_scored_steps}`, `*_eval/intrinsic_*_per_step_{mean,std}`, `*_eval/{intrinsic_scored_steps,length}_{mean,std}` | Debug-only (they stay in `evaluation/*.json`) | Derivable: total / steps; scored steps = length | — |
| Unseeded `*_eval/episode_{i}/*` | Merge | Up to 615 keys for 100 final episodes; per-episode detail belongs in `evaluation/*.json` | aggregates only, or set `final_reset_seeds` |
| `{train_dataset,test_dataset}/counts/{end_0,end_1}` | Remove | `end_0` = `num_steps`; `end_1` is always 0 (a true termination stops the run) | — |
| `{train_dataset,test_dataset}/episode_id` | Debug-only | Bookkeeping | — |
| `train_dataset/num_steps` | Remove | Equal to `progress/train_env_steps` in the same row | `progress/train_env_steps` |
| `{train_dataset,test_dataset}/new_env_steps`, `test_dataset/num_steps` | Keep | Per-collect cost | — |
| `length` | Merge | Mixes train and test episodes | `train_dataset/episode_length`, `test_dataset/episode_length` |
| `duration` | Remove | `timing/epoch_seconds` / 3600 | `timing/epoch_seconds` |
| `epoch`, `progress/train_env_steps` (every row) | Keep | Axes | — |
| `lcg/round` | Remove | Equal to `epoch` | `epoch` |
| `lcg/replay_steps` | Remove | Equal to `progress/train_env_steps` in the same row | `progress/train_env_steps` |
| `lcg/theta_s_dim`, `laplace/{selected_dim,posterior_temperature}`, `plan2explore/{feature_dim,head_parameters}`, `timing/pmpo_updates` | Remove from history (keep in `config.provenance`) | Run or schedule constants | config |
| `lcg/{h_D_fingerprint,banks_fingerprint}` | Keep | Identity checks for resume verification | — |
| `laplace/member_*_perturbation_norm` | Remove | `relative_perturbation` × `selected_weight_norm` | `laplace/member_{k}_relative_perturbation` |
| `laplace/{precision_*,perturbation_std_*,selected_weight_norm,member_*_relative_perturbation}` | Keep | Laplace scale diagnostics (the main open question for that method) | — |
| `laplace/refreshes`, `plan2explore/{refreshes,cache_reused,cache_mb,episodes,encoded_frames,excluded_no_successor,padded_context_transitions,successor_transitions}`, `plan2explore/member_*_fit_updates` | Debug-only | Bookkeeping; the fit counters are cumulative and equal across heads by schedule | — |
| `plan2explore/{eligible_transitions,feature_std_*,feature_mean_abs_mean,near_constant_coordinates,member_*_loss_last10_mean,member_*_empty_bootstrap_fallbacks,member_*_bootstrap_zero_episode_fraction}` | Keep | Fit-set size, feature health, head fit | — |
| `plan2explore/member_*_{loss_first,loss_last}` | Debug-only | Single-batch values; `loss_last10_mean` is the stable summary | — |
| `timing/pmpo_update_seconds`, `timing/scorer_refresh_seconds`, `timing/{lcg,p2e,laplace}_refresh_seconds`, `timing/test_diagnostic_seconds` | Remove | Exactly equal to `timing/epoch_{pmpo,scorer_refresh,test}_seconds` | `timing/epoch_{pmpo,scorer_refresh,test}_seconds` |
| `timing/epoch_*`, `timing/final_test_diagnostic_seconds`, `timing/pmpo_seconds_per_update`, method breakdowns (`lcg_{precision,scoring}`, `p2e_{identity,feature_preparation,fit,imagination_feature,scoring}`, `laplace_{precision,sampling,prediction}`) | Keep | Per-method cost comparison | — |
| `timing/laplace_{prediction_calls,shared_prefix_evaluations}`, `timing/gpu_peak_reserved_gb`, `timing/{gpu_peak_allocated_gb,p2e_refresh_gpu_peak_allocated_gb,laplace_refresh_gpu_peak_allocated_gb}`, `timing/denoiser_member_*_seconds` | Debug-only | Counts filed under timing; sub-window peaks of the epoch peak; member split | `timing/epoch_gpu_peak_allocated_gb` |
| `progress/*` (others) | Keep | Cumulative counters | — |
| `resume/*` | Keep | Resume accounting. The final-only resume drop (finding 1) is a separate fix | — |
| `milestone/*`, `classification_metrics/*` axis globs | Remove | Nothing emits them | — |
| System metrics | Keep (automatic) | Optionally restrict the sampled GPUs on shared hosts; the W&B setting for that is unverified for 0.30 | — |

**Effect if adopted (reviewed configuration):** the default per-update PMPO set shrinks from 140 keys per learned method to 23, or 21 when scaling is the identity:
- 15 core keys including the axis;
- `{alpha,beta}_mean` and the two near-floor fractions;
- raw and scaled intrinsic mean and max.

The other removals are the exact duplicates, constants and derived keys listed above. No core loss, axis, per-seed return or identity check is lost.

#### Traceability check

- `extract_keys.py --check-notes` expands every key pattern written in this section and compares it with the evidence keys.
  - Every key in the current-code evidence must match a pattern here.
  - Every pattern here must match an evidence key, or be tagged [U], [O] or [A].
- Output: `outputs/step7-wandb-inventory/check_notes.log`.

#### Verification

**Commands** (repository root, `lcg` env; read-only on the evidence):

```
python outputs/step7-wandb-inventory/extract_keys.py                 # exit 0
python outputs/step7-wandb-inventory/extract_keys.py --check-notes   # exit 0, RESULT: PASS
```

**First command:** writes `keys/<set>/<method>.json` (history, summary, config, system keys, metric definitions, mirror keys), `keys/key_matrix.tsv`, `keys/INDEX.txt` and `keys/same_row_equal_pairs.txt`.
- **Equality scan:**
  - same-row pairs: equal in every shared row, with a varying value;
  - constant pairs;
  - per-epoch pairs: logged once per epoch in different rows.
- The scan finds candidates, not conclusions. In a 2-epoch smoke with 2 updates, many equalities are coincidences:
  - no negative advantages;
  - 2 denoiser and 2 PMPO updates;
  - the prior is still the initial actor.
- Each redundancy above marked "exact" was confirmed against the source, not only against the scan.

**Traceability check:** 613 expanded patterns. All 517 current-code and 602 older evidence keys (history, summary and system) are covered, and so are all 19 configuration groups. Every pattern either has evidence or carries a [U], [O] or [A] tag.

**Spot checks:**
- Source line references were checked at `b1e10c2` (unchanged working tree for `src/`).
- The v2 snapshot differs from `62ecded` only in files with no W&B key changes. `62ecded` and `b1e10c2` differ in logging only in the offline resume row.

**Unverified:**
- the dashboard's own section counts and the visibility of the first online attempt (no W&B access in this task);
- whether "Charts" also draws `epoch`;
- why the kingplus0 uniform runs lack `gpu.process.*`;
- the W&B setting that restricts sampled GPUs.

### Downstream terminal format (2026-10-29)

Committed as `12f2829` (on `94408bb`). Review: `docs/changes/downstream_terminal_format_review.zip` (regenerated with the evaluate-only fix below). Evidence: `outputs/step15-downstream-terminal/`. Terminal presentation only: no W&B logging, comparison tools, new module or logging framework.

- **Files:** `src/downstream_main.py` (banner, plan and headings), `src/downstream_trainer.py` (phase and evaluation bars, evaluation summary, saved-path lines), `src/utils.py` (`Progress` gains `initial=`, the resumed count). Reused: `say`, `Progress`/`rate_text` (tqdm; a space before the unit, `s/update` and `s/ep` below one per second) and `device_report.device_lines`.
- **Once per command:** the banner and the batch plan, also for a single milestone (`Milestone directory:`; a folder: `Milestones folder:`). The banner has:
  - the compute and rendering device lines of the exploration banner: physical GPUs matched by UUID, never inferred from the process's `cuda:0`; anything unverified is labelled as such;
  - the training budget (omitted for evaluate-only);
  - the evaluation protocol, or `none after training (evaluation.after_training=false)`;
  - the output root `<run>/downstream`.

  The plan has domain and task, adaptation seed, `Ensemble member:` (only for a non-zero member or an ensemble milestone), `Mode:` (only for resume and evaluate-only) and the milestones in order, with their recorded `collected_steps`. Only metadata files are read: `read_manifest` for training or resume of one milestone, and the folder's existing preflight. Evaluate-only reads the completed output instead (see the fix below). No model is loaded or initialized for the banner. `downstream_main.py` now calls `setup_visible_cuda_devices` before the folder launcher too, so the launcher's banner names the children's device (they inherit and set the same value); the launcher held no GPU memory in the smoke.
- **Per milestone:** `=== Milestone i/N: <name> ===`, then without blank lines: `Reward-predictor training`, `Task-policy training`, `Evaluation episodes (seeds a-b)` (closed tqdm bars), `Evaluation summary: max | median | mean | min | std` (the stored `return_*` values, two decimals), `Evaluation saved: > <file>`, `Downstream outputs saved: > <output>`. One blank line between milestone blocks.
  - The outputs line is printed after the evaluation, and also when the evaluation fails (the outputs are already published).
  - Resume: bars start at the restored counts, and a phase completed before the checkpoint has no bar.
  - Evaluate-only: the evaluation bar and lines, no training bars and no outputs line.
  - Removed: `Starting on <device>`, `Downstream outputs: ...`, `Evaluation: mean return ...` and the old bar descriptions. Per-seed returns are only in the evaluation YAML (unchanged).
- **Folder children:** the launcher sets `LCG_DOWNSTREAM_BATCH_CHILD=1` in each child's environment, so a child prints only its milestone's lines. The variable is not part of the command or configuration: saved configs, resume checks and the printed recovery commands are unchanged (a recovery command run by hand prints the full banner). No success summary after the last milestone; on a failure the existing batch summary and recovery commands are printed as before.
- **Unchanged** (verified): training budgets, sampling, controller, frozen denoiser and reward model, evaluation seeds, actions, returns and scheduling, checkpoints and resume, output layout and overwrite protection, `metrics.jsonl` and evaluation file contents.
  - `compare_with_head.py`: the committed code (`94408bb`, exported) and this tree ran the same single-milestone and folder runs on identical copies of the tests' tiny run (evaluation on). Weights, `metrics.jsonl`, `summary.yaml`, `config.yaml` (milestone path aside) and the evaluation files are identical: 21 checks over 3 outputs.

**Tests** (`tests/downstream/`):
- `test_terminal_format_single_milestone_folder_and_evaluate_only` (real processes, evaluation on): the banner and plan line by line for one milestone, evaluate-only and a two-milestone folder; singular/plural; one banner; phase order and no blank lines between bars; bar format (100%, n/n, elapsed<remaining, rate with a space before its unit); the summary in the order max, median, mean, min, std, equal to the evaluation file, whose statistics equal its episodes; the saved-path lines; nothing after them (no duplicated child output, no success summary); per-seed episodes kept in the file.
- `test_folder_plan_singular_and_member_lines`: a one-milestone folder (`Milestones folder:`, `1 milestone`, no member line, no summary).
- `test_evaluate_only_needs_only_the_saved_output_not_the_milestone` (real processes; added with the fix below): a tiny run is trained, then its `milestones/` and `dataset/` are moved away. Evaluate-only with a new protocol (seeds 5, 9) succeeds:
  - the banner shows the not-found label, and the plan shows the step count recorded in `summary.yaml`;
  - the evaluation block is correct, and the training outputs are unchanged;
  - the same protocol again is refused and the file is unchanged;
  - with `collected_steps` removed from the summary, the plan says it is not recorded;
  - a missing output prints no banner and gives the existing "No completed downstream output" error.
- The folder tests check the new plan and headings, that every child gets `LCG_DOWNSTREAM_BATCH_CHILD=1`, and that the variable is not in any command.
- `test_resume_matches_uninterrupted_training` (3 crash points): bars start at the restored counts (`2/6`, `0/4`, `2/4`); completed phases have no bar; the last line is the outputs line.
- `test_terminal_output_changes_no_training_state_metrics_or_evaluation`: with progress output on and off, the same training state, global RNG, `metrics.jsonl` and evaluation file.

**Server smokes** (kingplus0 GPU 1, idle beforehand; GPU 0's other job untouched; new folder `/home/bigking/scfeng/LCG-downstream-terminal-20261029`; isolated snapshot, 382 files, hashes verified):
- **Sources (read-only):** copies of `milestones/` and `dataset/` (the replay files the milestones refer to) of two runs in `LCG-compute-matching-20261029`: the uniform-random pilot (`steps_010000`) and the LCG smoke (`steps_000400`, `steps_000500`). Both the originals and the copies were hashed before and after, and are unchanged. Outputs went only to the copies' `downstream/` folders.
- **Settings:** task walk, adaptation seed 0, 5 reward-predictor and 5 PMPO updates per milestone, evaluation seeds 3001–3002 with 50-step episodes, `common.devices=1`, `MUJOCO_GL=egl`, `MUJOCO_EGL_DEVICE_ID=1`, `WANDB_MODE=offline` (downstream does not use W&B). Run under `script` (a pseudo-terminal, 120 columns).
- **A** (`initialization.milestone_path`, steps_010000): exit 0, 9 s. **B** (`initialization.milestones_dir`, 400 then 500): exit 0, 18 s.
- **`verify_smokes.py`: 59 checks passed.**
  - The rendered text, line by line: the banner shows `NVIDIA GeForce RTX 5090 | physical GPU 1 | process cuda:0` and `EGL | physical GPU 1`.
  - Counts: 5 + 5 updates and 10 metrics rows per output; 2 evaluation episodes of at most 50 steps.
  - The summary values equal the evaluation files.
  - Three separate outputs, no work folders left; sources intact.
  - The launcher ran two children, one at a time, and only they used the GPU, only GPU 1.
  - stderr: only torch's `data_source` deprecation warning.
  - Attempt 1 failed one check on a checker bug: process command lines were cut before `milestone_path`. Kept as `verify_smokes_attempt1_checker_bug.log`.

**Fix: evaluate-only without the source milestone (2026-10-29, after review).**
- **Regression:** the first version of this format called `read_manifest` on the source milestone for every single-milestone command. Evaluate-only had never needed the milestone, since a completed output holds its config, summary and policy. Once the milestone was moved or deleted, it failed with `FileNotFoundError: No milestone manifest`.
- **Fix:** `downstream_main.saved_output_milestone` builds the evaluate-only banner from the completed output:
  - the domain comes from its `config.yaml`;
  - the step count comes from `summary.yaml` `milestone.collected_steps`, or the plan reads `training env steps not recorded in the saved output`;
  - a missing milestone path is labelled `(not found; evaluation uses the saved output)`;
  - without a completed output, no banner is printed and `evaluate_saved_output` raises its existing error.

  Nothing else changed: the output lookup, policy loading, evaluation protocol and overwrite protection are as before, and no model is loaded for the banner.
- **Evidence** (`outputs/step15-downstream-terminal/evaluate_only_fix/`):
  - `regression_before_after.log`: the same evaluate-only command after moving the milestone and replay fails with the smoke-snapshot source (exit 1, `No milestone manifest`) and succeeds with the fix (exit 0);
  - `downstream_tests.log`: 43 passed.
- **Server smoke:** not repeated (no GPU smoke needed; A and B do not use evaluate-only). The smoke evidence above is unchanged; its hashes are in `round1_evidence.sha256`.

Rendered output of B (the single-milestone run A is the same with `Milestone directory:`, `1 milestone` and one block):

```
==============================================================================
Downstream task adaptation
  Milestones folder: /home/bigking/scfeng/LCG-downstream-terminal-20261029/sources/lcg-smoke/milestones
  Compute device: NVIDIA GeForce RTX 5090 | physical GPU 1 | process cuda:0
  Rendering device: EGL | physical GPU 1
  Training budget: 5 reward-predictor updates + 5 policy updates (per milestone)
  Evaluation: 2 seeds (3001–3002) | maximum 50 steps per episode
  Output root: /home/bigking/scfeng/LCG-downstream-terminal-20261029/sources/lcg-smoke/downstream
==============================================================================

Downstream batch: 2 milestones
  Domain: walker; task: walk
  Adaptation seed: 0
  Milestones:
  1. steps_000400 | 400 training env steps
  2. steps_000500 | 500 training env steps

=== Milestone 1/2: steps_000400 ===
Reward-predictor training: 100%|██████████| 5/5 [00:01<00:00, 4.81 updates/s]
Task-policy training: 100%|██████████| 5/5 [00:02<00:00, 2.39 updates/s]
Evaluation episodes (seeds 3001-3002): 100%|██████████| 2/2 [00:00<00:00, 3.24 ep/s]
Evaluation summary: max 4.49 | median 4.00 | mean 4.00 | min 3.52 | std 0.48
Evaluation saved: > <...>/lcg-smoke/downstream/steps_000400/walk/seed_0/evaluations/eval_walk_policy-54597599b14e_protocol-33afa0ff3589.yaml
Downstream outputs saved: > <...>/lcg-smoke/downstream/steps_000400/walk/seed_0

=== Milestone 2/2: steps_000500 ===
...
Downstream outputs saved: > <...>/lcg-smoke/downstream/steps_000500/walk/seed_0
```

(Bars shortened here; full transcripts: `A_single.rendered.txt`, `B_folder.rendered.txt`; raw: `server/outputs/*.typescript`.)

### Downstream W&B logging (2026-10-29)

Committed as `9c08542` (on `12f2829`). Review: `docs/changes/downstream_wandb_logging_review.zip`. Evidence: `outputs/step16-downstream-wandb/`. Terminal format unchanged. No new module or logging framework.

**Files:**
- `src/downstream_trainer.py`: run setup, rows, resume, evaluation, evaluate-only.
- `src/models/rew_end_model.py`: masked reward means; `prediction_mae` with debug.
- `src/models/pmpo_beta.py`: `last_reward`, a reference to the rewards given to the lambda returns; `reward_statistics`, now shared with exploration's intrinsic statistics; the `task_reward/` paths in `metric_path`.
- `src/exploration_trainer.py`: uses `reward_statistics`, with unchanged values and keys.
- `config/downstream.yaml`: a `wandb:` section (default `disabled`), and `rew_end_model.debug_metrics: false`.

**Runs:**
- **One run per output:** milestone × task × adaptation seed × denoiser member.
  - Name: `<prefix>-<milestone>-<task>-adaptation-seed<s>[-member<k>]`.
  - Group: `<prefix>-<task>-adaptation-seed<s>[-member<k>]`, so the milestones of one exploration run, task, adaptation seed and member share a group.
  - `<prefix>` is `wandb.name`, or else `<method>-<domain>-exploration-seed<N>` from the run's saved configuration (since 2026-10-30; before, the exploration run folder's name, which under the default layout is `seed_<N>` for every method). `wandb.group` replaces the group; an existing run keeps its name and group.
  - Job type `adaptation`.
- **Folder mode:** each child process starts its own run; the launcher starts none.
- **Modes:** `disabled`, `offline`, `online`. System charts cover only the used GPU (`x_stats_gpu_device_ids`, as in exploration).
- **Config:** the resolved config plus `provenance`:
  - domain and task;
  - exploration method, collection mode and intrinsic-reward flag, from the manifest. The method is `not recorded` when the manifest has none, as for uniform runs.
  - exploration seed, from the run configuration the manifest refers to (`run_config`), or labelled `not recorded`;
  - the run folder name;
  - milestone name, path, collected steps, epoch, identity and replay checksum (`dataset.pt`);
  - denoiser member: index, member count and weights hash;
  - adaptation seed; controller settings and objective; debug flags; both training budgets; the evaluation protocol; source/package versions.
- **Summary:** `status` (`running`, `completed` or `failed` with `error`; a killed process stays `running` and W&B marks it crashed) and `activity` (`adaptation` or `evaluate_only`); `status_history/<UTC time>` (earlier failures); `output`, `resume`, `evaluations/<file>` (kind `after_training` or `evaluate_only`, step, seeds, file) and `last_evaluate_only`.

**Metrics.** Only the three groups, plus their axes:

| Group | Default | With debug |
| --- | --- | --- |
| `reward_model/train/core/` | `loss_reward` (= `loss_rew`, the reward MSE), `predicted_reward_mean`, `target_reward_mean`, both over exactly the unpadded transitions of the reward loss; the axis `num_updates` (hidden): **4 keys** | `rew_end_model.debug_metrics=true` adds `reward_model/train/debug/{loss_total, loss_end, grad_norm, prediction_mae}`: **8 keys** |
| `actor_critic/train/` | The accepted exploration schema (`core/` 11 with the axis `num_batch_train_actor_critic`, `EMA/` 3, `gaussian/` 23 at 6 action dimensions), without `weights/` (sign-only PMPO uses no advantage weights) and `intrinsic/`, plus `task_reward/{reward_mean, reward_max}`: **39 keys** | `agent.actor_critic.debug_metrics=true` adds 93 `debug/` keys: the exploration debug list without `weight_{mean,max,std}_*` and `intrinsic_*`, plus `task_reward_{std,min,p10,median,p90,p99}`. **132 keys** |
| `eval/` | `seed_<s>/return` for each configured seed (30 by default), and `return_{max,median,mean,min,std}` from the saved evaluation; the axis `exploration/train_env_steps` (hidden) = the milestone's recorded `collected_steps` | (no debug keys) |

- **Task-reward statistics:** taken over every imagined step (batch × horizon) of the reward tensor PMPO passed to `compute_lambda_returns`. No masking applies there: imagined episodes end only by horizon truncation. The reward model does not run again.
- **Steps:** one W&B row per update, with step = the update's index in the whole training (reward-model updates first). The evaluation row follows: after training it is at step = total updates; each evaluate-only row is one step later per earlier evaluation file.
- **`metrics.jsonl`:** new keys only: `predicted_reward_mean`, `target_reward_mean`, `task_reward_mean` and `task_reward_max` (plus the debug ones). Every W&B update row is derived from its `metrics.jsonl` row (`wandb_row`). Evaluation YAMLs and `summary.yaml` are unchanged.
- **Debug flags:** they change no loss, update or RNG draw. A test compares training state, global RNG, weights, evaluation file and every `metrics.jsonl` value with W&B on plus both debug flags against W&B off; debug only adds keys. The only difference is PMPO's documented debug probe cache (`fixed_obs` in the extra state).

**Resume:**
- `wandb_run.json` in the work folder (published with the output) names the run, and a resume continues it.
- W&B, the debug flags and `wandb.*` may change at resume; they are in `RESUME_MAY_CHANGE`. The scientific checks are otherwise unchanged.
- The checkpoint committed one row per completed update. The resume compares this with the server's row count (`starting_step`):
  - rows the server lacks are backfilled from `metrics.jsonl`;
  - updates the server already holds beyond the checkpoint are replayed but not logged again, each compared with the interrupted segment's local record (`replayed_{matched,mismatched,unverified}`);
  - an offline segment of an existing run backfills nothing (the server's rows are unknown);
  - a resume of a run trained without W&B starts a run and backfills its committed updates.
- `summary.resume` records what was found.
- W&B's own initialization cannot change training: the RNG state is restored around it.
- Old checkpoints, old configs and old `metrics.jsonl` rows without the new keys still resume; those rows are backfilled without the missing keys.

**Evaluate-only:**
- An existing evaluation (same policy and protocol) is refused before anything runs, and W&B is not touched.
- W&B starts before the evaluation: status `running` (activity `evaluate_only`), then `completed` or `failed` with the error.
- If the output has a `wandb_run.json`, the evaluation row and an `evaluations/<file>` entry (kind `evaluate_only`) are added to that adaptation run.
- Otherwise a new run with job type `evaluate-only` (name suffix `-evaluate-only`) is created. Its provenance comes only from the output (`summary.yaml`, `config.yaml`); exploration fields read `not recorded in the saved output`.
- The source milestone and replay are never read. Local overwrite protection is unchanged.

**Tests** (fake W&B module):
- exact key sets with debug off and on (39/132 actor keys; 4/8 reward keys);
- `loss_reward` = `loss_rew`; the masked means and MAE recomputed by hand on a padded batch; the task-reward statistics equal those of the tensor captured at `compute_lambda_returns`;
- axes and hidden axes; seed keys from the configured seeds; the evaluation row equals the saved file at `exploration/train_env_steps=26`;
- no intrinsic or weight keys; identities, names, groups and provenance;
- separate runs per milestone, and a launcher that never initializes W&B;
- W&B and debug on versus off;
- resume with the server equal to, short of, beyond or unaware of the checkpoint (offline). Every update appears once with the uninterrupted values; the first segment is marked failed (exit code 1);
- legacy resume without W&B and with old `metrics.jsonl` rows;
- evaluate-only with the milestone and replay moved away, into the adaptation run, and as a new labelled run.

Results: full suite **626 passed** (before the summary fix below); after it, downstream + exploration logging + models **292 passed**.

**Online smokes** (kingplus0 GPU 1, idle beforehand; GPU 0's other job untouched; new folder `/home/bigking/scfeng/LCG-downstream-wandb-20261029`; isolated snapshot, 382 files, hashes verified; W&B project `lcg-reward-free-downstream-logging`):
- **Sources:** copies of `milestones/`, `dataset/` and `config/` of the uniform pilot (`steps_010000`) and the LCG compute-matching smoke (`steps_000400`, `steps_000500`). Originals and copies are unchanged.
- **A:** one milestone, 6 + 6 updates, 30 seeds, 50 steps. https://wandb.ai/models-national-tsing-hua-university7947/lcg-reward-free-downstream-logging/runs/j2qlykat
- **B:** a folder of 2 milestones, 5 + 5 updates, 2 seeds:
  - https://wandb.ai/models-national-tsing-hua-university7947/lcg-reward-free-downstream-logging/runs/nb03aprp
  - https://wandb.ai/models-national-tsing-hua-university7947/lcg-reward-free-downstream-logging/runs/ohhfzu4x
- **C:** 6 + 6 updates, checkpoints every 2, SIGKILL after PMPO update 3, then `downstream.mode=resume`: https://wandb.ai/models-national-tsing-hua-university7947/lcg-reward-free-downstream-logging/runs/twig8663
  - The killed process had uploaded nothing: every synced row is timestamped after the resume, and the server held 0 rows. The resume therefore backfilled the 8 committed updates, replayed update 3 and continued.
  - The history has steps 0–12 once each.
- **`verify_wandb.py`: 71 checks passed** against the W&B API and the fetched records:
  - exact keys per row; update and evaluation counts;
  - every synced value equals `metrics.jsonl` and the evaluation files; `exploration/train_env_steps` = 10,000 / 400 / 500;
  - axis definitions;
  - run identities and provenance, including exploration method `lcg` or `not recorded` and exploration seed 0 from `run_config`;
  - exactly 4 runs in the project (no launcher run);
  - no duplicate steps after the resume;
  - sources intact;
  - processes and W&B system metrics only on GPU 1.

  Attempt 1 passed the system-metric check vacuously (checker bug: wrong part of `system.gpu.<i>.*` keys); kept as `verify_wandb_attempt1_checker_bug.log`.
- **After the smokes:** the replay counter now keeps the resume record in the trainer. W&B returns nested summary values flattened, so the counter can't reread them. This path (updates on the server beyond the checkpoint) did not occur in the smoke; it is covered by the fake-W&B tests.
- **Review fixes (2026-10-29, after review; no new GPU smoke):**
  - **Replay mismatch warning.** `check_replayed` built the key `replayed_mismatched` but tested for `mismatched`, so the warning could never print. Fixed: a genuine mismatch counts `replayed_mismatched` and warns `W&B warning: replayed <phase> update <n> (W&B step <s>) differs from the value the interrupted segment sent; the server keeps the earlier value`. The resume test now covers matched, mismatched and unavailable earlier records (the interrupted segment's local record changed or removed); the replayed update is never logged twice.
  - **Status lifecycle.** `begin_status` (adaptation start and resume, evaluate-only start) sets `status=running` and `activity`. A previous `failed` status moves to `status_history/<UTC time>` (status, activity, error), and the current `error` is removed; `finish_wandb` sets `completed`, or `failed` with the error. Evaluate-only used to start W&B only after a successful evaluation, so a failed evaluation was not recorded and a successful retry left the earlier `failed`/`error` in place. Now W&B starts before the evaluation. The overwrite check runs first (same file name as before: `evaluation_path`), so a refused duplicate never marks a run failed.
  - **Tests:** an adaptation run whose final evaluation fails, then a failing and a succeeding evaluate-only, all on the same run identity: failed → failed (history: adaptation) → completed (no current error; history: adaptation, evaluate_only; evaluation row and entry recorded); then a refused duplicate does not touch W&B. A standalone evaluate-only run fails, and its retry (a new run) completes. Downstream + exploration logging + models: **296 passed** on the final code (`review_fixes/focused_tests.log`).
  - **Evidence:** the smoke evidence is unchanged (hashes: `review_fixes/round1_evidence.sha256`; a copy of the round-1 ZIP is in `review_fixes/`).
- **Server housekeeping:** two of my idle wait loops (`while pgrep ...; sleep 10`) from this and the terminal-format smoke still sleep on kingplus0 (PIDs 739019, 746753). Ending them was not permitted from here; they use no GPU.

## Controller migration: Gaussian H10 EMA-critic controller (2026-10-28)

Review: `docs/changes/gaussian_h10_controller_migration_review.zip`. Evidence: `outputs/step8-gaussian-controller/`. Committed `3ee4c28`.

Both stages now use the Stage 3 Gaussian H10 EMA-critic controller. Exploration keeps advantage-weighted PMPO; downstream uses sign-only PMPO. Nothing else changed: intrinsic scoring, reward scaling, replay sampling, denoiser schedules, collection budgets, evaluation protocols, the terminal format, the W&B mirror repair and the frozen-scorer resume. The W&B metric cleanup is still deferred.

### Reference

- **Location:** `kingplus0:/home/bigking/scfeng/LCG-stage3-gaussian-h10-emacritic`. The requested path `LCG-stage3-guassian-h10-emacritic` does not exist ("guassian" is a misspelling).
- **What it is:** a git worktree of `/home/bigking/scfeng/LCG-stage3-gaussian-seed1`, at commit `a13b57c` ("replace beta with guassian controller while same PMPO objective", 2026-10-20), on top of `b976a6e`. That is also this branch's starting commit.
- **Working tree:**
  - modified: `config/agent/pmpo_beta.yaml` and `src/models/pmpo_beta.py` (the slow EMA critic);
  - untracked: `tests/models/test_ema_critic.py`.
- **The settings come from its H10 runs:**
  - Run folder: `outputs/k4-walker-100k-entropy-stage3-seed{0,1,2}-gaussian-h10-emacritic-*`.
  - Their copied source is byte-identical to the reference working tree: `pmpo_beta.py` sha256 `18f6ca55…`, `env_loop.py` `84cbf0b3…`.
  - Overrides:
    - `agent.actor_critic.actor_distribution=tanh_gaussian`;
    - `agent.actor_critic.slow_critic_coef=1.0`, with the default `slow_critic_rate=0.02`;
    - `world_model_env.horizon=10`;
    - `agent.actor_critic.entropy_coef=0.01`, `prior_refresh_interval=500`;
    - `rew_end_model.training.seq_length=19`.
  - Everything else is the shared default: actor and value lr 1e-4, γ 0.985, λ 0.95, α 0.5, β_KL 0.3, gradient clip 10, 32 rollouts per update, warmup 100, and the same sampler.
- **How it was read:** read-only (`git diff`, a tar of `src`, `config`, `tests`, `docs` and the run configs). Nothing on the server was modified. The copies, diffs and the run's resolved config are in `outputs/step8-gaussian-controller/reference/`.

### What the reference controller does (reused as is)

- **Policy.**
  - A diagonal Gaussian over pre-tanh actions: mean `mu`, std = 0.1 + 0.9·sigmoid(raw).
  - Actions are `low + (tanh(x) + 1)/2 · (high − low)`, always within bounds. The deterministic rule is tanh(mu).
  - **Log-probability:** log N(x) − log(1 − tanh²x) − log(width/2), summed over dimensions. Stored actions are inverted in float64, clamped at |tanh| ≤ 1 − 1e-7.
  - **Entropy:** that of the environment action, H(x) + E[log|da/dx|], with the expectation by 32-node Gauss-Hermite quadrature (differentiable, consumes no RNG).
  - **KL:** analytic, between the pre-tanh Normals of the current actor and the prior snapshot (refreshed every 500 updates), summed over dimensions.
- **Critic.**
  - The scalar critic is regularized toward a slow EMA copy: `value_loss = MSE(V, λ-return) + 1.0 · MSE(V, sg(V_slow))`, on the same states.
  - V_slow starts as an exact copy of the new critic. It has no gradients and is outside the optimizer.
  - After every critic optimizer step: V_slow ← 0.98·V_slow + 0.02·V.
  - **Bootstraps and advantages use the ONLINE critic.** The reference has no target network. Its "EMA critic" is this regularizer, so the migration keeps it. A separate target critic for bootstraps would be a different controller.
- **Horizon:** 10 imagined steps per rollout.
- **Repeat-frame initialization:**
  - The policy's four-frame stack at an imagined reset repeats the imagined start frame. That frame is the last frame of the replay prompt (the anchor), exactly as a real reset repeats its first observation.
  - Within the rollout the stack advances one frame per step, and a reset row repeats its new start frame.
  - The world model is still conditioned on the real replay prompt (frames and actions). Only the policy's stack ignores it.
  - In the reference this is a hard-coded change to `env_loop.py` ("repeat-final-frame"), replacing the replay-prompt stack introduced in `772ef85`.
- **Diagnostics:**
  - added: `pretanh_mean_*`, `pretanh_std_*`, `pretanh_std_near_floor_fraction`, `pretanh_mean_abs_gt_3_fraction`, `tanh_inverse_limited_fraction` and `policy_conditional_std_*`;
  - with the slow critic: `loss_value_target`, `loss_slowreg`, `slow_value_mean` and `slow_value_gap`;
  - removed: no `alpha_*`/`beta_*` for the Gaussian actor.
  - `pmpo_diagnostic.py` applies its Beta-concentration stop rules only to the Beta actor.

### Changes in this repository

- **`src/models/pmpo_beta.py`:** the reference code, merged with this branch's advantage-weighted objective option, its objective check at load, and its reward-source guard. A `diff --strip-trailing-cr` against the reference shows only those, plus the items below. New fields:
  - `actor_distribution`;
  - `slow_critic_coef` and `slow_critic_rate`;
  - `imagination_history` (`replay_prompt` | `repeat_anchor`).

  The dataclass defaults stay Beta, no slow critic and `replay_prompt`, as in the reference, so older saved configs keep their meaning.
- **Checkpoint compatibility:**
  - `controller_settings` (distribution, slow critic, frame-history convention, horizon) goes into the controller's `_extra_state`, next to the objective.
  - `check_controller_state` refuses a mismatch with a clear message before any tensor is copied. It runs from `PMPOBeta.load_state_dict`, `set_extra_state`, `Agent.load`, the exploration resume (before the trainer state loads) and the downstream resume (before the reward model loads).
  - State saved before the settings were recorded counts as the legacy Beta / no-slow-critic / replay-prompt controller and is never reinterpreted. The Beta and Gaussian heads have the same shapes, so without this check a Beta state would load silently.
- **`src/coroutines/frame_history.py` (`policy_prompt`), `env_loop.py`, `game/play_env.py`:**
  - the replay prompt seeds the policy stack only under `replay_prompt`;
  - real environments supply no prompt, so collection and evaluation histories are unchanged;
  - replay observations and denoiser targets are untouched.
- **`src/pmpo_diagnostic.py`:** the reference's three Gaussian-aware edits.
- **`src/exploration_trainer.py`:**
  - controller check before loading;
  - `config.provenance.controller` records distribution, slow critic, frame-history convention, horizon, objective, entropy, β_KL and prior refresh (learned methods only).
- **`src/downstream_trainer.py`:** refuses a non-sign objective; controller check before loading.
- **Configs:**
  - `config/agent/pmpo_beta.yaml`: `actor_distribution: tanh_gaussian`, `slow_critic_coef: 1.0`, `slow_critic_rate: 0.02`, `imagination_history: repeat_anchor`; `actor_objective: sign` (shared default).
  - `config/trainer.yaml`: `world_model_env.horizon: 10`.
  - `config/downstream.yaml`:
    - `horizon: 10` and `rew_end_model.training.seq_length: 19`;
    - an explicit `agent.actor_critic.actor_objective: sign`;
    - `_self_` moved last so that this explicit value wins. No other key overlaps the `env`/`agent` groups, so nothing else changes.
  - `config/experiment/walker_lcg_100k.yaml`: `actor_objective: advantage_weighted`, `advantage_weight_cap: 5.0`, as the ensemble, Plan2Explore and Laplace presets already set.
  - Uniform random trains no policy.
- **Downstream:** builds a fresh controller and optimizers from its own config. Milestones hold only the denoiser and replay, so no exploration-policy weights or optimizer state can be inherited. The milestone denoiser stays frozen. Existing milestones remain usable.

### Differences from the reference (and why)

1. **Advantage-weighted PMPO in exploration.** The reference is sign-only; this experiment requires advantage-weighted PMPO in exploration.
   - The weights are as before: per sign group, magnitudes normalized by the group mean, capped at 5, detached.
   - KL, entropy and critic terms are untouched.
   - Downstream is sign-only, like the reference.
2. **The frame-history convention is a recorded setting (`imagination_history`)**, not an unconditional edit of `env_loop.py`. This lets checkpoints record it and lets older configs keep `replay_prompt`. The behaviour under `repeat_anchor` is the reference's.
3. **Controller settings are recorded and checked at load.** The reference does neither.
4. **Gauss-Hermite rule as constants.** The reference calls `numpy.polynomial.hermite.hermgauss(32)`. On this Windows environment that LAPACK call aborts the process once torch has loaded its own OpenMP runtime (`OMP: Error #15`, libiomp5md already initialized). The constants equal `hermgauss(32)` exactly on numpy 1.26.0 (kingplus0) and 1.26.4 (local), and a test recomputes them in a torch-free process.
5. **Horizon 10 and reward-model sequence length 19 are config defaults** instead of command-line overrides.
   - 19 is the reference's override: it kept the H15 value 15 + 4 when moving to horizon 10.
   - The exploration reward model is not trained, so `trainer.yaml`'s `seq_length` formula is left as is.
6. **The reference's tests are merged into existing test files**, extended for this branch's objective, frame and resume behaviour:
   - `tests/models/test_pmpo_integration.py`, `test_pmpo_frames.py`;
   - `tests/exploration/test_pmpo_objective_resume.py`;
   - `tests/downstream/test_downstream_trainer.py`, `test_downstream_eval_resume.py`.

### Checkpoint compatibility

- **Saved and restored:**
  - Gaussian actor, online critic, slow critic (`slow_value.*`), prior and initial-actor snapshots;
  - both Adams and both schedulers;
  - the update counter;
  - PMPO's private RNG and fixed diagnostic states;
  - objective and controller settings.
- **Refused before any weights load:**
  - Beta vs Gaussian;
  - slow critic on, off or at a different rate;
  - `replay_prompt` vs `repeat_anchor`;
  - a different horizon;
  - controller state from before this migration (no recorded settings, read as the legacy Beta controller).

  Tests confirm that nothing was copied when a load is refused.
- **Exploration run folders from before this migration:** `scripts/resume.sh` resumes a folder with its own copied source and saved config, so such runs continue with the Beta controller they started with. Loading their checkpoints with this code and the new config is refused. Start fresh runs for the Gaussian controller.
- **Milestones:** existing denoiser and replay milestones stay valid for fresh downstream adaptation. Downstream never loads a policy from them.
- **Downstream outputs and checkpoints from before this migration:** not resumable (the saved settings differ). Completed outputs still evaluate with their own saved config.

### Verification

- **Full suite:** 576 passed, exit 0 (`tests/full_suite.log`, about 8 min).
- **Affected files:** 377 passed (`tests/focused_tests.log`): models, checkpoint, downstream, resume, the PMPO objective, presets, ensemble, Laplace, Plan2Explore, and exploration logging (the resume fixes).
- **Earlier runs:**
  - `first_run_hermgauss_abort.log`: the LAPACK abort that led to difference 4;
  - `second_run_4_expected_failures_before_test_updates.log`: 4 assertions that encoded the old behaviour (LCG sign-only; replay-prompt stacks in downstream imagination), updated.
- **New tests:**
  - Gaussian:
    - the bounds and the deterministic rule;
    - log-probabilities against torch's own `TransformedDistribution`;
    - finite values and gradients at the action limits;
    - entropy against Monte Carlo;
    - the analytic KL with a frozen prior.
  - The Gauss-Hermite constants equal `hermgauss(32)`.
  - Losses and gradients are finite under both objectives.
  - Only the PMPO term differs between objectives; KL, entropy and critic are identical.
  - The slow critic:
    - starts as an exact copy, with no gradients, outside the optimizer;
    - follows the loss formula;
    - EMA update with the post-step online critic at 0.02.
  - Online-critic bootstraps and advantages.
  - Repeat-anchor stacks:
    - with `env_loop`, endings and cuts;
    - world-model conditioning and replay draws are identical under both conventions;
    - real histories are unchanged.
  - Bit-exact CPU resume (controller, slow critic, prior refresh); exploration crash/resume equal to an uninterrupted run.
  - Refusals: Beta, frame history, slow critic, horizon, legacy state. Exploration and downstream resume refuse before any weights.
  - Downstream: sign-only enforced, fresh controller, slow critic restored.
  - The reference's trainer-level test.
- **Negative controls** (pytest plugins in `negative_control/`; no source edits):
  - With the controller check disabled, all 9 controller-refusal tests fail. The one other selected test passes: the pre-existing downstream incompatible-inputs test (format, milestone and package checks).
  - With replay-prompt stacks forced, all 12 repeat-anchor tests fail. The other 2 selected tests pass: dm_control action-repeat tests that the `-k repeat` filter matched, unrelated to the convention.
- **Server smoke** (kingplus0 GPU 1 idle beforehand; GPU 0's other job untouched; W&B offline, not synced; new folder `/home/bigking/scfeng/LCG-gaussian-h10-migration-20261028`; isolated snapshot, 382 files, hashes verified). Integration only, not learning quality.
  - **Exploration** (LCG preset; 300 + 2 × 100 steps; 20/10 denoiser updates; 4 PMPO updates per epoch; seeded evaluation every epoch and at the end; milestones 400 and 500):
    - phase 1 (`src/main.py`) killed within about 1 ms of epoch 1's commit: exit 137;
    - `scripts/resume.sh`: exit 0, `Resumed at epoch 1`, the pending milestone published, epoch 2, final evaluation.
  - **Downstream** (milestone `steps_000500`, task walk, seed 0; 20 reward-model and 10 PMPO updates; checkpoints every 4):
    - killed right after the first PMPO-phase checkpoint (4 updates): exit 137;
    - `downstream.mode=resume`: exit 0; 2 real evaluation episodes.
  - **`verify_smoke.py`: 48 checks passed.** Among them:
    - resolved configs: exploration advantage-weighted, downstream sign; both tanh-Gaussian, slow critic 1.0/0.02, `repeat_anchor`, horizon 10; downstream reward-model sequence 19;
    - controller and objective recorded in every checkpoint; slow critic saved and lagging;
    - PMPO updates 1–8 exactly once (4 before, 4 after the resume);
    - Gaussian and slow-critic metrics only, no `alpha_*`/`beta_*`;
    - `loss_value = loss_value_target + loss_slowreg`;
    - advantage weights varying in exploration, all 1 downstream;
    - our processes only on GPU 1 in all four phases; the banner shows compute and EGL rendering on physical GPU 1.
  - **Repeat-anchor runtime evidence:** `stack_temporal_variation_fraction` (the fraction of imagined states whose four frames are not all equal) is exactly 0.9 in every exploration and downstream PMPO row. Only the first of the 10 states per rollout is a repeated frame, and imagination has no sampled endings. Replay-prompt stacks would give 1.0 except at padded episode starts.
  - **EMA at run time:** `slow_value_gap` is 0 at update 1 (exact copy), then grows smoothly through the resume: 2e-6 → 8.8e-4 at update 4 → 1.7e-3 at update 5 (the resumed segment) → 6.0e-3.
  - **Verifier attempts:** attempts 1–3 failed on checker bugs, not on the smoke:
    1. an unresolved `${world_model_env.horizon}` in the saved config;
    2. a wrong expectation of 0.8, from misreading the per-state variation fraction as per-pair;
    3. `manifest.json` instead of `manifest.yaml`.

    Their logs are kept.

### Limitations

- Bit-exact resume is shown on CPU (tests). On GPU the smoke checks continuity, not equality with an uninterrupted run.
- The smoke is tiny (4 PMPO updates per epoch, 10 downstream). It says nothing about learning quality.
- The three other exploration presets (ensemble, Plan2Explore, Laplace) are covered by config and unit tests only. The controller code path is shared, and no method-specific problem appeared.
- The settings are the reference's, not tuned. `seq_length: 19` copies a reference override whose rationale is not documented in the reference.
- The W&B metric inventory above predates this migration. Gaussian runs log `pretanh_*`, `policy_conditional_std_*`, `tanh_inverse_limited_fraction` and the slow-critic keys (`loss_value_target`, `loss_slowreg`, `slow_value_mean`, `slow_value_gap`), and no `alpha_*`/`beta_*`. Update the inventory before the cleanup.

## actor_critic metric organization and debug switch (2026-10-28)

Review: `docs/changes/actor_critic_logging_review.zip`. Evidence: `outputs/step9-actor-critic-logging/`. Committed `4b21ea5`.

This implements the agreed exploration actor_critic W&B schema. It supersedes the actor_critic table of the W&B metric inventory above, which describes the earlier Beta keys.

**Scope:** actor_critic logging only. Unchanged:
- the controller (Gaussian H10, slow EMA critic), objectives, losses, checkpointed state and finite checks;
- the terminal format, the resume fixes, reward scaling and the evaluation protocol;
- every other W&B category.

The downstream trainer still writes only `metrics.jsonl`, with flat names.

### Debug switch

- **Setting:** `agent.actor_critic.debug_metrics` (default `false`). It is separate from `log_diagnostics`, which prints JSON to the terminal. It is a logging setting, not a controller setting: it is not part of the recorded controller settings, so a resume may change it.
- **When false, these are not computed:**
  - the fixed-state policy probes (`temporal_diagnostics` and the fixed mean-action spread), and their cache `_fixed_obs`;
  - the KL to the initial actor;
  - the parameter distance from the initial actor;
  - advantage quantiles;
  - per-dimension action, mean-action and entropy statistics;
  - frame-stack statistics;
  - sign-group weight statistics other than the clipped fractions;
  - intrinsic-reward quantiles.
- **Nothing debug-only enters a loss or consumes random numbers.**
  - Debug on and off give identical losses, gradients, parameters, optimizer and scheduler state, slow critic, prior, update counter, and controller and global RNG.
  - The only difference is the debug-only `_fixed_obs` cache in the controller's extra state: the first 32 imagined states seen while debug is on.
- **PMPO diagnostic mode:** turns debug metrics on, because its stop rules read them. This is recorded in `config.provenance.controller.debug_metrics`.

### Schema (exploration W&B, below `actor_critic/train/`)

**Default: 43 keys** (Gaussian with the slow EMA critic, Walker's 6 action dimensions):

| Group | Keys |
| --- | --- |
| `core/` (11) | `loss_actor`, `loss_value`, `kl_prior`, `policy_entropy`, `return_mean`, `value_mean`, `value_target_correlation`, `positive_fraction`, `actor_grad_norm`, `value_grad_norm`, `num_batch_train_actor_critic` (the update axis) |
| `EMA/` (3) | `loss_value_target`, `loss_slowreg`, `slow_value_gap` |
| `gaussian/` (23) | `pretanh_{mean,std}_{mean,std,min,max,p10,median,p90}`; `pretanh_std_near_floor_fraction`, `pretanh_mean_abs_gt_3_fraction`, `tanh_inverse_limited_fraction`; `policy_conditional_std_{0..A-1}` |
| `weights/` (2) | `clipped_fraction_{positive,negative}` |
| `intrinsic/` (4) | `{raw,scaled}_reward_{mean,max}`: per imagined step of the last update. The names are the same for LCG, the full ensemble, Laplace and Plan2Explore. Raw and scaled are both kept, even where they are equal |

**`debug/` (105 more with debug, 148 in total):**
- `loss_pmpo_raw`, `loss_positive`, `loss_negative`, `loss_kl`, `loss_entropy`;
- `log_probability`, `{positive,negative}_log_probability`;
- `value_std`, `value_bias`, `value_abs_max`, `return_std`, `return_abs_max`, `slow_value_mean`;
- `advantage_{mean,std,min,max,p10,median,p90}`, `advantage_zero_fraction`, `advantage_near_zero_fraction`, `negative_fraction`;
- `advantage_abs_{mean,max}_{positive,negative}`, `weight_{mean,max,std}_{positive,negative}`;
- `kl_initial`, `actor_rel_distance_from_init`;
- `action_{mean,std}_d`, `policy_{mean,mean_state_std,mean_min,mean_max}_d`, `policy_entropy_dim_d`, `policy_entropy_std`, `near_{lower,upper}_fraction`;
- `fixed_policy_mean_state_std_d`, `fixed_{temporal_difference_mean,temporal_variation_fraction,ordered_repeated_action_difference,ordered_reversed_action_difference}`;
- `stack_temporal_{difference_mean,variation_fraction}`;
- `intrinsic_{raw,scaled}_reward_{std,min,p10,median,p90,p99}`.

**Rules:**
- Per-dimension keys come from the actual action dimension. The test environment's 2 dimensions give 39 and 112 keys.
- Removed in both modes: `loss_actor_total` (= `loss_actor`), `rollout_length` (configuration) and `all_finite` (non-finite values raise).
- No legacy flat or method-prefixed (`lcg_`/`ens_`/`laplace_`/`p2e_`) names are emitted.
- A Beta controller logs its concentration statistics under `beta/`, never `gaussian/`. It has no `EMA/` keys without the slow critic.

### Implementation

- **`src/models/pmpo_beta.py`:**
  - `debug_metrics`;
  - the gated metric computations in `forward`, `diagnostics` and `objective_diagnostics`. Direct callers such as the PMPO diagnostic still get the full metrics by default.
  - `metric_path` (flat name → group path) and its exact inverse `flat_metric_name`. The controller keeps returning flat names, which the PMPO diagnostic and downstream `metrics.jsonl` use.
- **`src/exploration_trainer.py`:**
  - rows are logged under `metric_path`;
  - `intrinsic_reward_metrics` produces method-neutral names;
  - the update axis is `actor_critic/train/core/num_batch_train_actor_critic`;
  - the terminal intrinsic line and `epoch_summary/*` read the new keys.
  - `epoch_summary/pmpo_negative_fraction_mean` is kept as 1 − the epoch mean of `positive_fraction`. The two are complementary fractions of the same advantages, equal up to float rounding.
  - The PMPO diagnostic turns debug on; the controller provenance records `debug_metrics`.
- **`src/pmpo_diagnostic.py`:** `finish_epoch` maps grouped rows back to flat names.
- **`config/agent/pmpo_beta.yaml`:** `debug_metrics: false`.

### Verification

- **Tests** (existing files extended; the agreed list is written out once in `tests/conftest.py`): **586 passed**, full suite, exit 0. Covered:
  - exact key sets: 43 / 148 at 6 dimensions; 39 / 112 in the trainer with 2 dimensions;
  - the same default set for all four learned methods;
  - no legacy aliases or removed keys;
  - the paths invert exactly, and Beta is labelled `beta/`;
  - the debug-only probes are not called when off (they raise if called);
  - identical training with debug on and off, for the controller alone and through the trainer;
  - terminal and epoch summaries;
  - the PMPO diagnostic and resume.
- **Online smoke:**
  - **Setup:**
    - kingplus0 GPU 1 (idle; GPU 0's job untouched);
    - W&B project `lcg-reward-free-actor-critic-logging`, group `lcg-gaussian-h10-advantage-weighted-debug-off`, run `lcg-gaussian-h10-advweighted-debugoff-seed0`: https://wandb.ai/models-national-tsing-hua-university7947/lcg-reward-free-actor-critic-logging/runs/8jyqydhf
    - the LCG preset with `debug_metrics=false`: 300 + 2 × 100 steps, 2 denoiser and 2 PMPO updates per epoch;
    - evaluation every epoch and at the end (seeds 3001–3004, 50 steps); milestones at 400 and 500.
  - **Result:** exit 0; **29 checks passed** against the synced run (W&B API) and the run files:
    - exactly the 43 agreed keys, none under `debug/`, no legacy keys;
    - updates 1–4, each with all 43 keys;
    - `define_metric` and every per-key expansion on the update counter;
    - every other category's keys and point counts identical to the accepted terminal-format LCG smoke (149 keys);
    - two epoch summaries, two periodic evaluations and one final evaluation; both milestones;
    - processes only on GPU 1.
  - **Terminal line structure:** identical to the accepted run (220 lines; only mid-redraw progress-bar glyphs differ).
  - **Verifier attempts:** attempts 1–3 failed on checker bugs (dotted summary keys, cp950 decoding, W&B's own footer containing the run name). Their logs are kept.

### Baseline two-epoch smokes (2026-10-28)

Review: `docs/changes/baseline_two_epoch_smokes_review.zip`. Evidence: `outputs/step10-baseline-smokes/`.

The accepted LCG smoke (run `8jyqydhf`) was repeated for the other four baselines.
- **Source and project:** the accepted source (`4b21ea5`; the snapshot tree `8745228` is the same tree as the LCG run's). Same W&B project, group `baseline-gaussian-h10-two-epoch-debug-off`, W&B online.
- **Placement:** kingplus0 GPU 1, idle beforehand; computation and EGL rendering on GPU 1; GPU 0's job untouched.
- **Settings:** the same schedule and evaluation as the LCG run, `debug_metrics=false`, and each preset's method settings. Plan2Explore uses `fit_updates=20`, as in the earlier small smokes.

| Method | Run | Exit | Runtime | actor_critic keys | Denoiser updates | PMPO updates |
| --- | --- | --- | --- | --- | --- | --- |
| Uniform random | https://wandb.ai/models-national-tsing-hua-university7947/lcg-reward-free-actor-critic-logging/runs/8s3lu5cx | 0 | 21 s | 0 | 4 | 0 |
| Full ensemble | https://wandb.ai/models-national-tsing-hua-university7947/lcg-reward-free-actor-critic-logging/runs/8j12bthq | 0 | 36 s | 43 | 4 per member, 20 total | 4 |
| Plan2Explore | https://wandb.ai/models-national-tsing-hua-university7947/lcg-reward-free-actor-critic-logging/runs/4pzzc7jr | 0 | 26 s | 43 | 4 | 4 |
| Laplace | https://wandb.ai/models-national-tsing-hua-university7947/lcg-reward-free-actor-critic-logging/runs/pnlxosyy | 0 | 47 s | 43 | 4 | 4 |

`verify_baselines.py`: **136 checks passed** against the synced runs.
- **Learned methods:**
  - the agreed 43 keys, the same set as the LCG run; no debug, legacy or Beta keys;
  - updates 1–4, each with all 43 keys;
  - axis definitions identical to the LCG run;
  - scaled = raw (fixed divisor 1.0);
  - Gaussian, EMA, repeat-anchor and advantage-weighted settings;
  - method settings preserved: 5 members / heads / samples, bootstrap, temperature 1.0.
- **Uniform:** no actor_critic keys, no intrinsic values; evaluations marked not applicable; terminal lines read `intrinsic N/A`.
- **All four:**
  - both epochs, both milestones, two periodic evaluations and one final evaluation;
  - 500 training and 600 separately counted test environment steps;
  - finite values;
  - other categories' keys and point counts identical to the earlier accepted smokes;
  - accepted terminal lines;
  - finished online runs, processes only on GPU 1.
- **LCG run:** state, name, group, history and config hashes unchanged.
- **Verifier attempt 1:** failed on a checker bug (a method-settings lookup); its log is kept.

### W&B logging cleanup (2026-10-28)

Review: `docs/changes/wandb_logging_cleanup_review.zip`. Evidence: `outputs/step11-wandb-cleanup/`. Not committed.

This is the exploration-stage W&B logging now in force; the "W&B metric inventory" section above describes the state before it. Training, the controller, the 43 default actor_critic keys and the 148-key debug mode are unchanged. Only what is published to W&B (and mirrored to `wandb_history.jsonl`) changed. Replay, checkpoints, milestones, evaluation JSON and terminal output keep everything they had.

**Stopping a metric does not remove it from earlier runs.** Runs logged before this change keep their keys (checked: the five accepted runs still have 192/61/228/237/214 history keys). Their panels stay in the workspace until removed by hand (list below).

#### Retained W&B history keys

| Category | Keys | Axis |
| --- | --- | --- |
| `epoch` | the epoch number; defined `hidden=True` (an x-axis, no chart of its own) | — |
| `actor_critic/train/*` | the agreed 43 (148 with `debug_metrics`), unchanged; learned methods only | `actor_critic/train/core/num_batch_train_actor_critic` |
| `denoiser/train/{loss_denoising,num_batch_train_denoiser}` | single-denoiser methods | `denoiser/train/num_batch_train_denoiser` |
| `denoiser/member_<k>/train/{loss_denoising,num_batch_train_member}` | full ensemble, per member (no member-0 copy of the global counter) | `denoiser/member_<k>/train/num_batch_train_member` |
| `denoiser/test/loss_denoising` | one mean per periodic diagnostic | epoch |
| `denoiser/final_test/loss_denoising` | one mean for the final diagnostic | epoch |
| `progress/{train_env_steps,test_env_steps,denoiser_updates_total,pmpo_updates}` | cumulative; `pmpo_updates` for learned methods only; `train_env_steps` is also on every row (alternative x-axis) | epoch |
| `timing/{epoch_collection,epoch_denoiser,epoch_method_refresh,epoch_pmpo,epoch_test,epoch}_seconds`, `timing/pmpo_seconds_per_update`, `timing/intrinsic_scoring_seconds_per_update`, `timing/epoch_gpu_peak_allocated_gb`, `timing/final_test_diagnostic_seconds` | only the phases that ran: no refresh/PMPO/scoring timing for uniform random, no collection in training-only epochs, no GPU peak on CPU. Updated 2026-10-29: `epoch_method_refresh_seconds` replaces `epoch_scorer_refresh_seconds`, `intrinsic_scoring_seconds_per_update` is new (definitions: "Compute matching" below) | epoch |
| `test_eval/*`, `final_test_eval/*` (`test_dataset_diagnostic/*` and `final_test_dataset_diagnostic/*` when test data are reused) | per seed `seed_<s>/intrinsic_{raw,scaled}_return`, and `intrinsic_{raw,scaled}_return_{mean,std}`: 12 keys for 4 seeds; raw and scaled both kept when equal. Unseeded episodes: the 4 aggregates only. Unscored (uniform random, no frozen scorer): none; a missing scale: no scaled keys | epoch |
| `resume/backfilled` | only on rows re-logged from the mirror after a lost upload | epoch |

**Counts.** LCG on Walker: 85 keys (43 + 4 + 9 + 4 + 24 + epoch); P2E and Laplace the same; full ensemble 93 (5 members x 2 instead of 2); uniform random 14. Before: 192, 228, 237, 214 and 61. Since 2026-10-29 (one timing key added): LCG, P2E and Laplace 86, full ensemble 94, uniform random 14.

**progress/denoiser_updates_total** counts the updates actually taken: the sum over ensemble members (`EnsembleTraining.updates`), else the denoiser counter. **progress/test_env_steps** counts every fresh test collection, the final one included; the final diagnostic's row carries it (600 in the smoke schedule), separate from `train_env_steps`.

**Test-loss aggregation.** `mean_test_metrics`: the unweighted mean over the test batches of the per-batch values (each value is that batch's mean loss), the definition of the former `epoch_summary/denoiser_test_loss_mean`. It is now also computed for the final diagnostic. `num_batch_test` still counts batches in the checkpoint.

**Publication filter.** `published_row` drops `denoiser/**/lr` and `grad_norm_before_clip` at the W&B boundary. They stay in the trainer's rows, which the PMPO diagnostic reads. Gradient clipping, finite checks, schedules and counters are unchanged.

#### Removed from W&B history, and where the information is

| Removed | Now |
| --- | --- |
| `duration` | `timing/epoch_seconds` |
| `length`, `train_dataset/*`, `test_dataset/*` (collection rows) | replay and test datasets (episode lengths, counts, termination check); `progress/*`; evaluation JSON |
| `epoch_summary/*` | the per-update metrics it averaged; the terminal intrinsic line computes its own mean/max |
| `test/*`, `final_test/*` | `progress/test_env_steps`; evaluation JSON (`fresh`, `num_episodes`, `num_steps`, `new_test_env_steps`, `total_test_env_steps`); W&B summary `evaluation/<block>` |
| evaluation lengths, per-step values, scored steps, their aggregates, `num_episodes`, `intrinsic_applicable`, `intrinsic_scored`, unseeded `episode_<i>/*` | evaluation JSON (every episode); summary `evaluation/<block>` (status, reason, scorer, file) |
| `denoiser/*/lr`, `grad_norm_before_clip`, `denoiser/test/num_batch_test_denoiser`, `denoiser/member_0/train/num_batch_train_denoiser`, per-batch test losses | trainer rows (PMPO diagnostic); checkpoint counters; one mean per diagnostic |
| `progress/{denoiser_updates,member_denoiser_updates_total,new_train_env_steps}` | `progress/denoiser_updates_total`; cumulative counters |
| `lcg/*`, `ensemble/*`, `plan2explore/*`, `laplace/*` | run constants in `config.provenance.method_settings` (LCG `theta_s_dim`, P2E `head_parameters` and the extractor's `feature_dim`, Laplace `dim`); scorer identity and counters in summary `scorer` (LCG round and fingerprints; member updates and empty bootstrap batches; P2E refreshes, fit updates, fallbacks, cache identity; Laplace refreshes); checkpoints unchanged |
| `timing/{test_diagnostic,pmpo_update,scorer_refresh,*_refresh,lcg_*,p2e_*,laplace_*,denoiser_member_*}*`, `timing/pmpo_updates`, `timing/gpu_peak_{allocated,reserved}_gb`, refresh-window peaks | the canonical timing keys; method timers still run (not logged) |
| `resume/{from_epoch,count,wandb_*,backfilled_rows,server_rows_beyond_checkpoint}` row | summary `resume` (it is now also recorded for a final-only resume, which used to drop the row) |

#### System metrics

`wandb.init` gets `settings=wandb.Settings(x_stats_gpu_device_ids=[<physical index>])`, the run's GPU found by UUID through nvidia-smi (`device_report.physical_gpu_index`). An offline probe on kingplus0 showed that this W&B 0.30 setting uses nvidia-smi (NVML) indices, also under `CUDA_VISIBLE_DEVICES=1`: `[1]` kept only `gpu.1.*` (`outputs/step11-wandb-cleanup/gpu_stats_probe/`). In the smoke, the synced system stream has GPU keys only for GPU 1 (19). CPU, disk, network and process-memory metrics cannot be switched off one by one with a supported setting; hide them in the workspace.

#### Verification

- **Tests:**
  - updated the existing assertions on retired keys;
  - new: every method's exact key set, axes, mirror and summary metadata (all four learned methods and uniform, CPU);
  - new: the test-loss mean over several finite batches and the publication filter.
  - The full suite passed: **592 passed**.
- **Online smoke:** LCG, the accepted small schedule of run `8jyqydhf` (seed 0, 300 + 2 x 100 steps, 2 + 2 updates per epoch, 4 seeds, final evaluation, milestones 400/500). kingplus0 GPU 1 (idle beforehand), group `wandb-cleanup-two-epoch-debug-off`, run https://wandb.ai/models-national-tsing-hua-university7947/lcg-reward-free-actor-critic-logging/runs/nx4gnm77. Exit 0, 51 s.
- **`verify_cleanup.py`: 44 checks passed** against the synced run:
  - exactly the 85 retained keys; the 43 actor_critic keys equal to `8jyqydhf`'s, 4 updates;
  - counters 400/200/2/2, 500/400/4/4, then 600 test steps on the final row;
  - one test loss per diagnostic; 12 evaluation keys per block with population mean/std;
  - summary evaluation status, scorer and milestones;
  - `epoch` hidden; every key on exactly one axis;
  - system GPU keys only for GPU 1; mirror equal to the synced rows;
  - the same trainer terminal lines as `8jyqydhf`; processes only on GPU 1;
  - the five earlier runs unchanged.
- **Verifier attempts 1 and 2** failed on checker bugs (the update counter's own axis record; a path containing "wandb" dropped by the W&B-line filter). Both logs are kept.

#### Dashboard: manual steps (project `lcg-reward-free-actor-critic-logging`)

The workspace was not edited: no supported tool for it is installed (`wandb-workspaces` is absent), and the W&B API has no call for panels. In the W&B UI:
- **Delete these sections:** `epoch_summary`, `train_dataset`, `test_dataset`, `test`, `final_test`, `ensemble`, `plan2explore`, `laplace`, `lcg`, `resume`.
- **Delete these panels from Charts:** `duration`, `length`, `epoch`.
- **Delete these panels in `denoiser`:** `*/lr`, `*/grad_norm_before_clip`, `test/num_batch_test_denoiser`, `member_0/train/num_batch_train_denoiser`.
- **Delete these panels in `test_eval` and `final_test_eval`:** all except the per-seed `intrinsic_{raw,scaled}_return` and `intrinsic_{raw,scaled}_return_{mean,std}`.
- **Delete these panels in `progress`:** `denoiser_updates`, `member_denoiser_updates_total`, `new_train_env_steps`.
- **Delete these panels in `timing`:** all except the nine retained keys.
- **System:** keep GPU utilization and GPU memory (allocated %, allocated bytes) for GPU 1 and the process GPU; hide CPU, disk, network, system/process memory, threads, GPU temperature, power, clocks and fan, and every `gpu.0` panel (from older runs).

Old runs still hold these keys, so W&B can recreate their panels while those runs are shown. Filter the old runs out, or stop the workspace from regenerating panels automatically.

### Compute matching: LCG chunk 32, timing definitions, member counts (2026-10-29)

Review: `docs/changes/compute_matching_review.zip`. Evidence: `outputs/step13-compute-matching/`. Not committed. Per-update throughput only: equal seconds per PMPO update is not equal total cost, and nothing here measures exploration or learning quality.

#### LCG reference configuration (frozen)

- **Before:** the ten-epoch runs (2026-10-29) resolved `intrinsic_reward.candidate_chunk_size: 16` (`outputs/step12-ten-epoch-smokes/server/outputs/lcg/config/trainer.yaml`).
- **Now:** `config/experiment/walker_lcg_100k.yaml` sets 32, the fastest of 16/32/64 in the earlier RTX 5090 benchmark (1.90 / 1.60 / 1.70 s per 480 candidates). The preset test pins (MC 5, chunk 32), and the smoke's resolved config has 32. Old vs new resolved config: `lcg_resolved_config_old_vs_new.diff`; apart from run identity and schedule, only the chunk size differs (plus `compile_wm`, which the ten-epoch run overrode to false; the preset resolves true).
- **Scores do not depend on the chunk size** (`tests/lcg/test_crn.py`). The chunk size is part of the saved scorer settings, so a checkpoint from a chunk-16 run will not resume under 32.
- **Unchanged:** R1 (217,664 parameters), candidate MC 5, 320 historical references with MC 3, damping 1e-4, beta 1, fixed divisor 100, and the controller (Gaussian H10, EMA critic, repeat-anchor, advantage-weighted cap 5, batch 32, 3 denoising steps).
- **Confirmation under the current controller:** 1.328 s per complete update, of which 1.087 s is scoring (320 candidates per update, consistent with 1.60 s per 480), with a 2.94 GB peak.

#### Timing definitions (`timing/*`, epoch axis, monitoring only)

- **`timing/epoch_method_refresh_seconds`** (replaces `epoch_scorer_refresh_seconds`). The method-specific preparation, never the common single-denoiser training:
  - LCG: historical precision and JVP bank;
  - Laplace: historical precision and the K-sample posterior bank;
  - Plan2Explore: replay identity, feature encoding (cache), standardization and head fitting;
  - full ensemble: training of the additional members 1..K-1, plus freezing the members. The members' training is timed inside the existing member loop (one CUDA synchronization at member 1 and one after the last member); training order and behavior are unchanged.
- **`timing/epoch_denoiser_seconds`** is unchanged: all denoiser training of the epoch. For the full ensemble it overlaps the refresh value, so never add the two.
- **`timing/pmpo_seconds_per_update`** is unchanged and is the primary matching metric. It is the PMPO phase's wall time divided by its updates: imagination, intrinsic reward, actor/critic losses, backward passes, optimizer and EMA updates, and the per-update metrics. It excludes refresh, collection, evaluation, checkpoints and W&B.
- **`timing/intrinsic_scoring_seconds_per_update`** (new) is the reward-producing work inside PMPO updates, divided by the updates. It is part of the PMPO time; do not add the two. It covers:
  - LCG: the reward hook (candidate JVP scoring and scaling);
  - Plan2Explore: feature extraction during imagination (`WorldModelEnv.step`) and the head-disagreement hook;
  - full ensemble and Laplace: `WorldModelEnv.predict_next_obs_ensemble`, i.e. all members' predictions (Laplace including the shared prefix) and the disagreement, plus the scaling hook.
  - For the two ensembles the member predictions also produce the imagined next frame. That frame is counted once, inside scoring, where a single-denoiser method counts its one next-frame prediction as imagination.
- **How it is measured:** `utils.RegionTimer` records CUDA events per region (no host synchronization) and counts the outermost region only, so shared work is counted once. It is reset at the start of each PMPO phase and read after the phase's synchronization; the evaluation scoring paths are separate code, so they are never counted. Nothing is checkpointed, so a resumed epoch measures itself.
- **Distortion check:** LCG with the timer off gives 1.3280 s, the same as with it on (1.3283 s).
- **Not logged for uniform random.**

#### Tests

- **Changed files:** `tests/exploration/test_exploration_logging.py`, `test_lcg_preset.py`.
- **Per-method instrumented runs:** known delays were placed in each method's refresh path, scoring path, common denoiser and optimizer step. The runs check that:
  - the refresh covers its own path (K-1 members for the ensemble) but not the common denoiser;
  - scoring covers its own path but not the rest of an update, and sits inside the PMPO time;
  - the final real-test evaluation adds nothing to the scoring time.
- **Timer on vs removed:** weights, optimizers, RNG, replay, rewards and all non-timing values are identical.
- **Other:** exact key sets per method (86, ensemble 94, uniform 14); a nested-region unit test.
- **Full suite:** 601 passed.

#### Benchmark (kingplus0 GPU 1; `bench_update.py`, `run_trials.sh`)

- **Input:** the uniform pilot milestone that the earlier benchmarks used was deleted from the server on 2026-10-29. A replacement with the same recipe was regenerated: `retrain_milestone.sh` runs the uniform preset, seed 0, to 10,000 steps (13,600 denoiser updates) in 37 min. It is read only; its digest is unchanged across all batches.
- **Trials:**
  - every trial uses the same snapshot and inputs and runs sequentially, one process each;
  - each measures 5 warm-up updates, then 20 timed complete updates, synchronized at update boundaries;
  - all methods start from a fresh controller with seed 0 and draw prompts from the same replay;
  - refresh and compilation stay outside the timed updates, and each refresh is measured twice (the warm value is used);
  - full-ensemble members 1..K-1 and Plan2Explore heads beyond 5 are freshly initialized (throughput only);
  - LCG runs with its preset `compile_wm=true`; the other methods are uncompiled, as their presets require.

| Configuration | Median s/update | Ratio to LCG | Scoring s/update | Peak GB |
| --- | --- | --- | --- | --- |
| LCG, chunk 32 (start / end) | 1.3283 / 1.3277 | 1.000 | 1.087 | 2.94 |
| Full ensemble K = 4 / **5** / 6 | 1.170 / **1.446, 1.445** / 1.727 | 0.881 / **1.089, 1.088** / 1.301 | 1.110 / 1.386 / 1.667 | 2.9 |
| Laplace K = 4 / **5** / 6 | 1.078 / **1.319, 1.319** / 1.559 | 0.812 / **0.993, 0.993** / 1.174 | 1.017 / 1.257 / 1.497 | 2.86 |
| Plan2Explore heads 5 / 40 / 320 / 1280 / 2560 | 0.376 / 0.379 / 0.398 / 0.465 / OOM | 0.283 / 0.285 / 0.299 / 0.350 / — | 0.036 / 0.039 / 0.058 / 0.123 / — | 2.9 / 3.1 / 4.2 / 14.9 / >30 |

- **p90:** within 0.3% of the median everywhere.
- **Selected:**
  - full ensemble 5 (1.09, the only count in the 0.90-1.10 band);
  - Laplace 5 (0.99, the largest count in the band).
- **Plan2Explore is unmatched under the tested implementation and hardware constraints.** A head adds about 0.07 ms per update, so 0.90 would need roughly 11,700 heads. That is beyond the 32 GB GPU (2560 heads ran out of memory), and fitting costs 0.88 s per head per refresh (1280 heads: 19 min per refresh, about 75 h of fitting over the 240 refreshes).
  - The nearest feasible configuration measured is 1280 heads, at ratio 0.35.
  - Plan2Explore keeps 5 heads (0.28) as the cheaper baseline: it is the only method with a lower per-update cost than LCG.
- **Presets:** every member count is unchanged.

#### Cost at the preset schedule (`projections.md`)

The preset schedule is 190 collection epochs plus 50 training-only epochs (`training.num_final_epochs`): 240 refreshes, 100,600 PMPO updates and 105,600 updates per denoiser. Collection precedes the refresh, so the collection-epoch refreshes see 5,500 through 100,000 transitions and the 50 training-only refreshes see 100,000 each. The per-update and per-refresh values were measured at a 10k replay; the hour totals are projections. Corrected 2026-10-29: the first version counted only the 190 collection epochs and used replay sizes 5,000-99,500.

| Method | s/PMPO update | Scoring s/update | Refresh s (measured) | PMPO h | Method refresh h | Common denoiser h |
| --- | --- | --- | --- | --- | --- | --- |
| LCG | 1.328 | 1.087 | 7.2 | 37.1 | 0.48 | 1.90 |
| Full ensemble (5) | 1.446 | 1.385 | (members' training) | 40.4 | 7.62, extrapolated (4 x 0.0649 s x 105,600) | 1.90 |
| Laplace (5) | 1.319 | 1.258 | 7.0 | 36.9 | 0.47 | 1.90 |
| Plan2Explore (5) | 0.376 | 0.036 | 7.5 | 10.5 | 1.58, extrapolated (encoding 1.29, linear in each refresh's replay; fitting 0.29) | 1.90 |

- **Common denoiser update:** 0.0649 s (batch 32, 5.1 GB peak).
- **LCG and Laplace refreshes** use 320 reference transitions, so they are independent of the replay size.
- **Totals exclude** collection, evaluation, checkpoints and W&B.
- **Check:** `test_projections.py` (2 passed) pins 240 refreshes, 100,600 / 105,600 updates and the per-refresh replay sizes, and checks that training-only epochs add refreshes and updates.

#### Integration smoke (W&B online, `lcg-reward-free-10epoch-smoke`, group `compute-matching-two-epoch-20261029`)

The selected presets ran for two epochs each (the accepted small schedule), on kingplus0 GPU 1 for compute and EGL rendering:

| Method | Run | Epoch-2 s/update | Epoch-2 scoring s/update | Epoch-2 method refresh s |
| --- | --- | --- | --- | --- |
| LCG | [7cpg3j8i](https://wandb.ai/models-national-tsing-hua-university7947/lcg-reward-free-10epoch-smoke/runs/7cpg3j8i) | 1.325 | 1.081 | 7.25 |
| Full ensemble | [v3szy0xu](https://wandb.ai/models-national-tsing-hua-university7947/lcg-reward-free-10epoch-smoke/runs/v3szy0xu) | 1.464 | 1.399 | 0.52 (4 members x 2 updates) |
| Plan2Explore | [gav7z0nl](https://wandb.ai/models-national-tsing-hua-university7947/lcg-reward-free-10epoch-smoke/runs/gav7z0nl) | 0.384 | 0.037 | 4.15 |
| Laplace | [t1vju8zb](https://wandb.ai/models-national-tsing-hua-university7947/lcg-reward-free-10epoch-smoke/runs/t1vju8zb) | 1.334 | 1.268 | 7.26 |

- **Exit codes:** 0 for all four.
- **`verify_smokes.py` (59 checks):**
  - keys: 86, or 94 for the ensemble; the old refresh key is gone; 43 actor_critic keys;
  - values: all finite;
  - timing relations: scoring < PMPO per update, and the ensemble refresh < its denoiser total;
  - counters and placement: counters 500 / 600, the local mirror equals the synced history, and processes ran only on GPU 1;
  - the five ten-epoch runs are untouched.
- **Agreement:** the warmed epoch-2 per-update times match the benchmark within 2.2% (LCG -0.2%, ensemble +1.3%, Laplace +1.1%, Plan2Explore +2.2%). Epoch 1 includes startup costs (first CUDA kernels, and compilation for LCG) and is not comparable: 2.755 / 1.537 / 1.386 / 0.431 s per update (+107% / +6.3% / +5.1% / +14.8%). The first version of this note said "within 1.3%" without naming the epoch; the logged values are unchanged.

## File formats: generated metadata is JSON only (2026-10-29, simplified 2026-10-30)

Committed as `14db41b` (on `9c08542`). Review: `docs/changes/json_only_metadata_review.zip`. Evidence: `outputs/step18-json-only-smoke/`; the dual-format first version is in `outputs/step17-json-metadata/` and `docs/changes/json_metadata_review.zip`, both unchanged. No full experiment; W&B offline, nothing synced.

**Convention:** YAML for configuration, JSON for generated metadata, JSONL for append-only training metrics.

| Kind | Files | Format |
| --- | --- | --- |
| Configuration | `config/*.yaml`, a run's `config/trainer.yaml` and `.hydra/*.yaml`, a downstream output's `config.yaml` | YAML (unchanged) |
| Milestone metadata | `milestones/steps_N/manifest.json` | JSON only |
| Downstream completion | `<output>/summary.json` | JSON only |
| Downstream evaluation reports | `<output>/evaluations/eval_<task>_policy-<hash>_protocol-<hash>.json` (same stem as before) | JSON only |
| Training metrics | `metrics.jsonl`, `wandb_history.jsonl` | JSONL (unchanged) |
| Other | `wandb_run.json`, checkpoints, weights, replay | unchanged |

**Compatibility with historical generated YAML metadata is intentionally dropped:**
- The current code neither reads nor writes `manifest.yaml`, `summary.yaml` or `eval_*.yaml`.
- A milestone with only `manifest.yaml` is not a milestone: discovery, loading, folder preflight and resume fail with `No milestone manifest at <...>/manifest.json (only manifest.yaml, generated YAML metadata from before 2026-10-29, which is no longer read)`.
- An output with only `summary.yaml` is not a completed output: evaluate-only fails with `No completed downstream output at <...> (no summary.json)`, and the banner and failure report treat it the same way.
- An old `eval_*.yaml` report no longer blocks or counts as an evaluation.
- Nothing converts, deletes or rewrites old files. Historical artifacts stay as they are:
  - earlier smoke outputs on kingplus0 and in `outputs/`;
  - earlier review packages in `docs/changes/`;
  - synced W&B runs.

  Using them with the current code would need their metadata converted by hand; this was not done.

**Implementation (existing modules):**
- **`src/data/milestone.py`:**
  - `write_json`: standard JSON, `allow_nan=False` (a non-finite number raises; none occurs in the generated metadata), written aside and renamed;
  - `read_json`: the file must exist, parse as standard JSON and hold an object; `NaN`/`Infinity` constants are rejected. A missing file is reported, naming a leftover YAML file if one exists (not read).
  - `read_manifest`, `downstream_summary`.
- **`src/downstream_trainer.py`:**
  - `summary.json`;
  - `evaluation_path` (`.json`);
  - duplicate refusal by the report's existence, still before W&B starts;
  - W&B's evaluation step counts `eval_*.json`.
- **`src/downstream_main.py`:** the evaluate-only banner and the failure report check `summary.json`.

Checksums, ensemble members, replay identity and resume checks are unchanged. Training, controllers, reward scaling, sampling, budgets, evaluation protocols, terminal formatting and W&B metric names are untouched.

**Tests:**
- `test_milestone_metadata_is_json_only`: JSON publication, discovery, loading and folder launch. A YAML-only milestone gives the clear error and stays untouched. Malformed JSON, a `NaN` constant and a non-object are rejected. Checksums still checked. Non-finite values are never written.
- `test_downstream_outputs_are_json_only`:
  - the output's files, with summary and report parsing strictly;
  - evaluate-only after the milestone and replay are moved away: banner data, failure report, duplicate refusal before W&B with the report unchanged, a new report with the right W&B step;
  - a malformed `summary.json`;
  - a `summary.yaml`-only output is not complete.
- `test_resume_needs_the_milestone_manifest_json`: without the manifest, the resume stops clearly and keeps the work folder; with it, the resume completes.
- The remaining resume, folder-launch, ensemble-member, W&B and duplicate-evaluation tests run on JSON outputs.
- Results: affected (downstream, resume, Laplace, Plan2Explore) **151 passed**; full suite **633 passed** (exit 0).

**End-to-end smoke** (kingplus0 GPU 1, idle beforehand; GPU 0's other job untouched; compute and EGL rendering on GPU 1; W&B offline, not synced; new folder `/home/bigking/scfeng/LCG-json-only-smoke-20261030`; isolated snapshot, 382 files, hashes verified; `server_smoke.sh`):

| Step | Command (abridged; exact in `server/outputs/*.cmd`) | Exit | Time |
| --- | --- | --- | --- |
| 1. Exploration | `src/main.py +experiment=walker_uniform_100k common.devices=1 training.compile_wm=false collection.train.first_epoch.min=200 collection.train.steps_per_epoch=100 collection.train.num_steps_total=400 training.num_final_epochs=0 denoiser.training.steps_first_epoch=2 denoiser.training.steps_per_epoch=2 evaluation.should=false milestones.at_steps=[300,400] wandb.mode=offline` | 0 | 12 s |
| 2. Load check | `load_milestone` of both discovered milestones (task walk) | 0 | 2 s |
| 3. Downstream | `src/downstream_main.py initialization.milestone_path=<run>/milestones/steps_000400 downstream.task_name=walk downstream.seed=0 common.devices=1 rew_end_model.training.steps=5 actor_critic.training.steps=5 evaluation.seeds=[3001,3002] evaluation.max_steps=50 wandb.mode=offline` | 0 | 10 s |
| 4. Same evaluation again | step 3 + `downstream.mode=evaluate` | 1 (refused) | 3 s |

- **`verify_smoke.py`: 34 checks passed:**
  - exit codes;
  - exactly four generated metadata files (two `manifest.json`, `summary.json`, one `eval_*.json`), all fetched JSON parsing strictly;
  - no `manifest.yaml`, `summary.yaml` or `eval_*.yaml`;
  - configuration YAML (`config/trainer.yaml`, `.hydra/*.yaml`, the output's `config.yaml`) and `metrics.jsonl`/`wandb_history.jsonl` present;
  - both milestones load with their manifests' identities; two epochs, no exploration-policy training;
  - 5 + 5 updates in `summary.json` and `metrics.jsonl`; report seeds and step limit; aggregates equal to the episodes and to the terminal summary;
  - the repeat refused (`FileExistsError ... already exists; not overwriting it`), the report unchanged, no W&B run started (offline run folders unchanged, no `wandb:` output);
  - milestones and replay unchanged during adaptation;
  - processes and banners on GPU 1.

  Attempt 1 failed two checks on checker bugs (a repository script's `summary.json` in the run's copied `scripts/`, and Hydra's echo of `wandb.mode=offline`); kept as `verify_smoke_attempt1_checker_bugs.log`.
- **Output tree** (the run's copied `src/`, `scripts/` and the 3 replay episode files omitted):

```
exploration/
  .hydra/hydra.yaml, .hydra/overrides.yaml        config/trainer.yaml
  checkpoints/state.pt, checkpoints/agent_versions/agent_epoch_00002.pt, checkpoints/info_for_import_script.json
  dataset/test/info.pt, dataset/train/...          main.log, wandb_history.jsonl, wandb_run.json
  milestones/steps_000300/{dataset.pt, denoiser.pt, manifest.json}
  milestones/steps_000400/{dataset.pt, denoiser.pt, manifest.json}
  downstream/steps_000400/walk/seed_0/
    actor_critic.pt, rew_end_model.pt, config.yaml, metrics.jsonl, summary.json, wandb_run.json
    evaluations/eval_walk_policy-170609234f49_protocol-33afa0ff3589.json
```

These tiny budgets show file generation and integration only, not learning quality.

## Default output structure, result collection and comparison (2026-10-30)

Committed as `2c95eb0` (on `14db41b`). Review: `docs/changes/output_structure_review.zip`. Evidence: `outputs/step19-output-structure/`. **Custom output roots: pending.** The smoke is integration evidence (tiny budgets, 2 evaluation episodes, 50-step limit), not a learning-quality result.

**Renames:**
- `src/main.py` → `src/exploration_main.py` and `config/trainer.yaml` → `config/exploration.yaml` (git renames).
- New runs save their resolved configuration as `config/exploration.yaml`, and milestone manifests record `run_config: ../../config/exploration.yaml`. Downstream provenance reads whatever `run_config` names.
- `scripts/resume.sh` runs the run folder's own `src/exploration_main.py` with its `config/exploration.yaml`. Run folders created before the rename keep `src/main.py` + `config/trainer.yaml`, which it detects and uses.
- Updated: the launch scripts, `play.py`, preset comments, tests, `README.md` and `docs/AGENTS.md`. Historical run folders, copied sources and earlier evidence are untouched.
- Hydra now names an exploration run's job log `exploration_main.log` (it was `main.log`).

**Default exploration run folder:**
- `hydra.run.dir` in `config/exploration.yaml` resolves to `outputs/<method>/<domain>/seed_<exploration seed>/`, relative to the launch directory. This uses the resolver `exploration_run_dir`, registered in `exploration_main.py` (`exploration_trainer.default_run_dir`).
- `<method>` comes from `intrinsic_reward.enabled` and `intrinsic_reward.method` through `exploration_trainer.OUTPUT_LABELS`:

  | Configuration | Label |
  | --- | --- |
  | disabled | `uniform` |
  | `lcg` | `lcg` |
  | `ensemble_disagreement` | `full-ensemble` |
  | `laplace_disagreement` | `laplace-ensemble` |
  | `plan2explore_disagreement` | `plan2explore` |

  The intrinsic-reward identifiers are unchanged.
- `<domain>` and the seed are `env.train.domain_name` and `common.seed`. A null seed is an error, because the folder could not name the seed actually used.
- `hydra.run.dir` can still be overridden: the historical launch scripts and `resume.sh` do.
- Inside a run, as before: `config/exploration.yaml`, `dataset/`, `checkpoints/`, `milestones/steps_<N>/{manifest.json, dataset.pt, denoiser.pt}` (full-ensemble milestones also `denoiser_member_<k>.pt`), with replay episode files shared rather than copied. Downstream outputs and results add `downstream/` and `result/`.
- **Existing outputs are refused:** a new run (not `common.resume`) refuses a folder that holds any of `checkpoints`, `dataset`, `milestones`, `config`, `src`, `scripts`, `downstream` or `result`. The error points to resume.
  - The check runs in `exploration_main.main` before the finished-run guard, and again in `ExplorationTrainer` before anything is written.
  - The smoke's first attempt found that relaunching a *finished* run (`.run_is_over`) was skipped with exit 0 by the inherited guard instead of refused; hence the check in `main` (evidence: `attempt1_relaunch_not_refused/`).
  - Hydra itself still writes `.hydra/*.yaml` and its job log into the folder before any entry-point code runs; no run output is touched.
  - Explicit resume (`resume.sh`, `common.resume=true`) is unaffected.

**Downstream:** unchanged. `downstream_main.py` handles one task and one adaptation seed per command, and folder mode runs the milestones sequentially. Outputs go to `<run>/downstream/steps_<N>/<task>/seed_<s>[_member_<k>]/`, with fresh models per milestone, resume, overwrite protection, evaluate-only without the milestone, and the terminal format and W&B metrics as before. No all-task launching.

**Result collection** (`downstream_main.py downstream.mode=collect analysis.run_dir=<run> downstream.task_name=<task> downstream.seed=<s> [initialization.member_index=<k>] [analysis.protocol=<id>]`; CPU only; functions `collect_results` etc. in `downstream_trainer.py`):
- **Output:** `<run>/result/<task>/seed_<s>[_member_<k>]/result.csv`, one row per milestone in collected-step order.
- **Columns:**
  - `method`, `domain`, `exploration_seed` (from the run's `config/exploration.yaml`, never folder names);
  - `task`, `adaptation_seed`, `member_index`, `milestone`, `train_env_steps`, `num_eval_episodes`;
  - `return_{mean,median,std,min,max}`, `frac_over_{200,400,500,600}` (strictly greater);
  - `protocol` (the report's protocol id), `evaluation_max_steps`, `compatibility_sha256` (see the review fixes below);
  - `return_seed_<s>` for each evaluation seed, and `source_evaluation_file` (relative to the run).
- **Statistics** are recomputed from the episode returns. The saved aggregates must agree (else an error), and so must the episode seeds and the protocol id in the file name.
- **Missing or incomplete milestones** (no output, work folder only, no report of the protocol) are printed and left out; nothing is launched.
- **Not mixed:** several protocols require `analysis.protocol`; several policies under the selected protocol, and different environment, evaluation or downstream settings, are errors. Each denoiser member has its own table.

**Comparison** (`downstream_main.py downstream.mode=compare analysis.output_root=outputs analysis.domain=<d> analysis.exploration_seed=<N> downstream.task_name=<task> downstream.seed=<s>`; `compare_methods`):
- **Inputs:** every `<root>/*/<domain>/seed_<N>/` run with a saved configuration; the method comes from that configuration. Two runs recording the same method are an error, as are rows of another selection, duplicate milestones, different protocols (unless one is selected), different downstream settings and different evaluation seeds.
- **Output:** `<root>/comparisons/<domain>/exploration_seed_<N>/<task>/adaptation_seed_<s>/` with:
  - `comparison.csv`;
  - `analysis.json`: selection, protocol, settings hash, input paths with SHA-256 of each `result.csv` and evaluation file, methods, missing results, and a note that evaluation-seed spread is not training-seed uncertainty;
  - `task_return.{png,pdf}`: mean return against exploration environment steps, one curve per method;
  - `fraction_over_thresholds.{png,pdf}`: four panels.
- **Plot style:** fixed colours, markers and line styles per method. A missing milestone is a gap in the curve, never interpolated.
- **Plotting process:** the figures are drawn in a separate Python process that loads matplotlib but not torch (`PLOT_PROGRAM`, run like `device_report`'s EGL probe). In this Windows conda environment torch's bundled OpenMP runtime and the one matplotlib's numpy work loads abort a process holding both ("OMP: Error #15").
- **Dependency:** `requirements.txt` adds `matplotlib==3.10.9`, needed only for compare. kingplus0's `lcg` environment has no matplotlib, and none was installed, so the smoke's comparison ran on the laptop from the fetched outputs.

**Tests** (existing files):
- `tests/exploration/test_exploration_trainer.py`:
  - resolved default folders for all five presets, including another domain and seed, and the null-seed error;
  - existing-output refusal for four kinds of output, with nothing written;
  - the real entry point refusing a finished run folder.
- `tests/resume/test_resume_and_milestones.py`: `resume.sh` running the run folder's own entry point and config for the new and the historical layout, and a missing entry point refused; manifest `run_config`.
- `tests/downstream/test_downstream_trainer.py`:
  - 30-seed table format and column order;
  - statistics, and strict thresholds at returns exactly 200/400/500/600;
  - tampered aggregates rejected;
  - missing and incomplete milestones; ambiguous protocols and selection; member tables; mixed settings;
  - comparison files and inputs through the real CLI; missing milestones recorded;
  - duplicate method runs, protocol mismatch, and sources unchanged.
- Results: full suite **648 passed** (exit 0); affected (downstream, exploration trainer, resume) **109 passed** (exit 0). An earlier full run (22 failed) and affected run (13 failed, 5 h) failed only in MuJoCo/GLFW rendering (`OSError: access violation` opening a GLFW context) while the laptop display was unavailable; kept as `*_attempt_render_failures.log`.

**Smoke** (kingplus0 GPU 1, idle beforehand; GPU 0's other job untouched; compute and EGL rendering on GPU 1; W&B offline, not synced; isolated snapshot, 382 files, hashes verified; attempt 2 in `/home/bigking/scfeng/LCG-output-structure-smoke-20261030-r2`, started in the snapshot's repository root without `hydra.run.dir`):

| Step | Command (abridged; exact in `server/logs/*.cmd`) | Exit |
| --- | --- | --- |
| 1 | `src/exploration_main.py +experiment=walker_uniform_100k common.devices=1 common.seed=0 training.compile_wm=false collection.train.first_epoch.min=200 collection.train.steps_per_epoch=100 collection.train.num_steps_total=1200 training.num_final_epochs=0 denoiser.training.steps_first_epoch=2 denoiser.training.steps_per_epoch=2 evaluation.should=false milestones.at_steps=[300,...,1200] wandb.mode=offline` | 0 |
| 2 | step 1 again (the folder exists) | 1, refused |
| 3 | step 1's settings with `+experiment=walker_plan2explore_100k intrinsic_reward.num_members=5 intrinsic_reward.fit_updates=20 actor_critic.training.steps_first_epoch=2 actor_critic.training.steps_per_epoch=2` | 0 |
| 4, 5 | `src/downstream_main.py initialization.milestones_dir=<run>/milestones downstream.task_name=walk downstream.seed=0 common.devices=1 rew_end_model.training.steps=5 actor_critic.training.steps=5 evaluation.seeds=[3001,3002] evaluation.max_steps=50 wandb.mode=offline` per method | 0, 0 |
| 6, 7 | `src/downstream_main.py downstream.mode=collect analysis.run_dir=<run> downstream.task_name=walk downstream.seed=0 common.devices=cpu` (`CUDA_VISIBLE_DEVICES=""`) | 0, 0 |
| 8 (laptop) | `src/downstream_main.py downstream.mode=compare analysis.output_root=<fetched>/outputs analysis.domain=walker analysis.exploration_seed=0 downstream.task_name=walk downstream.seed=0` | 0 |

- **`verify_smoke.py`: 59 checks passed:**
  - exit codes; no `hydra.run.dir` anywhere; runs in `outputs/uniform/walker/seed_0` and `outputs/plan2explore/walker/seed_0` by the configured default;
  - the refused relaunch left every run output unchanged (Hydra rewrote only `.hydra/config.yaml`);
  - 10 epochs and 10 milestones (300..1,200, JSON manifests with `run_config ../../config/exploration.yaml`) per method; Plan2Explore 10 PMPO phases of 2 updates, uniform none;
  - downstream folder mode ran the 10 milestones in order; 10 outputs with 5 + 5 updates and one JSON report each per method;
  - each `result.csv` has 10 ordered rows whose per-seed returns, aggregates and threshold fractions equal the reports;
  - the comparison holds both methods and nothing missing, and its six files exist;
  - no generated YAML metadata; configuration YAML and 20 `metrics.jsonl` present; all 85 fetched JSON files parse strictly;
  - milestones and replay unchanged by downstream training, collection and comparison;
  - processes and banners on GPU 1; collection on CPU only.

  Attempt 1 of the verifier failed three checks on checker bugs (Hydra records `run.dir` unresolved; universal-newline reading doubled progress bars); kept as `verify_smoke_attempt1_checker_bugs.log`.
- **Results:**
  - Every evaluation return is about 3.5 or 4.5 (mean about 4.0) for both methods at every milestone, so every threshold fraction is 0.
  - The curves differ only in the third decimal place. That is expected from 5 + 5 updates and 50-step episodes, and says nothing about the methods.
  - The plots were restyled after the server run (markers and line styles per method, two-line subtitle). Only the laptop's comparison step draws plots; it was rerun.

```
outputs/
  uniform/walker/seed_0/            (plan2explore/walker/seed_0/ likewise)
    .hydra/{hydra.yaml, overrides.yaml}   .run_is_over   exploration_main.log   wandb_history.jsonl   wandb_run.json
    config/exploration.yaml
    checkpoints/{state.pt, info_for_import_script.json, agent_versions/agent_epoch_00005.pt, agent_epoch_00010.pt}
    dataset/{train/..., test/...}   src/   scripts/
    milestones/steps_000300/{dataset.pt, denoiser.pt, manifest.json}  ...  milestones/steps_001200/...
    downstream/steps_000300/walk/seed_0/{actor_critic.pt, rew_end_model.pt, config.yaml, metrics.jsonl, summary.json,
                                         wandb_run.json, evaluations/eval_walk_policy-<hash>_protocol-33afa0ff3589.json}
      ...  downstream/steps_001200/walk/seed_0/...
    result/walk/seed_0/result.csv
  comparisons/walker/exploration_seed_0/walk/adaptation_seed_0/   (drawn on the laptop from the fetched outputs)
    comparison.csv  analysis.json  task_return.png  task_return.pdf  fraction_over_thresholds.png  fraction_over_thresholds.pdf
```

**Review fixes (2026-10-30; `output_structure_review.zip` regenerated):** evidence in `outputs/step19-output-structure/review_fixes/`; the original smoke evidence is unchanged.
1. **Compatibility includes the environment.**
   - Before, `settings_sha256` left out `env.train`, so outputs with different action repeats compared as equal.
   - Now `compatibility_settings` takes resolved values of:
     - `env.train`: domain, tasks, action repeat, time limit, camera, image size;
     - `agent` (controller and denoiser), `world_model_env` (imagination), `rew_end_model.training`/`optimizer`, `actor_critic.training`;
     - the evaluation report's domain, task, action repeat, policy and return convention.
   - Not compared: `agent.actor_critic.debug_metrics`/`log_diagnostics`, `rew_end_model.debug_metrics`, W&B, checkpoint cadence, paths and the run identity (seeds and task are the selection).
   - A mismatch error names the differing settings, e.g. `env.train.action_repeat: 2 vs 4`.
   - The column is now `compatibility_sha256`. A `result.csv` with the old `downstream_settings_sha256` column is refused ("collect it again"), never treated as equivalent.
2. **Several policies under one protocol are refused.** Before, reports were keyed by protocol only, so a second policy's report silently replaced the first by filename order. Now every report is grouped per protocol first, and two or more for the selected protocol are an error naming the files. Nothing is chosen by name, time or return. A refused collection leaves the existing `result.csv` as it was.
3. **Gaps when every method lacks a milestone.** Before, the plot grid held only steps with rows. Now it is the union of the selected methods' known milestones and the observed rows (`analysis.json` `plot_grid`), with `None` (a gap) where a method has no row, in both figures. `comparison_plot_spec` builds the data; `plot_comparison` draws it.
4. **W&B default names identify the exploration run.**
   - Before, the prefix was the exploration folder's basename, now `seed_0` for every method.
   - Now the prefix is `<method>-<domain>-exploration-seed<N>` (`exploration_label`), from the run's saved `config/exploration.yaml`. Without it, the method and domain come from the milestone manifest or the saved downstream config, and whatever is not recorded reads `unrecorded`; folder names are never used.
   - Name `<prefix>-<milestone>-<task>-adaptation-seed<s>[-member<k>]`, group `<prefix>-<task>-adaptation-seed<s>[-member<k>]`, e.g. `lcg-walker-exploration-seed0-steps_010000-walk-adaptation-seed0` / `lcg-walker-exploration-seed0-walk-adaptation-seed0`.
   - Explicit `wandb.name`/`wandb.group` still win for new runs. A run with a `wandb_run.json` identity (resume, evaluate-only) keeps that run's own name and group; historical W&B runs are not renamed.
5. **`analysis.comparison_dir` removed.** Only the default comparison directory exists; custom output roots stay pending.

Tests (existing files):
- each environment, evaluation and controller difference refused with the setting named;
- logging-only differences and resolved references compatible;
- old tables refused;
- two policies under one protocol refused in either file order, with `result.csv` unchanged;
- a milestone missing from one or from every method checked in the plotted data;
- distinct names and groups per method, domain and seed; shared groups across milestones; member suffix; overrides; existing-run identity kept.

Corrected analysis on a local copy of the smoke outputs (`review_fixes/rerun_analysis.sh`):
- the old tables were refused (exit 1, "collect it again");
- collection exited 0 for both methods, with 10 rows each whose values equal the original tables apart from the fingerprint column;
- the comparison exited 0 with `plot_grid` 300..1,200 and nothing missing;
- the inputs were unchanged.

Results: full suite **653 passed** (exit 0); affected (downstream, exploration trainer, resume) **114 passed** (exit 0) (`review_fixes/*.log`).

## Source organization: intrinsic_reward package, device reporting, PMPO diagnostics (2026-10-31)

Committed as `88bc7a2` (on `2c95eb0`, clean working tree at the start). Review: `docs/changes/intrinsic_reward_cleanup_review.zip`. Evidence: `outputs/step20-intrinsic-reward-cleanup/`. Organization only: no budgets, schedules, milestones, evaluation seeds, objectives, reward scaling, sampling, RNG streams, logging keys, terminal output, output layout, checkpoint or milestone contents changed.

**Old path → new path** (git renames; contents unchanged apart from import lines):

| Old | New |
| --- | --- |
| `src/lcg/` (`__init__`, `forward_jvp`, `intrinsic_reward`, `lifecycle`, `precision`, `reward_normalization`, `theta_s`) | `src/intrinsic_reward/lcg/` |
| `src/ensemble.py` | `src/intrinsic_reward/ensemble.py` |
| `src/plan2explore.py` | `src/intrinsic_reward/plan2explore.py` |
| `src/laplace.py` | `src/intrinsic_reward/laplace.py` |
| `src/intrinsic_eval.py` | `src/intrinsic_reward/intrinsic_eval.py` (real-episode scoring with each method's frozen scorer: method-specific) |
| `src/device_report.py` | folded into `src/utils.py` (same functions and names; section "compute and rendering devices") |
| — | `src/intrinsic_reward/__init__.py` (docstring only, no imports) |

Hydra targets in `config/intrinsic_reward/*.yaml` are now `intrinsic_reward.lcg.LCGConfig`, `intrinsic_reward.lcg.theta_s.ThetaSConfig`, `intrinsic_reward.ensemble.EnsembleDisagreementConfig`, `intrinsic_reward.plan2explore.Plan2ExploreConfig` and `intrinsic_reward.laplace.LaplaceConfig`. Imports updated in `src/`, `tests/` and the maintained diagnostic scripts under `scripts/lcg_diagnostic/` and `scripts/historical_precision_diagnostic/` (they call the production LCG code). LCG and Laplace still share one historical-precision implementation (`intrinsic_reward.lcg.lcg_historical_precision`).

**Reviewed and kept in place:**
- `envs/world_model_env.ensemble_disagreement`: `WorldModelEnv` computes it during imagination from the member predictions it also takes the next frame from. Moving it into `intrinsic_reward/` would make `envs` import a package that imports `models` and `envs`, a circular dependency. `intrinsic_reward/ensemble.py` re-exports it, as before.
- `src/pmpo_diagnostic.py`: it is PMPO controller diagnostics, not an intrinsic reward. `exploration_trainer.py` imports `PMPORunDiagnostic` at runtime for the opt-in `+pmpo_diagnostic=<name>` mode, and the maintained `scripts/controller_diagnostic/eval_*` scripts and tests import it. There is no existing diagnostics package to move it into, and creating one is outside this cleanup. Its mode, commands and outputs are unchanged.
- `exploration_trainer.py` keeps orchestrating collection, denoiser training, scorer refresh and policy training; `models/` and `data/` are unchanged.

**Compatibility:**
- **Checkpoints and milestones:** the pickles of exploration checkpoints refer only to `torch`, `numpy`, `collections` and `_codecs` (checked for all five methods, before and after), and milestones are JSON plus tensors. Nothing needed handling.
- **Saved configurations:** `config/exploration.yaml` of earlier runs names the old `_target_` paths (only LCG instantiates its target). Those runs resume through `scripts/resume.sh` with their own copied `src/`, which has the old layout, so no shim was added.
- **New runs:** they copy the new package; a test runs the copied trainer and checks it imports `intrinsic_reward` from the run folder.
- **Intentional remaining old names:**
  - the Laplace settings label `precision.source = "lcg.lcg_historical_precision"`, recorded in checkpoints and W&B and compared on resume; changing it would change saved contents;
  - `docs/LAPLACE_PRECISION_AUDIT.md` and earlier notes, evidence and review packages, historical records of the layout when written;
  - historical run folders' copied sources.

**Before/after equivalence** (`equivalence.py`):
- **Method:** the same script ran on an export of `2c95eb0` (twice: identical, so the comparison is deterministic) and on the reorganized tree. It drives each method's trainer through its own test harness on CPU: two epochs with refreshes, PMPO updates on the intrinsic reward, seeded real-episode intrinsic evaluation and milestones.
- **Recorded:**
  - every logged scalar (intrinsic reward statistics included);
  - a hash of every checkpoint entry: weights, ensemble members, optimizers, schedulers, counters, replay index, RNG, LCG precision and bank, Laplace bank, Plan2Explore heads, RMS scale;
  - per-seed intrinsic evaluation scores and the pickled-module audit;
  - milestones at 18 and 24 steps: manifests with their file checksums, and each milestone loaded as downstream loads it (`load_milestone`, task walk: denoiser, member files, replay index).
- **Result:** all three result files are **byte-identical** (`before_run1.json` = `before_run2.json` = `after.json`, captured with `PYTHONHASHSEED=0`). The only excluded fields are the source identity, which names the tree, and the W&B mirror's byte length, which holds wall-clock timings; the starting tree differed between its own two runs only there.
- **Hash seed:** without a fixed seed, the milestones' `dataset.pt` file checksums differ between any two processes, the unchanged start tree included. The replay index pickles a hash-ordered container, so the byte order depends on Python's per-process hash seed. `replay_checksum_probe.py` shows the episode files byte-identical across processes and `info.pt`/`dataset.pt` equal in content (`replay_checksum_probe_random_hashseed.log`), and all byte-identical under a fixed seed (`replay_checksum_probe_fixed_hashseed.log`). This is pre-existing and unrelated to the cleanup; it is not changed here.

**Tests:**
- `test_intrinsic_reward_package_layout_targets_and_scorers` in `tests/exploration/test_exploration_logging.py`:
  - no stale top-level modules; the new modules import; every Hydra target resolves;
  - LCG and Laplace share the precision function; the trainer's device functions are `utils`';
  - each method constructs its own scorer and uniform none;
  - a new run's copied source imports the package from the run folder.
- `tests/exploration/test_device_report.py` now tests `utils`.
- Test-robustness fix (unrelated to the reorganization): `screen_lines` in `tests/downstream/test_downstream_trainer.py` drops trailing blanks. tqdm pads a final redraw that is shorter than the previous one (e.g. `9.84 updates/s` after a two-digit rate) with a space, which the strict bar pattern rejected under load. The first full and affected runs failed only there; their logs are kept as `*_attempt1_tqdm_padding_flake.log`.
- All other tests import the new paths.
- Results: full suite **654 passed** (exit 0); affected (`tests/lcg`, `tests/exploration`, PMPO diagnostic, `tests/downstream`) **441 passed** (exit 0). An earlier full run failed 21 rendering tests only (`OSError: access violation` opening a GLFW context with the laptop display unavailable); log kept.

**Final source tree** (`src/`):
```
src/
  __init__.py  agent.py  utils.py  play.py  pmpo_diagnostic.py
  exploration_main.py  exploration_trainer.py  downstream_main.py  downstream_trainer.py
  intrinsic_reward/  __init__.py  ensemble.py  plan2explore.py  laplace.py  intrinsic_eval.py
                     lcg/  __init__.py  forward_jvp.py  intrinsic_reward.py  lifecycle.py  precision.py
                           reward_normalization.py  theta_s.py
  models/  data/  envs/  coroutines/  game/
```

## Final independent review fixes (2026-10-31)

Starting checkout: `88bc7a2ecf271ea4910788c32ea6273f81a55a5f`, branch `reward-free-experiment`,
clean before this work. Changes are uncommitted. Existing modules and test files were reused;
no configuration, controller objective, reward scale, collection budget or training schedule changed.

1. **Denoiser padding:** traversal now knows the denoiser's conditioning length and excludes chunks
   with no real target after it. The loss retains the per-position mean pixel/example MSE and
   averages over positions with valid targets. Empty positions contribute no term; a wholly
   unusable batch raises before forward/RNG/optimizer work. Tests cover the eight-frame / six-frame
   chunk / four-conditioning-frame example, the left-padded single-target window, exact target
   coverage and finite gradients. Resume comparisons now require finite metrics instead of
   normalizing NaNs; the multi-batch diagnostic fixture retains multiple valid batches.
2. **Exploration replay:** resume verifies the checkpoint-referenced training and test episode
   checksums and readability once, before restoring collection or continuing training. Missing
   episodes retain `FileNotFoundError`; altered/unreadable episodes name the replay and path.
   Tests cover same-shaped changed observations, missing/unreadable files, unreadable payloads
   with matching recorded digests, and no repeated verification during subsequent updates.
3. **Comparison integrity:** comparison reuses result collection in read-only mode to revalidate
   source exploration/downstream configurations, milestone/summary identities, selected member,
   task, actual policy tensor hash, report filename/hash, seeds and evaluation protocol. It
   recomputes compatibility, aggregates and strict threshold fractions, and compares every CSV
   value with current sources before creating comparison artifacts. Stale tables require explicit
   recollection. Tests alter action repeat (both source configs), CSV mean/fraction, policy weights,
   report identities and milestones after collection, and check ambiguity and missing reports.
4. **Downstream metrics:** checkpoint format **3** records `metrics_bytes` and a SHA-256 of that
   committed prefix, after flushing/fsyncing the log and before atomic checkpoint publication.
   Resume refuses missing, truncated, malformed or readable-but-altered committed history.
   Only uncommitted interrupted tails are repaired; their exact bytes are kept and repair is
   idempotent. Tests also check fsync/publication ordering and older-format refusal. The older
   optional-metric-column fixture now writes those columns before committing, rather than
   rewriting already committed history.
5. **Exploration W&B:** logical row numbers resume at the checkpoint; server-held rows are
   compared and suppressed during replay, without assigning new server steps. Resume summaries
   distinguish matched, mismatched and unavailable comparisons; mismatches warn and retain the
   earlier server-held value. Committed backfill and offline no-backfill behavior are preserved.
   The exact two-committed / three-server-held case tests uniqueness and all three outcomes.
6. **Evaluation metadata:** Gaussian reports describe `deterministic (tanh(mu))`; Beta reports
   describe `deterministic (Beta mean)`. Evaluated actions and filename-based duplicate refusal
   are unchanged. The real-task action/history/return test now runs for both distributions.

**Compatibility:** older downstream checkpoints (format 1/2 or missing the committed-history
boundary/checksum) are refused by this source; use their own copied source to continue them, or
start fresh. Completed-output evaluate-only remains supported. Exploration replay without complete
checksums cannot be verified and is refused on resume. Existing Gaussian reports bearing the old
Beta label are not rewritten: they require a correct new evaluation artifact before collection
with this source (an existing identical policy/protocol remains protected from overwrite).
Existing server-side duplicate W&B history is not deleted; offline segments still cannot know
which rows the server holds. No generated-YAML compatibility was added.

Validation uses the local `lcg` environment, CPU only, short `outputs/f*` temporary paths and
installed Git Bash. No server access, GPU training, package changes or W&B synchronization.
The validation command ledger follows. Each `--basetemp` was a new, short test directory;
existing experiment outputs were not reused. Git's index was not changed.

Shared PowerShell environment and interpreter (all commands below; the f5 invocation omitted
the explicit `MKL_NUM_THREADS` assignment and retained `OMP_NUM_THREADS=1`):

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:CUDA_VISIBLE_DEVICES=''
$env:WANDB_MODE='disabled'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PATH='C:\Program Files\Git\bin;' + $env:PATH
```

Initial focused reproductions: **31 passed, 128 deselected**, exit 0 (42.70 s):
```powershell
& 'C:\Users\jerry\miniconda3\envs\lcg\python.exe' -m pytest -q -p no:cacheprovider --basetemp=outputs/f1 tests/exploration/test_test_diagnostics.py tests/exploration/test_exploration_logging.py tests/downstream/test_downstream_eval_resume.py tests/downstream/test_downstream_trainer.py -k 'denoiser_eight or denoiser_left or denoiser_rejects_completely or resume_verifies_checkpoint or two_committed_three or comparison_revalidates_sources or damaged_committed_metrics or metrics_boundary_fsync or without_metrics_boundary or evaluation_protocol_task'
```

Intermediate affected-file run: **157 passed, 3 failed**, exit 1 (711.36 s). It exposed the
missing-file exception-type compatibility, the diagnostic fixture's now-filtered final chunk,
and the old-column fixture rewriting committed history. These were corrected as described above;
numerical assertions were preserved or strengthened.
```powershell
& 'C:\Users\jerry\miniconda3\envs\lcg\python.exe' -m pytest -q -p no:cacheprovider --basetemp=outputs/f2 tests/exploration/test_test_diagnostics.py tests/exploration/test_exploration_logging.py tests/downstream/test_downstream_eval_resume.py tests/downstream/test_downstream_trainer.py
```

The exception-type failure was isolated with **18 passed, 1 failed**, exit 1 (46.67 s),
before preserving the existing `FileNotFoundError` contract:
```powershell
& 'C:\Users\jerry\miniconda3\envs\lcg\python.exe' -m pytest -xq -p no:cacheprovider --basetemp=outputs/f3 tests/exploration/test_test_diagnostics.py
```

Updated focused regressions and diagnostic compatibility: **35 passed, 126 deselected**, exit 0 (79.11 s):
```powershell
& 'C:\Users\jerry\miniconda3\envs\lcg\python.exe' -m pytest -q -p no:cacheprovider --basetemp=outputs/f4 tests/exploration/test_test_diagnostics.py tests/exploration/test_exploration_logging.py tests/downstream/test_downstream_eval_resume.py tests/downstream/test_downstream_trainer.py -k 'denoiser_eight or denoiser_left or denoiser_rejects_completely or resume_verifies_checkpoint or two_committed_three or comparison_revalidates_sources or damaged_committed_metrics or metrics_boundary_fsync or without_metrics_boundary or evaluation_protocol_task or reuse_only_after_resume or denoiser_test_loss_is_one'
```

Once-per-resume and payload-readability checks: **2 passed, 23 deselected**, exit 0 (12.35 s):
```powershell
& 'C:\Users\jerry\miniconda3\envs\lcg\python.exe' -m pytest -q -p no:cacheprovider --basetemp=outputs/f5 tests/exploration/test_test_diagnostics.py -k 'verification_runs_once or unreadable_episode_payload'
```

All three affected existing tests rerun after their corrections: **3 passed**, exit 0 (25.59 s):
```powershell
& 'C:\Users\jerry\miniconda3\envs\lcg\python.exe' -m pytest -q -p no:cacheprovider --basetemp=outputs/f6 tests/downstream/test_downstream_eval_resume.py::test_resume_of_a_run_without_wandb_starts_one_and_backfills_its_committed_updates tests/exploration/test_test_diagnostics.py::test_reuse_only_after_resume_diagnoses_without_collecting tests/exploration/test_exploration_logging.py::test_denoiser_test_loss_is_one_mean_per_diagnostic_and_unpublished_values_stay_internal
```

AST parsing of all 10 changed Python files passed, and `git diff --check` passed. No configuration
file changed. The notes file is already ignored by Git and remains unstaged, as it was before.

Final full suite on the completed implementation: **688 passed, 176 warnings**, exit 0
(1142.02 s / 19:02). The warnings are the existing PyTorch sampler `data_source` deprecation.
This includes all focused regressions, the unchanged real-environment checks, resume/state/RNG
comparisons and the copied-source Git Bash resume-script tests. No assertion was relaxed to
accommodate Windows paths or the new integrity checks.
```powershell
& 'C:\Users\jerry\miniconda3\envs\lcg\python.exe' -m pytest -q tests -p no:cacheprovider --basetemp=outputs/f7
```

Final checks: `git diff --check` passed; configuration diff and staged diff are empty. The five
source modules and five existing test files remain modified and uncommitted, together with this
already-ignored notes file. GPU execution, live W&B server recovery/synchronization and long-run
learning quality remain unverified; none were launched.

## Repeated-crash exploration W&B correction (2026-10-31)

The six-fix snapshot's server-ahead recovery still selected the latest local values after a
second crash before the same checkpoint. With two committed rows and three server-held rows,
the first suppressed replay changed the local third-row loss from 0.25 to 0.50. A second replay
of 0.50 then falsely compared equal to that suppressed attempt instead of the server's 0.25.

`ExplorationTrainer.log_rows` now mirrors a suppressed attempt's actual values for committed
backfill **and** an explicit `server_reference` containing the earlier epoch/values, or null
when unknown. `reconcile_wandb_history` restores that separate reference. An unknown reference
stays unknown on repeated crashes. Multiple legacy entries without this metadata are ambiguous
and report `replayed_unavailable` rather than guessing that the latest attempt reached W&B.
A unique legacy record remains usable. Invalid reference metadata is also treated as unknown.
This is additive JSONL metadata, not a checkpoint-format or training-state change. Mirror
repair, committed-byte backfill selection and offline no-backfill behavior remain unchanged.

Regression: `tests/exploration/test_exploration_logging.py::test_two_crashes_preserve_server_row_reference`,
parameterized with `present`, `missing` and `ambiguous_legacy` references. Two fresh trainer
instances restore the same two-row checkpoint and replay loss 0.50 while the fake server holds
0.25. The test asserts no server duplication, preservation of 0.25, mismatch and warning on
both resumes for a known reference, unavailable on both for an unknown reference, no false
matched count, unchanged committed prefix and RNG, and later committed backfill of actual 0.50.

Before the implementation correction, all three cases failed (exit 1): the known-reference
case lost its second mismatch warning, and missing/ambiguous references falsely became matched.
The continuous reproduction output is `outputs/wr6/reproduction.log`. After correction, the
new regression plus the original one-crash matched/mismatched/unavailable test passed:
**6 passed**, exit 0 (6.63 s; runner elapsed 8.297 s). No assertions were weakened.

Validation uses `C:/Users/jerry/miniconda3/envs/lcg/python.exe`, CPU only, OMP/MKL threads 1,
`CUDA_VISIBLE_DEVICES=''`, `WANDB_MODE=disabled`, `PYTHONDONTWRITEBYTECODE=1`, and
`C:/Program Files/Git/bin` prepended to PATH. Existing `outputs/f*` directories are not reused.
Continuous stdout/stderr logs and JSON command/environment/exit records are saved under the new
gitignored `outputs/wr6/` evidence location. The runner imports no project code itself.

```powershell
& 'C:\Users\jerry\miniconda3\envs\lcg\python.exe' -m pytest -q tests/exploration/test_exploration_logging.py::test_two_crashes_preserve_server_row_reference -p no:cacheprovider --basetemp=outputs/w1
# Before correction: 3 failed, exit 1, 10.33 s (runner elapsed 12.125 s).
& 'C:\Users\jerry\miniconda3\envs\lcg\python.exe' -m pytest -q tests/exploration/test_exploration_logging.py::test_two_crashes_preserve_server_row_reference tests/exploration/test_exploration_logging.py::test_two_committed_three_server_rows_replay_once -p no:cacheprovider --basetemp=outputs/w2
# After correction: 6 passed, exit 0, 6.63 s.
```

No server experiment, GPU training, live W&B sync or package upgrade was performed. W&B recovery
here is tested with a fake server; the reference comes from the local mirror, not a server query.
Historical ambiguous mirrors cannot recover a lost association and are deliberately unavailable.
Earlier checkpoint-refusal and mislabeled evaluation-report rules remain as documented above.
The prior review ZIP is immutable: `docs/changes/reward_free_six_fixes_review_20261031.zip`,
SHA-256 `506493418255911075e9a391bbe23f42aa19752a64656fbbad22ffcc33769ef8`.

Focused exploration logging/diagnostics/controller-resume/real-environment resume validation:
**117 passed, 76 existing sampler deprecation warnings**, exit 0 (555.04 s; runner 558.75 s).
```powershell
& 'C:\Users\jerry\miniconda3\envs\lcg\python.exe' -m pytest -q tests/exploration/test_exploration_logging.py tests/exploration/test_test_diagnostics.py tests/exploration/test_pmpo_objective_resume.py tests/resume/test_resume_and_milestones.py -p no:cacheprovider --basetemp=outputs/w3
```

The regression's final server-loss fixture explicitly removes the lost fake-server row before
backfill, then checks uniqueness and RNG again. That final regression passed **3 tests**, exit 0
(11.62 s; runner 13.843 s), before starting the full suite. The implementation did not change
after the focused suite.
```powershell
& 'C:\Users\jerry\miniconda3\envs\lcg\python.exe' -m pytest -q tests/exploration/test_exploration_logging.py::test_two_crashes_preserve_server_row_reference -p no:cacheprovider --basetemp=outputs/w4
```

**CPU binding correction and intermediate full-suite failure:** the first full attempt with
`CUDA_VISIBLE_DEVICES=''` finished with **8 failed, 683 passed, 173 warnings**, exit 1
(1146.00 s; runner 1149.187 s). All eight failures were CUDA device-selection/deserialization
errors, not W&B replay assertions. Fresh-process probes of this Windows PyTorch 2.7.1+cu128
environment showed that the empty value gives `torch.cuda.is_available() == True` while
`torch.cuda.device_count() == 0`. It unexpectedly selected CUDA-only tests and cannot be
described as strictly CPU-only execution. With `CUDA_VISIBLE_DEVICES='-1'`, both probes
consistently report no CUDA device (`False`, `0`). The runner now uses `-1` for strict CPU
validation; no repository source, configuration, test assertion or skip condition was changed
to resolve these device errors. Existing CUDA-only checks will use their existing skips.

Continuous initial-attempt log/exit/environment: `outputs/wr6/full.log` and `full.json`.
Probe results: `outputs/wr6/device_environment_probe.json`. The earlier six-fix ledger/package's
CPU-only label is not established by its empty-visibility setting; that historical evidence is
preserved verbatim, and no GPU validation of this correction is claimed from it. Apart from the
unintentionally selected synthetic CUDA tests in the failed attempt, no GPU experiment or live
W&B sync was launched.
```powershell
& 'C:\Users\jerry\miniconda3\envs\lcg\python.exe' -m pytest -q tests -p no:cacheprovider --basetemp=outputs/w5
# Initial attempt with empty CUDA visibility: 8 failed, 683 passed, exit 1.
```

Affected device-case retry with `CUDA_VISIBLE_DEVICES=-1`: **8 passed, 3 skipped, 3 existing
sampler warnings**, exit 0 (18.74 s; runner 20.859 s). The three skips are the existing CUDA
variants in these selected functions. No source/test change was needed after the failed full run.
```powershell
& 'C:\Users\jerry\miniconda3\envs\lcg\python.exe' -m pytest -q tests/checkpoint/test_pmpo_resume.py::test_next_actual_imagined_update_resume tests/collection/test_collection_task_labels.py::test_trainer_rejects_unknown_collection_mode_before_any_work tests/models/test_pmpo_diagnostic.py::test_fixed_real_uint8_checkpoint_round_trip_is_exact tests/models/test_pmpo_integration.py::test_static_dataset_trainer_world_model_and_pmpo_resume tests/models/test_pmpo_integration.py::test_h10_checkpoint_resume_is_bit_exact_with_slow_critic_and_prior_refresh tests/models/test_pmpo_integration.py::test_trainer_world_model_pmpo_diagnostic_and_resume_with_the_h10_controller -p no:cacheprovider --basetemp=outputs/w6
```

**Final verified CPU full suite:** **685 passed, 6 skipped, 176 existing sampler warnings**,
exit 0 (950.63 s / 15:50; runner 953.5 s), with `CUDA_VISIBLE_DEVICES=-1`. All 691 cases were
collected; the six existing CUDA-only skips are checkpoint PMPO resume, input-device JVP,
CUDA RNG preservation, fixed-real CUDA checkpoint, CUDA final-observation metadata and H10 CUDA
checkpoint resume. No assertion or test was disabled to obtain this result. This includes all
new repeated-crash cases and the other five fixes' regressions. Continuous final log and exact
command/environment/exit record: `outputs/wr6/full_final.log` and `full_final.json`.
```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:CUDA_VISIBLE_DEVICES='-1'
$env:WANDB_MODE='disabled'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PATH='C:\Program Files\Git\bin;' + $env:PATH
& 'C:\Users\jerry\miniconda3\envs\lcg\python.exe' -m pytest -q tests -rs -p no:cacheprovider --basetemp=outputs/w7
```

Only `ExplorationTrainer.log_rows`, `reconcile_wandb_history`, the added exploration logging
regression and this existing notes file changed for the correction. AST comparisons confirmed
all other trainer methods and all preexisting logging tests unchanged; the other previously
modified source/test files are byte-identical to the prior ZIP. `git diff --check` passed.
Configuration, controller objectives, reward scaling, schedules and packages are unchanged.
All work remains uncommitted; nothing was pushed.

New review package: `docs/changes/reward_free_wandb_repeated_crash_review_20261031.zip`, built
by `outputs/wr6/build_package.py` with the complete current source/config/tests/scripts, current
documentation, correction-only and complete diffs, all continuous validation logs and a SHA-256
manifest. The package result (size, entry count, SHA-256) is recorded outside the ZIP in
`outputs/wr6/PACKAGE_RESULT.json` to avoid a self-referential archive checksum. Both diffs are
replayed against the corresponding tracked Git trees in isolated storage, and ZIP bytes are
compared with the current working tree. Earlier ZIPs remain unchanged.

## Downstream adaptation throughput on the RTX 5090 (2026-11-02)

Throughput check, not a learning-quality experiment. Settings document: `REWARD_FREE_EXPERIMENT_SETTINGS.md`, Section 9A (the file is in the user's Downloads folder, not this repository). Review: `docs/changes/downstream_timing_review.zip`. Evidence: `outputs/step21-downstream-timing/`. Nothing is committed; no source, test, production budget or W&B sync changed.

**Source.**
- Requested: `uniform-pilot-run1-20261020`, `steps_010000`. The server original was already gone (deleted 2026-10-29, see "Compute matching"). The milestone exists as the read-only local copy `outputs/step6-laplace-mature/source/uniform-pilot-run1-20261020/` (25 files, fetched 2026-10-27).
- Verified before use: manifest file checksums, 20/20 episode files, denoiser state `11c4da27…` (file `45527705…`), replay index `986f7c97…` (file `477c217f…`), epoch 10, 10,000 steps, 13,600 denoiser updates, 0 PMPO updates, uniform collection.
- The current loader reads only `manifest.json`; this milestone has the older `manifest.yaml`. The benchmark's server copy received a derived `manifest.json` with equal content (`make_manifest_json.py`). The source was not written.
- Not used: the regenerated `uniform-10k-seed0`, a different run. Its `denoiser.pt` happens to have the same file hash; its `dataset.pt` differs.
- After the run: all 25 local files unchanged (`source_local_before/after.sha256`), and all 26 server-copy files passed `sha256sum -c`.

**Run.**
- `/home/bigking/scfeng/LCG-downstream-timing-20261102`, `lcg` environment, snapshot of clean HEAD `cc99758` (the user's uncommitted `config/exploration.yaml` edit excluded; downstream does not read that file).
- RTX 5090, physical GPU 1 (`GPU-923e4182…`), idle beforehand. Banner: compute physical GPU 1, EGL physical GPU 1. GPU 0's job (pid 1247789) untouched. Our process was the only one on GPU 1.
- `timing_driver.py` wraps the trainer's methods with the existing `RegionTimer` and runs `src/downstream_main.py` unchanged as `__main__`. Exact command: `server/logs/command.txt`. Exit 0, 289.8 s under the 1,800 s limit. W&B offline, not synced.
- Settings verified from the saved config (`verify_run.log`): tanh-Gaussian, H10, slow critic coefficient 1.0, `repeat_anchor`, sign-only PMPO, existing batch sizes, sequence length and optimizers. Only the two step counts were overridden.

**Measurements** (`server/logs/timing.json`, `results.json`). Totals are CUDA-complete `RegionTimer` regions, and synchronized wall clocks agree to under 1 ms. tqdm rates were not used.

| | Reward predictor | Task policy |
| --- | ---: | ---: |
| Phase, 500 updates | 47.97 s | 191.50 s |
| Warmed s/update, median / mean / p90 (updates 51-500) | 0.0929 / 0.0928 / 0.0939 | 0.3804 / 0.3805 / 0.3819 |
| Warmed loop s/update, excluding checkpoints (projection rate) | 0.0932 | 0.3814 |
| First update / first 50 updates | 0.887 s / 5.51 s | 0.581 s / 19.27 s |
| Checkpoints: count, median, file | 5, 0.099 s, 79 MB | 5, 0.106 s, 86 MB |
| Torch peak allocated / reserved | 7.38 / 7.80 GB | 3.16 / 7.81 GB |

- **Startup:** 2.46 s from process start to the trainer, 1.40 s trainer construction (including all checksums), 3.94 s in all outside the phases and evaluation. The downstream path has no compilation. Rates are steady from about update 4 (reward) and update 2 (policy); one policy update reached 0.488 s (update 285).
- **Evaluation:** 45.45 s, 30 episodes, 15,000 steps (3.03 ms/step). Mean return 23.06 after 500 policy updates, which says nothing about learning.
- **Peak GPU memory:** 8,192 MiB by nvidia-smi, sampled about once per second, including EGL.
- **Completed output:** 32.9 MB (`rew_end_model.pt` 23.6, `actor_critic.pt` 8.3, `metrics.jsonl` 1.0) plus 3.3 MB of W&B offline files.

**Projection model.** `phase(n) = first 50 updates + (n - 50) x warm loop rate + ((n - 1) // 100 + 1) x median checkpoint`; adaptation = 5,000-update phase + 10,000-update phase + the measured 49.4 s outside the phases. At 500 updates it gives 47.96 s and 191.45 s against the measured 47.97 s and 191.50 s.

| Scope (5,000 + 10,000 updates + 30-seed evaluation; one adaptation seed) | Adaptations | GPU-hours | Days, one GPU |
| --- | ---: | ---: | ---: |
| One milestone, one task | 1 | 1.21 (72.4 min) | 0.05 |
| 20 milestones x 1 method x 1 seed x 1 task | 20 | 24.1 | 1.0 |
| 5 methods x 1 seed x 1 task | 100 | 120.7 | 5.0 |
| 5 methods x 3 seeds x 1 task | 300 | 362.2 | 15.1 |
| 7 accepted groups, 18 tasks x 5 x 3 x 20 | 5,400 | 6,519.8 | 271.7 |

**These are projections from one uniform milestone.** Other domains, replay sizes and denoiser configurations can have different throughput. The policy phase is 88% of an adaptation. For comparison, 5k + 5k is 40.6 min and the pilot's 5k + 2k budget 21.4 min. Storage at the working plan is about 100 MB per adaptation including W&B (540 GB for 5,400), against 78 GB free on kingplus0. Earlier figures stay as superseded history in the settings document, including the pilot's 27.5 min per arm on the old controller.

**Driver fix during the dry run:** `torch.cuda.is_available()` stayed true after `common.devices=cpu` hid the devices, so the driver's GPU-name query failed; it now follows the trainer's device. The first dry run failed only for that reason, and the second passed (`dryrun_cpu/`).

**Not measured:** other domains and methods, larger replays, several adaptations on one GPU, full-length runs, online W&B.

## Standalone downstream analysis entry point (2026-11-02)

Organizational change; integration evidence only, not learning quality. Uncommitted on top of `cc99758`. Review: `docs/changes/downstream_analysis_separation_review.zip`. Evidence: `outputs/step22-analysis-separation/`.

**Responsibilities.**
- `src/downstream_main.py`: train, resume, evaluate. Rejects `collect`/`compare`, and unknown modes, showing the analysis commands.
- `src/downstream_trainer.py`: training and evaluation.
- `src/downstream_analysis.py` (new): `collect`/`compare`, validation, CSVs, plots. Rejects other modes, showing the correct command. Reuses `config/downstream.yaml`; no configuration values changed, only comments.

**One decision beyond a plain move.** With `MUJOCO_GL=egl`, importing dm_control opens `/dev/nvidia0`, `/dev/nvidia1` and `/dev/nvidiactl` (measured on kingplus0), although CUDA stays uninitialized. Importing either trainer would therefore give analysis GPU handles. So six definitions that training and analysis share moved to `data/milestone.py`, which already holds output paths and identities: `RETURN_CONVENTION`, `evaluation_path`, `deterministic_policy_description`, `check_evaluation_config` (the downstream one; the exploration trainer's function of the same name is unrelated), `OUTPUT_LABELS` with `method_label`, and `exploration_identity`. The trainers import them from there, so existing names still resolve and nothing is duplicated. The analysis process now loads no `dm_control`, `mujoco`, `envs`, `models`, `agent` or trainer. `wandb` is still imported through `utils`, never started.

Moved text was cut by script (`apply_refactor.py`), not retyped. `audit_move.py`: 90 definitions unchanged in place, none lost or duplicated, 33 moved or edited, and the 11 edited texts differ only in import lines, the `Any` annotation, entry-point names in guidance, and `main`'s mode check (`move_audit_diffs.txt`).

**Intended text differences.** Guidance says `downstream_analysis.py` rather than `downstream_main.py`; this appears in errors and in `analysis.json`'s `missing` reason for an uncollected method. The new rejection messages replace the old one-line mode error.

**Fixtures.** No suitable outputs existed: `~/scfeng` on the server held only `LCG-deleted-manifests`, and local runs have one or two milestones or predate `cc99758`. Generated at `/home/bigking/scfeng/LCG-analysis-separation-20261102` (3.9 GB, kept), `lcg` environment, snapshot of clean HEAD plus only the refactored files (the user's uncommitted `config/exploration.yaml` edit excluded; `milestones.every_steps=null` also explicit).
- GPU 1 (`GPU-923e4182…`) had no compute process beforehand; compute and EGL on physical GPU 1 per every banner; GPU 0's other job (pid 1247789) untouched. W&B offline, 22 offline runs, not synced.
- Exit 0 for all four: uniform exploration 17 s, Plan2Explore exploration 31 s, each downstream folder about 86 s.
- `verify_fixtures.py`, 106 checks: 200 initial + 10 epochs x 100; 10 milestones at 300..1,200; 2 denoiser updates per epoch; 2 PMPO per epoch for Plan2Explore, none for uniform; 5 members, `fit_updates=20`; no training-only epochs; diagnostics off; 5 + 5 updates; seeds 3001-3002, 50 steps. Resolved settings differ from the unmodified presets only in the requested overrides, so controller, feature and reward-scaling settings are unchanged (`resolved_differences.json`). `training.compile_wm` stayed at each preset's value, unlike the 2026-10-30 smoke.
- Milestones and replay unchanged by downstream training; all 480 files of each completed run unchanged after analysis (`sha256` lists).

**Old against new.** `before_after.py` runs HEAD's `downstream_main.py` and the new `downstream_analysis.py` on separate copies of the same outputs through their real entry points; `compare_sides.py` gives every step an expected outcome and normalizes only paths and the entry-point name.

| Where | Scope | Checks | Exit |
| --- | --- | --- | --- |
| kingplus0 | full copies; collect, error paths, GPU handles | 363/363 | 0 |
| Laptop (Windows) | files analysis reads (102, hashes equal the server's); collect, compare, plots | 415/415 | 0 |

- 16 cases: the full fixture, missing milestones, one or no method collected, missing arguments, second protocol, two policies, malformed report name, tampered aggregate, report policy identity, settings differences, four kinds of stale table.
- `result.csv` and `comparison.csv` byte-identical; `analysis.json` equal after normalization; plotted data identical and equal to `comparison.csv`; gaps exactly the missing results in return and threshold plots. PNGs byte-identical 12/12; PDFs differ only in their creation date 12/12. Each fixture method: 10 ordered rows. Repeating the commands, and hiding all GPUs, gave identical tables.
- Under `MUJOCO_GL=egl`: old process held `/dev/nvidia0`, `/dev/nvidia1`, `/dev/nvidiactl`; new process held none. CUDA uninitialized on both sides; no trainer constructed, no policy evaluated. The analysis's only GPU-relevant difference is this improvement.
- **The first server comparison is invalid and kept as history** (`server/superseded_first_comparison/`): its 286/286 only required the sides to agree, and `compare` was failing on both for lack of matplotlib, with 0 plots compared. The checker now requires each step's outcome.
- **kingplus0's `lcg` environment has no matplotlib.** `compare` writes its CSV and `analysis.json`, then fails at plotting with `ModuleNotFoundError` (exit 1), before and after this change. Nothing was installed. Plots are drawn on the laptop, as in the 2026-10-30 smoke.
- **Plotting subprocess still needed:** torch and matplotlib in one Windows process abort with `OMP: Error #15`, exit 3 (`logs/probe_plot_in_process.log`).
- **Existing rough edge, preserved:** an `eval_*.json` with no `_protocol-` in its name gives a bare `IndexError` rather than a clear refusal. Same on both sides.

**Tests.** Baseline `tests/downstream` at HEAD snapshot: 89 passed. After: `tests/downstream` 98 passed; full suite 700 passed, 18.5 min, in this worktree with the user's `config/exploration.yaml` edit present (the fixtures did not use it). Added: real-entry collect/compare with CUDA and module checks, seven rejection cases, argument errors, and separation of names. `git diff --check` clean.

## PMPO updates per epoch: 200 vs 400 on one frozen LCG model (2026-11-03)

Bounded controller-optimization check on one frozen model; not real exploration, not other methods or training stages. Nothing committed; no source, test or preset changed. Review: `docs/changes/pmpo_update_budget_review.zip`. Evidence: `outputs/step23-pmpo-update-budget/`.

**Source.** No new fixture was trained: a suitable verified source exists. `uniform-pilot-run1-20261020`, milestone `steps_010000` (the server original was deleted 2026-10-29; the read-only local copy fetched 2026-10-27, as in "Downstream adaptation throughput"). Walker, seed 0, current denoiser architecture, uniform collection, 10,000 steps in 20 episodes of 500, epoch 10, 13,600 denoiser updates, 0 PMPO updates. Denoiser state `11c4da27…` (file `45527705…`), replay index `986f7c97…` (`dataset.pt` `477c217f…`). Checked in the run: manifest file checksums, all 20 episode checksums, both recorded identities, architecture and environment against the current configuration. Copied to `/home/bigking/scfeng/LCG-pmpo-update-budget-20261103/source/` and made read-only; all 25 files unchanged afterwards, there and locally.

**Code and settings.** Snapshot of HEAD `7c1a383` (`git archive`; the user's uncommitted `config/exploration.yaml` milestone edit is not in it and is irrelevant here). Scripts adapted from the 2026-10-20 frozen-model comparison (`pmpo_compare.py`): `common.py`, `update_budget.py`, `run.sh`, `run_prefix.sh`, `analyze.py`; no source module added. Resolved `+experiment=walker_lcg_100k` with no overrides besides `common.seed=0` (the preset's value): tanh-Gaussian actor, H10, `repeat_anchor`, EMA critic regularization 1.0 / 0.02 with online bootstraps and advantages, advantage-weighted PMPO cap 5, entropy 0.01, KL 0.3, prior refresh 500, actor/critic lr 1e-4 with 100 warmup updates, batch 32; LCG R1 (10 tensors, 217,664 scalars), candidate MC 5, chunk 32, reference 320, historical MC 3, damping 1e-4, fixed raw / 100; `training.compile_wm=true` as in the preset.

**Method.** One LCG refresh (production `LCGLifecycle`, round identifier 11, the next epoch after the milestone's 10), then frozen: precision, training bank and denoiser (digests checked after training). A separate evaluation bank from seed 7,000,003 with the same frozen precision. One fresh controller trained 400 updates exactly as `ExplorationTrainer.train_component` (production prompt loader, compiled world model, production reward hook). Snapshots before update 1 and after 100, 200 and 400: 128 imagined rollouts (4 repeats x 32, 128 distinct prompts), the same prompts, world-model noise and policy-sampling base draws for every snapshot; each rollout's 10-step raw LCG sum under both banks; uniform actions under the same conditions. Evaluation restores the global RNG and module modes and leaves the controller's private RNG, weights and optimizer unchanged (asserted).

**Runs** (kingplus0, RTX 5090, physical GPU 1 `GPU-923e4182…`; GPU 0 had another user's idle 6.8 GB kernel, pid 1247789, untouched; compute via `CUDA_VISIBLE_DEVICES=<UUID>`, `MUJOCO_EGL_DEVICE_ID=1`, no rendering; only our process on GPU 1 throughout).
- Main: exit 1 after all results were written (632 s). The crash came after `summary.json`, `per_rollout.csv` and the per-update metrics: `wandb.summary` rejected integer dictionary keys. That ended the process before its in-process control. The W&B run `8msx10s3` holds all training rows, the four evaluations and the paired summary, but is marked failed.
- Prefix check, after a fix converting those keys: exit 0, 663 s. A controller evaluated at 0/100/200/210 against one never evaluated, in one process. Total GPU time 1,295 s of the 30-minute limit. W&B run `i84jq71w`.
- Shared prefix: all 210 per-update metric rows identical, and full states (weights, EMA critic, prior, counter, private RNG, both Adam states) identical at 100, 200 and 210. Across the two processes, the evaluated controller's 210 updates and its evaluations at 0/100/200 equal the main run's exactly. So updates 1-200 are the exact shared prefix, and snapshotting did not change training.

**Results** (raw imagined LCG return, 10-step sum; paired mean +- SE over 128 rollouts):

| Update | Policy, training bank | Policy, separate bank | Uniform, training / separate |
| ---: | ---: | ---: | ---: |
| 0 | 159.70 | 196.70 | 168.55 / 209.39 |
| 100 | 160.04 | 197.19 | same |
| 200 | 159.99 | 197.55 | same |
| 400 | 159.75 | 196.64 | same |

| Paired | Training bank | Separate bank |
| --- | ---: | ---: |
| 200 - 0 | +0.29 +- 0.30 | +0.85 +- 0.33 |
| 400 - 200 | -0.24 +- 0.42 | -0.91 +- 0.51 |
| 400 - 0 | +0.05 +- 0.38 | -0.06 +- 0.40 |
| 400 - uniform | -8.81 +- 0.96 | -12.75 +- 1.29 |

- The banks agree (per-rollout correlation 0.82-0.84); the result is not bank-specific.
- Policy on 1,280 fixed states: entropy 3.319 / 3.352 / 3.333 / 3.321 nats; KL to the initial policy 0 / 0.003 / 0.018 / 0.040; lowest per-dimension entropy 0.552 -> 0.533. The actor moved 6.8% (relative parameter distance) by 400.
- Critic: value-target correlation -0.07 (1-10), 0.42 (91-100), 0.76 (191-200), 0.85 (391-400); value mean 0.01 -> 3.47 against return mean 0.78 -> 3.83. Positive-advantage fraction 1.000 through update 100 and 0.998 at 191-200 (0.987 at 391-400): a fresh critic starts near zero under positive rewards, so for about 200 updates almost every advantage is positive and the sign split carries no direction. Weight clipping 0. The update-500 prior refresh is not reached.
- Compute: 1.330 s/update warmed (median; first update 4.2 s), 82% LCG scoring; updates 1-200 268.6 s, 201-400 265.9 s; each evaluation about 20 s; LCG refresh 8.2 s. Peak torch memory 3.92 GB allocated / 7.30 GB reserved; nvidia-smi 8,222 MiB.
- At this rate, after epoch 1's 5,000 updates: 200/epoch gives 52,800 PMPO updates (19.5 GPU-hours, 4.4 min per epoch), 400/epoch 100,600 (37.2 GPU-hours, 8.9 min per epoch). The difference is 17.7 GPU-hours of PMPO per LCG run.

**Recommendation.** Neither budget improved the frozen objective meaningfully. Updates 0-200 gained +0.29 +- 0.30 (training bank) and +0.85 +- 0.33 (independent bank), under 0.5% of the return. Updates 201-400 gave nothing: -0.24 +- 0.42 and -0.91 +- 0.51, at 266 s more per epoch. The policy stayed 5-6% below uniform actions at every snapshot. This is not a plateau after improvement, so it does not show that 200 is enough; it shows that 200 more updates do not fix the optimization. On this evidence 400 buys nothing, and 200 is the cheaper choice. More updates should not be assumed to solve the controller's failure to beat uniform. Likely contributors visible here: the critic needs about 200 updates before advantages change sign, the learning-rate warmup covers the first 100, and the initial tanh-Gaussian (pre-tanh std 0.55) samples a narrower action spread than uniform, which LCG scores lower. These are observations, not tested causes.

**Limits.** One frozen model, one controller seed, one uniform-data source. A fresh controller: production epochs after the first continue a controller already trained for 5,000 updates, whose critic is fitted, so the per-epoch budget there may behave differently; this check cannot settle that. Imagined LCG return is not real exploration, and other methods have different costs.

## PMPO actor objective: sign-only vs advantage-weighted on one frozen LCG model (2026-11-03)

Bounded controller-optimization comparison on one frozen model; not real exploration. Nothing committed; no source, test or preset changed. Review: `docs/changes/pmpo_objective_review.zip`. Evidence: `outputs/step24-pmpo-objective/` (`server/analysis.md` has every table).

**Setup.** The diagnostic of "PMPO updates per epoch: 200 vs 400" (`update_budget.py`, copied; additions: a W&B group argument, the full controller state digest at update 0, and `control=0` to skip the in-process control, which that section's prefix run already established). Same source, `uniform-pilot-run1-20261020/steps_010000`, freshly copied to `/home/bigking/scfeng/LCG-pmpo-objective-20261103/source/` (25/25 checksums equal before and after, there and locally; identities and episode checksums checked in each run), same HEAD `7c1a383` snapshot, LCG round 11, separate bank seed 7,000,003, 128 paired rollouts with the same seeds. Two processes run one after the other on kingplus0 physical GPU 1 (`GPU-923e4182…`; GPU 0 again held another user's idle kernel, pid 1247789, untouched; only our process on GPU 1):
- A: `agent.actor_critic.actor_objective=sign`; B: `advantage_weighted`, cap 5 (the preset). Both with `agent.actor_critic.debug_metrics=true` for the weight, advantage and action-spread metrics.
- Debug metrics are logging only: on the laptop, 6 updates with and without them gave identical shared metrics and evaluations. On the server, arm B reproduces the earlier 400-update run (debug off): per-rollout returns and all 52 shared per-update metrics identical.
- Matched start: the initial state digest (actor, critic, EMA critic, prior, counter, private RNG, optimizer) is equal; prompts, uniform references, update-0 returns, frozen scorer and denoiser are identical. The only resolved configuration difference is `agent.actor_critic.actor_objective`.
- Exit 0 for both, 637 s each; 1,274 s of GPU time in all. W&B group `pmpo-objective-sign-vs-advweighted-20261103`: `diag-lcg-pmpo-objective-sign-400upd-uniformpilot-steps10000-seed0` (`pbzzipne`) and `…-advweighted-…` (`mny2ws79`), both finished.

**Results** (raw imagined LCG return, 10-step sum, paired mean +- SE over 128 rollouts; training bank / independent bank):

| | Sign-only | Advantage-weighted | Weighted - sign |
| --- | ---: | ---: | ---: |
| Update 0 | 159.70 / 196.70 | identical | 0 |
| 100 - 0 | +0.14 +- 0.18 / +0.39 +- 0.12 | +0.35 +- 0.10 / +0.49 +- 0.10 | +0.21 +- 0.16 / +0.10 +- 0.11 |
| 200 - 0 | +0.11 +- 0.27 / +0.17 +- 0.29 | +0.29 +- 0.30 / +0.85 +- 0.33 | +0.18 +- 0.35 / +0.68 +- 0.43 |
| 400 - 0 | -0.15 +- 0.28 / +0.99 +- 0.28 | +0.05 +- 0.38 / -0.06 +- 0.40 | +0.20 +- 0.35 / -1.05 +- 0.42 |
| 400 - 200 | -0.27 +- 0.26 / +0.82 +- 0.30 | -0.24 +- 0.42 / -0.91 +- 0.51 | +0.03 +- 0.59 / -1.72 +- 0.72 |
| Policy - uniform at 400 | -9.01 +- 0.96 / -11.70 +- 1.27 | -8.81 +- 0.96 / -12.75 +- 1.29 | |

- Every change is under 1% of the return, and both objectives stay 5-6% below uniform actions at every snapshot.
- The banks disagree on the later updates. Sign's 200 -> 400 change is -0.27 on the training bank but +0.82 on the independent bank. Weighted leads at 100 and 200 on both banks, within about 1.5 SE. At 400 it leads on the training bank (+0.20) and trails on the independent bank (-1.05 +- 0.42). Per-rollout bank correlation is 0.82-0.84 in both arms.

**Diagnostics** (window means; `server/analysis.md`):
- Advantages: every advantage was non-negative in 238 (sign) and 248 (weighted) of 400 updates; the first negative one came at update 136 / 143; the positive fraction was still 0.99 at 391-400. Mean |A| fell from 0.77 to about 0.37 while the critic caught up (value bias -0.77 -> -0.36).
- Weights: sign-only weights are 1 by construction. Advantage-weighted positive-group weights have mean 1 (by normalization), standard deviation 0.43-0.48 and maximum 2.1-2.4; the cap of 5 was never reached (clipping 0 in both groups). Negative-group window means (0.2-1.0 in both arms) mostly reflect how many updates had any negative advantage at all.
- **All-positive advantages do not remove the learning direction under advantage weighting.** With every advantage positive, sign-only PMPO reduces to maximizing the mean log-probability of the policy's own samples. That has zero expected gradient, so only noise, the KL and the entropy term act. Advantage weighting multiplies each log-probability by |A| over its mean. Samples with larger advantages are still pushed up more than others, so the update keeps a direction even when no advantage is negative.
- Critic (similar in both): value-target correlation about -0.07 / 0.42 / 0.76 / 0.85 at 1-10 / 91-100 / 191-200 / 391-400; target MSE 0.73 -> 0.16-0.17.
- Policy: KL to the initial policy at 400 is 0.023 (sign) and 0.040 (weighted) on the fixed states. The actor moved 6.6% / 6.8%. Entropy 3.37 / 3.32 nats (3.32 at start). The deterministic action spread across states stays about 0.002, so the policy is nearly state-independent in both arms. Training action standard deviation is about 0.44 per dimension. Actor gradient norms are higher with weighting (window peak 4.7 vs 1.8); value gradient norms are similar.
- Compute: 1.341 s per update in both arms, warmed median, debug metrics on (1.330 s without them); 81% LCG scoring; peak 3.93 GB allocated / 7.30 GB reserved, nvidia-smi 8,222 MiB.
- Proposed schedule (190 collection epochs, the same budget in every epoch including the first, no training-only epochs): 200 updates per epoch = 38,000 updates = 14.2 GPU-hours of PMPO per LCG run; 400 = 76,000 = 28.3 GPU-hours, at this model's rate. The objective does not change the cost.

**Interpretation.** Inconclusive. Neither objective optimizes the frozen LCG return meaningfully within 400 updates. Their differences are within about 1 return unit (under 1%), change sign between snapshots, and disagree between banks. Advantage weighting does keep a learning direction while advantages are all positive, and it moves the actor somewhat further, but no consistent gain in return follows here. No basis for changing the presets' advantage-weighted objective, and none for preferring it over sign-only. Both stay below uniform actions, and the policy's actions barely depend on the state.

**Limits.** One frozen model, one controller seed, a fresh controller; evaluation-seed SEs do not cover training-seed variation. Imagined LCG return is not real exploration.

## Training batch-size throughput on the RTX 5090 (2026-11-03)

Computational benchmark only; it says nothing about learning quality. No source, test, preset or update count changed; nothing committed. Review: `docs/changes/batch_size_benchmark_review.zip`. Evidence: `outputs/step25-batch-benchmark/` (`server/benchmark.md` has every trial).

**Batch-size keys (resolved, HEAD `7c1a383`).**

| Phase | Key | Value | One example is |
| --- | --- | ---: | --- |
| Exploration denoiser | `denoiser.training.batch_size` (`config/exploration.yaml`) | 32 | a replay sequence of 6 frames (4 conditioning + 1 target + 1 autoregressive step) |
| Exploration PMPO | `actor_critic.training.batch_size` (`config/exploration.yaml`) | 32 | an imagined trajectory from one replay prompt: H = 10 transitions, so 320 LCG candidates per update |
| Downstream reward predictor | `rew_end_model.training.batch_size` (`config/downstream.yaml`) | 32 | a replay sequence of `seq_length` 19 |
| Downstream sign-only PMPO | `actor_critic.training.batch_size` (`config/downstream.yaml`) | 32 | an imagined trajectory of 10 transitions |

Not training batch sizes: `intrinsic_reward.candidate_chunk_size` (32; how many candidates one JVP pass scores, scores do not depend on it), `candidate_num_mc` (5, JVP bank size), `precision_reference_size` (320 historical transitions for the precision) with `precision_num_mc` (3), `world_model_env.num_batches_to_preload` (256; PMPO restarts its prompt block with 1 batch each update), and Plan2Explore's predictor-fitting and replay-encoding batches (`intrinsic_reward.batch_size` and `encode_batch_size`, 256 each).

**Source and method.** The verified uniform milestone `uniform-pilot-run1-20261020/steps_010000` (10,000 steps, 20 episodes, 13,600 denoiser updates; denoiser `11c4da27…`). It was copied read-only to `/home/bigking/scfeng/LCG-batch-benchmark-20261103/source/`; all 25 files were unchanged afterwards, there and locally.
- Exploration (`bench_exploration.py`): the production `ExplorationTrainer` with `+experiment=walker_lcg_100k`, run on the static replay copy, which it never writes. The milestone denoiser is loaded through `initialization.path_to_ckpt`, with a fresh controller and optimizers. One production `train_agent()` epoch as epoch 11: denoiser updates, LCG refresh, then PMPO with its real data loaders. Gaussian H10, `repeat_anchor`, advantage-weighted cap 5, the LCG preset with chunk 32, raw / 100, `compile_wm=true`, debug metrics off. Each optimizer step's end is a CUDA event (step post-hooks; no added synchronization), so an interval is one complete update including data loading. LCG scoring comes from the trainer's own per-update RegionTimer events.
- Downstream (`timing_driver.py` of 2026-11-02, unchanged): `src/downstream_main.py` on a fresh copy of the milestone per trial, with the derived JSON manifest. The reward predictor trains first, then sign-only PMPO on the frozen denoiser and reward predictor; evaluation off. An update's time is the interval between end marks taken after its logging.
- One process per trial, each starting from the same saved milestone state with fresh models. The first 10 (denoiser, reward predictor) or 5 (PMPO) updates are warm-up, reported separately. W&B offline, not synced.

**Runs.** kingplus0, RTX 5090 physical GPU 1 (`GPU-923e4182…`), driver 570.195.03, torch 2.7.1+cu128, cuDNN 9.7.1, Triton 3.3.1, Python 3.10.20. GPU 0 held another user's idle kernel (pid 1247789, 6.8 GB) throughout and was not used. Each trial's only GPU 1 process was its own. 16 trials, all exit 0: phase 1 at 16/32/64/128, phase 2 repeating 32 and 64 and adding 48 and 96. 1,015 s of the 30-minute limit.

**Results** (warmed median; throughput = batch / median s/update):

| Phase | Batch | s/update (p90) | Throughput | vs 32 | Peak torch alloc / reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Denoiser | 16 / 32 / 48 / 64 / 96 / 128 | 0.047 / 0.064 / 0.088 / 0.111 / 0.172 / 0.230 | 341 / 502 / 545 / 576 / 558 / 558 sequences/s | 0.68 / 1 / 1.08 / 1.15 / 1.11 / 1.11 | 2.6 / 5.1 / 7.6 / 10.0 / 15.0 / 19.9 (reserved up to 21.3) |
| Exploration PMPO | 16 / 32 / 64 / 128 | 0.684 / 1.330 / 2.620 / 5.376 | 23.4 / 24.1 / 24.5 / 23.8 trajectories/s (x10 transitions) | 0.97 / 1 / 1.02 / 0.99 | 1.5 / 3.0 / 5.8 / 11.5 (reserved 2.8-21.4) |
| Reward predictor | 16 / 32 / 64 / 128 | 0.048 / 0.093 / 0.198 / 0.395 | 333 / 344 / 323 / 324 sequences/s | 0.96 / 1 / 0.93 / 0.94 | 3.5 / 6.9 / 13.6 / 27.1 (reserved 29.9) |
| Downstream PMPO | 16 / 32 / 48 / 64 / 96 / 128 | 0.260 / 0.380 / 0.555 / 0.705 / 1.155 / 1.533 | 61.5 / 84.2 / 86.6 / 90.8 / 83.1 / 83.5 trajectories/s | 0.73 / 1 / 1.03 / 1.08 / 0.99 / 0.99 | 1.6 / 2.9 / 4.3 / 5.7 / 8.5 / 11.3 |

- Repeats: 32 and 64 reproduce within 0.5% in every phase (denoiser 503 / 501 and 576 / 576 sequences/s; PMPO 1.329 / 1.331 and 2.623 / 2.616 s; downstream PMPO 84.3 / 84.0 and 90.8 / 90.8 trajectories/s).
- p90 within 1-3% of the median everywhere. LCG scoring is 80-83% of an exploration PMPO update at every batch, which is why its cost grows linearly with the batch.
- Startup is excluded from the rates: about 9 s to a ready exploration trainer and about 7.3 s per LCG refresh. The first compiled PMPO update took 4-23 s, the first denoiser update about 1 s, and the downstream trainer was ready after 2.3 s. nvidia-smi peak: 6.2 GB (E32) to 29.9 GB (reward predictor at 128).

**Exploration PMPO budget at the planned 200 updates per epoch.** (A) Fixed update count: batch 64 gives 128,000 imagined transitions per epoch in 524 s, against 64,000 in 266 s at 32. That is twice the training data, not an execution-only optimization. (B) Fixed transitions (64,000 per epoch): 100 updates at 64 take 262 s, 50 at 128 take 269 s. Within 1.5% of 32, so no saving.

**Recommendation** (for review; nothing changed):

| Phase | Current batch | Best tested batch | Current throughput | Candidate throughput | Speedup | Peak memory (candidate) | Learning-budget implication |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Exploration denoiser | 32 | 64 | 502 sequences/s | 576 sequences/s | 1.15x (repeated) | 10.0 GiB alloc / 10.6 reserved | Same update counts: twice the sequences per update, 1.74x the time per update. Same sequences: halve the updates (5,000 + 200 per epoch), 13% less denoiser time. Either way the optimization changes (lr 1e-4 and warm-up were set at batch 32). |
| Exploration PMPO | 32 | 32 (16-128 flat) | 24.1 trajectories/s | - | none (0.97-1.02x) | 3.0 / 5.5 GiB | Keep 32. A larger batch at 200 updates is more imagined data at proportional cost. |
| Downstream reward predictor | 32 | 32 | 344 sequences/s | - | none (0.93-0.96x elsewhere) | 6.9 / 7.3 GiB | Keep 32. |
| Downstream sign-only PMPO | 32 | 64 | 84.2 trajectories/s | 90.8 trajectories/s | 1.08x (repeated) | 5.7 / 9.5-15.1 GiB | Same updates: twice the trajectories, 1.85x the time. Same trajectories: half the updates, about 7% less policy time. |

- Computationally faster: denoiser batch 64 (15%) and downstream PMPO batch 64 (8%); both repeatable, with memory far below 32 GB. Proposed overrides for review: exploration `denoiser.training.batch_size=64`, downstream `actor_critic.training.batch_size=64`, each with its update counts either kept (more data, more time) or halved (same data, less time). Exploration PMPO and the reward predictor stay at 32.
- Unknown: whether either change preserves learning quality (gradient noise; the learning rates and warm-ups were set at batch 32). Not measured here; it would need a learning comparison.
- Cross-method PMPO check: not run, since no different exploration PMPO batch is recommended. A real 5-member ensemble state does not exist (only 3-member test fixtures), so that check would have been unavailable; Laplace and Plan2Explore could be built from this milestone through their own refresh.
- A denoiser batch change affects every method: the full ensemble trains five members, each at the denoiser batch, and the 2026-10-29 compute matching (member counts, timing definitions) was measured at batch 32. Those measurements would need updating before any change.

**Limits.** One model size and replay size (10,000 steps), Walker, one GPU model, one controller state. Rates may differ for other replays, methods and when two processes share a GPU.

## Online pilot: 200 vs 400 PMPO updates per epoch, with a uniform baseline (2026-11-03)

A matched online pilot: one exploration seed and one adaptation seed. Differences are evidence for this pilot only, and an inconclusive difference is not proof of equivalence. Nothing was committed, and no source, test or preset changed. The user's uncommitted `config/exploration.yaml` edit is not in the snapshot. Review: `docs/changes/online_pilot_200v400_review.zip`. Evidence: `outputs/step26-online-pilot/`; `analysis/pilot_summary.md` and `analysis/downstream_comparison.md` hold every table.

**Design.** All three arms use the production `ExplorationTrainer` from a HEAD `7c1a383` snapshot, with presets `walker_uniform_100k` and `walker_lcg_100k`. Walker `[stand, walk, run]`, seed 0.
- **Data:** 5,000 initial uniform steps, then 10 epochs x 500 steps, for 10,000 training env steps. No training-only epochs, and no test collection (`evaluation.should=false`), so no task returns appear in the exploration logs or W&B.
- **Updates:** the denoiser gets 10,000 + 9 x 400 = 13,600 updates. PMPO gets 200 or 400 in every epoch including the first: 2,000 or 4,000.
- **Batches and logging:** every batch is 32, and debug metrics are off.
- **Milestones:** at 5,500, 7,500 and 10,000 env steps (epochs 1, 5 and 10).
- **Production order per epoch:** collect, train the denoiser, refresh the scorer, train the policy.
- **Shared initial replay:** the 5,000 initial steps (10 episodes) were collected once by the production `collect_initial_dataset`. They were saved as a bundle of episode files, index, environment collection state and RNG states.
- **Injection into each arm:** a driver (`scripts/exploration_driver.py`, run via runpy around `src/exploration_main.py`) copies the episodes in, verifies their checksums, restores the collection state and sets the global RNG to the collector's post-collection state.
- **Snapshots:** the driver also copies `state.pt` at epochs 0, 1, 5 and 10.

**Runs** (kingplus0, RTX 5090).
- **GPU 1** (`GPU-923e4182…`) ran, in order: the shared collection, uniform, LCG-200 and its diagnostics, then the uniform and LCG-200 adaptations.
- **GPU 0** (`GPU-c92ce703…`) ran LCG-400, its diagnostics and its adaptation, from 18:22. LCG-400 was planned for GPU 1 after LCG-200 and was moved on the user's instruction. GPU 0 held another user's idle Jupyter kernel throughout (pid 1247789, 6.8 GB, about 0% utilization); it was tolerated and never touched. That GPU's runner refused to start beside any other process.
- **Placement:** compute and EGL rendering were bound to each assigned physical GPU (banners; `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `MUJOCO_EGL_DEVICE_ID`). Across 4,137 monitor samples, at most one of our processes was on a GPU at a time.
- **Exit codes:** every stage exited 0.
  - Exploration: uniform 902 s, LCG-200 3,662 s, LCG-400 6,355 s.
  - Diagnostics: 128 s and 129 s.
  - Adaptations: 4,352 s, 4,352 s and 4,389 s.
- **Budget:** the ledger records 24,314 s (6.75 GPU-hours) of the 8-hour budget: GPU 1 13,433 s, GPU 0 10,881 s.
- **CPU checks:** the verification and `downstream_analysis.py` compare ran on the server's CPU. The first verification attempt crashed before writing anything (a duplicate summary key in the script). The script was fixed and rerun; the attempt's logs are kept.
- **W&B:** online, project `LCG-exploration-policy-unpdate-intervation`, group `online-pilot-200-vs-400-20261103`. Runs: `pilot200v400-{uniform,lcg200,lcg400}-exploration` (`fb3p6s6l`, `4qqnmsg9`, `okru5blw`) and `pilot200v400-<arm>-downstream-steps_010000-walk-adaptation-seed0` (`94v5jkh2`, `ml5f2ekp`, `k27e06vg`). Afterwards, at the user's request, copies were uploaded to project `LCG-exploration-policy-update-intervention` (spelling corrected) with `wandb sync --no-skip-online` from the saved run files. Same IDs, names, group, state, history steps, summary keys and configuration; the originals are unchanged, and the stored configuration still names the original project. Logs: server `logs/wandb_resync/`.

**Integrity** (`verification/verification.json`, 37/37 checks).
- **Counts:** each arm reached epoch 10 with 10,000 training env steps, 13,600 denoiser updates (10,000 then 400 per epoch), and 0, 2,000 or 4,000 PMPO updates (exactly 200 or 400 per epoch). Test env steps and test episodes were 0.
- **Finiteness:** every W&B-mirrored value is finite: 40,870, 128,920 and 216,920 values.
- **Replay:** episode files match their checkpoint checksums. The first 10 episodes of every arm are the shared bundle's. Episode 11, collected in epoch 1 by the untrained policy, is identical in both LCG arms; all later episodes differ between arms.
- **Milestones:** epochs, step counts and counters are as planned.
- **Identical start:** all arms and the collector share the initial denoiser `fdea3376…`, controller `b88d0b83…`, controller private RNG, pre-collection RNG, and post-injection RNG `14dc58d3…` (equal to the collector's post-collection RNG).
- **LCG arms in epoch 1:** denoiser losses are identical, and PMPO updates 1–200 are identical, even though the arms ran on different GPUs.
- **Downstream:**
  - The frozen denoiser's digest equals the milestone's after init, after the reward predictor and after the policy, in all three arms.
  - The milestone and replay files hashed identically before and after adaptation: 62 files on GPU 1, 31 on GPU 0.
  - The interaction counts are separate. Exploration training used 10,000 env steps per arm, 20,000 physical in all, and no test steps. Downstream training interacts with no environment. Evaluation used 30 episodes x 500 = 15,000 env steps per arm.
- **Note:** the milestone identity field `dataset_sha256` (`index_sha256`) fingerprints only episode counts and lengths, and every Walker episode is 500 steps, so it is equal across the arms. Contents are pinned by `dataset.pt`'s checksum, which differs: `e2d79c8a…`, `9899f60c…` and `3e8dbdcf…`.

**Imagined diagnostics** (`scripts/pilot_diagnostics.py`, forward only, from the saved snapshots).
- **Rollouts:** 128 paired H10 rollouts per policy and milestone: 32 prompts from the shared initial replay x 4 repeats, with the same seeds for the current policy, the epoch-0 policy and uniform actions.
- **Scoring:** by the milestone's own scorer (training bank) and by a separate bank (seed 7,000,003) under the same precision.
- **Table:** paired mean +- SE (fraction positive) of the raw 10-step LCG sum:

| Arm | Env steps | PMPO updates | Bank | Current - initial | Current - uniform |
| --- | ---: | ---: | --- | ---: | ---: |
| LCG-200 | 5,500 | 200 | train / separate | +1.05 +- 0.46 / +0.74 +- 0.60 | -12.19 +- 1.82 / -17.69 +- 2.43 |
| LCG-200 | 7,500 | 1,000 | | +4.64 +- 0.60 / +6.60 +- 0.64 | -11.87 +- 1.93 / -10.97 +- 1.95 |
| LCG-200 | 10,000 | 2,000 | | +14.76 +- 1.04 / +11.94 +- 0.94 | +4.23 +- 1.53 (0.53) / +0.37 +- 1.45 (0.50) |
| LCG-400 | 5,500 | 400 | | +4.36 +- 0.59 / +2.11 +- 0.73 | -8.88 +- 1.83 / -16.31 +- 2.46 |
| LCG-400 | 7,500 | 2,000 | | +16.42 +- 1.28 / +14.65 +- 1.21 | +3.51 +- 2.03 (0.56) / +1.87 +- 1.78 (0.51) |
| LCG-400 | 10,000 | 4,000 | | +31.71 +- 1.40 / +27.02 +- 1.49 | +23.74 +- 1.58 (0.89) / +17.14 +- 1.64 (0.84) |

- **Initial policy vs uniform:** the epoch-0 policy stays below uniform at every milestone, by 8–18 return units.
- **Changing model:** each milestone is scored under its own model and scorer, which change between milestones. Gains across milestones combine policy change and model change.
- **Cumulative updates:** at equal cumulative updates (2,000), the two arms are similar on both banks (+4.2 / +0.4 and +3.5 / +1.9 over uniform).
- **Critic** (training probe on the milestone replay):
  - Value-target correlation was 0.80 / 0.86 / 0.97 (LCG-200) and 0.89 / 0.93 / 0.99 (LCG-400).
  - The positive-advantage fraction was 1.00 / 0.85 / 0.32 and 0.99 / 0.58 / 0.11.
  - Updates with every advantage non-negative, by epoch: LCG-200 196, 103, 7, then 0; LCG-400 272, 51, then 0.
  - Advantage weights averaged about 1 in both groups, with essentially no clipping.
- **Policy probe** (1,280 states of the uniform rollouts):
  - KL to initial: 0.02 / 0.08 / 0.56 (LCG-200) and 0.05 / 0.56 / 4.49 (LCG-400). The actor's relative distance from initial reached 0.13 and 0.22.
  - Entropy (initial 3.31 nats): LCG-200 3.35 / 3.45 / 3.46; LCG-400 3.40 / 3.50 / 2.22.
  - Deterministic action spread across states (std of tanh(mu), mean over dimensions; initial 0.0025): LCG-200 0.003 / 0.019 / 0.148; LCG-400 0.004 / 0.119 / 0.393.
  - An observation only: by 10,000 steps the LCG-400 policy's actions depend strongly on the state and are less random.

**Downstream** (only the 10,000-step milestones; Walker walk; adaptation seed 0).
- **Training:** 5,000 reward-predictor and 10,000 sign-only PMPO updates, batch 32.
- **Evaluation:** deterministic, seeds 3001–3030, at most 500 steps, protocol `1e5d8307ecab`, the same for all arms.
- **Comparison:** `downstream_analysis.py` compare (`comparison.csv`, compatibility hash `8edc5d6f…`, equal for all arms) matches `scripts/analyze_downstream.py`. Its plot step needs matplotlib, which the server lacks; the figure is drawn on the laptop.

| Arm | Mean | Median | Std | Min | Max | > 200 / 400 / 500 / 600 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Uniform | 45.6 | 45.4 | 8.9 | 23.9 | 66.5 | 0 / 0 / 0 / 0 |
| LCG-200 | 57.3 | 51.0 | 19.4 | 4.5 | 109.3 | 0 / 0 / 0 / 0 |
| LCG-400 | 51.7 | 46.1 | 14.6 | 44.3 | 118.2 | 0 / 0 / 0 / 0 |

| Paired over the 30 evaluation seeds | Mean +- SE | Seeds won / lost |
| --- | ---: | ---: |
| LCG-200 - uniform | +11.6 +- 3.2 | 26 / 4 |
| LCG-400 - uniform | +6.1 +- 2.7 | 20 / 10 |
| LCG-400 - LCG-200 | -5.6 +- 3.9 | 9 / 21 |

The SEs reflect evaluation-seed variation of one trained policy per arm. They do not cover exploration-seed or adaptation-seed variation.

**Compute.**
- **Exploration totals:** uniform 883 s; LCG-200 3,637 s, of which PMPO 2,667 s; LCG-400 6,327 s, of which PMPO 5,360 s.
- **PMPO per epoch:** about 1.33 s per update on both GPUs, so 400 per epoch costs about 270 s more per epoch.
- **Other per-epoch costs:** the denoiser takes 26 s and the LCG refresh 7.5–7.8 s.
- **Memory:** peak 5.2 GB allocated in exploration.

**Answers.**
- **Do the continuing controllers improve after their first 400 cumulative updates?** In the imagined objective, yes. In both arms, the margins over the epoch-0 policy and over uniform keep growing past 400 cumulative updates, on both banks. This differs from the frozen-model check, where a fresh controller gained nothing in 400 updates. Here the critic fits, and advantages change sign, within the first one to three epochs. The imagined gain looks tied to cumulative updates rather than to env steps.
- **Does 400 per epoch beat 200?** On imagined LCG return, yes: at 10,000 steps, +23.7 vs +4.2 over uniform on the training bank and +17.1 vs +0.4 on the separate bank. Downstream, no: LCG-400 - LCG-200 is -5.6 +- 3.9 and wins 9 of 30 seeds. That is inconclusive, leaning against 400.
- **Does LCG beat uniform downstream?** Numerically, yes: LCG-200 +11.6 +- 3.2 (26/30 seeds) and LCG-400 +6.1 +- 2.7 (20/30). But no arm learned to walk: means are 46–57 out of a 500-step maximum, and no episode exceeds 200. So this compares failing policies with one exploration seed and one adaptation seed. It is not evidence of a useful downstream benefit.
- **Does any benefit justify 400's extra compute?** Not on this evidence. 400 per epoch doubles PMPO time (+2,693 s over 10 epochs; about 14 more GPU-hours per 190-epoch LCG run, at the rate measured in "PMPO actor objective"). Its larger imagined gain did not carry over to the downstream task here.

**Limits.**
- One exploration seed and one adaptation seed.
- 10,000 env steps, far below the planned 100k.
- The downstream returns are uninformatively low for all arms.
- The LCG-400 arm ran on the other GPU: same model, and exact epoch-1 agreement was observed.
- Imagined LCG return is not real exploration.
- No tuning, budget extension, checkpoint selection or additional arm followed any intermediate result.

## Follow-up to the online pilot: full world-model ensemble at 200 PMPO updates per epoch (2026-11-04)

This arm was requested by the user after the pilot's results, so it is not part of the pilot's pre-set plan. It is the first online run of the full ensemble (`walker_ensemble_100k`: 5 bootstrapped denoisers, PMPO on their disagreement, advantage-weighted cap 5, `compile_wm=false`). Nothing was committed, and no source, test or preset changed. Review: `docs/changes/online_pilot_ensemble200_review.zip`. Evidence is in the same folder, `outputs/step26-online-pilot/`:
- `analysis/with_ensemble/`
- `server/verification/verification_ensemble.json`
- the `ensemble*` logs

**Setup.** The pilot's code snapshot, driver, settings and shared initial replay (`scripts/ensemble_common.sh`, `queue_ensemble.tsv`, `run_queue_ensemble.sh`, a 6-hour cap with its own ledger), at 200 updates per epoch:
- **Data and schedule:** 10,000 env steps; 13,600 denoiser updates per member; 2,000 PMPO updates; batches 32; milestones at 5,500, 7,500 and 10,000 steps.
- **Downstream:** the 10,000-step milestone was adapted with member 0 (`initialization.member_index`, fixed in advance). The downstream settings, adaptation seed and evaluation protocol are the pilot's.
- **Dry run:** a laptop dry run with tiny budgets passed first.
- **Not run:** imagined diagnostics. The pilot's diagnostic script needs LCG's scorer.

**Run.** kingplus0 physical GPU 1, with compute and EGL rendering on GPU 1. Only our process was on that GPU (1,155 samples).
- **Exit codes:** all stages exited 0. Exploration took 7,379 s (to 11:55) and adaptation 4,348 s; 11,736 s in all (3.26 GPU-hours).
- **W&B:** project `LCG-exploration-policy-update-intervention`, group `online-pilot-200-vs-400-20261103`. Runs `pilot200v400-ensemble200-exploration` (`358iz9pp`) and `pilot200v400-ensemble200-downstream-steps_010000-walk-adaptation-seed0-member0` (`cl2hm5px`).

**Integrity** (`scripts/verify_ensemble.py`, 16/16 checks).
- **Counts:** epoch 10, 10,000 training env steps, and no test interaction. Every member had 13,600 denoiser updates (10,000 then 400 per epoch). There were 2,000 PMPO updates, exactly 200 per epoch.
- **Finiteness:** every logged value is finite.
- **Replay:** episode checksums match, and the first 10 episodes are the shared bundle's.
- **Milestones:** member counters are as planned, and the 5 members are distinct.
- **Shared start:** member 0's initial weights, the controller, the controller's private RNG and the post-injection RNG equal the collector's and the pilot arms'.
  - The pre-collection RNG differs by design: building the 4 extra members draws random numbers before collection.
- **Downstream:** the initial reward predictor and controller digests are the pilot's. The frozen member 0 is unchanged through adaptation and equals the milestone's. The milestone and replay files were unchanged (43 files).

**Downstream** (Walker walk, seeds 3001–3030, at most 500 steps; `downstream_analysis.py` compare agrees, compatibility hash `8edc5d6f…`, the same as the pilot's arms):

| Arm | Mean | Median | Std | Min | Max | > 200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform | 45.6 | 45.4 | 8.9 | 23.9 | 66.5 | 0 |
| LCG-200 | 57.3 | 51.0 | 19.4 | 4.5 | 109.3 | 0 |
| LCG-400 | 51.7 | 46.1 | 14.6 | 44.3 | 118.2 | 0 |
| Ensemble-200 | 79.0 | 69.7 | 40.3 | 3.7 | 165.9 | 0 |

| Paired | Mean +- SE | Seeds won / lost |
| --- | ---: | ---: |
| Ensemble-200 - uniform | +33.4 +- 7.2 | 26 / 4 |
| Ensemble-200 - LCG-200 | +21.7 +- 6.5 | 25 / 5 |
| Ensemble-200 - LCG-400 | +27.3 +- 7.4 | 27 / 3 |

**Training.**
- **Advantages:** the PMPO positive-advantage fraction stayed between 0.72 and 0.96 in every epoch. Only one update, in epoch 1, had all advantages non-negative. Value-target correlation was 0.71 in epoch 1 and 0.84–0.91 afterwards.
- **Reward trend:** the raw intrinsic reward per step (ensemble disagreement) rose from 0.0035 (epoch 1) to 0.0042 (epoch 10). LCG's reward fell over the same epochs, because its scorer is refreshed each epoch.
- **Compute:** exploration took 7,343 s over the epochs.
  - Denoisers 4,396 s, five times a single denoiser's: about 54 minutes in epoch 1, then about 129 s per epoch.
  - PMPO 2,924 s at 1.46 s per update, against LCG's 1.33.
  - Peak 5.4 GB allocated.

**Reading.**
- **Downstream:** the ensemble arm adapted better than all three pilot arms on this seed pair, by 22–33 return units, winning 25–27 of 30 seeds.
- **Same failure as the pilot:** no arm learned to walk. Means are 46–79 of a 500-step maximum, and no episode exceeds 200.
- **What the SEs cover:** one exploration seed and one adaptation seed. The SEs cover only evaluation-seed variation of one trained policy per arm.
- **Confounded compute:** the ensemble had five times the world-model training of the LCG arms. This is a difference worth testing on more seeds, not evidence that the ensemble explores better.

## LCG policy-reward scaling: data-scaled and reference-scaled divisors (2026-11-04)

The user asked for this test. Branch `lcg-reward-scaling`, from `7c1a383`, uncommitted. No configuration file changed: the arms set the new modes with launch overrides. Review: `docs/changes/lcg_reward_scaling_review.zip`. Evidence: `outputs/step27-lcg-reward-scaling/`.

**Motivation.**
- **The drop:** under the preset's fixed raw / 100, LCG's policy reward falls at every scorer refresh. The precision grows with the replay, and every score scales as 1/precision. In the online pilot, LCG-200's policy reward fell from 0.37 to 0.21 per step over 10 epochs. Uniform actions' scores fell just as much.
- **Effect on the policy:** the PMPO actor update ignores reward scale (sign split, mean-normalized weights), so the drop changes which transitions are preferred only through the critic.
- **The pilot's critic:** by epoch 10, LCG-200's critic overestimated (value bias +0.09), and only 32% of advantages were positive.

**What the raw score is** (approximately; damping negligible, h_j ≈ N · mean_replay G_j):

    r_raw(c) ≈ (1/N) · S(c),   S(c) = Σ_j G_j(c) / mean_replay G_j

- G_j(c) is the denoiser output's squared sensitivity to parameter j for candidate c.
- A typical replay transition has S ≈ d_S = 217,664, so r_raw ≈ d_S / N. That is 39.6 at N = 5,500 (observed 36.9) and 21.8 at N = 10,000 (observed 21.1). The raw reward's fall is almost entirely this 1/N.

**Implementation** (existing files only).
- `src/intrinsic_reward/lcg/lifecycle.py`: two new `reward_scaling` modes. Each sets one divisor per refresh, frozen for the round, included in the scorer's checkpoint state and restored on resume. Both are validated and recorded in the resume-compatibility settings.
  - `data_scaled`: divisor = `fixed_divisor` × `data_reference_steps` / N, so the reward is (raw / 100) × N / 5,500 ≈ S(c) / 550,000. A typical replay transition scores about 0.40.
  - `reference`: divisor = the mean raw score of reference rollouts, set by a new `calibrate()`. The reward is S(c) divided by the mean S of uniform-action imagined transitions; 1 means as novel as random actions. N and the precision's scale cancel exactly.
- `src/envs/world_model_env.py`: `imagine_candidates(obs, act, actions)` imagines open-loop from given prompts. It produces exactly `step()`'s candidates (tested) and does not touch the env's own rollout.
- `src/exploration_trainer.py`, after each LCG refresh in reference mode (`lcg_reference_scores`):
  - draws 32 replay prompts, seeded by the round in its own namespace (+2,000,000);
  - imagines the 10-step horizon under uniform random actions within the env bounds;
  - scores the 320 transitions with the refreshed scorer and calibrates.
  - The global RNG is seeded for this and restored afterwards. The divisor is logged per epoch as `lcg/reward_divisor`, and `lcg/reference_raw_mean` in reference mode.
- `tests/lcg/test_reward_scaling.py`: 9 new tests. The full suite passes: 713 tests (16 min, laptop).

**Runs.** The online pilot's settings, driver and shared initial replay, copied (11 files, checksums equal). LCG-200: 200 PMPO updates per epoch, 10 epochs, 10,000 env steps; the downstream protocol is the pilot's.
- **Option 1** (`datascaled`, kingplus0 GPU 1): `intrinsic_reward.reward_scaling=data_scaled +intrinsic_reward.data_reference_steps=5500`. The value 5,500 is the replay size at the first refresh, so epoch 1 is identical to the preset.
- **Option 2** (`reference`, GPU 0, beside another user's idle kernel, untouched): `intrinsic_reward.reward_scaling=reference intrinsic_reward.fixed_divisor=null +intrinsic_reward.reference_num_prompts=32`.
- **Dry run:** a laptop dry run of both passed first.
- **Exit codes and budget:** all stages exited 0. Exploration took 3,664 s and 3,696 s; adaptation 4,356 s and 4,405 s. GPU time was 8,156 s and 8,238 s, against a 4-hour cap per arm.
- **W&B:** project `LCG-exploration-policy-update-intervention`, group `lcg-reward-scaling-20261104`. Runs `lcgscale-{datascaled,reference}-exploration` and `-downstream`.
- **Calibration cost:** about 1.2 s per epoch (refresh 9.1 s against 7.9 s).
- **Launch connection:** the SSH session used to launch both queues dropped at 15:55. The detached queues were unaffected.

**Integrity** (`scripts/verify_scaling.py`, 40/40 checks).
- **Counts:** exact; no test interaction; every logged value finite.
- **Replay and milestones:** replay checksums match, the shared first 10 episodes are present, and the milestones are as planned.
- **Divisors:** every update's policy reward equals its raw reward over the round's logged divisor (relative error < 1e-5). The data-scaled divisor equals 100 × 5,500 / N at every refresh.
- **Shared start:** initial weights, controller RNG and post-injection RNG equal the pilot collector's and LCG-200's.
- **Match with LCG-200:** epoch-1 denoiser losses are identical in both arms.
  - Data-scaled: all 200 epoch-1 PMPO metric rows are identical to LCG-200's.
  - Reference: the first update's raw reward is identical, so the reference rollouts did not change the random stream.
- **Downstream:** the frozen denoiser is unchanged through adaptation, and the milestone and replay files are unchanged (31 files per arm).

**Policy reward per step** (epoch 1 → 10):
- LCG-200: 0.37 → 0.21.
- Data-scaled: 0.37 → 0.38; range 0.32–0.49, around the 0.40 of an average replay transition.
- Reference: 0.98 → 1.03; range 0.96–1.13. Its divisor fell from 37.7 to 19.0 as the raw score fell.

**Critic and imagined objective** (milestones at 5,500 / 7,500 / 10,000 steps; diagnostics as in the pilot):

| Arm | Positive-advantage fraction | Critic value bias (own reward units) | Policy - uniform, training bank |
| --- | --- | --- | --- |
| LCG-200 | 1.00 / 0.85 / 0.32 | -1.04 / -0.18 / +0.09 | -12.2 / -11.9 / +4.2 +- 1.5 |
| Data-scaled | 1.00 / 0.91 / 0.70 | -1.04 / -0.34 / -0.15 | -12.2 / -8.5 / +5.1 +- 1.6 |
| Reference | 1.00 / 0.87 / 0.79 | -2.71 / -0.78 / -0.67 | -13.0 / -2.9 / +3.1 +- 1.4 |

- **Per epoch in training:** the positive fraction in epochs 9–10 was 0.62 / 0.79 (data-scaled) and 0.73 / 0.81 (reference), against 0.42 / 0.41 (LCG-200).
- **Critic sign:** with either divisor the critic no longer ends up overestimating. This is consistent with the falling reward scale as the cause in LCG-200, on one seed.
- **Imagined objective:** at 10,000 steps it is within one SE across arms. The reference arm got close to uniform earlier (7,500 steps).

**Downstream** (Walker walk, seeds 3001–3030, at most 500 steps, the pilot's protocol). `downstream_analysis.py` compare agrees; the compatibility hash `8edc5d6f…` is the same as the pilot's arms.

| Arm | Mean | Median | Std | Min | Max | > 200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform | 45.6 | 45.4 | 8.9 | 23.9 | 66.5 | 0 |
| LCG-200 | 57.3 | 51.0 | 19.4 | 4.5 | 109.3 | 0 |
| LCG-200 data-scaled | 60.8 | 46.6 | 21.7 | 42.1 | 113.6 | 0 |
| LCG-200 reference-scaled | 65.4 | 60.7 | 21.2 | 34.2 | 127.3 | 0 |
| Full ensemble 200 | 79.0 | 69.7 | 40.3 | 3.7 | 165.9 | 0 |

| Paired | Mean +- SE | Seeds won / lost |
| --- | ---: | ---: |
| Data-scaled - LCG-200 | +3.5 +- 4.0 | 12 / 18 |
| Reference - LCG-200 | +8.2 +- 5.3 | 18 / 12 |
| Reference - data-scaled | +4.7 +- 5.9 | 16 / 14 |
| Reference - uniform | +19.8 +- 4.4 | 28 / 2 |
| Ensemble - reference | +13.6 +- 9.0 | 21 / 9 |

**Reading.**
- **Critic:** both divisors do what they were built for. The policy reward's scale stays steady, and the critic's late overestimation and collapsing positive-advantage fraction are gone.
- **Downstream:** the differences from LCG-200 are within about 1.5 SE (+3.5 and +8.2), so this pilot cannot show a benefit. No arm learned to walk, and the SEs cover only evaluation-seed variation of one trained policy per arm.
- **Choice:** reference scaling has the clearer meaning (1 = random actions) and cancels every overall scale change exactly, at about 1.2 s per epoch. Data scaling is free and deterministic and compensated almost as well in this run. Neither is in a preset; that decision is the user's.

## Reward-scaling continuation: both arms to 15,000 steps (2026-11-04)

The user asked for this continuation. Both scaling arms were continued from their epoch-10 checkpoints for exactly 10 more epochs, to 15,000 training env steps, and their new `steps_015000` milestones were adapted. The decision rule was fixed in advance:
- Downstream returns are primary.
- No new arms, setting changes, checkpoint selection or further training.

Branch `lcg-reward-scaling`, uncommitted. Review: `docs/changes/lcg_scaling_continuation_review.zip`. Evidence: `outputs/step28-scaling-continuation/`.

**Budget extension** (existing files only).
- **What blocked a plain resume:** `num_epochs_collect` is computed once from `collection.train.num_steps_total` at the initial collection and saved in the checkpoint. Raising the budget at resume therefore changed nothing. The `.run_is_over` sentinel also refuses a finished run.
- `src/exploration_trainer.py`, `extend_collection_budget()`, runs right after the checkpoint is loaded on resume:
  - Compares the configured budget with the planned one: steps so far + remaining epochs × steps per epoch.
  - **Equal:** no change.
  - **Lower, not a whole number of epochs, or past the planned end:** refused with an error.
  - **Otherwise:** adds the epochs and clears `final_test_done`, so the run's final test still happens at the new end.
  - Each extension is appended to `budget_extensions` (saved in the checkpoint), reported in the W&B resume summary (`resume.budget_extension`) and printed.
- `scripts/resume.sh`: the help text documents the procedure (rename the sentinel, pass the higher `collection.train.num_steps_total`, add the new milestone to `milestones.at_steps`). `resume.sh` runs the run folder's own copied source, so an extension needs a run created by code that has the method.
- `tests/resume/test_resume_and_milestones.py`:
  - A 30-step run extended to 42 trains exactly like a planned 42-step run: the same weights, counters, replay and metrics. Only the extra final test differs.
  - Lower and partial-epoch budgets are refused.
  - The full suite passes: 716 tests (19 min, laptop).

**How the runs were resumed.** These run folders' copied source predates the extension. So the continuation used the branch code (`repo_continuation/`) with each run's saved config and `resume.sh`'s overrides, plus:

    collection.train.num_steps_total=15000 "milestones.at_steps=[5500,7500,10000,15000]"

The diff between each run's own `src/` and the continuation source is the extension's additions only (`logs/<arm>_run_source_vs_continuation.diff`).

Before launch:
- The run folders and logs were copied read-only to `preserved_epoch10/` (980 files hashed).
- `.run_is_over` was renamed to `.run_is_over.20261104-epoch10`.

A laptop dry run of the full procedure (smoke W&B project) passed first.

**Runs** (kingplus0; one arm per RTX 5090, launched 17:04):

| Arm | GPU | Exploration | Diagnostics | Adaptation | GPU time (cap 4 h) |
| --- | --- | ---: | ---: | ---: | ---: |
| Data-scaled | 1 | 3,082 s | 46 s | 4,357 s | 7,495 s (2.08 h) |
| Reference | 0 | 3,104 s | 47 s | 4,399 s | 7,560 s (2.10 h) |

- **Stages:** all 12 exited 0.
- **Placement:** each process appears on its own GPU only in the monitors, and diagnostics record EGL on the same GPU. GPU 0 also held another user's idle kernel, which was not touched.
- **W&B** (project `LCG-exploration-policy-update-intervention`, group `lcg-reward-scaling-20261104`):
  - Exploration continued the same runs: `fwstc4da` (data-scaled) and `gmrsly90` (reference).
  - New adaptation runs: `r395td8d` and `0ksdkjlc` (`lcgscale-<arm>-downstream-steps_015000-walk-adaptation-seed0`).

**Integrity** (`scripts/verify_continuation.py`, 57/57 checks; the preserved copy's 980 hashes still match).
- **Resume:** each resume read the preserved epoch-10 `state.pt` (equal to the epoch-10 snapshot). The restored denoiser and controller weights equal that checkpoint's. Epoch 10, 10,000 steps, 13,600 / 2,000 updates, scorer round 10 and the epoch-10 divisor were all restored. No initial collection was run.
- **Final counters:** epoch 20, 15,000 steps, 17,600 denoiser and 4,000 PMPO updates. Each of epochs 11–20 had exactly 400 denoiser and 200 PMPO updates. No test interaction.
- **Extension record:** the extension `{at_epoch 10, 10,000 → 15,000, +10 epochs}` is in the record, the checkpoint and the W&B summary.
- **Replay:** checksums match, and the replay extends the epoch-10 replay.
- **Milestones:** the 5,500 / 7,500 / 10,000 milestones are byte-identical to the preserved copies, and the new 15,000 milestone has the expected counters.
- **History:** the W&B mirror's epoch-1–10 rows are unchanged, followed by epochs 11–20 only, with strictly increasing steps (no duplicated history). Every value is finite.
- **Divisors:** the data-scaled divisor equals 100 × 5,500 / N at all 20 refreshes; the reference divisor equals the logged reference mean. Every update's reward equals raw / divisor.
- **Adaptation:**
  - The frozen denoiser equals the 15k milestone's throughout.
  - The milestone and replay files are unchanged, and the 10k adaptation outputs are untouched.
  - The fresh reward predictor and controller start identically in both arms and equal the 10k adaptations' starts (seed-determined).
  - Each report covers seeds 3001–3030 with at most 500 steps.

**Downstream** (Walker walk, adaptation seed 0, the pilot's protocol `1e5d8307ecab`). `downstream_analysis.py` compare (laptop, with plots) agrees.

| Milestone | Mean | Median | Std | Min | Max | > 200 / 400 / 500 / 600 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Data-scaled, 15,000 | 40.7 | 37.4 | 12.1 | 22.8 | 72.6 | 0 / 0 / 0 / 0 |
| Reference, 15,000 | 50.6 | 51.2 | 21.2 | 4.3 | 91.8 | 0 / 0 / 0 / 0 |
| Data-scaled, 10,000 (step 27) | 60.8 | 46.6 | 21.7 | 42.1 | 113.6 | 0 / 0 / 0 / 0 |
| Reference, 10,000 (step 27) | 65.4 | 60.7 | 21.2 | 34.2 | 127.3 | 0 / 0 / 0 / 0 |
| Uniform, 10,000 (pilot, context) | 45.6 | 45.4 | 8.9 | 23.9 | 66.5 | 0 / 0 / 0 / 0 |

| Paired over 30 evaluation seeds | Mean +- SE | Seeds won / lost |
| --- | ---: | ---: |
| Reference − data-scaled, 15k | +9.9 +- 5.0 | 19 / 11 |
| Reference − data-scaled, 10k | +4.7 +- 5.9 | 16 / 14 |
| Data-scaled 15k − 10k | −20.1 +- 4.1 | 4 / 26 |
| Reference 15k − 10k | −14.8 +- 5.9 | 11 / 19 |
| Data-scaled 15k − uniform 10k | −4.9 +- 2.3 | 9 / 21 |
| Reference 15k − uniform 10k | +5.0 +- 3.9 | 19 / 11 |

**Supporting** (epochs 11–20; critic quantities are in each arm's own reward units).
- **Policy reward per step:**
  - Data-scaled: 0.33–0.63, epoch 20 0.45; the divisor fell from 52.4 to 36.7.
  - Reference: 1.05–1.25, rising late (1.25 at epoch 20); the divisor fell from 15.7 to 13.3.
  - Both scales stayed bounded.
- **Positive-advantage fraction (training):** swung more after epoch 10.
  - Data-scaled: 0.40–0.96, epoch 20 0.42.
  - Reference: 0.48–0.92, epoch 20 0.55.
- **Value–target correlation:** rose to 0.98 in both.
- **Imagined diagnostics** at milestones 1 / 5 / 10 / 20 (epochs 1–10 from step 27):

| Arm | Positive fraction | Value bias | Train bank: policy − uniform | Independent bank: policy − uniform |
| --- | --- | --- | --- | --- |
| Data-scaled | 1.00 / 0.91 / 0.70 / 0.30 | −1.04 / −0.34 / −0.14 / +0.14 | −12.2 / −8.5 / +5.1 / +26.3 +- 1.2 | −17.7 / −10.5 / +0.1 / +21.4 +- 1.2 |
| Reference | 1.00 / 0.87 / 0.79 / 0.25 | −2.71 / −0.78 / −0.67 / +0.50 | −13.0 / −2.9 / +3.1 / +23.4 +- 1.3 | −16.8 / −1.9 / +1.6 / +23.4 +- 1.5 |

The imagined objectives are raw scores under each arm's own epoch-20 scorer, so they are not comparable across arms.

**Reading** (under the fixed rule).
- **Downstream got worse with more exploration in both arms:**
  - Data-scaled by 20 (26 of 30 seeds), ending below the 10k uniform baseline.
  - Reference by 15 (about 2.5 SE).
  - No episode exceeded 200 in either arm at either milestone.
- **Reference vs data-scaled at 15k:** reference is ahead by +9.9 ± 5.0 (19/30 seeds). That is about 2 SE on one exploration seed and one adaptation seed, and the SE covers evaluation-seed variation only. Inconclusive: it does not establish that reference scaling is better.
- **Exploration objective vs downstream:** both policies became much better than uniform at their own imagined objective from epoch 10 to 20 (+5 → +26 and +3 → +23). The downstream returns moved the other way, so on this seed pair the scaled LCG objective's gains did not carry over to walking.
- **Critic:** with either scaling, the critic's sign change (value bias turning positive, positive fraction around 0.3) came later than LCG-200's (which had it at 10k). It was delayed, not prevented. A low positive fraction does not by itself mean a worse critic.
- **Not established:** whether the 15k drop is a property of more LCG exploration, of this seed, or of the adaptation. The pilot has no 15k uniform or LCG-200 milestone to compare with. Multiple seeds would be needed.

## Milestone schedule and pending-change commits (2026-11-04)

Configuration and repository housekeeping; no training was run.

- **Milestones** (`57fb6c3`): `config/exploration.yaml` now sets `milestones.every_steps: 10000` (was `null` at HEAD; a temporary working-tree `500` was replaced), with `at_steps: []` and `keep_last: null`. Targets count replay steps, including the 5,000 initial steps. Under the 100k presets' schedule (5,000 + 190 × 500), every target is an exact epoch boundary. Running `ExplorationTrainer.mark_due_milestone` over all 240 planned epochs of each of the five Walker presets publishes exactly `steps_010000` … `steps_100000` (epochs 11, 31, …, 191) and retires none; the training-only epochs add none. **Correction:** the epochs were wrong, from an off-by-one error in that check script; they are 10, 30, …, 190. See "Selected 100k exploration schedule".
- **Collection-budget extension** (step "Reward-scaling continuation") committed as `1c557b3`; `wandb==0.30.0` pinned as `30eb734`, the version in the verification environment.
- **Documents:** `docs/` is now tracked; only `docs/changes/` (review packages) stays ignored. `AGENTS.md`, the progress list and the Laplace design/audit status lines were updated to the committed settings: LCG reference scaling with 200 PMPO updates per epoch, the milestone schedule, and the Laplace commit and later package paths. Earlier sections of these notes are historical records and were not rewritten.

## Selected 100k exploration schedule (2026-11-04)

Configuration only; no training was run.

- **Settings** (`config/exploration.yaml`, inherited by all five Walker presets without overrides): `training.num_final_epochs: 0` (was 50) and `actor_critic.training.steps_first_epoch` / `steps_per_epoch` 200 / 200 (were 5,000 / 400). `walker_lcg_100k` no longer repeats the 200 / 200 override. Collection (5,000 initial + 190 × 500), the denoiser schedule (10,000 then 400), milestones, reward scaling, controller, architecture, evaluation protocol and budget extension are unchanged.
- **Epoch numbering** (`ExplorationTrainer._run`, `collect_initial_dataset`): the initial uniform collection fills replay to 5,000 steps before epoch 1, and `collect_initial_dataset` returns (100,000 − 5,000) / 500 = 190 collection epochs. Epoch k collects 500 steps before training, so its boundary is at 5,000 + 500k steps: 5,500 at epoch 1, 10,000 at epoch 10, 100,000 at epoch 190. Milestones fall at epochs 10, 30, …, 190. The earlier check ("Milestone schedule and pending-change commits") reported epochs 11, 31, …, 191 because its script set epoch k's boundary to 5,000 + 500(k − 1) and counted 191 collection epochs; the trainer was right.
- **Totals:** 190 epochs, no training-only tail; 85,600 denoiser updates per denoiser (428,000 over the full ensemble's 5 members); 38,000 PMPO updates for each learned method and 0 for uniform, which builds no PMPO optimizer. Test diagnostics are extra: 19 periodic collections of 4 episodes plus the final 100.
- **Test:** `tests/exploration/test_uniform_preset.py::test_walker_100k_presets_resolve_the_selected_schedule` composes each preset, steps through the trainer's epoch arithmetic, calls `ExplorationTrainer.mark_due_milestone` at every boundary, and asserts the values above. `test_lcg_preset.py` now expects LCG's PMPO schedule to equal uniform's.
- **Earlier records:** the pilots, continuation and projections above ran or assumed other schedules, often by override (for example 240 epochs, 5,000-then-400 or 200 PMPO updates). They are historical and were not changed.

## Physical-state recording for exploration coverage analysis (2026-11-05)

Implemented and validated; committed at the user's request before the independent Codex review (the review package was built from the identical uncommitted tree). Starting point: `e4a2371`, clean working tree. Evidence: `outputs/step30-physical-state/`; review package `docs/changes/physical_state_review.zip`. No coverage metric, feature, clustering, plot or W&B record is part of this step.

**Configuration** (`config/exploration.yaml`, saved with each run in `config/exploration.yaml`):
- `collection.train.record_physical_state: true`: new exploration training collections record states.
- `collection.test.record_physical_state: false`: evaluation (test) episodes do not.
- A saved config without the key (runs from before this step) records nothing (`OmegaConf.select(..., default=False)`). The flag reaches `make_dm_control_env(record_physical_state=...)` → `DMControlEnv` and `TorchEnv`. `env.train` is unchanged, so milestone manifests' `model.env`, downstream comparisons and the resume env-config check are unaffected. The W&B config (the full resolved config) gains the key; no W&B metric or summary entry is added.

**Storage:** `Episode.physical_state`, an optional field saved in the existing episode `.pt` file: a dict of CPU tensors.

| Key | Shape, dtype | Meaning |
| --- | --- | --- |
| `qpos` | (T, nq) float64 | MuJoCo `qpos` at which `obs[t]` was rendered, i.e. before `act[t]` |
| `qvel` | (T, nv) float64 | likewise `qvel` |
| `recorded` | (T,) bool | False where the frame's state is unknown; those `qpos`/`qvel` rows are NaN |
| `final_qpos`, `final_qvel` | (nq,), (nv,) float64 | state of `info["final_observation"]`: after the last action, before the vector env's autoreset. Ended episodes with a recorded final state only |

- nq and nv come from the simulator (Walker: 9 and 9); nothing assumes Walker.
- A milestone's `dataset.pt` adds `successor_physical_states` (`{episode_id: {qpos, qvel}}`), the state of each recorded `successor_observations` frame. `load_milestone` returns it as `Milestone.successor_physical_states`, `{}` for older milestones.
- The milestone manifest (JSON) gains a descriptive `physical_state` entry: the recording setting, keys, alignment, missing-data convention and the analysis-only use. There is no per-state JSON file and no YAML metadata.

**Exact temporal alignment** (`DMControlEnv` → `TorchEnv` → collector):
- `DMControlEnv.reset` and `step` copy `physics.data.qpos`/`qvel` (`np.array(copy=True)`) after the frame's last physics substep. That is the state the returned frame is rendered from; rendering does not change it.
- `TorchEnv` keeps each env's current state. It returns it as step `info["physical_state"]`: the state of the observation the action was taken from. A finished env's own step state comes from `final_info` (`info["final_physical_state"]`, in `final_observation` order), because its top-level info is the autoreset's, which supplies the new episode's first state.
- `set_env_states` (resume) re-reads the restored simulators' states. That is the current observation's state; nothing is stepped, reset or drawn.
- The collector buffers these states beside the transitions, not in the transition columns. When it writes a partial or finished episode, the rows and final state are added with `stack_physical_states`, and `Episode.__add__` extends them with the continuation (`join_physical_states`).

**Missing and mixed data:**
- Episodes collected without recording, including every historical file, have `physical_state=None`. Their files are byte-for-byte the previous format: the key is not written.
- When an episode mixes recorded and unrecorded parts (recording switched on at a resume, or a frame whose state is unknown), the unrecorded rows are NaN with `recorded=False`. No values are made up or carried over.
- Dimension mismatches and malformed fields (shapes, dtypes, non-NaN unrecorded rows, half a final state) are refused when the Episode is built.
- A missing successor entry means unknown.

**Analysis only:**
- Training reads none of it. Segments, batches and `info` never carry it, because `make_segment` copies only the transition columns and `info` holds no physical state.
- No policy input, world-model or reward-model training, intrinsic scoring, replay weighting or downstream adaptation reads it.
- Recording adds no random draws, resets or simulation steps. Exact-resume simulator snapshots (`collection_state`) are unchanged.

**Validation:**
- **Focused tests** (`tests/collection/test_collection_task_labels.py`):
  - Frames and states align at reset, ordinary steps and termination before autoreset, across a partial-episode extension and save/load. Each stored frame and state equals an independently replayed raw dm_control environment's rendered frame and `qpos`/`qvel` at the same instant.
  - Copies never alias simulator memory.
  - Recording on versus off gives identical frames, rewards, labels, endings and random streams for every verified domain group, with the domain's own nq/nv.
  - Episode extension, mixing, validation and save/load work. Unrecorded files keep the old format. Segments and batches never carry the field.
  - The collector marks unknown states and aligns final states.
- **Trainer tests** (`tests/resume/test_resume_and_milestones.py`):
  - The exact-resume comparison now includes physical states.
  - Milestone readback covers cut rows and the successor state equal to the next stored row.
  - Evaluation episodes are not recorded.
  - Learned exploration with recording on versus off is bit-identical in replay transitions, weights, optimizers, schedulers, RNG, logs and milestone identities; files without recording have no key.
  - A run collected without recording and resumed with it has explicitly mixed episodes, whose recorded rows equal the always-recorded run's.
- **Smoke** (CPU, W&B disabled, `outputs/step30-physical-state/scripts/run_smoke.sh`):
  - The run was a real uniform-random Walker exploration of 750 + 4 × 250 = 1,750 steps (4 episodes), with milestones at 1,250 (cutting episode 2 at 250 steps) and 1,750. Downstream training and evaluation then ran from the 1,250 milestone. All exit codes were 0, about 1¼ minutes in total.
  - `verify_smoke.py` passed 23/23. All 1,750 stored frames, the 3 final frames and both successor frames equal frames re-rendered from their own recorded states in a fresh simulator. States are float64, finite, all recorded and shaped (T, 9). Each successor state equals the continued episode's next row, and the evaluation episode has no states.
- **Test runs:**
  - Focused suites (`tests/collection`, `tests/resume`, `tests/downstream`, `tests/exploration`): 427 passed.
  - First full suite: 771 passed, 2 failed. `test_pmpo_frames.py` builds a `DMControlEnv` with `__new__` and lacked the new attribute; this was an implementation failure, fixed by a class-level default `record_physical_state = False`.
  - After the fix, the smoke was rerun (identical results) and the full suite on the final code gave 773 passed, 0 failed.
- **Storage overhead (smoke):** 147.6 bytes per frame, +1.19% of the episode files (21.99 MB versus 21.74 MB without the field; 64×64 RGB frames). Over 100,000 frames this is about 14.8 MB per run.
- **Environment note (pre-existing, not changed):**
  - `common.devices=cpu` sets `CUDA_VISIBLE_DEVICES=""`. On this laptop's torch 2.7.1+cu128, that reports CUDA as available with 0 devices, so the trainer chose `cuda` and failed in its GPU report.
  - The smoke instead used `CUDA_VISIBLE_DEVICES=-1` with `common.devices=all`, and ran on CPU. The two failed attempts are kept under `logs/failed_attempt_{1,2}/`.

## Downstream folder mode: milestone-step selection; progress bars in logs (2026-11-05)

Committed at the user's request; the review package was built from the identical uncommitted tree. Starting point `6cf8a07`, clean working tree. Evidence: `outputs/step31-milestone-selection/`; review package `docs/changes/milestone_selection_review.zip`. No training, controller, W&B or evaluation change; the exploration milestone-saving configuration is unchanged.

**Selection (`config/downstream.yaml`, `initialization.milestone_steps: null`):**
- `null`: every discovered milestone, as before.
- A non-empty list of distinct positive integers: only the milestones whose manifest `collected_steps` is listed. Example production selection: `[10000, 20000, ..., 100000]`.
- Rejected before anything runs: a non-list, an empty list, booleans, non-integers, values below 1, duplicates, and a non-null filter with `initialization.milestone_path` (single-milestone mode). It is never silently ignored.
- Launcher-only, like `milestones_dir`:
  - `downstream_main.pop_milestone_source` (formerly `pop_milestones_dir`) validates and removes both keys. It is listed in `LAUNCHER_KEYS`, and every child command and recovery command carries `++initialization.milestone_steps=null`, so a filter set in YAML is cleared too.
  - `DownstreamTrainer` drops the key and refuses a non-null value.
  - Each trainer still handles one milestone × one task × one adaptation seed. Trainer configs, saved configs and resume checks are unchanged.

**Launcher (`run_milestones` → `preflight`):**
- Milestones are discovered and validated by the existing `discover_milestones`. Selection uses manifest step counts, never folder names, and keeps ascending step order whatever the order of the list.
- Every requested step that no milestone records is reported in one error, together with the selected milestones' file, label and output problems, before any child starts.
- Output-conflict and task-label checks apply to the selected milestones only. An unselected milestone's completed or incomplete outputs, or broken files, neither block the batch nor are touched.
- The plan lists only the selected milestones, numbered 1..N. Its heading says `Downstream batch: N milestones (selected by initialization.milestone_steps from M milestones in the folder)`.
- Single-milestone train, resume and evaluate, and the printed recovery commands, behave as before; recovery commands gain the clearing override.

**Progress bars written to a log or pipe (`utils.Progress`):**
- Reported symptom: the downstream 30-seed evaluation bar showed as many repeated lines.
- Cause: with stdout not a terminal (`nohup ... > out`, watched with `tail -f`, or `| tee`), tqdm cannot know the window width and draws a 10-cell bar regardless of the line's length. The pilot's server logs show the evaluation line at 86 characters, the longest bar (reward predictor 84, policy 82). In a narrower window it wraps, and each `\r` redraw then appears as a new line.
- Fix:
  - When stdout is not a terminal, a line that would exceed `log_columns()` with tqdm's default 10-cell bar gets a narrower bar, one cell at least, so it fits. Lines that already fit keep the default 10-cell bar and are never widened. `log_columns()` is stderr's terminal width if stderr is a terminal (`| tee`), else 80.
  - The text is never cut, and labels are unchanged.
  - On a terminal, tqdm's dynamic width applies as before.
  - The real 30-seed evaluation bar (`evidence/eval_bar_probe_30_seeds_compact.out`): the first redraw (`? ep/s`, 79 characters) keeps 10 cells; once the rate appears, the bar narrows to 4–5 cells. All 32 redraws are 78–80 characters, with nothing cut.
- `tests/exploration/test_device_report.py` pins it.
- The downstream evaluation bar's label is now `Evaluation` (was `Evaluation episodes (seeds 3001-3030)`), at the user's request. The seeds remain in the banner (`Evaluation: 30 seeds (3001–3030)`) and the saved evaluation report. The 30-seed bar is 51–59 characters with the full 10-cell bar (`evidence/eval_bar_probe_30_seeds_label_evaluation.out`); focused formatting tests: 87 passed. The exploration test-collection bar keeps its `Evaluation episodes (seeds ...)` label.
- Not changed: a window narrower than 80 columns (or than a long label) can still wrap. On this Windows laptop, a redirected stdout is encoded in the locale code page (cp950), which is separate and does not affect the server.

**Tests:**
- `tests/downstream/test_downstream_trainer.py`:
  - validation of the setting, including single-mode rejection;
  - manifest-based selection in ascending order despite misleading folder names;
  - all missing steps and the selected milestones' problems reported before any child;
  - completed, incomplete and broken unselected milestones neither blocking nor touched;
  - the plan listing only selected milestones, numbered 1..N;
  - a real command-line folder run adapting only the selected milestone with its inputs unchanged;
  - a real YAML-defined selection reaching the children cleared;
  - the existing folder-mode tests, updated for the second clearing override.
- `tests/downstream`: 102 passed (before the progress change). Terminal-output tests after the progress change: 61 passed. Full suite: 778 passed, 0 failed, before the compact-bar refinement; after it, the focused formatting tests (device report, exploration logging, downstream screen and plan) gave 75 passed.

## Production configuration cleanup and requirements check (2026-11-05)

Uncommitted, pending independent review. Starting point `073774c`, clean working tree. Evidence: `outputs/step32-config-cleanup/`. Review package: `docs/changes/config_cleanup_review.zip`. No source module, algorithm or hyperparameter changed; no training was run.

**Layout:**
- `config/exploration.yaml` / `config/downstream.yaml`: shared settings.
- `config/intrinsic_reward/*.yaml`: method scoring.
- `config/experiment/*.yaml`: domain-free method presets.
- Launch scripts: domain and task group, `common.seed`, `common.devices`, `hydra.run.dir` and the W&B identity.
- `config/env/dm_control.yaml` is kept, with Walker `[stand, walk, run]` only as the default. No per-environment YAML files.

**Exploration (`exploration.yaml`):**
- `collection.train.steps_per_epoch` 100 → 500. The other schedule values were already the selected ones: 100,000 steps, 5,000 initial uniform, denoiser 10,000/400, PMPO 200/200, batch 32, no training-only epochs, recording on for training and off for evaluation, debug metrics off.
- Milestones: `at_steps` lists 20 counts (5,500, then 10,000, 15,000, …, 100,000), `every_steps: null`, `keep_last: null`. They are published at epochs 1, 10, 20, …, 190 and all retained (checked with `mark_due_milestone`).
- Unchanged and still undecided: test diagnostics every 10 epochs (4 episodes) and a separate final 100, with `evaluation.reset_seeds` / `final_reset_seeds: null` (random reset seeds).

**Downstream (`downstream.yaml`):**
- Reward predictor 1,000 → 5,000 updates; policy 1,000 → 10,000 updates; both batch 32.
- Unchanged: the Gaussian H10 sign-only controller, seed 0, member 0, evaluation seeds 3001–3030 and 500 steps, debug off.
- `initialization.milestone_steps: null` stays generic.
- New preset `experiment/downstream_10_milestones.yaml`: `milestone_path: null`, `milestones_dir: ???`, `milestone_steps: [10000, …, 100000]`. Without a folder, the launcher's "exactly one of milestone_path and milestones_dir" error applies.

**Presets:**
- `walker_{uniform,lcg,ensemble,laplace,plan2explore}_100k.yaml` were renamed with `git mv` to `{…}_100k.yaml`. Removed from them: the Walker environment, the collection schedule, evaluation settings, `common.seed: 0` and the W&B settings (`online`, project, name). W&B is now disabled unless a launcher enables it.
- Kept: uniform collection with the scorer disabled for `uniform_100k`; advantage-weighted PMPO with cap 5 for all four learned presets; `training.compile_wm: false` for ensemble, Laplace and Plan2Explore (LCG and uniform use the default `true`).

**Scoring:**
- `intrinsic_reward/lcg.yaml` now holds the production LCG settings: R1 `unet.u_blocks.3.resblocks.0.*` with guards 10 / 217,664; 5 candidate probes, chunk 32; 320 × 3 historical reference; damping 1e-4; beta 1.0; `reward_scaling: reference`, `reference_num_prompts: 32`, `rms_enabled: false`.
  - Previously the group default was the broad selector, MC 12, chunk 16 and RMS scaling, and the preset overrode it.
  - The other scaling modes and selectors remain available by override.
- `laplace_disagreement.yaml` gains the same R1 guards.
- Ensemble and Plan2Explore settings are unchanged.
- Stale comments were corrected, including "the maintained LCG preset is sign-only" and the "future baseline" header.

**Removed:** `config/pmpo_diagnostic/bounded*.yaml` (4 files). `src/pmpo_diagnostic.py` and its tests are kept; the two tests that used `+pmpo_diagnostic=bounded` pass the same values inline. The historical `scripts/run_walker_100k_pmpo_*.sh` Stage 3 launchers still name `+pmpo_diagnostic=bounded_full_loop`. They were already not runnable against the current trainer (`env.train.task_name`, disabled intrinsic reward under policy collection), are kept for provenance, and the README now says so.

**Verification:**
- `outputs/step32-config-cleanup/scripts/resolve_presets.py` composed every method preset with every accepted group (35 compositions) and the downstream preset. Results are in `evidence/config_resolution.json`, with 0 failures. It checked that:
  - the launched domain and task list reach `env.train` and `env.test`, with no preset resetting them to Walker;
  - the schedule, 20 milestones, recording, debug flags, objective and compilation are the selected ones;
  - the R1 selector with its guards holds on the real denoiser for every accepted domain's action dimension (14 checks, all 10 tensors / 217,664 scalars).
- Default run folders are `outputs/<method>/<domain>/seed_<N>`, so the two Cartpole groups collide unless their launchers set distinct `hydra.run.dir` values. This is a launcher requirement.

**Tests:** tiny test configurations now null the R1 guards and keep RMS scaling explicitly (tiny models and replays; `TINY` in `test_exploration_trainer.py`, `REAL` in the resume tests, `test_pmpo_integration.py`, the Laplace list). Preset names were updated, and the schedule test now covers all 5 presets × 7 task groups. The full suite before the test updates gave 82 failed (expected: guards, renamed presets, removed diagnostic preset, scaling defaults); on the final code it gave 808 passed, 0 failed.

**Requirements (laptop, Windows, Python 3.10.20):**
- `pip check`: no broken requirements.
- Every pin equals the installed version (`hydra-core==1.3` is 1.3.0). torch 2.7.1+cu128 with CUDA available, and W&B 0.30.0 kept.
- Maintained imports in `src/` are all declared:
  - `matplotlib` is imported only by the analysis's plotting subprocess;
  - `pillow` and `pygame` are used only by `src/game` / `play.py`;
  - `h5py` is not imported by maintained code;
  - `opencv-python` is used only by historical diagnostic scripts.
  Both are kept pending a decision.
- Tests also need `pytest`, which is not pinned.
- The README now documents the cu128 wheel index for the RTX 5090 (driver 570, CUDA 12.8) and the EGL requirements.
- **Not checked here (servers):** `pip check`, imports, CUDA/compilation, EGL rendering and W&B login on both RTX 5090 servers. The kingplus0 environment is known to lack matplotlib. No server environment was touched.

**Open decisions:**
- Exploration test reset seeds: periodic, proposed 3001–3004; final, 100 random episodes kept for now.
- W&B projects, groups and run names for both phases.
- The launch scripts themselves, including distinct Cartpole run folders.
- Whether to drop `h5py` / `opencv-python`.
- The server environment setup.

## Production launch scripts, scripts/ cleanup and Plan2Explore storage smoke (2026-11-05)

**Launchers** (`scripts/run_<method>_100k.sh` for lcg, uniform, ensemble, laplace, plan2explore), built from the user's `scripts/reference.sh`:
- **Order:** each explores the seven accepted groups first, then adapts every task of every group on the ten selected milestones, one task at a time.
- **Groups:** walker, hopper, acrobot, cartpole-balance, cartpole-swingup, humanoid and humanoid_CMU. The two Cartpole groups use separate folders but the same `cartpole` domain.
- **Outputs:**
  - run folders: `outputs/<method label>/<group>/seed_0` (`lcg`, `uniform`, `full-ensemble`, `laplace-ensemble`, `plan2explore`);
  - terminal logs: one per run in `logs/<label>/<date>-<id>/`, written through `tee`.
- **Settings at the top:** GPU, Python, seeds and the W&B projects (`EXPLORATION_PROJECT`, `DOWNSTREAM_PROJECT`). The scripts run with `set -euo pipefail` and unbuffered Python.
- **Device binding:** `CUDA_VISIBLE_DEVICES`, `MUJOCO_EGL_DEVICE_ID` and `common.devices` all name the selected GPU.
- **Budgets:** read from the YAML.
- **Safety:** existing run folders are refused before anything starts; resume works as before (`scripts/resume.sh`).
- **Changes from the reference:**
  - preset and output label per method;
  - CUDA bound explicitly;
  - downstream `wandb.name` = `<label>-<group>-exploration-seed0`, so the two Cartpole groups have distinct run names;
  - the log folder is created only after the existing-output check;
  - the reference's stray last line (`previouse generate one is wrong?`) is dropped; it would have exited 127 after all runs. The user later removed `reference.sh`; `run_lcg_100k.sh` replaces it.
- **Downstream W&B files:** the downstream entry point sets no W&B directory, so with online W&B these local files go to the launch checkout's `wandb/`.

**scripts/ cleanup** (`git rm`; all files were tracked):
- **Removed:**
  - `controller_diagnostic/` (27 files);
  - `historical_precision_diagnostic/` (2);
  - `lcg_diagnostic/` (235, including 7.3 MB of datasets and frames);
  - the four `run_walker_100k_pmpo_*.sh`.

  Exploration run folders copy `scripts/`, so each run folder is now about 7 MB smaller.
- **Kept:** `resume.sh` and `import_run.py` (DIAMOND's remote-run import utility, unreferenced).
- **Final `scripts/`:** `import_run.py`, `resume.sh`, `run_{lcg,uniform,ensemble,laplace,plan2explore}_100k.sh`, `smoke.sh`.
- **Test removed:** `test_historical_initialization_fingerprints_ignore_tensor_storage_identity` loaded `lcg_diagnostic/pretrained_collectors/create_theta0.py`, a historical script, so it went with the folder. `src/pmpo_diagnostic.py` is kept.

**Exploration evaluation (final settings):**
- Periodic: every 10 epochs, 4 episodes with reset seeds 3001–3004.
- Final: one separate evaluation of 30 episodes with reset seeds 3001–3030 (`collection.test.num_final_episodes: 30`, `evaluation.final_reset_seeds: [3001, …, 3030]`).
- History: while `num_final_episodes` was already 30, the seed list was still 3001–3004. `check_reset_seeds` rejected that in the trainer's constructor, after W&B had started and the run folder existed, for all 35 production compositions. The user's correction added seeds 3005–3030.
- After the correction, all 35 compositions pass the trainer's episode-count/reset-seed check (`resolve_launch_configs.py`: 0 mismatches, 0 failures).

**Correction to the previous section:** commit `0874335` included the user's change of `evaluation.reset_seeds` / `final_reset_seeds` to 3001–3004. The "808 passed" run predates that change.
- With fixed seeds, 91 tests failed. They set their own episode counts (1–4) but inherited the four seeds.
- The tiny override lists (`TINY`, `REAL`, `test_pmpo_integration.py`) now also set `evaluation.reset_seeds=null` and `evaluation.final_reset_seeds=null`.
- Affected files: 361 passed. Full suite: 807 passed, 0 failed (808 minus the removed fingerprint test).

**Launcher checks** (CPU; `outputs/step33-launchers/scripts/`):
- **Stub runs:** every launcher ran against a stub interpreter that records its commands, in a throwaway copy of the repository. 25 commands each: 7 explorations, then 18 downstream tasks.
- **Composition:** all commands were composed with Hydra: 35 production explorations and 90 production downstream commands, plus the smoke's. Checked: preset to output label, domain/tasks, seed, devices, 100k schedule, 20 milestones, update budgets, evaluation, compilation, W&B identities, the ten selected milestones, 30 evaluation seeds and 500 steps. 0 failures apart from the final-seed mismatch above.
- **Failure propagation:**
  - an exploration failure stops at that call (exit 3);
  - a downstream failure stops at that call (exit 3);
  - a missing milestones folder stops before the first downstream call (exit 1);
  - an existing run folder is refused before any command or log folder (exit 1).

**Smoke** (`scripts/smoke.sh`, written by the user):
- **Structure:** `run_plan2explore_100k.sh` with the same paths, logs and W&B identities. Only these differ:
  - the groups: Walker `[walk]` and Hopper `[stand]`;
  - inline overrides: one denoiser, one PMPO and one predictor-fit update per epoch, all warmups 0;
  - one-episode evaluations with seed 3001;
  - one reward-model and one policy downstream update.
- **Why not LCG:** LCG was tried first. Its one denoiser update fell inside the warmup (learning rate 0 at step 0), so the zero-initialized output stayed zero and LCG's reference calibration refused (`Reference scores must be non-empty with a finite positive mean, got 0.0`) at epoch 1.
- **Where it ran:** kingplus0 GPU 1, in the separate checkout `~/scfeng/LCG-plan2explore-smoke-20261105` (working-tree tar `6892b73c…`, `smoke.sh` `e1a78c4b…`). W&B was online in `lcg-reward-free-exploration-smoke` / `lcg-reward-free-downstream-smoke`. 17:24–19:25, exit 0.
- **Verification** (`server/verify_smoke.py`, 0 failures):
  - **Exploration (each run):** 100,000 training env steps; 190 denoiser and 190 PMPO updates; 190 fits per predictor head.
  - **Milestones:** exactly the 20 at 5,500 and 10k–100k (epochs 1 and 10–190), with manifest counters equal to their epoch.
  - **Exploration evaluations:** 20 reports (epochs 10–190 and the final one), each one episode with seed 3001.
  - **Physical state:** in all 200 training episode files (qpos/qvel/recorded); none in test replay.
  - **Downstream:** each task adapted exactly steps 10k–100k from its own run's milestones, with `num_updates` {rew_end_model 1, actor_critic 1}. Each output has one evaluation report with one episode (seed 3001), no checkpoints and no work folders.
  - **Sources unchanged:** milestones and replay (263 files per run) had the same hashes as snapshots taken when each exploration finished.
  - **Formats:** YAML configs, JSON metadata, JSONL metrics.
  - **W&B:** 2 + 20 runs, all finished, with ids matching the local `wandb_run.json` files.
- **Elapsed time:** exploration 57.7 min (Walker) + 58.2 min (Hopper); downstream 3.0 min (walk) + 2.0 min (stand); 120.8 min total.
  - Of each exploration's ~56 epoch-minutes, the Plan2Explore refresh took ~47 (pool encoding and scoring, independent of `fit_updates`), collection ~7, and denoiser/PMPO/test ~3.
- **Storage** (each inode counted once; no hard links):

| Phase / category | Walker | Hopper |
| --- | --- | --- |
| Exploration: replay (200 episodes) | 1.264 GB | 1.260 GB |
| Exploration: milestones (20 × denoiser + index) | 0.355 GB | 0.355 GB |
| Exploration: checkpoints (state + kept agent versions) | 0.391 GB (peak 0.420) | 0.391 GB (peak 0.461) |
| Exploration: W&B, metrics, logs, evaluation, run copy | 0.010 GB | 0.010 GB |
| **Exploration total** | **2.019 GB** | **2.015 GB** |
| Downstream: 10 outputs (reward model 23.6 MB + policy 8.3 MB each) | 0.319 GB | 0.319 GB |
| Downstream: metrics, evaluation, logs; W&B (both tasks, `wandb/`) | < 1 MB | < 1 MB (W&B 1 MB total) |
| **Downstream total** | **0.319 GB** | **0.319 GB** |

- **Peaks:** final usage equals the sampled peak except the checkpoint rotation (+30–70 MB while a new agent version is written before an old one is removed).

**Production projection for metrics and logs** (these scale with updates and runtime, not with the smoke):
- **Downstream metrics:** one JSONL row per update, about 0.3 KB for the reward model and 1.6 KB for the policy. That is about 17.5 MB per production adaptation (5,000 + 10,000 updates), about 3.2 GB per method for 180 adaptations, plus a comparable local W&B history.
- **Exploration metrics:** `wandb_history.jsonl` holds about 3 rows per epoch whatever the update count, about 0.74 MB per run.
- **Terminal logs:** about 2 MB per smoke exploration (progress bars, time-throttled), so they grow with runtime.
- **Replay, milestones, checkpoints and downstream models:** the same per run in production as in the smoke.
