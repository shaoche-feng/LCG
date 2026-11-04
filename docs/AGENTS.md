# AGENTS.md

Instructions for Codex, Claude Code and other coding agents working on this reward-free experiment repository.

Updated: 2026-11-04, branch `lcg-reward-scaling`: reward scaling (`e5b1e61`), the collection-budget extension, and the selected milestone schedule are committed. This identifies the snapshot the guide was checked against; inspect the actual checkout before claiming its current commit or status.

## Project and authority

This project studies task-agnostic exploration with DIAMOND-style diffusion world models, followed by task-specific adaptation using a frozen milestone and its fixed replay. The main scientific comparison is downstream task performance across exploration methods and data budgets.

- The user’s current instructions take precedence over this guide.
- Inspect the actual checkout, resolved configuration and current code before making changes. Do not assume an old server path or worktree is the active repository.
- Treat saved configurations and executable code as the description of existing behavior. Design documents describe intent; the notes record implementation changes and limitations. Report contradictions instead of silently choosing a historical description.
- Historical Stage 3 Beta/H15 experiments are context, not this repository’s current controller specification. Do not copy their settings or results into the reward-free experiment as defaults.
- Current software implementation and bounded validation are complete. Selected so far: LCG reference reward scaling with 200 PMPO updates per epoch (`walker_lcg_100k`), and ten milestones at 10k, 20k, …, 100k training environment steps (`milestones.every_steps: 10000`). Other budgets, schedules and the seed design are still pending; the remaining numerical defaults are executable placeholders, not an approved final protocol.

## Working rules

- Follow the existing DIAMOND code structure. Reuse appropriate modules and existing test/documentation files.
- Do not add registries, factories, helper modules, frameworks or compatibility layers without a concrete need within the requested scope.
- Keep a structural refactor separate from algorithm, configuration and training-behavior changes.
- Do not weaken checks, numerical assertions or resume comparisons merely to make tests pass.
- Preserve unrelated changes. Do not commit, push, launch long experiments, sync W&B or delete experiment data unless authorized.
- Historical outputs, copied source, logs and review evidence are immutable evidence. Do not edit them to match new code.
- Put new smoke outputs and review evidence in the worktree’s gitignored `outputs/` hierarchy. Keep as-built notes and progress in the existing documents.
- Report what was changed, what was actually tested, exit codes, limitations and whether anything remains uncommitted. Separate integration evidence from learning-quality evidence.

## Code map

```text
src/
  exploration_main.py       exploration entry point
  exploration_trainer.py    exploration orchestration
  downstream_main.py        single/folder adaptation launcher: train, resume, evaluate
  downstream_trainer.py     one-milestone adaptation, evaluation and W&B
  downstream_analysis.py    collect/compare entry point: result tables, validation and plots (imports no trainer)
  agent.py                  model construction
  utils.py                  shared utilities, progress, timing, RNG and device reporting
  pmpo_diagnostic.py         existing opt-in PMPO controller diagnostics
  intrinsic_reward/
    __init__.py             package documentation; no eager method initialization
    lcg/                    historical precision, JVP bank, scoring, scaling and lifecycle
    ensemble.py             full diffusion-ensemble training and disagreement scorer
    plan2explore.py          frozen features, predictor fitting and feature disagreement
    laplace.py              sampled-parameter disagreement using shared LCG precision
    intrinsic_eval.py       real-episode scoring with frozen intrinsic scorers
  models/                   diffusion denoiser, reward/end predictor and PMPO controller
  data/                     replay, sampling, task-label batching, milestones and output/report identity helpers
  envs/                     real and imagined environments
  coroutines/               collectors and frame history
config/
  exploration.yaml
  downstream.yaml
  agent/pmpo_beta.yaml
  intrinsic_reward/
  experiment/
scripts/resume.sh
```

There is no separate `device_report.py`: its helpers now live in `utils.py`. Device probes run only when called, not at import time. `pmpo_diagnostic.py` stays separate because the trainer imports its opt-in diagnostic mode. The small `ensemble_disagreement` calculation remains in `envs/world_model_env.py` and is re-exported by the ensemble module; do not create an import cycle to relocate it.

## Exploration contract

One exploration run represents one method, one domain/verified compatible task group and one exploration seed. The initial group is Walker `[stand, walk, run]`.

1. Collect initial uniform actions and record all compatible task-reward columns.
2. For each collection epoch: collect more transitions, train the denoiser(s), refresh the intrinsic method, then train the exploration policy in imagination.
3. Run scheduled diagnostics and publish configured milestones. Optional training-only epochs collect no additional training transitions.

Collection precedes that epoch’s model/policy updates. Consequently, the first policy collection uses the initial policy. Do not reorder the loop during cleanup.

- Task rewards are retained in replay for later adaptation but are excluded from denoiser inputs and exploration learning. They must not affect intrinsic scores, features, predictor fitting, bootstrap choices or cache identity.
- Exploration constructs no task-reward/end predictor. Uniform random never acts with or updates the policy and constructs no intrinsic scorer.
- Full ensemble is the only method that trains multiple denoisers. The other methods train one denoiser.
- Imagined trajectories use the established horizon cutoff and bootstrap handling, not an untrained termination predictor.
- Test collection is separate from the training-data budget and must remain isolated from training RNG, modes and replay.

## Current shared controller

The implementation retains the name `PMPOBeta`, but the current configured actor is a tanh-squashed diagonal Gaussian. Do not infer its distribution from the class or filename.

| Setting | Current value/meaning |
| --- | --- |
| Actor distribution | `tanh_gaussian` |
| Policy frame stack | 4 RGB frames |
| Imagination horizon | 10 |
| Imagined policy initialization | `repeat_anchor`: repeat the last replay-prompt frame |
| World-model initialization | Preserve the real replay frames/actions; do not replace its context with repeated frames |
| Exploration actor objective | `advantage_weighted` in all four learned-method Walker presets |
| Advantage weights | Detached; magnitude divided by its sign-group mean, capped at 5 |
| Downstream actor objective | `sign`, explicitly enforced |
| Discount / lambda | 0.985 / 0.95 |
| PMPO alpha / prior KL coefficient | 0.5 / 0.3 |
| Entropy coefficient | 0.01 |
| Prior refresh interval | 500 updates |
| EMA critic regularization coefficient / rate | 1.0 / 0.02 |
| Controller gradient clip | 10 |

The EMA critic is a regularizer, not the bootstrap target network. The online critic supplies bootstraps and advantages. Preserve the reference implementation rather than substituting a conventional target-critic algorithm.

The shared agent config’s actor objective remains sign-only; the learned exploration presets override it. Resolve the selected preset before judging actual behavior. Some historical config comments incorrectly call the LCG preset sign-only; its actual current override is advantage-weighted.

## Exploration methods and working settings

| Output label | Preset | Current distinguishing settings |
| --- | --- | --- |
| `uniform` | `walker_uniform_100k` | Uniform actions throughout; denoiser only; no intrinsic reward or PMPO training |
| `lcg` | `walker_lcg_100k` | R1 selector, candidate MC 5, chunk 32, historical reference 320 with MC 3, damping 1e-4, beta 1; reference reward scaling (32 prompts; see `docs/REWARD_SCALING.md`); 200 PMPO updates per epoch |
| `full-ensemble` | `walker_ensemble_100k` | 5 independently initialized denoisers; episode bootstrap enabled; shared diffusion noise; pixel disagreement |
| `plan2explore` | `walker_plan2explore_100k` | One denoiser; 5 independent lightweight MLP predictors; replay-feature cache; bootstrap enabled; feature disagreement |
| `laplace-ensemble` | `walker_laplace_100k` | One denoiser; 5 sampled R1 parameter sets; shared LCG historical precision; temperature 1; pixel disagreement |

- Base `intrinsic_reward/lcg.yaml` differs from the Walker preset: broad selector, candidate MC 12, chunk 16 and RMS scaling. Do not confuse base-group defaults with resolved Walker settings.
- R1 is `unet.u_blocks.3.resblocks.0.*`, guarded as 10 tensors / 217,664 scalars for the current architecture.
- Full ensemble and Laplace compare completed sampler predictions on the uint8 grid represented as float tensors. They use population pixel variance, shared noise and one uniformly selected member per trajectory/transition for the next observation. Do not advance the world with the ensemble mean.
- Full ensemble trains the full denoiser schedule per member. Bootstrap multiplicities are fixed per episode/member. Equal PMPO update time does not imply equal total compute.
- Plan2Explore uses independent MLPs sharing frozen denoiser features, not multiple denoisers. Current layers are `unet.d_blocks.2` and `.3`, pooled to 4x4, yielding 2,048 features. Pixel sigma 0.1 plus offset noise 0.3 gives effective conditioning sigma about 0.316. Replay-only standardization, per-refresh episode bootstrap and fresh predictor fitting must remain reward-isolated. Defaults: hidden sizes `[256,256]`, Adam lr 1e-3, batch 256, 500 updates per predictor per refresh. The single denoiser generates imagined frames.
- Laplace samples `theta_hat + sqrt(T) * epsilon / sqrt(h)`. Replay-size scaling and damping are already included in the shared precision; do not apply them twice. Samples stay fixed during the epoch’s PMPO phase and each denoising chain. Preserve the stored settings label `lcg.lcg_historical_precision`; it is an identity string, not an import path.
- Temperature 1 is provisional. Extremely undertrained smoke models collapsed, while the tested first-refresh model under the actual pilot schedule retained coherent predictions. Neither observation establishes calibrated uncertainty or exploration benefit.
- Full ensemble, Plan2Explore and Laplace currently divide raw reward by 1.0; these are provisional scales. Raw intrinsic rewards are not comparable across methods.
- Current presets leave world-model compilation enabled for LCG/uniform and disable it for full ensemble, Plan2Explore and Laplace. Record the resolved setting in timing comparisons; do not silently align it during a refactor.

## Downstream contract

One `DownstreamTrainer` handles exactly one milestone, task, adaptation seed and selected denoiser member.

1. Validate and load the exact milestone/replay; freeze the selected denoiser.
2. Train a fresh reward predictor on the requested task’s recorded reward column.
3. Freeze the reward predictor and train a fresh sign-only PMPO policy/critic in imagination.
4. Evaluate the resulting policy on the requested real task without collecting additional task-training data.

- Never restore the exploration policy or its optimizer into a fresh adaptation.
- The action-spec lookup creates one dm_control environment, reads its spec and closes it; it is never reset or stepped. Real interaction is reserved for evaluation.
- Preserve task-reward alignment, padding masks, episode final observations and recorded successors at milestone cuts.
- Folder mode processes milestones sequentially in independent child processes. Each command selects ONE task and ONE adaptation seed; do not add automatic all-task training.
- Downstream resume is at completed updates, with optimizer/scheduler/counters/RNG and source replay identity checked.
- Evaluate-only uses completed output files and must work after the source milestone/replay are moved away.

## Environment and provisional schedules

Current Walker configuration: RGB 64x64, camera 0, action repeat 2, four conditioning frames, `time_limit: null` (the existing environment default). Do not claim compatibility for additional task groups without the compatibility evidence.

The current 100k presets use 5,000 initial steps and 190 collection epochs of 500 steps, followed by 50 training-only epochs. Denoiser updates are 10,000 in epoch 1 and 400 thereafter. PMPO updates are 200 in every epoch for LCG (48,000 in total) and 5,000 then 400 for the other learned methods (100,600). This gives 240 epochs and 105,600 updates per denoiser. Apart from the selections listed under "Project and authority", these are current defaults, not a finalized experimental schedule. Milestones are published every 10,000 replay steps (`config/exploration.yaml`, `milestones.every_steps: 10000`, `at_steps: []`, `keep_last: null`): ten per 100k run, at 10k, 20k, …, 100k, all on epoch boundaries of this schedule, and all retained.

Downstream currently defaults to 1,000 reward-model + 1,000 policy updates, batches of 32 and reward-model sequence length 19. Its evaluation defaults are deterministic actions, seeds 3001–3030 and at most 500 environment steps per episode. Budgets and the final seed protocol remain open decisions.

Return convention: sum the requested task’s reward over environment steps; each step already sums action-repeat substeps. Do not divide by action repeat or silently normalize returns.

## Output formats and identity

```text
outputs/<method>/<domain>/seed_<exploration_seed>/
  config/exploration.yaml
  dataset/
  checkpoints/
  milestones/steps_<N>/{manifest.json,dataset.pt,denoiser.pt,...}
  downstream/steps_<N>/<task>/seed_<adaptation_seed>/
    config.yaml
    summary.json
    metrics.jsonl
    rew_end_model.pt
    actor_critic.pt
    evaluations/eval_<task>_policy-<hash>_protocol-<hash>.json
  result/<task>/seed_<adaptation_seed>/result.csv
outputs/comparisons/<domain>/exploration_seed_<N>/<task>/adaptation_seed_<N>/
  comparison.csv
  analysis.json
  task_return.{png,pdf}
  fraction_over_thresholds.{png,pdf}
```

- Default exploration paths come from the config’s Hydra resolver. A fixed exploration seed is required. New runs refuse existing run outputs.
- Selected nonzero denoiser members add the established member suffix to downstream/result paths. Full-ensemble milestones retain all member files. `dataset.pt` references immutable shared replay episode versions; do not duplicate the full replay per milestone.
- Configuration is YAML; generated metadata is JSON only; append-only training metrics are JSONL. Historical generated YAML metadata is deliberately unsupported and must not be automatically converted or deleted.
- Custom output-root support is pending. Existing Hydra overrides do not constitute an implemented generalized custom-root feature.
- Never choose a checkpoint, denoiser member or evaluation report by its best task score.

## Logging and analysis contracts

- Use the shared tqdm helper: elapsed/remaining time, a space before the unit, concise units and automatic rate inversion. Keep the accepted phase labels and banner layout; no blank lines between phases within an epoch, one between epochs.
- Terminal losses, PMPO diagnostics, separate timing summaries and memory are hidden. Exploration displays intrinsic training statistics and per-seed intrinsic evaluation returns. Downstream displays only the evaluation aggregate summary and saved paths, not per-seed returns.
- Do not log downstream task returns during exploration. Keep task reward columns in replay.
- Exploration diagnostics are controlled by `evaluation.should`, `evaluation.every` and `evaluation.collect_test`. Fresh/reused metadata stays in artifacts/summary, not terminal noise.
- Four-seed intrinsic evaluation is supported, but current defaults do NOT pin four reset seeds: periodic collection defaults to 4 episodes, final collection to 100, and both reset-seed lists are null. Set the protocol explicitly when required. Extra test interactions are counted separately, including final collection.
- Preserve periodic diagnostics and one final diagnostic after exploration training. Evaluation must restore RNG and each submodule’s own mode and must not update frozen reward scaling.
- W&B debug metrics are off by default. Current Walker schemas: exploration actor_critic 43 keys (148 with debug), downstream actor_critic 39 (132 with debug), downstream reward_model 4 (8 with debug), plus configured evaluation returns. Different action dimensions change per-dimension counts.
- Training curves use component update counters. Downstream evaluation’s x-axis is the milestone’s exploration training steps. Preserve existing W&B identities on resume/evaluate-only.
- Base W&B mode is disabled, but the Walker experiment presets enable online logging. Explicitly select disabled/offline for authorized smokes when no sync is requested.
- `python src/downstream_analysis.py downstream.mode=collect` and `compare` perform CPU-only analysis. They do not train, collect data or evaluate policies. Each entry point rejects the other's modes. The analysis module must not import `downstream_trainer`, `exploration_trainer` or `envs`: with `MUJOCO_GL=egl`, importing dm_control opens the GPU devices. Definitions it shares with training live in `data/milestone.py`.
- Recompute result aggregates from episode returns. Threshold fractions use strict `return > 200/400/500/600`. Preserve individual evaluation returns and source paths.
- Reject incompatible effective training/environment/return settings and ambiguous reports. Old result fingerprints require recollection. Do not silently combine protocols or select among multiple policies under one protocol.
- Keep known missing milestones as gaps in both return and threshold plots. Store selection, hashes, protocol and missing results in `analysis.json`.
- Current comparison selects one exploration seed and one adaptation seed. Thirty evaluation seeds are episodes of one trained policy, not 30 independent training runs. Multi-training-seed aggregation and fixed analysis-candidate tooling are not implemented by the current comparison command.

## Resume and evidence

- Use `scripts/resume.sh <run_dir>` with the run’s own copied entry point, configuration and source. New runs use `src/exploration_main.py` and `config/exploration.yaml`; historical copied runs may use `main.py` and `trainer.yaml`.
- The canonical saved config is `config/exploration.yaml`, not a later relaunch’s `.hydra/` files. Hydra may rewrite its own files before rejecting an occupied directory; do not mistake those for the original run configuration.
- Preserve atomic saves, immutable milestones, checksums, task-label order, replay identity, source provenance and optimizer/RNG restoration.
- Preserve repaired-tail handling and committed-row boundaries for logging recovery. Do not duplicate W&B rows or silently hide replay mismatches.
- A finished run’s `.run_is_over` sentinel is not an invitation to delete it or force continuation. Deliberate extension requires user authorization and the documented workflow: `scripts/resume.sh` with a higher `collection.train.num_steps_total` (`ExplorationTrainer.extend_collection_budget`; whole collection epochs only, recorded as `budget_extensions`).
- Do not promise arbitrary independent GPU runs are bit-identical. Small rendering differences have been observed. Describe observed differences precisely; do not attribute a cause without evidence.

## Local environment, remote work and tests

- Local development: Windows, Conda environment `lcg`; use the project environment rather than base Python. A known local interpreter is `C:/Users/jerry/miniconda3/envs/lcg/python.exe`; verify it exists.
- Remote: `kingplus0`; a known interpreter is `/home/bigking/anaconda3/envs/lcg/bin/python`. Verify current paths and dependencies. Keep authorized server work under `~/scfeng/`.
- GPU 1 is the usual authorized device, only when idle. Check current occupancy before each authorized launch and leave other users’ processes untouched.
- On Linux, bind both compute and rendering: the project’s physical-device setting/visibility plus `MUJOCO_GL=egl` and `MUJOCO_EGL_DEVICE_ID` for the intended GPU. Verify actual placement; process `cuda:0` does not imply physical GPU 0.
- Standard tests: `python -m pytest tests -q -p no:cacheprovider`. For a small change, run relevant existing tests first, then required broader checks.
- Rendering integration tests need a working backend/display. Report genuine rendering failures separately; do not discard failed logs or call an unrun suite passed.
- At the reviewed cleanup snapshot, supplied logs report 654 full-suite and 441 affected tests passing. Supplied two-before/one-after CPU captures are identical for all five methods with `PYTHONHASHSEED=0`. These are snapshot evidence, not a permanent guarantee for later changes.
- Before/after refactors should compare logged values and scientific state, not only file existence. Exclude only documented volatile timing/source-identity fields. Pre-existing pickle ordering can affect raw replay-index checksums across processes; preserve content checks and document any fixed hash seed used.

## Documentation and review

Read the existing documents relevant to the task: `docs/DM_CONTROL_TASK_COMPATIBILITY.md`, `docs/LAPLACE_ENSEMBLE_DESIGN.md`, `docs/LAPLACE_PRECISION_AUDIT.md`, `docs/PLAN2EXPLORE_DESIGN.md`, `docs/REWARD_SCALING.md`, `docs/REWARD_FREE_EXPERIMENT_NOTES.md` and `docs/REWARD_FREE_EXPERIMENT_PROGRESS.md`. They are tracked in Git; `docs/changes/` (review packages) is gitignored. Update existing files rather than adding parallel trackers.

Review packages should contain the requested complete current source/config/scripts/tests, a diff against the stated starting commit, exact commands/exit codes, evidence and SHA-256 manifest. Exclude credentials, caches, large weights and replay unless specifically requested. Codex performs independent review; do not describe an implementation as approved before that review.

When this guide conflicts with code or a newer user decision, identify the conflict and correct the relevant documentation within scope. Do not launch additional experiments merely to resolve a stale comment.
