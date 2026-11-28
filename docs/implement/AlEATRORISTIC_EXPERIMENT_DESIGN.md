# Cartpole noisy TV test

Status: Implemented 2026-11-28 (uncommitted, awaiting review); no 100k comparison launched. Cartpole only; Walker is deferred.
Selected settings and deviations from the draft: "Selected settings (2026-11-28)" at the end. As-built notes and
evidence: `REWARD_FREE_EXPERIMENT_NOTES.md`, "Cartpole noisy TV (2026-11-28)".

Objective: Test whether a controllable source of unpredictable visual observations attracts the exploration policy and reduces exploration of Cartpole’s physical dynamics.

Compare three conditions:

| Condition | Observation |
| --- | --- |
| Clean | Original background |
| TV16 | A 16 × 16 random RGB panel while the trigger is active |
| TV24 | A 24 × 24 random RGB panel under the same trigger |

Initial methods: Uniform, Plan2Explore and predictor-LCG.

A static-TV condition is not included in this first experiment.

## Control

Use the original Cartpole environment without changing its rendered observations.

Keep the following identical across conditions:

- Physics, action space and action repeat.
- Reset distribution and episode length.
- Task-reward computation and recording.
- Observation resolution and preprocessing.
- Exploration budget, update schedule and milestone schedule.

Within each method, keep the controller, intrinsic-reward configuration and reward scaling identical across Clean, TV16 and TV24.

The clean condition must record whether the cart occupies the designated trigger region, even though no TV appears. This provides the baseline region occupancy.

Diagnostic simulator state and trigger metadata must not become additional policy inputs.

Save separate output directories, resolved configurations and source identities for every run. Reuse existing clean runs only when their relevant configurations, code and budgets match.

## Noisy TV

**Activation rule**

Let x_t be the cart’s physical horizontal position corresponding to observation o_t.

    I_t = 1 if x_L <= x_t <= x_R, otherwise 0.

Provisional interval:

    [x_L, x_R] = [0.3, 0.6] metres.

Before training, inspect existing clean calibration trajectories to verify that the region is repeatedly reachable and escapable.

Any threshold adjustment must occur before examining the experiment’s evaluation results. Freeze the same interval across all methods and TV sizes.

The TV remains active only while the cart is inside the interval. Leaving the interval immediately restores the original background; activation is not permanent.

**Appearance**

- Retain the existing 64 × 64 RGB observation resolution.
- Place the panel at a fixed background location, provisionally the upper-right corner.
- Anchor both panel sizes to the same corner with the same margin.
- Preserve the visibility of the cart, pole and track.
- Verify placement with representative trajectories before freezing it.
- Outside the trigger region, return the original image without a panel.

While active, sample every panel RGB value independently and uniformly from the integers 0 through 255.

Sample a fresh independent texture for each successive agent-observed step.

The agent controls whether it enters the region, but cannot predict the next texture.

**Observation timing**

Generate one texture per agent-observed environment step, after action repeat.

For observation o_t, determine activation using x_t. After action a_t, determine activation of o_(t+1) using x_(t+1). Apply the same rule to reset observations when applicable.

Reuse the resulting observation for:

- Policy input.
- Replay storage.
- Observation history.
- Denoiser training.
- Feature extraction.
- Evaluation and video recording.

Repeated reads or rendering of the same observation must not generate another TV texture. Reading replay must not resample TV content.

The fixed denoiser feature-probe noise remains a separate operation and is unchanged.

**Randomness**

Use a dedicated TV random-number generator, independent of physics, policy, training and feature-probe RNG streams.

Use separate training and evaluation TV seeds.

Save sufficient TV RNG and cached-observation state to support the existing checkpoint/resume guarantees.

## Evaluation

**Training comparison**

Train each method separately in Clean, TV16 and TV24.

Use the existing 100k reward-free exploration schedule and matching training seeds. The initial pilot uses one training seed; independent training repetitions are needed for reproducibility claims.

Keep each method’s configuration fixed across conditions. Do not tune hyperparameters separately for each TV size.

**Thirty-seed evaluation**

Evaluate the trained exploration policies in their corresponding real environments:

- Clean-trained policy in Clean.
- TV16-trained policy in TV16.
- TV24-trained policy in TV24.

The TV remains active during noisy-condition evaluation.

Use the same:

- 30 reset seeds.
- Normal exploration reset convention.
- Policy action-selection mode.
- 500-step episode length.

Use evaluation TV seeds separate from training. Pair TV16 and TV24 evaluation noise streams where applicable.

**Attraction metrics**

Record the following per episode:

| Metric | Definition |
| --- | --- |
| Region occupancy — primary | Fraction of observations with I_t = 1 |
| Region-entry count | Number of outside-to-inside transitions |
| Mean dwell duration | Mean consecutive duration inside the region |
| Maximum dwell duration | Longest consecutive duration inside the region |
| Reach indicator | Whether the episode enters the region |

Report reach rate across episodes.

Record episodes beginning inside the interval separately from outside-to-inside entries. Report dwell durations conditional on visits and identify episodes with no visits.

For each paired reset seed s:

    delta_occupancy_s = occupancy_TV_s - occupancy_clean_s

**Physical exploration**

Use simulator state for analysis only.

Measure coverage using shared, predefined bins of cart position and pole angle. Fix bin boundaries using clean calibration data and handle angle periodicity and out-of-range states consistently.

Also report cart-velocity and pole-angular-velocity summaries.

Do not count unique rendered images as physical-state coverage: random TV textures would inflate that count.

**Supporting diagnostics**

Record:

- Intrinsic rewards inside and outside the region.
- Rewards on entry versus during sustained occupancy.
- Representative videos.
- Existing actor and critic health diagnostics.

Compare intrinsic-score changes within each method. Raw intrinsic-reward magnitudes are not directly comparable across methods.

Report per-episode distributions, mean ± std and paired occupancy differences. Episode-level confidence intervals describe evaluation variability for a trained policy, not variability across independent training runs.

**Interpretation**

High occupancy alone does not establish distraction.

Evidence of a stochastic trap would be a persistent increase in occupancy or dwell time relative to Clean, accompanied by reduced physical-state coverage.

TV16 versus TV24 tests sensitivity to disturbance size. Ignoring a small panel alone does not establish aleatoric-uncertainty discrimination; the representation may simply be insensitive to those pixels.

Clean versus noisy TV measures the overall effect of adding the stochastic distractor. It does not separately isolate unpredictability from the panel’s appearance. A static-TV control can be added later if needed.

Downstream adaptation is outside this initial test. Specify its observation conditions and evaluation protocol separately before adding it.

# Implementation

## Environment wrapper

Add a dedicated observation wrapper:

    src/envs/cartpole_noisy_tv.py

Reuse the existing Cartpole physics implementation. Add a small hook in the existing environment factory to select Clean, TV16 or TV24 through configuration.

The wrapper must:

- Read cart position to determine trigger-region membership.
- Modify only the rendered RGB observation.
- Preserve physics, actions, rewards, termination, observation shape and dtype.
- Support region logging in Clean mode without changing the image.
- Use a dedicated TV RNG and support checkpoint restoration.
- Reject enabled noisy-TV settings for non-Cartpole domains.

Proposed configuration, with nesting adapted to the existing repository:

```yaml
noisy_tv:
  enabled: false
  size: 16
  position_interval: [0.3, 0.6]
  panel_location: top_right
  seed: 0
```

| Condition | enabled | size |
| --- | --- | --- |
| Clean | false | Not used |
| TV16 | true | 16 |
| TV24 | true | 24 |

Apply the panel to the final 64 × 64 RGB observation before frame stacking, normalization and policy/replay preprocessing.

Generate each observation once. Repeated reads, video rendering and feature extraction must reuse that observation without generating another TV texture.

Record diagnostic metadata separately from policy inputs:

- Cart position and velocity.
- Pole angle and angular velocity.
- Trigger-region membership.
- Actual TV activation.
- TV panel size.

Distinguish region membership from TV activation: the cart can occupy the region in Clean while the TV remains disabled.

## Exploration training integration

Reuse `exploration_trainer.py`. Do not create a separate trainer or duplicate its training loop.

| Component | Required behavior |
| --- | --- |
| Real collection environment | Configured Clean, TV16 or TV24 environment |
| Real evaluation environment | Corresponding condition |
| Replay | Store the actual emitted observations |
| Denoiser training | Existing procedure using that replay |
| Intrinsic scorer | Existing method-specific refresh and scoring |
| Exploration-policy training | Existing controller training in imagination |
| Imagined environment | Existing learned world-model environment |

The training flow remains:

1. Collect observations from the configured real environment.
2. Train the denoiser.
3. Refresh the intrinsic scorer.
4. Train the exploration policy in imagination.
5. Evaluate in the corresponding real environment.
6. Save checkpoints and milestones.

Do not apply the noisy-TV wrapper to the imagined environment. The denoiser learns the modified observation dynamics from real replay and generates its own imagined observations.

Keep each method’s training configuration unchanged across conditions. Save the resolved TV configuration, seeds and required restoration state with the run.

## Evaluation and analysis integration

Use `exploration_analysis.py` as the analysis entry point. Add a `noisy-tv` analysis option alongside the existing exploration analyses.

Separate responsibilities as follows:

| Component | Responsibility |
| --- | --- |
| `cartpole_noisy_tv.py` | Produce TV observations and diagnostic metadata |
| Existing real-environment evaluator | Collect the 30-seed policy rollouts |
| `exploration_analysis.py` | Compute metrics, compare conditions and export results |
| `AlEATRORISTIC_EXPERIMENT_DESIGN.md` | Specify the protocol and metric definitions |

Prefer reading previously saved evaluation trajectories.

If the required trajectories are missing, the analysis entry point may call a helper that:

1. Restores the selected milestone’s exploration policy.
2. Constructs the corresponding real environment.
3. Evaluates the policy over the specified 30 reset seeds.
4. Saves trajectories, diagnostic state and evaluation settings.

For Uniform, use the existing uniform action-selection procedure rather than requiring a learned policy checkpoint.

Evaluation must not resume training, update weights or refit intrinsic scorers. Optional intrinsic-reward diagnostics must restore the saved scorer state.

Keep the noisy TV active during TV16 and TV24 evaluation. Use separate evaluation TV seeds from training, and preserve the same reset seeds, action-selection mode and episode length across comparisons.

## Analysis outputs

Compute per-episode:

- Trigger-region occupancy.
- Region-entry count.
- Mean and maximum dwell duration.
- Reach indicator and initial region membership.
- Physical-state coverage.
- Cart-velocity and pole-angular-velocity summaries.

Pair results by reset seed and compare TV16 and TV24 against Clean.

Use identical physical-state coverage bins across methods and conditions. Record the bin definitions and handle angle periodicity and out-of-range states consistently.

Export:

- Per-episode CSV and JSON results.
- Aggregate summaries and paired occupancy differences.
- Occupancy, dwell-time and physical-coverage plots.
- Representative videos.
- Evaluation settings and source/checkpoint provenance.

Episode-level uncertainty intervals must not be presented as uncertainty across independent training runs.

## Required validation

Verify that:

- Clean mode reproduces the original observations.
- Identical physics seeds and fixed actions produce identical physical trajectories, rewards and episode boundaries across conditions.
- Pixels outside the active panel remain unchanged.
- Inactive observations match the original rendering.
- Repeated observation reads do not consume TV RNG draws.
- TV RNG activity does not affect other RNG streams.
- Reset, action repeat, final observations, truncations and collection cuts are handled consistently.
- Checkpoint restoration preserves the required random state and cached observation.
- Policy inputs, replay frames and recorded videos use the same emitted observations.
- Restored evaluation reconstructs the correct TV condition.
- Analysis computes the defined metrics correctly on simple known trajectories.
- Neither evaluation nor analysis changes model weights.
- The imagined environment receives no additional TV-noise injection.

Implement and validate these changes before launching the comparison. Preserve existing experiments, other methods and unrelated running jobs.
## Selected settings (2026-11-28)

Frozen before any noisy-TV training, from the existing clean Cartpole runs' recorded training replays (calibration
evidence: `REWARD_FREE_EXPERIMENT_NOTES.md`, "Cartpole noisy TV (2026-11-28)"). No evaluation result was examined.

| Setting | Selected | Draft |
| --- | --- | --- |
| Trigger interval | `[0.3, 0.6]` m, inclusive, on the slider position `qpos[0]` | `[0.3, 0.6]` (kept) |
| Panel location | `top_left` | `top_right` |
| Margin | 0 px (both sizes) | unspecified |
| Panel sizes | 16 (TV16), 24 (TV24) | as drafted |
| TV seeds | training 0, evaluation 1000 | `seed: 0` |
| Texture | `size x size x 3` uint8 values, uniform 0..255, one fresh texture per emitted observation while active | as drafted |

Configuration: the Hydra group `config/noisy_tv/{clean,tv16,tv24}.yaml` (`+noisy_tv=<condition>`, combined with any
method preset). It sets `env.train.noisy_tv`, `env.test.noisy_tv` (same condition, seed 1000) and
`collection.{train,test}.record_physical_state: true`. Without the group, no wrapper is built (all other runs are
unchanged).

Deviations from the draft, and why:

- **Panel location `top_left` instead of `top_right`.** Camera 0 is fixed and +x is image right: while the cart is
  in `[0.3, 0.6]` m the cart and pole occupy image columns 18–63 for some pole angles, so the top-right corner
  covers the pole. On in-region calibration frames (1,500 per run, ten runs) the top-right panel covered cart/pole
  pixels in 0–5% (TV16) of frames, and over all in-region states 21–27% (TV24). The top-left panel covers nothing
  for TV16 (0 of all checked frames and states) and at most 6 pole pixels in 0–4.5% of in-region calibration frames
  for TV24 (2.6% of a dense state grid). No 24-pixel square fits anywhere in the 64 x 64 image without some
  overlap for some pole angle: the pole spans rows 12–51 across the full width.
- **Explicit TV seeding rule.** A reset with an explicit seed `s` restarts the TV stream from
  `SeedSequence([tv_seed, s])`; an autoreset (no seed) continues it. Evaluation episodes of one reset seed therefore
  get the same TV draws in every evaluation of a condition, and TV16 and TV24 share their stream (paired).
- **Diagnostics storage.** The per-frame diagnostics extend the existing analysis-only `Episode.physical_state`
  (`tv_in_region`, `tv_active`, `tv_panel_size`, and their `final_*` values) beside `qpos`/`qvel`; the panel size
  and interval are also in the run config and milestone manifest.
- **Evaluation helper.** The analysis evaluates missing trajectories with the existing exploration test collector
  (sampled policy actions, as the run's own test diagnostics; uniform actions for Uniform), not the downstream
  evaluator (deterministic actions).
- **Downstream.** Adapting a milestone explored with an enabled TV is refused: downstream evaluation builds a plain
  environment and would silently drop the TV. Clean milestones adapt as usual.
