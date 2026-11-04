# Laplace-sampled diffusion ensemble disagreement

Date: 2026-10-26\
Status: agreed initial design; implemented 2026-10-26/27 and committed in `9176c13` (modules later moved under `src/intrinsic_reward/`, `88bc7a2`; file paths below are as of the design). Tests pass. In the bounded local GPU smoke the training state (models, optimizers, sample bank, replay, collector/env incl. task RNG) matched across the uninterrupted and the resumed runs; the resumed run's final test episode diverged from step 260 (cause not isolated), so its two exact comparisons failed and the smoke driver's overall exit status was 1. Not experimentally validated. A frozen-model diagnostic (2026-10-27) found that at T = 1 every sampled member predicts a blank grey frame, a sampling-scale concern reported without tuning. Implementation notes: `docs/REWARD_FREE_EXPERIMENT_NOTES.md`, Step 6, "Laplace-sampled diffusion ensemble"; precision audit: `docs/LAPLACE_PRECISION_AUDIT.md`.\
Intended repository path: `docs/LAPLACE_ENSEMBLE_DESIGN.md`.

## 1. Goal and agreed choices

Add a fourth learned exploration signal beside LCG, the full diffusion ensemble, and the Plan2Explore-style lightweight ensemble. Use the existing `ExplorationTrainer`, DIAMOND denoiser, imagination, and PMPO controller.

Train **one denoiser**. After its training phase, reuse **the existing LCG historical-precision calculation** to construct a Gaussian distribution over its selected parameters. Draw **five parameter samples**, keep every unselected parameter fixed, and measure **pixel variance between the five completed next-frame predictions**.

Keep member count configurable, with `num_members: 5` as the default. These are five sampled parameter sets, not five independently trained models. Do not include the unchanged central model as one of the five samples. No bootstrap or additional denoiser optimizers are needed.

Share computation where dependencies permit it. Shared parameters reduce storage and training cost; they do not automatically allow all activations to be shared throughout a diffusion sampling chain.

The method is a **Laplace-inspired subnetwork ensemble**. It is not an exact reproduction of either reference below, an independently trained deep ensemble, or an exact information-gain reward.

## 2. Reference 1 — Plou et al. (2024 preprint)

**Paper:** Carlos Plou, Ana C. Murillo, and Ruben Martinez-Cantin, *Active Exploration in Bayesian Model-based Reinforcement Learning for Robot Manipulation*. [Paper, Sections III–V](https://arxiv.org/html/2404.01867v1).

**What the work does:** learns a probabilistic dynamics network and uses model uncertainty to guide task-independent exploration. Laplace inference approximates parameter uncertainty around a trained model. Its subnetwork Monte Carlo variant samples selected weights and fixes the remaining weights. Collected data subsequently supports task-specific policy learning.

**What we take:** the exploration structure, one trained dynamics model, Gaussian uncertainty over a parameter subset, and sampled model predictions as the basis of an intrinsic utility.

**What we change:** DIAMOND predicts images instead of their low-dimensional dynamics outputs. We reuse LCG's parameter selection and historical curvature rather than their weight-selection procedure. We use mean pixel variance rather than their entropy/Jensen–Rényi utilities, and retain our PMPO controller. This paper supports the overall exploration design, not our pixel metric or its calibration.

## 3. Reference 2 — Jazbec et al. (UAI 2025)

**Paper:** Metod Jazbec et al., *Generative Uncertainty in Diffusion Models*. [Published paper](https://proceedings.mlr.press/v286/jazbec25a.html); [method, Sections 3.1–3.3](https://arxiv.org/html/2502.20946v2).

**What the work does:** applies post-hoc Laplace inference to diffusion-model weights. With sampling randomness fixed, sampled models generate complete images. Variation in frozen semantic embeddings is converted to a Gaussian entropy score for detecting poor-quality generations.

**What we take:** compare completed diffusion outputs under shared sampling randomness; hold each sampled parameter set fixed for the entire denoising chain; use only a parameter subset to make inference tractable.

**What we change:** use LCG's selected block and precision instead of their last-layer construction. Compare next-frame pixels directly, without CLIP or embedding entropy. Use disagreement for exploration rather than image filtering. Our rollout-member choice matches our full-ensemble baseline. Their results do not establish exploration performance. They also discuss why applying Laplace directly to a practical diffusion loss is not automatically a fully justified Bayesian posterior.

## 4. Our decisions versus the references

| Decision | Reason in this project |
| --- | --- |
| One trained denoiser and Gaussian parameter samples | Avoid training five complete models; relates to both references |
| Same selected weights and historical precision as LCG | Isolate a change from analytic LCG scoring to sampled output disagreement |
| Five sampled members | User-selected initial count; matches the full-ensemble default |
| Pixel population variance | Match the existing full-ensemble scoring function |
| Same sampling randomness across members | Separate parameter-induced disagreement from independent diffusion noise |
| Fixed sample bank within an epoch | Keep the reward function fixed during that PMPO phase |
| Uniform member per trajectory and transition | Match the existing full-ensemble imagination behavior |
| No CLIP, entropy, or bootstrap | Keep the first implementation small and its comparison interpretable |

The LCG parameter subset, precision convention, sample-bank lifetime, population variance, and rollout selection are our design decisions. Neither paper validates this exact combination.

## 5. Gaussian construction: reuse LCG literally and document its scale

Let the current trained denoiser parameters be `theta_hat`. Let `S` be the selected parameter subset. Start with the existing LCG R1 configuration:

```text
unet.u_blocks.3.resblocks.0.*
```

For the current Walker architecture, the recorded expectation is 10 tensors and 217,664 scalars. Reuse the existing selector and its shape/count checks rather than duplicating a list of tensors.

Let `P_LCG` be the positive precision operator returned by the existing historical-precision construction for this subset, **including its existing damping exactly once**. Define the initial sampling distribution explicitly:

```text
q(theta_S) = Normal(theta_hat_S, T * inverse(P_LCG))
T = 1.0 initially
theta_not_S = theta_hat_not_S
```

`T` is a configurable covariance multiplier called `posterior_temperature`. It multiplies covariance, so standard deviations scale with `sqrt(T)`. It is unrelated to intrinsic-reward scaling.

For diagonal precision `p`, sampling is:

```text
epsilon_k ~ Normal(0, I)
theta_S[k] = theta_hat_S + sqrt(T) * epsilon_k / sqrt(p)
```

For a non-diagonal precision, use the matching factorization or inverse-square-root operation. A diagonal approximation is a different method and must not be silently substituted. Never multiply noise by the inverse precision itself when the required quantity is its inverse square root.

### Required implementation audit

Before wiring sampling, inspect the actual LCG implementation and put the exact formula and code locations in the report:

- What is accumulated: squared gradients, a Fisher/GGN approximation, or another curvature statistic?
- Is the reference bank summed or averaged? How are historical Monte Carlo probes combined?
- Are replay size, loss reduction over pixels/batches, likelihood scale, or other multipliers already applied?
- Is damping already present? Does it act as numerical stabilization, a stated prior, or both?
- What is the representation: diagonal, block, low-rank, or dense?

Reuse those conventions unchanged for the first version. Reuse the LCG preset's historical settings, initially bank size 320, historical MC 3, and damping `1e-4`, subject to verifying their current configuration names and meaning. **320 is the reference-bank size, not automatically a posterior data count.** LCG's candidate-scoring MC count and reward divisor do not enter the Gaussian construction.

Do not insert an extra replay-size factor, prior precision, damping term, or likelihood multiplier merely because a generic Laplace formula contains one. If LCG's scale does not correspond to a conventional likelihood curvature, record the literal Gaussian above as a curvature-based approximation. The current trained weights need not be an exact posterior mode. Do not claim calibrated Bayesian uncertainty without further work.

Use `T=1` for the bounded correctness smoke. If it produces excessive perturbations or degenerate images, report that result; do not silently shrink the samples, clamp their norm, tune temperature, or change damping to make the smoke appear successful. Invalid/nonpositive/nonfinite precision must fail clearly, without a silent fallback to an unrelated covariance.

## 6. Refresh and parameter storage

Keep the existing epoch ordering: collect real data, train the denoiser, refresh the intrinsic scorer, train PMPO in imagination, log, and save.

At refresh:

1. Freeze the current denoiser for imagination and establish the selected-parameter ordering.
2. Run the shared LCG historical-precision code on reward-free training replay.
3. Draw five independent selected-parameter perturbations using a private, checkpointed RNG stream.
4. Freeze the resulting bank for the whole PMPO phase. Refresh only after the next denoiser phase.

Store one base denoiser and only the selected parameter samples or perturbations. Do not allocate five full training models or optimizers. Prefer functional parameter substitution or another scoped mechanism that cannot leave the base model modified after an exception. Normal denoiser training, milestones, and downstream loading must always see the base trained weights.

Private sampling must not reseed global CUDA generators. Record enough state to reproduce the exact sample bank on resume; saving the actual selected perturbations avoids relying on a numerically different curvature rebuild.

## 7. Prediction and pixel reward

For each imagined environment transition:

1. All five members receive identical frame/action history and the candidate action.
2. Draw one initial diffusion-noise tensor and give the same tensor to every member.
3. Run the existing diffusion sampler to completion for each sampled parameter set. Each member keeps its weights unchanged for **all denoising steps**.
4. Apply the existing full-ensemble disagreement function to the completed float predictions.

Use the existing deterministic sampler convention, initially `s_churn=0`. Reject unsupported stochastic settings rather than comparing members with different random noise. If stochastic sampling is supported later, share its entire random-noise path, not just its initial noise.

For predictions `x[k,b,c,h,w]` in the existing model-space float range, the raw reward is:

```text
mean_image = mean(x, across members)
raw[b] = mean_pixels(mean_members((x - mean_image)^2))
```

This is **population variance**, dividing by `K`, matching the current full diffusion ensemble. Do not use the Plan2Explore heads' `K-1` variance convention. Score at the same stage as the existing full ensemble: the sampler's completed output. That output is a float tensor, but its values lie on the uint8 grid, because the final Euler step returns the last denoiser output, which `wrap_model_output` quantizes. (Corrected 2026-10-27: this sentence previously said "before uint8 quantization", which does not describe the existing stage.) Add no extra clipping or preprocessing that differs between methods.

Default `num_noise_samples: 1`. If greater than one, repeat the shared-noise comparison and average its scalar rewards; do not mix diffusion-noise variance with member variance. Reuse the existing full ensemble's rule that the first noise sample supplies the next observation.

To choose that observation, draw a member uniformly for each trajectory at each transition and use its completed prediction. Do not average images, choose the highest-scoring member, or add a sixth central-model rollout. This matches our full ensemble; it is not a claimed reproduction of Jazbec et al.'s output-selection protocol.

Reuse fixed reward scaling: `policy_reward = raw / fixed_divisor`, initially divisor `1.0`. This is a provisional setting, separate from posterior temperature. Do not copy LCG's divisor 100 just because its precision code is reused. Tiny rewards remain a calibration issue for later diagnostics.

## 8. Efficient computation without changing the sampler

The straightforward reference runs five complete predictions using the shared base model with different selected parameters. Implement or retain this reference for correctness checks before optimizing.

An operation's output can be shared only when its **parameters, inputs, conditioning, and randomness are identical**. Unchanged parameters alone are insufficient.

### First denoising evaluation

At the first evaluation, all members have the same noisy input. For a supported late-block subset such as R1, run the common U-Net prefix once and branch at the first operation affected by sampled parameters. Cache all required skip tensors and conditioning, not just one activation. Run each selected branch and its dependent suffix separately.

### Later denoising evaluations

After the first sampler update, member-specific noisy images generally differ. The unchanged U-Net prefix now receives different inputs, so its image-dependent activations generally must be recomputed per member. Do not reuse the first member's activations or the central model's latent trajectory for the other members.

Conditioning-only computations may still be shared when independent of the member's evolving image. Member-specific passes can also be batched where useful; batching is not equivalent to having identical activations.

### Implementation boundary

Provide a small exact reuse path for the initial supported subset if practical, and compare it against the full-forward reference under identical samples and noise. A configurable unsupported subset may use the correct full-forward path with an explicit log message. Unsupported graph dependencies must never silently reuse stale activations.

Do not change to a one-pass approximation, propagate only output variance, or resample weights at each denoising step to claim speed. These define different algorithms.

The reliable efficiency benefit is one denoiser training schedule and one full weight set plus five small parameter banks. Actual imagination speedup depends on valid reuse and hardware; measure it rather than promising near-single-model inference.

## 9. Configuration and comparison settings

Use a new selector and file:

```text
intrinsic_reward=laplace_disagreement
config/intrinsic_reward/laplace_disagreement.yaml
```

All method settings live there, following the accepted full-ensemble configuration layout. Reuse existing schemas and field names for historical precision, selected parameters, and reward scaling. The following is a logical field list, not a requirement to rename existing LCG fields:

| Setting | Initial value |
| --- | --- |
| `method` | `laplace_disagreement` |
| `num_members` | `5`, configurable, integer at least 2 |
| Selected parameters | Same R1 selector and validation as LCG |
| Historical precision | Same bank/probe/damping settings and implementation as LCG |
| `posterior_temperature` | `1.0`, finite and strictly positive |
| Sample-bank lifetime | One epoch's PMPO phase |
| `num_noise_samples` | `1` |
| Member selection | Uniform per trajectory and transition |
| Reward | Existing pixel population variance |
| Reward scaling | Fixed divisor `1.0`, provisional |
| World-model compilation | Off initially; reject unsupported compilation |

Add `walker_laplace_100k` as a configuration preset only. Match the full-ensemble preset's collection budget, denoiser schedule, milestones, test diagnostics, and weighted PMPO settings, including cap 5. Do not modify existing presets or global defaults. Report that the maintained LCG preset still uses sign-only; controlled LCG comparisons must explicitly align the controller settings used in a run.

No temperature sweep, CLIP alternative, new posterior-fitting library, full covariance expansion, or new trainer is needed for this first implementation.

## 10. Checkpoints, milestones, and isolation

- Save method settings, selected names/order/shapes, precision convention/state, actual sample-bank tensors, RNG state, refresh epoch, and relevant counters through the existing full-state checkpoint workflow.
- Validate method and posterior settings before loading weights. Reject a changed subset, temperature, member count, precision configuration, or scoring protocol on resume.
- Preserve existing single-model and other-baseline checkpoint behavior. Older checkpoints must not be silently interpreted as this method.
- Milestones remain **single-denoiser** milestones containing the unperturbed trained model and fixed replay. Downstream training stays unchanged; it does not load one of the sampled posterior models.
- Refresh, sampling, scoring, and any activation cache must not read task rewards or reward-bearing metadata. Changing recorded task rewards must leave the method unchanged under matched seeds.
- Do not use test data to fit precision, tune the Gaussian, or construct sample banks. Test diagnostics retain their separate data and RNG streams.
- Restore every submodule's original train/eval state and protect original weights, buffers, and training RNG even when scoring fails.

Keep posterior samples out of ordinary snapshot initialization unless the existing workflow can validate and restore complete method state. A clear unsupported-initialization error is preferable to partial loading.

## 11. Validation and logging

Add meaningful tests for:

1. Shared precision: the same denoiser/replay/seeds/configuration produce the same precision as the LCG path; damping and normalization are applied once.
2. Sampling: a small known precision yields the correct covariance factor; Gaussian draws modify only selected parameters. Cover diagonal or the actual supported precision representation, parameter ordering, and invalid settings.
3. Sampler semantics: each sampled model is fixed throughout its denoising chain, members share noise, and final-image variance matches the existing full-ensemble function.
4. Reuse: optimized and reference per-member predictions/rewards agree under a stated tight numerical tolerance; later image-dependent prefixes are recomputed after trajectories diverge. Test outputs, not just call counts.
5. Isolation: base weights and buffers remain unchanged during scoring, including exceptions; changing task rewards changes no exploration state; other methods remain unchanged.
6. Resume: exact sample-bank restoration and deterministic CPU continuation; integration through the supported checkpoint/resume workflow; configuration changes refused before model mutation.
7. Integration: denoiser and PMPO actually update, milestones load downstream as the base model, and collection uses the exploration policy after training.

Log precision quantiles, selected dimension, temperature, sampled perturbation norms and their ratio to selected-weight norms, raw/scaled rewards, finite-value checks, refresh/scoring/full-PMPO times, and peak memory. Distinguish shared-prefix savings from total update time. Record near-bound output/saturation diagnostics if useful; do not alter predictions to improve these statistics.

Run affected regressions and the full suite. Run a bounded local Walker GPU smoke through `src/main.py` and `scripts/resume.sh`, with offline W&B and fresh gitignored `outputs/` directories. Reuse the short ensemble smoke schedule where practical: five members, real denoiser updates, multiple refreshes, actual PMPO and policy collection, milestone loading, and uninterrupted versus hard-kill/resume comparisons. Measure the reference and optimized prediction path on matched inputs.

Report exact comparisons separately from numerical optimization tolerances. Do not weaken resume comparisons to accommodate a rendering difference. If processes diverge, record the first observed difference and distinguish evidence from a suspected cause. Keep the actual smoke-driver exit status visible.

This validates implementation and bounded runtime only. It does not establish calibrated uncertainty, better exploration, downstream benefit, or a final schedule.

## 12. Implementation scope and delivery

Read repository instructions and current progress/notes first. Work locally in `reward-free-experiment`; preserve unrelated changes. Reuse the current codebase. Do not access kingplus0, commit, push, run an online pilot, train downstream policies, tune hyperparameters, or launch 100k experiments.

Update progress and implementation notes only for verified work. Keep this document's scientific rationale separate from as-built details, and record any deviation explicitly.

Deliver `docs/changes/step6_laplace_ensemble_review.zip` with the complete diff, changed source/config/tests, this design, updated docs, the precision audit, a short report, resolved smoke configuration, commands, stdout/stderr, exit codes, and comparison/timing evidence. Report the absolute Windows path and remaining limitations.
