# DIAMOND PMPO Beta controller prototype

## Git isolation

- Base branch: `lcg-undersampling-diagnostic`
- Base commit: `1708a7cded943ca8f50b0664ad4e48da2c3c1e9b`
- New branch: `pmpo-beta-continuous`
- Worktree: `C:\Users\jerry\Project_LCG\LCG-pmpo-beta`
- Initial new-worktree `git status --short`: empty (clean).
- The base checkout had a modified `.gitignore` and untracked
  `sample_walker_walk_epoch26.mp4`; neither was copied or changed.
- No files were edited in the base or DrQ worktrees. Implementation is left as
  an uncommitted diff on the new branch for review.

## References inspected and attribution

Inspected 2026-09-23; the GitHub links identify the inspected main-branch files,
not immutable upstream release pins.

- [Official Dreamer 4 paper, section 3.3, equations 10–11](https://arxiv.org/html/2509.24527v1):
  sign-only advantage groups, separate group normalization, current-to-prior KL,
  mixing coefficient 0.5 and KL coefficient 0.3.
- [lucidrains continuous policy](https://github.com/lucidrains/dreamer4/blob/main/dreamer4/dreamer4.py)
  selects a unimodal Beta readout.
  [The actual readout implementation](https://github.com/lucidrains/discrete-continuous-embed-readout/blob/main/discrete_continuous_embed_readout/discrete_continuous_embed_readout.py)
  uses `softplus(raw) + 1` for both concentrations when unimodal.
- [HalfCheetah imagination example](https://github.com/lucidrains/dreamer4/blob/main/train_halfcheetah_imagination_rl.py):
  freezes a copied policy, refreshes it before an imagination-training block,
  generates experience without gradients, then evaluates likelihoods for learning.
- The inspected lucidrains PMPO code multiplies likelihoods by
  `abs(tanh(advantage))` and normalizes using the total sample count. This prototype
  deliberately follows the paper's separate sign-only groups instead.
- [Nicklas Hansen's DMControl Dreamer 4 implementation](https://github.com/nicklashansen/dreamer4/blob/main/README.md)
  describes a continuous-control world-model implementation, not a complete agent.
  No controller or world-model code was imported from it. The existing project's
  environment action-spec adapter remains the authority for bounds.

## Distribution and action bounds

For action dimension j:

```
alpha_j = softplus(raw_alpha_j) + concentration_min
beta_j  = softplus(raw_beta_j)  + concentration_min
x_j ~ Beta(alpha_j, beta_j)
a_canonical_j = 2*x_j - 1
a_env_j = low_j + (a_canonical_j + 1)*(high_j-low_j)/2
```

The default floor is 1, matching the inspected unimodal readout and avoiding
U-shaped densities at initialization. Initialization gives concentrations near
1.693. Small nonzero output weights preserve initial state dependence. Sampling
uses `Beta.sample()`; actions are detached. Deterministic action requests use the
Beta mean, not its mode. There is no tanh action transform or Q-maximization path.

`get_action_space_kwargs()` passes dimension and vector bounds from the real
environment into `AgentConfig`, then into PMPO. The affine implementation uses
the algebraically equivalent `low + x*(high-low)`. Finite, strictly ordered bounds
are required. Tests use asymmetric bounds `[-3, 0.5]` to `[2, 7]`; the actual
Walker smoke obtained its six-dimensional bounds from DMControl.

Log probability is `sum_j(log Beta(x_j) - log(high_j-low_j))`. Entropy adds the
same affine log width and sums dimensions. Both are reported in nats. Reconstructing
x from float32 environment actions can round to 0 or 1; its argument is clamped
to `[finfo.eps, 1-finfo.eps]` only for density evaluation. Log probabilities are
never clamped; non-finite values raise an error.

## Value target and PMPO equations

`Vbar` is the detached value prediction made during rollout, before either
optimizer step. It is a fixed target for that update, not an EMA/target-Q network.

```
R_t = r_t + gamma*(1-end_t)*[(1-lambda)*Vbar(s_next) + lambda*R_next]
R_after_last = Vbar(s_after_last)
A_t = R_t - Vbar(s_t)
D_pos = {t: A_t >= 0}; D_neg = {t: A_t < 0}

L_actor = -alpha_pmpo * mean_D_pos(log pi(a_t|s_t))
          +(1-alpha_pmpo) * mean_D_neg(log pi(a_t|s_t))
          +beta_kl * mean_all(KL(pi_current || pi_prior))
L_value = mean_all((V(s_t) - stop_gradient(R_t))**2)
```

A true termination removes the bootstrap. At an artificial truncation, the
return is `r + gamma*Vbar(final_observation)` and the trace stops before the reset
episode. An ordinary rollout cut bootstraps from the last successor value.
The existing `compute_lambda_returns(..., continuous_reward=True)` implements
these semantics without reward-sign clipping. Empty positive/negative groups
contribute zero; their surviving coefficient is not renormalized. Exact zero is
positive; there is no epsilon threshold or advantage-magnitude weighting.

KL uses PyTorch's analytic `kl_divergence(Beta, Beta)`, sums action dimensions,
then averages batch/time. An identical affine transform does not change KL.

## Architecture, rollout and optimizer ownership

The actor and scalar value head each own an independent instance of the base
`ActorCriticEncoder`, followed by an MLP. Actor Adam owns only actor encoder/head;
value Adam owns only value encoder/head. They have separate learning rates,
schedulers and clipping. A combined backward is valid because the two loss
graphs have disjoint parameters. The prior copies the entire actor, including
its encoder, and is frozen and excluded from both optimizers.

This is a **single-frame feedforward prototype**. It retains the base pixel
encoder blocks and normalized observation input, but omits the base LSTM. The
selected base has no DrQ frame-stack implementation to reuse. A dummy recurrent
state preserves the existing `env_loop` interface; it has no learned role and
cannot leak episode information. This temporal-information limitation must be
considered before Walker performance comparisons.

Training remains `main.py -> Trainer.train_component("actor_critic", steps)`.
`PMPOBeta.forward()` takes no real-data batch and `setup_training()` rejects
anything other than `WorldModelEnv`. It uses the existing `make_env_loop` under
`no_grad`, retaining pre-action observations, sampled actions, predicted rewards,
end/truncation flags and bootstrap values. Learning reevaluates likelihoods on
those detached observations/actions; neither DIAMOND nor the reward/end predictor
receives controller gradients. Real transitions only initialize imagination and
train the existing world-model components.

Each update starts a fresh prompt/rollout block and discards its coroutine state.
Prompt preloading is one batch on demand for this opt-in path, rather than
discarding 256 preloaded batches after every short rollout. The world model's
transition, reward and termination calculations are unchanged. A shared-loop
fix clones each pre-action observation so in-place imagined resets cannot
overwrite likelihood-replay states; its regression test also checks the final
observation bootstrap.

The existing `set_intrinsic_reward_fn(infos, rewards)` contract is preserved.
No reward scaling/composition changes were introduced. LCG, disagreement or
control reward providers can use the same learner interface; a new disagreement
implementation is outside this controller prototype.

## Defaults and prior block semantics

Select `agent=pmpo_beta env=dm_control` through the existing entry point.
Controller settings live in `agent.actor_critic.*`:

| Setting | Default | Rationale |
|---|---:|---|
| actor/value hidden dimensions | [256, 256] each | Small conventional MLP heads |
| actor/value learning rate | 0.0001 each | Retain base controller LR scale |
| gamma / lambda | 0.985 / 0.95 | Existing DIAMOND defaults |
| alpha_pmpo / beta_kl | 0.5 / 0.3 | Dreamer 4 equation 11 defaults |
| concentration_min | 1.0 | Inspected unimodal Beta behavior |
| prior_refresh_interval | 10 updates | Simple prototype choice, not paper-tuned |
| imagination_horizon | world_model_env.horizon (15) | Base horizon |
| max_grad_norm | 10 for each optimizer | Conservative prototype choice |
| seed | common.seed, or 0 when null | Private imagination stream |
| log_diagnostics | false | Opt-in JSON diagnostic lines |

The actor is copied into the prior **before updates 0, 10, 20, ...**. One block
is ten optimizer steps, each with new imagined experience; it is not ten epochs
over a stored trajectory. The first step after refresh has zero KL, but later
steps constrain departure from that frozen reference. Setting the interval to
1 therefore makes the prior term locally ineffective with this one-step scheme.

Legacy `actor_critic.actor_critic_loss.*` and its single optimizer LR do not
control PMPO. The old entropy bonus is not applied. Training counts, batch size,
warmup and sample weights remain under `actor_critic.training.*`. Budget defaults
are untouched. Gradient accumulation and multi-device PMPO are explicitly
rejected in this minimal implementation.

## Checkpoint behavior and limits

State includes actor/value/prior weights, update counter, both optimizers and
schedulers, fixed diagnostic observations, and private Python/NumPy/CPU/CUDA
imagination RNG state. Explicit prompt-loader seeding and fresh rollout blocks
avoid uncheckpointed prompt-prefetch/coroutine state. Worker count must be zero.
CPU and CUDA tests compare two actual subsequent imagined updates, losses,
diagnostics and parameters exactly across a saved/loaded checkpoint, including
a prior refresh. PMPO enables deterministic cuDNN in the trainer.

**This is learner resume evidence, not full online experiment resume evidence.**
The selected base does not checkpoint real collector coroutine/simulator state
or all whole-trainer RNG state; its LCG lifecycle has additional existing resume
limitations. Those were not imported from other branches or silently claimed
fixed. Resolve and test whole-run resume fidelity before a long experiment that
depends on resumption. Identical hardware/software is assumed by bitwise tests.

## Verification

From this worktree, final command:

```powershell
& C:/Users/jerry/miniconda3/envs/lcg/python.exe -m pytest tests -q -p no:cacheprovider --basetemp outputs/pytest-all-final
```

Result: **107 passed**, one pre-existing W&B/Sentry deprecation warning, 16.33 s.
See [tests.log](tests.log). This includes all 83 existing LCG tests and 24 new
model/checkpoint tests. The base had no `tests/models` or `tests/checkpoint`
suite; the new focused tests establish them.

Coverage includes Beta positivity/support/asymmetric bounds/density/entropy,
actor/value gradient isolation, optimizer ownership, likelihood direction for
both advantage signs, missing groups, mixed/near-zero groups, analytic KL,
frozen prior and refresh timing, hand-computed returns/termination/truncation,
real tiny DIAMOND reward/end and denoising integration, reward-hook compatibility,
detached rollout experience, observation-reset aliasing, and exact next-update
checkpoint replay on CPU and CUDA. Hydra composition also resolved successfully.

Initial sandboxed test invocations could not access Python-created temporary
directories, even inside the worktree. The reported complete runs used approved
execution outside that Windows sandbox. An initial CUDA exactness test exposed
a 5.8e-11 parameter difference; deterministic cuDNN fixed it without weakening
the exact comparison.

## Bounded main.py smoke

The smoke used the actual `main.py`, DMControl Walker/Walk, and GPU 0 (RTX 5060
Laptop, 8151 MiB), seed 7, PyTorch 2.7.1+cu128. It exited 0. No pretrained
checkpoint was used. LCG was explicitly disabled: reward was DIAMOND's scalar
task-reward prediction. There was no controller training on real rewards.

Smoke-only overrides: 16x16 images, tiny eight-channel world models and encoder,
32-unit actor/value heads, two conditioning frames, two denoising steps,
two imagined environments, horizon 3, zero LR warmup, compilation off. The run
collected eight real training steps, performed one denoiser and one reward/end
update, then three PMPO updates (18 imagined transitions). The inherited final
collector added one four-step evaluation episode. The 0.2-second environment
time limit was solely to bound this check. No return comparison is warranted.

Artifacts:

- [Full config](smoke_config.yaml) and [exact Hydra overrides](smoke_overrides.yaml).
- [Full smoke log](smoke.log), [all three diagnostic records](smoke_metrics.json).
- [Final checkpoint/fixed-state check](post_smoke_check.json).
- Runtime dataset/checkpoints: `outputs/pmpo-smoke-1/` (Git-ignored).

To reproduce this bounded smoke in a new output directory, use the recorded
override list with the existing main entry point (no alternate trainer):

```powershell
$pmpoOverrides = & C:/Users/jerry/miniconda3/envs/lcg/python.exe -c "import json,yaml; print(json.dumps(yaml.safe_load(open('docs/pmpo_beta/smoke_overrides.yaml'))))" | ConvertFrom-Json
& C:/Users/jerry/miniconda3/envs/lcg/python.exe src/main.py @pmpoOverrides hydra.run.dir=outputs/pmpo-smoke-repeat
```

Final update metrics, measured before that optimizer step unless indicated:

| Diagnostic | Value |
|---|---:|
| Rollout length / trajectories | 3 / 2 |
| Actor loss | 2.3086078 |
| Value MSE | 0.0018416423 |
| Value mean / std | 0.00862682 / 0.00802613 |
| Lambda return mean / std | 0.04578766 / 0.02150883 |
| Positive / negative fraction | 1.0 / 0.0 |
| Policy entropy / mean log probability (nats) | 3.6940627 / -4.6172104 |
| KL to prior | 0.00000965147 |
| Alpha mean / median / min / max | 1.691814 / 1.691624 / 1.686836 / 1.696185 |
| Beta mean / median / min / max | 1.694580 / 1.692954 / 1.688967 / 1.700068 |
| Near lower / upper bound fraction | 0.0277778 / 0 |
| Actor / value gradient norm before clipping | 0.4194968 / 0.0839684 |
| Losses, diagnostics, gradients finite | yes |

Near-bound means within 1% of the environment range, over all sampled action
components. One out of 36 components was near the lower bound on update 3;
both fractions were zero on updates 1 and 2.

Final-update action statistics by dimension (environment units):

| Dimension | Sample mean | Sample std | Policy mean | Fixed-state policy-mean std |
|---|---:|---:|---:|---:|
| 0 | -0.204398 | 0.509754 | -0.001273 | 0.002038 |
| 1 | 0.332057 | 0.401084 | -0.002392 | 0.000591 |
| 2 | 0.054336 | 0.577012 | -0.000633 | 0.000912 |
| 3 | 0.398318 | 0.425473 | 0.000235 | 0.000681 |
| 4 | -0.193560 | 0.565211 | -0.001507 | 0.000719 |
| 5 | -0.222928 | 0.454903 | 0.000670 | 0.000452 |

The fixed observation set is the same six observations retained from the first
rollout. Using deterministic means isolates state dependence from sampling noise.
All six dimensions remained state-dependent; the effect is small, as expected
for a near-initialization policy. The post-smoke JSON checks the saved weights
after all three optimizer steps as well.

## Files and remaining concerns

New implementation: `src/models/pmpo_beta.py`, `config/agent/pmpo_beta.yaml`.
Integration edits: `src/agent.py`, `src/trainer.py`,
`src/coroutines/env_loop.py`, `src/envs/world_model_env.py`.
New tests: `tests/conftest.py`, `tests/models/test_pmpo_beta.py`,
`tests/models/test_pmpo_integration.py`, `tests/checkpoint/test_pmpo_resume.py`.
This directory contains the report and evidence artifacts.

Departures from the paper: DIAMOND remains the world model; Beta policy instead
of its original action head; online snapshot prior instead of a frozen BC prior;
scalar MSE value instead of symexp/two-hot; independent single-frame encoders;
base gamma 0.985 rather than paper 0.997. From lucidrains: sign-only separately
normalized groups, full actor/encoder snapshot, explicit ten-update blocks,
separate Adam optimizers, no magnitude weighting or imported Dreamer machinery.

Before a longer Walker test, resolve full-run resume fidelity and assess the
single-frame temporal limitation. Validate with a trained world model and a
larger fixed set of actual Walker observations. Exercise the actual LCG and
disagreement reward modes end to end under identical controller settings;
this smoke only exercised task reward. All smoke advantages were positive,
so mixed/negative behavior is established by focused tests rather than this
trajectory sample. Refresh interval, KL strength and gradient clipping are
prototype defaults, not tuned claims. Boundary/entropy diagnostics should remain
enabled for the first longer diagnostic. No long experiment was launched.
