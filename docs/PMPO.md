# PMPO controller: design and history

How the PMPO controller in `src/models/pmpo_beta.py` is built, how it got its current settings, and the
history-initialization experiment that fixed how its imagined frame stack starts. Summarizes the original
implementation reports (`docs/pmpo_beta/`, 2026-09-23; removed 2026-11-12, still in git history) and the
history-initialization write-up (`docs/experiments/stage3_history_init_seed0.md`, 2026-10-19; removed 2026-11-12, never
committed). Results of all later controller variants are in
[CONTROLLER.md](CONTROLLER.md).

## 1. Origin and references

- **Started:** 2026-09-23 on branch `pmpo-beta-continuous` (from `lcg-undersampling-diagnostic`, `1708a7c`). First a
  single-frame prototype (`f6148d2`), then the four-frame (K=4) version.
- **PMPO objective:** Dreamer 4 paper, section 3.3, equations 10–11 (arXiv 2509.24527): sign-only advantage groups
  normalized separately, KL from the current policy to a prior, `alpha_pmpo=0.5`, `beta_kl=0.3`.
- **Beta actor and prior refresh:** lucidrains `dreamer4` (unimodal Beta readout with `softplus(raw) + 1`
  concentrations; a frozen copy of the policy refreshed before each imagination block). Its PMPO code weights
  likelihoods by `|tanh(advantage)|`; this project deliberately follows the paper's sign-only groups instead.
- **Not imported:** no controller or world-model code from other Dreamer 4 implementations. Action bounds come from the
  environment's action spec.
- **Departures from the paper:** DIAMOND stays the world model; Beta (later also tanh-Gaussian) actor; online snapshot
  prior instead of a frozen behaviour-cloning prior; scalar MSE critic (two-hot tried later); separate encoders for
  actor and critic; γ = 0.985 (DIAMOND default) instead of 0.997.

## 2. Design

### Actor distribution and action bounds (Beta)

```
alpha_j = softplus(raw_alpha_j) + concentration_min      # concentration_min = 1.0
beta_j  = softplus(raw_beta_j)  + concentration_min
x_j ~ Beta(alpha_j, beta_j)
a_env_j = low_j + x_j * (high_j - low_j)
```

- A floor of 1 keeps the density unimodal (no U shape); initial concentrations are about 1.69.
- Sampled actions are detached. Deterministic evaluation uses the Beta **mean**, not the mode. No tanh, no
  Q-maximization.
- Log-probability and entropy include the affine log-width term and sum over dimensions (nats). Reconstructing x from
  float32 actions is clamped to `[eps, 1 − eps]` only for density evaluation; log-probabilities are never clamped, and
  non-finite values raise.
- Bounds come from the real environment (`get_action_space_kwargs()` → `AgentConfig` → PMPO) and must be finite and
  strictly ordered.
- The tanh-Gaussian actor (`actor_distribution: tanh_gaussian`) was added later on `a13b57c`.

### Returns, advantages and losses

`V̄` is the detached value predicted during the rollout, fixed for that update (not a target network):

```
R_t = r_t + γ(1 − end_t)[(1 − λ)V̄(s_{t+1}) + λR_{t+1}],   R_after_last = V̄(s_after_last)
A_t = R_t − V̄(s_t);   D+ = {A_t ≥ 0},   D− = {A_t < 0}

L_actor = −α · mean_{D+} log π(a|s) + (1 − α) · mean_{D−} log π(a|s) + β_kl · mean KL(π ‖ π_prior) − entropy_coef · H
L_value = mean (V(s_t) − sg(R_t))²
```

- True termination removes the bootstrap. Truncation (environment or imagination horizon) bootstraps from
  `V̄(final_observation)` and stops the trace before the reset.
- Empty groups contribute zero and the other coefficient is not renormalized. Exactly zero counts as positive. No
  epsilon threshold, no magnitude weighting.
- KL is analytic and summed over action dimensions.
- The entropy term was added later (`entropy_coef=0.01`, Stage 3); 0.0 reproduces the original loss bit for bit.

### Architecture and training path

- **Separate networks:** actor and value each own an `ActorCriticEncoder` plus an MLP head (256, 256). Separate Adam
  optimizers (lr 1e-4 each), schedulers and gradient clipping (max norm 10). The prior is a frozen copy of the whole
  actor, encoder included, outside both optimizers.
- **Training in imagination only:** `Trainer.train_component("actor_critic")` → `PMPOBeta.forward()`, which accepts only a
  `WorldModelEnv`. Rollouts run under `no_grad` with `make_env_loop`; learning re-evaluates likelihoods and values on the
  stored, detached observations and actions. No gradient reaches DIAMOND or the reward/end model. Real transitions only
  provide imagination start states and train the world model.
- **Fresh rollout block per update:** prompts are loaded on demand, one batch at a time; coroutine state is discarded.
  Pre-action observations are cloned so in-place imagined resets cannot overwrite them.
- **Reward hook:** `set_intrinsic_reward_fn(infos, rewards)` is preserved for LCG or other intrinsic rewards (disabled
  for all controller experiments).

### Four-frame state (K=4)

- **Where it lives:** `FrameHistory` belongs to each `make_env_loop` coroutine, not to a network or to DIAMOND. It builds
  `(B, 12, H, W)` RGB states, oldest to newest. Actor, value and prior all consume exactly this stack. No recurrent
  state.
- **Real episodes:** reset gives `[o0, o0, o0, o0]`, then `[o0, o0, o0, o1]`, … `[o1, o2, o3, o4]`. A batched reset
  replaces only its own rows. The successor stack is built from `info['final_observation']` before an automatic reset.
- **Imagined rollouts:** imagination stores the exact action-generating stack (`store_policy_observations=True`).
  How the imagined stack starts is the subject of section 4: **repeat the replay start frame**, not the chronological
  replay prompt.
- **Datasets** keep only the current RGB frame; the stack is rebuilt by the loop.
- Old single-frame checkpoints are intentionally incompatible (the first convolution's input channels changed).

### Prior refresh

- The actor is copied into the prior before updates 0, N, 2N, …, and held fixed for N fresh-rollout updates (each with
  new imagined experience). The first update after a refresh has zero KL.
- The prototype default was N = 10. **A moving prior refreshed every 10 updates collapsed** in validation (Beta
  concentration runaway, entropy collapse, actions drifting to the bounds, a near state-independent policy). N = 500 is
  the validated setting; N = 250 later performed much worse (CONTROLLER.md).

### Current defaults (`config/agent/pmpo_beta.yaml`)

| Setting | Value |
|---|---|
| Actor / value hidden dims | [256, 256] / [256, 256] |
| Actor / value lr | 1e-4 / 1e-4 |
| γ / λ | 0.985 / 0.95 |
| `alpha_pmpo` / `beta_kl` | 0.5 / 0.3 |
| `entropy_coef` | 0.01 |
| `prior_refresh_interval` | 500 |
| `concentration_min` | 1.0 |
| `imagination_horizon` | `world_model_env.horizon` (15 by default; the best controller uses 10) |
| `max_grad_norm` | 10 (each optimizer) |
| `actor_distribution` | `beta` (best controller: `tanh_gaussian`) |
| `slow_critic_coef` / `slow_critic_rate` | 0.0 / 0.02 (best controller: 1.0 / 0.02) |

### Checkpointing and determinism

- Learner state includes actor, value and prior weights, the update counter, both optimizers and schedulers, fixed
  diagnostic observations, and a private Python/NumPy/CPU/CUDA imagination RNG.
- Data-loader workers must be 0; one device; no gradient accumulation. PMPO enables deterministic cuDNN (a 5.8e-11 CUDA
  difference disappeared with it).
- Learner checkpoints continue bit-exactly on CPU and CUDA, including across a prior refresh. Whole-run resume is not
  bit-exact for world-model training (AGENTS.md).

### Diagnostics (`+pmpo_diagnostic=<name>`)

- An observer inside the existing Trainer (not a second trainer). Writes every update to
  `pmpo_diagnostic/updates.jsonl` and every epoch to `epochs.jsonl`.
- **eval4:** each epoch, deterministic evaluation on seeds 1001–1004 with RNG state restored afterwards; never used for
  training. Up to 128 real K=4 states are kept for input probes (ordered, repeated-last, reversed).
- **Health stops** (diagnostic only, no learning change): non-finite losses, gradients or parameters; Beta concentration
  above 10,000; KL above 10 nats per state; value or λ-return above 1e6; more than 95% of sampled components within 1% of
  a bound for 50 updates; fixed-state action std below 1e-6 for 100 updates; canonical entropy below −5 nats per
  dimension, or a one-update fall of more than 2 nats with mean concentration above 100.
- Configurations: `bounded`, `bounded_fixed_prior`, `bounded_refresh500`, `bounded_full_loop` (used by the 100k
  launch scripts; checkpoint pruning rules in AGENTS.md).

### Validation at implementation time

- 107 tests passed (83 existing LCG tests + 24 new model and checkpoint tests), covering Beta support and density,
  gradient isolation, optimizer ownership, advantage signs and empty groups, analytic KL, prior refresh timing,
  hand-computed returns with termination and truncation, a tiny real DIAMOND integration, and exact checkpoint replay.
- A bounded `main.py` smoke on Walker (tiny models, 16×16 images, three PMPO updates) exited 0 with finite losses and
  state-dependent actions. It checked plumbing only, not performance.

## 3. Evaluation protocol

- **eval30 (authoritative):** 30 deterministic episodes, seeds 3001–3030, 500 steps, at 14 milestones (10k … 100k
  collection end = epoch 190, 100k final = epoch 240).
- **eval4:** the in-training proxy above. Not for comparisons.
- Report mean, median and fraction above 500, with paired per-seed differences.

## 4. Imagined policy-history initialization (Stage 3, seeds 0 and 1)

**Question.** After the code cleanup (`b976a6e`), Stage 3 (Beta, entropy 0.01, refresh 500, H15) no longer reproduced. The
cleanup had changed how the policy's imagined four-frame stack starts: from **repeating the replay start frame** to the
**chronological replay prompt**.

**Seed 0 (controlled comparison).** The reproduction run differs from the cleanup run only in an 8-line `env_loop.py`
patch restoring repeated-frame initialization; resolved configs differ only in `wandb.name`.

| Final eval30 (epoch 240) | Median / mean / frac > 500 | W&B |
|---|---|---|
| Historical Stage 3 (repeated frame) | 595.8 / 561.4 / 83% | `ef22guzq` |
| Reproduction (cleanup + patch) | 595.8 / 561.4 / 83% | `mx970ez4` |
| Cleanup (replay prompt) | 189.9 / 300.9 / 27% | `whe35fmy` |

- The reproduction's 30 returns equal historical Stage 3's **exactly** at every historical milestone (25k, 50k, 75k,
  100k).
- Reproduction vs cleanup at final: +260.5 mean, higher on 29 of 30 seeds. This attributes the seed-0 gap to history
  initialization.
- Both runs are non-monotonic (reproduction: 525 median at 50k, 215 at 60k).

**Seed 1.**

| Final eval30 | Replay prompt (`3i6dvzj1`) | Repeated frame (`4op80ok2`) |
|---|---|---|
| Median / mean / frac > 500 | 253.8 / 246.0 / 0% | 272.2 / 321.3 / 7% |

- Repeated frame is ahead at 10 of 14 milestones by mean; +75.3 mean at final, but 15/15 wins and losses (gains
  +156.6, losses −6.0 on average). Neither reaches the seed-0 level.
- Both sit near 250 from about 40k: **seated scooting**, an absorbing mode (0 of 183 seated bouts exit to an upright
  run). Repeated frame scores higher by entering it later or never.

**Why seed 0's repeated-frame policy does better (final checkpoints).**
- Both policies fall backward equally often, mostly at reset; the historical policy recovers from every backward fall
  (15 of 15), the cleanup policy from none (0 of 14) and slides along on its back.
- Reward models, world models (both lose a real get-up within 5–8 open-loop steps) and critics (both overvalue
  back-fallen states by 20–40) are similarly flawed in both runs. From the same restored back states the historical
  policy recovers 8 of 8 times, the cleanup policy 0 of 8: the difference is in the policy.
- PMPO advantages in grounded states are near zero; λ-targets there are 95–99% bootstrapped value.

**Conclusions.** Repeated-frame initialization is the project default since then (all later runs use it). The
mechanism by which history initialization changes training is not established, and two training seeds do not prove it
is better in general. Historical Stage 3 and its reproduction are one training seed, not two.
