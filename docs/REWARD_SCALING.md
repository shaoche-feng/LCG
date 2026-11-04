# LCG reward scaling

How the LCG scorer's raw score becomes the PMPO policy reward. Implementation and selected configuration independently checked on 2026-11-04, from `7c1a383` on branch `lcg-reward-scaling`. This document is `docs/REWARD_SCALING.md`; there is no second repository-root copy.

## Raw score and policy reward

- **Raw LCG score** (`LCGLifecycle.score_candidates`): one non-negative value per imagined transition. It is computed with the round's frozen precision `h_D` and probe bank, both built at `refresh()`.
  - The implementation averages `2 * ||J_F (h_D^(-1/2) * eta)||²` over the frozen MC bank. Write `G_j = 2 * ||dF/dtheta_j||²`, including the output-coordinate sum and corruption/sigma expectation. Historical precision is `h_j = damping + beta * N * Ghat_j`, where `Ghat_j` estimates the replay mean of that quantity. Taking the probe expectation gives approximately `sum_j G_j(candidate) / (damping + beta * N * Ghat_j)`.
  - For `beta=1`, negligible damping, comparable candidate/replay sensitivities and adequate MC estimates, this is approximately `(1/N) * sum_j G_j(candidate) / Ghat_j`, with a typical replay score near `d_S/N`. For `d_S=217,664`, this gives about 40 at N=5,500 and 22 at N=10,000. This is a scale heuristic, not a guarantee of monotonic scores while the model and replay change.
- **Policy reward** (`LCGLifecycle.scale`): the raw score transformed by `intrinsic_reward.reward_scaling`. PMPO's critic is trained on this value.
  - The actor's sign split is invariant to multiplying **all advantages** by the same positive constant; mean-normalised magnitude weights share that property away from their `1e-8` denominator floor. Changing rewards does not in general multiply all advantages by that constant: current/bootstrapped critic values are not automatically rescaled. The actor's sign groups and weights can therefore change immediately, and critic targets, losses, optimisation and clipping also change. Full learning dynamics are not scale-invariant.
- Both values are logged per imagined step of each PMPO update: `actor_critic/train/intrinsic/raw_reward_mean` / `_max` and `.../scaled_reward_mean` / `_max`. With debug metrics, further statistics are logged as well.

## Modes

Keys are under `intrinsic_reward.` (`LCGConfig`, `src/intrinsic_reward/lcg/lifecycle.py`). The `data_scaled` and `reference` modes have one divisor per scorer round. A round runs from one `refresh()`, at the start of each active epoch's PMPO phase, to the next. `fixed` keeps its configured divisor across rounds; RMS updates its statistic during policy training and freezes it only for evaluation.

| `reward_scaling` | Policy reward | Required keys | Must be null / false |
| --- | --- | --- | --- |
| `none` | raw | — | `fixed_divisor`, `data_reference_steps`, `reference_num_prompts`; `rms_enabled=false` |
| `fixed` | raw / `fixed_divisor` | `fixed_divisor` (finite, > 0) | `data_reference_steps`, `reference_num_prompts`; `rms_enabled=false` |
| `rms` | `rms_alpha` · raw / (√s² + `rms_eps`), with s² ← ρ·s² + (1−ρ)·mean(raw²) per PMPO batch, ρ = `rms_ema_decay`, s² starting at the first batch's mean square and reset at each refresh | `rms_enabled=true` | `fixed_divisor`, `data_reference_steps`, `reference_num_prompts` |
| `data_scaled` | raw / d, with d = `fixed_divisor` · `data_reference_steps` / N | `fixed_divisor`, `data_reference_steps` (integer > 0) | `reference_num_prompts`; `rms_enabled=false` |
| `reference` | raw / d, with d = the mean raw score of the round's reference transitions | `reference_num_prompts` (integer > 0) | `fixed_divisor`, `data_reference_steps`; `rms_enabled=false` |

**Defaults.**
- **`LCGConfig`:** `reward_scaling=None`, `fixed_divisor=None`, `data_reference_steps=None`, `reference_num_prompts=None`, `rms_enabled=True`, `rms_alpha=1.0`, `rms_ema_decay=0.99`, `rms_eps=1e-8`.
  - `reward_scaling=None` resolves to `rms` if `rms_enabled` is true and to `none` otherwise. Saved configs from before the field existed keep that meaning.
- **`config/intrinsic_reward/lcg.yaml`:** `reward_scaling: rms`, `fixed_divisor: null`, `data_reference_steps: null`, `reference_num_prompts: null`, `rms_enabled: True`. These keys use ordinary launch overrides, without `+`.
- **Preset `config/experiment/walker_lcg_100k.yaml`:** `reward_scaling: reference`, `reference_num_prompts: 32`, `fixed_divisor: null`, `rms_enabled: false`. The 200 PMPO updates in the first and every later epoch come from `config/exploration.yaml`, shared by all learned methods.
- **Scope:** `data_scaled` and `reference` exist for LCG only. The ensemble accepts `fixed`, `rms` and `none`; Laplace and Plan2Explore accept `fixed` and `none`.

### Data-scaled

- **Formula:** d = `fixed_divisor` · `data_reference_steps` / N, where N = `train_dataset.num_steps` at the refresh (initial collection included). So the policy reward = (raw / `fixed_divisor`) · N / `data_reference_steps`.
  - The factor N compensates the leading approximate 1/N dependence. Additive damping and coordinate-specific curvature/model changes prevent an exact cancellation in general.
  - At N = `data_reference_steps` the mode equals `fixed`.
- **Anchor:** the runs used `fixed_divisor=100` and `data_reference_steps=5500`, the replay size at the first refresh (5,000 initial + 500 steps). Epoch 1 is then identical to the preset's raw/100, and a typical replay transition scores about 0.40 (≈ S/550,000). The anchor is not a default; it must be set.
- **Determinism:** d is computed in `refresh()` and needs no extra compute or randomness. It corrects only the leading N-scaling; other drifts in G_j / mean(G_j) remain. The code multiplies the replay-size ratio first to avoid intermediate overflow and refuses a non-finite or zero derived divisor.

### Reference

- **Calibration** (`ExplorationTrainer.lcg_reference_scores`): runs once per refresh, after `refresh()` and before any PMPO update of the round. It:
  1. draws `reference_num_prompts` (selected: 32) replay prompts uniformly **without replacement** over all stored replay steps, with `sample_uniform_historical_transitions`, seeded by the round. Each window has the denoiser's conditioning length and ends on the sampled real frame; earlier positions are left-padded when necessary. The replay must have at least that many steps;
  2. imagines each open-loop for `world_model_env.horizon` (10) steps under independent uniform random actions within the training env's action bounds. This uses `WorldModelEnv.imagine_candidates`, with this epoch's denoiser and sampler, and does not touch the env's own rollout state;
  3. scores the 32 × 10 = 320 imagined transitions with the just-refreshed scorer.
- **Divisor:** `LCGLifecycle.calibrate(scores)` sets d to the float64 arithmetic mean over all `prompts * horizon` transition scores, not a sum of trajectory returns. With the selected 32 prompts and horizon 10 this is 320 equally weighted transitions.
- **What "policy reward = 1" means:** the raw score equals this round's mean raw reference score. This is a ratio in scorer units, not proof that a transition is equally novel or that the scorer is calibrated uncertainty. For fixed candidates, model and probe bank, a uniform multiplication `h_D -> c * h_D` divides both numerator and divisor by c and cancels in the ratio (up to floating-point error). Replay growth need not be such a uniform multiplication, so this does not remove every change across rounds.
  - It is a relative scale and says nothing absolute about novelty.
  - The reference itself moves as the model and replay change.
- **Cost:** with the selected setting, 320 extra candidate scores per refresh. The earlier RTX 5090 run reported about 1.2 s additional refresh time (9.1 s versus 7.9 s); this is a historical observation, not a new benchmark or a hardware-independent cost estimate.

## Divisor lifecycle and safeguards

- **Frozen per round:** `refresh()` clears `reward_divisor` and `reference_raw_mean`, then sets the divisor (data-scaled) or leaves it for `calibrate()` (reference). It is the same for every PMPO batch of the round.
  - In reference mode, `scale()` raises `RuntimeError` until `calibrate()` has run (`needs_reference`), so PMPO can never use an uncalibrated round.
  - `calibrate()` refuses calls before a refresh and a second calibration of an already calibrated round. Only the next refresh can clear the divisor.
- **Invalid divisors:**
  - `calibrate()` requires non-empty, individually finite, non-negative scores with a strictly positive finite mean. Zero scores are allowed when the overall mean is positive. There is no fallback divisor or clipping of invalid data.
  - `LCGConfig.__post_init__` validates the mode-specific keys, a boolean `rms_enabled`, and mode agreement. Fixed divisors must be finite positive numbers; counts must be positive integers (booleans are rejected). Active RMS alpha/epsilon must be finite and positive, and decay finite in `[0,1)`; booleans and strings are rejected. Inactive RMS fields retain their legacy ignored meaning. Ensemble RMS has the same finite-number checks; Laplace and Plan2Explore validate their fixed divisors and reject unsupported modes.
- **Checkpoints:** `LCGLifecycle.state_dict()` includes `reward_divisor`, `reference_raw_mean` and the RMS `s2`. `load_state_dict()` restores them, so a resumed round continues with the same divisor and needs no recalibration.
  - Before restoring a scorer, load rejects missing, non-numeric, non-finite or non-positive per-round divisors, an absent/inconsistent reference mean, and non-finite/negative/non-scalar RMS mean squares. Reference mean and divisor must agree exactly because calibration stores the same value in both fields.
  - Modes without a divisor accept checkpoints that predate the fields.
  - Ensemble RMS likewise rejects non-finite, negative or non-scalar restored mean squares.
- **Compatibility:** `LCGConfig.reward_scaling_settings()` (mode plus that mode's keys) is saved in the checkpoint (`lcg_reward_scaling`) and in the scorer settings. A resume with different settings is refused, by both `ExplorationTrainer.load_state_dict` and `LCGLifecycle.load_state_dict`. The settings are also recorded in the W&B config (`provenance`).
- **RNG isolation:**
  - Calibration uses its own seed namespace: epoch + 2,000,000. Precision draws use the epoch, and the probe bank uses epoch + 1,000,000.
  - The global Python, NumPy, Torch and CUDA RNG state is saved before calibration and restored after (`rng_state` / `set_rng_state`), so PMPO's random stream and prompt stream are unchanged.
- **Logged per epoch:** `lcg/reward_divisor` (data-scaled and reference) and `lcg/reference_raw_mean` (reference).

## Selected setting for new LCG experiments

**Selected:** `reward_scaling=reference` with `reference_num_prompts=32`, and 200 PMPO updates per epoch, including the first (`actor_critic.training.steps_first_epoch=200`, `steps_per_epoch=200`). Everything else is the preset's.

Hydra composition now selects this setting for new LCG runs:

| Configuration | `reward_scaling` settings | PMPO updates, first epoch / later epochs |
| --- | --- | --- |
| `exploration.yaml` alone | `rms` (α 1.0, ρ 0.99, ε 1e-8) | 200 / 200 (5,000 / 400 before the shared 100k schedule) |
| `+experiment=walker_lcg_100k` (maintained preset) | `reference`, 32 prompts | 200 / 200 |
| historical scaling test / continuation (explicit launch overrides) | `reference`, 32 prompts | 200 / 200 |

```
python src/exploration_main.py +experiment=walker_lcg_100k
```

The base group remains RMS for compatibility. The LCG preset changes only its scaling selection; the policy-update count is the shared repository schedule (see `docs/AGENTS.md`, "Environment and schedules"). Its controller, denoiser, collection budget and other settings remain those of the preset. The earlier historical commands used `+intrinsic_reward.reference_num_prompts=32` because that key was then absent from the base YAML; current commands use the preset or an ordinary override.

**Existing runs retain their saved selection:** the new preset does not rewrite their saved config. Valid saved round state restores its exact divisor without recalibration, and resume refuses changed scaling settings. The numerical/configuration guards above reject invalid state that older code accepted. Continuing with different code versions is not a claim of bit-identical future training; a run's own copied source remains its historical implementation.

## Tests and evidence

Tests:
- **`tests/lcg/test_reward_scaling.py`:**
  - fixed constancy and outlier independence; RMS still applied;
  - validation of every mode's keys; legacy meaning;
  - data-scaled divisor vs N;
  - reference: exactly one calibration per ready round, then dividing by the mean; invalid score rejection;
  - per-round divisor checkpointed, with invalid/missing divisor, inconsistent mean and invalid RMS statistic refusal;
  - settings recorded for resume;
  - `imagine_candidates` reproduces `WorldModelEnv.step()`'s candidates exactly.
- **`tests/lcg/test_reward_scaling.py`** also covers the full `lcg_reference_scores` helper: seeded distinct prompts, asymmetric action bounds, horizon/transition count, repeatability, live-buffer/prompt-stream/weight/scorer preservation, and Python/NumPy/Torch RNG restoration on success and on imagination/scoring exceptions.
- **`tests/exploration/test_lcg_preset.py`:** composed selected reference/200-update configuration, once-per-refresh calibration before policy updates, the frozen divisor shared with real-episode intrinsic scoring, restored divisor/bank without refresh or recalibration, and explicit LCG mode/prompt-count resume refusal. Fixed scaling and legacy preset checks remain covered.
- The budget-extension implementation and its tests are separate work and are excluded from this reward-scaling commit. Ordinary epoch-boundary resume remains covered by the existing resume suite.

Evidence:
- **Earlier reported unit tests:** full suite, 716 passed (laptop, 2026-11-04). This is historical supplied evidence, not the result of this independent verification; current verification results are recorded below. Empty CUDA visibility does not reliably hide CUDA in this Windows build, so current CPU checks use `CUDA_VISIBLE_DEVICES=-1`.
- **Scaling test** (`outputs/step27-lcg-reward-scaling/`, 40/40 checks):
  - Every update's policy reward = raw / logged divisor (relative error < 1e-5).
  - The data-scaled divisor = 100·5,500/N at every refresh.
  - The data-scaled epoch 1 is identical to raw/100.
  - The reference arm's first PMPO update has the same raw reward as the raw/100 arm. This is a bounded consistency observation; the new RNG/state unit tests independently check isolation, including exception paths.
- **Continuation to 15,000 steps** (`outputs/step28-scaling-continuation/`, 57/57 checks): the divisors stayed correct over 20 refreshes and were restored exactly on resume.

## Limitations

- **These modes target reward-scale control.** They do not guarantee steady policy rewards as the policy/reference distribution changes, accurate critic values, useful exploration, calibrated uncertainty or improved downstream performance.
- **Critic in the runs:**
  - In the 10k runs, both modes avoided the late critic overestimation seen with raw/100.
  - At the 15k imagined diagnostic (epoch 20), value bias was positive in both arms and the positive-advantage fractions were 0.30 (data-scaled) and 0.25 (reference). These are diagnostic-rollout results: epoch-20 training means were respectively 0.42 and 0.55, with value-minus-target means +0.050 and −0.143. They must not be presented as the same measurement.
- **Downstream in the runs:**
  - At 10k, neither mode was distinguishable from raw/100 downstream (one seed pair).
  - From 10k to 15k, Walker-walk returns fell in both arms: reference 65.4 → 50.6, data-scaled 60.8 → 40.7.
  - Reference − data-scaled at 15k was +9.9 ± 5.0, which is inconclusive.
  - The ± values are paired standard errors over 30 evaluation episode seeds for one exploration/adaptation seed pair. They are not uncertainty across independently trained policies. The means and paired summaries agree with the existing step-27/28 analysis tables; no new learning run was performed for this review.
- **What the selection rests on:** reference was chosen for its ratio interpretation and cancellation of a common scalar precision factor. The existing runs do not demonstrate a downstream benefit over fixed scaling or over data-scaled reward.
- **Units:** policy rewards and critic values are in each mode's own units. Do not compare their sizes across modes or methods.

## Independent verification (2026-11-04)

Confirmed and fixed within reward scaling:
- An already calibrated reference round could be recalibrated, replacing its frozen divisor. Calibration now requires a ready round and refuses a second call.
- Invalid restored divisors/reference means and invalid RMS statistics were accepted. Load now rejects these before restoring the scorer; the RMS guard applies to LCG and the ensemble scorer.
- RMS configuration accepted infinity and boolean values as numbers. Active RMS parameters now require finite real numeric settings, and LCG's enable flag must be boolean.
- Data-scaled multiplication could overflow before dividing by N even when the final result was finite. The ratio is now evaluated first; a zero/non-finite derived divisor is refused.
- The maintained preset did not implement the selected reference/32-prompt/200-update configuration. Its composition is now explicitly tested.

The calibration helper's prompt sampling, action bounds, horizon, score count, RNG restoration and live prompt-stream preservation were already correct in the inspected implementation. The new end-to-end tests verify those paths on success and injected failures. No replacement calibration mechanism was added.

CPU environment: local Conda `lcg` interpreter, OMP/MKL threads 1, `CUDA_VISIBLE_DEVICES=-1`, `WANDB_MODE=disabled`, `PYTHONDONTWRITEBYTECODE=1`, installed Git Bash first on PATH, and fresh short temporary paths. No learning experiment, package upgrade or new GPU benchmark was run.

| Check | Result | Exit |
| --- | --- | --- |
| Existing scaling/preset baseline | 39 passed | 0 |
| Confirmed-gap regressions before fixes | 35 failed, 3 calibration-isolation cases passed | 1 |
| Lifecycle/scaling/preset checks after initial fixes | 102 passed | 0 |
| Preliminary broader affected suite | 339 passed, 2 existing CUDA skips | 0 |
| Final scaling/preset suite from the exact staged snapshot | 79 passed | 0 |
| Full suite from the exact staged snapshot | **747 passed, 6 existing CUDA skips**, 178 sampler deprecation warnings; 909.74 s | **0** |

The staged snapshot excludes the unrelated collection-budget extension and uses the committed `config/exploration.yaml`, while the user's working-tree milestone edit remains unchanged. The final RMS-restore test also reproduced four ensemble failures before its guard was added. An initial fixture comparison used an unhashable segment object and an unavailable cross-folder import; those fixtures were corrected before the 35-failure reproduction. A first staged-export run lacked its ignored `outputs/` parent and had fixture setup errors; creating that parent yielded the 79-pass result without changing assertions or source.

Final staged-snapshot commands (working directory: the exported index, with its `outputs/` parent created):
```text
python -m pytest -q tests/lcg/test_reward_scaling.py tests/exploration/test_lcg_preset.py -p no:cacheprovider --basetemp=outputs/t2
python -m pytest -q tests -rs -p no:cacheprovider --basetemp=outputs/t3
```
Exact interpreter paths, environment variables, commands, timings, exit codes and continuous stdout/stderr logs are retained locally under `outputs/rs4/`; generated evidence is excluded from the commit. Source/config/test blobs in the exported staged snapshot are checked against the final index before committing. CUDA calibration is not newly verified by these CPU tests; all six CUDA-only tests retain their existing skips.

Read-only checks of the supplied evidence found 40/40 and 57/57 recorded checks marked successful. Recomputed episode-return means are data-scaled 60.7628/40.6852 and reference 65.4340/50.6208 at 10k/15k; the 15k paired difference is 9.9356 with episode-seed SE 4.9825. These support the rounded reported observations, not a new learning or physical-state result.
