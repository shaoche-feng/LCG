# Precision audit for the Laplace-sampled ensemble

Date: 2026-10-26. Code at base `cd78477` plus the Laplace change (committed as `9176c13`). Paths and line numbers are as of that code; `src/lcg/` is now `src/intrinsic_reward/lcg/` (`88bc7a2`). This audit records what LCG's historical precision actually computes before it is reused as an inverse covariance. Required by `docs/LAPLACE_ENSEMBLE_DESIGN.md`, Section 5.

## 1. Where it is computed

- `src/lcg/precision.py`:
  - `historical_precision` (line 104), the formula;
  - `_backward_vjp_probe` (line 84), one probe;
  - `sample_uniform_historical_transitions` (line 20), the reference draw.
- `src/lcg/lifecycle.py`, `lcg_historical_precision` (line 200): new and shared. It holds the exact call LCG has always made: `B = precision_reference_size`, `N = dataset.num_steps`, `num_mc = precision_num_mc`, `beta`, `damping`, `seed = round identifier`.
  - `LCGLifecycle.refresh` (line 152) and `LaplaceEnsemble.refresh` (`src/laplace.py`) both call it.
  - No precision code is duplicated, and Laplace does not build LCG's candidate JVP bank or run candidate scoring.
- A test checks that Laplace's precision is bit-identical to `LCGLifecycle.refresh`'s `h_D` for the same denoiser, replay, settings and round.

## 2. The formula as implemented

For selected parameters `theta_S` (dimension `d_S`), reference transitions `i = 1..B`, and probes `m = 1..M`:

```text
v_im = grad_{theta_S} [ sqrt(2) * sum( xi_im * F_theta(y_sigma_im, x_i) ) ]      (precision.py:96-100)
       xi_im ~ N(0, I), the shape of the raw network output F_theta
       sigma_im ~ p_train(sigma) (the denoiser's training log-normal, clipped)  (line 141)
       y_sigma_im = y_i + sigma * eps + sigma_offset_noise * eps_offset         (lines 142-145)
g_i  = (1/M) * sum_m v_im^2   (elementwise)                                     (line 146)
h    = damping * 1  +  beta * (N / B) * sum_i g_i                               (lines 124, 135, 147)
```

**What is accumulated.** `F_theta` is the raw inner-model output, the quantity the training loss regresses on (`denoiser.py:157-158`). For a fixed input, `E_xi[v^2] = 2 * diag(J^T J)`, where `J = dF/dtheta_S`. So `g_i` is a Hutchinson-style single-probe Monte Carlo estimate of the diagonal of the Gauss-Newton matrix (2·JᵀJ) of the *summed* squared error over output pixels and channels. The noise level and corruption are averaged over M probes.

**What it is not.** It is not the curvature of the training loss as implemented. That loss is `F.mse_loss`, a mean over pixels and batch, and is not multiplied by its pixel count here. It is also not an empirical-Fisher squared gradient of the loss.

**Probe combination.** The squared VJPs are averaged within each transition (1/M), then summed over the B transitions.

**Replay-size factor.** Already applied: `N / B` rescales the B-transition sum to an estimate of the sum over the whole replay (N = current replay size). The precision therefore grows roughly in proportion to the replay. B = 320 is the reference-bank size used to estimate that sum; it is not the data count.

**Likelihood and other multipliers.** `beta` (1.0) multiplies the data term. The sqrt(2) in the probe gives the factor 2 of the squared-error Gauss-Newton matrix. There is no other temperature, noise-variance, pixel-count or likelihood scale. Reference transitions are drawn uniformly without replacement (`rng = default_rng(seed)`, line 48), using only obs/act windows of the replay; rewards are never read.

**Damping.** `1e-4` is added once, as the initial value (line 124). Within LCG it serves as numerical stabilization of `h^-1/2`. Read as a Gaussian prior, it is an isotropic prior precision of 1e-4, i.e. a prior std of 100 per coordinate. Laplace inherits exactly this; a test checks that a zero-curvature probe gives `h = damping` exactly, and that a unit probe gives `damping + beta * N`.

**Representation.** Diagonal: one vector of length `d_S` (R1: 217,664). Laplace supports only this representation (`perturbation_scale` rejects anything else). Non-finite or non-positive values raise an error, with no fallback.

**Seeding.** Reference draw and probes are seeded by the round identifier (the epoch), LCG's existing convention; `common.seed` does not enter. Laplace's own sample bank uses a separate private CPU generator seeded by `SeedSequence([common.seed, 21, epoch])`.

## 3. The Gaussian Laplace builds from it

```text
q(theta_S) = Normal(theta_hat_S, T * diag(h)^-1),  T = posterior_temperature = 1.0
theta_S[k] = theta_hat_S + sqrt(T) * eps_k / sqrt(h),  eps_k ~ N(0, I),  k = 1..5
theta_not_S = theta_hat_not_S
```

- Code: `perturbation_scale` (laplace.py:106) and `LaplaceEnsemble.refresh` (lines 169-171).
- Nothing is added: no extra replay-size factor, prior, damping, likelihood multiplier, clamp or norm limit.
- Because of the scale conventions above (summed-pixel Gauss-Newton matrix, replay-sum rescaling, damping as the only prior), this is a curvature-based approximation. It is not a calibrated posterior, and the trained weights need not be a mode.

## 4. Consequence observed in the smoke (reported, not corrected)

- **Damping-only coordinates.** At 600-800 replay transitions on a denoiser with 40 updates:
  - many R1 coordinates have curvature far below the damping;
  - the precision minimum is ~1e-4 (damping only) in every refresh;
  - the precision median is 2.1e-4, 1.5e-3 and 2.1e-2 in epochs 1-3.
  - Their std is therefore close to 1/sqrt(1e-4) = 100.
- **Relative perturbation size.** Each sample's perturbation norm was ~2,500x, ~1,900x and ~1,160x the norm of the selected weights.
- **Effect on images (corrected 2026-10-27).** The earlier statement here, that the effect was modest because members differ from the base by RMSE 0.14-0.18 with no saturated values, was wrong. It was inferred from those statistics without looking at images. The frozen-model diagnostic's image sheets show that at T = 1 every member's completed prediction is a nearly uniform grey frame with the walker gone, on both the final checkpoint and the epoch-2 milestone. The members agree with each other (spread RMS 0.017) but have all lost the content (99% of member-to-base deviation is common shift).
- **Outside the local approximation.** The raw-output change at T = 1 bears little relation to the first-order (JVP) prediction (relative error ~1.0-1.5), so the curvature-based Gaussian is used far outside the region where it describes the network. See `docs/REWARD_FREE_EXPERIMENT_NOTES.md`, "Review corrections and frozen-model diagnostic".
- **No tuning.** As the design requires, temperature, damping and sample norms were not changed in response to these results.
