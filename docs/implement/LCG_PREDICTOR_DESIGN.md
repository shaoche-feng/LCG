# Predictor-based LCG: design

Status (2026-11-25, revised after review): implemented as `intrinsic_reward.method=lcg_predictor` (output label `lcg-predictor`, preset `lcg_predictor_100k`). Refresh is transactional, and `exact_trace` is the default scorer. Evidence so far: integration and numerical tests, one frozen-replay fitting-budget check and one bounded 6,000-step Walker controller check. No 100k exploration run has been launched. As-built notes and evidence: `REWARD_FREE_EXPERIMENT_NOTES.md`, Step 6, "Predictor-based LCG".

## Idea

Plan2Explore scores a candidate transition by how much five independently fitted feature predictors disagree. Predictor-based LCG keeps Plan2Explore's frozen-denoiser features. It replaces the ensemble with one predictor and a **diagonal GGN-based Laplace approximation around the fitted predictor**: the posterior over all of the predictor's parameters is approximated as N(ψ̂, diag(h_D)⁻¹), where ψ̂ are the fitted weights.

- **Not necessarily the MAP.** ψ̂ comes from a fixed budget of optimizer steps on the explicit MAP objective. Nothing guarantees convergence to the MAP, so the approximation is centred on the fitted weights, whatever they are.
- **Score.** A candidate is scored by r = tr(J diag(h_D)⁻¹ Jᵀ) / τ², the approximate posterior's output covariance relative to the likelihood variance. This is an **information-gain surrogate**: the leading, small-variance term of the expected information gain about ψ from observing z_{t+1}, ½ log det(I + J Σ Jᵀ / τ²) ≈ ½ tr(J Σ Jᵀ) / τ². It is not a calibrated uncertainty, and the diagonal approximation ignores parameter correlations.
- **Relation to LCG.** This is the LCG score of a small, fully parameterized model, instead of the denoiser's selected parameters (R1).

## Features (reused from Plan2Explore, unchanged)

- **Extractor E.** The frozen denoiser's `unet.d_blocks.2` and `unet.d_blocks.3` outputs, each average-pooled to 4×4 and concatenated, give D = 2,048.
- **Extraction input.** The input is o_t + 0.1·eps + 0.3·eps_offset, one fixed probe from `probe_seed`, so the effective conditioning sigma is about 0.316. It is conditioned on o_{t−4..t−1} and a_{t−4..t−1}, zero-padded before an episode start, with EDM preconditioning.
- **Standardization.** A coordinate-wise mean and std over the replay frames used by valid transitions, with std ≥ 1e-3. It is recomputed whenever the feature cache is rebuilt.
- **Transitions.** (o_t, a_t) → o_{t+1}, where o_{t+1} is the next stored frame, the episode's final observation, or the collector's successor at a cut. The input is u = [z_t, ā_t], with z_t standardized and ā_t = 2(a − low)/(high − low) − 1. The target is y = standardized z_{t+1}.
- **Reuse.** The code reuses `FeatureExtractor`, `build_feature_cache`, `replay_identity`, `context_window` and `PredictorHead`. Imagination reports features through the same `WorldModelEnv` hook.

## Predictor and MAP objective

- **Model.** f_ψ is one MLP: (D + A) → 256 → 256 → D, SiLU, with a linear output. For Walker (A = 6) that is 2054 → 256 → 256 → 2048, with P = 1,118,208 parameters. Every parameter is used for curvature. Other domains change only A; other feature selections change D.
- **Likelihood and prior.** y_i ~ N(f_ψ(u_i), τ² I), and ψ ~ N(0, λ⁻¹ I).

  L_D(ψ) = Σ_i ‖y_i − f_ψ(u_i)‖² / (2τ²) + λ‖ψ‖² / 2

- **Minibatch objective.** This is L_D / N, with N the number of valid replay transitions and B uniform with-replacement draws:

  L_batch = Σ_{i∈batch} ‖y_i − f_ψ(u_i)‖² / (2τ²B) + λ‖ψ‖² / (2N)

  - Reductions: ‖·‖² sums over the D feature dimensions, and the data term averages over the B transitions. It is D × PyTorch's element-mean MSE.
  - Weight decay is never used. The prior is the explicit penalty term (`training.map_terms`), on weights and biases.
- **Not the diffusion noise.** τ² is the feature-likelihood variance, in standardized-feature units. It is unrelated to the diffusion noise σ and to `feature_sigma`.
- **Budget.** Plan2Explore's per-head budget: Adam, lr 1e-3, batch 256, 500 updates per refresh, no bootstrap.
  - `init_policy: reinitialize` uses a fresh initialization from the round's seed.
  - `warm_start` starts from the previous round's weights. In both cases the optimizer state is new.
  - L_D/N over the whole replay is evaluated before and after fitting and recorded. A fixed budget does not guarantee convergence to the MAP.
  - On the frozen Walker 100k replay, L_D/N was still falling at 2,000 updates (183, 158, 137 after 500, 1,000, 2,000). The exact-trace scores barely changed (Spearman ≥ 0.966, top-10% overlap ≥ 0.83, mean within 1%; `REWARD_FREE_EXPERIMENT_NOTES.md`). The 500-update budget is kept.

## Historical precision

h_D = λ·1 + s · diag(Σ_{i∈H} J_iᵀ J_i) / τ², with J_i = ∂f_ψ(u_i)/∂ψ ∈ R^{D×P}.

- **Set H (default: the full replay).** `curvature_num_transitions: null` uses every one of the N valid transitions, so the sum is exact and s = 1. The replay is moved to the device in blocks of `curvature_block_size` (8,192) and accumulated in float64. At 100k transitions this takes about 0.8 s on the RTX 5090 (peak 180 MiB), against about 23 s for replay encoding. Changed from a 320-transition subset on 2026-11-26.
- **Optional subset.** An integer B_H draws B_H transitions uniformly without replacement with a private generator (if N ≤ B_H, every transition is used, unscaled). Scaling: s = N / B_H (`curvature_scaling: full_replay`; consistent with the MAP objective's full-replay likelihood) or s = 1 (`none`).
- **Record.** N, the number of transitions used, the sampling rule, the seed (None for the full replay) and the scale factor are recorded.
- **Backends.** Both are exact GGN diagonals; neither uses squared loss gradients.
  - `closed_form` (default): with layer inputs h_l, pre-activations a_l and column norms c_l[p] = ‖∂f/∂a_l[p]‖²:
    - hidden layers: W_l[p, q] gets c_l[p]·h_{l−1}[q]² and b_l[p] gets c_l[p];
    - output layer: W_L[k, j] gets h_{L−1}[j]² and b_L[k] gets 1.
    - c_l comes from the backward recursion P ← W_lᵀ Q_l W_l, Q_l = diag(σ′(a_l)) P diag(σ′(a_l)), starting from P = W_Lᵀ W_L.
  - `autograd`: per-sample VJPs against every output basis vector, squared and summed. It is generic and slow.
- **Per refresh.** h_D is recomputed from scratch at every refresh, for the newly fitted predictor. Nothing is accumulated across rounds, so no replay is counted twice.

## Candidate scoring

Scoring uses only the current z_t and a_t: no next-feature target and no diffusion-noise draw.

- **`exact_trace` (default).**
  - Score: r_raw = Σ_j ‖J(u)[:, j]‖² / h_D[j] / τ², computed in closed form from the same c_l, h_l pieces. It is MLP-specific.
  - No bank is used, and the milestone still saves `lcg_predictor_bank.pt` as an explicit record of that: backend `exact_trace`, `tangents: None`, `bank_seed: None`.
  - Why it is the default: for this MLP it is exact and was faster than the 5-probe bank in the 2026-11-25 laptop smoke (5.5 vs 11 ms per 320 candidates). The bank's estimate had a median relative error of 2.5% against it.
- **`jvp_bank`.** Still configurable, saved and restorable:
  - Bank: η_m ~ N(0, I_P) for m = 1..M (M = 5), from `torch.Generator()` on CPU seeded with SeedSequence([seed, 33, bank_seed, round]). The tangents are v_m = h_D^{−1/2} ⊙ η_m, frozen for the round and shared by every candidate.
  - Score: r_raw = Σ_m ‖J(u) v_m‖² / (M τ²), with one `torch.func.jvp` per probe over a candidate chunk.
  - E_η[r_raw] equals the exact backend's value.
- **Numerics.** `score_dtype` is float32 or float64. `allow_tf32` (default false) controls TF32 for float32 matmuls during fitting, curvature and scoring; the trainer enables TF32 globally, and the setting is restored afterwards. A non-finite or negative score, loss, precision or bank raises an error.

## Policy reward

The divisor is frozen per round, and raw and scaled values are logged separately (`intrinsic/raw_*`, `intrinsic/scaled_*`; evaluation `intrinsic_raw_*`, `intrinsic_scaled_*`).

| Mode | Policy reward |
| --- | --- |
| `data_scaled` (default) | raw / (fixed_divisor · data_reference_steps / N_steps), where N_steps is the replay size at the refresh |
| `fixed` | raw / fixed_divisor |
| `none` | raw |

- Under `full_replay` scaling, h_D grows roughly linearly with N, so raw scores shrink as about 1/N. Data scaling is LCG's existing correction for this (`REWARD_SCALING.md`).
- The defaults (fixed_divisor 500, data_reference_steps 5,500; 2026-11-26) give a divisor of 500 at the first 100k-schedule refresh and 27.5 at 100k, about 0.4 per step in the bounded pilot. With fixed_divisor 1 (the earlier default) the scaled reward was about 200 per step and the critic could not follow.

## Lifecycle (per epoch)

1. The denoiser phase trains the denoiser.
2. `LCGPredictor.refresh`, with the denoiser only read:
   1. Compute the identity: denoiser SHA-256, replay content SHA-256 and extraction-settings SHA-256.
   2. Rebuild the feature cache and refit the standardization unless the cache has exactly this identity.
   3. Fit the MAP predictor.
   4. Compute h_D.
   5. Draw the bank (`jvp_bank` only).
   6. Fix the divisor.
   7. Record the predictor SHA-256.

   **Transactional.** The new round is built in local state: cache, standardization, predictor, h_D, bank, divisor and SHA-256. It is installed in one step only after every part has been computed. A failure at any stage (encoding, fitting, curvature or bank) raises and leaves the previous round untouched: its cache, standardization, predictor, h_D, bank, identity, divisor, counters and scores stay as they were. A later successful refresh installs a complete new round. A failure-injection test covers every stage.
3. The denoiser is put in eval mode, and the PMPO hook is set to the frozen predictor round. PMPO trains only the controller.
4. Milestones are published after PMPO. `milestone_files()` first checks that the predictor still has its recorded SHA-256.

**Seeds** (all private; global, NumPy and CUDA RNG are untouched):

| What | Seed |
| --- | --- |
| Predictor initialization and minibatches | SeedSequence([seed, 31, round]) |
| Curvature subset | SeedSequence([seed, 32, round]) |
| Bank | SeedSequence([seed, 33, bank_seed, round]) |

The round is the refresh epoch.

## Checkpoint and milestone schema

**`checkpoints/state.pt["lcg_predictor"]`** (schema 1) holds:
- `settings` (resolved config, extractor settings, action bounds, parameter layout) and `seed`;
- `predictor` (state_dict) and `predictor_sha256`;
- `mean`, `std`, `raw_std`;
- `h_D` (P), and `bank` (M × P, or None);
- `round_id`, `reward_divisor`, `replay_steps`, `num_transitions`;
- `curvature_info`, `fit_info`, `cache_identity`;
- `refreshes`, `fit_updates_total`, `seed_spec`.

On resume:
- every setting and the seed must match, and this is checked before any weights load;
- after the agent loads, the restored round's recorded denoiser SHA-256 must equal the loaded denoiser;
- the feature cache is not saved and is rebuilt deterministically at the next refresh.

Refreshes are atomic within an epoch, and checkpoints are written at epoch boundaries, so no mid-fit resume is needed or supported. An interrupted epoch refits identically from its seeds, so no optimizer state is saved.

**Milestone files** (`milestones/steps_N/`, listed in the manifest with checksums):

| File | Contents |
| --- | --- |
| `lcg_predictor.pt` | state_dict, SHA-256, architecture (input/feature/action dims, hidden dims, activation), parameter order (name, shape, numel, offset) |
| `lcg_predictor_precision.pt` | h_D, parameter order, τ², λ, curvature record |
| `lcg_predictor_bank.pt` | backend, tangents and bank seed (both None for the default `exact_trace`: an explicit record that no bank is used), num_probes, generator spec |
| `lcg_predictor_normalization.pt` | mean, std, raw_std, floor, extractor settings, probe SHA-256 |
| `lcg_predictor_scorer.pt` | resolved settings, seed, round, denoiser/replay/extraction identity, reward mode and frozen divisor, likelihood/prior, fit record, counters |

`manifest.json["lcg_predictor"]` repeats the provenance summary. Restoration uses `LCGPredictor.load_milestone_files`, or `exploration_analysis.restore_milestone_scorer("lcg-predictor", ...)`. Each file is about P floats, except the bank (M·P floats), so the Walker defaults take about 31 MB per milestone.

## Configuration

`config/intrinsic_reward/LCG_predictor.yaml` documents every key. The preset `config/experiment/lcg_predictor_100k.yaml` selects it with advantage-weighted PMPO (cap 5) and `training.compile_wm=false`, as the Plan2Explore preset does.

| Group | Keys (defaults) |
| --- | --- |
| Features | `feature_layers` [d_blocks.2, d_blocks.3], `pool_size` 4, `feature_sigma` 0.1, `probe_seed` 0, `feature_std_floor` 1e-3, `encode_batch_size` 256 |
| Predictor and fit | `hidden_dims` [256, 256], `activation` silu, `init_policy` reinitialize, `optimizer` adam, `lr` 1e-3, `adam_betas` [0.9, 0.999], `adam_eps` 1e-8, `grad_clip_norm` null, `batch_size` 256, `fit_updates` 500 |
| Bayesian model | `tau_squared` 1.0, `prior_precision` 1.0 |
| Curvature | `curvature_backend` closed_form, `curvature_num_transitions` null (full replay), `curvature_block_size` 8192, `curvature_scaling` full_replay (subset only), `curvature_chunk_size` 64, `autograd_output_chunk` 128, `autograd_sample_chunk` 4 |
| Scoring | `scorer_backend` exact_trace, `num_probes` 5 and `bank_seed` 0 (both `jvp_bank` only), `score_dtype` float32, `allow_tf32` false, `candidate_chunk_size` 1024 |
| Reward | `reward_scaling` data_scaled, `fixed_divisor` 500.0, `data_reference_steps` 5500 |

All values are working placeholders. The training budget follows Plan2Explore, and τ², λ and the reward scale are untuned.
