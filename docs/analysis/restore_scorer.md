# Restore intrinsic reward scorers

This document supports the **intrinsic reward following test** in `docs/analysis/reward_free_experiment_analysis.md`. At each selected milestone, save the scorer used to train the exploration policy that collects the next batch. Use the same frozen scorer to score that batch and the corresponding batch from the separate Uniform run.

For example, use the **50k Walker Plan2Explore scorer** on the **50k–50.5k** batches from the Plan2Explore and Uniform runs. Recover both batches from the existing replay files. The pilot audit verified that this window is episode 100, rows 0–499.

Paths below are relative to the run directory and use the 5,500-step milestone as an example.

## Exploration policy

| What | Where to save it |
| --- | --- |
| The exploration policy’s inference weights and buffers, including any encoder needed for action selection. Save the policy after training and before it collects the next batch. This supports an optional matched-start test. | `milestones/steps_005500/exploration_policy.pt` |

Use the run’s resolved policy configuration to load the policy. Measure one export to estimate the storage cost for 20 milestones.

## Full ensemble

| What | Where to save it |
| --- | --- |
| All trained ensemble members. | Already saved in the milestone. |

## Plan2Explore

| What | Where to save it |
| --- | --- |
| The complete `state_dict()` of each of the five trained MLP predictor heads, in member order. | `milestones/steps_005500/MLP_member_1.pt` through `milestones/steps_005500/MLP_member_5.pt` |
| The fitted normalization means and scales used when computing disagreement, if the scoring path reads them. Verify the exact tensors in `plan2explore.py`. | `milestones/steps_005500/plan2explore_normalization.pt` |

Use the run’s resolved config for predictor architecture, feature extraction, and reward scaling.

## Laplace ensemble

| What | Where to save it |
| --- | --- |
| The actual sampled parameter bank, in member order and keyed by full parameter name. Preserve the representation used by the scorer, whether it stores sampled values or additive perturbations. | `milestones/steps_005500/parameter_bank.pt` |

## LCG

| What | Where to save it |
| --- | --- |
| The historical precision tensors read by the scorer, in their existing representation and parameter order. | `milestones/steps_005500/historical_precision.pt` |
| The cached JVP probes, if the scorer uses a fixed bank. Preserve their order. | `milestones/steps_005500/jvp_probe_bank.pt` |
| The fitted scaling state for the configured mode, if the scorer uses any. For data scaling, save the actual divisor or statistics read by the scorer. | `milestones/steps_005500/lcg_scaling.pt` |

Use the **existing scaling config** to select the mode. Set it to **data scaling** for the Walker and Hopper reruns. The resolved config records the mode, and the milestone saves any fitted values needed to reproduce the scaled reward. Verify the config key and the exact tensors or scalars read by the scorer before implementing the save.

## LCG predictor

| What | Where it is saved |
| --- | --- |
| The MAP predictor's `state_dict()`, its architecture, its SHA-256 and the flat parameter order (names, shapes, offsets) used by every tensor below. | `milestones/steps_005500/lcg_predictor.pt` |
| The historical diagonal precision `h_D` (flat, in that order), `tau_squared`, `prior_precision` and the curvature record (replay size, sampling rule, number of transitions used and seed, scaling factor; by default the full replay, seed None). | `milestones/steps_005500/lcg_predictor_precision.pt` |
| With the default `scorer_backend: exact_trace`: an explicit record that no bank is used (`tangents` and `bank_seed` are None). With `jvp_bank`: the actual JVP bank used for scoring, `v_m = h_D^{-1/2} * eta_m` (rows in probe order), with its seed. | `milestones/steps_005500/lcg_predictor_bank.pt` |
| The feature standardization (mean, floored std) and the feature-extraction settings, including the probe SHA-256. | `milestones/steps_005500/lcg_predictor_normalization.pt` |
| The resolved settings, the run seed, the round id, the denoiser/replay/extraction identity, the frozen reward divisor and its replay size, and the fit record. | `milestones/steps_005500/lcg_predictor_scorer.pt` |

`LCGPredictor.load_milestone_files` restores these files without refitting anything or drawing a new bank. It refuses files from different rounds, a different parameter order, probe or bank seed, and a denoiser whose weights differ from the recorded identity. The policy's action bounds (`exploration_policy.pt`) give the canonical actions. `exploration_analysis.restore_milestone_scorer("lcg-predictor", ...)` uses this path.

## Uniform

Use the existing Uniform replay from the corresponding collection window. Uniform has no learned scorer or policy weights.

## Verify restoration

Score a fixed batch with the active scorer, then reload it in a fresh process and score the same batch again. Use identical observation history, actions, and any sampled scoring inputs. Compare **raw scores per transition** and **scaled rewards per transition** when scaling is enabled.
