# AGENTS.md

Context for any AI coding agent (Claude Code, Codex, etc.) working in this repo.
This is a PMPO-Beta actor-critic controller built on a DIAMOND-style world model,
evaluated on dm_control Walker Walk. Keep this file in sync with reality — if a
fact here turns out stale, fix it rather than leaving it wrong.

## Environment

- Repo root (canonical, only full checkout): `~/scfeng/LCG-pmpo-beta-k4-20260923`
  on host `kingplus0` (2x RTX 5090, 32GB each). Reachable via `ssh kingplus0`
  (jump-host chain through `bujo0`; ask the user for `~/.ssh/config` if it's not
  already set up in your environment).
- Conda env: `lcg`, Python at `/home/bigking/anaconda3/envs/lcg/bin/python`.
  Bare `python` is NOT on PATH over non-interactive SSH — always use the full
  path, or pass it via the launch scripts' `PYTHON=` env var.
- GPU isolation gotcha: `common.devices=N` sets `CUDA_VISIBLE_DEVICES=N` correctly,
  but `MUJOCO_GL=egl` (dm_control's headless rendering) does its own GPU
  enumeration and defaults to device 0 regardless. Always also set
  `MUJOCO_EGL_DEVICE_ID=N` to the same device on launch.
- Convention on kingplus0: all work stays under `~/scfeng/`, never at the
  server's home-directory top level (other users share this machine).
- wandb: project `lcg-controller-baselines`, entity
  `models-national-tsing-hua-university7947`. The one-off zero-shot-adaptation
  diagnostic used a separate project, `LCG-zero-shot-adaptation`.

## Known-good baseline config (PMPO-Beta, Walker Walk)

K=4 (imagination samples), sign-only PMPO, `alpha_pmpo=0.5`, `beta_kl=0.3`,
`imagination_horizon=15` (`H=15`), `collection.train.steps_per_epoch=500`,
`collection.train.num_steps_total=100000`, seed 0, fresh initialization.
Evaluation: 30-seed deterministic eval (seeds 3001-3030) is the authoritative
metric; the in-training 4-seed "eval4" (seeds 1001-1004) is a cheap, noisy
proxy — don't trust it for real comparisons.

## Experiment results so far

| Run | Change from baseline | Final 30-seed median | frac > 500 | Notes |
|---|---|---|---|---|
| Stage 3 | `entropy_coef=0.01` | 595.8 | 83% | Best result of the family. Non-monotonic across milestones (25k/50k/75k/100k: 92/525/224/596). |
| Stage 3B | `entropy_coef=0.02` | 490.4 | 47% | Worse than Stage 3 — entropy effect is non-monotonic, not "more is better." |
| Refresh=250 | `prior_refresh_interval=500→250` (from Stage 3) | 210.9 | 3% | Much worse. Contradicts a within-block KL diagnostic that suggested 500 wasn't constraining movement. **Single seed per config on both sides** — could partly be trajectory luck, not a clean causal effect. A second seed per setting would resolve this. |
| Zero-shot adaptation | Freeze Stage 3's final world model, train a fresh PMPO actor-critic through imagination only, zero real-env training steps | Bimodal outcome; frac>500 plateaus at ~0.47-0.50 from ~10k updates onward and never improves through 100k | ~50% | Never matches Stage 3's real-data-trained 83%. Recommended adaptation budget: ~10k updates — more doesn't help, and the median is unreliable near the bimodal boundary (use frac>500 instead). |

**Diagnostic findings on the Stage 3 gait:**
- The dive/recover walker gait is a genuine, reward-permitted behavior (dm_control's
  `walker_walk` reward is multiplicative `stand_reward * move_reward`, which legitimizes
  it), not primarily a world-model hallucination — though a small WM bias for
  near-stationary states was separately quantified.
- It does NOT alternate legs across cycles (checked directly, not assumed).
- Real action-probability barriers to the "other leg" action exist at some gait
  phases but not others (phase-dependent), and even where a barrier exists, the
  frozen world model predicts no return benefit to crossing it — evidence against
  "exploration barrier" as a *sufficient* explanation for the gait on its own.

## Checkpoint / disk policy

Training saves a dense per-epoch and per-prior-refresh-interval checkpoint history by
default (`src/pmpo_diagnostic.py`) — this gets large fast (~60GB per 100k-step run at
the default 500-update interval, more at shorter intervals). **Do not let this
accumulate.** After training (or periodically during long runs), prune to: the
checkpoint nearest each 10k-real-step milestone (10k, 20k, ..., 100k) + the true
final checkpoint + `fixed_real_stacks.pt` (small diagnostic reference file, always
keep). Never touch `updates.jsonl`, `epochs.jsonl`, `checkpoints.jsonl`,
`.hydra/`, the dataset directory, or `checkpoints/state.pt` (Trainer's own resume
checkpoint) — those are small and needed for reproducibility/resumability.

## Open items (as of 2026-09-29)

- No second training seed yet for either refresh-interval setting — the
  refresh=250 result should be treated as suggestive, not conclusive.
- Refresh=250's own `updates.jsonl` hasn't been run through the within-prior-block
  KL analysis that motivated the ablation in the first place.
- Refresh=250's outputs haven't been downloaded off kingplus0 (Stage 3 and
  Stage 3B were, byte-verified).

## Evaluation protocol reminder

When comparing any two runs, confirm the resolved config differs in *exactly* the
intended field before trusting a comparison (`diff` the Hydra `overrides.yaml` or
launch-script `LAUNCH_CMD` arrays). Report that diff before drawing conclusions,
not after.
