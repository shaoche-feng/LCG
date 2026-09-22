# K=4 implementation and pre-run protocol

The single-frame prototype was reverified (107 tests passed) and preserved in
commit `f6148d2a8a7ed9f1093f5a194497e18152554786` before any temporal changes.

## Exact temporal state

`FrameHistory` belongs to each `make_env_loop` coroutine, not to any neural
network or to DIAMOND. It builds `(B,12,H,W)` channel-concatenated RGB states in
oldest-to-newest order. The separate actor/value encoders and frozen actor prior
all consume that exact representation. No learned recurrent state is added.

Reset initializes `[o0,o0,o0,o0]`; advances produce `[o0,o0,o0,o1]`,
`[o0,o0,o1,o2]`, `[o0,o1,o2,o3]`, then `[o1,o2,o3,o4]`. Every advancement
allocates new storage. A batched episode reset affects only the corresponding
rows, replacing them with four copies of the new initial observation.

Before installing an automatic reset observation, the loop constructs the
successor stack using `info['final_observation']`. True termination masks its
value in returns. Environment/artificial-horizon truncation bootstraps that
pre-reset successor and ends the trace. A coroutine rollout cut does not reset
history: its final bootstrap sees the next rolling stack, and the next send
continues that stack. PMPO's existing fresh-imagination-block boundaries remain
unchanged; each newly reset imagined episode starts with repeated `o0`, not the
world model's independent reward-model burn-in frames.

Imagination calls `make_env_loop(..., store_policy_observations=True)` so the
exact action-generation stack is stored for likelihood/value replay. Real
collectors leave that flag false, retaining only the current RGB observation in
DIAMOND's dataset even though the policy sees K=4. Fixed imagined observations
in learner checkpoints now store twelve channels. Old single-frame checkpoint
weights are intentionally architecture-incompatible; no implicit conversion.

All existing Beta/PMPO/KL/value/optimizer/default-budget settings are unchanged.
The only architecture change is the first encoder convolution's input channel
count. Tests retain exact CPU/CUDA learner-continuation comparisons.

## Opt-in run instrumentation

`+pmpo_diagnostic=bounded` installs an observer in the existing `Trainer`; it is
not a second trainer. `max_epochs` caps the total epoch count at 20 independently
of DIAMOND's collection-plus-final-epochs formula. Every PMPO update is written
to `pmpo_diagnostic/updates.jsonl`; each epoch to `epochs.jsonl`. Losses and all
requested distribution/value/advantage/group/KL/gradient/temporal metrics are
recorded. Epoch training summaries average update statistics, except global
alpha/beta minima and maxima. Thus p10/p90 and correlations in the epoch row are
means of per-update statistics, not pooled-trajectory estimates. Exact fixed-real
state summaries are separately prefixed `real_fixed_`.

Primary evaluation uses deterministic Beta means on the same four full episodes
with seeds `[1001,1002,1003,1004]` each epoch. It uses the existing DMControlEnv
adapter and the same FrameHistory helper. It restores Python/NumPy/Torch RNG state
afterward, does not supply real rewards to PMPO, and does not add evaluation data
to training. Up to 128 K=4 states from the first evaluation are retained unchanged
for ordered/repeated-last/reversed-stack probes. Evaluation-distribution fields
`real_fixed_action_*` describe deterministic mean actions, while the unprefixed
training `action_*` fields describe sampled actions.

Opting into this diagnostic replaces the inherited extra final evaluation batch
with the final epoch's fixed-seed evaluation. The usual evaluation collector can
still be selected separately with `evaluation.should`; it is disabled in the
bounded diagnostic to avoid a second changing-seed primary comparison.

## Frozen health-stop definitions

These are diagnostic stop rules, not learning changes. Non-finite controller or
world-model loss/gradient/parameter diagnostics stop immediately. Other rules:

- Any Beta concentration above 10000.
- KL above 10 nats per state, summed across action dimensions.
- Value or lambda-return absolute magnitude above 1e6.
- More than 95% sampled components within 1% of a bound for 50 consecutive updates.
- Maximum fixed-state action std across dimensions below 1e-6 for 100 updates.
- Canonical differential entropy per dimension below -5 nats, or a one-update
  fall exceeding 2 nats with a mean concentration above 100.

Differential entropy can validly be negative, so merely crossing zero is not a
stop. Mean concentrations above 100, near-floor fractions, and at least three
epochs with a minority advantage fraction below 1% are reported for diagnosis;
they do not automatically change or normalize the loss. Checkpoint/runtime
exceptions stop the run, retain the previous normal checkpoint, and attempt to
save failed-state weights and the exception. No run resumes automatically.

## Intended normal Walker diagnostic

Use standard 64x64 RGB, action repeat 2, no shortened episode limit, four-frame
DIAMOND conditioning, horizon 15, and unchanged normal networks. The project's
existing `LCG/outputs/phase1/walker_walk/config/trainer.yaml` supplies the validated
normal experiment convention: initial 5000 real transitions, 500 more per
collection epoch up to 10000 total, denoiser batches 32, reward/end batches 8,
controller batches 32, first-epoch updates 10000/10000/5000, subsequent updates
400/400/400, and compilation disabled. PMPO uses its own unchanged optimizer and
loss settings. These are run overrides; no default budgets are edited.

Use training seed 0. Explicitly disable LCG for this downstream task-reward
controller diagnostic, preserving the previous PMPO smoke reward composition.
Checkpoint every epoch and retain all epochs. The cap permits ten collection
epochs and ten subsequent epochs if numerically healthy. Whole-experiment resume
fidelity remains outside this change. Exact commands, commits, smoke results,
epoch records, and final assessment will accompany the run report.
