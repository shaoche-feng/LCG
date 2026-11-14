# Reward-free experiment analysis

## Exploration policy

### Actor

| W&B panel | Why check it |
| --- | --- |
| `actor_critic/train/core/policy_entropy` | Look for an abrupt or sustained drop. This is the first panel to check for a narrowing action distribution. |
| `actor_critic/train/core/kl_prior` | Look for a persistent jump in how far the policy moves from its prior. Check whether it happens at the same time as an entropy drop. |
| `actor_critic/train/core/actor_grad_norm` | Look for repeated spikes or a prolonged period near zero. Use the entropy and KL panels to see whether the change affected the policy. |

If those panels show an anomaly, open `actor_critic/train/core/loss_actor`. If the debug panels are available, `actor_critic/train/debug/action_std_0`, `action_std_1`, and the remaining dimensions can show which actions lost variation.

### Critic

| W&B panel | Why check it |
| --- | --- |
| `actor_critic/train/core/loss_value` | Look for persistent spikes, a sustained increase, or non-finite values. |
| `actor_critic/train/core/value_target_correlation` | Check whether predicted values track differences in the training targets. A sustained drop is worth investigating, especially if value loss rises at the same time. |
| `actor_critic/train/core/return_mean` and `actor_critic/train/core/value_mean` | Overlay them to see whether predictions drift away from the mean training target or whether both shift together. The return is from imagination and reflects the intrinsic reward used in training. |

If the critic looks unstable, open `actor_critic/train/core/value_grad_norm` and `actor_critic/train/EMA/slow_value_gap` around the same steps. These show whether the anomaly coincided with gradient spikes or a widening gap between the critic and its EMA copy.

### Denoiser

Use epoch as the x-axis for the main exploration plots.

| W&B panel | Why check it |
| --- | --- |
| `denoiser/train/loss_denoising` | Check whether training loss decreases and stays finite. Note persistent jumps around changes in the collected data. |
| `denoiser/test/loss_denoising` | Check whether test loss improves alongside training loss. If the curves diverge, verify whether the test rollouts changed. |
| `denoiser/final_test/loss_denoising` | Record the final test loss. Compare methods directly when the test data and loss settings match. |
| `denoiser/member_k/train/loss_denoising` | For the full ensemble, check members individually for persistently high or non-finite losses. |

If a curve stops early, check `denoiser/train/num_batch_train_denoiser` or `denoiser/member_k/train/num_batch_train_member` to see whether updates continued. Denoising loss measures performance on the denoising objective. Assess preservation of task-relevant behavior with rollout diagnostics.

## Collected exploration dataset

### 1. follow intrinsic reward test

Did the exploration policy collect transitions that its intrinsic reward scores highly? Freeze the scorer at a saved milestone and use it to score both the next batch from the learned policy and a batch from Uniform. Match their starting conditions where possible.

**Example.** Freeze the LCG scorer at 50k. If the next LCG batch scores higher than the Uniform batch, that’s evidence the policy collected more of what LCG rewarded at that point.

### 2. task-reward coverage test

How often did the 100k replay contain high task rewards? For each task, report the fraction of steps above a reward threshold and the fraction of episodes with at least one such step. Use the same threshold across methods for that task.

**Example.** In each Hopper replay, count high-reward steps for `stand` and `hop` separately. The same replay might contain plenty of standing experience and very little hopping experience.

### 3. task-reward lasting test

How long did high-reward behavior last? Within each episode, count consecutive steps above the same reward threshold. Report how long those bouts lasted and how often they occurred.

**Example.** Two Hopper replays might each contain 30 high-`stand`-reward steps. In one replay they form a single 30-step bout. In the other they are scattered across brief attempts. For `hop`, also count how many distinct hopping bouts occur in each episode.

## Downstream policy

For each task, compare all five methods, including Uniform, at the 100k exploration milestone. Use downstream actor–critic update count for the actor and critic curves. Verify the logged metric keys against the downstream run.

### Real-environment evaluation

| W&B panel | Why check it |
| --- | --- |
| `eval/return_mean` | Compare final real-environment task return under the same evaluation protocol. The pilot specifies 30 reset seeds (3001–3030) and up to 500 steps per episode. |
| `eval/return_median` | Check whether the typical episode supports the comparison based on mean return. Per-episode returns are logged as `eval/seed_<seed>/return`. |

`eval/return_max`, `eval/return_min` and `eval/return_std` are logged as well. These keys belong to each run's primary protocol; any other protocol is logged under `eval_protocols/protocol-<id>/`. Locally, the same values are in each task's `evaluations/eval_<task>_policy-<hash>_protocol-<hash>.json`, including every episode's return.

The downstream pilot provides an endpoint for each adapted checkpoint. Real-environment learning during adaptation remains unmeasured.

### Reward model

Use `reward_model/train/core/num_updates` as the x-axis. It counts this milestone's updates and restarts at each milestone. W&B charts default to `reward_model/train/core/cumulative_updates`, which counts across the milestones of the run; with one 100k milestone per pilot run, the two are equal.

| W&B panel | Why check it |
| --- | --- |
| `reward_model/train/core/loss_reward` | Check whether training loss decreases and stays finite. Note sustained increases or repeated spikes. |
| `reward_model/train/core/predicted_reward_mean` and `reward_model/train/core/target_reward_mean` | Overlay the means. Look for a persistent gap or predictions that stay flat while the targets shift. |

Each task's local `metrics.jsonl` holds the same values in rows with `"phase": "rew_end_model"`. Two names differ from W&B, the update counter and the loss:

| W&B key | `metrics.jsonl` key |
| --- | --- |
| `reward_model/train/core/num_updates` | `update` |
| `reward_model/train/core/loss_reward` | `loss_rew` |
| `reward_model/train/core/predicted_reward_mean` | `predicted_reward_mean` |
| `reward_model/train/core/target_reward_mean` | `target_reward_mean` |

`metrics.jsonl` also records `loss_end`, `loss_total`, `grad_norm_before_clip` and `lr`. W&B publishes the first three only with `rew_end_model.debug_metrics` on, as `reward_model/train/debug/loss_end`, `loss_total` and `grad_norm`, together with `prediction_mae`. None of the 90 pilot `metrics.jsonl` files (5 methods × 18 tasks, checked 2026-11-14) records `prediction_mae`, so debug metrics were off throughout the pilot.

A falling loss and aligned means indicate that the model is fitting the training rewards on average. Assess rare high-reward transitions with per-transition predictions, ideally on held-out episodes. Record held-out accuracy as unverified when only training metrics are available.

### Actor

| W&B panel | Why check it |
| --- | --- |
| `actor_critic/train/core/policy_entropy` | Look for abrupt or sustained narrowing of the action distribution. |
| `actor_critic/train/core/kl_prior` | Check for a sustained shift away from the policy prior around sharp entropy changes. |
| `actor_critic/train/core/actor_grad_norm` | Look for repeated spikes or a prolonged period near zero. Check whether these coincide with changes in entropy or KL. |

### Critic

| W&B panel | Why check it |
| --- | --- |
| `actor_critic/train/core/loss_value` | Look for persistent spikes or non-finite values. Read the loss alongside the scale of imagined returns. |
| `actor_critic/train/core/value_target_correlation` | Check whether predicted values track variation in the training targets. Investigate a sustained drop alongside rising value loss. |
| `actor_critic/train/core/return_mean` and `actor_critic/train/core/value_mean` | Check whether imagined task returns and predicted values move together or drift apart. Compare the imagined trend with the final real-environment result. |
| `actor_critic/train/core/value_grad_norm`, if needed | Check whether changes in loss or correlation coincide with repeated gradient spikes. |

For each method and task, record the training-health verdict as **stable**, **concerning**, or **inconclusive**, along with the imagined-return trend and the final real-environment return.
