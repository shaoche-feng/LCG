import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest


def actor_critic_schema(action_dim, debug, slow_critic=True):
    """The agreed exploration actor_critic/train/ keys of the Gaussian controller (paths below
    actor_critic/train/): 43 by default and 148 with debug metrics for Walker's 6 action
    dimensions with the slow EMA critic. Written out from the agreed list, not from the code."""
    dims, stats = range(action_dim), ("mean", "std", "min", "max", "p10", "median", "p90")
    keys = [f"core/{k}" for k in ("loss_actor", "loss_value", "kl_prior", "policy_entropy", "return_mean", "value_mean",
                                  "value_target_correlation", "positive_fraction", "actor_grad_norm", "value_grad_norm",
                                  "num_batch_train_actor_critic")]
    keys += [f"EMA/{k}" for k in ("loss_value_target", "loss_slowreg", "slow_value_gap")] if slow_critic else []
    keys += [f"gaussian/pretanh_{p}_{s}" for p in ("mean", "std") for s in stats]
    keys += ["gaussian/pretanh_std_near_floor_fraction", "gaussian/pretanh_mean_abs_gt_3_fraction",
             "gaussian/tanh_inverse_limited_fraction"] + [f"gaussian/policy_conditional_std_{d}" for d in dims]
    keys += ["weights/clipped_fraction_positive", "weights/clipped_fraction_negative"]
    keys += [f"intrinsic/{kind}_reward_{s}" for kind in ("raw", "scaled") for s in ("mean", "max")]
    if debug:
        debug_keys = ["loss_pmpo_raw", "loss_positive", "loss_negative", "loss_kl", "loss_entropy",
                      "log_probability", "positive_log_probability", "negative_log_probability",
                      "value_std", "value_bias", "value_abs_max", "return_std", "return_abs_max"]
        debug_keys += ["slow_value_mean"] if slow_critic else []
        debug_keys += [f"advantage_{s}" for s in stats]
        debug_keys += ["advantage_zero_fraction", "advantage_near_zero_fraction", "negative_fraction"]
        debug_keys += [f"advantage_abs_{s}_{g}" for s in ("mean", "max") for g in ("positive", "negative")]
        debug_keys += [f"weight_{s}_{g}" for s in ("mean", "max", "std") for g in ("positive", "negative")]
        debug_keys += ["kl_initial", "actor_rel_distance_from_init"]
        debug_keys += [f"action_{s}_{d}" for s in ("mean", "std") for d in dims]
        debug_keys += [f"policy_{s}_{d}" for s in ("mean", "mean_state_std", "mean_min", "mean_max") for d in dims]
        debug_keys += [f"policy_entropy_dim_{d}" for d in dims] + ["policy_entropy_std", "near_lower_fraction",
                                                                    "near_upper_fraction"]
        debug_keys += [f"fixed_policy_mean_state_std_{d}" for d in dims]
        debug_keys += ["fixed_temporal_difference_mean", "fixed_temporal_variation_fraction",
                       "fixed_ordered_repeated_action_difference", "fixed_ordered_reversed_action_difference",
                       "stack_temporal_difference_mean", "stack_temporal_variation_fraction"]
        debug_keys += [f"intrinsic_{kind}_reward_{s}" for kind in ("raw", "scaled")
                       for s in ("std", "min", "p10", "median", "p90", "p99")]
        keys += [f"debug/{k}" for k in debug_keys]
    assert len(keys) == len(set(keys))
    return set(keys)


@pytest.fixture
def ac_schema():
    return actor_critic_schema
