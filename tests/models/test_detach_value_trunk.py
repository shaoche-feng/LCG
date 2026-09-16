"""Regression tests for the `detach_value_trunk` experimental flag (causal-isolation study for
the epoch-20->30 walker/walk return regression -- see the actor_critic.py docstring on
ActorCriticConfig.detach_value_trunk). CPU-only, fast, no GPU or checkpoint required.
"""
import copy

import torch

from models.actor_critic import ActorCritic, ActorCriticConfig


def make_ac(detach_value_trunk=False, action_dim=2, lstm_dim=8, img_size=8):
    cfg = ActorCriticConfig(
        lstm_dim=lstm_dim, img_channels=3, img_size=img_size, channels=[4], down=[0],
        continuous_action_dim=action_dim, action_low=[-1.0] * action_dim, action_high=[1.0] * action_dim,
        detach_value_trunk=detach_value_trunk,
    )
    return ActorCritic(cfg)


def _same_init(ac_control, ac_treatment):
    """Copies control's (randomly-initialized-by-test) weights onto treatment so both start
    from literally the same parameters -- isolates the flag as the only difference."""
    ac_treatment.load_state_dict(copy.deepcopy(ac_control.state_dict()))


def _random_encoder_init(ac):
    """actor_critic.py zero-inits actor_linear/critic_linear/lstm-bias by design (see
    __init__): with critic_linear.weight == 0, d(val)/d(hx) is identically zero, so the value
    loss's gradient into hx (and hence encoder/lstm) would be trivially zero regardless of
    detach_value_trunk -- and likewise for actor_linear.weight == 0 zeroing out mean/log_std's
    gradient into hx. Randomize encoder, lstm, AND both actor_linear/critic_linear so that
    trunk-reaching-or-not is actually discriminating in these tests, not a init-time accident."""
    torch.manual_seed(0)
    for p in ac.encoder.parameters():
        p.data.normal_(0, 0.1)
    for name, p in ac.lstm.named_parameters():
        if "weight" in name:
            p.data.normal_(0, 0.1)
    ac.actor_linear.weight.data.normal_(0, 0.1)
    ac.critic_linear.weight.data.normal_(0, 0.1)


def _forward(ac, obs, hx, cx):
    logits_act, val, (hx, cx) = ac.predict_act_value(obs, (hx, cx))
    action, z = ac.sample_action(logits_act, deterministic=True)
    return logits_act, val, action, (hx, cx)


# ---------------------------------------------------------------------------------------------
# 1. Control and treatment produce identical actions and values before training.
# ---------------------------------------------------------------------------------------------

def test_control_and_treatment_numerically_identical_before_training():
    torch.manual_seed(1)
    ac_control = make_ac(detach_value_trunk=False)
    _random_encoder_init(ac_control)
    ac_treatment = make_ac(detach_value_trunk=True)
    _same_init(ac_control, ac_treatment)

    obs = torch.randn(3, 3, 8, 8)
    hx = torch.zeros(3, ac_control.lstm_dim)
    cx = torch.zeros(3, ac_control.lstm_dim)

    logits_c, val_c, action_c, _ = _forward(ac_control, obs, hx, cx)
    logits_t, val_t, action_t, _ = _forward(ac_treatment, obs, hx, cx)

    assert torch.equal(logits_c, logits_t), "dist_params (mean, log_std) must be bit-identical"
    assert torch.equal(val_c, val_t), "detach() must not change the value's numeric output"
    assert torch.equal(action_c, action_t)


# ---------------------------------------------------------------------------------------------
# 2 & 3. In treatment: value loss gives zero encoder/LSTM gradient, but still updates
#         critic_linear.
# ---------------------------------------------------------------------------------------------

def test_treatment_value_loss_zero_trunk_grad_nonzero_critic_grad():
    torch.manual_seed(2)
    ac = make_ac(detach_value_trunk=True)
    _random_encoder_init(ac)

    obs = torch.randn(4, 3, 8, 8)
    hx = torch.zeros(4, ac.lstm_dim)
    cx = torch.zeros(4, ac.lstm_dim)
    _, val, _, _ = _forward(ac, obs, hx, cx)

    target = torch.randn(4)
    loss_values = torch.nn.functional.mse_loss(val, target)
    ac.zero_grad(set_to_none=True)
    loss_values.backward()

    for p in ac.encoder.parameters():
        assert p.grad is None or torch.all(p.grad == 0), "encoder must get zero gradient from value loss when detached"
    for p in ac.lstm.parameters():
        assert p.grad is None or torch.all(p.grad == 0), "lstm must get zero gradient from value loss when detached"

    assert ac.critic_linear.weight.grad is not None
    assert ac.critic_linear.weight.grad.norm().item() > 0, "critic_linear must still be updated by the value loss"
    assert ac.critic_linear.bias.grad is not None
    assert ac.critic_linear.bias.grad.norm().item() > 0


# ---------------------------------------------------------------------------------------------
# 4. Policy and entropy losses still update the shared trunk normally, even with
#    detach_value_trunk=True (only the VALUE path is detached).
# ---------------------------------------------------------------------------------------------

def test_treatment_policy_and_entropy_losses_still_reach_trunk():
    torch.manual_seed(3)
    ac = make_ac(detach_value_trunk=True)
    _random_encoder_init(ac)

    obs = torch.randn(4, 3, 8, 8)
    hx = torch.zeros(4, ac.lstm_dim)
    cx = torch.zeros(4, ac.lstm_dim)
    logits_act, val, (hx2, cx2) = ac.predict_act_value(obs, (hx, cx))
    action, z = ac.sample_action(logits_act, deterministic=False)

    log_prob, entropy_per_sample = ac.log_prob_and_entropy(logits_act, action, z)
    advantage = torch.full((4,), 1.5)
    loss_actions = (-log_prob * advantage).mean()

    ac.zero_grad(set_to_none=True)
    loss_actions.backward(retain_graph=True)
    assert any(p.grad is not None and p.grad.norm().item() > 0 for p in ac.encoder.parameters()), \
        "policy loss must still reach the encoder even when detach_value_trunk=True"
    assert any(p.grad is not None and p.grad.norm().item() > 0 for p in ac.lstm.parameters()), \
        "policy loss must still reach the lstm even when detach_value_trunk=True"

    ac.zero_grad(set_to_none=True)
    entropy = entropy_per_sample.mean()
    entropy.backward()
    assert any(p.grad is not None and p.grad.norm().item() > 0 for p in ac.encoder.parameters()), \
        "entropy loss must still reach the encoder even when detach_value_trunk=True"
    assert any(p.grad is not None and p.grad.norm().item() > 0 for p in ac.lstm.parameters()), \
        "entropy loss must still reach the lstm even when detach_value_trunk=True"


# ---------------------------------------------------------------------------------------------
# 5. detach_value_trunk=False exactly reproduces current (pre-flag) behavior: value loss DOES
#    reach the shared trunk, matching the architecture before this flag was introduced.
# ---------------------------------------------------------------------------------------------

def test_control_value_loss_still_reaches_trunk_as_before():
    torch.manual_seed(4)
    ac = make_ac(detach_value_trunk=False)
    _random_encoder_init(ac)

    obs = torch.randn(4, 3, 8, 8)
    hx = torch.zeros(4, ac.lstm_dim)
    cx = torch.zeros(4, ac.lstm_dim)
    _, val, _, _ = _forward(ac, obs, hx, cx)

    target = torch.randn(4)
    loss_values = torch.nn.functional.mse_loss(val, target)
    ac.zero_grad(set_to_none=True)
    loss_values.backward()

    assert any(p.grad is not None and p.grad.norm().item() > 0 for p in ac.encoder.parameters()), \
        "with the flag off, value loss must still reach the encoder (unchanged prior behavior)"
    assert any(p.grad is not None and p.grad.norm().item() > 0 for p in ac.lstm.parameters()), \
        "with the flag off, value loss must still reach the lstm (unchanged prior behavior)"


def test_default_config_value_is_false():
    from dataclasses import fields
    default = next(f.default for f in fields(ActorCriticConfig) if f.name == "detach_value_trunk")
    assert default is False


# ---------------------------------------------------------------------------------------------
# 6. Checkpoint parameter shapes (and key set) are unaffected by the flag -- the flag adds no
#    new parameters, so a checkpoint trained with one setting loads cleanly under the other.
# ---------------------------------------------------------------------------------------------

def test_checkpoint_shapes_unaffected_by_flag():
    ac_false = make_ac(detach_value_trunk=False)
    ac_true = make_ac(detach_value_trunk=True)
    sd_false = ac_false.state_dict()
    sd_true = ac_true.state_dict()
    assert set(sd_false.keys()) == set(sd_true.keys())
    for k in sd_false:
        assert sd_false[k].shape == sd_true[k].shape, f"shape mismatch at {k}"

    # A checkpoint saved under one setting must load cleanly under the other (the whole point
    # of the flag being a pure forward/gradient-path switch, not an architecture change).
    ac_true.load_state_dict(sd_false)
    ac_false.load_state_dict(sd_true)
