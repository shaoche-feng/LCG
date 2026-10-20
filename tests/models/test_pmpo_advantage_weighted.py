"""Optional advantage-weighted PMPO actor objective: formula, grouping, detachment, edge
cases, cap, default sign-only behaviour and checkpoint compatibility."""
import math

import pytest
import torch

from models.pmpo_beta import actor_sample_weights, pmpo_loss
from test_pmpo_beta import make_controller


def sign_only_reference(log_prob, advantage, alpha):
    """The sign-only loss exactly as implemented before the objective setting existed."""
    positive = advantage.detach() >= 0
    negative = ~positive
    pos = log_prob.masked_fill(~positive, 0).sum() / positive.sum().clamp_min(1)
    neg = log_prob.masked_fill(~negative, 0).sum() / negative.sum().clamp_min(1)
    return (1 - alpha) * neg - alpha * pos


def weighted(log_prob, advantage, alpha=0.5, cap=5.0):
    return pmpo_loss(log_prob, advantage, alpha, "advantage_weighted", cap)


@pytest.mark.parametrize("advantages", [[2., 1., 0., -1., -3.], [1e-12, -1e-12, 0.], [5., 5., 5.], [-1., -2., -3.]])
def test_default_sign_only_is_unchanged(advantages):
    torch.manual_seed(0)
    adv = torch.tensor(advantages)
    for alpha in (0.0, 0.3, 0.5, 1.0):
        log_prob = torch.randn(len(advantages), requires_grad=True)
        expected = sign_only_reference(log_prob, adv, alpha)
        g_ref, = torch.autograd.grad(expected, log_prob)
        for loss in (pmpo_loss(log_prob, adv, alpha), pmpo_loss(log_prob, adv, alpha, "sign", 0.1)):
            assert torch.equal(loss, expected)  # bit-identical value; the cap is ignored
            g, = torch.autograd.grad(loss, log_prob)
            assert torch.equal(g, g_ref)


def test_weighted_loss_matches_a_hand_calculated_example():
    log_prob = torch.tensor([-1., -2., -3., -4., -5.], requires_grad=True)
    adv = torch.tensor([2., 1., 0., -1., -3.])
    # Positive group {2, 1, 0}: mean |A| = 1 -> w = (2, 1, 0); mean(w logp) = (-2 - 2 + 0) / 3.
    # Negative group {-1, -3}: mean |A| = 2 -> w = (0.5, 1.5); mean(w logp) = (-2 - 7.5) / 2.
    torch.testing.assert_close(actor_sample_weights(adv), torch.tensor([2., 1., 0., 0.5, 1.5]), rtol=0, atol=0)
    loss = weighted(log_prob, adv, alpha=0.5)
    expected = 0.5 * (-9.5 / 2) - 0.5 * (-4 / 3)
    assert loss.item() == pytest.approx(expected, rel=1e-6)
    # Gradient: positive i -> -alpha w_i / n_pos; negative i -> (1 - alpha) w_i / n_neg.
    loss.backward()
    torch.testing.assert_close(log_prob.grad, torch.tensor([-0.5 * 2 / 3, -0.5 / 3, 0., 0.5 * 0.5 / 2, 0.5 * 1.5 / 2]))
    # alpha weighting of the two groups is kept.
    assert weighted(log_prob.detach(), adv, alpha=0.2).item() == pytest.approx(0.8 * (-9.5 / 2) - 0.2 * (-4 / 3), rel=1e-6)


def test_larger_magnitudes_get_larger_weights_within_each_group_until_capped():
    adv = torch.tensor([0.1, 0.5, 1., 2., 4., -0.2, -1., -3.])
    w = actor_sample_weights(adv)
    pos, neg = w[:5], w[5:]
    assert (pos[1:] > pos[:-1]).all() and (neg[1:] > neg[:-1]).all()  # strictly increasing below the cap
    # Groups are normalized separately: same relative size -> same weight in either group.
    w2 = actor_sample_weights(torch.tensor([1., 3., -10., -30.]))
    torch.testing.assert_close(w2[:2], w2[2:], rtol=0, atol=0)
    # Above the cap: equal weight, no renormalization.
    adv = torch.tensor([1.] * 30 + [100., 300.])  # mean 12.19: 100 and 300 exceed 5x the mean
    w = actor_sample_weights(adv, weight_cap=5.0)
    scale = adv.mean()
    assert (w[-2:] == 5.0).all()
    torch.testing.assert_close(w[:30], (1 / scale).expand(30), rtol=0, atol=0)
    assert w.mean() < 1  # clipping removes weight; nothing is redistributed


@pytest.mark.parametrize("cap", [1.0, 2.0, 5.0, 100.0])
def test_cap_is_applied_exactly(cap):
    adv = torch.tensor([0.01, 0.1, 1., 10., 100., -0.01, -100.])
    w = actor_sample_weights(adv, weight_cap=cap)
    uncapped = actor_sample_weights(adv, weight_cap=math.inf)
    torch.testing.assert_close(w, uncapped.clamp(max=cap), rtol=0, atol=0)
    assert w.max() <= cap


def test_advantages_and_weights_are_detached():
    log_prob = torch.randn(6, requires_grad=True)
    adv = torch.tensor([3., 1., 0.5, -0.5, -2., -4.], requires_grad=True)
    loss = weighted(log_prob, adv)
    loss.backward()
    assert adv.grad is None and log_prob.grad is not None
    assert not actor_sample_weights(adv).requires_grad


@pytest.mark.parametrize("advantages", [[1., 2., 3.], [-1., -2., -3.], [0., 0., 0.], [1e-12, 3e-12, -2e-12],
                                        [1e-30, 0., -1e-30], [1e6, -1e6, 1e-6]])
def test_empty_groups_zero_and_tiny_advantages_stay_finite(advantages):
    log_prob = torch.tensor([-1., -2., -3.], requires_grad=True)
    adv = torch.tensor(advantages)
    loss = weighted(log_prob, adv)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(log_prob.grad).all()
    w = actor_sample_weights(adv)
    assert torch.isfinite(w).all() and (w >= 0).all() and (w <= 5).all()
    if all(a == 0 for a in advantages):
        assert loss.item() == 0 and (log_prob.grad == 0).all()  # all-zero advantages contribute zero
    if all(a > 0 for a in advantages):  # empty negative group contributes zero
        assert loss.item() == pytest.approx(-0.5 * (w * log_prob.detach()).mean().item())


@pytest.mark.parametrize("kwargs,match", [
    (dict(actor_objective="weighted"), "actor_objective must be one of"),
    (dict(actor_objective="advantage_weighted", advantage_weight_cap=0.0), "advantage_weight_cap"),
    (dict(actor_objective="advantage_weighted", advantage_weight_cap=-5.0), "advantage_weight_cap"),
    (dict(actor_objective="advantage_weighted", advantage_weight_cap=math.inf), "advantage_weight_cap"),
    (dict(actor_objective="advantage_weighted", advantage_weight_cap=math.nan), "advantage_weight_cap"),
])
def test_invalid_objective_settings_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_controller(**kwargs)


def test_objective_survives_checkpoints_and_mismatches_are_rejected():
    torch.manual_seed(0)
    model = make_controller(actor_objective="advantage_weighted", advantage_weight_cap=5.0)
    state = model.state_dict()
    assert model.objective_settings() == dict(mode="advantage_weighted", weight_cap=5.0)
    make_controller(actor_objective="advantage_weighted", advantage_weight_cap=5).load_state_dict(state)  # same
    for other in (dict(), dict(actor_objective="advantage_weighted", advantage_weight_cap=3.0)):
        with pytest.raises(ValueError, match="different objective"):
            make_controller(**other).load_state_dict(state)
    # Sign-only states (including ones saved before the setting existed) load into sign-only only.
    sign_state = make_controller().state_dict()
    old = {k: v for k, v in sign_state.items()}
    extra = next(k for k in old if k.endswith("_extra_state"))
    old[extra] = {k: v for k, v in old[extra].items() if k != "objective"}
    make_controller().load_state_dict(old)
    make_controller(advantage_weight_cap=2.0).load_state_dict(sign_state)  # cap is irrelevant to sign-only
    with pytest.raises(ValueError, match="different objective"):
        make_controller(actor_objective="advantage_weighted").load_state_dict(old)
