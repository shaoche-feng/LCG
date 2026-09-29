import pytest
import torch
from torch.distributions import kl_divergence

from models.pmpo_beta import PMPOBeta, PMPOBetaConfig, pmpo_loss
from models.actor_critic import compute_lambda_returns


def make_controller(**kwargs):
    cfg = dict(lstm_dim=1, img_channels=3, img_size=8, channels=[8], down=[0],
               continuous_action_dim=2, action_low=[-3., 0.5], action_high=[2., 7.],
               actor_hidden_dims=(16,), value_hidden_dims=(16,), imagination_horizon=3)
    cfg.update(kwargs)
    return PMPOBeta(PMPOBetaConfig(**cfg))


def test_beta_actions_density_entropy_and_gradients():
    torch.manual_seed(3)
    model = make_controller()
    obs = torch.randn(32, 12, 8, 8)
    raw = model.actor(obs)
    dist = model.distribution(raw)
    assert torch.isfinite(dist.concentration1).all() and (dist.concentration1 > 0).all()
    assert torch.isfinite(dist.concentration0).all() and (dist.concentration0 > 0).all()
    canonical = 2 * dist.sample() - 1
    assert ((canonical >= -1) & (canonical <= 1)).all()
    actions = model.sample_action(raw)
    assert actions.dtype == torch.float32 and actions.shape == (32, 2)
    assert not actions.requires_grad
    assert ((actions >= model.action_low) & (actions <= model.action_high)).all()
    logp, entropy = model.log_prob_and_entropy(raw, actions)
    assert torch.isfinite(logp).all() and torch.isfinite(entropy).all()
    expected = (dist.log_prob((actions - model.action_low) / (model.action_high - model.action_low))
                - (model.action_high - model.action_low).log()).sum(-1)
    torch.testing.assert_close(logp, expected)
    (-logp.mean()).backward()
    assert sum(p.grad.abs().sum() for p in model.actor.parameters()) > 0
    assert all(p.grad is None for p in model.value.parameters())
    torch.testing.assert_close(model.sample_action(raw, True), model.to_environment(dist.mean))
    assert model.sample_action(raw, True).std(0).min() > 0


@pytest.mark.parametrize("advantage", [1., -1.])
def test_likelihood_moves_in_correct_direction(advantage):
    model = make_controller()
    raw = torch.nn.Parameter(torch.zeros(1, 4))
    action = model.to_environment(torch.tensor([[0.2, 0.7]]))
    opt = torch.optim.SGD([raw], lr=0.05)
    before = model.log_prob_and_entropy(raw, action)[0]
    pmpo_loss(before, torch.tensor([advantage])).backward()
    opt.step()
    after = model.log_prob_and_entropy(raw, action)[0]
    assert (after - before).item() * advantage > 0


@pytest.mark.parametrize("advantages", [[1, 2, 3], [-1, -2, -3], [-1, 0, 1], [-1e-12, 0, 1e-12]])
def test_pmpo_groups(advantages):
    logp = torch.tensor([1., 2., 4.], requires_grad=True)
    adv = torch.tensor(advantages)
    loss = pmpo_loss(logp, adv, 0.3)
    pos, neg = adv >= 0, adv < 0
    expected = (0.7 * logp[neg].mean() if neg.any() else 0) - (0.3 * logp[pos].mean() if pos.any() else 0)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logp.grad).all()
    torch.testing.assert_close(loss, pmpo_loss(logp, adv * 100, 0.3))


def test_beta_kl_and_frozen_prior():
    model = make_controller()
    obs = torch.randn(4, 12, 8, 8)
    prior = model.distribution(model.prior_actor(obs))
    same = kl_divergence(model.distribution(model.actor(obs)), prior).sum(-1)
    torch.testing.assert_close(same, torch.zeros_like(same), atol=1e-6, rtol=0)
    with torch.no_grad():
        model.actor.head[-1].bias[0] += 1
    changed = kl_divergence(model.distribution(model.actor(obs)), prior).sum(-1)
    assert torch.isfinite(changed).all() and (changed > 0).all()
    changed.mean().backward()
    assert all(p.grad is None and not p.requires_grad for p in model.prior_actor.parameters())


def test_initial_actor_separate_from_prior_and_never_refreshed():
    model = make_controller(prior_refresh_interval=1)  # refreshes prior_actor every update
    obs = torch.randn(4, 12, 8, 8)
    initial_before = {k: v.clone() for k, v in model.initial_actor.state_dict().items()}
    same = kl_divergence(model.distribution(model.actor(obs)), model.distribution(model.initial_actor(obs))).sum(-1)
    torch.testing.assert_close(same, torch.zeros_like(same), atol=1e-6, rtol=0)
    with torch.no_grad():
        model.actor.head[-1].bias[0] += 1
    # prior_actor refreshes to match actor (interval=1); initial_actor must not.
    model.prior_actor.load_state_dict(model.actor.state_dict())
    for key, p in model.initial_actor.state_dict().items():
        torch.testing.assert_close(p, initial_before[key], rtol=0, atol=0)
    assert all(not p.requires_grad for p in model.initial_actor.parameters())
    changed = kl_divergence(model.distribution(model.actor(obs)), model.distribution(model.initial_actor(obs))).sum(-1)
    assert torch.isfinite(changed).all() and (changed > 0).all()


@pytest.mark.parametrize("end,trunc,gamma,lam,expected", [
    ([0, 0], [0, 0], 1., 1., [13., 12.]),
    ([1, 0], [0, 0], 1., 1., [1., 12.]),
    ([0, 0], [1, 0], 1., 1., [6., 12.]),
    ([0, 0], [0, 0], 1., 0., [6., 12.]),
    ([0, 0], [0, 0], 0.5, 0.5, [4., 7.]),
    ([0, 0], [0, 0], 0., 0.95, [1., 2.]),
])
def test_lambda_returns(end, trunc, gamma, lam, expected):
    actual = compute_lambda_returns(torch.tensor([[1., 2.]]), torch.tensor([end]), torch.tensor([trunc]),
                                    torch.tensor([[5., 10.]]), gamma, lam, continuous_reward=True)
    torch.testing.assert_close(actual, torch.tensor([expected]))


def test_real_environment_training_rejected():
    with pytest.raises(ValueError, match="WorldModelEnv"):
        make_controller().setup_training(object())


def test_nonfinite_fails_fast():
    with pytest.raises(FloatingPointError):
        pmpo_loss(torch.tensor([float("nan")]), torch.tensor([1.]))


def test_optimizer_ownership_and_value_gradient_isolation():
    from models.pmpo_beta import PMPOOptimizers
    model = make_controller()
    opt = PMPOOptimizers(model, 0)
    actor_ids = {id(p) for g in opt.actor.param_groups for p in g["params"]}
    value_ids = {id(p) for g in opt.value.param_groups for p in g["params"]}
    prior_ids = {id(p) for p in model.prior_actor.parameters()}
    assert not actor_ids & value_ids and not (actor_ids | value_ids) & prior_ids
    assert actor_ids | value_ids == {id(p) for p in model.parameters() if p.requires_grad}
    model.value(torch.randn(2, 12, 8, 8)).square().mean().backward()
    assert all(p.grad is None for p in model.actor.parameters())
    assert all(p.grad is None for p in model.prior_actor.parameters())
    assert model.value.encoder.encoder[0].weight.grad.abs().sum() > 0
