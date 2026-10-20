"""Gaussian-PMPO experiment (agent.actor_critic.actor_distribution=tanh_gaussian)."""
import json
from copy import deepcopy

import pytest
import torch
from torch.distributions import Normal, TransformedDistribution, kl_divergence
from torch.distributions.transforms import AffineTransform, TanhTransform

import test_pmpo_integration as integ
from models.pmpo_beta import GAUSSIAN_STD_MIN, GAUSSIAN_STD_SPAN, PMPOOptimizers, pmpo_loss
from test_pmpo_beta import make_controller
from test_pmpo_integration import make_system

G = dict(actor_distribution="tanh_gaussian")


def reference(model, raw):
    """Independent reference: torch's own transformed distribution, per dimension."""
    dist = model.distribution(raw)
    width = model.action_high - model.action_low
    # cache_size=1: scoring a tensor this distribution just sampled reuses the exact pre-image.
    return TransformedDistribution(dist, [TanhTransform(cache_size=1),
                                          AffineTransform(model.action_low + width / 2, width / 2, cache_size=1)])


def test_parameterization_bounds_and_deterministic_rule():
    torch.manual_seed(0)
    model = make_controller(**G)
    raw = torch.randn(64, 4) * 3
    dist = model.distribution(raw)
    mu, raw_std = raw.chunk(2, dim=-1)
    torch.testing.assert_close(dist.mean, mu)
    torch.testing.assert_close(dist.stddev, GAUSSIAN_STD_MIN + GAUSSIAN_STD_SPAN * torch.sigmoid(raw_std))
    assert (dist.stddev > GAUSSIAN_STD_MIN).all() and (dist.stddev < GAUSSIAN_STD_MIN + GAUSSIAN_STD_SPAN).all()
    actions = model.sample_action(raw)
    assert actions.dtype == torch.float32 and actions.shape == (64, 2) and not actions.requires_grad
    assert ((actions >= model.action_low) & (actions <= model.action_high)).all()
    det = model.sample_action(raw, deterministic=True)
    width = model.action_high - model.action_low
    torch.testing.assert_close(det, model.action_low + (torch.tanh(mu) + 1) / 2 * width)
    assert torch.equal(det, model.sample_action(raw, deterministic=True))
    # Saturating means still give in-bounds (possibly exactly boundary) actions.
    huge = torch.tensor([[50., -50., 0., 0.]])
    sat = model.sample_action(huge, deterministic=True)
    assert ((sat >= model.action_low) & (sat <= model.action_high)).all()


def test_log_prob_matches_independent_transformed_distribution():
    torch.manual_seed(1)
    model = make_controller(**G)
    raw = torch.randn(256, 4) * torch.tensor([1.5, 1.5, 2., 2.])
    actions = model.sample_action(raw)
    logp, _ = model.log_prob_and_entropy(raw, actions)
    expected = reference(model, raw).log_prob(actions).sum(-1)
    canonical = 2 * (actions - model.action_low) / (model.action_high - model.action_low) - 1
    safe = (canonical.abs() < 0.999).all(-1)  # away from float32 inversion limits
    assert safe.sum() > 100
    torch.testing.assert_close(logp[safe], expected[safe], rtol=1e-4, atol=1e-3)


def test_log_prob_and_gradients_finite_at_and_near_action_limits():
    model = make_controller(**G)
    low, high = model.action_low, model.action_high
    width = high - low
    actions = torch.stack([low, high, low + 1e-6 * width, high - 1e-6 * width, low + 1e-3 * width, high - 1e-3 * width])
    for scale in (0., 5., -12.):
        raw = torch.zeros(len(actions), 4)
        raw[:, :2] = scale
        raw.requires_grad_(True)
        logp, entropy = model.log_prob_and_entropy(raw, actions)
        assert torch.isfinite(logp).all() and torch.isfinite(entropy).all()
        (logp.sum() + entropy.sum()).backward()
        assert torch.isfinite(raw.grad).all()
    with pytest.raises(ValueError):
        model.log_prob_and_entropy(torch.zeros(1, 4), (high + 1e-3).unsqueeze(0))


def test_transformed_entropy_matches_monte_carlo_and_has_gradient():
    torch.manual_seed(2)
    model = make_controller(**G)
    raw = torch.tensor([[0., 0., -2., 2.], [1.5, -0.5, 1., -1.], [3., 0.2, 0., 3.]], requires_grad=True)
    _, entropy = model.log_prob_and_entropy(raw, model.sample_action(raw, deterministic=True))
    ref = reference(model, raw.detach())
    samples = ref.sample((200_000,))
    mc = -ref.log_prob(samples).sum(-1).mean(0)
    torch.testing.assert_close(entropy.detach(), mc, rtol=0, atol=0.02)
    entropy.sum().backward()
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0
    # Wider pre-tanh spread around 0 means larger entropy of the environment action.
    narrow = model.entropy_per_dim(model.distribution(torch.tensor([[0., 0., -3., -3.]])))
    wide = model.entropy_per_dim(model.distribution(torch.tensor([[0., 0., 3., 3.]])))
    assert (wide > narrow).all()


def test_kl_analytic_zero_for_identical_and_frozen_prior():
    model = make_controller(**G)
    obs = torch.randn(4, 12, 8, 8)
    prior = model.distribution(model.prior_actor(obs))
    current = model.distribution(model.actor(obs))
    same = kl_divergence(current, prior).sum(-1)
    torch.testing.assert_close(same, torch.zeros_like(same), atol=1e-6, rtol=0)
    with torch.no_grad():
        model.actor.head[-1].bias[0] += 1
        model.actor.head[-1].bias[3] -= 1
    current = model.distribution(model.actor(obs))
    kl = kl_divergence(current, prior).sum(-1)
    p, q = current, prior
    closed = (torch.log(q.stddev / p.stddev) + (p.stddev ** 2 + (p.mean - q.mean) ** 2) / (2 * q.stddev ** 2) - 0.5).sum(-1)
    torch.testing.assert_close(kl, closed, rtol=1e-5, atol=1e-6)
    assert (kl > 0).all()
    kl.mean().backward()
    assert all(p.grad is None and not p.requires_grad for p in model.prior_actor.parameters())


def test_full_update_prior_refresh_and_sign_only_objective():
    model, _ = make_system(**G, prior_refresh_interval=2)
    opt = PMPOOptimizers(model, 0)
    original = deepcopy(model.prior_actor.state_dict())
    for _ in range(2):
        opt.zero_grad()
        loss, metrics = model()
        assert all(torch.isfinite(torch.as_tensor(v)).all() for v in metrics.values())
        assert "pretanh_std_mean" in metrics and "alpha_mean" not in metrics
        assert all(f"policy_conditional_std_{d}" in metrics for d in range(2))
        for key, p in model.prior_actor.state_dict().items():
            torch.testing.assert_close(p, original[key], rtol=0, atol=0)
        loss.backward()
        opt.step()
    expected = deepcopy(model.actor.state_dict())
    _, metrics = model()
    for key, p in model.prior_actor.state_dict().items():
        torch.testing.assert_close(p, expected[key], rtol=0, atol=0)
    assert float(metrics["kl_prior"]) == pytest.approx(0.0, abs=1e-6)
    # The raw PMPO term is still the sign-only loss: invariant to advantage magnitudes.
    logp = torch.randn(10)
    adv = torch.randn(10)
    torch.testing.assert_close(pmpo_loss(logp, adv), pmpo_loss(logp, adv * 7.0))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_checkpoint_resume_bit_exact_across_prior_refresh(tmp_path, device):
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    model, _ = make_system(device, **G, prior_refresh_interval=2)
    opt = PMPOOptimizers(model, 0)
    loss, _ = model()
    loss.backward()
    opt.step()
    path = tmp_path / "pmpo.pt"
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict()), path)
    restored, _ = make_system(device, **G, prior_refresh_interval=2)
    restored_opt = PMPOOptimizers(restored, 0)
    saved = torch.load(path, map_location=device, weights_only=False)
    assert not any(k.startswith("_gh_") for k in saved["model"])  # quadrature buffers not persisted
    restored.load_state_dict(saved["model"])
    restored_opt.load_state_dict(saved["optimizer"])
    for _ in range(2):  # crosses a prior-refresh boundary
        opt.zero_grad(); restored_opt.zero_grad()
        loss, metrics = model()
        actual, actual_metrics = restored()
        torch.testing.assert_close(actual, loss, rtol=0, atol=0)
        for key in metrics:
            torch.testing.assert_close(torch.as_tensor(actual_metrics[key]), torch.as_tensor(metrics[key]), rtol=0, atol=0)
        loss.backward(); actual.backward()
        opt.step(); restored_opt.step()
        for p, q in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        for (k, p), q in zip(model.prior_actor.state_dict().items(), restored.prior_actor.state_dict().values()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_instrumentation_and_imagination_consume_no_global_rng():
    model, _ = make_system(**G)
    before = torch.get_rng_state().clone()
    rollout = model.collect_imagination()
    assert torch.equal(torch.get_rng_state(), before)
    obs, act = rollout[0].flatten(0, 1), rollout[1].flatten(0, 1)
    raw = model.actor(obs).detach()
    logp, entropy = model.log_prob_and_entropy(raw, act)
    model.diagnostics(raw, act, logp, entropy)
    model.entropy_per_dim(model.distribution(raw))
    assert torch.equal(torch.get_rng_state(), before)


def test_trainer_world_model_pmpo_diagnostic_and_resume_with_gaussian_actor(tmp_path, monkeypatch):
    original = integ.compose_cleanup_config
    monkeypatch.setattr(integ, "compose_cleanup_config",
                        lambda overrides=(): original([*overrides, "agent.actor_critic.actor_distribution=tanh_gaussian"]))
    integ.test_static_dataset_trainer_world_model_and_pmpo_resume(tmp_path, monkeypatch, True)
    assert "actor_distribution: tanh_gaussian" in (tmp_path / "config/trainer.yaml").read_text()
    rows = [json.loads(line) for line in (tmp_path / "pmpo_diagnostic/updates.jsonl").read_text().splitlines()]
    assert rows and all("pretanh_std_mean" in r and "alpha_mean" not in r for r in rows)
