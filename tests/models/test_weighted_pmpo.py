"""Advantage-weighted Beta PMPO experiment (agent.actor_critic.pmpo_weighting=abs_advantage)."""
import json
from copy import deepcopy

import pytest
import torch

import test_pmpo_integration as integ
from models.pmpo_beta import PMPOOptimizers, pmpo_loss
from test_pmpo_beta import make_controller
from test_pmpo_integration import make_system

W = dict(pmpo_weighting="abs_advantage")


def spec_loss(logp, adv, alpha=0.5):
    """The specified formula, written independently of pmpo_loss."""
    a = adv.detach()
    pos, neg, w = a >= 0, a < 0, a.abs()
    pos_term = (w[pos] * logp[pos]).sum() / max(int(pos.sum()), 1)
    neg_term = (w[neg] * logp[neg]).sum() / max(int(neg.sum()), 1)
    return (1 - alpha) * neg_term - alpha * pos_term


def grad_of(fn, logp):
    x = logp.clone().requires_grad_(True)
    out = fn(x)
    out.backward()
    return out.detach(), x.grad


@pytest.mark.parametrize("alpha", [0.5, 0.3])
def test_matches_specified_formula(alpha):
    torch.manual_seed(0)
    logp, adv = torch.randn(64), torch.randn(64) * 3
    loss, g = grad_of(lambda x: pmpo_loss(x, adv, alpha, "abs_advantage"), logp)
    ref, g_ref = grad_of(lambda x: spec_loss(x, adv, alpha), logp)
    torch.testing.assert_close(loss, ref)
    torch.testing.assert_close(g, g_ref)


def test_unit_magnitude_advantages_reproduce_sign_only_loss_and_gradients_exactly():
    torch.manual_seed(1)
    logp = torch.randn(64)
    adv = torch.where(torch.rand(64) < 0.4, -1.0, 1.0)
    weighted, gw = grad_of(lambda x: pmpo_loss(x, adv, 0.5, "abs_advantage"), logp)
    sign, gs = grad_of(lambda x: pmpo_loss(x, adv, 0.5, "sign"), logp)
    torch.testing.assert_close(weighted, sign, rtol=0, atol=0)
    torch.testing.assert_close(gw, gs, rtol=0, atol=0)


@pytest.mark.parametrize("c", [0.25, 3.0, 1000.0])
def test_positive_scaling_scales_loss_and_actor_gradient(c):
    torch.manual_seed(2)
    model = make_controller(**W)
    obs = torch.randn(32, 12, 8, 8)
    actions = model.sample_action(model.actor(obs))
    adv = torch.randn(32)

    def run(a):
        model.zero_grad()
        logp, _ = model.log_prob_and_entropy(model.actor(obs), actions)
        loss = pmpo_loss(logp, a, 0.5, "abs_advantage")
        loss.backward()
        return loss.detach(), [p.grad.clone() for p in model.actor.parameters()]
    base, g0 = run(adv)
    scaled, g1 = run(adv * c)
    torch.testing.assert_close(scaled, base * c, rtol=1e-5, atol=1e-6)
    for a, b in zip(g1, g0):
        torch.testing.assert_close(a, b * c, rtol=1e-4, atol=1e-6 * c)


def test_all_zero_advantages_give_zero_loss_and_gradient():
    logp = torch.randn(16)
    loss, g = grad_of(lambda x: pmpo_loss(x, torch.zeros(16), 0.5, "abs_advantage"), logp)
    assert loss.item() == 0.0 and torch.count_nonzero(g) == 0
    # Exactly-zero entries contribute nothing; denominators remain the original sample counts.
    adv = torch.tensor([0., 2., -1., 0.])
    lp = torch.tensor([5., 1., 3., -7.])
    torch.testing.assert_close(pmpo_loss(lp, adv, 0.5, "abs_advantage"),
                               torch.tensor(0.5 * (1 * 3.) / 1 - 0.5 * (2 * 1.) / 3))


@pytest.mark.parametrize("adv", [[1., 2., 3.], [-1., -2., -3.]])
def test_empty_group_contributes_zero(adv):
    logp = torch.tensor([1., 2., 4.])
    a = torch.tensor(adv)
    loss = pmpo_loss(logp, a, 0.5, "abs_advantage")
    if a[0] > 0:
        torch.testing.assert_close(loss, -0.5 * (a * logp).sum() / 3)
    else:
        torch.testing.assert_close(loss, 0.5 * (a.abs() * logp).sum() / 3)


def test_advantages_are_detached():
    logp = torch.randn(8, requires_grad=True)
    adv = torch.randn(8, requires_grad=True)
    pmpo_loss(logp, adv, 0.5, "abs_advantage").backward()
    assert adv.grad is None and logp.grad is not None


def test_unknown_weighting_rejected():
    with pytest.raises(ValueError):
        pmpo_loss(torch.zeros(2), torch.zeros(2), 0.5, "normalized")
    with pytest.raises(ValueError):
        make_controller(pmpo_weighting="normalized")


def test_default_sign_path_unchanged_and_weighted_forward_uses_weights():
    torch.manual_seed(11)
    default, _ = make_system()
    torch.manual_seed(11)
    explicit, _ = make_system(pmpo_weighting="sign")
    loss_a, m_a = default()
    loss_b, m_b = explicit()
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
    for key in m_a:
        torch.testing.assert_close(torch.as_tensor(m_a[key]), torch.as_tensor(m_b[key]), rtol=0, atol=0)
    torch.manual_seed(11)
    weighted, _ = make_system(**W)
    _, m_w = weighted()
    # Same imagination (same RNG, same weights) -> same advantages; only the raw PMPO term differs.
    for key in ("return_mean", "value_mean", "positive_fraction", "loss_kl", "loss_entropy", "advantage_mean"):
        torch.testing.assert_close(torch.as_tensor(m_w[key]), torch.as_tensor(m_a[key]), rtol=0, atol=0)
    assert float(m_w["loss_pmpo_raw"]) != float(m_a["loss_pmpo_raw"])


def test_conditional_std_metric_matches_analytic_beta_variance():
    model = make_controller(**W)
    raw = torch.randn(20, 4)
    actions = model.sample_action(raw)
    metrics = model.diagnostics(raw, actions)
    dist = model.distribution(raw)
    a, b = dist.concentration1, dist.concentration0
    width = model.action_high - model.action_low
    var = a * b / ((a + b) ** 2 * (a + b + 1)) * width ** 2
    for d in range(2):
        torch.testing.assert_close(metrics[f"policy_conditional_std_{d}"], var[:, d].mean().sqrt())


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_checkpoint_resume_bit_exact_across_prior_refresh(tmp_path, device):
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    model, _ = make_system(device, **W, prior_refresh_interval=2)
    opt = PMPOOptimizers(model, 0)
    loss, _ = model()
    loss.backward()
    opt.step()
    path = tmp_path / "pmpo.pt"
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict()), path)
    restored, _ = make_system(device, **W, prior_refresh_interval=2)
    restored_opt = PMPOOptimizers(restored, 0)
    saved = torch.load(path, map_location=device, weights_only=False)
    restored.load_state_dict(saved["model"])
    restored_opt.load_state_dict(saved["optimizer"])
    for _ in range(2):
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


def test_instrumentation_consumes_no_global_rng():
    model = make_controller(**W)
    raw = torch.randn(8, 4)
    actions = torch.zeros(8, 2) + model.action_low + 0.5
    before = torch.get_rng_state().clone()
    model.diagnostics(raw, actions)
    assert torch.equal(torch.get_rng_state(), before)


def test_trainer_world_model_pmpo_diagnostic_and_resume_with_weighted_pmpo(tmp_path, monkeypatch):
    original = integ.compose_cleanup_config
    monkeypatch.setattr(integ, "compose_cleanup_config",
                        lambda overrides=(): original([*overrides, "agent.actor_critic.pmpo_weighting=abs_advantage"]))
    integ.test_static_dataset_trainer_world_model_and_pmpo_resume(tmp_path, monkeypatch, True)
    assert "pmpo_weighting: abs_advantage" in (tmp_path / "config/trainer.yaml").read_text()
    rows = [json.loads(line) for line in (tmp_path / "pmpo_diagnostic/updates.jsonl").read_text().splitlines()]
    assert rows and all("policy_conditional_std_0" in r and "alpha_mean" in r for r in rows)
