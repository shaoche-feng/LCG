"""Slow EMA critic regularization (agent.actor_critic.slow_critic_coef > 0) for Gaussian-PMPO. CPU only."""
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from models.pmpo_beta import PMPOOptimizers
from test_pmpo_beta import make_controller
from test_pmpo_integration import make_system

G = dict(actor_distribution="tanh_gaussian")
E = dict(actor_distribution="tanh_gaussian", slow_critic_coef=1.0)


def test_disabled_creates_no_slow_critic_and_keeps_state_dict():
    torch.manual_seed(0)
    base = make_controller(**G)
    torch.manual_seed(0)
    off = make_controller(**G, slow_critic_coef=0.0)
    assert not hasattr(off, "slow_value")
    assert list(base.state_dict()) == list(off.state_dict())
    for k, v in base.state_dict().items():
        if torch.is_tensor(v):
            assert torch.equal(v, off.state_dict()[k]), k


def test_invalid_settings_rejected():
    with pytest.raises(ValueError, match="slow critic"):
        make_controller(**G, slow_critic_coef=-1.0)
    with pytest.raises(ValueError, match="slow critic"):
        make_controller(**G, slow_critic_coef=1.0, slow_critic_rate=0.0)


def test_slow_critic_exact_copy_no_grad_and_outside_optimizer():
    torch.manual_seed(0)
    model = make_controller(**E)
    for (k, s), (_, v) in zip(model.slow_value.state_dict().items(), model.value.state_dict().items()):
        assert torch.equal(s, v), k
    assert not any(p.requires_grad for p in model.slow_value.parameters())
    opt = PMPOOptimizers(model, warmup_steps=0)
    optimized = {id(p) for g in opt.actor.param_groups + opt.value.param_groups for p in g["params"]}
    assert not optimized & {id(p) for p in model.slow_value.parameters()}


def test_update_rule_loss_formula_and_gradient_paths():
    torch.set_num_threads(1)
    model, env = make_system(**E)
    opt = PMPOOptimizers(model, warmup_steps=0)
    captured = {}
    collect = model.collect_imagination

    def spy():
        out = collect()
        captured["obs"] = out[0]
        return out
    model.collect_imagination = spy
    with torch.no_grad():  # make the slow critic differ so the regularizer is non-trivial
        for p in model.slow_value.parameters():
            p.add_(0.05 * torch.randn_like(p))
    slow_before = deepcopy(model.slow_value.state_dict())
    loss, metrics = model()
    flat = captured["obs"].flatten(0, 1)
    with torch.no_grad():
        v = model.value(flat).squeeze(-1)
        vs = model.slow_value(flat).squeeze(-1)
    assert float(metrics["loss_slowreg"]) == pytest.approx(float(F.mse_loss(v, vs)), rel=1e-6)
    assert float(metrics["loss_value"]) == pytest.approx(float(metrics["loss_value_target"]) + 1.0 * float(metrics["loss_slowreg"]), rel=1e-6)
    loss.backward()
    assert all(p.grad is None for p in model.slow_value.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.value.parameters())
    opt.step()
    for k, online in model.value.state_dict().items():
        torch.testing.assert_close(model.slow_value.state_dict()[k], 0.98 * slow_before[k] + 0.02 * online, rtol=0, atol=1e-7)


def test_online_critic_used_for_bootstrap_and_advantage():
    torch.set_num_threads(1)
    model, env = make_system(**E)
    with torch.no_grad():
        for p in model.slow_value.parameters():
            p.add_(1.0)  # very different slow critic
    obs, act, rew, end, trunc, _, values, bootstrap, infos = model.collect_imagination()
    with torch.no_grad():
        online = model.value(obs.flatten(0, 1)).squeeze(-1).view_as(values)
        slow = model.slow_value(obs.flatten(0, 1)).squeeze(-1).view_as(values)
    torch.testing.assert_close(values, online, rtol=0, atol=1e-6)  # batched recomputation: float rounding only
    assert not torch.allclose(values, slow)
    # the rollout's next-state bootstraps are online-critic values of the following stored states
    torch.testing.assert_close(bootstrap[:, :-1][~(end + trunc).bool()[:, :-1]],
                               online[:, 1:][~(end + trunc).bool()[:, :-1]], rtol=0, atol=1e-6)


def test_checkpoint_round_trip_preserves_slow_critic():
    torch.set_num_threads(1)
    model, env = make_system(**E)
    opt = PMPOOptimizers(model, warmup_steps=5)
    for _ in range(2):
        opt.zero_grad()
        loss, _ = model()
        loss.backward()
        opt.step()
    state = deepcopy(dict(model=model.state_dict(), opt=opt.state_dict()))
    assert any(k.startswith("slow_value.") for k in state["model"])
    opt.zero_grad()
    loss_a, _ = model()
    loss_a.backward()
    opt.step()
    after = deepcopy(model.state_dict())
    model2, env2 = make_system(**E)
    torch.manual_seed(99)
    model2.load_state_dict(state["model"])
    opt2 = PMPOOptimizers(model2, warmup_steps=5)
    opt2.load_state_dict(state["opt"])
    env2.sampler.denoiser.load_state_dict(env.sampler.denoiser.state_dict())
    env2.rew_end_model.load_state_dict(env.rew_end_model.state_dict())
    opt2.zero_grad()
    loss_b, _ = model2()
    loss_b.backward()
    opt2.step()
    assert loss_a.item() == loss_b.item()
    for k, v in after.items():
        if torch.is_tensor(v):
            assert torch.equal(v, model2.state_dict()[k]), k
