"""Slow EMA critic regularization (agent.actor_critic.slow_critic_coef > 0) for sign-only Beta-PMPO. CPU only.
Ported from the Gaussian EMA tests; the actor is the default Beta actor throughout."""
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from models.pmpo_beta import PMPOOptimizers
from test_pmpo_beta import make_controller
from test_pmpo_integration import make_system

E = dict(slow_critic_coef=1.0)


def train_steps(model, opt, n):
    out = []
    for _ in range(n):
        opt.zero_grad()
        loss, metrics = model()
        loss.backward()
        opt.step()
        out.append((loss.item(), metrics))
    return out


def test_disabled_creates_no_slow_critic_and_keeps_state_dict():
    torch.manual_seed(0)
    base = make_controller()
    torch.manual_seed(0)
    off = make_controller(slow_critic_coef=0.0)
    assert not hasattr(off, "slow_value")
    assert list(base.state_dict()) == list(off.state_dict())
    for k, v in base.state_dict().items():
        if torch.is_tensor(v):
            assert torch.equal(v, off.state_dict()[k]), k


def test_disabled_training_is_bit_identical_to_default():
    torch.set_num_threads(1)
    model_a, _ = make_system()
    model_b, _ = make_system(slow_critic_coef=0.0, slow_critic_rate=0.5)
    out_a = train_steps(model_a, PMPOOptimizers(model_a, warmup_steps=2), 3)
    out_b = train_steps(model_b, PMPOOptimizers(model_b, warmup_steps=2), 3)
    assert [l for l, _ in out_a] == [l for l, _ in out_b]
    assert set(out_a[-1][1]) == set(out_b[-1][1]) and "loss_slowreg" not in out_b[-1][1]
    for k, v in model_a.state_dict().items():
        if torch.is_tensor(v):
            assert torch.equal(v, model_b.state_dict()[k]), k


def test_invalid_settings_rejected():
    with pytest.raises(ValueError, match="slow critic"):
        make_controller(slow_critic_coef=-1.0)
    with pytest.raises(ValueError, match="slow critic"):
        make_controller(slow_critic_coef=1.0, slow_critic_rate=0.0)


def test_beta_actor_unchanged_by_slow_critic():
    torch.manual_seed(0)
    off = make_controller()
    torch.manual_seed(0)
    on = make_controller(**E)
    for k, v in off.state_dict().items():
        if torch.is_tensor(v):
            assert torch.equal(v, on.state_dict()[k]), k  # same actor/critic/prior initialization


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


def test_slow_update_happens_after_critic_step_once_per_update():
    torch.set_num_threads(1)
    model, env = make_system(**E)
    opt = PMPOOptimizers(model, warmup_steps=0)
    order = []
    value_step, update = opt.value.step, model.update_slow_critic
    opt.value.step = lambda *a, **k: (order.append("value_step"), value_step(*a, **k))[1]
    model.update_slow_critic = lambda: (order.append("slow_update"), update())[1]
    slow_before = deepcopy(model.slow_value.state_dict())
    loss, _ = model()
    assert order == [] and all(torch.equal(v, model.slow_value.state_dict()[k]) for k, v in slow_before.items())
    loss.backward()
    assert order == []  # backward alone does not move the slow critic
    opt.step()
    assert order == ["value_step", "slow_update"]


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
    train_steps(model, opt, 2)
    state = deepcopy(dict(model=model.state_dict(), opt=opt.state_dict()))
    assert any(k.startswith("slow_value.") for k in state["model"])
    loss_a = train_steps(model, opt, 1)[0][0]
    after = deepcopy(model.state_dict())
    model2, env2 = make_system(**E)
    torch.manual_seed(99)
    model2.load_state_dict(state["model"])
    opt2 = PMPOOptimizers(model2, warmup_steps=5)
    opt2.load_state_dict(state["opt"])
    env2.sampler.denoiser.load_state_dict(env.sampler.denoiser.state_dict())
    env2.rew_end_model.load_state_dict(env.rew_end_model.state_dict())
    loss_b = train_steps(model2, opt2, 1)[0][0]
    assert loss_a == loss_b
    for k, v in after.items():
        if torch.is_tensor(v):
            assert torch.equal(v, model2.state_dict()[k]), k
