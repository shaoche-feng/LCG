"""Two-hot (DreamerV3-style, 255-bin) critic with EMA regularization for Gaussian-PMPO. CPU only."""
import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

import models.pmpo_beta as pmpo_module
from models.pmpo_beta import PMPOOptimizers
from models.twohot import symexp, symexp_bins, symlog, twohot_encode, twohot_loss, twohot_mean
from test_pmpo_beta import make_controller
from test_pmpo_integration import make_system

G = dict(actor_distribution="tanh_gaussian")
S = dict(actor_distribution="tanh_gaussian", slow_critic_coef=1.0)            # scalar EMA (h10ema5bc11033)
T = dict(actor_distribution="tanh_gaussian", slow_critic_coef=1.0, value_head="twohot")
DREAMER = Path("/home/bigking/scfeng/LCG-stage3-dreamerv3-controller/src/models/dreamerv3_controller.py")


def train_steps(model, opt, n):
    out = []
    for _ in range(n):
        opt.zero_grad()
        loss, metrics = model()
        loss.backward()
        opt.step()
        out.append((loss.item(), metrics))
    return out


# ---------------------------------------------------------------- reference building blocks

def test_bins_construction_and_boundaries():
    bins = symexp_bins(255)
    assert bins.shape == (255,) and bins.dtype == torch.float32
    assert torch.all(bins[1:] > bins[:-1])
    assert bins[127] == 0 and torch.equal(bins, -bins.flip(0))
    torch.testing.assert_close(bins[-1], symexp(torch.tensor(20.0)))
    torch.testing.assert_close(symlog(bins), torch.cat([torch.linspace(-20, 0, 128), torch.linspace(0, 20, 128)[1:]]),
                               rtol=0, atol=2e-5)


def test_encoding_interpolates_hits_bins_and_clamps_out_of_range():
    bins = symexp_bins(255)
    t = torch.tensor([0.0, 0.37, -3.2, 12.5, 250.0, float(bins[140]), 1e9, -1e9])
    enc = twohot_encode(t, bins)
    assert torch.all(enc >= 0) and torch.allclose(enc.sum(-1), torch.ones(len(t)))
    assert ((enc > 0).sum(-1) <= 2).all()
    inside = slice(0, 6)
    torch.testing.assert_close((enc[inside] * bins).sum(-1), t[inside], rtol=1e-5, atol=1e-5)  # linear interpolation
    assert enc[0, 127] == 1 and enc[5, 140] == 1  # exact bin hits are one-hot
    assert enc[6, -1] == 1 and enc[7, 0] == 1  # out of range: all weight on the edge bin
    nz = enc[1].nonzero().flatten()
    assert len(nz) == 2 and nz[1] - nz[0] == 1 and bins[nz[0]] <= t[1] <= bins[nz[1]]


def test_decoding_uniform_is_zero_and_peaked_recovers_bin():
    bins = symexp_bins(255)
    assert twohot_mean(torch.zeros(3, 255), bins).abs().max() == 0
    logits = torch.full((255,), -1e4)
    logits[150] = 0.0
    torch.testing.assert_close(twohot_mean(logits, bins), bins[150])
    target = torch.tensor([4.2, -0.5])
    enc = twohot_encode(target, bins)
    torch.testing.assert_close(twohot_mean(torch.log(enc.clamp_min(1e-30)), bins), target, rtol=1e-4, atol=1e-4)
    loss = twohot_loss(torch.zeros(2, 255), target, bins)
    torch.testing.assert_close(loss, torch.full((2,), float(torch.log(torch.tensor(255.0)))))


@pytest.mark.skipif(not DREAMER.exists(), reason="reviewed Dreamer controller worktree not present")
def test_matches_reviewed_dreamer_reference():
    src = DREAMER.read_text()
    ns = {}
    start = src.index("def symlog(x):")
    end = src.index("def imagination_weights")
    exec("import torch\nfrom torch import nn\n" + src[start:end], ns)  # the pure functions only
    g = torch.Generator().manual_seed(0)
    for n in (255, 7, 8):
        torch.testing.assert_close(symexp_bins(n), ns["symexp_bins"](n), rtol=0, atol=0)
    bins = symexp_bins(255)
    t = torch.randn(64, generator=g) * 50
    t[:3] = torch.tensor([0.0, 1e10, -1e10])
    logits = torch.randn(64, 255, generator=g)
    torch.testing.assert_close(twohot_encode(t, bins), ns["twohot_encode"](t, bins), rtol=0, atol=0)
    torch.testing.assert_close(twohot_loss(logits, t, bins), ns["twohot_loss"](logits, t, bins), rtol=0, atol=0)
    torch.testing.assert_close(twohot_mean(logits, bins), ns["twohot_mean"](logits, bins), rtol=0, atol=0)


# ---------------------------------------------------------------- construction / initialization

def test_scalar_default_unchanged_and_invalid_rejected():
    torch.manual_seed(0)
    a = make_controller(**S)
    torch.manual_seed(0)
    b = make_controller(**S, value_head="scalar")
    assert list(a.state_dict()) == list(b.state_dict()) and not hasattr(b, "value_bin_locations")
    for k, v in a.state_dict().items():
        if torch.is_tensor(v):
            assert torch.equal(v, b.state_dict()[k]), k
    with pytest.raises(ValueError, match="critic head"):
        make_controller(**G, value_head="quantile")
    with pytest.raises(ValueError, match="critic head"):
        make_controller(**G, value_head="twohot", value_bins=2)


def test_twohot_zero_output_init_and_exact_zero_initial_value():
    torch.manual_seed(0)
    m = make_controller(**T)
    assert m.value.head[-1].weight.shape[0] == 255
    assert torch.count_nonzero(m.value.head[-1].weight) == 0 and torch.count_nonzero(m.value.head[-1].bias) == 0
    v = m.predict_act_value(torch.randn(16, 12, 8, 8))[1]
    assert v.shape == (16,) and v.abs().max() == 0
    for (k, s), (_, o) in zip(m.slow_value.state_dict().items(), m.value.state_dict().items()):
        assert torch.equal(s, o), k  # exact copy
    assert not any(p.requires_grad for p in m.slow_value.parameters())
    opt = PMPOOptimizers(m, warmup_steps=0)
    optimized = {id(p) for g in opt.actor.param_groups + opt.value.param_groups for p in g["params"]}
    assert not optimized & {id(p) for p in m.slow_value.parameters()}


def test_critic_head_does_not_change_actor_trunk_or_global_rng():
    torch.manual_seed(5)
    sc = make_controller(**S)
    rng_scalar = torch.random.get_rng_state()
    torch.manual_seed(5)
    th = make_controller(**T)
    assert torch.equal(rng_scalar, torch.random.get_rng_state())  # downstream construction/sampling unchanged
    for k, v in sc.state_dict().items():
        if not torch.is_tensor(v) or k.startswith("value.head.") and k.split(".")[2] == str(len(th.value.head) - 1):
            continue
        if k.startswith("slow_value.head.") and k.split(".")[2] == str(len(th.value.head) - 1):
            continue
        assert torch.equal(v, th.state_dict()[k]), k  # actor, prior, initial actor, critic trunk identical


def test_world_and_reward_model_init_unchanged():
    sm, senv = make_system(**S)  # make_system builds the controller first, then the denoiser and reward model
    tm, tenv = make_system(**T)
    for a, b in ((senv.sampler.denoiser, tenv.sampler.denoiser), (senv.rew_end_model, tenv.rew_end_model)):
        for (k, x), (_, y) in zip(a.state_dict().items(), b.state_dict().items()):
            assert torch.equal(x, y), k
    for k, v in sm.actor.state_dict().items():
        assert torch.equal(v, tm.actor.state_dict()[k]), k


# ---------------------------------------------------------------- learning path

def capture(model, monkeypatch):
    rec = {}
    collect, lam = model.collect_imagination, pmpo_module.compute_lambda_returns

    def spy_collect():
        out = collect()
        rec["obs"] = out[0]
        return out

    def spy_lambda(*a, **k):
        rec["targets"] = lam(*a, **k)
        return rec["targets"]
    model.collect_imagination = spy_collect
    monkeypatch.setattr(pmpo_module, "compute_lambda_returns", spy_lambda)
    return rec


def test_losses_formula_finite_gradients_and_detached_targets(monkeypatch):
    torch.set_num_threads(1)
    model, env = make_system(**T)
    with torch.no_grad():  # make online and slow critics non-trivial and different
        for p in model.value.parameters():
            p.add_(0.05 * torch.randn_like(p))
        for p in model.slow_value.parameters():
            p.add_(0.05 * torch.randn_like(p))
    rec = capture(model, monkeypatch)
    loss, metrics = model()
    targets, flat = rec["targets"], rec["obs"].flatten(0, 1)
    assert not targets.requires_grad
    bins = model.value_bin_locations
    with torch.no_grad():
        logits = model.value(flat)
        slow = twohot_mean(model.slow_value(flat), bins)
        values = twohot_mean(logits, bins)
    tgt_ce = twohot_loss(logits, targets.flatten(), bins).mean()
    slow_ce = twohot_loss(logits, slow, bins).mean()
    assert float(metrics["loss_value_target"]) == pytest.approx(float(tgt_ce), rel=1e-6)
    assert float(metrics["loss_slowreg"]) == pytest.approx(float(slow_ce), rel=1e-6)
    assert float(metrics["loss_value"]) == pytest.approx(float(tgt_ce + 1.0 * slow_ce), rel=1e-6)
    err = values - targets.flatten()
    assert float(metrics["value_target_abs_error"]) == pytest.approx(float(err.abs().mean()), rel=1e-5)
    assert float(metrics["value_target_rmse"]) == pytest.approx(float(err.square().mean().sqrt()), rel=1e-5)
    assert float(metrics["target_clip_fraction"]) == 0 and float(metrics["slow_target_clip_fraction"]) == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None for p in model.slow_value.parameters())
    vgrads = [p.grad for p in model.value.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in vgrads) and any(g.abs().sum() > 0 for g in vgrads)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.actor.parameters())
    # the critic gradient equals that of the CE losses with targets and slow values as constants
    model.zero_grad()
    (twohot_loss(model.value(flat), targets.flatten(), bins).mean()
     + twohot_loss(model.value(flat), slow, bins).mean()).backward()
    for g, p in zip(vgrads, model.value.parameters()):
        torch.testing.assert_close(g, p.grad, rtol=1e-5, atol=1e-7)


def test_ema_update_after_critic_step_once_per_update():
    torch.set_num_threads(1)
    model, env = make_system(**T)
    opt = PMPOOptimizers(model, warmup_steps=0)
    order = []
    value_step, update = opt.value.step, model.update_slow_critic
    opt.value.step = lambda *a, **k: (order.append("value_step"), value_step(*a, **k))[1]
    model.update_slow_critic = lambda: (order.append("slow_update"), update())[1]
    before = deepcopy(model.slow_value.state_dict())
    loss, _ = model()
    loss.backward()
    assert order == [] and all(torch.equal(v, model.slow_value.state_dict()[k]) for k, v in before.items())
    opt.step()
    assert order == ["value_step", "slow_update"]
    for k, online in model.value.state_dict().items():
        torch.testing.assert_close(model.slow_value.state_dict()[k], 0.98 * before[k] + 0.02 * online, rtol=0, atol=1e-7)


def test_online_critic_decoded_for_bootstrap_and_advantage():
    torch.set_num_threads(1)
    model, env = make_system(**T)
    with torch.no_grad():
        for p in model.value.parameters():
            p.add_(0.05 * torch.randn_like(p))
        for p in model.slow_value.parameters():
            p.add_(1.0)
    obs, act, rew, end, trunc, _, values, bootstrap, infos = model.collect_imagination()
    bins = model.value_bin_locations
    with torch.no_grad():
        online = twohot_mean(model.value(obs.flatten(0, 1)), bins).view_as(values)
        slow = twohot_mean(model.slow_value(obs.flatten(0, 1)), bins).view_as(values)
    torch.testing.assert_close(values, online, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(values, slow)
    keep = ~(end + trunc).bool()[:, :-1]
    torch.testing.assert_close(bootstrap[:, :-1][keep], online[:, 1:][keep], rtol=1e-5, atol=1e-5)


def test_checkpoint_round_trip_twohot():
    torch.set_num_threads(1)
    model, env = make_system(**T)
    opt = PMPOOptimizers(model, warmup_steps=5)
    train_steps(model, opt, 2)
    state = deepcopy(dict(model=model.state_dict(), opt=opt.state_dict()))
    assert any(k.startswith("slow_value.") for k in state["model"]) and "value_bin_locations" not in state["model"]
    loss_a = train_steps(model, opt, 1)[0][0]
    after = deepcopy(model.state_dict())
    model2, env2 = make_system(**T)
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


def test_scalar_mode_training_bit_identical_to_explicit_scalar():
    torch.set_num_threads(1)
    a, _ = make_system(**S)
    b, _ = make_system(**S, value_head="scalar", value_bins=7)
    out_a = train_steps(a, PMPOOptimizers(a, warmup_steps=2), 3)
    out_b = train_steps(b, PMPOOptimizers(b, warmup_steps=2), 3)
    assert [l for l, _ in out_a] == [l for l, _ in out_b]
    assert set(out_a[-1][1]) == set(out_b[-1][1]) and "value_target_abs_error" not in out_b[-1][1]
    for k, v in a.state_dict().items():
        if torch.is_tensor(v):
            assert torch.equal(v, b.state_dict()[k]), k
