"""Predictor-based LCG (intrinsic_reward.method=lcg_predictor): configuration and resolved dimensions, the
exact GGN diagonal and trace against explicit Jacobians and generic autograd, JVPs against finite
differences, the MAP objective's scaling, precision scaling and subset sampling, the probe bank, scoring
(finite, nonnegative, repeatable, dtype), RNG isolation, action/history alignment, freezing during PMPO,
checkpoint and milestone save/load parity, analysis restoration, resume, and the other methods unchanged."""
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import func

import exploration_trainer as trainer_module
import test_exploration_trainer as base
import test_plan2explore as p2e_test
import test_test_diagnostics as diag
from exploration_trainer import ExplorationTrainer
from data.milestone import load_milestone, read_manifest
from intrinsic_reward.lcg_predictor import LCGPredictor, LCGPredictorConfig, MILESTONE_FILES
from intrinsic_reward.lcg_predictor.curvature import exact_trace, ggn_diagonal_autograd, ggn_diagonal_closed_form
from intrinsic_reward.lcg_predictor.scoring import jvp_squared_norms, ParameterLayout, probe_bank
from intrinsic_reward.lcg_predictor.training import full_objective, make_optimizer, map_terms
from intrinsic_reward.plan2explore import context_window, FeatureExtractor, PredictorHead
from models.diffusion import Denoiser, DenoiserConfig
from models.diffusion.inner_model import InnerModelConfig
from test_exploration_trainer import COLLECT, compose, run_trainer
from test_test_diagnostics import BASE, Crash, same, start
from utils import rng_state

LCGP = [o for o in p2e_test.P2E if not o.startswith("intrinsic_reward")] + [
    "intrinsic_reward=LCG_predictor", "intrinsic_reward.feature_layers=[unet.d_blocks.0,unet.d_blocks.1]",
    "intrinsic_reward.pool_size=2", "intrinsic_reward.hidden_dims=[16]", "intrinsic_reward.batch_size=8",
    "intrinsic_reward.fit_updates=5", "intrinsic_reward.encode_batch_size=7", "intrinsic_reward.curvature_num_transitions=6",
    "intrinsic_reward.num_probes=3", "intrinsic_reward.candidate_chunk_size=5"]
OFF = ["evaluation.should=false"]


def run_lcgp(tmp_path, monkeypatch, extra=(), build_only=False):
    monkeypatch.setattr(base, "TINY", LCGP)
    return run_trainer(tmp_path, monkeypatch, [*COLLECT, *extra], build_only=build_only)


def make_lcgp(denoiser=None, seed=3, **overrides):
    settings = dict(feature_layers=["unet.d_blocks.0", "unet.d_blocks.1"], pool_size=2, hidden_dims=[16], batch_size=8,
                    fit_updates=20, encode_batch_size=5, curvature_num_transitions=6, num_probes=4)
    settings.update(overrides)
    denoiser = denoiser or p2e_test.tiny_denoiser()
    return LCGPredictor(LCGPredictorConfig(**settings), denoiser, 8, [-1., -1.], [1., 1.], seed=seed,
                        device=torch.device("cpu"))


def refreshed(tmp_path, epoch=1, **overrides):
    lp = make_lcgp(**overrides)
    dataset = p2e_test.random_dataset(tmp_path / "data")
    lp.refresh(lp.extractor.denoiser, dataset, {}, epoch=epoch)
    return lp, dataset


def candidates(lp, n=11, seed=5):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, lp.feature_dim, generator=g) * 0.3, torch.rand(n, lp.action_dim, generator=g) * 2 - 1


def explicit_jacobian(head, u):
    """(N, out, P) J of every row of u by reverse-mode autograd on the flat parameter vector."""
    layout = ParameterLayout(head)
    flat = layout.flatten({k: v.detach() for k, v in head.named_parameters()})

    def f(fl, x):
        return func.functional_call(head.net, {k[len("net."):]: v for k, v in layout.split(fl).items()}, (x,))
    return torch.stack([func.jacrev(f)(flat, x) for x in u]), layout


# ---------------------------------------------------------------- configuration and dimensions

def test_config_group_and_preset():
    group = compose(["intrinsic_reward=LCG_predictor"]).intrinsic_reward
    cfg = LCGPredictorConfig(**{k: v for k, v in OmegaConf.to_container(group).items() if k != "_target_"})
    assert (cfg.feature_layers, cfg.pool_size, cfg.feature_sigma, cfg.probe_seed, cfg.feature_std_floor) == (
        ["unet.d_blocks.2", "unet.d_blocks.3"], 4, 0.1, 0, 1e-3)
    assert (cfg.hidden_dims, cfg.activation, cfg.lr, cfg.batch_size, cfg.fit_updates, cfg.optimizer, cfg.init_policy) == (
        [256, 256], "silu", 1e-3, 256, 500, "adam", "reinitialize")  # Plan2Explore's per-head budget
    assert (cfg.tau_squared, cfg.prior_precision, cfg.curvature_backend, cfg.curvature_num_transitions,
            cfg.curvature_scaling) == (1.0, 1.0, "closed_form", 320, "full_replay")
    assert (cfg.scorer_backend, cfg.num_probes, cfg.score_dtype, cfg.allow_tf32) == ("exact_trace", 5, "float32", False)
    assert cfg.reward_scaling_settings() == dict(method="lcg_predictor", mode="data_scaled", fixed_divisor=1.0,
                                                 data_reference_steps=5500)
    preset = compose(["+experiment=lcg_predictor_100k"])
    OmegaConf.resolve(preset)
    assert preset.intrinsic_reward.method == "lcg_predictor" and preset.training.compile_wm is False
    assert preset.agent.actor_critic.actor_objective == "advantage_weighted"
    p2e = compose(["+experiment=plan2explore_100k"])
    OmegaConf.resolve(p2e)
    for key in ("collection", "denoiser", "milestones", "evaluation", "env", "actor_critic", "agent", "world_model_env",
                "common", "training"):
        assert OmegaConf.to_container(preset[key]) == OmegaConf.to_container(p2e[key]), key
    assert trainer_module.method_label(True, "lcg_predictor") == "lcg-predictor"


def test_resolved_dimensions_on_the_real_architecture():
    """Walker: 2048 features, 2054 -> 256 -> 256 -> 2048 SiLU, 1,118,208 parameters, all in the layout."""
    cfg = compose(["intrinsic_reward=LCG_predictor"])
    inner = InnerModelConfig(**{k: v for k, v in OmegaConf.to_container(cfg.agent.denoiser.inner_model).items()
                                if k != "_target_"})
    inner.continuous_action_dim = 6
    denoiser = Denoiser(DenoiserConfig(inner_model=inner, sigma_data=0.5, sigma_offset_noise=0.3))
    lp = LCGPredictor(LCGPredictorConfig(), denoiser, 64, [-1.] * 6, [1.] * 6, seed=0, device=torch.device("cpu"))
    assert lp.feature_dim == 2048 and lp.num_parameters == 1_118_208
    assert [(e["name"], e["shape"]) for e in lp.layout.describe()] == [
        ("net.0.weight", [256, 2054]), ("net.0.bias", [256]), ("net.2.weight", [256, 256]), ("net.2.bias", [256]),
        ("net.4.weight", [2048, 256]), ("net.4.bias", [2048])]
    # Hopper (4 actions): only the input width changes.
    hop = LCGPredictor(LCGPredictorConfig(), denoiser, 64, [-1.] * 4, [1.] * 4, seed=0, device=torch.device("cpu"))
    assert hop.layout.describe()[0]["shape"] == [256, 2052]


@pytest.mark.parametrize("overrides,match", [
    (dict(tau_squared=0.0), "tau_squared must be a finite positive"),
    (dict(prior_precision=-1.0), "prior_precision must be a finite positive"),
    (dict(prior_precision=float("inf")), "prior_precision must be a finite positive"),
    (dict(curvature_num_transitions=0), "curvature_num_transitions must be a positive integer"),
    (dict(num_probes=0), "num_probes must be a positive integer"),
    (dict(curvature_backend="kfac"), "curvature_backend must be one of"),
    (dict(curvature_scaling="half"), "curvature_scaling must be one of"),
    (dict(scorer_backend="mc"), "scorer_backend must be one of"),
    (dict(score_dtype="float16"), "score_dtype must be one of"),
    (dict(init_policy="keep"), "init_policy must be one of"),
    (dict(optimizer="adamw"), "optimizer must be one of"),
    (dict(adam_betas=[0.9]), "adam_betas must be two numbers"),
    (dict(grad_clip_norm=0.0), "grad_clip_norm must be null or a finite positive"),
    (dict(allow_tf32=1), "allow_tf32 must be true or false"),
    (dict(reward_scaling="rms"), "reward_scaling must be one of"),
    (dict(reward_scaling="none"), "fixed_divisor is used only with"),
    (dict(reward_scaling="fixed", data_reference_steps=5500), "data_reference_steps is used only with"),
    (dict(data_reference_steps=0), "data_reference_steps must be a positive integer"),
    (dict(feature_sigma=0.0), "feature_sigma must be a finite positive"),
    (dict(hidden_dims=[]), "hidden_dims must be positive"),
    (dict(method="lcg"), "method must be lcg_predictor"),
])
def test_invalid_settings_are_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        LCGPredictorConfig(**overrides)


@pytest.mark.parametrize("overrides,match", [
    (["intrinsic_reward.tau_squared=0"], "tau_squared must be a finite positive"),
    (["intrinsic_reward.feature_layers=[unet.nope]"], "are not InnerModel modules"),
    (["training.compile_wm=true"], "validated uncompiled"),
])
def test_trainer_rejects_bad_settings_before_any_work(tmp_path, monkeypatch, overrides, match):
    with pytest.raises(ValueError, match=match):
        run_lcgp(tmp_path, monkeypatch, overrides, build_only=True)
    assert not (tmp_path / "run" / "checkpoints").exists()


# ---------------------------------------------------------------- exact curvature, trace and JVPs

@pytest.mark.parametrize("activation,hidden", [("silu", [7, 5]), ("silu", [6]), ("elu", [4, 6]), ("relu", [5, 3, 4])])
def test_closed_form_ggn_equals_generic_autograd_and_the_explicit_jacobian(activation, hidden):
    torch.manual_seed(0)
    head = PredictorHead(5, 2, hidden, activation).double()
    u = torch.randn(9, 7, dtype=torch.float64)
    jac, _ = explicit_jacobian(head, u)
    reference = jac.square().sum((0, 1))  # diag(sum_i J_i^T J_i)
    torch.testing.assert_close(ggn_diagonal_closed_form(head, activation, u, chunk_size=4), reference, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(ggn_diagonal_autograd(head, u, output_chunk=2, sample_chunk=3), reference, rtol=1e-12,
                               atol=1e-12)


def test_closed_form_is_not_a_squared_loss_gradient():
    """The GGN diagonal depends on the inputs only, never on targets or residuals."""
    torch.manual_seed(1)
    head = PredictorHead(4, 2, [5], "silu").double()
    u = torch.randn(6, 6, dtype=torch.float64)
    ggn = ggn_diagonal_closed_form(head, "silu", u)
    grads = []
    for y in (torch.zeros(6, 4, dtype=torch.float64), torch.randn(6, 4, dtype=torch.float64)):
        head.zero_grad()
        for i in range(6):
            (head.net(u[i]) - y[i]).square().sum().backward()
        grads.append(torch.cat([p.grad.reshape(-1).clone() for p in head.parameters()]))
    assert not torch.allclose(grads[0], grads[1])  # loss gradients change with the targets; the GGN cannot
    assert torch.equal(ggn, ggn_diagonal_closed_form(head, "silu", u))


@pytest.mark.parametrize("activation,hidden", [("silu", [7, 5]), ("elu", [6]), ("relu", [4, 3])])
def test_exact_trace_equals_the_explicit_jacobian(activation, hidden):
    torch.manual_seed(2)
    head = PredictorHead(5, 2, hidden, activation).double()
    u = torch.randn(8, 7, dtype=torch.float64)
    jac, layout = explicit_jacobian(head, u)
    h = 0.5 + torch.rand(layout.size, dtype=torch.float64)
    expected = (jac.square() / h).sum((1, 2))
    torch.testing.assert_close(exact_trace(head, activation, u, layout.split(1 / h)), expected, rtol=1e-12, atol=1e-14)


def test_jvp_equals_explicit_jacobian_and_central_finite_differences():
    torch.manual_seed(3)
    head = PredictorHead(5, 2, [6, 4], "silu").double()
    z, a = torch.randn(7, 5, dtype=torch.float64), torch.rand(7, 2, dtype=torch.float64) * 2 - 1
    params = {k: v.detach() for k, v in head.named_parameters()}
    layout = ParameterLayout(head)
    v = torch.randn(layout.size, dtype=torch.float64)
    jac, _ = explicit_jacobian(head, torch.cat((z, a), -1))
    torch.testing.assert_close(jvp_squared_norms(head, params, z, a, [layout.split(v)]), (jac @ v).square().sum(-1),
                               rtol=1e-12, atol=1e-14)
    _, jv = func.jvp(lambda p: func.functional_call(head, p, (z, a)), (params,), (layout.split(v),))
    eps, flat = 1e-6, layout.flatten(params)
    plus = func.functional_call(head, layout.split(flat + eps * v), (z, a))
    minus = func.functional_call(head, layout.split(flat - eps * v), (z, a))
    fd = (plus - minus) / (2 * eps)
    assert float((jv - fd).norm() / jv.norm()) < 1e-8


def test_bank_estimate_is_unbiased_for_the_exact_trace():
    torch.manual_seed(4)
    head = PredictorHead(4, 2, [5], "silu").double()
    layout = ParameterLayout(head)
    h = 1 + 5 * torch.rand(layout.size)
    bank = probe_bank(h.float(), 20000, seed=11).double()
    torch.testing.assert_close(bank, probe_bank(h.float(), 20000, seed=11).double(), rtol=0, atol=0)  # reproducible
    eta = bank * h.float().double().sqrt()
    assert abs(float(eta.mean())) < 0.01 and abs(float(eta.var()) - 1) < 0.01
    z, a = torch.randn(6, 4, dtype=torch.float64), torch.rand(6, 2, dtype=torch.float64)
    params = {k: v.detach() for k, v in head.named_parameters()}
    est = torch.stack([jvp_squared_norms(head, params, z, a, [layout.split(r)]) for r in bank]).mean(0)
    exact = exact_trace(head, "silu", torch.cat((z, a), -1), layout.split(1 / h.float().double()))
    assert float(((est - exact) / exact).abs().max()) < 0.05


# ---------------------------------------------------------------- MAP objective

def test_map_terms_use_the_stated_reductions_and_no_weight_decay():
    torch.manual_seed(5)
    head = PredictorHead(4, 2, [5], "silu").double()
    z, a, y = torch.randn(8, 4).double(), torch.rand(8, 2).double(), torch.randn(8, 4).double()
    tau2, lam, n = 0.7, 3.0, 1000
    data, prior = map_terms(head, z, a, y, tau2, lam, n)
    pred = head(z, a)
    assert torch.allclose(data, ((y - pred) ** 2).sum() / (2 * tau2 * 8))
    assert torch.allclose(data, torch.nn.functional.mse_loss(pred, y) * 4 / (2 * tau2))  # D x element-mean MSE
    norm = sum((p ** 2).sum() for p in head.parameters())
    assert torch.allclose(prior, lam * norm / (2 * n))
    cfg = LCGPredictorConfig()
    for name in ("adam", "sgd"):
        cfg.optimizer = name
        assert all(g["weight_decay"] == 0 for g in make_optimizer(head, cfg).param_groups)
    full = full_objective(head, lambda idx: (z[idx], a[idx], y[idx]), 8, tau2, lam, chunk=3)
    assert abs(full["data_per_transition"] - float(data)) < 1e-12  # the minibatch is the whole set here
    assert abs(full["prior_per_transition"] - lam * float(norm) / (2 * 8)) < 1e-12


def test_fit_learns_only_the_predictor_and_freezes_it(tmp_path):
    lp, _ = refreshed(tmp_path, fit_updates=200, lr=3e-3)
    info = lp.fit_info
    assert info["after_objective_per_transition"] < info["before_objective_per_transition"]
    assert info["updates"] == 200 and info["init"] == "fresh" and "no bootstrap" in info["sampling"]
    assert all(not p.requires_grad for p in lp.predictor.parameters()) and not lp.predictor.training
    assert all(p.grad is None for p in lp.extractor.denoiser.parameters())
    assert lp.predictor_sha256 is not None
    lp.verify_frozen()
    with torch.no_grad():
        next(lp.predictor.parameters()).add_(1.0)
    with pytest.raises(RuntimeError, match="changed after its refresh"):
        lp.milestone_files()


def test_warm_start_continues_from_the_previous_round(tmp_path):
    lp, dataset = refreshed(tmp_path, init_policy="warm_start", fit_updates=3)
    first = {k: v.clone() for k, v in lp.predictor.state_dict().items()}
    fresh = make_lcgp(init_policy="reinitialize", fit_updates=3)
    fresh.refresh(fresh.extractor.denoiser, dataset, {}, epoch=1)
    assert same(first, fresh.predictor.state_dict())  # round 1: both start from the round's fresh init
    lp.refresh(lp.extractor.denoiser, dataset, {}, epoch=2)
    fresh.refresh(fresh.extractor.denoiser, dataset, {}, epoch=2)
    assert lp.fit_info["init"] == "warm_start (round 1)" and fresh.fit_info["init"] == "fresh"
    assert not same(lp.predictor.state_dict(), fresh.predictor.state_dict())


def snapshot(lp, f, a):
    return dict(predictor={k: v.clone() for k, v in lp.predictor.state_dict().items()}, mean=lp.mean.clone(),
                std=lp.std.clone(), raw_std=lp.raw_std.clone(), h_D=lp.h_D.clone(),
                bank=None if lp.bank is None else lp.bank.clone(), identity=dict(lp.cache_identity),
                cache=lp.cache, divisor=lp.reward_divisor, round=lp.round_id, refreshes=lp.refreshes,
                sha=lp.predictor_sha256, fit_updates=lp.fit_updates_total, curvature=dict(lp.curvature_info),
                fit=dict(lp.fit_info), replay_steps=lp.replay_steps, n=lp.num_transitions,
                raw=lp.score_features(f, a), scaled=lp.scale(lp.score_features(f, a)))


@pytest.mark.parametrize("backend,stage", [(b, s) for b in ("exact_trace", "jvp_bank")
                                           for s in ("encoding", "fit", "curvature", "bank") if (b, s) != ("exact_trace", "bank")])
def test_a_failed_refresh_leaves_the_previous_round_intact(tmp_path, monkeypatch, backend, stage):
    """A refresh on a NEW replay (so new features and normalization would be built) that fails at one stage:
    every part of the previous round stays as it was and keeps scoring identically; the next refresh works."""
    import intrinsic_reward.lcg_predictor.lifecycle as lifecycle
    lp, _ = refreshed(tmp_path, scorer_backend=backend)
    f, a = candidates(lp)
    before = snapshot(lp, f, a)
    new_replay = p2e_test.random_dataset(tmp_path / "new", lengths=(8, 6, 7), finished=(True, True, False), seed=3)

    def boom(*args, **kwargs):
        raise FloatingPointError(f"injected failure at {stage}")
    with monkeypatch.context() as m:
        target = {"encoding": (lifecycle.p2e, "build_feature_cache"), "fit": (lifecycle, "fit_map"),
                  "curvature": (lifecycle, "ggn_diagonal_closed_form"), "bank": (lifecycle, "probe_bank")}[stage]
        m.setattr(*target, boom)
        with pytest.raises(FloatingPointError, match="injected failure"):
            lp.refresh(lp.extractor.denoiser, new_replay, {}, epoch=2)
    after = snapshot(lp, f, a)
    assert after["cache"] is before["cache"]
    for key in before:
        if key != "cache":
            assert same(before[key], after[key]), key
    lp.verify_frozen()
    # The PMPO hook still scores with the previous round.
    infos = [dict(plan2explore_feature=f[i:i + 1], plan2explore_action=a[i:i + 1]) for i in range(len(f))]
    torch.testing.assert_close(lp(infos, torch.zeros(1, len(f)))[0], before["scaled"], rtol=0, atol=0)
    # A subsequent successful refresh installs the new round consistently.
    lp.refresh(lp.extractor.denoiser, new_replay, {}, epoch=2)
    assert lp.round_id == 2 and lp.refreshes == before["refreshes"] + 1 and lp.cache is not before["cache"]
    assert lp.cache_identity["replay_sha256"] != before["identity"]["replay_sha256"]
    assert not torch.equal(lp.mean, before["mean"]) and lp.num_transitions == 8 + 6 + 6  # finished episodes: every step; the unfinished one: all but the last
    fresh = make_lcgp(scorer_backend=backend)
    fresh.refresh(fresh.extractor.denoiser, new_replay, {}, epoch=2)
    for name in ("mean", "std", "h_D"):
        assert torch.equal(getattr(lp, name), getattr(fresh, name)), name
    assert lp.predictor_sha256 == fresh.predictor_sha256 and same(lp.bank, fresh.bank)
    assert torch.equal(lp.score_features(f, a), fresh.score_features(f, a))


# ---------------------------------------------------------------- historical precision and bank

@pytest.mark.parametrize("scaling", ["full_replay", "none"])
def test_precision_scaling_subset_and_record(tmp_path, scaling):
    lp, _ = refreshed(tmp_path, curvature_scaling=scaling, tau_squared=0.5, prior_precision=2.0)
    info, n = lp.curvature_info, lp.num_transitions
    # Episodes of 9 (finished: final observation), 7 and 5 steps: 9 + 6 + 4 valid transitions.
    assert info["num_samples"] == 6 and info["replay_transitions"] == n == 19 and info["replay_steps"] == 21
    scale = n / 6 if scaling == "full_replay" else 1.0
    assert info["scale_factor"] == scale and info["scaling"] == scaling and "without replacement" in info["sampling"]
    idx = torch.from_numpy(np.sort(np.random.default_rng(lp.curvature_seed(1)).choice(n, size=6, replace=False)))
    z, a, _ = lp.transition_batch(idx)
    ggn = ggn_diagonal_closed_form(lp.predictor, "silu", torch.cat((z, a), -1))
    torch.testing.assert_close(lp.h_D, (2.0 + scale * ggn / 0.5).float(), rtol=0, atol=0)
    assert (lp.h_D >= 2.0).all()


def test_a_replay_smaller_than_the_subset_uses_every_transition_unscaled(tmp_path):
    lp, _ = refreshed(tmp_path, curvature_num_transitions=1000)
    assert lp.curvature_info["num_samples"] == lp.num_transitions and lp.curvature_info["scale_factor"] == 1.0


def test_precision_backends_agree_and_are_recomputed_each_round(tmp_path):
    closed, dataset = refreshed(tmp_path, curvature_backend="closed_form")
    auto = make_lcgp(curvature_backend="autograd", autograd_output_chunk=7)
    auto.refresh(auto.extractor.denoiser, dataset, {}, epoch=1)
    torch.testing.assert_close(closed.h_D, auto.h_D, rtol=1e-5, atol=1e-6)
    first = closed.h_D.clone()
    closed.refresh(closed.extractor.denoiser, dataset, {}, epoch=2)
    # Round 2 is computed from scratch: a different predictor and subset, not round 1 plus more curvature.
    assert not torch.equal(first, closed.h_D) and closed.curvature_info["round"] == 2
    assert closed.curvature_info["ggn_sum"] < 3 * float(first.sum())


def test_bank_is_the_rounds_frozen_h_scaled_gaussian(tmp_path):
    lp, dataset = refreshed(tmp_path, scorer_backend="jvp_bank")
    assert lp.bank.shape == (4, lp.num_parameters)
    generator = torch.Generator().manual_seed(lp.bank_seed(1))
    eta = torch.randn(4, lp.num_parameters, generator=generator)
    torch.testing.assert_close(lp.bank, eta * lp.h_D.rsqrt(), rtol=0, atol=0)
    other = make_lcgp(bank_seed=1, scorer_backend="jvp_bank")
    other.refresh(other.extractor.denoiser, dataset, {}, epoch=1)
    assert torch.equal(other.h_D, lp.h_D) and not torch.equal(other.bank, lp.bank)
    exact, _ = refreshed(tmp_path / "x", scorer_backend="exact_trace")
    assert exact.bank is None and "no scoring bank" in exact.milestone_files()["lcg_predictor_bank.pt"]["representation"]


# ---------------------------------------------------------------- scoring

@pytest.mark.parametrize("backend", ["exact_trace", "jvp_bank"])
def test_scores_are_finite_nonnegative_repeatable_and_use_current_inputs_only(tmp_path, backend):
    lp, _ = refreshed(tmp_path, scorer_backend=backend)
    f, a = candidates(lp)
    raw = lp.score_features(f, a)
    assert raw.shape == (11,) and torch.isfinite(raw).all() and (raw >= 0).all() and raw.dtype == torch.float32
    assert torch.equal(raw, lp.score_features(f, a))
    # Chunking does not change scores beyond rounding; every candidate is scored independently.
    torch.testing.assert_close(lp.score_features(f[3:4], a[3:4]), raw[3:4], rtol=1e-5, atol=1e-7)
    if backend == "exact_trace":
        torch.testing.assert_close(raw, lp.exact_scores(f, a), rtol=1e-6, atol=0)
        return
    manual = sum(((func.jvp(lambda p: func.functional_call(lp.predictor, p, lp.inputs(f, a)), (lp._score_params,),
                            (t,))[1]) ** 2).sum(-1) for t in lp._tangents) / (4 * lp.cfg.tau_squared)
    torch.testing.assert_close(raw, manual.float(), rtol=1e-5, atol=1e-7)


def test_exact_backend_matches_the_explicit_jacobian_and_the_bank_on_average(tmp_path):
    lp, dataset = refreshed(tmp_path, scorer_backend="exact_trace", score_dtype="float64", tau_squared=0.5)
    f, a = candidates(lp, n=5)
    raw = lp.score_features(f, a)
    z, ac = lp.inputs(f, a)
    head64 = PredictorHead(lp.feature_dim, lp.action_dim, [16], "silu").double()
    head64.load_state_dict({k: v.double() for k, v in lp.predictor.state_dict().items()})
    jac, _ = explicit_jacobian(head64, torch.cat((z, ac), -1))
    expected = (jac.square() / lp.h_D.double()).sum((1, 2)) / 0.5
    torch.testing.assert_close(raw, expected.float(), rtol=1e-6, atol=0)
    many = make_lcgp(num_probes=4000, tau_squared=0.5, scorer_backend="jvp_bank")
    many.refresh(many.extractor.denoiser, dataset, {}, epoch=1)
    assert torch.equal(many.h_D, lp.h_D)
    est = many.score_features(f, a)
    assert float(((est - raw) / raw).abs().max()) < 0.1  # 4000-probe JVP estimate of the exact value
    torch.testing.assert_close(many.exact_scores(f, a), raw, rtol=1e-5, atol=0)


@pytest.mark.parametrize("backend", ["exact_trace", "jvp_bank"])
def test_float64_scoring_agrees_with_float32(tmp_path, backend):
    lp32, dataset = refreshed(tmp_path, scorer_backend=backend)
    lp64 = make_lcgp(score_dtype="float64", scorer_backend=backend)
    lp64.refresh(lp64.extractor.denoiser, dataset, {}, epoch=1)
    assert torch.equal(lp32.h_D, lp64.h_D) and same(lp32.bank, lp64.bank)
    f, a = candidates(lp32)
    torch.testing.assert_close(lp64.score_features(f, a), lp32.score_features(f, a), rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize("mode,divisor", [("data_scaled", 2.0 * 50 / 21), ("fixed", 2.0), ("none", None)])
def test_reward_scaling_is_frozen_per_round_and_raw_stays_separate(tmp_path, mode, divisor):
    kwargs = dict(reward_scaling=mode, fixed_divisor=None if mode == "none" else 2.0,
                  data_reference_steps=50 if mode == "data_scaled" else None)
    lp, _ = refreshed(tmp_path, **kwargs)
    assert lp.reward_divisor == (None if divisor is None else pytest.approx(divisor))
    f, a = candidates(lp)
    raw = lp.score_features(f, a)
    torch.testing.assert_close(lp.scale(raw), raw if divisor is None else raw / divisor)
    infos = [dict(plan2explore_feature=f[i:i + 1], plan2explore_action=a[i:i + 1]) for i in range(11)]
    reward = lp(infos, torch.zeros(1, 11))
    assert reward.shape == (1, 11) and torch.equal(lp.last_raw[0], raw) and torch.equal(lp.last_scaled, reward)


def test_refresh_uses_private_rng_only_and_is_deterministic(tmp_path):
    a = make_lcgp(scorer_backend="jvp_bank")  # the fixture's tiny denoiser seeds the global RNG itself: build it first
    dataset = p2e_test.random_dataset(tmp_path / "data")
    torch.manual_seed(123)
    np.random.seed(123)
    before = rng_state()
    a.refresh(a.extractor.denoiser, dataset, {}, epoch=1)
    f, act = candidates(a)
    a.score_features(f, act)
    after = rng_state()
    for k in ("python", "numpy", "torch"):
        assert same(before[k], after[k]), k
    b = make_lcgp(scorer_backend="jvp_bank")
    b.refresh(b.extractor.denoiser, dataset, {}, epoch=1)
    assert same(a.state_dict(), b.state_dict())
    c = make_lcgp(seed=4, scorer_backend="jvp_bank")
    c.refresh(c.extractor.denoiser, dataset, {}, epoch=1)
    assert not torch.equal(a.bank, c.bank)


# ---------------------------------------------------------------- alignment of features, history and actions

def test_transitions_pair_the_current_feature_and_action_with_the_next_feature(tmp_path):
    lp, dataset = refreshed(tmp_path)
    cache, n, row = lp.cache, lp.extractor.num_conditioning, 0
    rows = []
    for e in range(dataset.num_episodes):
        ep = dataset.load_episode(e)
        frames = torch.cat((ep.obs, ep.info["final_observation"].unsqueeze(0))) if "final_observation" in ep.info else ep.obs
        for t in range(frames.shape[0] - 1):
            rows.append((row + t, row + t + 1, ep.act[t]))
        for j in (0, frames.shape[0] - 1):
            ctx, ctx_act, _ = context_window(frames, ep.act, j, n)
            torch.testing.assert_close(cache.features[row + j], lp.extractor(frames[j:j + 1], ctx[None], ctx_act[None])[0],
                                       rtol=1e-5, atol=1e-6)
        row += frames.shape[0]
    assert [(int(c), int(t)) for c, t in zip(cache.current, cache.target)] == [(c, t) for c, t, _ in rows]
    assert torch.equal(cache.actions, torch.stack([r[2] for r in rows]).float())


def test_real_episode_evaluation_scores_each_step_with_its_own_feature_and_action(tmp_path):
    from intrinsic_reward.intrinsic_eval import IntrinsicEvaluator
    lp, dataset = refreshed(tmp_path)
    evaluator = IntrinsicEvaluator("lcg_predictor", lp.extractor.num_conditioning, torch.device("cpu"), chunk_size=3,
                                   lcg_predictor=lp)
    episode = dataset.load_episode(0)
    raw, scaled = evaluator.score_episode(episode)
    n = lp.extractor.num_conditioning
    for t in range(len(episode)):
        ctx, ctx_act, _ = context_window(episode.obs, episode.act, t, n)
        feature = lp.extractor(episode.obs[t:t + 1], ctx[None], ctx_act[None])
        torch.testing.assert_close(raw[t:t + 1], lp.score_features(feature, episode.act[t:t + 1]), rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(scaled, lp.scale(raw))
    edited = episode.act.clone()
    edited[3] = -edited[3]  # a_3 enters step 3's score directly and later steps' features as context only
    episode.act = edited
    raw_edit, _ = evaluator.score_episode(episode)
    assert torch.equal(raw_edit[:3], raw[:3]) and not torch.equal(raw_edit[3], raw[3])
    assert lp.last_raw is None  # evaluation never touches the PMPO hook's monitoring state


# ---------------------------------------------------------------- state, checkpoints and milestone files

def test_state_dict_roundtrip_reproduces_scores_without_refitting(tmp_path):
    lp, _ = refreshed(tmp_path)
    f, a = candidates(lp)
    raw = lp.score_features(f, a)
    state = lp.state_dict()
    restored = make_lcgp()
    restored.load_state_dict(state)
    assert restored.cache is None and restored.fit_updates_total == lp.fit_updates_total
    assert torch.equal(restored.score_features(f, a), raw) and torch.equal(restored.scale(raw), lp.scale(raw))
    with pytest.raises(ValueError, match="LCG-predictor settings"):
        make_lcgp(num_probes=5).load_state_dict(state)
    with pytest.raises(ValueError, match="LCG-predictor settings"):
        make_lcgp(seed=9).load_state_dict(state)
    tampered = dict(state, predictor={k: v + 1 for k, v in state["predictor"].items()})
    with pytest.raises(ValueError, match="recorded SHA-256"):
        make_lcgp().load_state_dict(tampered)


@pytest.mark.parametrize("backend", ["exact_trace", "jvp_bank"])
def test_milestone_files_restore_identical_raw_and_scaled_scores(tmp_path, backend):
    lp, _ = refreshed(tmp_path, scorer_backend=backend)
    folder = tmp_path / "milestone"
    folder.mkdir()
    files = lp.milestone_files()
    assert sorted(files) == sorted(MILESTONE_FILES)
    for name, obj in files.items():
        torch.save(obj, folder / name)
    pred = files["lcg_predictor.pt"]
    assert pred["num_parameters"] == lp.num_parameters and pred["parameter_order"] == lp.layout.describe()
    assert files["lcg_predictor_scorer.pt"]["denoiser_identity"] == lp.cache_identity
    bank_file = files["lcg_predictor_bank.pt"]
    if backend == "jvp_bank":
        assert bank_file["bank_seed"] == lp.bank_seed(1) and torch.equal(bank_file["tangents"], lp.bank.cpu())
    else:  # the file stays, as an explicit record that no bank is used
        assert bank_file["tangents"] is None and bank_file["bank_seed"] is None and bank_file["backend"] == "exact_trace"
        assert "no scoring bank" in bank_file["representation"]
    f, a = candidates(lp)
    raw = lp.score_features(f, a)
    restored = make_lcgp(scorer_backend=backend)
    restored.load_milestone_files(folder)
    assert torch.equal(restored.score_features(f, a), raw) and torch.equal(restored.scale(raw), lp.scale(raw))
    assert same(restored.bank, lp.bank) and torch.equal(restored.h_D, lp.h_D) and torch.equal(restored.mean, lp.mean)
    with pytest.raises(ValueError, match="refreshed on denoiser"):
        make_lcgp(denoiser=p2e_test.tiny_denoiser(seed=1), scorer_backend=backend).load_milestone_files(folder)
    with pytest.raises(ValueError, match="LCG-predictor settings"):  # the other backend is another scorer
        make_lcgp(scorer_backend="jvp_bank" if backend == "exact_trace" else "exact_trace").load_milestone_files(folder)


# ---------------------------------------------------------------- trainer integration

def test_trainer_smoke_refresh_scoring_milestone_and_analysis_restore(tmp_path, monkeypatch):
    trainer, logs = run_lcgp(tmp_path, monkeypatch, OFF + ["milestones.at_steps=[24]"])
    lp = trainer.lcg_predictor
    assert len(trainer.agent.denoisers) == 1 and trainer.plan2explore is None
    assert lp.refreshes == 2 and lp.round_id == 2 and trainer.agent.actor_critic.updates.item() == 3
    assert trainer.agent.actor_critic.intrinsic_reward_fn is lp and trainer._rl_env.feature_encoder is lp.extractor
    keys = set().union(*logs)
    for key in ("actor_critic/train/intrinsic/raw_reward_mean", "actor_critic/train/intrinsic/scaled_reward_mean"):
        assert key in keys, key
    rewards = [row["actor_critic/train/intrinsic/raw_reward_mean"] for row in logs
               if "actor_critic/train/intrinsic/raw_reward_mean" in row]
    assert rewards and all(np.isfinite(r) and r >= 0 for r in rewards)
    path = Path("milestones/steps_000024")
    manifest = read_manifest(path)
    assert sorted(manifest["files"]) == sorted(["dataset", "denoiser", "exploration_policy",
                                                 *(f[:-3] for f in MILESTONE_FILES)])
    assert manifest["exploration"]["method"] == "lcg_predictor" and manifest["lcg_predictor"]["round_id"] == 2
    assert manifest["lcg_predictor"]["num_parameters"] == lp.num_parameters
    milestone = load_milestone(path, "walk")
    assert same(milestone.denoiser_state, trainer.agent.denoiser.state_dict())
    # Restoration with the analysis tooling (no trainer): identical raw and scaled scores on identical inputs.
    import exploration_analysis as ea
    scorer, _, _ = ea.restore_milestone_scorer("lcg-predictor", path, torch.device("cpu"))
    restored = scorer.lcg_predictor
    assert restored.round_id == 2 and lp.bank is None and restored.bank is None and torch.equal(restored.h_D, lp.h_D)
    assert restored.cfg.scorer_backend == "exact_trace"
    batch = ea.window_batch(Path.cwd(), ea.run_replay(Path.cwd()), 0, 12, scorer.n)
    raw, scaled = scorer.score(batch, seed=0)
    features = lp.extractor(batch["frame"], batch["context"], batch["context_act"])
    expected = lp.score_features(features, batch["action"])
    assert torch.equal(raw, expected.double()) and torch.equal(scaled, lp.scale(expected).double())
    assert scorer.describe()["round"] == 2


def test_default_exact_backend_records_no_bank_and_the_jvp_bank_runs_in_the_trainer(tmp_path, monkeypatch):
    trainer, _ = run_lcgp(tmp_path / "exact", monkeypatch, OFF + ["milestones.at_steps=[24]"])
    assert trainer.lcg_predictor.cfg.scorer_backend == "exact_trace" and trainer.lcg_predictor.bank is None
    bank = torch.load("milestones/steps_000024/lcg_predictor_bank.pt", weights_only=False)
    assert bank["tangents"] is None and bank["backend"] == "exact_trace" and "no scoring bank" in bank["representation"]
    trainer, _ = run_lcgp(tmp_path / "bank", monkeypatch, OFF + ["intrinsic_reward.scorer_backend=jvp_bank",
                                                                 "milestones.at_steps=[24]"])
    lp = trainer.lcg_predictor
    assert lp.bank.shape == (3, lp.num_parameters) and lp.refreshes == 2
    bank = torch.load("milestones/steps_000024/lcg_predictor_bank.pt", weights_only=False)
    assert torch.equal(bank["tangents"], lp.bank.cpu()) and bank["bank_seed"] == lp.bank_seed(2)
    import exploration_analysis as ea
    scorer, _, _ = ea.restore_milestone_scorer("lcg-predictor", Path("milestones/steps_000024"), torch.device("cpu"))
    assert torch.equal(scorer.lcg_predictor.bank, lp.bank)


def test_predictor_precision_and_bank_do_not_change_during_pmpo(tmp_path, monkeypatch):
    snapshots, after_pmpo = [], []
    original = LCGPredictor.refresh

    def refresh(self, *args, **kwargs):
        logs = original(self, *args, **kwargs)
        snapshots.append(({k: v.clone() for k, v in self.predictor.state_dict().items()}, self.h_D.clone(),
                          self._inv_h, {k: v.clone() for k, v in trainer_ref[0].agent.denoiser.state_dict().items()},
                          {k: v.clone() for k, v in trainer_ref[0].agent.actor_critic.actor.state_dict().items()}))
        return logs

    monkeypatch.setattr(LCGPredictor, "refresh", refresh)
    trainer_ref = []
    trainer, _ = run_lcgp(tmp_path, monkeypatch, OFF, build_only=True)
    trainer_ref.append(trainer)
    original_train_component = ExplorationTrainer.train_component

    def train_component(self, name, steps, member=None):
        out = original_train_component(self, name, steps, member)
        if name == "actor_critic":
            lp = self.lcg_predictor
            after_pmpo.append(({k: v.clone() for k, v in lp.predictor.state_dict().items()}, lp.h_D.clone(),
                               lp._inv_h, {k: v.clone() for k, v in self.agent.denoiser.state_dict().items()},
                               {k: v.clone() for k, v in self.agent.actor_critic.actor.state_dict().items()}))
            lp.verify_frozen()
        return out

    monkeypatch.setattr(ExplorationTrainer, "train_component", train_component)
    trainer._run()
    assert len(snapshots) == len(after_pmpo) == 2
    for before, after in zip(snapshots, after_pmpo):
        assert all(same(x, y) for x, y in zip(before[:4], after[:4]))  # predictor, h_D, 1/h_D views, denoiser
        assert not same(before[4], after[4])  # the controller did update


def test_changing_task_rewards_does_not_change_the_lcg_predictor(tmp_path, monkeypatch):
    runs = []
    for scale in (0.0, 1e6):
        root = tmp_path / f"scale{scale:g}"
        base.write_static_dataset(root / "static", scale)
        overrides = [f"static_dataset.path={(root / 'static').as_posix()}", "training.num_final_epochs=2",
                     "evaluation.every=1"]
        monkeypatch.setattr(base, "TINY", LCGP)
        runs.append(run_trainer(root, monkeypatch, overrides))
    (a, _), (b, _) = runs
    assert same(a.lcg_predictor.state_dict(), b.lcg_predictor.state_dict())
    assert same(a.agent.state_dict(), b.agent.state_dict())


def test_crash_resume_is_identical_and_changed_settings_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "TINY", LCGP)
    full, _ = start(tmp_path / "full", monkeypatch, BASE + OFF)
    full._run()
    full_rng = rng_state()
    original = ExplorationTrainer.train_agent

    def crash(self):
        if self.epoch == 2:
            raise Crash()
        return original(self)

    monkeypatch.setattr(ExplorationTrainer, "train_agent", crash)
    crashed, _ = start(tmp_path / "split", monkeypatch, BASE + OFF)
    with pytest.raises(Crash):
        crashed._run()
    monkeypatch.setattr(ExplorationTrainer, "train_agent", original)
    saved = torch.load(tmp_path / "split/checkpoints/state.pt", weights_only=False)["lcg_predictor"]
    assert saved["predictor"] is not None and saved["round_id"] == 1 and saved["bank"] is None  # exact_trace
    loads = []
    monkeypatch.setattr(torch.nn.Module, "load_state_dict",
                        (lambda f: lambda self, *a, **k: loads.append(1) or f(self, *a, **k))(torch.nn.Module.load_state_dict))
    for override, match in (("intrinsic_reward.tau_squared=2.0", "LCG-predictor settings"),
                            ("intrinsic_reward.prior_precision=2.0", "LCG-predictor settings"),
                            ("intrinsic_reward.num_probes=4", "LCG-predictor settings"),
                            ("intrinsic_reward.curvature_scaling=none", "LCG-predictor settings"),
                            ("intrinsic_reward.scorer_backend=jvp_bank", "LCG-predictor settings"),
                            ("intrinsic_reward.hidden_dims=[8]", "LCG-predictor settings"),
                            ("intrinsic_reward.feature_sigma=0.2", "LCG-predictor settings"),
                            ("intrinsic_reward.fixed_divisor=2.0", "must not switch")):
        loads.clear()
        with pytest.raises(ValueError, match=match):
            start(tmp_path / "split", monkeypatch, [override], resume=True)
        assert not loads, override  # refused before any weights were loaded
    resumed, _ = start(tmp_path / "split", monkeypatch, [], resume=True)
    assert resumed.agent.actor_critic.intrinsic_reward_fn is resumed.lcg_predictor  # restored round, wired
    resumed._run()
    assert same(full.agent.state_dict(), resumed.agent.state_dict())
    assert same(full.opt.state_dict(), resumed.opt.state_dict())
    assert same(full.lcg_predictor.state_dict(), resumed.lcg_predictor.state_dict())
    for k in ("python", "numpy", "torch"):
        assert same(full_rng[k], rng_state()[k])


def test_other_methods_refuse_an_lcg_predictor_checkpoint_and_build_none(tmp_path, monkeypatch):
    lcgp, _ = run_lcgp(tmp_path / "lcgp", monkeypatch, OFF, build_only=True)
    lcgp_state = torch.load(tmp_path / "lcgp/run/checkpoints/state.pt", weights_only=False)
    monkeypatch.setattr(base, "TINY", p2e_test.P2E)
    p2e, _ = run_trainer(tmp_path / "p2e", monkeypatch, COLLECT, build_only=True)
    p2e_state = torch.load(tmp_path / "p2e/run/checkpoints/state.pt", weights_only=False)
    assert p2e.lcg_predictor is None and p2e_state["lcg_predictor"] is None
    with pytest.raises(ValueError, match="must not switch"):  # reward settings differ first
        p2e.load_state_dict(dict(lcgp_state))
    with pytest.raises(ValueError, match="Plan2Explore settings None"):  # then the method's own state
        p2e.load_state_dict(dict(lcgp_state, lcg_reward_scaling=p2e_state["lcg_reward_scaling"]))
    with pytest.raises(ValueError, match="Plan2Explore feature predictors"):
        lcgp.load_state_dict(dict(p2e_state, lcg_reward_scaling=lcgp._lcg_reward_scaling))
    with pytest.raises(ValueError, match="LCG-predictor settings None"):
        lcgp.load_state_dict(dict(p2e_state, lcg_reward_scaling=lcgp._lcg_reward_scaling, plan2explore=None))
    for name, extra in (("lcg", []), ("uniform", ["collection.train.mode=uniform", "intrinsic_reward.enabled=false"])):
        monkeypatch.setattr(base, "TINY", base.TINY_LCG)
        trainer, _ = run_trainer(tmp_path / name, monkeypatch, [*COLLECT, *extra], build_only=True)
        assert trainer.lcg_predictor is None
        assert trainer.agent.actor_critic.rl_env is None or trainer.agent.actor_critic.rl_env.feature_encoder is None
        if name == "lcg":  # a method without feature predictors refuses the LCG-predictor state
            with pytest.raises(ValueError, match="LCG feature predictor"):
                trainer.load_state_dict(dict(lcgp_state, lcg_reward_scaling=trainer._lcg_reward_scaling))


def test_older_checkpoints_without_the_key_still_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "TINY", base.TINY_LCG)
    run, _ = start(tmp_path / "lcg", monkeypatch, BASE + OFF)
    run._run()
    path = tmp_path / "lcg/checkpoints/state.pt"
    state = torch.load(path, weights_only=False)
    assert state["lcg_predictor"] is None
    del state["lcg_predictor"]  # as saved before this method existed
    torch.save(state, path)
    resumed, _ = start(tmp_path / "lcg", monkeypatch, [], resume=True)
    assert resumed.lcg_predictor is None and resumed.epoch == run.epoch
