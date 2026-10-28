"""Laplace-inspired sampled diffusion ensemble: configuration, reuse of LCG's historical precision
(damping and scale applied once), Gaussian sampling, parameter isolation (also on exceptions),
fixed-parameter shared-noise diffusion chains, optimized-vs-reference agreement, imagination
scoring/selection, task-reward independence, checkpoint compatibility, exact resume, milestones."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import exploration_trainer as trainer_module
import laplace as laplace_module
import lcg.precision as precision_module
import test_exploration_trainer as base
import test_test_diagnostics as diag
from data import BatchSampler, collate_segments_without_rewards, Dataset, Episode
from data.milestone import load_milestone, publish_milestone, read_manifest
from envs import WorldModelEnv, WorldModelEnvConfig
from envs.world_model_env import ensemble_disagreement
from exploration_trainer import ExplorationTrainer
from laplace import (LaplaceConfig, LaplaceEnsemble, perturbation_scale, shared_prefix_first_evaluation)
from lcg import LCGConfig, LCGLifecycle, selected_named_parameters, ThetaSConfig
from models.diffusion import Denoiser, DenoiserConfig, DiffusionSampler, DiffusionSamplerConfig, SigmaDistributionConfig
from models.diffusion.inner_model import InnerModelConfig
from test_exploration_trainer import COLLECT, compose, run_trainer, TASKS, write_static_dataset
from test_test_diagnostics import BASE, Crash, same, start
from utils import rng_state

R1_TINY = "unet.u_blocks.1.resblocks.0.*"  # first ResBlock of the last decoder level (2-level UNet)
LAP = [o for o in base.TINY if not o.startswith("intrinsic_reward.") and not o.startswith("agent.denoiser.inner_model.")] + [
    "agent.denoiser.inner_model.num_steps_conditioning=2", "agent.denoiser.inner_model.cond_channels=16",
    "agent.denoiser.inner_model.depths=[1,1]", "agent.denoiser.inner_model.channels=[8,8]",
    "agent.denoiser.inner_model.attn_depths=[0,0]",
    "intrinsic_reward=laplace_disagreement", "intrinsic_reward.num_members=3",
    f"intrinsic_reward.theta_s.include=[{R1_TINY}]", "intrinsic_reward.precision_reference_size=8",
    "intrinsic_reward.precision_num_mc=1", "agent.actor_critic.actor_objective=advantage_weighted"]
OFF = ["evaluation.should=false"]
TINY_LCG = list(base.TINY)  # the default (LCG) tiny settings, captured before any test patches them
SIGMA = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=2e-3, sigma_max=20)
SAMPLER = DiffusionSamplerConfig(num_steps_denoising=3)


def run_lap(tmp_path, monkeypatch, extra=(), build_only=False):
    monkeypatch.setattr(base, "TINY", LAP)
    return run_trainer(tmp_path, monkeypatch, [*COLLECT, *extra], build_only=build_only)


def tiny_denoiser(seed=0):
    torch.manual_seed(seed)
    inner = InnerModelConfig(img_channels=3, num_steps_conditioning=2, cond_channels=16, depths=[1, 1], channels=[8, 8],
                             attn_depths=[False, False], continuous_action_dim=2)
    denoiser = Denoiser(DenoiserConfig(inner_model=inner, sigma_data=0.5, sigma_offset_noise=0.3))
    with torch.no_grad():  # nonzero conv2 weights: the selected block really affects outputs
        for p in denoiser.parameters():
            if p.ndim > 1 and not p.any():
                p.normal_(0, 0.2)
    return denoiser


def make_laplace(denoiser=None, include=(R1_TINY,), **overrides):
    settings = dict(num_members=3, theta_s=ThetaSConfig(include=tuple(include)), precision_reference_size=6,
                    precision_num_mc=2)
    settings.update(overrides)
    return LaplaceEnsemble(LaplaceConfig(**settings), denoiser or tiny_denoiser(), SIGMA, seed=5, device=torch.device("cpu"))


def random_dataset(path, seed=0, lengths=(9, 7)):
    g = torch.Generator().manual_seed(seed)
    dataset = Dataset(path, "train_dataset", cache_in_ram=True)
    for length in lengths:
        obs = torch.randint(0, 256, (length, 3, 8, 8), generator=g).float().div(255).mul(2).sub(1)
        act = torch.rand(length, 2, generator=g) * 2 - 1
        rew = torch.rand(length, generator=g)
        trunc = torch.zeros(length, dtype=torch.long)
        trunc[-1] = 1
        dataset.add_episode(Episode(obs, act, rew, torch.zeros(length, dtype=torch.long), trunc,
                                    {"final_observation": obs[-1].clone()}, task_rew=torch.stack([rew, -rew], 1),
                                    task_names=TASKS))
    return dataset


def inputs(b=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(b, 2, 3, 8, 8, generator=g) * 2 - 1, torch.rand(b, 2, 2, generator=g) * 2 - 1,
            torch.randn(b, 3, 8, 8, generator=g))


def materialized(denoiser, params):
    """An independent copy of the denoiser with the member's selected values written in."""
    copy = deepcopy(denoiser)
    named = dict(copy.inner_model.named_parameters())
    with torch.no_grad():
        for name, value in params.items():
            named[name].copy_(value)
    return copy


def snapshot(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


# ---------------------------------------------------------------- configuration

def test_config_group_and_preset():
    group = compose(["intrinsic_reward=laplace_disagreement"]).intrinsic_reward
    cfg = LaplaceConfig(**{k: v for k, v in OmegaConf.to_container(group).items() if k != "_target_"})
    assert (cfg.num_members, cfg.posterior_temperature, cfg.theta_s.include, cfg.precision_reference_size,
            cfg.precision_num_mc, cfg.damping, cfg.beta) == (5, 1.0, ("unet.u_blocks.3.resblocks.0.*",), 320, 3, 1e-4, 1.0)
    assert (cfg.num_noise_samples, cfg.reward_scaling, cfg.fixed_divisor, cfg.shared_prefix) == (1, "fixed", 1.0, True)
    preset = compose(["+experiment=walker_laplace_100k"])
    OmegaConf.resolve(preset)
    ens = compose(["+experiment=walker_ensemble_100k"])
    OmegaConf.resolve(ens)
    lcg = compose(["+experiment=walker_lcg_100k"])
    OmegaConf.resolve(lcg)
    assert preset.intrinsic_reward.method == "laplace_disagreement" and preset.training.compile_wm is False
    for key in ("collection", "denoiser", "milestones", "evaluation", "env", "actor_critic", "agent",
                "world_model_env", "common"):
        assert OmegaConf.to_container(preset[key]) == OmegaConf.to_container(ens[key]), key
    for key in ("precision_reference_size", "precision_num_mc", "damping", "beta"):
        assert preset.intrinsic_reward[key] == lcg.intrinsic_reward[key], key
    assert OmegaConf.to_container(preset.intrinsic_reward.theta_s) == OmegaConf.to_container(lcg.intrinsic_reward.theta_s)
    # LCG uses the same exploration controller (advantage-weighted) and the default method is LCG.
    assert OmegaConf.to_container(lcg.agent) == OmegaConf.to_container(preset.agent)
    assert lcg.agent.actor_critic.actor_objective == "advantage_weighted" and compose([]).intrinsic_reward._target_ == "lcg.LCGConfig"
    # The R1 guards hold on the real architecture (10 tensors, 217,664 scalars).
    from hydra.utils import instantiate
    den_cfg = instantiate(preset.agent.denoiser)
    den_cfg.inner_model.continuous_action_dim = 6
    lap = LaplaceEnsemble(LaplaceConfig(**{k: v for k, v in OmegaConf.to_container(preset.intrinsic_reward).items()
                                           if k != "_target_"}), Denoiser(den_cfg), SIGMA, 0, torch.device("cpu"))
    assert (len(lap.names), lap.dim, lap.prefix_supported) == (10, 217664, True)


@pytest.mark.parametrize("overrides,match", [
    (dict(num_members=1), "num_members must be an integer >= 2"),
    (dict(posterior_temperature=0.0), "posterior_temperature must be a finite positive"),
    (dict(posterior_temperature=float("nan")), "posterior_temperature must be a finite positive"),
    (dict(damping=0.0), "damping must be a finite positive"),
    (dict(precision_reference_size=0), "precision_reference_size must be a positive integer"),
    (dict(reward_scaling="rms"), "reward_scaling must be one of"),
    (dict(fixed_divisor=0.0), "fixed_divisor must be a finite positive"),
    (dict(member_selection="best"), "member_selection must be one of"),
])
def test_invalid_settings_are_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        LaplaceConfig(**overrides)


@pytest.mark.parametrize("overrides,match", [
    (["intrinsic_reward.num_members=1"], "num_members must be an integer >= 2"),
    (["training.compile_wm=true"], "not compiled"),
    (["world_model_env.diffusion_sampler.s_churn=0.5"], "needs s_churn=0"),
])
def test_trainer_rejects_bad_settings_before_any_work(tmp_path, monkeypatch, overrides, match):
    with pytest.raises(ValueError, match=match):
        run_lap(tmp_path, monkeypatch, overrides, build_only=True)
    assert not (tmp_path / "run" / "checkpoints").exists()


def test_selection_guards_and_bad_patterns_fail():
    with pytest.raises(ValueError, match="matched no parameters"):
        make_laplace(include=("unet.nope.*",))
    with pytest.raises(ValueError, match="expected 99 tensors"):
        make_laplace(theta_s=ThetaSConfig(include=(R1_TINY,), expected_num_tensors=99))


# ---------------------------------------------------------------- precision reuse

def test_precision_is_lcgs_exactly(tmp_path):
    dataset = random_dataset(tmp_path / "d")
    denoiser = tiny_denoiser()
    lap = make_laplace(denoiser)
    lap.refresh(dataset, epoch=4)
    lcg_cfg = LCGConfig(enabled=True, theta_s=ThetaSConfig(include=(R1_TINY,)), precision_reference_size=6,
                        precision_num_mc=2, damping=1e-4, beta=1.0, candidate_num_mc=1)
    lifecycle = LCGLifecycle(lcg_cfg, SIGMA, img_channels=3, img_size=8, device=torch.device("cpu"))
    lifecycle.refresh(denoiser, dataset, round_identifier=4)
    torch.testing.assert_close(lap.precision, lifecycle.h_D, rtol=0, atol=0)


def test_damping_and_replay_scale_are_applied_once(tmp_path, monkeypatch):
    dataset = random_dataset(tmp_path / "d")
    lap = make_laplace(damping=3e-3, beta=2.0)
    monkeypatch.setattr(precision_module, "_backward_vjp_probe", lambda d, params, *a, **k: torch.cat(
        [torch.zeros(p.numel()) for p in params]))
    with pytest.raises(ValueError, match="finite and strictly positive"):
        lap.cfg.damping = 0.0  # a zero-curvature precision without damping must not fall back
        lap.refresh(dataset, epoch=1)
    lap.cfg.damping = 3e-3
    lap.refresh(dataset, epoch=1)
    assert torch.equal(lap.precision, torch.full((lap.dim,), 3e-3))  # damping exactly once
    monkeypatch.setattr(precision_module, "_backward_vjp_probe", lambda d, params, *a, **k: torch.cat(
        [torch.ones(p.numel()) for p in params]))
    lap.refresh(dataset, epoch=2)
    # Probes averaged (1 per transition), summed over B and scaled by N / B: damping + beta * N.
    torch.testing.assert_close(lap.precision, torch.full((lap.dim,), 3e-3 + 2.0 * dataset.num_steps), rtol=1e-6, atol=0)


# ---------------------------------------------------------------- Gaussian sampling

def test_covariance_factor_and_gaussian_draws(tmp_path, monkeypatch):
    h = torch.tensor([4.0, 0.25, 1.0])
    torch.testing.assert_close(perturbation_scale(h, 1.0), torch.tensor([0.5, 2.0, 1.0]))
    torch.testing.assert_close(perturbation_scale(h, 4.0), torch.tensor([1.0, 4.0, 2.0]))  # std scales with sqrt(T)
    for bad in (torch.tensor([1.0, 0.0]), torch.tensor([1.0, -2.0]), torch.tensor([1.0, float("nan")])):
        with pytest.raises(ValueError, match="finite and strictly positive"):
            perturbation_scale(bad, 1.0)
    dataset = random_dataset(tmp_path / "d")
    lap = make_laplace(num_members=4000, posterior_temperature=2.0)
    known = torch.linspace(0.5, 8.0, lap.dim)
    monkeypatch.setattr(laplace_module, "lcg_historical_precision", lambda *a, **k: known.clone())
    rng_before = rng_state()
    lap.refresh(dataset, epoch=3)
    assert same(rng_before, rng_state())  # private generator only (no global or CUDA reseeding)
    eps = torch.randn(4000, lap.dim, generator=torch.Generator().manual_seed(lap.bank_seed(3)))
    torch.testing.assert_close(lap.perturbations, eps * (2.0 / known).sqrt(), rtol=1e-6, atol=0)
    var = lap.perturbations.var(dim=0)
    ratio = var / (2.0 / known)
    assert 0.9 < float(ratio.mean()) < 1.1 and float(lap.perturbations.mean(0).abs().max()) < 0.2 * float((2.0 / known).sqrt().max())
    first = lap.perturbations.clone()
    lap.refresh(dataset, epoch=3)
    assert torch.equal(first, lap.perturbations)  # reproducible per refresh epoch
    lap.refresh(dataset, epoch=4)
    assert not torch.equal(first, lap.perturbations)


def test_members_change_only_selected_parameters_in_order(tmp_path):
    dataset = random_dataset(tmp_path / "d")
    denoiser = tiny_denoiser()
    lap = make_laplace(denoiser)
    lap.refresh(dataset, epoch=1)
    named = dict(denoiser.inner_model.named_parameters())
    selected = selected_named_parameters(denoiser, ThetaSConfig(include=(R1_TINY,)))
    assert lap.names == list(selected) and lap.names == [n for n in named if n.startswith("unet.u_blocks.1.resblocks.0.")]
    for k, params in enumerate(lap.member_params()):
        assert list(params) == lap.names  # substitution dict holds the selected tensors only
        flat = torch.cat([params[n].reshape(-1) for n in lap.names])
        torch.testing.assert_close(flat - lap.selected_flat(), lap.perturbations[k], rtol=0, atol=1e-6)
        for name in lap.names:
            assert params[name].shape == named[name].shape and not torch.equal(params[name], named[name])


# ---------------------------------------------------------------- sampler semantics and isolation

def test_each_member_is_a_fixed_parameter_chain_from_shared_noise(tmp_path):
    dataset = random_dataset(tmp_path / "d")
    denoiser = tiny_denoiser()
    lap = make_laplace(denoiser)
    lap.refresh(dataset, epoch=1)
    obs, act, noise = inputs()
    sampler = DiffusionSampler(denoiser, SAMPLER)
    before = snapshot(denoiser)
    predictions = lap.predict_members(sampler, obs, act, noise, reuse_prefix=False)
    assert predictions.shape == (3, 4, 3, 8, 8)
    for k, params in enumerate(lap.member_params()):
        # The same chain on an independent copy holding member k's weights throughout.
        expected = DiffusionSampler(materialized(denoiser, params), SAMPLER).sample(obs, act, noise)[0]
        torch.testing.assert_close(predictions[k], expected, rtol=0, atol=0)
    assert not torch.equal(predictions[0], predictions[1])
    assert same(before, snapshot(denoiser))  # base weights and buffers untouched
    # Population variance (divisor K), as the full ensemble.
    centered = predictions - predictions.mean(0, keepdim=True)
    torch.testing.assert_close(ensemble_disagreement(predictions), centered.square().mean(0).flatten(1).mean(1))


def test_shared_prefix_matches_the_full_forward_reference(tmp_path):
    dataset = random_dataset(tmp_path / "d")
    denoiser = tiny_denoiser()
    lap = make_laplace(denoiser)
    lap.refresh(dataset, epoch=1)
    obs, act, noise = inputs(b=5, seed=3)
    sampler = DiffusionSampler(denoiser, SAMPLER)
    params = lap.member_params()
    # First evaluation: shared prefix vs a full forward per member, same input.
    flat_obs = obs.reshape(5, 6, 8, 8)
    first = shared_prefix_first_evaluation(denoiser, sampler.sigmas[0], noise, flat_obs, act, params)
    for f, p in zip(first, params):
        torch.testing.assert_close(f, denoiser.denoise(noise, sampler.sigmas[0], flat_obs, act, params=p), rtol=0, atol=0)
    optimized = lap.predict_members(sampler, obs, act, noise, reuse_prefix=True)
    reference = lap.predict_members(sampler, obs, act, noise, reuse_prefix=False)
    torch.testing.assert_close(optimized, reference, rtol=0, atol=0)  # stated tolerance: exact on CPU
    torch.testing.assert_close(ensemble_disagreement(optimized), ensemble_disagreement(reference), rtol=0, atol=0)
    # After the first update the members' images differ, so the prefix cannot be reused later:
    # reusing member 0's second-evaluation activations for member 1 would give another result.
    s0, s1 = sampler.sigmas[0], sampler.sigmas[1]
    x1 = [noise + (noise - d) / s0 * (s1 - s0) for d in first]
    assert not torch.equal(x1[0], x1[1])
    stale = shared_prefix_first_evaluation(denoiser, s1, x1[0], flat_obs, act, params)[1]
    correct = denoiser.denoise(x1[1], s1, flat_obs, act, params=params[1])
    assert not torch.allclose(stale, correct)


def test_unsupported_selection_uses_the_full_forward_path(tmp_path):
    dataset = random_dataset(tmp_path / "d")
    lap = make_laplace(include=("unet.d_blocks.0.*",))
    assert not lap.prefix_supported and not lap.use_prefix
    lap.refresh(dataset, epoch=1)
    obs, act, noise = inputs()
    sampler = DiffusionSampler(lap.denoiser, SAMPLER)
    with pytest.raises(ValueError, match="not supported"):
        lap.predict_members(sampler, obs, act, noise, reuse_prefix=True)
    out = lap.predict_members(sampler, obs, act, noise)
    for k, params in enumerate(lap.member_params()):
        expected = DiffusionSampler(materialized(lap.denoiser, params), SAMPLER).sample(obs, act, noise)[0]
        torch.testing.assert_close(out[k], expected, rtol=0, atol=0)


def test_base_weights_buffers_and_modes_survive_an_exception(tmp_path):
    dataset = random_dataset(tmp_path / "d")
    denoiser = tiny_denoiser().train()
    denoiser.inner_model.unet.eval()
    lap = make_laplace(denoiser)
    lap.refresh(dataset, epoch=1)
    before, tensors = snapshot(denoiser), {n: p for n, p in denoiser.named_parameters()}
    modes = {n: m.training for n, m in denoiser.named_modules()}
    sampler = DiffusionSampler(denoiser, SAMPLER)
    original = type(denoiser.inner_model.conv_out).forward
    calls = []

    def failing(self, x):
        calls.append(1)
        if len(calls) == 2:  # inside a member's substituted forward
            raise RuntimeError("boom")
        return original(self, x)

    obs, act, noise = inputs()
    for reuse in (True, False):
        calls.clear()
        torch.nn.Conv2d.forward = failing
        try:
            with pytest.raises(RuntimeError, match="boom"):
                lap.predict_members(sampler, obs, act, noise, reuse_prefix=reuse)
        finally:
            torch.nn.Conv2d.forward = original
        assert same(before, snapshot(denoiser))
        assert all(p is tensors[n] for n, p in denoiser.named_parameters())  # original Parameters restored
        assert {n: m.training for n, m in denoiser.named_modules()} == modes


def test_imagination_scores_and_selects_like_the_full_ensemble(tmp_path):
    dataset = random_dataset(tmp_path / "d", lengths=(12, 12))
    denoiser = tiny_denoiser()
    lap = make_laplace(denoiser)
    lap.refresh(dataset, epoch=1)
    recorded = []

    def predictor(sampler, obs, act, noise):
        out = lap.predict_members(sampler, obs, act, noise)
        recorded.append(out)
        return out

    loader = DataLoader(dataset, batch_sampler=BatchSampler(dataset, 0, 1, 4, 2), collate_fn=collate_segments_without_rewards)
    env = WorldModelEnv(denoiser, None, loader, WorldModelEnvConfig(horizon=5, num_batches_to_preload=1,
                                                                     diffusion_sampler=SAMPLER), member_predictor=predictor)
    np.random.seed(0)
    torch.manual_seed(0)
    env.reset()
    for _ in range(2):
        next_obs, *_, info = env.step(torch.zeros(4, 2))
        predictions = recorded[-1]
        torch.testing.assert_close(info["ensemble_disagreement"], ensemble_disagreement(predictions), rtol=0, atol=0)
        member = info["ensemble_member"]
        assert member.shape == (4,) and int(member.max()) < 3
        torch.testing.assert_close(next_obs, predictions[member, torch.arange(4)], rtol=0, atol=0)
    with pytest.raises(ValueError, match="either an ensemble"):
        WorldModelEnv(denoiser, None, loader, WorldModelEnvConfig(5, 1, SAMPLER), ensemble=[denoiser, tiny_denoiser()],
                      member_predictor=predictor)


# ---------------------------------------------------------------- trainer integration

def test_trainer_runs_one_denoiser_with_frozen_bank(tmp_path, monkeypatch):
    banks = []
    original = LaplaceEnsemble.refresh

    def refresh(self, dataset, epoch):
        logs = original(self, dataset, epoch)
        banks.append((self.perturbations.clone(), self.precision.clone()))
        return logs

    monkeypatch.setattr(LaplaceEnsemble, "refresh", refresh)
    trainer, logs = run_lap(tmp_path, monkeypatch, OFF)
    assert len(trainer.agent.denoisers) == 1 and trainer.ensemble is None and trainer.laplace.refreshes == 2
    assert trainer.agent.actor_critic.updates.item() == 3 and trainer.num_batch_train.denoiser == 3
    assert trainer.agent.actor_critic.intrinsic_reward_fn is trainer._ensemble_scorer
    # The checkpointed bank is the last refresh's, unchanged by PMPO.
    assert torch.equal(trainer.laplace.perturbations, banks[-1][0]) and not torch.equal(banks[0][0], banks[1][0])
    keys = set().union(*logs)
    for key in ("laplace/precision_median", "laplace/member_2_relative_perturbation", "laplace/selected_dim",
                "timing/laplace_refresh_seconds", "timing/laplace_prediction_seconds",
                "timing/laplace_precision_seconds", "actor_critic/train/intrinsic/raw_reward_mean",
                "actor_critic/train/intrinsic/scaled_reward_max"):
        assert key in keys, key
    assert all(row["timing/laplace_shared_prefix_evaluations"] > 0 for row in logs if "timing/laplace_shared_prefix_evaluations" in row)


def test_changing_task_rewards_does_not_change_laplace(tmp_path, monkeypatch):
    runs = []
    for scale in (0.0, 1e6):
        root = tmp_path / f"scale{scale:g}"
        write_static_dataset(root / "static", scale)
        monkeypatch.setattr(base, "TINY", LAP)
        runs.append(run_trainer(root, monkeypatch, [f"static_dataset.path={(root / 'static').as_posix()}",
                                                    "training.num_final_epochs=2", "evaluation.every=1"]))
    (a, logs_a), (b, logs_b) = runs
    assert a.train_dataset.load_episode(0).rew.abs().max() == 0 and b.train_dataset.load_episode(0).rew.abs().max() > 1
    assert same(a.laplace.state_dict(), b.laplace.state_dict()) and same(a.agent.state_dict(), b.agent.state_dict())

    def strip(rows):
        return [{k: v for k, v in r.items() if "/return_" not in k} for r in base.numeric_logs(rows)]
    assert strip(logs_a) == strip(logs_b)


def test_crash_resume_is_identical_and_changed_settings_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "TINY", LAP)
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
    saved = torch.load(tmp_path / "split/checkpoints/state.pt", weights_only=False)["laplace"]
    assert saved["perturbations"].shape == (3, crashed.laplace.dim) and saved["refresh_epoch"] == 1
    loads = []
    monkeypatch.setattr(torch.nn.Module, "load_state_dict",
                        (lambda f: lambda self, *a, **k: loads.append(1) or f(self, *a, **k))(torch.nn.Module.load_state_dict))
    for override, match in (("intrinsic_reward.num_members=4", "Laplace settings"),
                            ("intrinsic_reward.posterior_temperature=2.0", "Laplace settings"),
                            ("intrinsic_reward.theta_s.include=[unet.u_blocks.1.*]", "Laplace settings"),
                            ("intrinsic_reward.precision_reference_size=7", "Laplace settings"),
                            ("intrinsic_reward.precision_num_mc=2", "Laplace settings"),
                            ("intrinsic_reward.damping=0.001", "Laplace settings"),
                            ("intrinsic_reward.beta=2.0", "Laplace settings"),
                            ("intrinsic_reward.shared_prefix=false", "Laplace settings"),
                            ("intrinsic_reward.num_noise_samples=2", "must not switch"),
                            ("intrinsic_reward.fixed_divisor=2.0", "must not switch")):
        loads.clear()
        with pytest.raises(ValueError, match=match):
            start(tmp_path / "split", monkeypatch, [override], resume=True)
        assert not loads, override  # refused before any weights were loaded
    resumed, _ = start(tmp_path / "split", monkeypatch, [], resume=True)
    assert torch.equal(resumed.laplace.perturbations, saved["perturbations"])  # exact bank restored
    resumed._run()
    assert same(full.agent.state_dict(), resumed.agent.state_dict())
    assert same(full.opt.state_dict(), resumed.opt.state_dict())
    assert same(full.laplace.state_dict(), resumed.laplace.state_dict())
    for k in ("python", "numpy", "torch"):
        assert same(full_rng[k], rng_state()[k])


def test_checkpoint_compatibility_across_methods(tmp_path, monkeypatch):
    lap, _ = run_lap(tmp_path / "lap", monkeypatch, OFF, build_only=True)
    lap_state = torch.load(tmp_path / "lap/run/checkpoints/state.pt", weights_only=False)
    monkeypatch.setattr(base, "TINY", TINY_LCG)
    lcg, _ = run_trainer(tmp_path / "lcg", monkeypatch, COLLECT, build_only=True)
    lcg_state = torch.load(tmp_path / "lcg/run/checkpoints/state.pt", weights_only=False)
    assert lcg_state["laplace"] is None and lcg.laplace is None and lcg.agent.actor_critic.rl_env.member_predictor is None
    with pytest.raises(ValueError, match="must not switch"):
        lcg.load_state_dict(lap_state)
    lap_state["lcg_reward_scaling"] = lcg_state["lcg_reward_scaling"]
    with pytest.raises(ValueError, match="Laplace sample bank"):
        lcg.load_state_dict(lap_state)
    lcg_state["lcg_reward_scaling"] = lap._lcg_reward_scaling
    with pytest.raises(ValueError, match="Laplace settings None"):
        lap.load_state_dict(lcg_state)
    # Older checkpoints (no key) of other methods still resume.
    monkeypatch.setattr(diag, "TINY", TINY_LCG)
    run, _ = start(tmp_path / "old", monkeypatch, BASE + OFF)
    run._run()
    path = tmp_path / "old/checkpoints/state.pt"
    state = torch.load(path, weights_only=False)
    del state["laplace"]
    torch.save(state, path)
    resumed, _ = start(tmp_path / "old", monkeypatch, [], resume=True)
    assert resumed.laplace is None and resumed.epoch == run.epoch


def test_milestones_hold_the_unperturbed_denoiser_and_load_downstream(tmp_path, monkeypatch):
    trainer, _ = run_lap(tmp_path, monkeypatch, OFF + ["milestones.at_steps=[24]"])
    path = Path("milestones/steps_000024")
    milestone = load_milestone(path, "walk")
    manifest = milestone.manifest
    assert sorted(manifest["files"]) == ["dataset", "denoiser"] and "ensemble" not in manifest
    assert manifest["exploration"]["method"] == "laplace_disagreement" and manifest["laplace"]["num_members"] == 3
    assert same(milestone.denoiser_state, trainer.agent.denoiser.state_dict())  # base, not a sample
    monkeypatch.syspath_prepend(str(base.ROOT / "tests" / "downstream"))
    import test_downstream_trainer as down
    single = down.build_milestone(tmp_path / "down")
    m = load_milestone(single, "walk")
    keep = {k: read_manifest(single)[k] for k in ("created_by", "domain", "task_names", "collected_steps",
                                                   "num_episodes", "epoch", "counters", "model", "run_config", "source")}
    keep["exploration"] = dict(collection_mode="policy", intrinsic_reward_enabled=True, method="laplace_disagreement")
    keep["laplace"] = manifest["laplace"]
    lap_path = publish_milestone(single.parent, "steps_laplace", m.denoiser_state, m.dataset, m.successor_observations, keep)
    downstream = down.make_trainer(lap_path, monkeypatch)
    assert down.states_equal(m.denoiser_state, downstream.agent.denoiser.state_dict())
    assert (downstream.run() / "summary.yaml").is_file()
