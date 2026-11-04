"""LCG policy-reward scaling (fixed divisor vs RunningRMS), its configuration, and the R1
theta_S selection on the real denoiser architecture."""
import math
from copy import deepcopy

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from intrinsic_reward.lcg.lifecycle import LCGConfig, LCGLifecycle
from intrinsic_reward.lcg.reward_normalization import RunningRMS, RunningRMSConfig
from intrinsic_reward.lcg.theta_s import ThetaSConfig, selected_named_parameters
from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.diffusion.inner_model import InnerModelConfig

SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)
R1 = ThetaSConfig(include=("unet.u_blocks.3.resblocks.0.*",), expected_num_tensors=10, expected_dim=217_664)


def config(**overrides):
    kwargs = dict(enabled=True, theta_s=ThetaSConfig(include=("conv_out.*",)))
    kwargs.update(overrides)
    return LCGConfig(**kwargs)


def fixed_lifecycle(divisor=100.0):
    cfg = config(reward_scaling="fixed", fixed_divisor=divisor, rms_enabled=False)
    return LCGLifecycle(cfg, SIGMA_CFG, 3, 8, torch.device("cpu"))


# ---------------------------------------------------------------- fixed scaling

def test_fixed_scaling_divides_by_a_constant_across_batches_and_rounds():
    lifecycle = fixed_lifecycle(100.0)
    generator = torch.Generator().manual_seed(0)
    for _ in range(5):  # batches of very different magnitude and sign
        raw = torch.randn(4, 15, generator=generator) * 10 ** torch.randint(-3, 6, (1,), generator=generator).item()
        torch.testing.assert_close(lifecycle.scale(raw), raw / 100.0, rtol=0, atol=0)
    assert lifecycle.rms is None  # no running statistics exist in this mode
    # Declining novelty is not compensated: a smaller raw score gives a proportionally smaller reward.
    assert lifecycle.scale(torch.tensor([1e-6])).item() == pytest.approx(1e-8, rel=1e-6)
    # Not centered or clipped: negative and huge raw values pass through divided.
    raw = torch.tensor([-5.0, 0.0, 3e8])
    torch.testing.assert_close(lifecycle.scale(raw), torch.tensor([-0.05, 0.0, 3e6]), rtol=0, atol=0)


def test_fixed_scaling_is_independent_of_outliers_unlike_rms():
    batch = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    outlier = batch.clone()
    outlier[0, 3] = 1e9
    lifecycle = fixed_lifecycle(100.0)
    # Same batch and later batches: the other entries' rewards are unchanged by an outlier.
    torch.testing.assert_close(lifecycle.scale(outlier)[0, :3], lifecycle.scale(batch)[0, :3], rtol=0, atol=0)
    torch.testing.assert_close(lifecycle.scale(batch), batch / 100.0, rtol=0, atol=0)
    # RunningRMS (the explicit alternative) is not: the outlier shrinks everything, now and later.
    rms = RunningRMS(RunningRMSConfig(enabled=True))
    with_outlier = rms(outlier)[0, :3]
    assert (with_outlier < RunningRMS(RunningRMSConfig(enabled=True))(batch)[0, :3] * 1e-3).all()
    assert (rms(batch) < batch * 1e-3).all()


def test_rms_mode_still_applies_running_rms(tmp_path):
    cfg = config()  # defaults: rms_enabled=True, no reward_scaling -> rms
    assert cfg.reward_scaling == "rms"
    lifecycle = LCGLifecycle(cfg, SIGMA_CFG, 3, 8, torch.device("cpu"))
    lifecycle.rms = RunningRMS(RunningRMSConfig(enabled=True))  # what refresh() builds in this mode
    raw = torch.tensor([[2.0, -2.0]])
    torch.testing.assert_close(lifecycle.scale(raw), raw / (2.0 + 1e-8), rtol=1e-6, atol=0)


# ---------------------------------------------------------------- configuration

@pytest.mark.parametrize("divisor", [0, 0.0, -100.0, math.inf, -math.inf, math.nan, None, True, "100"])
def test_fixed_divisor_must_be_finite_and_positive(divisor):
    with pytest.raises(ValueError, match="fixed_divisor must be a finite positive number"):
        config(reward_scaling="fixed", fixed_divisor=divisor, rms_enabled=False)


@pytest.mark.parametrize("overrides,match", [
    (dict(reward_scaling="fixed", fixed_divisor=100.0, rms_enabled=True), "conflicts with rms_enabled=True"),
    (dict(reward_scaling="rms", rms_enabled=False), "conflicts with rms_enabled=False"),
    (dict(reward_scaling="none", rms_enabled=True), "conflicts with rms_enabled=True"),
    (dict(reward_scaling="rms", fixed_divisor=100.0), "used only with reward_scaling=fixed"),
    (dict(reward_scaling="clip", rms_enabled=False), "must be one of"),
])
def test_inconsistent_scaling_settings_are_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        config(**overrides)


def test_older_configs_keep_their_meaning():
    # Saved configs from before reward_scaling existed: rms_enabled alone decides.
    old = OmegaConf.create(dict(_target_="intrinsic_reward.lcg.LCGConfig", enabled=True, rms_enabled=True,
                                theta_s=dict(_target_="intrinsic_reward.lcg.theta_s.ThetaSConfig", include=["conv_out.*"], exclude=[])))
    assert instantiate(old).reward_scaling == "rms"
    old.rms_enabled = False
    cfg = instantiate(old)
    assert cfg.reward_scaling == "none" and cfg.reward_scaling_settings() == dict(mode="none")
    assert config(reward_scaling="fixed", fixed_divisor=100, rms_enabled=False).reward_scaling_settings() == dict(
        mode="fixed", fixed_divisor=100.0)


# ---------------------------------------------------------------- R1 selection

def real_architecture_denoiser():
    torch.manual_seed(0)
    inner = InnerModelConfig(img_channels=3, num_steps_conditioning=4, cond_channels=256, depths=[2, 2, 2, 2],
                             channels=[64, 64, 64, 64], attn_depths=[0, 0, 0, 0], continuous_action_dim=6)
    return Denoiser(DenoiserConfig(inner_model=inner, sigma_data=0.5, sigma_offset_noise=0.3))


def test_r1_selects_the_first_resblock_of_the_last_decoder_level():
    denoiser = real_architecture_denoiser()
    selected = selected_named_parameters(denoiser, R1)
    block = denoiser.inner_model.unet.u_blocks[3].resblocks[0]
    assert list(selected) == [f"unet.u_blocks.3.resblocks.0.{name}" for name, _ in block.named_parameters()]
    assert all(selected[f"unet.u_blocks.3.resblocks.0.{n}"] is p for n, p in block.named_parameters())
    assert len(selected) == 10 and sum(p.numel() for p in selected.values()) == 217_664
    assert {n.split(".")[-2] for n in selected} == {"proj", "linear", "conv1", "conv2"}


@pytest.mark.parametrize("guards", [dict(expected_num_tensors=11), dict(expected_dim=217_665),
                                    dict(expected_num_tensors=10, expected_dim=1)])
def test_r1_guards_fail_clearly_on_a_different_block(guards):
    with pytest.raises(ValueError, match="expected .* tensors and .* scalars"):
        selected_named_parameters(real_architecture_denoiser(), ThetaSConfig(include=R1.include, **guards))
    # Same guards on a different architecture (another block size) also fail.
    small = InnerModelConfig(img_channels=3, num_steps_conditioning=4, cond_channels=256, depths=[2, 2, 2, 2],
                             channels=[32, 32, 32, 32], attn_depths=[0, 0, 0, 0], continuous_action_dim=6)
    other = Denoiser(DenoiserConfig(inner_model=small, sigma_data=0.5, sigma_offset_noise=0.3))
    with pytest.raises(ValueError, match="expected 10 tensors and 217664 scalars"):
        selected_named_parameters(other, R1)


# ---------------------------------------------------------------- per-round divisors (data_scaled, reference)

def tiny_refresh_setup(tmp_path, steps=(4, 6), **overrides):
    """A refreshable lifecycle on a tiny denoiser and replay (CPU)."""
    from data import Dataset, Episode
    torch.manual_seed(0)
    inner = InnerModelConfig(img_channels=3, num_steps_conditioning=1, cond_channels=16, depths=[1], channels=[8],
                             attn_depths=[False], continuous_action_dim=2)
    denoiser = Denoiser(DenoiserConfig(inner_model=inner, sigma_data=0.5, sigma_offset_noise=0.3)).eval()
    dataset = Dataset(tmp_path / "ds", "scaling_test_ds", cache_in_ram=True)
    for length in steps:
        dataset.add_episode(Episode(obs=torch.randn(length, 3, 8, 8), act=torch.randn(length, 2), rew=torch.zeros(length),
                                    end=torch.zeros(length, dtype=torch.uint8), trunc=torch.zeros(length, dtype=torch.uint8),
                                    info={}))
    cfg = config(precision_reference_size=4, precision_num_mc=1, candidate_num_mc=2, candidate_chunk_size=2,
                 rms_enabled=False, **overrides)
    return LCGLifecycle(cfg, SIGMA_CFG, 3, 8, torch.device("cpu")), denoiser, dataset


def test_data_scaled_divides_by_the_divisor_scaled_to_the_replay_size(tmp_path):
    lifecycle, denoiser, dataset = tiny_refresh_setup(tmp_path, reward_scaling="data_scaled", fixed_divisor=100.0,
                                                      data_reference_steps=5)
    lifecycle.refresh(denoiser, dataset, round_identifier=1)  # N = 10 transitions
    assert lifecycle.reward_divisor == pytest.approx(100.0 * 5 / 10)
    raw = torch.tensor([[2.0, -4.0, 3e6]])
    torch.testing.assert_close(lifecycle.scale(raw), raw / 50.0, rtol=0, atol=0)  # raw / 100 * N / 5
    torch.testing.assert_close(lifecycle.scale_frozen(raw), raw / 50.0, rtol=0, atol=0)
    assert not lifecycle.needs_reference
    # At N = data_reference_steps it is the fixed divisor.
    same, denoiser2, dataset2 = tiny_refresh_setup(tmp_path / "b", steps=(5,), reward_scaling="data_scaled",
                                                   fixed_divisor=100.0, data_reference_steps=5)
    same.refresh(denoiser2, dataset2, round_identifier=1)
    assert same.reward_divisor == 100.0
    # A later refresh on a larger replay rescales.
    dataset.add_episode(dataset.load_episode(0))
    lifecycle.refresh(denoiser, dataset, round_identifier=2)  # N = 14
    assert lifecycle.reward_divisor == pytest.approx(100.0 * 5 / 14)


def test_reference_mode_needs_calibration_and_divides_by_the_reference_mean(tmp_path):
    lifecycle, denoiser, dataset = tiny_refresh_setup(tmp_path, reward_scaling="reference", reference_num_prompts=3)
    lifecycle.refresh(denoiser, dataset, round_identifier=1)
    assert lifecycle.needs_reference and lifecycle.reward_divisor is None
    with pytest.raises(RuntimeError, match="no divisor yet"):
        lifecycle.scale(torch.ones(1, 2))
    for bad in (torch.tensor([]), torch.tensor([-1.0, 0.5]), torch.tensor([math.nan])):
        with pytest.raises(ValueError, match="finite positive mean"):
            lifecycle.calibrate(bad)
    lifecycle.calibrate(torch.tensor([1.0, 2.0, 6.0]))
    assert not lifecycle.needs_reference and lifecycle.reward_divisor == lifecycle.reference_raw_mean == 3.0
    torch.testing.assert_close(lifecycle.scale(torch.tensor([[3.0, 9.0]])), torch.tensor([[1.0, 3.0]]), rtol=0, atol=0)
    # The next refresh needs a new calibration (the divisor belongs to one round).
    lifecycle.refresh(denoiser, dataset, round_identifier=2)
    assert lifecycle.needs_reference and lifecycle.reward_divisor is None
    with pytest.raises(RuntimeError, match="reward_scaling=reference"):
        tiny_refresh_setup(tmp_path / "b", reward_scaling="data_scaled", fixed_divisor=1.0,
                           data_reference_steps=1)[0].calibrate(torch.ones(2))


def test_per_round_divisor_is_checkpointed_with_the_round(tmp_path):
    lifecycle, denoiser, dataset = tiny_refresh_setup(tmp_path, reward_scaling="reference", reference_num_prompts=3)
    lifecycle.refresh(denoiser, dataset, round_identifier=1)
    lifecycle.calibrate(torch.tensor([4.0]))
    state = lifecycle.state_dict()
    restored = LCGLifecycle(lifecycle.cfg, SIGMA_CFG, 3, 8, torch.device("cpu"))
    restored.load_state_dict(state, denoiser)
    assert restored.reward_divisor == restored.reference_raw_mean == 4.0 and not restored.needs_reference
    # A divisor mode cannot resume from a round without one (e.g. a checkpoint from before them).
    state["reward_divisor"] = None
    with pytest.raises(ValueError, match="no reward divisor"):
        LCGLifecycle(lifecycle.cfg, SIGMA_CFG, 3, 8, torch.device("cpu")).load_state_dict(state, denoiser)
    # Modes without a divisor restore checkpoints that lack the field.
    fixed, denoiser2, dataset2 = tiny_refresh_setup(tmp_path / "b", reward_scaling="fixed", fixed_divisor=100.0)
    fixed.refresh(denoiser2, dataset2, round_identifier=1)
    old = fixed.state_dict()
    del old["reward_divisor"], old["reference_raw_mean"]
    again = LCGLifecycle(fixed.cfg, SIGMA_CFG, 3, 8, torch.device("cpu"))
    again.load_state_dict(old, denoiser2)
    assert again.reward_divisor is None


@pytest.mark.parametrize("overrides,match", [
    (dict(reward_scaling="data_scaled", fixed_divisor=100.0), "data_reference_steps must be a positive integer"),
    (dict(reward_scaling="data_scaled", fixed_divisor=100.0, data_reference_steps=0), "data_reference_steps must be"),
    (dict(reward_scaling="data_scaled", data_reference_steps=5000), "fixed_divisor must be a finite positive number"),
    (dict(reward_scaling="reference"), "reference_num_prompts must be a positive integer"),
    (dict(reward_scaling="reference", reference_num_prompts=2.5), "reference_num_prompts must be"),
    (dict(reward_scaling="reference", reference_num_prompts=32, fixed_divisor=1.0), "used only with reward_scaling=fixed"),
    (dict(reward_scaling="fixed", fixed_divisor=1.0, data_reference_steps=5000), "used only with reward_scaling=data_scaled"),
    (dict(reward_scaling="fixed", fixed_divisor=1.0, reference_num_prompts=32), "used only with reward_scaling=reference"),
])
def test_per_round_divisor_settings_are_validated(overrides, match):
    with pytest.raises(ValueError, match=match):
        config(rms_enabled=False, **overrides)


def test_per_round_divisor_settings_are_recorded_for_resume():
    assert config(reward_scaling="data_scaled", fixed_divisor=100, data_reference_steps=5000,
                  rms_enabled=False).reward_scaling_settings() == dict(mode="data_scaled", fixed_divisor=100.0,
                                                                       data_reference_steps=5000)
    assert config(reward_scaling="reference", reference_num_prompts=32,
                  rms_enabled=False).reward_scaling_settings() == dict(mode="reference", reference_num_prompts=32)


def test_reference_imagination_reproduces_the_env_steps_candidates():
    """WorldModelEnv.imagine_candidates from a prompt with given actions gives exactly the
    candidates step() reports from that prompt under the same random stream, and leaves the
    env's own rollout state and its inputs alone."""
    from types import SimpleNamespace
    from torch.utils.data import DataLoader, IterableDataset
    from envs import WorldModelEnv, WorldModelEnvConfig
    from models.diffusion import DiffusionSamplerConfig

    class Replay(IterableDataset):
        def __iter__(self):
            generator = torch.Generator().manual_seed(3)
            while True:
                yield SimpleNamespace(obs=torch.randn(2, 3, 8, 8, generator=generator),
                                      act=torch.rand(2, 2, generator=generator) * 2 - 1, mask=torch.ones(2, dtype=torch.bool))

    def collate(samples):
        return SimpleNamespace(obs=torch.stack([s.obs for s in samples]), act=torch.stack([s.act for s in samples]),
                               mask_padding=torch.stack([s.mask for s in samples]))

    torch.manual_seed(0)
    inner = InnerModelConfig(img_channels=3, num_steps_conditioning=2, cond_channels=16, depths=[1], channels=[8],
                             attn_depths=[False], continuous_action_dim=2)
    denoiser = Denoiser(DenoiserConfig(inner_model=inner, sigma_data=0.5, sigma_offset_noise=0.3)).eval()
    env = WorldModelEnv(denoiser, None, DataLoader(Replay(), batch_size=2, collate_fn=collate),
                        WorldModelEnvConfig(horizon=3, num_batches_to_preload=1,
                                            diffusion_sampler=DiffusionSamplerConfig(num_steps_denoising=2)),
                        return_imagined_candidate=True)
    env.reset()
    prompt_obs, prompt_act = env.obs_buffer.clone(), env.act_buffer.clone()
    inputs = (prompt_obs.clone(), prompt_act.clone())
    actions = torch.rand(3, 2, 2) * 2 - 1
    torch.manual_seed(7)
    stepped = [env.step(a)[4]["imagined_candidate"] for a in actions]
    after = (env.obs_buffer.clone(), env.act_buffer.clone(), env.ep_len.clone())
    torch.manual_seed(7)
    imagined = env.imagine_candidates(prompt_obs, prompt_act, actions)
    assert len(imagined) == 3
    for a, b in zip(stepped, imagined):
        for name in ("x_obs", "x_act", "y_star"):
            torch.testing.assert_close(getattr(b, name), getattr(a, name), rtol=0, atol=0)
    for before, now in zip(after, (env.obs_buffer, env.act_buffer, env.ep_len)):
        torch.testing.assert_close(now, before, rtol=0, atol=0)
    torch.testing.assert_close(prompt_obs, inputs[0], rtol=0, atol=0)
    torch.testing.assert_close(prompt_act, inputs[1], rtol=0, atol=0)


@pytest.mark.parametrize("name,value", [("rms_alpha", math.inf), ("rms_eps", math.inf),
                                       ("rms_alpha", True), ("rms_eps", True),
                                       ("rms_ema_decay", False), ("rms_alpha", "1")])
def test_rms_settings_require_finite_numbers(name, value):
    from intrinsic_reward.ensemble import EnsembleDisagreementConfig
    for construct in (lambda: config(**{name: value}),
                      lambda: EnsembleDisagreementConfig(reward_scaling="rms", **{name: value})):
        with pytest.raises((AssertionError, ValueError), match="rms_"):
            construct()


@pytest.mark.parametrize("value", ["false", 1, None])
def test_rms_enabled_requires_a_boolean(value):
    with pytest.raises(ValueError, match="rms_enabled must be"):
        config(rms_enabled=value)


def test_reference_calibration_is_once_per_ready_round(tmp_path):
    lifecycle, denoiser, dataset = tiny_refresh_setup(tmp_path, reward_scaling="reference", reference_num_prompts=3)
    with pytest.raises(RuntimeError, match="refresh"):
        lifecycle.calibrate(torch.tensor([2.0]))
    lifecycle.refresh(denoiser, dataset, round_identifier=1)
    lifecycle.calibrate(torch.tensor([2.0]))
    with pytest.raises(RuntimeError, match="already calibrated"):
        lifecycle.calibrate(torch.tensor([9.0]))
    assert lifecycle.reward_divisor == lifecycle.reference_raw_mean == 2.0


def test_reference_rejects_negative_scores_even_when_the_mean_is_positive(tmp_path):
    lifecycle, denoiser, dataset = tiny_refresh_setup(tmp_path, reward_scaling="reference", reference_num_prompts=3)
    lifecycle.refresh(denoiser, dataset, round_identifier=1)
    with pytest.raises(ValueError, match="non-negative"):
        lifecycle.calibrate(torch.tensor([-1.0, 3.0]))
    assert lifecycle.needs_reference


@pytest.mark.parametrize("mode", ["reference", "data_scaled"])
@pytest.mark.parametrize("divisor", [0.0, -1.0, math.nan, math.inf, True, "3"])
def test_checkpoint_rejects_invalid_divisors_before_restoring(tmp_path, mode, divisor):
    options = dict(reference_num_prompts=3) if mode == "reference" else dict(fixed_divisor=2.0, data_reference_steps=10)
    lifecycle, denoiser, dataset = tiny_refresh_setup(tmp_path, reward_scaling=mode, **options)
    lifecycle.refresh(denoiser, dataset, round_identifier=1)
    if mode == "reference":
        lifecycle.calibrate(torch.tensor([3.0]))
    state = lifecycle.state_dict()
    state["reward_divisor"] = divisor
    restored = LCGLifecycle(lifecycle.cfg, SIGMA_CFG, 3, 8, torch.device("cpu"))
    with pytest.raises(ValueError, match="reward divisor"):
        restored.load_state_dict(state, denoiser)
    assert not restored.ready


@pytest.mark.parametrize("mean", [None, 0.0, -1.0, math.nan, math.inf, True, 4.0])
def test_reference_checkpoint_requires_its_recorded_mean_to_match(tmp_path, mean):
    lifecycle, denoiser, dataset = tiny_refresh_setup(tmp_path, reward_scaling="reference", reference_num_prompts=3)
    lifecycle.refresh(denoiser, dataset, round_identifier=1)
    lifecycle.calibrate(torch.tensor([3.0]))
    state = lifecycle.state_dict()
    state["reference_raw_mean"] = mean
    with pytest.raises(ValueError, match="reference.*mean"):
        LCGLifecycle(lifecycle.cfg, SIGMA_CFG, 3, 8, torch.device("cpu")).load_state_dict(state, denoiser)


@pytest.mark.parametrize("s2", [torch.tensor(-1.), torch.tensor(math.nan), torch.tensor(math.inf), torch.ones(2)])
def test_rms_checkpoint_rejects_invalid_mean_square(tmp_path, s2):
    from intrinsic_reward.ensemble import EnsembleDisagreementConfig, EnsembleScorer
    lifecycle, denoiser, dataset = tiny_refresh_setup(tmp_path, reward_scaling="fixed", fixed_divisor=2.)
    lifecycle.cfg = config(candidate_num_mc=2, precision_reference_size=4, precision_num_mc=1, candidate_chunk_size=2)
    lifecycle.refresh(denoiser, dataset, round_identifier=1)
    state = lifecycle.state_dict()
    state["rms_s2"] = s2
    with pytest.raises(ValueError, match="RMS"):
        LCGLifecycle(lifecycle.cfg, SIGMA_CFG, 3, 8, torch.device("cpu")).load_state_dict(state, denoiser)
    ensemble = EnsembleScorer(EnsembleDisagreementConfig(reward_scaling="rms"))
    state = ensemble.state_dict()
    state["rms_s2"] = s2
    with pytest.raises(ValueError, match="RMS"):
        ensemble.load_state_dict(state, torch.device("cpu"))


def test_data_scaled_avoids_intermediate_overflow_and_rejects_zero_divisor(tmp_path):
    lifecycle, denoiser, dataset = tiny_refresh_setup(tmp_path, reward_scaling="data_scaled", fixed_divisor=1e308,
                                                      data_reference_steps=10)
    lifecycle.refresh(denoiser, dataset, round_identifier=1)
    assert lifecycle.reward_divisor == 1e308  # (1e308 * 10) / 10 must not overflow
    lifecycle.cfg.fixed_divisor = 5e-324
    lifecycle.cfg.data_reference_steps = 1
    with pytest.raises(ValueError, match="reward divisor"):
        lifecycle.refresh(denoiser, dataset, round_identifier=2)


@pytest.mark.parametrize("fail_at", [None, "imagination", "scoring"])
def test_reference_calibration_prompts_bounds_horizon_scores_and_rng(tmp_path, monkeypatch, fail_at):
    import random
    from types import SimpleNamespace
    import numpy as np
    import exploration_trainer as trainer_module
    from exploration_trainer import ExplorationTrainer
    from envs import WorldModelEnv
    from models.diffusion import DiffusionSampler, DiffusionSamplerConfig
    from utils import rng_state

    lifecycle, denoiser, dataset = tiny_refresh_setup(tmp_path, reward_scaling="reference", reference_num_prompts=3)
    lifecycle.refresh(denoiser, dataset, round_identifier=7)
    trainer = ExplorationTrainer.__new__(ExplorationTrainer)
    trainer._cfg = OmegaConf.create(dict(agent=dict(denoiser=dict(inner_model=dict(num_steps_conditioning=1)))))
    trainer._device, trainer.epoch = torch.device("cpu"), 7
    trainer._lcg_lifecycle, trainer.train_dataset = lifecycle, dataset
    low, high = torch.tensor([-2.0, .3]), torch.tensor([-.7, 3.0])
    trainer._train_env = SimpleNamespace(action_low=low, action_high=high)
    env = WorldModelEnv.__new__(WorldModelEnv)
    env.horizon, env.sampler = 3, DiffusionSampler(denoiser, DiffusionSamplerConfig(num_steps_denoising=2))
    env.obs_buffer, env.act_buffer = torch.ones(1), torch.ones(1)
    env.generator_init = object()  # the live PMPO prompt stream must not be touched
    trainer._rl_env = env
    captured = {}
    original_sample = trainer_module.sample_uniform_historical_transitions
    original_imagine, original_score = env.imagine_candidates, lifecycle.score_candidates

    def sample(ds, count, conditioning, seed):
        assert ds is dataset and count == 3 and conditioning == 0 and seed == 2_000_007
        result = original_sample(ds, count, conditioning, seed=seed)
        captured["segments"] = result
        return result

    def fail():
        random.random(), np.random.random(), torch.rand(1)
        raise RuntimeError("injected calibration failure")

    def imagine(obs, act, actions):
        captured.update(obs=obs.clone(), act=act.clone(), actions=actions.clone())
        if fail_at == "imagination":
            fail()
        return original_imagine(obs, act, actions)

    def score(candidates):
        assert len(candidates) == 3 * 3
        if fail_at == "scoring":
            fail()
        return original_score(candidates)

    monkeypatch.setattr(trainer_module, "sample_uniform_historical_transitions", sample)
    monkeypatch.setattr(env, "imagine_candidates", imagine)
    monkeypatch.setattr(lifecycle, "score_candidates", score)
    before, weights = rng_state(), deepcopy(denoiser.state_dict())
    prompt_stream, round_state = env.generator_init, deepcopy(lifecycle.state_dict())
    if fail_at:
        with pytest.raises(RuntimeError, match="injected calibration failure"):
            trainer.lcg_reference_scores()
    else:
        scores = trainer.lcg_reference_scores()
        assert scores.shape == (9,) and torch.isfinite(scores).all() and (scores >= 0).all()
        again = trainer.lcg_reference_scores()
        torch.testing.assert_close(again, scores, rtol=0, atol=0)
        expected = low + torch.rand(3, 3, 2, generator=torch.Generator().manual_seed(2_000_007)) * (high - low)
        torch.testing.assert_close(captured["actions"], expected, rtol=0, atol=0)
        assert (captured["actions"] >= low).all() and (captured["actions"] <= high).all()
        assert len({(s.episode_id, s.start, s.stop) for s in captured["segments"]}) == 3
        for i, segment in enumerate(captured["segments"]):
            torch.testing.assert_close(captured["obs"][i], dataset[segment].obs, rtol=0, atol=0)
            torch.testing.assert_close(captured["act"][i], dataset[segment].act, rtol=0, atol=0)
    def assert_same(a, b):
        if torch.is_tensor(a):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                assert_same(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                assert_same(x, y)
        else:
            assert a == b
    assert_same(before, rng_state())
    assert_same(weights, denoiser.state_dict())
    assert_same(round_state, lifecycle.state_dict())
    assert denoiser.training is False
    assert env.generator_init is prompt_stream and torch.equal(env.obs_buffer, torch.ones(1))
    assert torch.equal(env.act_buffer, torch.ones(1)) and lifecycle.last_raw is lifecycle.last_scaled is None
