"""LCG policy-reward scaling (fixed divisor vs RunningRMS), its configuration, and the R1
theta_S selection on the real denoiser architecture."""
import math

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from lcg.lifecycle import LCGConfig, LCGLifecycle
from lcg.reward_normalization import RunningRMS, RunningRMSConfig
from lcg.theta_s import ThetaSConfig, selected_named_parameters
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
    old = OmegaConf.create(dict(_target_="lcg.LCGConfig", enabled=True, rms_enabled=True,
                                theta_s=dict(_target_="lcg.theta_s.ThetaSConfig", include=["conv_out.*"], exclude=[])))
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
