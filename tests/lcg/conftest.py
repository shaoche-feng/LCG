"""
Shared fixtures for the permanent LCG regression suite. Deliberately tiny (CPU, small
UNet, small image size) so the whole suite runs in seconds without a GPU or the large
480-candidate/checkpoint diagnostic fixtures -- those fixtures were pre-offset-noise-fix
anyway (see docs/lcg_diagnostic/checkpoint/PRE_OFFSET_NOISE_FIX_README.txt) and must not
be used as expected-value fixtures here.
"""
import sys
from pathlib import Path

import pytest
import torch


def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while not ((p / "src").is_dir() and (p / "scripts").is_dir()):
        if p.parent == p:
            raise RuntimeError("could not locate LCG repo root")
        p = p.parent
    return p


_REPO_ROOT = _find_repo_root(Path(__file__).parent)
sys.path.insert(0, str(_REPO_ROOT / "src"))

from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig  # noqa: E402
from models.diffusion.inner_model import InnerModelConfig  # noqa: E402

TINY_IMG_CHANNELS = 3
TINY_IMG_SIZE = 8
TINY_NUM_STEPS_CONDITIONING = 1
TINY_ACTION_DIM = 2
TINY_SIGMA_OFFSET_NOISE = 0.3
TINY_SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)


@pytest.fixture(scope="session")
def tiny_denoiser() -> Denoiser:
    """A freshly-initialized, real (not mocked) Denoiser with a minimal 1-level UNet --
    exercises the exact same GroupNorm/AdaGroupNorm/conv code path as the production
    model, just tiny enough to run explicit-Jacobian construction on CPU in milliseconds.
    conv_out.weight is zero-initialized (matches production init), which does not affect
    JVP-vs-explicit-Jacobian correctness: a directional derivative through a layer that is
    linear in its own weight does not vanish just because the current weight value is
    zero."""
    torch.manual_seed(0)
    inner_cfg = InnerModelConfig(
        img_channels=TINY_IMG_CHANNELS,
        num_steps_conditioning=TINY_NUM_STEPS_CONDITIONING,
        cond_channels=16,
        depths=[1],
        channels=[8],
        attn_depths=[False],
        continuous_action_dim=TINY_ACTION_DIM,
    )
    cfg = DenoiserConfig(inner_model=inner_cfg, sigma_data=0.5, sigma_offset_noise=TINY_SIGMA_OFFSET_NOISE)
    denoiser = Denoiser(cfg)
    denoiser.eval()
    return denoiser


@pytest.fixture
def tiny_transition():
    """One (obs, act, y) transition shaped for tiny_denoiser, batch size 1."""
    n = TINY_NUM_STEPS_CONDITIONING
    obs = torch.randn(1, n * TINY_IMG_CHANNELS, TINY_IMG_SIZE, TINY_IMG_SIZE)
    act = torch.randn(1, n, TINY_ACTION_DIM)
    y = torch.randn(1, TINY_IMG_CHANNELS, TINY_IMG_SIZE, TINY_IMG_SIZE)
    return obs, act, y
