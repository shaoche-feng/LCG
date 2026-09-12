"""
Permanent regression tests for the configurable theta_S selector (lcg.theta_s). theta_S
used to be hard-coded in Python (the final decoder u_block + norm_out + conv_out); it is
now an experiment/configuration choice (ThetaSConfig.include/exclude glob patterns),
resolved by ONE canonical function, selected_named_parameters, whose selection ORDER
always follows the model's own denoiser.inner_model.named_parameters() traversal --
never the order patterns are written in config.
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


sys.path.insert(0, str(_find_repo_root(Path(__file__).parent) / "src"))

from lcg.precision import assert_setup_valid, historical_precision  # noqa: E402
from lcg.forward_jvp import make_jvp_bank, score_one_jvp_bank  # noqa: E402
from lcg.theta_s import (  # noqa: E402
    ThetaSConfig,
    frozen_named_parameters,
    selected_dim,
    selected_named_parameters,
    selected_parameter_names,
    selected_parameters,
)
from data import Dataset, Episode  # noqa: E402
from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig  # noqa: E402
from models.diffusion.inner_model import InnerModelConfig  # noqa: E402

SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)


def _real_arch_denoiser() -> Denoiser:
    """A fresh (untrained -- structure/selection only, no checkpoint needed), REAL-
    architecture Denoiser: depths=[2,2,2,2] matches config/agent/default.yaml exactly, so
    this exercises the actual production default patterns ("unet.u_blocks.3.*" etc.), not
    just the single-u_block tiny_denoiser used elsewhere in this suite."""
    torch.manual_seed(0)
    inner_cfg = InnerModelConfig(
        img_channels=3, num_steps_conditioning=4, cond_channels=256,
        depths=[2, 2, 2, 2], channels=[64, 64, 64, 64], attn_depths=[0, 0, 0, 0],
        continuous_action_dim=6,
    )
    cfg = DenoiserConfig(inner_model=inner_cfg, sigma_data=0.5, sigma_offset_noise=0.3)
    denoiser = Denoiser(cfg)
    denoiser.eval()
    return denoiser


def _old_hardcoded_selection(denoiser: Denoiser):
    """Reconstructs the DELETED hard-coded selection (src/lcg/theta_s.py's old
    selected_submodules + forward_jvp.py's old selected_named_parameters) locally, purely
    as an independent reference for proving the new config-driven default is behaviorally
    identical -- production no longer contains this logic anywhere."""
    inner = denoiser.inner_model
    last_idx = len(inner.unet.u_blocks) - 1
    prefixed_modules = [
        (f"unet.u_blocks.{last_idx}", inner.unet.u_blocks[-1]),
        ("norm_out", inner.norm_out),
        ("conv_out", inner.conv_out),
    ]
    named = {}
    for prefix, module in prefixed_modules:
        for name, p in module.named_parameters():
            named[f"{prefix}.{name}"] = p
    return named


# --------------------------------------------------------------------------------------
# A: current production selection, reproduced exactly via config
# --------------------------------------------------------------------------------------


def test_default_config_reproduces_old_hardcoded_selection_on_real_architecture():
    denoiser = _real_arch_denoiser()
    old_named = _old_hardcoded_selection(denoiser)

    cfg = ThetaSConfig(include=("unet.u_blocks.3.*", "norm_out.*", "conv_out.*"), exclude=())
    new_named = selected_named_parameters(denoiser, cfg)

    assert list(new_named.keys()) == list(old_named.keys())  # exact ordered names
    assert all(new_named[k] is old_named[k] for k in old_named)  # exact parameter objects
    assert len(new_named) == len(old_named) == 34  # exact tensor count
    assert sum(p.numel() for p in new_named.values()) == sum(p.numel() for p in old_named.values()) == 654_851
    assert selected_dim(denoiser, cfg) == 654_851
    assert selected_parameter_names(denoiser, cfg) == list(old_named.keys())
    assert all(a is b for a, b in zip(selected_parameters(denoiser, cfg), old_named.values()))


# --------------------------------------------------------------------------------------
# B/C: narrower selections
# --------------------------------------------------------------------------------------


def test_conv_out_only_selection(tiny_denoiser):
    cfg = ThetaSConfig(include=("conv_out.*",))
    named = selected_named_parameters(tiny_denoiser, cfg)
    assert set(named.keys()) == {"conv_out.weight", "conv_out.bias"}
    ref = dict(tiny_denoiser.inner_model.conv_out.named_parameters())
    assert all(named[f"conv_out.{k}"] is v for k, v in ref.items())


def test_norm_out_and_conv_out_selection(tiny_denoiser):
    cfg = ThetaSConfig(include=("norm_out.*", "conv_out.*"))
    named = selected_named_parameters(tiny_denoiser, cfg)
    expected_names = {f"norm_out.{n}" for n, _ in tiny_denoiser.inner_model.norm_out.named_parameters()}
    expected_names |= {f"conv_out.{n}" for n, _ in tiny_denoiser.inner_model.conv_out.named_parameters()}
    assert set(named.keys()) == expected_names


# --------------------------------------------------------------------------------------
# D: broad include + exclude (needs a multi-u_block model; tiny_denoiser only has one)
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_block_denoiser() -> Denoiser:
    torch.manual_seed(0)
    inner_cfg = InnerModelConfig(
        img_channels=3, num_steps_conditioning=1, cond_channels=16,
        depths=[1, 1], channels=[8, 8], attn_depths=[False, False], continuous_action_dim=2,
    )
    cfg = DenoiserConfig(inner_model=inner_cfg, sigma_data=0.5, sigma_offset_noise=0.3)
    denoiser = Denoiser(cfg)
    denoiser.eval()
    return denoiser


def test_broad_include_with_exclude(two_block_denoiser):
    cfg = ThetaSConfig(include=("unet.u_blocks.*",), exclude=("unet.u_blocks.0.*",))
    named = selected_named_parameters(two_block_denoiser, cfg)
    assert len(named) > 0
    assert all(name.startswith("unet.u_blocks.1.") for name in named)
    assert not any(name.startswith("unet.u_blocks.0.") for name in named)

    # sanity: block 0 does have parameters, so the exclude genuinely removed something
    all_u_blocks_named = selected_named_parameters(two_block_denoiser, ThetaSConfig(include=("unet.u_blocks.*",)))
    assert len(all_u_blocks_named) > len(named)


# --------------------------------------------------------------------------------------
# E: pattern-order independence
# --------------------------------------------------------------------------------------


def test_pattern_order_independence(two_block_denoiser):
    cfg_forward = ThetaSConfig(include=("conv_out.*", "norm_out.*", "unet.u_blocks.1.*"))
    cfg_reversed = ThetaSConfig(include=("unet.u_blocks.1.*", "norm_out.*", "conv_out.*"))

    named_forward = selected_named_parameters(two_block_denoiser, cfg_forward)
    named_reversed = selected_named_parameters(two_block_denoiser, cfg_reversed)

    assert list(named_forward.keys()) == list(named_reversed.keys())
    assert all(a is b for a, b in zip(named_forward.values(), named_reversed.values()))


# --------------------------------------------------------------------------------------
# F/G: invalid pattern, empty selection
# --------------------------------------------------------------------------------------


def test_invalid_include_pattern_raises(tiny_denoiser):
    cfg = ThetaSConfig(include=("this_module_does_not_exist.*",))
    with pytest.raises(ValueError, match="matched no parameters"):
        selected_named_parameters(tiny_denoiser, cfg)


def test_empty_final_selection_raises(tiny_denoiser):
    # both patterns individually match something, but exclude cancels out include entirely
    cfg = ThetaSConfig(include=("conv_out.*",), exclude=("conv_out.*",))
    with pytest.raises(ValueError, match="empty"):
        selected_named_parameters(tiny_denoiser, cfg)


def test_unmatched_exclude_pattern_is_not_an_error(tiny_denoiser):
    """An exclude pattern matching nothing in the current architecture is harmless, not an
    error -- documented choice (item 5)."""
    cfg = ThetaSConfig(include=("conv_out.*",), exclude=("this_matches_nothing.*",))
    named = selected_named_parameters(tiny_denoiser, cfg)
    assert set(named.keys()) == {"conv_out.weight", "conv_out.bias"}


# --------------------------------------------------------------------------------------
# Historical precision with an alternative theta_S (item 11)
# --------------------------------------------------------------------------------------


def _small_dataset(tmp_path):
    dataset = Dataset(tmp_path / "ds", "theta_s_test_ds", cache_in_ram=True)
    for L in [4, 6]:
        ep = Episode(
            obs=torch.randn(L, 3, 8, 8), act=torch.randn(L, 2), rew=torch.zeros(L),
            end=torch.zeros(L, dtype=torch.uint8), trunc=torch.zeros(L, dtype=torch.uint8), info={},
        )
        dataset.add_episode(ep)
    return dataset


@pytest.mark.parametrize("include", [
    ("unet.u_blocks.0.*", "norm_out.*", "conv_out.*"),  # current/default-equivalent selection
    ("conv_out.*",),  # conv_out-only
])
def test_historical_precision_is_parameter_subset_agnostic(tiny_denoiser, tmp_path, include):
    denoiser = tiny_denoiser
    dataset = _small_dataset(tmp_path)
    cfg = ThetaSConfig(include=include)
    params = selected_parameters(denoiser, cfg)
    d_S = selected_dim(denoiser, cfg)
    N = dataset.num_steps

    np_state_before = torch.get_rng_state()
    h_D = historical_precision(denoiser, params, dataset, SIGMA_CFG, B=N, N=N, num_mc=3, beta=1.0, damping=1e-4, seed=11)
    np_state_after = torch.get_rng_state()

    assert h_D.shape == (d_S,)
    assert torch.isfinite(h_D).all()
    assert (h_D > 0).all()  # strictly positive after damping
    assert torch.equal(np_state_before, np_state_after)  # RNG isolation intact regardless of theta_S


# --------------------------------------------------------------------------------------
# Forward JVP with an alternative theta_S (item 12)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("include", [
    ("unet.u_blocks.0.*", "norm_out.*", "conv_out.*"),
    ("conv_out.*",),
])
def test_forward_jvp_is_parameter_subset_agnostic(tiny_denoiser, include):
    denoiser = tiny_denoiser
    cfg = ThetaSConfig(include=include)
    theta_s_named = selected_named_parameters(denoiser, cfg)
    frozen_named = frozen_named_parameters(denoiser, theta_s_named)
    d_S = selected_dim(denoiser, cfg)

    h_D = torch.rand(d_S) + 0.5
    assert_setup_valid(theta_s_named, h_D, d_S=d_S)  # h_D dimension check works for any theta_S

    n = denoiser.cfg.inner_model.num_steps_conditioning
    obs = torch.randn(1, n * 3, 8, 8)
    act = torch.randn(1, n, 2)
    y = torch.randn(1, 3, 8, 8)
    candidates = [(obs, act, y), (obs, act, y)]  # duplicate, to also check Full-CRN identity across chunks

    bank = make_jvp_bank(SIGMA_CFG, torch.Size([1, 3, 8, 8]), d_S, device=torch.device("cpu"), num_samples=3, seed=0)
    h_D_inv_sqrt = h_D.rsqrt()
    scores_chunk1 = score_one_jvp_bank(denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, bank, candidates, chunk_size=1)
    scores_chunk2 = score_one_jvp_bank(denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, bank, candidates, chunk_size=2)

    assert scores_chunk1.shape == (2,)
    assert torch.isfinite(scores_chunk1).all()
    assert (scores_chunk1 >= 0).all()
    assert torch.equal(scores_chunk1[0], scores_chunk1[1])  # identical candidates -> identical score
    assert torch.allclose(scores_chunk1, scores_chunk2, atol=1e-5)  # Full CRN: chunking must not change scores
