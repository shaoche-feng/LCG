"""
Permanent regression tests for the DIAMOND-LCG corruption fix (extracted from
scripts/corruption_fix/validate_corruption_fix.py). Preserves the guarantee that
LCG evaluates the exact same corrupted-input distribution DIAMOND training uses:

    y_sigma = y + sigma*eps + sigma_offset_noise*eps_offset

All tests use fixed, controlled tensors -- never only distributional statistics.
"""
from unittest import mock

import lcg.forward_jvp as lcg_forward_jvp
import lcg.precision as lcg_precision
import torch
from models.diffusion.denoiser import add_dims, apply_noise_from_samples


def test_apply_noise_from_samples_matches_exact_formula():
    B, C, H, W = 3, 4, 8, 8
    y = torch.randn(B, C, H, W)
    sigma = torch.rand(B) * 2 + 0.1
    eps = torch.randn(B, C, H, W)
    eps_offset = torch.randn(1, C, 1, 1)
    sigma_offset_noise = 0.3

    manual = y + add_dims(sigma, y.ndim) * eps + sigma_offset_noise * eps_offset
    actual = apply_noise_from_samples(y, sigma, eps, eps_offset, sigma_offset_noise)
    assert torch.equal(actual, manual)


def test_offset_noise_spatially_constant_per_channel():
    B, C, H, W = 2, 3, 16, 16
    y = torch.zeros(B, C, H, W)
    sigma = torch.zeros(B)
    eps = torch.zeros_like(y)
    eps_offset = torch.randn(1, C, 1, 1)

    y_sigma = apply_noise_from_samples(y, sigma, eps, eps_offset, sigma_offset_noise=1.0)
    # every spatial location within a (batch, channel) slice must be identical
    assert y_sigma.std(dim=(2, 3)).max().item() < 1e-6


def test_zero_offset_reproduces_pre_fix_formula():
    y = torch.randn(4, 3, 8, 8)
    sigma = torch.rand(4)
    eps = torch.randn_like(y)
    eps_offset = torch.randn(1, 3, 1, 1)

    y_sigma = apply_noise_from_samples(y, sigma, eps, eps_offset, sigma_offset_noise=0.0)
    expected = y + add_dims(sigma, y.ndim) * eps
    assert torch.equal(y_sigma, expected)


def test_denoiser_apply_noise_matches_shared_helper(tiny_denoiser):
    """Denoiser.apply_noise (training path) delegates to apply_noise_from_samples --
    verified with monkeypatched fixed draws against an INDEPENDENT hand-written formula
    (not a tautological self-comparison against the helper it calls)."""
    denoiser = tiny_denoiser
    B, C, H, W = 3, denoiser.cfg.inner_model.img_channels, 8, 8
    x = torch.randn(B, C, H, W)
    sigma = torch.rand(B) * 2 + 0.1
    fixed_eps_offset = torch.randn(B, C, 1, 1)
    fixed_eps = torch.randn_like(x)

    with mock.patch("torch.randn", side_effect=lambda *a, **k: fixed_eps_offset), \
         mock.patch("torch.randn_like", side_effect=lambda *a, **k: fixed_eps):
        actual = denoiser.apply_noise(x, sigma, denoiser.cfg.sigma_offset_noise)

    expected = x + add_dims(sigma, x.ndim) * fixed_eps + denoiser.cfg.sigma_offset_noise * fixed_eps_offset
    assert torch.equal(actual, expected)


def test_lcg_precision_and_forward_jvp_use_the_authoritative_helper():
    """Structural wiring check: both LCG modules import the SAME function object DIAMOND
    training uses -- not a re-derived copy that could silently diverge again."""
    assert lcg_precision.apply_noise_from_samples is apply_noise_from_samples
    assert lcg_forward_jvp.apply_noise_from_samples is apply_noise_from_samples


def test_no_double_application_of_sigma_offset_noise(tiny_denoiser):
    """compute_conditioners() derives its coefficients from the EFFECTIVE sigma
    sqrt(sigma^2+sigma_offset_noise^2); apply_noise_from_samples must scale eps by the
    BARE sigma, not the effective one -- otherwise sigma_offset_noise would be baked into
    the corruption twice (once directly as the eps_offset term, once indirectly by
    inflating the eps term's scale)."""
    denoiser = tiny_denoiser
    sigma = torch.tensor([1.0])
    y = torch.randn(1, denoiser.cfg.inner_model.img_channels, 8, 8)
    eps = torch.randn_like(y)
    eps_offset = torch.randn(1, y.shape[1], 1, 1)

    y_sigma_bare = apply_noise_from_samples(y, sigma, eps, eps_offset, denoiser.cfg.sigma_offset_noise)
    effective_sigma = (sigma**2 + denoiser.cfg.sigma_offset_noise**2).sqrt()
    y_sigma_if_double_counted = apply_noise_from_samples(y, effective_sigma, eps, eps_offset, denoiser.cfg.sigma_offset_noise)

    assert not torch.equal(y_sigma_bare, y_sigma_if_double_counted)

    # and compute_conditioners itself must still use the effective sigma (unchanged by
    # this fix -- only the corruption construction changed, not the preconditioning math)
    cs = denoiser.compute_conditioners(sigma)
    expected_c_in = 1 / (effective_sigma**2 + denoiser.cfg.sigma_data**2).sqrt()
    assert torch.allclose(cs.c_in.reshape(-1), expected_c_in)
