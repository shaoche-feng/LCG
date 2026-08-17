#! /usr/bin/env python
"""
LCG Stage 1 validation: isolated checks for the diagnostic building blocks, with no
Trainer/ActorCritic/H_D/intrinsic-reward wiring.

Builds a tiny (but real) Denoiser via models.diffusion.Denoiser/InnerModelConfig and checks:
  1. theta_S selector picks the expected modules (final decoder ResBlocks level +
     output GroupNorm + output conv) and a nonzero parameter count.
  2. compute_vjp's single-transition Monte Carlo estimator E_xi[v v^T] matches the
     explicit Gauss-Newton matrix 2*w(sigma)*J^T J (both the full matrix and its
     diagonal, since only the diagonal is used downstream).
  3. The 3-stratum sigma sampler's pooled mixture reproduces the unstratified
     log-normal p_train(sigma)'s quantiles, and num_strata=1 reduces to the same
     distribution as Denoiser's own training sampler.

Usage:
    python scripts/validate_lcg_stage1.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from lcg.gauss_newton import compute_vjp, differentiable_denoise
from lcg.sigma_strata import sample_sigma_strata
from lcg.theta_s import selected_dim, selected_parameters, selected_submodules
from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.diffusion.inner_model import InnerModelConfig

IMG_CHANNELS = 1
IMG_SIZE = 8
NUM_STEPS_CONDITIONING = 1
CONTINUOUS_ACTION_DIM = 1


def build_toy_denoiser(device: torch.device) -> Denoiser:
    inner_cfg = InnerModelConfig(
        img_channels=IMG_CHANNELS,
        num_steps_conditioning=NUM_STEPS_CONDITIONING,
        cond_channels=8,
        depths=[1, 1],
        channels=[2, 2],
        attn_depths=[0, 0],
        continuous_action_dim=CONTINUOUS_ACTION_DIM,
    )
    cfg = DenoiserConfig(inner_model=inner_cfg, sigma_data=0.5, sigma_offset_noise=0.0)
    denoiser = Denoiser(cfg).to(device)
    denoiser.eval()
    return denoiser


def check_theta_s(denoiser: Denoiser):
    modules = selected_submodules(denoiser)
    params = selected_parameters(denoiser)
    d_s = selected_dim(denoiser)
    print(f"[theta_S] modules: {[type(m).__name__ for m in modules]}")
    print(f"[theta_S] num parameter tensors: {len(params)}, d_S = {d_s}")
    assert len(modules) == 3
    assert d_s > 0
    return params, d_s


def explicit_jacobian(denoiser, params, y_sigma, sigma, obs, act):
    d_theta, w = differentiable_denoise(denoiser, y_sigma, sigma, obs, act)
    flat = d_theta.reshape(-1)
    d_y, d_s = flat.numel(), sum(p.numel() for p in params)
    J = torch.zeros(d_y, d_s, device=y_sigma.device)
    for i in range(d_y):
        grads = torch.autograd.grad(flat[i], params, retain_graph=True)
        J[i] = torch.cat([g.reshape(-1) for g in grads])
    return J, w


def check_vjp_matches_gauss_newton(denoiser, params, device, num_mc=6000, seed=0):
    torch.manual_seed(seed)
    obs = torch.randn(1, NUM_STEPS_CONDITIONING * IMG_CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
    act = torch.randn(1, NUM_STEPS_CONDITIONING, CONTINUOUS_ACTION_DIM, device=device)
    y = torch.randn(1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
    sigma = torch.tensor([0.7], device=device)
    eps = torch.randn_like(y)
    y_sigma = (y + sigma.view(-1, 1, 1, 1) * eps).detach()

    J, w = explicit_jacobian(denoiser, params, y_sigma, sigma, obs, act)
    G_exact = 2 * w.reshape(()) * (J.T @ J)

    d_s = G_exact.shape[0]
    G_accum = torch.zeros(d_s, d_s, device=device)
    for _ in range(num_mc):
        v, _ = compute_vjp(denoiser, params, y_sigma, sigma, obs, act)
        G_accum += torch.outer(v, v)
    G_mc = G_accum / num_mc

    rel_err_full = (G_mc - G_exact).norm() / G_exact.norm()
    rel_err_diag = (G_mc.diagonal() - G_exact.diagonal()).norm() / G_exact.diagonal().norm()
    print(f"[GN check] num_mc={num_mc}, d_S={d_s}, d_y={J.shape[0]}")
    print(f"[GN check] ||G_mc - G_exact||_F / ||G_exact||_F = {rel_err_full.item():.4f}")
    print(f"[GN check] ||diag_mc - diag_exact|| / ||diag_exact|| = {rel_err_diag.item():.4f}")
    assert rel_err_full < 0.15, "Monte Carlo GN estimate deviates too much from explicit J^T J"
    assert rel_err_diag < 0.15, "Monte Carlo GN diagonal deviates too much from explicit diag(J^T J)"


def check_sigma_strata(device, num_strata=3, n_per_stratum=100_000):
    cfg = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=2e-3, sigma_max=20.0)

    def sample_unstratified(n):
        s = torch.randn(n, device=device) * cfg.scale + cfg.loc
        return s.exp().clip(cfg.sigma_min, cfg.sigma_max)

    baseline = sample_unstratified(n_per_stratum * num_strata)
    strata = sample_sigma_strata(cfg, num_strata, n_per_stratum, device)
    mixture = torch.cat(strata)

    qs = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=device)
    q_base, q_mix = torch.quantile(baseline, qs), torch.quantile(mixture, qs)
    rel_err = ((q_mix - q_base).abs() / q_base).max()
    print(f"[sigma strata] baseline quantiles:    {[round(x, 4) for x in q_base.tolist()]}")
    print(f"[sigma strata] 3-stratum mixture quantiles: {[round(x, 4) for x in q_mix.tolist()]}")
    print(f"[sigma strata] max relative quantile error: {rel_err.item():.4f}")
    assert rel_err < 0.1

    single_stratum = sample_sigma_strata(cfg, 1, n_per_stratum * num_strata, device)[0]
    q_single = torch.quantile(single_stratum, qs)
    rel_err_single = ((q_single - q_base).abs() / q_base).max()
    print(f"[sigma strata] num_strata=1 quantiles: {[round(x, 4) for x in q_single.tolist()]}")
    print(f"[sigma strata] num_strata=1 max relative quantile error: {rel_err_single.item():.4f}")
    assert rel_err_single < 0.05


def main():
    device = torch.device("cpu")
    denoiser = build_toy_denoiser(device)
    params, _ = check_theta_s(denoiser)
    check_vjp_matches_gauss_newton(denoiser, params, device)
    check_sigma_strata(device)
    print("\nStage 1 validation passed.")


if __name__ == "__main__":
    main()
