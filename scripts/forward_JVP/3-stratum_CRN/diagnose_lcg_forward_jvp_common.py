#! /usr/bin/env python
"""
Shared forward-mode JVP scorer for the LCG forward/JVP diagnostic (Stage F1).

Genuine forward-mode automatic differentiation via torch.func.jvp + torch.func.functional_call
-- NOT autograd.grad, NOT reverse-over-reverse, NOT an explicit Jacobian for the real model.
theta_S (the existing LCG parameter subset: final decoder level's 3 conditional ResBlocks,
norm_out, conv_out) is passed as the jvp "primal" via functional_call; every other denoiser
parameter is captured as an ordinary (non-dual) closure tensor, so forward-mode AD only
propagates dual numbers through the ops downstream of theta_S -- confirmed empirically
(scratchpad/test_jvp_feasibility.py) to work on the real InnerModel with zero unsupported
operators, and the functional primal output is bit-identical to a normal forward call.

Uses the EDM cancellation validated in Stage 2.5 (w(sigma)*c_out(sigma)^2 = 1, so
2*w*||J_D z||^2 = 2*||J_F z||^2): the JVP is taken through the raw inner-model output F_theta
directly (denoiser.compute_model_output's un-preconditioned path), never touching c_out/c_skip,
matching lcg.gauss_newton's own analytic-cancellation finding -- not a new assumption.

z = H_D^{-1/2} @ eta is computed once (elementwise, H_D diagonal, no_grad) and never
appears inside any autograd/functorch-traced region.
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while not ((p / "src").is_dir() and (p / "scripts").is_dir()):
        if p.parent == p:
            raise RuntimeError("could not locate LCG repo root (no ancestor has both src/ and scripts/)")
        p = p.parent
    return p


_REPO_ROOT = _find_repo_root(Path(__file__).parent)
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# scripts/ was reorganized into topic subfolders after this file was first written; this
# extra entry lets this module find diagnose_lcg_backward_variance_setup regardless of
# exactly where this file itself ends up nested.
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "backward_VJP" / "3-stratum"))

import torch
import torch.func as func
from torch import Tensor

import diagnose_lcg_backward_variance_setup as setup
from lcg.sigma_strata import sample_sigma_stratum
from lcg.theta_s import selected_parameters
from models.diffusion.denoiser import Denoiser

NUM_STRATA = setup.NUM_STRATA


# --------------------------------------------------------------------------------------
# theta_S <-> named-dict plumbing (functional_call needs dotted names; ordering must match
# selected_parameters()/h_D exactly)
# --------------------------------------------------------------------------------------


def selected_named_parameters(denoiser: Denoiser) -> Dict[str, torch.nn.Parameter]:
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

    ref = selected_parameters(denoiser)
    assert len(named) == len(ref) and all(a is b for a, b in zip(named.values(), ref)), (
        "selected_named_parameters ordering/membership does not match lcg.theta_s.selected_parameters()"
    )
    return named


def frozen_named_parameters(denoiser: Denoiser, theta_s_named: Dict[str, torch.nn.Parameter]) -> Dict[str, Tensor]:
    inner = denoiser.inner_model
    selected_ids = {id(p) for p in theta_s_named.values()}
    return {name: p for name, p in inner.named_parameters() if id(p) not in selected_ids}


def flatten_dict(named: Dict[str, Tensor]) -> Tensor:
    return torch.cat([t.reshape(-1) for t in named.values()])


def unflatten_to_dict(flat: Tensor, template: Dict[str, Tensor]) -> Dict[str, Tensor]:
    result = {}
    offset = 0
    for name, p in template.items():
        n = p.numel()
        result[name] = flat[offset : offset + n].reshape(p.shape)
        offset += n
    assert offset == flat.numel(), f"flat tensor size {flat.numel()} does not match template total {offset}"
    return result


# --------------------------------------------------------------------------------------
# F1: genuine forward-mode JVP through F_theta (EDM-cancelled, no c_out/c_skip)
# --------------------------------------------------------------------------------------


def jvp_through_F(
    denoiser: Denoiser,
    theta_s_named: Dict[str, Tensor],
    frozen_named: Dict[str, Tensor],
    tangent_named: Dict[str, Tensor],
    y_sigma_batch: Tensor,
    sigma: Tensor,
    obs_batch: Tensor,
    act_batch: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Returns (F_theta_primal, JVP_output), both shape matching y_sigma_batch. Uses
    torch.func.jvp + torch.func.functional_call exclusively -- true forward-mode AD.

    Wrapped in torch.no_grad(): frozen_named's tensors are real nn.Parameters with
    requires_grad=True, so without this, functional_call's forward pass also builds a
    reverse-mode autograd graph (visible as primal_out.grad_fn/jvp_out.grad_fn being
    non-None) IN ADDITION TO the forward-mode dual computation -- pure waste, since nothing
    here ever calls .backward(), and it retains the entire inner_model activation graph per
    call. torch.no_grad() only disables reverse-mode graph construction; forward-mode AD
    (torch.autograd.forward_ad, which torch.func.jvp is built on) is an orthogonal
    mechanism and is unaffected -- confirmed empirically: found via a real CUDA OOM in
    Stage F6's profiling run (unbounded graph accumulation across the per-stratum loop),
    fixed here, and reverified against F2's tiny-model correctness checks after the fix
    (bit-identical primal/JVP values, grad_fn now None on both outputs)."""
    cs = denoiser.compute_conditioners(sigma)
    rescaled_obs = obs_batch / denoiser.cfg.sigma_data
    rescaled_noise = y_sigma_batch * cs.c_in
    c_noise = cs.c_noise

    def f(theta_s_dict):
        full_params = {**frozen_named, **theta_s_dict}
        return func.functional_call(denoiser.inner_model, full_params, (rescaled_noise, c_noise, rescaled_obs, act_batch))

    with torch.no_grad():
        primal_out, jvp_out = func.jvp(f, (theta_s_named,), (tangent_named,))
    return primal_out, jvp_out


@dataclass(frozen=True)
class JVPBank:
    """One Full-CRN bank: one shared (sigma, eps, eta) triple per stratum, reused across
    every candidate. sigmas/epsilons: same broadcastable shapes as lcg.crn.CRNBank. etas:
    one flat (d_S,) tensor per stratum (parameter-space probe, inherently "global" -- there
    is no per-candidate analog for a parameter-space direction)."""

    sigmas: Tuple[Tensor, ...]
    epsilons: Tuple[Tensor, ...]
    etas: Tuple[Tensor, ...]

    @property
    def num_strata(self) -> int:
        return len(self.sigmas)


def make_jvp_bank(sigma_cfg, y_shape, d_S: int, device, num_strata: int = NUM_STRATA, seed=None) -> JVPBank:
    if seed is not None:
        torch.manual_seed(seed)
    sigmas, epsilons, etas = [], [], []
    for m in range(num_strata):
        sigmas.append(sample_sigma_stratum(sigma_cfg, m, num_strata, 1, device).detach())
        epsilons.append(torch.randn(y_shape, device=device).detach())
        etas.append(torch.randn(d_S, device=device).detach())
    return JVPBank(tuple(sigmas), tuple(epsilons), tuple(etas))


def score_one_jvp_bank(
    denoiser: Denoiser,
    theta_s_named: Dict[str, Tensor],
    frozen_named: Dict[str, Tensor],
    template: Dict[str, Tensor],
    h_D: Tensor,
    h_D_inv_sqrt: Tensor,
    bank: JVPBank,
    candidates: List,
    chunk_size: int,
) -> Tensor:
    """r_hat_k(x_j) = (1/num_strata) * sum_s 2*||JVP_F(x_j; sigma_s,eps_s; z_s)||^2, for
    every candidate. z_s = h_D_inv_sqrt * eta_s (elementwise, H_D diagonal, no_grad)."""
    device = h_D.device
    num_strata = bank.num_strata
    all_scores = []
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        obs_batch = torch.cat([c[0] for c in chunk], dim=0)
        act_batch = torch.cat([c[1] for c in chunk], dim=0)
        y_batch = torch.cat([c[2] for c in chunk], dim=0)
        B = len(chunk)

        chunk_score = torch.zeros(B, device=device)
        for m in range(num_strata):
            sigma, eps, eta = bank.sigmas[m], bank.epsilons[m], bank.etas[m]
            with torch.no_grad():
                z_flat = h_D_inv_sqrt * eta
            tangent_named = unflatten_to_dict(z_flat, template)

            y_sigma_batch = (y_batch + sigma.view(-1, 1, 1, 1) * eps).detach()
            _, jvp_out = jvp_through_F(
                denoiser, theta_s_named, frozen_named, tangent_named, y_sigma_batch, sigma, obs_batch, act_batch
            )
            contribution = 2.0 * jvp_out.reshape(B, -1).square().sum(dim=1)
            chunk_score = chunk_score + contribution / num_strata
        all_scores.append(chunk_score.detach().cpu())
    return torch.cat(all_scores)


def assert_setup_valid(theta_s_named: Dict[str, Tensor], h_D: Tensor, d_S: int) -> None:
    flat = flatten_dict(theta_s_named)
    assert flat.numel() == d_S, f"theta_S flat dim {flat.numel()} != d_S {d_S}"
    assert h_D.numel() == d_S, f"h_D dim {h_D.numel()} != d_S {d_S}"
    assert torch.isfinite(h_D).all(), "h_D contains non-finite values"
    assert (h_D > 0).all(), "h_D is not strictly positive"
    assert not h_D.requires_grad, "h_D must not require grad"
    print(f"assert_setup_valid: dim(theta_S)=dim(h_D)={d_S}, h_D finite and >0, h_D.requires_grad=False -- OK")
