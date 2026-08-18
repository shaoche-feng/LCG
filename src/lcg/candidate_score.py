from typing import List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn

from models.diffusion.denoiser import Denoiser, SigmaDistributionConfig

from .gauss_newton import compute_vjp
from .sigma_strata import sample_sigma_stratum


def candidate_score_per_stratum(
    denoiser: Denoiser,
    params: List[nn.Parameter],
    sigma_cfg: SigmaDistributionConfig,
    x_obs: Tensor,
    x_act: Tensor,
    y_star: Tensor,
    num_strata: int = 3,
) -> Tuple[Tensor, Tensor]:
    """One VJP per stratum for a single candidate transition (x*, y*), using the exact
    Stage-1 compute_vjp implementation (no second VJP derivation). Corrupts y_star as
    y_sigma = y_star + sigma*eps, the same convention validated in Stage 1/2 -- no
    different corruption process is introduced here.

    Returns (v_squared, sigmas): v_squared has shape (num_strata, d_S) -- v(.)^2 per
    stratum, uncombined with h_D -- and sigmas has shape (num_strata,). Kept separate from
    candidate_score() so callers needing coordinate/module-level detail (diagnostics) don't
    have to re-derive the VJP.
    """
    assert x_obs.size(0) == 1 and y_star.size(0) == 1
    device = y_star.device
    d_S = sum(p.numel() for p in params)
    v_squared = torch.empty(num_strata, d_S, device=device)
    sigmas = torch.empty(num_strata, device=device)

    for m in range(num_strata):
        sigma = sample_sigma_stratum(sigma_cfg, m, num_strata, 1, device)
        eps = torch.randn_like(y_star)
        y_sigma = (y_star + sigma.view(-1, 1, 1, 1) * eps).detach()
        v, _ = compute_vjp(denoiser, params, y_sigma, sigma, x_obs, x_act)
        v_squared[m] = v * v
        sigmas[m] = sigma.detach()

    return v_squared, sigmas


def candidate_score(
    denoiser: Denoiser,
    params: List[nn.Parameter],
    h_D: Tensor,
    sigma_cfg: SigmaDistributionConfig,
    x_obs: Tensor,
    x_act: Tensor,
    y_star: Tensor,
    num_strata: int = 3,
    return_diagnostics: bool = False,
) -> "float | Tuple[float, dict]":
    """Algorithm 3: the LCG intrinsic score for one candidate transition (x*, y*), given a
    frozen historical diagonal precision h_D (from Algorithm 2, unmodified):

        r_LCG = (1/M) * sum_{m=1}^{M} sum_k (v_k^(m))^2 / h_{D,k}

    h_D is a plain input tensor here -- this module never estimates or updates it (that
    remains lcg.precision.historical_precision, untouched). No candidate-side beta: once
    h_D is fixed, beta no longer affects ranking across candidates (see Stage 2.5/2.6), so
    it is intentionally omitted from this formula.

    Never touches denoiser parameters or .grad, never retains a graph between strata
    (inherited directly from compute_vjp's retain_graph=False, create_graph=False).
    """
    d_S = sum(p.numel() for p in params)
    assert h_D.shape == (d_S,), f"h_D shape {tuple(h_D.shape)} != (d_S,)=({d_S},)"
    assert torch.all(h_D > 0), "h_D must be strictly positive everywhere (Algorithm 2's damping guarantees this)"

    v_squared, sigmas = candidate_score_per_stratum(denoiser, params, sigma_cfg, x_obs, x_act, y_star, num_strata)
    per_coord = v_squared / h_D.unsqueeze(0)  # (num_strata, d_S)
    per_stratum_r = per_coord.sum(dim=1)  # (num_strata,)
    score = per_stratum_r.mean().item()

    if return_diagnostics:
        diagnostics = dict(
            per_stratum_r=per_stratum_r.detach().cpu(),
            sigmas=sigmas.detach().cpu(),
            v_squared=v_squared.detach().cpu(),
        )
        return score, diagnostics
    return score
