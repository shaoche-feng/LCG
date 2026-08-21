#! /usr/bin/env python
"""
Shared sampling/scoring utilities for the simple-MC-vs-3-stratum LCG diagnostic.

Reuses diagnose_lcg_backward_variance_setup.py's cached model/h_D_full/candidates and
lcg.batched_vjp.compute_vjp_batched (Stage 4.5, unmodified) as the sole VJP primitive --
same as the backward/VJP variance diagnostic. No src/lcg/*.py changes.

Key simplification vs. the 3-stratum estimator: here one "MC sample" m IS the complete
estimate contribution (no inner strata loop, no /num_strata averaging) --
r_hat_m(x) = v_m^T H_D^{-1} v_m, r_hat_M(x) = mean_m r_hat_m(x). sigma_m is drawn from the
UNSTRATIFIED p_train(sigma) distribution; reusing lcg.sigma_strata.sample_sigma_stratum
with (stratum_idx=0, num_strata=1) gives exactly this (its quantile window collapses to
the full [eps, 1-eps] range when num_strata=1), so no new sigma-sampling code is needed.

Three sharing conditions per MC sample, implemented purely via tensor shape (broadcast
(1,...) vs per-candidate (N,...)) -- compute_vjp_batched already supports both:
  - independent: sigma (N,), eps (N,C,H,W), xi (N,C,H,W) -- all independent per candidate.
  - xi_crn:       sigma (N,), eps (N,C,H,W), xi (1,C,H,W) -- only xi shared.
  - full_crn:     sigma (1,), eps (1,C,H,W), xi (1,C,H,W) -- everything shared.
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

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
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "backward_VJP" / "3-stratum"))

import torch
from torch import Tensor

import diagnose_lcg_backward_variance_setup as setup
from lcg.batched_vjp import compute_vjp_batched
from lcg.sigma_strata import sample_sigma_stratum

CONDITIONS = ("independent", "xi_crn", "full_crn")
SINGLE_SHAPE = (3, 64, 64)


@dataclass(frozen=True)
class MCSample:
    sigma: Tensor  # (1,) or (N,)
    eps: Tensor    # (1,C,H,W) or (N,C,H,W)
    xi: Tensor     # (1,C,H,W) or (N,C,H,W)


def make_mc_sample(condition: str, sigma_cfg, num_candidates: int, device, seed: int) -> MCSample:
    assert condition in CONDITIONS
    torch.manual_seed(seed)
    if condition == "independent":
        sigma = sample_sigma_stratum(sigma_cfg, 0, 1, num_candidates, device).detach()
        eps = torch.randn((num_candidates,) + SINGLE_SHAPE, device=device).detach()
        xi = torch.randn((num_candidates,) + SINGLE_SHAPE, device=device).detach()
    elif condition == "xi_crn":
        sigma = sample_sigma_stratum(sigma_cfg, 0, 1, num_candidates, device).detach()
        eps = torch.randn((num_candidates,) + SINGLE_SHAPE, device=device).detach()
        xi = torch.randn((1,) + SINGLE_SHAPE, device=device).detach()
    else:  # full_crn
        sigma = sample_sigma_stratum(sigma_cfg, 0, 1, 1, device).detach()
        eps = torch.randn((1,) + SINGLE_SHAPE, device=device).detach()
        xi = torch.randn((1,) + SINGLE_SHAPE, device=device).detach()
    return MCSample(sigma, eps, xi)


def score_mc_sample(denoiser, params, h_D, sample: MCSample, candidates, chunk_size, device) -> Tensor:
    """r_hat_m for every candidate, one VJP per candidate (no stratum sub-loop)."""
    all_scores = []
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        obs_batch = torch.cat([c[0] for c in chunk], dim=0)
        act_batch = torch.cat([c[1] for c in chunk], dim=0)
        y_batch = torch.cat([c[2] for c in chunk], dim=0)
        B = len(chunk)

        sigma = sample.sigma if sample.sigma.shape[0] == 1 else sample.sigma[start : start + B]
        eps = sample.eps if sample.eps.shape[0] == 1 else sample.eps[start : start + B]
        xi = sample.xi if sample.xi.shape[0] == 1 else sample.xi[start : start + B]

        y_sigma_batch = (y_batch + sigma.view(-1, 1, 1, 1) * eps).detach()
        v_batch = compute_vjp_batched(denoiser, params, y_sigma_batch, sigma, obs_batch, act_batch, xi)
        contribution = (v_batch.square() / h_D.unsqueeze(0)).sum(dim=1)
        all_scores.append(contribution.detach().cpu())
        del v_batch, contribution
    return torch.cat(all_scores)
