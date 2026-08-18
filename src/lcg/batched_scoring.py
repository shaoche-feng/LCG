from typing import List, Tuple

import torch
from torch import Tensor
import torch.nn as nn

from models.diffusion.denoiser import Denoiser

from .crn import CRNBank
from .gauss_newton import compute_vjp

Candidate = Tuple[Tensor, Tensor, Tensor]  # (x_obs_flat, x_act, y_star), each batch size 1


def imagined_candidates_from_batch(x_obs: Tensor, x_act: Tensor, y_star: Tensor) -> List[Candidate]:
    """Bridges envs.world_model_env.ImaginedCandidate's batched fields to a list of
    per-candidate tuples for scoring. x_obs: (num_envs, t, C, H, W); x_act: (num_envs, t,
    ...); y_star: (num_envs, C, H, W) -- exactly WorldModelEnv's own buffer/output shapes,
    unflattened. Slices into num_envs single-candidate tuples shaped for
    compute_vjp/differentiable_denoise: x_obs_flat (1, t*C, H, W), x_act (1, t, ...),
    y_star (1, C, H, W).
    """
    num_envs = x_obs.size(0)
    candidates = []
    for i in range(num_envs):
        obs_i = x_obs[i : i + 1]
        act_i = x_act[i : i + 1]
        y_i = y_star[i : i + 1]
        obs_flat = obs_i.reshape(1, -1, obs_i.shape[-2], obs_i.shape[-1])
        candidates.append((obs_flat, act_i, y_i))
    return candidates


def score_candidate_with_banks(
    denoiser: Denoiser,
    params: List[nn.Parameter],
    h_D: Tensor,
    banks: Tuple[CRNBank, ...],
    candidate: Candidate,
) -> float:
    """r(c) = (1/K) sum_b (1/3) sum_m sum_k v_{c,k}^(b,m)^2 / h_{D,k}, for one candidate
    scored against a fixed CRN bank set. Built only from compute_vjp (Stage 1, unmodified):
    reuses Stage 3's exact scoring formula (sum over coordinates, average over strata),
    additionally averaged over banks here, with probes supplied by the (fixed, immutable)
    bank rather than resampled -- no call here ever draws new randomness.
    """
    x_obs, x_act, y_star = candidate
    bank_scores = []
    for bank in banks:
        v_sq_per_stratum = []
        for sigma, eps, xi in zip(bank.sigmas, bank.epsilons, bank.xis):
            y_sigma = (y_star + sigma.view(-1, 1, 1, 1) * eps).detach()
            v, _ = compute_vjp(denoiser, params, y_sigma, sigma, x_obs, x_act, xi=xi)
            v_sq_per_stratum.append(v * v)
        v_sq = torch.stack(v_sq_per_stratum)  # (num_strata, d_S)
        bank_scores.append((v_sq / h_D.unsqueeze(0)).sum(dim=1).mean())
    return torch.stack(bank_scores).mean().item()


def score_candidates_with_banks(
    denoiser: Denoiser,
    params: List[nn.Parameter],
    h_D: Tensor,
    banks: Tuple[CRNBank, ...],
    candidates: List[Candidate],
) -> Tensor:
    """Scores a batch/list of candidates against the same frozen (denoiser, h_D, banks) --
    every candidate receives the identical bank set (CRN's entire point), so per-candidate
    order cannot affect any individual score. A simple per-candidate loop: Stage 4 does not
    assume vectorizing VJPs across candidates is correct or more memory-efficient without
    testing it (see scripts/validate_lcg_stage4.py Part 5's profiling).
    """
    return torch.tensor([score_candidate_with_banks(denoiser, params, h_D, banks, c) for c in candidates])
