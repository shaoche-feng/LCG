from typing import List, Tuple

import torch
from torch import Tensor
import torch.nn as nn

from models.diffusion.denoiser import Denoiser

from .crn import CRNBank
from .gauss_newton import differentiable_denoise

Candidate = Tuple[Tensor, Tensor, Tensor]


def compute_vjp_batched(
    denoiser: Denoiser,
    params: List[nn.Parameter],
    y_sigma_batch: Tensor,
    sigma: Tensor,
    obs_batch: Tensor,
    act_batch: Tensor,
    xi: Tensor,
) -> Tensor:
    """Per-candidate VJP for a batch/chunk of B candidates sharing one (sigma, xi) probe
    (the CRN case: the same probe value is broadcast across every candidate in the chunk).

    Returns v_batch of shape (B, d_S), where
        v_batch[i] = grad_{theta_S}[ sqrt(2 w(sigma)) xi^T D_theta(y_sigma_batch[i], sigma, x_i) ],
    i.e. EACH candidate's own gradient -- not the gradient of the sum over candidates
    (sum_i v_i != grad(sum_i s_i) in general terms of what we need here; we need each v_i
    individually since (sum v_i)^2 != sum v_i^2).

    Computed via torch.autograd.grad(..., is_grads_batched=True) with an identity
    grad_outputs: outputs has shape (B,) (one scalar objective per candidate), and
    grad_outputs=eye(B) supplies B different cotangents (the standard basis vectors e_k),
    so row k of the result is exactly d(outputs_k)/d(theta) -- candidate k's own gradient,
    for all k at once, via a single vmapped backward call instead of a Python loop of B
    separate torch.autograd.grad calls.

    Reuses differentiable_denoise (Stage 1, unmodified) for the forward pass -- only the
    backward strategy differs from compute_vjp, which reduces to a single scalar summed
    over the whole batch and would otherwise silently compute d(sum_i s_i)/d(theta) instead
    of each v_i.
    """
    d_theta, w = differentiable_denoise(denoiser, y_sigma_batch, sigma, obs_batch, act_batch)
    B = d_theta.size(0)
    per_example = (torch.sqrt(2 * w) * xi * d_theta).reshape(B, -1).sum(dim=1)  # (B,)
    grad_outputs = torch.eye(B, device=per_example.device, dtype=per_example.dtype)
    grads = torch.autograd.grad(
        per_example, params, grad_outputs=grad_outputs, is_grads_batched=True,
        retain_graph=False, create_graph=False,
    )
    return torch.cat([g.reshape(B, -1) for g in grads], dim=1)


def score_candidates_batched(
    denoiser: Denoiser,
    params: List[nn.Parameter],
    h_D: Tensor,
    banks: Tuple[CRNBank, ...],
    candidates: List[Candidate],
    chunk_size: int,
) -> Tensor:
    """Mathematically identical to lcg.batched_scoring.score_candidates_with_banks (same
    formula, same CRNBank objects/probes), but processes candidates in chunks of
    chunk_size: each chunk's denoiser forward is batched, and compute_vjp_batched (an exact
    per-candidate VJP) is used for the backward pass instead of one-candidate-at-a-time
    autograd.grad calls.

    Processes one (bank, stratum) probe at a time across the whole chunk and releases that
    probe's (chunk_size, d_S) gradient tensor before moving to the next probe, so at most
    one such tensor is alive at once regardless of num_strata/num_crn_banks.
    """
    device = h_D.device
    num_banks = len(banks)
    num_strata = banks[0].num_strata
    all_scores = []
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        obs_batch = torch.cat([c[0] for c in chunk], dim=0)
        act_batch = torch.cat([c[1] for c in chunk], dim=0)
        y_batch = torch.cat([c[2] for c in chunk], dim=0)
        B = len(chunk)

        score_accum = torch.zeros(B, device=device)
        for bank in banks:
            for sigma, eps, xi in zip(bank.sigmas, bank.epsilons, bank.xis):
                y_sigma_batch = (y_batch + sigma.view(-1, 1, 1, 1) * eps).detach()
                v_batch = compute_vjp_batched(denoiser, params, y_sigma_batch, sigma, obs_batch, act_batch, xi)
                contribution = (v_batch.square() / h_D.unsqueeze(0)).sum(dim=1)
                score_accum = score_accum + contribution / (num_strata * num_banks)
                del v_batch, contribution
        all_scores.append(score_accum.detach().cpu())
    return torch.cat(all_scores)
