from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.func as func
from torch import Tensor

from models.diffusion.denoiser import (
    Denoiser,
    SigmaDistributionConfig,
    apply_noise_from_samples,
    sample_sigma_training_distribution,
)

Candidate = Tuple[Tensor, Tensor, Tensor]  # (x_obs_flat, x_act, y_star), each batch size 1


# --------------------------------------------------------------------------------------
# theta_S <-> named-dict plumbing (functional_call needs dotted names; ordering must match
# lcg.theta_s.selected_named_parameters()/h_D exactly). The canonical selector and its
# complement (frozen_named_parameters) both live in theta_s.py -- this module only
# flatten/unflatten support around them.
# --------------------------------------------------------------------------------------

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
# Genuine forward-mode JVP through F_theta (EDM-cancelled, no c_out/c_skip), validated in
# Stage F1-F6: w(sigma) = c_out(sigma)^-2 exactly under this EDM setup, so
# w(sigma)*c_out(sigma)^2 = 1 identically, and 2*w*||J_D z||^2 = 2*||J_F z||^2 exactly --
# the JVP can be taken through the raw (un-preconditioned) inner-model output directly.
# lcg.precision._backward_vjp_probe relies on the same identity for the historical
# (backward-VJP) side; see tests/lcg/test_forward_jvp.py and tests/lcg/test_precision.py
# for the permanent numerical proofs on each side.
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class JVPBank:
    sigmas: Tuple[Tensor, ...]
    epsilons: Tuple[Tensor, ...]
    epsilons_offset: Tuple[Tensor, ...]
    etas: Tuple[Tensor, ...]

    def __post_init__(self) -> None:
        assert len(self.sigmas) == len(self.epsilons) == len(self.epsilons_offset) == len(self.etas)

    @property
    def num_samples(self) -> int:
        return len(self.sigmas)


def make_jvp_bank(
    sigma_cfg: SigmaDistributionConfig,
    y_shape: torch.Size,
    d_S: int,
    device: torch.device,
    num_samples: int = 12,
    seed: Optional[int] = None,
) -> JVPBank:
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    c = y_shape[1]
    sigmas, epsilons, epsilons_offset, etas = [], [], [], []
    for _ in range(num_samples):
        sigmas.append(sample_sigma_training_distribution(sigma_cfg, 1, device, generator=gen).detach())
        epsilons.append(torch.randn(y_shape, device=device, generator=gen).detach())
        epsilons_offset.append(torch.randn(1, c, 1, 1, device=device, generator=gen).detach())
        etas.append(torch.randn(d_S, device=device, generator=gen).detach())
    return JVPBank(tuple(sigmas), tuple(epsilons), tuple(epsilons_offset), tuple(etas))

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


def score_one_jvp_bank(
    denoiser: Denoiser,
    theta_s_named: Dict[str, Tensor],
    frozen_named: Dict[str, Tensor],
    h_D_inv_sqrt: Tensor,
    bank: JVPBank,
    candidates: List[Candidate],
    chunk_size: int,
) -> Tensor:
    """theta_s_named doubles as the flatten/unflatten template (it is always the same
    ordered mapping the tangent must be unflattened against, so no separate `template`
    argument is needed). Only H_D^{-1/2} is required mathematically -- the caller is
    responsible for h_D's own validity (finite/positive/dimension); this function only
    checks h_D_inv_sqrt's dimension matches theta_S's, and reads its device from it
    directly rather than taking a separate h_D argument just for that.

    Returns the score tensor on the SAME device as h_D_inv_sqrt (no implicit GPU->CPU
    transfer) -- callers that need it elsewhere are responsible for their own `.to(...)`."""
    device = h_D_inv_sqrt.device
    d_S = sum(p.numel() for p in theta_s_named.values())
    assert h_D_inv_sqrt.numel() == d_S, f"h_D_inv_sqrt dim {h_D_inv_sqrt.numel()} != theta_S dim {d_S}"

    num_entries = bank.num_samples
    num_candidates = len(candidates)
    total_scores = torch.zeros(num_candidates, device=device)

    chunks = []
    for start in range(0, num_candidates, chunk_size):
        chunk = candidates[start : start + chunk_size]
        obs_batch = torch.cat([c[0] for c in chunk], dim=0)
        act_batch = torch.cat([c[1] for c in chunk], dim=0)
        y_batch = torch.cat([c[2] for c in chunk], dim=0)
        chunks.append((start, obs_batch, act_batch, y_batch, len(chunk)))

    for m in range(num_entries):
        sigma, eps, eps_offset, eta = bank.sigmas[m], bank.epsilons[m], bank.epsilons_offset[m], bank.etas[m]
        with torch.no_grad():
            z_flat = h_D_inv_sqrt * eta
        tangent_named = unflatten_to_dict(z_flat, theta_s_named)  # hoisted: once per probe, not once per (chunk, probe)

        for start, obs_batch, act_batch, y_batch, B in chunks:
            y_sigma_batch = apply_noise_from_samples(y_batch, sigma, eps, eps_offset, denoiser.cfg.sigma_offset_noise).detach()
            _, jvp_out = jvp_through_F(
                denoiser, theta_s_named, frozen_named, tangent_named, y_sigma_batch, sigma, obs_batch, act_batch
            )
            contribution = 2.0 * jvp_out.reshape(B, -1).square().sum(dim=1)
            total_scores[start : start + B] += contribution / num_entries

    return total_scores.detach()
