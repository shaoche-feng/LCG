from typing import List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn

from models.diffusion.denoiser import Denoiser


def edm_weight(c_out: Tensor) -> Tensor:
    """w(sigma) = 1 / c_out(sigma)^2, matching DIAMOND's existing F-space MSE training loss
    (||F_theta - target||^2 = c_out^-2 ||D_theta - y||^2, see Denoiser.forward)."""
    return c_out.pow(-2)


def differentiable_denoise(
    denoiser: Denoiser, y_sigma: Tensor, sigma: Tensor, obs: Tensor, act: Tensor
) -> Tuple[Tensor, Tensor]:
    """D_theta(y_sigma, sigma, x), fully differentiable w.r.t. denoiser parameters
    (unlike Denoiser.denoise/wrap_model_output, which are @torch.no_grad and quantize to
    uint8 for autoregressive rollout conditioning). Also returns w(sigma), broadcastable
    against the returned tensor's shape.
    """
    cs = denoiser.compute_conditioners(sigma)
    model_output = denoiser.compute_model_output(y_sigma, obs, act, cs)
    d_theta = cs.c_skip * y_sigma + cs.c_out * model_output
    return d_theta, edm_weight(cs.c_out)


def compute_vjp(
    denoiser: Denoiser,
    params: List[nn.Parameter],
    y_sigma: Tensor,
    sigma: Tensor,
    obs: Tensor,
    act: Tensor,
    xi: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]:
    """v(theta_S; sigma, eps, xi) = grad_{theta_S}[ sqrt(2 w(sigma)) xi^T D_theta(y_sigma,sigma,x) ],
    for one transition (x, y) and one already-sampled (sigma, eps, xi) -- i.e. y_sigma is
    expected to already equal y + sigma*eps. Computed as a single VJP (one backward pass),
    without ever materializing the Jacobian J = d D_theta / d theta_S.

    xi defaults to a fresh N(0, I) draw shaped like the denoiser output (denoiser-output
    space, per the LCG spec). `generator=None` (default) draws it via the global torch RNG
    (byte-identical to the original torch.randn_like(d_theta) behavior); passing an
    explicit torch.Generator (device-matched to d_theta) makes this draw not consume/
    mutate global RNG state -- used by LCG's production callers for RNG isolation from
    DIAMOND. Ignored if `xi` is supplied explicitly.

    Handles exactly one transition at a time (y_sigma.size(0) == 1): batching multiple
    dataset examples into a single backward pass would sum their per-example VJPs together
    rather than keep them separate, which Algorithm 2/3's per-example accumulation needs.
    """
    assert y_sigma.size(0) == 1, "compute_vjp handles one transition (batch size 1) at a time"
    d_theta, w = differentiable_denoise(denoiser, y_sigma, sigma, obs, act)
    if xi is None:
        xi = torch.randn(d_theta.shape, dtype=d_theta.dtype, device=d_theta.device, generator=generator)
    scalar = (torch.sqrt(2 * w) * xi * d_theta).sum()
    grads = torch.autograd.grad(scalar, params, retain_graph=False, create_graph=False)
    v = torch.cat([g.reshape(-1) for g in grads])
    return v, d_theta.detach()
