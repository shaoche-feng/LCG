"""Candidate scoring of the frozen predictor (design: docs/implement/LCG_PREDICTOR_DESIGN.md).

Both backends use only the current standardized feature z_t and canonical action a_t: no next-feature
target, no diffusion-noise draw.

- jvp_bank: r = sum_m ||J(u) v_m||^2 / (M tau^2) with the round's frozen bank v_m = h_D^{-1/2} * eta_m,
  eta_m ~ N(0, I); one forward-mode JVP (torch.func.jvp) per probe over the whole candidate chunk.
  E_eta[||J v||^2] = sum_j ||J[:, j]||^2 / h_D[j], the exact backend's value.
- exact_trace: r = sum_j ||J(u)[:, j]||^2 / h_D[j] / tau^2 in closed form (curvature.exact_trace).
"""
from contextlib import contextmanager
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import func, nn, Tensor


class ParameterLayout:
    """Flat parameter vector <-> named tensors, in head.named_parameters() order."""

    def __init__(self, head: nn.Module) -> None:
        self.names = [n for n, _ in head.named_parameters()]
        self.shapes = [tuple(p.shape) for _, p in head.named_parameters()]
        self.numels = [int(np.prod(s)) for s in self.shapes]
        self.offsets = [int(o) for o in np.cumsum([0] + self.numels[:-1])]
        self.size = int(sum(self.numels))

    def split(self, flat: Tensor) -> Dict[str, Tensor]:
        """Views of flat (..., size) as name -> (..., *shape)."""
        lead = flat.shape[:-1]
        return {n: flat[..., o:o + k].reshape(*lead, *s)
                for n, s, k, o in zip(self.names, self.shapes, self.numels, self.offsets)}

    def flatten(self, tensors: Dict[str, Tensor]) -> Tensor:
        return torch.cat([tensors[n].reshape(-1) for n in self.names])

    def describe(self) -> List[Dict]:
        return [dict(name=n, shape=list(s), numel=k, offset=o)
                for n, s, k, o in zip(self.names, self.shapes, self.numels, self.offsets)]


def probe_bank(h_D: Tensor, num_probes: int, seed: int) -> Tensor:
    """(M, P) tangents v_m = h_D^{-1/2} * eta_m, eta ~ N(0, I) from a dedicated CPU generator (the same
    numbers on every device; no global RNG). float32, on h_D's device."""
    generator = torch.Generator().manual_seed(int(seed))
    eta = torch.randn(num_probes, h_D.numel(), generator=generator, dtype=torch.float32)
    return eta.to(h_D.device) * h_D.float().rsqrt().unsqueeze(0)


def jvp_squared_norms(head: nn.Module, params: Dict[str, Tensor], z: Tensor, a: Tensor,
                      tangents: Sequence[Dict[str, Tensor]]) -> Tensor:
    """(B,) sum_m ||J(z, a) v_m||^2 (summed over outputs and probes; the caller divides by M tau^2)."""
    total = None
    for tangent in tangents:
        _, jv = func.jvp(lambda p: func.functional_call(head, p, (z, a)), (params,), (tangent,))
        s = jv.square().sum(-1)
        total = s if total is None else total + s
    return total


@contextmanager
def matmul_tf32(allowed: bool):
    """TF32 for float32 CUDA matmuls inside (the trainer enables it globally), restored afterwards."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(allowed)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
