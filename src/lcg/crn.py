from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from models.diffusion.denoiser import SigmaDistributionConfig

from .sigma_strata import sample_sigma_stratum

DEFAULT_NUM_CRN_BANKS = 2


@dataclass(frozen=True)
class CRNBank:
    """One common-random-numbers bank: exactly one (sigma, eps, xi) sample for each of the
    num_strata equal-probability log-sigma strata, held as concrete, already-materialized
    tensors -- not generators or seeds. There is no random state to consume or mutate:
    scoring any number of candidates with this bank just rereads the same fixed tensors."""

    sigmas: Tuple[Tensor, ...]
    epsilons: Tuple[Tensor, ...]
    xis: Tuple[Tensor, ...]

    def __post_init__(self) -> None:
        assert len(self.sigmas) == len(self.epsilons) == len(self.xis)

    @property
    def num_strata(self) -> int:
        return len(self.sigmas)


def make_crn_bank(
    sigma_cfg: SigmaDistributionConfig,
    y_shape: torch.Size,
    device: torch.device,
    num_strata: int = 3,
    seed: Optional[int] = None,
) -> CRNBank:
    """RNG isolation: uses a local, device-matched torch.Generator (seeded from `seed`
    when given, else auto-seeded from entropy) rather than global torch.manual_seed, so
    building a bank never mutates the global torch RNG stream DIAMOND's own code
    observes. Same seed -> same bank; different seeds -> different banks (unchanged)."""
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    sigmas, epsilons, xis = [], [], []
    for m in range(num_strata):
        sigmas.append(sample_sigma_stratum(sigma_cfg, m, num_strata, 1, device, generator=gen).detach())
        epsilons.append(torch.randn(y_shape, device=device, generator=gen).detach())
        xis.append(torch.randn(y_shape, device=device, generator=gen).detach())
    return CRNBank(tuple(sigmas), tuple(epsilons), tuple(xis))


def make_crn_bank_set(
    sigma_cfg: SigmaDistributionConfig,
    y_shape: torch.Size,
    device: torch.device,
    num_crn_banks: int = DEFAULT_NUM_CRN_BANKS,
    num_strata: int = 3,
    seed: Optional[int] = None,
) -> Tuple[CRNBank, ...]:
    """num_crn_banks independent CRN banks (default 2, per the Stage 3.7 initial
    integration default). All candidates scored against this set receive identical
    per-stratum, per-bank (sigma, eps, xi) draws -- CRN's entire purpose."""
    banks = []
    for b in range(num_crn_banks):
        bank_seed = None if seed is None else seed + b
        banks.append(make_crn_bank(sigma_cfg, y_shape, device, num_strata=num_strata, seed=bank_seed))
    return tuple(banks)
