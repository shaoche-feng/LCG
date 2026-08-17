from typing import List

import torch
from torch import Tensor

from models.diffusion.denoiser import SigmaDistributionConfig

_QUANTILE_EPS = 1e-6


def sample_sigma_stratum(
    cfg: SigmaDistributionConfig,
    stratum_idx: int,
    num_strata: int,
    n: int,
    device: torch.device,
) -> Tensor:
    """Draw n sigmas from one equal-probability stratum of p_train(sigma) in log-sigma
    space. p_train is the same log-normal used by Denoiser.sample_sigma_training
    (sigma = exp(loc + scale*z).clip(sigma_min, sigma_max), z ~ N(0,1)); a stratum
    restricts z to the quantile range [stratum_idx/num_strata, (stratum_idx+1)/num_strata]
    of the standard normal via inverse-CDF sampling. num_strata=1 reproduces the
    unstratified distribution exactly (the single-sample ablation case).
    """
    assert 0 <= stratum_idx < num_strata
    lo_q = max(stratum_idx / num_strata, _QUANTILE_EPS)
    hi_q = min((stratum_idx + 1) / num_strata, 1 - _QUANTILE_EPS)
    u = torch.empty(n, device=device).uniform_(lo_q, hi_q)
    standard_normal = torch.distributions.Normal(
        torch.zeros((), device=device), torch.ones((), device=device)
    )
    z = standard_normal.icdf(u)
    sigma = (z * cfg.scale + cfg.loc).exp().clip(cfg.sigma_min, cfg.sigma_max)
    return sigma


def sample_sigma_strata(
    cfg: SigmaDistributionConfig,
    num_strata: int,
    n: int,
    device: torch.device,
) -> List[Tensor]:
    """One draw of n sigmas per stratum, for m = 0 .. num_strata-1."""
    return [sample_sigma_stratum(cfg, m, num_strata, n, device) for m in range(num_strata)]
