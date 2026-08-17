from typing import List

import torch.nn as nn

from models.diffusion.denoiser import Denoiser


def selected_submodules(denoiser: Denoiser) -> List[nn.Module]:
    """The parameter subset theta_S from Algorithm 1, line 4: the final decoder level's
    three conditional ResBlocks, output GroupNorm, and output convolution.

    `unet.u_blocks` is built bottom-up per resolution level then reversed (see
    InnerModel/UNet), so `u_blocks[-1]` is the finest-resolution decoder level -- the one
    whose output feeds directly into `norm_out` -> `conv_out` in InnerModel.forward.
    """
    inner = denoiser.inner_model
    return [inner.unet.u_blocks[-1], inner.norm_out, inner.conv_out]


def selected_parameters(denoiser: Denoiser) -> List[nn.Parameter]:
    params = []
    for module in selected_submodules(denoiser):
        params.extend(module.parameters())
    return params


def selected_dim(denoiser: Denoiser) -> int:
    return sum(p.numel() for p in selected_parameters(denoiser))
