"""Visual policy encoder and raw-reward lambda returns shared by PMPO."""
from collections import namedtuple

import torch
from torch import Tensor, nn
from .blocks import Conv3x3, SmallResBlock

ActorCriticOutput = namedtuple("ActorCriticOutput", "logits_act val")

class ActorCriticEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        assert len(cfg.channels) == len(cfg.down)
        encoder_layers = [Conv3x3(cfg.img_channels, cfg.channels[0])]
        for i in range(len(cfg.channels)):
            encoder_layers.append(SmallResBlock(cfg.channels[max(0, i - 1)], cfg.channels[i]))
            if cfg.down[i]:
                encoder_layers.append(nn.MaxPool2d(2))
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


@torch.no_grad()
def compute_lambda_returns(
    rew: Tensor,
    end: Tensor,
    trunc: Tensor,
    val_bootstrap: Tensor,
    gamma: float,
    lambda_: float,
) -> Tensor:
    assert rew.ndim == 2 and rew.size() == end.size() == trunc.size() == val_bootstrap.size()

    end_or_trunc = (end + trunc).clip(max=1)
    not_end = 1 - end
    not_trunc = 1 - trunc

    lambda_returns = rew + not_end * gamma * (not_trunc * (1 - lambda_) + trunc) * val_bootstrap

    if lambda_ == 0:
        return lambda_returns

    last = val_bootstrap[:, -1]
    for t in reversed(range(rew.size(1))):
        lambda_returns[:, t] += end_or_trunc[:, t].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, t]

    return lambda_returns
