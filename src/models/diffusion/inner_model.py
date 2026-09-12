from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import Conv3x3, FourierFeatures, GroupNorm, UNet


@dataclass
class InnerModelConfig:
    img_channels: int
    num_steps_conditioning: int
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[bool]
    num_actions: Optional[int] = None  # discrete action count, e.g. Atari
    continuous_action_dim: Optional[int] = None  # continuous action dimension, e.g. DM Control

    # NOTE: no `__post_init__` validation here on purpose. `num_actions` is left unset (None) in
    # config/agent/default.yaml and patched onto an already-constructed InnerModelConfig instance
    # by AgentConfig.__post_init__ (agent.py), i.e. *after* hydra's bottom-up instantiate() has
    # already built this dataclass. Validating "exactly one of the two is set" here would fire
    # during that intermediate state and break the existing Atari path. The check instead lives in
    # InnerModel.__init__, which only runs once the config is fully populated.


class InnerModel(nn.Module):
    def __init__(self, cfg: InnerModelConfig) -> None:
        super().__init__()
        assert (cfg.num_actions is None) != (cfg.continuous_action_dim is None), (
            "InnerModelConfig requires exactly one of `num_actions` (discrete action space) "
            "or `continuous_action_dim` (continuous action space) to be set."
        )
        self.noise_emb = FourierFeatures(cfg.cond_channels)

        act_emb_dim = cfg.cond_channels // cfg.num_steps_conditioning

        if cfg.continuous_action_dim is None:
            act_proj = nn.Embedding(cfg.num_actions, act_emb_dim)  # discrete: action index -> embedding
        else:
            act_proj = nn.Linear(cfg.continuous_action_dim, act_emb_dim)  # continuous: R^action_dim -> embedding
        self.act_emb = nn.Sequential(
            act_proj,
            nn.Flatten(),  # b t e -> b (t e)
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )
        
        self.conv_in = Conv3x3((cfg.num_steps_conditioning + 1) * cfg.img_channels, cfg.channels[0])

        self.unet = UNet(cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)

        self.norm_out = GroupNorm(cfg.channels[0])
        self.conv_out = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out.weight)

    def forward(self, noisy_next_obs: Tensor, c_noise: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        cond = self.cond_proj(self.noise_emb(c_noise) + self.act_emb(act))
        x = self.conv_in(torch.cat((obs, noisy_next_obs), dim=1))
        x, _, _ = self.unet(x, cond)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x
