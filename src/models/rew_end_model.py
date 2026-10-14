from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_confusion_matrix

from .blocks import Conv3x3, Downsample, ResBlocks
from data import Batch
from utils import init_lstm, LossAndLogs


@dataclass
class RewEndModelConfig:
    lstm_dim: int
    img_channels: int
    img_size: int
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[int]
    continuous_action_dim: Optional[int] = None  # continuous action dimension, e.g. DM Control

    # Populated by AgentConfig after Hydra instantiates component configs.


class RewEndModel(nn.Module):
    def __init__(self, cfg: RewEndModelConfig) -> None:
        super().__init__()
        if cfg.continuous_action_dim is None or cfg.continuous_action_dim < 1:
            raise ValueError("An environment-derived continuous action dimension is required")
        self.cfg = cfg

        self.encoder = RewEndEncoder(2 * cfg.img_channels, cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)
        self.act_emb = nn.Linear(cfg.continuous_action_dim, cfg.cond_channels)
        input_dim_lstm = cfg.channels[-1] * (cfg.img_size // 2 ** (len(cfg.depths) - 1)) ** 2
        self.lstm = nn.LSTM(input_dim_lstm, cfg.lstm_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(cfg.lstm_dim, cfg.lstm_dim),
            nn.SiLU(),
            nn.Linear(cfg.lstm_dim, 3, bias=False),
        )
        init_lstm(self.lstm)

    def predict_rew_end(
        self,
        obs: Tensor,
        act: Tensor,
        next_obs: Tensor,
        hx_cx: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
        b, t, c, h, w = obs.shape
        obs, next_obs = obs.reshape(b * t, c, h, w), next_obs.reshape(b * t, c, h, w)
        act = act.reshape(b * t, -1)
        x = self.encoder(torch.cat((obs, next_obs), dim=1), self.act_emb(act))
        x = x.reshape(b, t, -1)  # (b t) e h w -> b t (e h w)
        x, hx_cx = self.lstm(x, hx_cx)
        out = self.head(x)
        rew_out = out[:, :, :1]
        rew_out = rew_out.squeeze(-1)  # (b, t, 1) -> (b, t) scalar reward prediction
        return rew_out, out[:, :, 1:], hx_cx

    def forward(self, batch: Batch) -> LossAndLogs:
        obs = batch.obs[:, :-1]
        act = batch.act[:, :-1]
        next_obs = batch.obs[:, 1:]
        rew = batch.rew[:, :-1]
        end = batch.end[:, :-1]
        mask = batch.mask_padding[:, :-1]

        # When dead, replace frame (gray padding) by true final obs
        dead = end.bool().any(dim=1)
        if dead.any():
            final_obs = torch.stack([i["final_observation"] for i, d in zip(batch.info, dead) if d]).to(obs.device)
            next_obs[dead, end[dead].argmax(dim=1)] = final_obs

        pred_rew, logits_end, _ = self.predict_rew_end(obs, act, next_obs)
        pred_rew = pred_rew[mask]
        logits_end = logits_end[mask]
        target_end = end[mask]

        target_rew = rew[mask]
        loss_rew = F.mse_loss(pred_rew, target_rew)

        loss_end = F.cross_entropy(logits_end, target_end)
        loss = loss_rew + loss_end

        metrics = {
            "loss_rew": loss_rew.detach(),
            "loss_end": loss_end.detach(),
            "loss_total": loss.detach(),
            "confusion_matrix": {
                "end": multiclass_confusion_matrix(logits_end, target_end, num_classes=2),
            },
        }
        return loss, metrics


class RewEndEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        depths: List[int],
        channels: List[int],
        attn_depths: List[int],
    ) -> None:
        super().__init__()
        assert len(depths) == len(channels) == len(attn_depths)
        self.conv_in = Conv3x3(in_channels, channels[0])
        blocks = []
        for i, n in enumerate(depths):
            c1 = channels[max(0, i - 1)]
            c2 = channels[i]
            blocks.append(
                ResBlocks(
                    list_in_channels=[c1] + [c2] * (n - 1),
                    list_out_channels=[c2] * n,
                    cond_channels=cond_channels,
                    attn=attn_depths[i],
                )
            )
        blocks.append(
            ResBlocks(
                list_in_channels=[channels[-1]] * 2,
                list_out_channels=[channels[-1]] * 2,
                cond_channels=cond_channels,
                attn=True,
            )
        )
        self.blocks = nn.ModuleList(blocks)
        self.downsamples = nn.ModuleList([nn.Identity()] + [Downsample(c) for c in channels[:-1]] + [nn.Identity()])

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        x = self.conv_in(x)
        for block, down in zip(self.blocks, self.downsamples):
            x = down(x)
            x, _ = block(x, cond)
        return x
