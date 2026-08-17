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
    num_actions: Optional[int] = None  # discrete action count, e.g. Atari
    continuous_action_dim: Optional[int] = None  # continuous action dimension, e.g. DM Control
    continuous_reward: bool = False  # False: Atari-style reward-sign classification; True: scalar regression

    # NOTE: no `__post_init__` validation of num_actions/continuous_action_dim here, on purpose,
    # mirroring InnerModelConfig (models/diffusion/inner_model.py). `num_actions` is left unset
    # (None) in config/agent/default.yaml and patched onto an already-constructed
    # RewEndModelConfig instance by AgentConfig.__post_init__ (agent.py), i.e. *after* hydra's
    # bottom-up instantiate() has already built this dataclass. Validating "exactly one of the
    # two is set" here would fire during that intermediate state and break the existing Atari
    # path. The check instead lives in RewEndModel.__init__, which only runs once the config is
    # fully populated.


class RewEndModel(nn.Module):
    def __init__(self, cfg: RewEndModelConfig) -> None:
        super().__init__()
        assert (cfg.num_actions is None) != (cfg.continuous_action_dim is None), (
            "RewEndModelConfig requires exactly one of `num_actions` (discrete action space) "
            "or `continuous_action_dim` (continuous action space) to be set."
        )
        self.cfg = cfg
        self.continuous_action = cfg.continuous_action_dim is not None
        self.continuous_reward = cfg.continuous_reward
        self.num_rew_outputs = 1 if cfg.continuous_reward else 3

        self.encoder = RewEndEncoder(2 * cfg.img_channels, cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)
        if self.continuous_action:
            self.act_emb = nn.Linear(cfg.continuous_action_dim, cfg.cond_channels)  # R^action_dim -> embedding
        else:
            self.act_emb = nn.Embedding(cfg.num_actions, cfg.cond_channels)  # action index -> embedding
        input_dim_lstm = cfg.channels[-1] * (cfg.img_size // 2 ** (len(cfg.depths) - 1)) ** 2
        self.lstm = nn.LSTM(input_dim_lstm, cfg.lstm_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(cfg.lstm_dim, cfg.lstm_dim),
            nn.SiLU(),
            nn.Linear(cfg.lstm_dim, self.num_rew_outputs + 2, bias=False),
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
        act = act.reshape(b * t, -1) if self.continuous_action else act.reshape(b * t)
        x = self.encoder(torch.cat((obs, next_obs), dim=1), self.act_emb(act))
        x = x.reshape(b, t, -1)  # (b t) e h w -> b t (e h w)
        x, hx_cx = self.lstm(x, hx_cx)
        out = self.head(x)
        rew_out = out[:, :, : self.num_rew_outputs]
        if self.continuous_reward:
            rew_out = rew_out.squeeze(-1)  # (b, t, 1) -> (b, t) scalar reward prediction
        return rew_out, out[:, :, self.num_rew_outputs :], hx_cx

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

        if self.continuous_reward:
            target_rew = rew[mask]  # raw continuous reward, no sign()
            loss_rew = F.mse_loss(pred_rew, target_rew)
        else:
            target_rew = rew[mask].sign().long().add(1)  # clipped to {-1, 0, 1}
            loss_rew = F.cross_entropy(pred_rew, target_rew)

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
        if not self.continuous_reward:
            metrics["confusion_matrix"]["rew"] = multiclass_confusion_matrix(pred_rew, target_rew, num_classes=3)
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
