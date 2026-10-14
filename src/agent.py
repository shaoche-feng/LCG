from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from envs import TorchEnv, WorldModelEnv
from models.pmpo_beta import PMPOBeta, PMPOBetaConfig
from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.rew_end_model import RewEndModel, RewEndModelConfig
from utils import extract_state_dict


@dataclass
class AgentConfig:
    denoiser: DenoiserConfig
    rew_end_model: RewEndModelConfig
    actor_critic: PMPOBetaConfig
    continuous_action_dim: Optional[int] = None  # continuous action dimension, e.g. DM Control
    action_low: Optional[List[float]] = None  # required iff continuous_action_dim is set
    action_high: Optional[List[float]] = None  # required iff continuous_action_dim is set

    def __post_init__(self) -> None:
        if self.continuous_action_dim is None or self.continuous_action_dim < 1:
            raise ValueError("AgentConfig requires an environment-derived continuous action dimension")
        if not isinstance(self.actor_critic, PMPOBetaConfig):
            raise TypeError("AgentConfig.actor_critic must be PMPOBetaConfig")

        self.denoiser.inner_model.continuous_action_dim = self.continuous_action_dim

        self.rew_end_model.continuous_action_dim = self.continuous_action_dim

        self.actor_critic.continuous_action_dim = self.continuous_action_dim
        self.actor_critic.action_low = self.action_low
        self.actor_critic.action_high = self.action_high


def get_action_space_kwargs(env: TorchEnv) -> Dict[str, Any]:
    """Read continuous action dimensions and bounds once for all three models."""
    return dict(
        continuous_action_dim=int(env.action_dim),
        action_low=env.action_low.tolist(),
        action_high=env.action_high.tolist(),
    )


class Agent(nn.Module):
    def __init__(self, cfg: AgentConfig) -> None:
        super().__init__()
        self.denoiser = Denoiser(cfg.denoiser)
        self.rew_end_model = RewEndModel(cfg.rew_end_model)
        self.actor_critic = PMPOBeta(cfg.actor_critic)

    @property
    def device(self):
        return self.denoiser.device

    def setup_training(
        self,
        sigma_distribution_cfg: SigmaDistributionConfig,
        rl_env: WorldModelEnv,
    ) -> None:
        self.denoiser.setup_training(sigma_distribution_cfg)
        self.actor_critic.setup_training(rl_env)

    def load(
        self,
        path_to_ckpt: Path,
        load_denoiser: bool = True,
        load_rew_end_model: bool = True,
        load_actor_critic: bool = True,
    ) -> None:
        # Trusted local agent snapshots may include PMPO's numpy/Python RNG state.
        sd = torch.load(Path(path_to_ckpt), map_location=self.device, weights_only=False)
        sd = {k: extract_state_dict(sd, k) for k in ("denoiser", "rew_end_model", "actor_critic")}
        if load_denoiser:
            self.denoiser.load_state_dict(sd["denoiser"])
        if load_rew_end_model:
            self.rew_end_model.load_state_dict(sd["rew_end_model"])
        if load_actor_critic:
            self.actor_critic.load_state_dict(sd["actor_critic"])
