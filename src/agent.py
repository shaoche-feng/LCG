from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from envs import TorchEnv, WorldModelEnv
from models.actor_critic import ActorCritic, ActorCriticConfig, ActorCriticLossConfig
from models.pmpo_beta import PMPOBeta, PMPOBetaConfig
from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.rew_end_model import RewEndModel, RewEndModelConfig
from utils import extract_state_dict


@dataclass
class AgentConfig:
    denoiser: DenoiserConfig
    rew_end_model: RewEndModelConfig
    actor_critic: ActorCriticConfig
    num_actions: Optional[int] = None  # discrete action count, e.g. Atari
    continuous_action_dim: Optional[int] = None  # continuous action dimension, e.g. DM Control
    action_low: Optional[List[float]] = None  # required iff continuous_action_dim is set
    action_high: Optional[List[float]] = None  # required iff continuous_action_dim is set
    continuous_reward: bool = False  # False: Atari-style reward-sign lambda returns/classification

    def __post_init__(self) -> None:
        assert (self.num_actions is None) != (self.continuous_action_dim is None), (
            "AgentConfig requires exactly one of `num_actions` (discrete action space) or "
            "`continuous_action_dim` (continuous action space) to be set."
        )

        self.denoiser.inner_model.num_actions = self.num_actions
        self.denoiser.inner_model.continuous_action_dim = self.continuous_action_dim

        self.rew_end_model.num_actions = self.num_actions
        self.rew_end_model.continuous_action_dim = self.continuous_action_dim
        self.rew_end_model.continuous_reward = self.continuous_reward

        self.actor_critic.num_actions = self.num_actions
        self.actor_critic.continuous_action_dim = self.continuous_action_dim
        self.actor_critic.action_low = self.action_low
        self.actor_critic.action_high = self.action_high
        self.actor_critic.continuous_reward = self.continuous_reward


def get_action_space_kwargs(env: TorchEnv) -> Dict[str, Any]:
    """Central place that reads an environment's action-space metadata (discrete vs continuous,
    dimension, bounds) and turns it into the kwargs `AgentConfig` needs to configure InnerModel,
    RewEndModel, and ActorCritic consistently -- so the three models are never patched
    individually from separate call sites. `env` is expected to be a `TorchEnv` exposing
    `is_discrete` plus either `num_actions` or `action_dim`/`action_low`/`action_high`.
    """
    if env.is_discrete:
        return dict(
            num_actions=int(env.num_actions),
            continuous_action_dim=None,
            action_low=None,
            action_high=None,
            continuous_reward=False,
        )
    else:
        return dict(
            num_actions=None,
            continuous_action_dim=int(env.action_dim),
            action_low=env.action_low.tolist(),
            action_high=env.action_high.tolist(),
            continuous_reward=True,
        )


class Agent(nn.Module):
    def __init__(self, cfg: AgentConfig) -> None:
        super().__init__()
        self.denoiser = Denoiser(cfg.denoiser)
        self.rew_end_model = RewEndModel(cfg.rew_end_model)
        controller = PMPOBeta if isinstance(cfg.actor_critic, PMPOBetaConfig) else ActorCritic
        self.actor_critic = controller(cfg.actor_critic)

    @property
    def device(self):
        return self.denoiser.device

    def setup_training(
        self,
        sigma_distribution_cfg: SigmaDistributionConfig,
        actor_critic_loss_cfg: ActorCriticLossConfig,
        rl_env: Union[TorchEnv, WorldModelEnv],
    ) -> None:
        self.denoiser.setup_training(sigma_distribution_cfg)
        self.actor_critic.setup_training(rl_env, actor_critic_loss_cfg)

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
