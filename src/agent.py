from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from envs import TorchEnv, WorldModelEnv
from models.actor_critic import ActorCritic, ActorCriticConfig, ActorCriticLossConfig
from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.drq_actor_critic import DrQActorCritic, DrQActorCriticConfig, DrQLossConfig
from models.rew_end_model import RewEndModel, RewEndModelConfig
from utils import extract_state_dict


@dataclass
class AgentConfig:
    denoiser: DenoiserConfig
    rew_end_model: RewEndModelConfig
    # Which class this resolves to (ActorCriticConfig vs DrQActorCriticConfig) is driven by the
    # `_target_` in config/agent/*.yaml's actor_critic section -- see Agent.__init__'s dispatch
    # on the resolved type.
    actor_critic: Union[ActorCriticConfig, DrQActorCriticConfig]
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

        self.actor_critic.continuous_action_dim = self.continuous_action_dim
        self.actor_critic.action_low = self.action_low
        self.actor_critic.action_high = self.action_high
        if isinstance(self.actor_critic, ActorCriticConfig):
            # DrQActorCriticConfig has no num_actions/continuous_reward fields: DrQ is
            # continuous-action-only (see models.drq_actor_critic's module docstring) and its
            # TD target always uses raw rewards directly (no discrete reward-sign-clipping
            # branch), so neither has a DrQ equivalent to populate.
            self.actor_critic.num_actions = self.num_actions
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
        # Dispatches on the resolved config TYPE (set by config/agent/*.yaml's actor_critic
        # `_target_`), not an explicit flag -- cfg.actor_critic is already a fully-instantiated
        # ActorCriticConfig or DrQActorCriticConfig by the time Agent.__init__ runs (Hydra's
        # instantiate() resolves nested _target_s bottom-up before AgentConfig.__post_init__
        # even runs), so this is a plain, ordinary isinstance check, not polymorphic Hydra
        # instantiation of the model itself.
        if isinstance(cfg.actor_critic, DrQActorCriticConfig):
            self.actor_critic = DrQActorCritic(cfg.actor_critic)
        else:
            self.actor_critic = ActorCritic(cfg.actor_critic)

    @property
    def device(self):
        return self.denoiser.device

    def setup_training(
        self,
        sigma_distribution_cfg: SigmaDistributionConfig,
        actor_critic_loss_cfg: Union[ActorCriticLossConfig, DrQLossConfig],
        rl_env: Union[TorchEnv, WorldModelEnv],
        actor_critic_noise_generator: Optional[torch.Generator] = None,
    ) -> None:
        self.denoiser.setup_training(sigma_distribution_cfg)
        if isinstance(self.actor_critic, DrQActorCritic):
            assert actor_critic_noise_generator is not None, (
                "DrQActorCritic.setup_training requires an explicit noise_generator for the "
                "imagined-training loop's own exploration stream -- see "
                "data.batch_sampler.COMPONENT_SEED_ID's drq_imagination_noise id."
            )
            self.actor_critic.setup_training(rl_env, actor_critic_loss_cfg, actor_critic_noise_generator)
        else:
            self.actor_critic.setup_training(rl_env, actor_critic_loss_cfg)

    def load(
        self,
        path_to_ckpt: Path,
        load_denoiser: bool = True,
        load_rew_end_model: bool = True,
        load_actor_critic: bool = True,
    ) -> None:
        sd = torch.load(Path(path_to_ckpt), map_location=self.device)
        sd = {k: extract_state_dict(sd, k) for k in ("denoiser", "rew_end_model", "actor_critic")}
        if load_denoiser:
            self.denoiser.load_state_dict(sd["denoiser"])
        if load_rew_end_model:
            self.rew_end_model.load_state_dict(sd["rew_end_model"])
        if load_actor_critic:
            self.actor_critic.load_state_dict(sd["actor_critic"])
