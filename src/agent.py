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
    rew_end_model: Optional[RewEndModelConfig]  # None: no reward/end model (exploration)
    actor_critic: PMPOBetaConfig
    continuous_action_dim: Optional[int] = None  # continuous action dimension, e.g. DM Control
    action_low: Optional[List[float]] = None  # required iff continuous_action_dim is set
    action_high: Optional[List[float]] = None  # required iff continuous_action_dim is set
    # Denoisers in total (member 0 is `denoiser`); > 1 only for ensemble disagreement.
    ensemble_members: int = 1

    def __post_init__(self) -> None:
        if self.continuous_action_dim is None or self.continuous_action_dim < 1:
            raise ValueError("AgentConfig requires an environment-derived continuous action dimension")
        if not isinstance(self.actor_critic, PMPOBetaConfig):
            raise TypeError("AgentConfig.actor_critic must be PMPOBetaConfig")

        self.denoiser.inner_model.continuous_action_dim = self.continuous_action_dim

        if self.rew_end_model is not None:
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
        self.rew_end_model = RewEndModel(cfg.rew_end_model) if cfg.rew_end_model is not None else None
        self.actor_critic = PMPOBeta(cfg.actor_critic)
        # Members 1..K-1, built last (independent initializations); empty for a single denoiser,
        # so single-model state dicts are unchanged.
        self.ensemble_members = nn.ModuleList(Denoiser(cfg.denoiser) for _ in range(cfg.ensemble_members - 1))

    @property
    def device(self):
        return self.denoiser.device

    @property
    def denoisers(self) -> List[Denoiser]:
        return [self.denoiser, *self.ensemble_members]

    def setup_training(
        self,
        sigma_distribution_cfg: SigmaDistributionConfig,
        rl_env: Optional[WorldModelEnv],
    ) -> None:
        for denoiser in self.denoisers:
            denoiser.setup_training(sigma_distribution_cfg)
        if rl_env is not None:  # None: no policy training (uniform baseline)
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
        # Only member 0 would be restored here, leaving other members randomly initialized:
        # refuse before loading anything.
        source_members = 1 + len({k.split(".")[1] for k in sd if k.startswith("ensemble_members.")})
        if load_denoiser and (source_members > 1 or len(self.ensemble_members) > 0):
            raise ValueError(
                f"Cannot initialize denoisers from {path_to_ckpt}: the snapshot holds {source_members} denoiser(s) and "
                f"this agent {len(self.denoisers)}. Denoiser ensembles are not warm-started through "
                "initialization.path_to_ckpt; start ensemble training fresh, or resume it with the full-state "
                "checkpoint workflow (scripts/resume.sh)."
            )
        sd = {k: extract_state_dict(sd, k) for k in ("denoiser", "rew_end_model", "actor_critic")}
        if load_denoiser:
            self.denoiser.load_state_dict(sd["denoiser"])
        if load_rew_end_model:
            if self.rew_end_model is None:
                raise ValueError("load_rew_end_model=True, but this agent has no reward/end model")
            self.rew_end_model.load_state_dict(sd["rew_end_model"])
        if load_actor_critic:
            self.actor_critic.load_state_dict(sd["actor_critic"])
