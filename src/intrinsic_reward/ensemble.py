"""MAX-inspired diffusion ensemble disagreement baseline (not an exact MAX reproduction and
not an exact information gain).

K independently initialized denoisers are trained on the same reward-free replay, each with
its own optimizer, scheduler, sampling and diffusion corruption (a private RNG stream), and,
with bootstrap enabled, its own episode-level Poisson(1) bootstrap of the replay. In
imagination every member predicts the next frame from the same history and action with the
same initial diffusion noise; the intrinsic reward is their disagreement (see
ensemble_disagreement), and one uniformly chosen member's prediction (per trajectory, per
transition) becomes the next observation.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from envs.world_model_env import ensemble_disagreement  # noqa: F401  (re-exported: the score)
from intrinsic_reward.lcg.reward_normalization import RunningRMS, RunningRMSConfig
from utils import rng_state, set_rng_state, set_seed

REWARD_SCALINGS = ("fixed", "rms", "none")
MEMBER_SELECTIONS = ("uniform_per_transition",)


@dataclass
class EnsembleDisagreementConfig:
    """All ensemble-disagreement settings: the ensemble (num_members, bootstrap) and the
    reward scoring (the rest)."""
    method: str = "ensemble_disagreement"
    enabled: bool = True
    num_members: int = 5  # ensemble construction/training: denoisers (>= 2)
    bootstrap: bool = True  # episode-level Poisson(1) bootstrap per member
    num_noise_samples: int = 1  # shared-noise samples per transition; scores averaged
    member_selection: str = "uniform_per_transition"
    reward_scaling: str = "fixed"  # fixed: raw / fixed_divisor only; rms: RunningRMS (reset each epoch); none: raw
    fixed_divisor: Optional[float] = 1.0
    rms_alpha: float = 1.0
    rms_ema_decay: float = 0.99
    rms_eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.method != "ensemble_disagreement":
            raise ValueError(f"EnsembleDisagreementConfig.method must be ensemble_disagreement, got {self.method!r}")
        check_ensemble_settings(self.num_members, self.bootstrap)
        if isinstance(self.num_noise_samples, bool) or not isinstance(self.num_noise_samples, int) or self.num_noise_samples < 1:
            raise ValueError(f"num_noise_samples must be a positive integer, got {self.num_noise_samples!r}")
        if self.member_selection not in MEMBER_SELECTIONS:
            raise ValueError(f"member_selection must be one of {MEMBER_SELECTIONS}, got {self.member_selection!r}")
        if self.reward_scaling not in REWARD_SCALINGS:
            raise ValueError(f"reward_scaling must be one of {REWARD_SCALINGS}, got {self.reward_scaling!r}")
        if self.reward_scaling == "fixed":
            d = self.fixed_divisor
            if isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(d) or d <= 0:
                raise ValueError(f"fixed_divisor must be a finite positive number, got {d!r}")
        if self.reward_scaling == "rms" and not (self.rms_alpha > 0 and 0 <= self.rms_ema_decay < 1 and self.rms_eps > 0):
            raise ValueError("Invalid rms_alpha / rms_ema_decay / rms_eps")

    def reward_scaling_settings(self) -> Dict:
        """Everything that turns member predictions into policy reward; a resume must not change it."""
        settings = dict(method=self.method, num_noise_samples=self.num_noise_samples,
                        member_selection=self.member_selection, mode=self.reward_scaling)
        if self.reward_scaling == "fixed":
            settings["fixed_divisor"] = float(self.fixed_divisor)
        if self.reward_scaling == "rms":
            settings.update(alpha=self.rms_alpha, ema_decay=self.rms_ema_decay, eps=self.rms_eps)
        return settings


def check_ensemble_settings(members, bootstrap) -> None:
    if isinstance(members, bool) or not isinstance(members, int) or members < 2:
        raise ValueError(f"intrinsic_reward.num_members must be an integer >= 2 for ensemble disagreement, got {members!r}")
    if not isinstance(bootstrap, bool):
        raise ValueError(f"intrinsic_reward.bootstrap must be true or false, got {bootstrap!r}")


LEGACY_FIELDS = (("members", "num_members"), ("bootstrap", "bootstrap"))


def migrate_ensemble_layout(cfg):
    """Read a config saved before the ensemble settings moved into intrinsic_reward: a
    top-level `ensemble: {members, bootstrap}` becomes intrinsic_reward.num_members/bootstrap
    (saved values kept). Disagreeing layouts are rejected; for other methods the old section
    was unused and is dropped. Returns a copy; the input (and any file) is unchanged."""
    from omegaconf import OmegaConf, open_dict
    legacy = cfg.get("ensemble") if hasattr(cfg, "get") else None
    if legacy is None:
        return cfg
    cfg = OmegaConf.create(OmegaConf.to_container(cfg))  # unresolved copy
    intrinsic = cfg.get("intrinsic_reward")
    is_ensemble = intrinsic is not None and intrinsic.get("method") == "ensemble_disagreement"
    with open_dict(cfg):
        for old, new in LEGACY_FIELDS:
            if old not in legacy:
                continue
            if is_ensemble:
                if new in intrinsic and intrinsic[new] != legacy[old]:
                    raise ValueError(f"Conflicting ensemble settings: ensemble.{old}={legacy[old]!r} (older layout) but "
                                     f"intrinsic_reward.{new}={intrinsic[new]!r}; keep one layout")
                intrinsic[new] = legacy[old]
        del cfg["ensemble"]
    return cfg


def episode_multiplicity(seed: int, member: int, episode_id: int) -> int:
    """Poisson(1) bootstrap multiplicity of one episode for one member: a pure function of
    (seed, member, episode id), so it never changes as the replay grows or after a resume."""
    return int(np.random.default_rng([int(seed), 3, int(member), int(episode_id)]).poisson(1.0))


class EnsembleTraining:
    """Per-member training state beyond the weights (held in Agent): optimizers and schedulers
    of members 1..K-1 (member 0 uses the trainer's denoiser optimizer), update counters,
    private RNG streams, bootstrap multiplicities and empty-bootstrap fallbacks. Saved in
    checkpoints."""

    def __init__(self, members: int, bootstrap: bool, seed: int, optimizers: List, schedulers: List) -> None:
        self.members, self.bootstrap, self.seed = members, bootstrap, int(seed)
        self.optimizers, self.schedulers = optimizers, schedulers  # members 1..K-1
        self.updates = [0] * members
        self.empty_bootstrap_batches = [0] * members
        self.multiplicities: List[List[int]] = [[] for _ in range(members)]
        self.rng = []
        saved = rng_state()
        for k in range(members):
            set_seed(int(np.random.SeedSequence([self.seed, 100 + k]).generate_state(1)[0]))
            self.rng.append(rng_state())
        set_rng_state(saved)

    def settings(self) -> Dict:
        return dict(members=self.members, bootstrap=self.bootstrap)

    def multiplicity(self, member: int, num_episodes: int) -> np.ndarray:
        """Multiplicities of episodes 0..num_episodes-1; new episodes get theirs on first use."""
        m = self.multiplicities[member]
        while len(m) < num_episodes:
            m.append(episode_multiplicity(self.seed, member, len(m)))
        return np.asarray(m[:num_episodes], dtype=np.float64)

    @contextmanager
    def member_rng(self, member: int):
        """Run member training (sampling and corruption) on its own checkpointed RNG stream."""
        outer = rng_state()
        set_rng_state(self.rng[member])
        try:
            yield
        finally:
            self.rng[member] = rng_state()
            set_rng_state(outer)

    def state_dict(self) -> Dict:
        return dict(settings=self.settings(), seed=self.seed, updates=list(self.updates),
                    empty_bootstrap_batches=list(self.empty_bootstrap_batches),
                    multiplicities=[list(m) for m in self.multiplicities], rng=self.rng,
                    optimizers=[o.state_dict() for o in self.optimizers],
                    schedulers=[s.state_dict() for s in self.schedulers])

    def load_state_dict(self, state: Dict) -> None:
        if state is None or state["settings"] != self.settings():
            saved = None if state is None else state["settings"]
            raise ValueError(f"Checkpoint was trained with ensemble settings {saved}, the config gives "
                             f"{self.settings()}; a resume must not change them")
        self.seed = state["seed"]
        self.updates = list(state["updates"])
        self.empty_bootstrap_batches = list(state["empty_bootstrap_batches"])
        self.multiplicities = [list(m) for m in state["multiplicities"]]
        self.rng = state["rng"]
        for o, s in zip(self.optimizers, state["optimizers"]):
            o.load_state_dict(s)
        for sch, s in zip(self.schedulers, state["schedulers"]):
            sch.load_state_dict(s)


class EnsembleScorer:
    """PMPO intrinsic-reward hook: reads each step's ensemble disagreement (computed by
    WorldModelEnv from the members' shared-noise predictions) and scales it."""

    metric_prefix = "ens"

    def __init__(self, cfg: EnsembleDisagreementConfig) -> None:
        self.cfg = cfg
        self.rms: Optional[RunningRMS] = None
        self.last_raw: Optional[Tensor] = None
        self.last_scaled: Optional[Tensor] = None
        self.reset()

    def reset(self) -> None:
        """Start of an epoch: RunningRMS (if used) restarts, as LCG's does at each refresh."""
        if self.cfg.reward_scaling == "rms":
            self.rms = RunningRMS(RunningRMSConfig(enabled=True, alpha=self.cfg.rms_alpha,
                                                   ema_decay=self.cfg.rms_ema_decay, eps=self.cfg.rms_eps))

    def scale(self, raw: Tensor) -> Tensor:
        if self.cfg.reward_scaling == "fixed":
            return raw / self.cfg.fixed_divisor
        if self.cfg.reward_scaling == "rms":
            return self.rms(raw)
        return raw

    def scale_frozen(self, raw: Tensor) -> Optional[Tensor]:
        """Policy reward of raw scores without updating RunningRMS (evaluation; None before its
        first update this epoch)."""
        if self.cfg.reward_scaling == "rms":
            return self.rms.apply_frozen(raw)
        return self.scale(raw)

    def state_dict(self) -> Dict:
        """The frozen scale (RunningRMS s2 when reward_scaling=rms): with it a resume scores
        evaluation exactly as before the checkpoint. Members and banks are checkpointed elsewhere."""
        s2 = None if self.rms is None or self.rms.s2 is None else self.rms.s2.cpu()
        return dict(settings=self.cfg.reward_scaling_settings(), rms_s2=s2)

    def load_state_dict(self, state: Dict, device: torch.device) -> None:
        if state["settings"] != self.cfg.reward_scaling_settings():
            raise ValueError(f"Checkpoint's {self.metric_prefix} scorer was built with {state['settings']}, the config "
                             f"gives {self.cfg.reward_scaling_settings()}; a resume must not change them")
        self.reset()
        if self.rms is not None and state["rms_s2"] is not None:
            self.rms.s2 = state["rms_s2"].to(device)

    def __call__(self, infos: List[Dict], env_rew: Tensor) -> Tensor:
        raw = torch.stack([info["ensemble_disagreement"] for info in infos], dim=1)  # (envs, steps)
        raw = raw.to(device=env_rew.device, dtype=env_rew.dtype)
        reward = self.scale(raw)
        self.last_raw, self.last_scaled = raw.detach(), reward.detach()
        return reward
