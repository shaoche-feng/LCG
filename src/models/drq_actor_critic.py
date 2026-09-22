"""DrQ-v2-style continuous-action actor-critic, trained exclusively on trajectories imagined
inside a DIAMOND WorldModelEnv (see envs.world_model_env.WorldModelEnv). Deliberately named
"DrQ-v2-style", not "DrQ-v2": several choices deviate from the original paper to fit DIAMOND's
imagined-rollout architecture without a rewrite of the surrounding pipeline --

  - No persistent replay buffer (see DrQActorCritic.forward's docstring): each imagined rollout
    IS the training minibatch, matching how DIAMOND already produces rollouts on-policy, one
    `env_loop.send()` at a time.
  - No target encoder: only the twin Q-heads are soft-updated; the (online) encoder is shared,
    trained by the critic loss, and used (no_grad) to featurize the n-step-ahead observation too.
  - Exploration noise std is an externally scheduled scalar (NoiseSchedule below), not learned.

This module implements the SAME external interface `coroutines.env_loop.make_env_loop` expects
of `model` (`predict_act_value(obs, hx_cx) -> (dist_params, val, (hx, cx))` and
`sample_action(dist_params, deterministic) -> (action, aux)`), so `env_loop.py` and
`coroutines/collector.py` need zero changes to drive either this or the original REINFORCE
`ActorCritic` (see agent.py, which selects between them via Hydra `_target_`).

Frame stacking: WorldModelEnv/env_loop only ever hand the model a SINGLE most-recent frame per
call (env.obs_buffer[:, -1]) -- the original ActorCritic's only source of temporal context is
its own LSTM hx/cx, not any multi-frame buffer. A feedforward DrQ actor needs an explicit short
frame stack, so this module repurposes env_loop's existing (hx, cx) threading: `hx` carries the
flattened (num_envs, frame_stack * img_channels * img_size * img_size) stack, `cx` is an unused
placeholder tensor. env_loop.py's hx/cx handling (.detach(), boolean indexing, multiplying by a
reset_gate, zero-init via `model.lstm_dim`) is generic tensor manipulation with no LSTM-specific
assumption, so this achieves the frame stack with ZERO changes to env_loop.py, and the stack
inherits env_loop's existing cross-rollout-call persistence AND RolloutHxCxState's existing
checkpoint/resume machinery for free (it round-trips as "hx" without any new checkpointed state).
`self.lstm_dim` below is a real attribute (required by env_loop.py's zero-init line), kept under
that name deliberately for zero env_loop.py diff even though this model has no LSTM -- see the
docstring on that attribute.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Conv3x3, SmallResBlock
from coroutines.env_loop import make_env_loop, RolloutHxCxState
from envs import TorchEnv, WorldModelEnv
from utils import LossAndLogs


@dataclass
class NoiseScheduleConfig:
    std_start: float = 1.0
    std_end: float = 0.1
    decay_steps: int = 50_000
    clip: float = 0.3  # truncation half-width for the exploration/target-smoothing noise


def _noise_std_at(cfg: NoiseScheduleConfig, step: int) -> float:
    """Externally scheduled exploration std -- NOT a learned log_std (unlike the original
    REINFORCE ActorCritic). Linear decay from std_start to std_end over decay_steps, std_end
    afterward. Pure function of `step`; the step counter itself lives on DrQExplorationState
    (see its docstring for why), not here."""
    frac = min(1.0, step / max(1, cfg.decay_steps))
    return cfg.std_start + frac * (cfg.std_end - cfg.std_start)


class DrQExplorationState:
    """Auxiliary, non-nn.Module state for exploration-noise sampling: the component-isolated
    torch.Generator (see utils.derive_torch_generator) driving all of this actor's noise draws,
    and the noise-schedule step counter. Checkpointed explicitly by Trainer's
    ResumeFidelityState (mirroring coroutines.env_loop.RolloutHxCxState) rather than through
    nn.Module's own state_dict/load_state_dict.

    This is deliberate, not a style choice: nn.Module.state_dict() recursion DOES call an
    overridden `state_dict()` on each child module (normal Python method dispatch), but
    nn.Module.load_state_dict() recursion uses the internal `_load_from_state_dict` hook on
    each child, NOT the public `load_state_dict()` method -- so an override of the public
    method on a module nested under another (as DrQActorCritic is, under Agent) would silently
    never fire when loaded via a parent's `load_state_dict()` (e.g. Trainer's own resume path,
    via StateDictMixin calling `self.agent.load_state_dict(...)`) even though it WOULD fire
    when called directly (e.g. Agent.load()'s `self.actor_critic.load_state_dict(...)`). Rather
    than rely on an asymmetry like that for state this critical, exploration state is
    checkpointed the same explicit way rollout_hx_cx already is.
    """

    def __init__(self) -> None:
        self.generator: Optional[torch.Generator] = None
        self.schedule_step: int = 0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "generator_state": self.generator.get_state() if self.generator is not None else None,
            "schedule_step": self.schedule_step,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        generator_state = state_dict.get("generator_state")
        if generator_state is not None and self.generator is not None:
            self.generator.set_state(generator_state)
        self.schedule_step = state_dict.get("schedule_step", 0)


@dataclass
class DrQActorCriticConfig:
    img_channels: int
    img_size: int
    frame_stack: int  # number of most-recent frames stacked along the channel dim, e.g. 3
    encoder_channels: List[int]
    encoder_down: List[int]
    feature_dim: int  # flattened conv-encoder output size (computed by caller, asserted here)
    actor_hidden_dim: int
    critic_hidden_dim: int
    continuous_action_dim: int
    action_low: List[float]
    action_high: List[float]
    noise_schedule: NoiseScheduleConfig
    use_augmentation: bool = True
    augmentation_pad: int = 4


@dataclass
class DrQLossConfig:
    backup_every: int  # rollout length per env_loop.send() call -- same role as ActorCriticLossConfig's
    n_step: int
    gamma: float
    target_tau: float  # soft target-critic update rate
    noise_clip: float  # target-policy-smoothing clip, may differ from the exploration clip


class RandomShiftsAug(nn.Module):
    """DrQ-v2's random-shift augmentation: reflect-pad by `pad` then take a random pad-sized
    crop back to the original size, one independent shift per batch element. Applied to the
    frame-stacked observation before the encoder. Kept enabled by default -- augmenting
    world-model-imagined frames the same way DrQ-v2 augments real camera frames is the
    intended initial design; disabling it is a later ablation, not a default."""

    def __init__(self, pad: int) -> None:
        super().__init__()
        self.pad = pad

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        assert h == w
        padding = (self.pad,) * 4
        x = F.pad(x, padding, mode="replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[: h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2).unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift = shift * (2.0 / (h + 2 * self.pad))
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class DrQEncoder(nn.Module):
    """Same building blocks as ActorCriticEncoder (blocks.Conv3x3/SmallResBlock/MaxPool2d), just
    with `frame_stack * img_channels` input channels instead of `img_channels`."""

    def __init__(self, cfg: DrQActorCriticConfig) -> None:
        super().__init__()
        assert len(cfg.encoder_channels) == len(cfg.encoder_down)
        in_channels = cfg.frame_stack * cfg.img_channels
        layers = [Conv3x3(in_channels, cfg.encoder_channels[0])]
        for i in range(len(cfg.encoder_channels)):
            layers.append(SmallResBlock(cfg.encoder_channels[max(0, i - 1)], cfg.encoder_channels[i]))
            if cfg.encoder_down[i]:
                layers.append(nn.MaxPool2d(2))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x).flatten(start_dim=1)


class DrQActor(nn.Module):
    def __init__(self, cfg: DrQActorCriticConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.actor_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.actor_hidden_dim, cfg.actor_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.actor_hidden_dim, cfg.continuous_action_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Returns `mu`, already bounded to [-1, 1] via tanh -- see module docstring point 2:
        the actor produces an already-bounded mean, exploration noise is added AROUND it by the
        caller (sample_action / the target-policy-smoothing step in forward()), not baked in
        here."""
        return torch.tanh(self.net(features))


class DrQQHead(nn.Module):
    def __init__(self, cfg: DrQActorCriticConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.feature_dim + cfg.continuous_action_dim, cfg.critic_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.critic_hidden_dim, cfg.critic_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.critic_hidden_dim, 1),
        )

    def forward(self, features: Tensor, action: Tensor) -> Tensor:
        return self.net(torch.cat([features, action], dim=-1)).squeeze(-1)


class DrQCritic(nn.Module):
    """Twin Q-functions, Q1 and Q2, each its own DrQQHead -- no shared parameters between the
    two, matching DrQ-v2/TD3's overestimation-bias mitigation (conservative target estimation
    via min(Q1, Q2), see DrQActorCritic.forward)."""

    def __init__(self, cfg: DrQActorCriticConfig) -> None:
        super().__init__()
        self.q1 = DrQQHead(cfg)
        self.q2 = DrQQHead(cfg)

    def forward(self, features: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        return self.q1(features, action), self.q2(features, action)


def _soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p_target, p_online in zip(target.parameters(), online.parameters()):
            p_target.lerp_(p_online, tau)


class DrQActorCritic(nn.Module):
    def __init__(self, cfg: DrQActorCriticConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.frame_stack = cfg.frame_stack
        self.img_channels = cfg.img_channels
        self.img_size = cfg.img_size

        # Required by env_loop.py's zero-init line (`torch.zeros(env.num_envs, model.lstm_dim,
        # ...)`) -- see module docstring on why this repurposes the hx/cx slot for a frame stack
        # instead of touching env_loop.py. Kept under this exact name for that reason alone.
        self.lstm_dim = cfg.frame_stack * cfg.img_channels * cfg.img_size * cfg.img_size

        self.encoder = DrQEncoder(cfg)
        with torch.no_grad():
            dummy = torch.zeros(1, cfg.frame_stack * cfg.img_channels, cfg.img_size, cfg.img_size)
            actual_feature_dim = self.encoder(dummy).shape[1]
        assert actual_feature_dim == cfg.feature_dim, (
            f"DrQActorCriticConfig.feature_dim={cfg.feature_dim} does not match the encoder's "
            f"actual flattened output size {actual_feature_dim} for frame_stack={cfg.frame_stack}, "
            f"img_channels={cfg.img_channels}, img_size={cfg.img_size}, "
            f"encoder_channels={cfg.encoder_channels}, encoder_down={cfg.encoder_down} -- fix "
            f"feature_dim in config rather than letting a shape mismatch surface later inside "
            f"DrQActor/DrQQHead's first Linear layer."
        )

        self.actor = DrQActor(cfg)
        self.critic = DrQCritic(cfg)
        self.target_critic = DrQCritic(cfg)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        self.aug = RandomShiftsAug(cfg.augmentation_pad) if cfg.use_augmentation else nn.Identity()

        self.register_buffer("action_low", torch.tensor(cfg.action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(cfg.action_high, dtype=torch.float32))

        # Exploration-noise generator + schedule step counter: see DrQExplorationState's
        # docstring for why this is checkpointed externally (by Trainer's ResumeFidelityState)
        # rather than through nn.Module's own state_dict. The generator itself is
        # component-isolated (utils.derive_torch_generator with a dedicated component id, see
        # data.batch_sampler.COMPONENT_SEED_ID), set once setup_training runs -- never drawing
        # from torch's shared global CPU/CUDA generator.
        self.exploration_state = DrQExplorationState()

        self.env_loop = None
        self.rollout_hx_cx_state = RolloutHxCxState()
        self.loss_cfg = None
        self.intrinsic_reward_fn = None
        self._rl_env = None

        # One-shot gate for cold-start stack seeding -- see predict_act_value and
        # _seed_cold_stack. Consumed (set False) by the very first predict_act_value call in
        # this process, REGARDLESS of which call that happens to be, so it can never
        # accidentally re-fire later and disturb the (already self-correcting, see
        # _seed_cold_stack's docstring) dead-env burn-in mechanism.
        self._cold_start_pending = True
        # Cache of the cold-seeded t=0 stack, for forward()'s _reconstruct_stacks to reuse
        # exactly rather than re-deriving it from obs_buffer (which has moved on by the time
        # forward() runs) -- see _reconstruct_stacks' docstring.
        self._cold_seeded_stack_flat: Optional[Tensor] = None

    @property
    def device(self) -> torch.device:
        return self.action_low.device

    def setup_training(
        self,
        rl_env: Union[TorchEnv, WorldModelEnv],
        loss_cfg: DrQLossConfig,
        noise_generator: torch.Generator,
    ) -> None:
        assert self.env_loop is None and self.loss_cfg is None
        assert isinstance(rl_env, WorldModelEnv), (
            "DrQActorCritic must only ever be trained against a WorldModelEnv (imagined "
            "rollouts) -- see the module docstring and the project's exploration-boundary "
            "requirement: real transitions may drive real-env DATA COLLECTION via a separate "
            "env_loop (see coroutines.collector.make_collector), but must never reach this "
            "training env_loop."
        )
        # DrQActorCritic requires frame_stack <= the world model's num_steps_conditioning: the
        # dead-env burn-in mechanism in env_loop.py feeds exactly (num_steps_conditioning - 1)
        # real context frames before normal stepping resumes, which is what makes the dead-env
        # case self-correct to the right K-frame window without any special-casing (see
        # _seed_cold_stack's docstring for the full trace) -- this is a config-level invariant
        # (agent/drq.yaml's frame_stack vs agent/default.yaml's
        # denoiser.inner_model.num_steps_conditioning), not something checkable here without
        # WorldModelEnv exposing that value directly.
        self._rl_env = rl_env
        self.env_loop = make_env_loop(rl_env, self, hx_cx_state=self.rollout_hx_cx_state)
        self.loss_cfg = loss_cfg
        self.exploration_state.generator = noise_generator

    def set_intrinsic_reward_fn(self, fn) -> None:
        """Same hook/contract as ActorCritic.set_intrinsic_reward_fn -- substitutes the reward
        used for THIS actor-critic's own training signal (e.g. an LCG/ensemble-disagreement
        exploration score) without affecting RewEndModel's own training, which always learns to
        predict the true env reward regardless of what drives the actor. None (default) uses
        the world model's own predicted reward, unmodified -- this is what a downstream
        zero-shot TASK policy should use (never set this hook for that phase)."""
        self.intrinsic_reward_fn = fn

    def _shift_and_append(self, hx_flat: Tensor, obs: Tensor) -> Tensor:
        """hx_flat: (num_envs, frame_stack * C * H * W) flattened stack. obs: (num_envs, C, H,
        W) the single newest frame. Returns the new flattened stack with the oldest frame
        dropped and `obs` appended as the newest -- the shared implementation used both during
        rollout collection (predict_act_value) and loss-time reconstruction (forward)."""
        n = hx_flat.size(0)
        stack = hx_flat.view(n, self.frame_stack, self.img_channels, self.img_size, self.img_size)
        stack = torch.cat([stack[:, 1:], obs.unsqueeze(1)], dim=1)
        return stack.reshape(n, -1)

    def _flat_to_chw(self, stack_flat: Tensor) -> Tensor:
        n = stack_flat.size(0)
        return stack_flat.view(n, self.frame_stack * self.img_channels, self.img_size, self.img_size)

    def _seed_cold_stack(self, obs: Tensor) -> Tensor:
        """Builds an initial K-frame stack from real history instead of zero-padding, for the
        ONE case that genuinely has no other source of context: env_loop.py's very first
        predict_act_value call in a truly fresh (non-resumed) process, made right after
        env.reset() with hx still exactly zero-init. Prefers self._rl_env.obs_buffer's own
        conditioning window (WorldModelEnv always maintains num_steps_conditioning >= K real
        frames there, populated by generator_init's initial-condition draw, by the time
        reset() returns) -- falling back to repeating `obs` K times only if obs_buffer isn't
        available (e.g. this model were ever used with a plain TorchEnv real-env collector,
        which is architecturally intended -- see the module/setup_training docstrings -- but
        has no multi-frame buffer of its own).

        This is NOT used for the dead-env mid-rollout reset case (env_loop.py's `dead.any()`
        branch): that case is already self-correcting WITHOUT any special seeding. Trace: (1)
        WorldModelEnv.step() calls self.reset_dead(dead) -- which fully updates obs_buffer for
        those rows to the NEW episode's initial condition -- strictly BEFORE returning, so
        info["burnin_obs"] already reflects the new episode only, zero old-episode leakage; (2)
        env_loop.py's reset_gate zeroes hx (this model's stack) for dead rows; (3) the burn-in
        loop feeds exactly (num_steps_conditioning - 1) real frames via ordinary
        _shift_and_append calls; (4) the immediately-following normal step feeds one more (the
        row's obs_buffer[:, -1]) -- num_steps_conditioning frames fed in total, so as long as
        frame_stack <= num_steps_conditioning (see setup_training's docstring), the stack has
        converged to exactly obs_buffer[dead, -frame_stack:] by the time normal action
        selection resumes for those rows. No row's action is ever read mid-burn-in (env_loop.py
        discards predict_act_value's mu/val there, keeping only the updated hx/cx), so the
        transiently-incomplete intermediate stack never affects behavior. Special-casing this
        path too, on top of the above, risks the opposite bug: overwriting the burn-in
        loop's OWN progressive state mid-sequence and corrupting it."""
        rl_env = self._rl_env
        if rl_env is not None and hasattr(rl_env, "obs_buffer") and rl_env.obs_buffer.size(1) >= self.frame_stack:
            return rl_env.obs_buffer[:, -self.frame_stack :].clone()
        return obs.unsqueeze(1).repeat(1, self.frame_stack, 1, 1, 1)

    @torch.no_grad()
    def predict_act_value(self, obs: Tensor, hx_cx: Tuple[Tensor, Tensor]):
        """See env_loop.py's calling convention. `val` is an unused placeholder (DrQ bootstraps
        exclusively from the target critics in forward(), never from this or val_bootstrap --
        see module docstring). Always no_grad, regardless of ambient autograd context: this
        call exists purely to pick an action and advance the frame stack during rollout
        collection; forward() recomputes actor/critic outputs WITH gradient from the stored
        (s, a, r, s') sequence separately, so building a graph here would be pure waste."""
        hx, cx = hx_cx
        if self._cold_start_pending:
            self._cold_start_pending = False
            if not self.rollout_hx_cx_state.initialized:
                # Truly fresh process (not a resume with a valid restored stack, which would
                # have rollout_hx_cx_state.initialized=True and hx already correct) -- see
                # _seed_cold_stack's docstring.
                seeded = self._seed_cold_stack(obs)
                stack_flat = seeded.reshape(seeded.size(0), -1)
                self._cold_seeded_stack_flat = stack_flat.clone()
            else:
                stack_flat = self._shift_and_append(hx, obs)
        else:
            stack_flat = self._shift_and_append(hx, obs)
        features = self.encoder(self._flat_to_chw(stack_flat))
        mu = self.actor(features)  # already tanh-bounded, see DrQActor.forward
        val = torch.zeros(obs.size(0), device=obs.device)
        return mu, val, (stack_flat, cx)

    def _rescale(self, canonical_action: Tensor) -> Tensor:
        scale = 0.5 * (self.action_high - self.action_low)
        return self.action_low + (canonical_action + 1.0) * scale

    def _sample_noise(self, shape: Tuple[int, ...], std: float, clip: float, device: torch.device) -> Tensor:
        eps = torch.randn(shape, device=device, generator=self.exploration_state.generator) * std
        return eps.clamp(-clip, clip)

    def sample_action(self, dist_params: Tensor, deterministic: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        """dist_params here is `mu` (already tanh-bounded, from predict_act_value). Deterministic
        eval returns `mu` rescaled with no noise (module docstring point 2). Training-time
        sampling adds truncated-Gaussian exploration noise around mu in the bounded [-1, 1]
        space, clamps back into [-1, 1], THEN rescales -- not the unbounded
        `tanh(mean + std*eps)` DrQ-v2 explicitly avoids. `aux` (the `z` slot in the original
        ActorCritic interface) is unused by DrQ's loss and returned as None."""
        mu = dist_params
        if deterministic:
            canonical = mu
        else:
            std = _noise_std_at(self.cfg.noise_schedule, self.exploration_state.schedule_step)
            eps = self._sample_noise(mu.shape, std, self.cfg.noise_schedule.clip, mu.device)
            canonical = (mu + eps).clamp(-1.0, 1.0)
            self.exploration_state.schedule_step += 1
        action = self._rescale(canonical).detach()
        return action, None

    def _reconstruct_stacks(self, pre_stack_flat: Optional[Tensor], all_obs: Tensor) -> Tensor:
        """all_obs: (num_envs, T, C, H, W), the single-frame-per-step sequence env_loop.py
        returns. Rebuilds the EXACT K-frame stack predict_act_value saw at each step t, so the
        critic loss pairs each action with the observation that actually produced it.

        Normal case (pre_stack_flat is not None, i.e. rollout_hx_cx_state was already
        initialized going into this env_loop.send() call): shift-and-append all_obs onto
        pre_stack_flat, step by step -- exactly what predict_act_value did.

        Cold-start case (pre_stack_flat is None, the very first rollout call in this process):
        predict_act_value's t=0 call did NOT shift-and-append -- it called _seed_cold_stack,
        reading self._rl_env.obs_buffer directly. That buffer has since moved on (mutated by
        every env.step() call during THIS rollout), so it can't be re-read here to recover
        what it held at t=0 -- instead this reuses self._cold_seeded_stack_flat, the exact
        value predict_act_value cached at the time. t=1 onward then proceeds by ordinary
        shift-and-append from that cached starting point, same as the normal case."""
        n, t, c, h, w = all_obs.shape
        if pre_stack_flat is not None:
            stack = pre_stack_flat.view(n, self.frame_stack, c, h, w)
            stacks = []
            for step in range(t):
                stack = torch.cat([stack[:, 1:], all_obs[:, step].unsqueeze(1)], dim=1)
                stacks.append(stack)
            return torch.stack(stacks, dim=1)  # (n, t, frame_stack, c, h, w)

        assert self._cold_seeded_stack_flat is not None, (
            "forward() has no pre_stack_flat (rollout_hx_cx_state was never initialized) but "
            "also no cached cold-seeded stack -- predict_act_value must run, via "
            "env_loop.send(), before _reconstruct_stacks is called"
        )
        stack = self._cold_seeded_stack_flat.view(n, self.frame_stack, c, h, w)
        stacks = [stack.clone()]
        for step in range(1, t):
            stack = torch.cat([stack[:, 1:], all_obs[:, step].unsqueeze(1)], dim=1)
            stacks.append(stack)
        return torch.stack(stacks, dim=1)

    def _n_step_returns(
        self, rew: Tensor, end: Tensor, trunc: Tensor, gamma: float, n: int, usable: int
    ) -> Tuple[Tensor, Tensor]:
        """rew/end/trunc: (num_envs, T). Returns (n_step_return, not_done_mask), both (num_envs,
        usable) -- not_done_mask is 0 wherever the episode ended/truncated at or before t+n-1
        (so the target critic's bootstrap term at s_(t+n) is zeroed out for those rows), 1
        otherwise. Pure reward accumulation with early stopping on end/trunc, no dependence on
        any value head."""
        num_envs = rew.size(0)
        returns = torch.zeros(num_envs, usable, device=rew.device, dtype=rew.dtype)
        not_done = torch.ones(num_envs, usable, device=rew.device, dtype=rew.dtype)
        alive = torch.ones(num_envs, usable, device=rew.device, dtype=torch.bool)
        for k in range(n):
            returns = returns + alive.to(rew.dtype) * (gamma ** k) * rew[:, k : k + usable]
            dead_this_step = torch.logical_or(end[:, k : k + usable].bool(), trunc[:, k : k + usable].bool())
            alive = alive & (~dead_this_step)
        not_done = alive.to(rew.dtype)
        return returns, not_done

    def forward(self) -> LossAndLogs:
        """Trains exclusively on the imagined rollout `self.env_loop.send()` just produced --
        NOT a persistent replay buffer (see module docstring: direct fresh imagined batches are
        the approved initial design). Every observation/action/reward here originates from
        WorldModelEnv.step() (see setup_training's assertion); real environment transitions
        never reach this method -- they only ever pass through coroutines.collector's SEPARATE
        env_loop instance, which trains nothing.
        """
        c = self.loss_cfg
        pre_stack_flat = self.rollout_hx_cx_state.hx.clone() if self.rollout_hx_cx_state.initialized else None

        all_obs, act, rew, end, trunc, _dist_params, _val, _val_bootstrap, _z, infos = self.env_loop.send(c.backup_every)

        if self.intrinsic_reward_fn is not None:
            rew = self.intrinsic_reward_fn(infos, rew)

        stacks = self._reconstruct_stacks(pre_stack_flat, all_obs)  # (n, T, K, C, H, W)
        num_envs, T = all_obs.size(0), all_obs.size(1)
        n = c.n_step
        usable = T - n
        assert usable > 0, f"n_step={n} must be < backup_every={T} so s_(t+n) is available within the rollout"

        def chw(x):  # (n, K, C, H, W) -> (n, K*C, H, W)
            return x.reshape(x.size(0), -1, self.img_size, self.img_size)

        s_t = torch.stack([chw(stacks[:, t]) for t in range(usable)], dim=1)  # (n, usable, K*C,H,W)
        s_tpn = torch.stack([chw(stacks[:, t + n]) for t in range(usable)], dim=1)
        a_t = act[:, :usable]

        returns, not_done = self._n_step_returns(rew, end, trunc, c.gamma, n, usable)

        s_t_flat = s_t.reshape(num_envs * usable, *s_t.shape[2:])
        s_tpn_flat = s_tpn.reshape(num_envs * usable, *s_tpn.shape[2:])
        a_t_flat = a_t.reshape(num_envs * usable, -1)

        s_t_aug = self.aug(s_t_flat)
        s_tpn_aug = self.aug(s_tpn_flat)

        features_t = self.encoder(s_t_aug)  # grad-tracked -- this is what trains the encoder
        q1, q2 = self.critic(features_t, a_t_flat)

        with torch.no_grad():
            features_tpn = self.encoder(s_tpn_aug)
            mu_tpn = self.actor(features_tpn)
            noise_std = _noise_std_at(self.cfg.noise_schedule, self.exploration_state.schedule_step)
            eps = self._sample_noise(mu_tpn.shape, noise_std, c.noise_clip, mu_tpn.device)
            a_tpn = (mu_tpn + eps).clamp(-1.0, 1.0)
            tq1, tq2 = self.target_critic(features_tpn, a_tpn)
            target_q = torch.min(tq1, tq2)
            td_target = returns.reshape(-1) + (c.gamma ** n) * not_done.reshape(-1) * target_q

        loss_critic = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        features_t_detached = features_t.detach()  # actor never trains the encoder, see module docstring
        mu_online = self.actor(features_t_detached)
        q1_pi, q2_pi = self.critic(features_t_detached, mu_online)
        loss_actor = -torch.min(q1_pi, q2_pi).mean()

        loss = loss_critic + loss_actor

        _soft_update(self.target_critic, self.critic, c.target_tau)

        metrics = {
            "loss_critic": loss_critic.detach(),
            "loss_actor": loss_actor.detach(),
            "loss_total": loss.detach(),
            "q1_mean": q1.detach().mean(),
            "q2_mean": q2.detach().mean(),
            "target_q_mean": target_q.detach().mean(),
            "noise_std": torch.tensor(noise_std),
        }
        return loss, metrics
