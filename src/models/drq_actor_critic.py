"""DrQ-v2-style continuous-action actor-critic, trained exclusively on trajectories imagined
inside a DIAMOND WorldModelEnv (see envs.world_model_env.WorldModelEnv). Deliberately named
"DrQ-v2-style", not "DrQ-v2": several choices deviate from the original paper to fit DIAMOND's
imagined-rollout architecture without a rewrite of the surrounding pipeline --

  - No persistent replay buffer: each imagined rollout (one env_loop.send() call) IS the
    training minibatch for one collect_rollout() + critic_update() + actor_update() cycle.
  - No target encoder: only the twin Q-heads (and critic's own projection trunk, see below) are
    soft-updated; the (online) encoder is shared, trained by the critic loss, and used
    (no_grad, at its POST-critic-update weights) to featurize the bootstrap observation for the
    actor's own update too.
  - Exploration noise std is an externally scheduled scalar (see NoiseScheduleConfig), not
    learned, and the schedule advances exactly once per collect_rollout() call -- NOT inside
    sample_action(), which is also called during real-env collection and evaluation, neither
    of which should perturb subsequent training exploration (see sample_action's docstring).

Optimization is intentionally NOT a single `loss = critic_loss + actor_loss` exposed as one
nn.Module.forward() for a single external optimizer -- the actor objective -Q(s, actor(s))
produces gradients through the critic's own parameters unless explicitly guarded, and the
target-critic soft update must happen strictly after the critic's OWN optimizer step (not
before, which would silently make targets lag the online critic by one extra step). Call
sequence: collect_rollout() once, then critic_update(opt_critic) (which also performs the
target soft-update internally, after opt_critic.step()), then actor_update(opt_actor) (which
explicitly freezes the critic's requires_grad for the duration of its own backward pass, so no
stray gradient ever reaches critic parameters even though the actor loss reads through them).
Trainer/Hydra integration (constructing opt_critic/opt_actor, calling this sequence from
train_component, adding the freeze-world-model downstream-policy mode) is a separate,
not-yet-implemented stage -- this module is usable and fully tested standalone first.

Separate actor/critic projection trunks: the CNN encoder is shared (trained only by the critic
loss), but the actor and critic each own an independent `Linear -> LayerNorm -> Tanh`
projection trunk on top of the shared raw conv features -- DrQEncoder itself returns UNPROJECTED
conv features; DrQActor/DrQCritic each apply their own trunk internally. This means
opt_critic's parameter group (encoder + critic, i.e. critic's trunk + Q1 + Q2) and opt_actor's
(actor, i.e. actor's trunk + its MLP) own fully disjoint parameter sets below the shared
encoder -- there is no post-conv parameter both optimizers ever touch.

Interface with coroutines.env_loop.make_env_loop: this module implements the same
predict_act_value(obs, hx_cx) / sample_action(dist_params, deterministic) contract ActorCritic
does, so env_loop.py's core loop needs no DrQ-specific changes. Two small, additive,
backward-compatible extensions WERE made to env_loop.py to support this module correctly (see
env_loop.py's own comments at each site) -- ActorCritic and coroutines.collector are
unaffected:
  1. An optional `model.initial_hx_cx(num_envs) -> (hx, cx)` hook, used instead of the
     hardcoded `torch.zeros(...)` zero-init when the model defines it. DrQActorCritic uses this
     to seed a NaN sentinel (see initial_hx_cx's docstring) rather than zeros, which
     predict_act_value needs to reliably distinguish "this loop's state has never been
     seeded" from a mid-rollout reset_gate zero-out (both of which would otherwise look like
     plain zeros) -- see predict_act_value's docstring for why that distinction matters and
     what breaks without it.
  2. env_loop.py's yield tuple now additionally returns `all_hx`: the exact per-step model
     state that produced each step's action (captured immediately after predict_act_value
     returns, BEFORE any dead-env reset_gate/burn-in touches hx for the FOLLOWING step -- an
     earlier version of this same feature captured it too late, after that mutation, and was
     caught by this module's own mid-rollout-reset test, not by inspection). Frame-stack
     loss-time reconstruction from `all_obs` alone cannot correctly reproduce what a
     mid-rollout per-env reset actually did (the dead-env burn-in loop consumes MULTIPLE
     context frames not individually present in `all_obs`), so collect_rollout() below uses
     `all_hx` directly instead of re-deriving it.

Frame stacking: WorldModelEnv/env_loop only ever hand the model a SINGLE most-recent frame per
call (env.obs_buffer[:, -1]) -- the original ActorCritic's only source of temporal context is
its own LSTM hx/cx, not any multi-frame buffer (traced directly, not assumed). A feedforward DrQ
actor needs an explicit short frame stack, so this module repurposes env_loop's existing
(hx, cx) threading: `hx` carries the flattened (num_envs, frame_stack * img_channels * img_size
* img_size) stack, `cx` is an unused placeholder tensor. env_loop.py's hx/cx handling (.detach(),
boolean indexing, multiplying by a reset_gate, the zero-init line) is generic tensor
manipulation with no LSTM-specific assumption, so this achieves the frame stack with no other
env_loop.py changes, and the stack inherits env_loop's existing cross-rollout-call persistence
AND RolloutHxCxState's existing checkpoint/resume machinery for free (it round-trips as "hx"
without any new checkpointed state).

Cold-start seeding differs between an imagined WorldModelEnv (which always has an obs_buffer
conditioning window to seed from, and whose dead-env burn-in loop already self-corrects a
mid-rollout reset with zero special-casing -- see predict_act_value's docstring) and a real
env (no obs_buffer, no burn-in `info` key at all): for the latter, EVERY episode reset --  not
just the process's first-ever call -- must reseed the stack to K repeats of the new episode's
first observation, or it silently degrades to zero-padding after the first real episode ends.
See predict_act_value's `has_obs_buffer` branch.

Per-loop isolation: a single DrQActorCritic instance's trainable parameters are shared across
MULTIPLE independent env_loop instances -- the imagined-training loop, plus real-env train/test
collectors that use this same actor to drive data collection (see the module's exploration-
boundary requirement: the SAME learned policy explores in both imagination and reality, but
real transitions never train it). Neither cold-start/frame-stack state NOR exploration-noise
RNG state may live on `self` (the shared model) for this reason -- see DrQPolicyBinding, which
gives each loop its own env reference AND its own noise generator, and
predict_act_value/sample_action's `env`/`generator` parameters, which are always threaded
through from the calling binding rather than read from any actor-owned default (the one
exception, by design: the imagined-training loop's OWN env_loop, built in setup_training, which
legitimately owns `self.exploration_state.generator` since collect_rollout/critic_update are
themselves only ever invoked for that one loop).
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
    afterward. Pure function of `step`; see DrQExplorationState for where the counter itself
    lives, and DrQActorCritic.collect_rollout for the ONE place it advances. This single
    schedule value (a function of overall training progress) is shared by every loop's noise
    draw -- what differs per loop is the RANDOM STREAM used to sample it (see DrQGeneratorState/
    DrQPolicyBinding), not the std value itself."""
    frac = min(1.0, step / max(1, cfg.decay_steps))
    return cfg.std_start + frac * (cfg.std_end - cfg.std_start)


class DrQExplorationState:
    """Auxiliary, non-nn.Module state for the IMAGINED-TRAINING loop's exploration-noise
    sampling: its own component-isolated torch.Generator (see utils.derive_torch_generator),
    and the shared noise-schedule step counter (training progress -- read by every loop, but
    only ever advanced here, via collect_rollout). Checkpointed explicitly by Trainer's
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


class DrQGeneratorState:
    """A single independent, checkpointed torch.Generator with NO schedule of its own -- used
    for exploration-noise streams OTHER than the imagined-training loop's (see
    DrQExplorationState): real-env train collection and evaluation each get their own instance
    (DrQActorCritic.real_collection_exploration_state / .eval_exploration_state), so consuming
    noise in one loop can never perturb another's future draws. All read the SAME schedule
    value (_noise_std_at(..., self.exploration_state.schedule_step) -- training progress is a
    single global notion), but draw from their own independent stream."""

    def __init__(self) -> None:
        self.generator: Optional[torch.Generator] = None

    def state_dict(self) -> Dict[str, Any]:
        return {"generator_state": self.generator.get_state() if self.generator is not None else None}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        generator_state = state_dict.get("generator_state")
        if generator_state is not None and self.generator is not None:
            self.generator.set_state(generator_state)


@dataclass
class DrQActorCriticConfig:
    img_channels: int
    img_size: int
    frame_stack: int  # number of most-recent frames stacked along the channel dim, e.g. 3
    encoder_channels: List[int]
    encoder_down: List[int]
    feature_dim: int  # flattened RAW conv-encoder output size (asserted against the real encoder)
    projection_dim: int  # EACH of actor's/critic's own trunk output size, see module docstring
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
    frame-stacked observation before the encoder -- since all frame_stack*img_channels channels
    of one batch element go through ONE grid_sample call with ONE shift, every stacked frame
    receives the IDENTICAL spatial shift (grid_sample's sampling grid is per-batch-element, not
    per-channel) -- never independently shifted per frame, which would fabricate motion. Kept
    enabled by default -- augmenting world-model-imagined frames the same way DrQ-v2 augments
    real camera frames is the intended initial design; disabling it is a later ablation, not a
    default."""

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


class DrQProjectionTrunk(nn.Module):
    """DrQ-v2-style lightweight projection: Linear -> LayerNorm -> Tanh. Bounds the feature
    magnitude fed into an MLP head regardless of raw conv-feature scale. DrQActor and DrQCritic
    each own an independent instance (see module docstring point 4) -- never shared between
    them, so opt_critic's and opt_actor's parameter groups are fully disjoint below the shared
    conv encoder."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.Tanh())

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DrQEncoder(nn.Module):
    """Conv stack ONLY (same building blocks as ActorCriticEncoder: Conv3x3/SmallResBlock/
    MaxPool2d), with `frame_stack * img_channels` input channels instead of `img_channels`.
    Returns RAW flattened conv features -- no projection trunk here; DrQActor and DrQCritic
    each apply their own (see DrQProjectionTrunk / module docstring point 4)."""

    def __init__(self, cfg: DrQActorCriticConfig) -> None:
        super().__init__()
        assert len(cfg.encoder_channels) == len(cfg.encoder_down)
        in_channels = cfg.frame_stack * cfg.img_channels
        layers = [Conv3x3(in_channels, cfg.encoder_channels[0])]
        for i in range(len(cfg.encoder_channels)):
            layers.append(SmallResBlock(cfg.encoder_channels[max(0, i - 1)], cfg.encoder_channels[i]))
            if cfg.encoder_down[i]:
                layers.append(nn.MaxPool2d(2))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x).flatten(start_dim=1)


class DrQActor(nn.Module):
    def __init__(self, cfg: DrQActorCriticConfig) -> None:
        super().__init__()
        self.trunk = DrQProjectionTrunk(cfg.feature_dim, cfg.projection_dim)
        self.net = nn.Sequential(
            nn.Linear(cfg.projection_dim, cfg.actor_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.actor_hidden_dim, cfg.actor_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.actor_hidden_dim, cfg.continuous_action_dim),
        )

    def forward(self, conv_features: Tensor) -> Tensor:
        """Returns `mu`, already bounded to [-1, 1] via tanh -- see the module docstring point
        2: the actor produces an already-bounded mean, exploration noise is added AROUND it by
        the caller (sample_action / the target-policy-smoothing step in critic_update), not
        baked in here. `conv_features` are the SHARED encoder's raw (unprojected) output --
        this method applies the actor's OWN trunk internally."""
        return torch.tanh(self.net(self.trunk(conv_features)))


class DrQQHead(nn.Module):
    def __init__(self, cfg: DrQActorCriticConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.projection_dim + cfg.continuous_action_dim, cfg.critic_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.critic_hidden_dim, cfg.critic_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.critic_hidden_dim, 1),
        )

    def forward(self, projected_features: Tensor, action: Tensor) -> Tensor:
        return self.net(torch.cat([projected_features, action], dim=-1)).squeeze(-1)


class DrQCritic(nn.Module):
    """Twin Q-functions, Q1 and Q2, sharing ONE projection trunk (this critic's own, never the
    actor's -- see DrQProjectionTrunk) but otherwise independent DrQQHeads, matching DrQ-v2/
    TD3's overestimation-bias mitigation (conservative target estimation via min(Q1, Q2))."""

    def __init__(self, cfg: DrQActorCriticConfig) -> None:
        super().__init__()
        self.trunk = DrQProjectionTrunk(cfg.feature_dim, cfg.projection_dim)
        self.q1 = DrQQHead(cfg)
        self.q2 = DrQQHead(cfg)

    def forward(self, conv_features: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        projected = self.trunk(conv_features)
        return self.q1(projected, action), self.q2(projected, action)


def _soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p_target, p_online in zip(target.parameters(), online.parameters()):
            p_target.lerp_(p_online, tau)


class DrQPolicyBinding:
    """Binds a shared DrQActorCritic to ONE specific env AND ONE specific exploration-noise
    generator, for ONE specific env_loop instance -- so cold-start seeding and RNG state are
    both correctly scoped per loop instead of living on the shared actor -- see the module
    docstring's "per-loop isolation" section. Exposes exactly the surface
    coroutines.env_loop.make_env_loop / coroutines.collector.make_collector need
    (predict_act_value, sample_action, initial_hx_cx, lstm_dim, device), delegating all actual
    computation (and all trainable parameters) to the wrapped model -- multiple bindings over
    the SAME DrQActorCritic share every weight, but never share env references, cold-start
    state, or noise-generator state: none of it lives on either the binding or the model,
    predict_act_value/sample_action derive it fresh from their own arguments every call."""

    def __init__(
        self,
        model: "DrQActorCritic",
        env: Optional[Union[TorchEnv, WorldModelEnv]] = None,
        noise_generator: Optional[torch.Generator] = None,
    ) -> None:
        self.model = model
        self.env = env
        self.noise_generator = noise_generator

    @property
    def lstm_dim(self) -> int:
        return self.model.lstm_dim

    @property
    def device(self) -> torch.device:
        return self.model.device

    def initial_hx_cx(self, num_envs: int) -> Tuple[Tensor, Tensor]:
        return self.model.initial_hx_cx(num_envs)

    def predict_act_value(self, obs: Tensor, hx_cx: Tuple[Tensor, Tensor]):
        return self.model.predict_act_value(obs, hx_cx, env=self.env)

    def sample_action(self, dist_params: Tensor, deterministic: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        return self.model.sample_action(dist_params, deterministic=deterministic, generator=self.noise_generator)


class DrQActorCritic(nn.Module):
    def __init__(self, cfg: DrQActorCriticConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.frame_stack = cfg.frame_stack
        self.img_channels = cfg.img_channels
        self.img_size = cfg.img_size

        # Required by env_loop.py's zero-init line (`torch.zeros(env.num_envs, model.lstm_dim,
        # ...)`, only reached when initial_hx_cx is absent -- ActorCritic's own case) and by
        # OUR OWN initial_hx_cx below. Kept under this name deliberately for zero env_loop.py
        # diff on that specific line even though this model has no LSTM.
        self.lstm_dim = cfg.frame_stack * cfg.img_channels * cfg.img_size * cfg.img_size

        self.encoder = DrQEncoder(cfg)
        with torch.no_grad():
            dummy = torch.zeros(1, cfg.frame_stack * cfg.img_channels, cfg.img_size, cfg.img_size)
            actual_feature_dim = self.encoder(dummy).shape[1]
        assert actual_feature_dim == cfg.feature_dim, (
            f"DrQActorCriticConfig.feature_dim={cfg.feature_dim} does not match the encoder's "
            f"actual flattened conv output size {actual_feature_dim} for frame_stack="
            f"{cfg.frame_stack}, img_channels={cfg.img_channels}, img_size={cfg.img_size}, "
            f"encoder_channels={cfg.encoder_channels}, encoder_down={cfg.encoder_down} -- fix "
            f"feature_dim in config rather than letting a shape mismatch surface later inside "
            f"the actor's/critic's own trunk's first Linear layer."
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

        # Three independent exploration-noise streams (module docstring, RNG-isolation
        # section): imagined-training owns exploration_state (generator + the shared
        # training-progress schedule_step); real-env train collection and evaluation each get
        # their own DrQGeneratorState (no schedule of their own -- they read
        # exploration_state.schedule_step for the std value, see _noise_std_at). All are
        # constructed with generator=None here; a caller (setup_training for the first,
        # make_collector_binding's caller for the other two) supplies an actual
        # utils.derive_torch_generator(..., component_id, device) generator before any
        # stochastic sampling happens on that stream.
        self.exploration_state = DrQExplorationState()
        self.real_collection_exploration_state = DrQGeneratorState()
        self.eval_exploration_state = DrQGeneratorState()

        self.env_loop = None
        self.rollout_hx_cx_state = RolloutHxCxState()
        self.loss_cfg = None
        self.intrinsic_reward_fn = None
        self._cached_rollout: Optional[Dict[str, Any]] = None

    @property
    def device(self) -> torch.device:
        return self.action_low.device

    def initial_hx_cx(self, num_envs: int) -> Tuple[Tensor, Tensor]:
        """NaN sentinel for "this loop's frame stack has never been seeded" -- deliberately NOT
        zeros, which env_loop.py's dead-env reset_gate ALSO produces mid-rollout
        (`hx = hx * reset_gate`). Using the same value for both would make predict_act_value
        unable to tell "truly cold, needs real seeding from history" apart from "mid-rollout
        reset, already being correctly rebuilt by the burn-in loop" for a WorldModelEnv -- see
        predict_act_value's docstring for exactly why conflating them corrupts the burn-in
        loop's own progressive state there. NaN survives the zero-init/reset_gate distinction
        cleanly because reset_gate only ever multiplies an ALREADY-real (non-NaN) hx -- by the
        time any env can die, predict_act_value has already replaced its NaN with a real seeded
        value in that same step (predict_act_value always runs before env.step() can trigger a
        death, see env_loop.py's loop order), so NaN never has a chance to propagate through
        `NaN * 0 = NaN`."""
        hx = torch.full((num_envs, self.lstm_dim), float("nan"), device=self.device)
        cx = torch.zeros(num_envs, 1, device=self.device)
        return hx, cx

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
            "env_loop/DrQPolicyBinding (see coroutines.collector.make_collector and "
            "make_collector_binding below), but must never reach this training env_loop."
        )
        self.exploration_state.generator = noise_generator
        self.env_loop = make_env_loop(
            rl_env,
            DrQPolicyBinding(self, env=rl_env, noise_generator=noise_generator),
            hx_cx_state=self.rollout_hx_cx_state,
        )
        self.loss_cfg = loss_cfg

    def make_collector_binding(
        self, env: Optional[Union[TorchEnv, WorldModelEnv]] = None, noise_generator: Optional[torch.Generator] = None
    ) -> DrQPolicyBinding:
        """Constructs an INDEPENDENT DrQPolicyBinding for a real-env collector (train or test),
        so its cold-start/frame-stack state AND its noise stream are scoped to that collector's
        own env_loop instance -- never shared with the imagined-training loop or with another
        collector (a separate call for the train collector and the test collector each gets its
        own binding, hence its own frame-stack initialization and its own RNG stream -- pass
        `self.real_collection_exploration_state.generator` / `self.eval_exploration_state.
        generator` respectively, or None for a purely deterministic caller, e.g. evaluation
        that always passes deterministic=True and so never touches `noise_generator` at all).
        `env` is typically a real TorchEnv (no obs_buffer), so cold-start there uses the
        repeat-current-observation path on every episode reset, not just the first -- see
        predict_act_value's docstring."""
        return DrQPolicyBinding(self, env=env, noise_generator=noise_generator)

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
        dropped and `obs` appended as the newest."""
        n = hx_flat.size(0)
        stack = hx_flat.view(n, self.frame_stack, self.img_channels, self.img_size, self.img_size)
        stack = torch.cat([stack[:, 1:], obs.unsqueeze(1)], dim=1)
        return stack.reshape(n, -1)

    def _flat_to_chw(self, stack_flat: Tensor) -> Tensor:
        n = stack_flat.size(0)
        return stack_flat.view(n, self.frame_stack * self.img_channels, self.img_size, self.img_size)

    def _seed_cold_stack(self, obs: Tensor, env: Optional[Union[TorchEnv, WorldModelEnv]]) -> Tensor:
        """Builds an initial K-frame stack from real history instead of zero-padding, for a
        row detected as cold (see predict_act_value). Prefers `env.obs_buffer`'s own
        conditioning window (WorldModelEnv always maintains num_steps_conditioning >= K real
        frames there by the time reset()/step() return) when `env` exposes one, falling back to
        repeating `obs` K times otherwise (a plain TorchEnv real-env collector, which has no
        multi-frame buffer of its own -- `[o0, o0, o0]` for K=3, per the reviewed plan)."""
        if env is not None and hasattr(env, "obs_buffer") and env.obs_buffer.size(1) >= self.frame_stack:
            return env.obs_buffer[:, -self.frame_stack :].clone()
        return obs.unsqueeze(1).repeat(1, self.frame_stack, 1, 1, 1)

    @torch.no_grad()
    def predict_act_value(
        self, obs: Tensor, hx_cx: Tuple[Tensor, Tensor], env: Optional[Union[TorchEnv, WorldModelEnv]] = None
    ):
        """See env_loop.py's calling convention (via DrQPolicyBinding, which supplies `env`).
        `val` is an unused placeholder (DrQ bootstraps exclusively from the target critics in
        critic_update, never from this or val_bootstrap -- see module docstring). Always
        no_grad, regardless of ambient autograd context: this call exists purely to pick an
        action and advance the frame stack during rollout collection; critic_update/
        actor_update recompute actor/critic outputs WITH gradient from the cached rollout
        separately, so building a graph here would be pure waste.

        Cold-start detection is PER-ROW and driven entirely by `hx`/`env`, never by state
        stored on `self` -- this is what makes the method safe to share across multiple
        independent loops via DrQPolicyBinding.

        Two DIFFERENT rules, selected by whether `env` exposes an obs_buffer:

        WorldModelEnv (has obs_buffer): a row is cold exactly when `hx` contains NaN. This
        happens in exactly one place -- env_loop.py's outer zero-init, via initial_hx_cx,
        before this loop's very first call ever (a true fresh start, not a resume). It does
        NOT happen during a mid-rollout dead-env reset: env_loop.py's reset_gate zeroes hx with
        ordinary 0.0 (`hx = hx * reset_gate`), which is NOT NaN, so this method takes the
        ordinary shift-and-append path there and lets the existing dead-env burn-in loop
        progressively rebuild the stack across (num_steps_conditioning - 1) real context
        frames -- exactly reproducing obs_buffer[dead, -frame_stack:] by the time normal
        stepping resumes, PROVIDED frame_stack <= num_steps_conditioning (a config-level
        invariant, see setup_training's docstring). Treating a reset_gate zero-out as cold here
        (an earlier version of this method did) would short-circuit that progressive rebuild
        with an immediately-correct answer that the REMAINING burn-in iterations then corrupt
        by continuing to feed already-incorporated frames on top of it.

        Real env / no obs_buffer: there is no burn-in mechanism at all (info never carries
        "burnin_obs" -- that key is WorldModelEnv-specific), so a reset_gate zero-out is NEVER
        naturally corrected by anything else. Without special-casing this, the stack would
        silently degrade to zero-padding `[0, 0, new_obs]` after every episode past the first.
        Since there is no burn-in loop here to corrupt, it's safe to ALSO treat an
        exactly-all-zero (not just NaN) `hx` as cold in this branch -- re-seeding immediately
        to `[new_obs, new_obs, new_obs]` via _seed_cold_stack's repeat-fallback, which then
        naturally becomes `[new_obs, new_obs, obs_1]`, `[new_obs, obs_1, obs_2]`, ... as
        ordinary steps follow, exactly as specified. This works independently per row, so a
        vectorized batch where only some envs reset this step is handled correctly without any
        extra bookkeeping."""
        hx, cx = hx_cx
        has_obs_buffer = env is not None and hasattr(env, "obs_buffer") and env.obs_buffer.size(1) >= self.frame_stack
        nan_cold = torch.isnan(hx).any(dim=1)
        if has_obs_buffer:
            cold_mask = nan_cold
        else:
            zero_cold = (~nan_cold) & torch.all(hx == 0, dim=1)
            cold_mask = nan_cold | zero_cold

        hx_clean = torch.nan_to_num(hx, nan=0.0)
        shifted = self._shift_and_append(hx_clean, obs)
        if cold_mask.any():
            seeded = self._seed_cold_stack(obs, env)
            seeded_flat = seeded.reshape(seeded.size(0), -1)
            stack_flat = torch.where(cold_mask.unsqueeze(1), seeded_flat, shifted)
        else:
            stack_flat = shifted

        features = self.encoder(self._flat_to_chw(stack_flat))
        mu = self.actor(features)  # already tanh-bounded, see DrQActor.forward
        val = torch.zeros(obs.size(0), device=obs.device)
        return mu, val, (stack_flat, cx)

    def _rescale(self, canonical_action: Tensor) -> Tensor:
        scale = 0.5 * (self.action_high - self.action_low)
        return self.action_low + (canonical_action + 1.0) * scale

    def _sample_noise(
        self, shape: Tuple[int, ...], std: float, clip: float, device: torch.device, generator: torch.Generator
    ) -> Tensor:
        eps = torch.randn(shape, device=device, generator=generator) * std
        return eps.clamp(-clip, clip)

    def sample_action(
        self, dist_params: Tensor, deterministic: bool = False, generator: Optional[torch.Generator] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """dist_params here is `mu` (already tanh-bounded, from predict_act_value).
        Deterministic eval returns `mu` rescaled with NO random draw at all -- consumes zero
        exploration RNG, from any stream, by construction (the `else` branch below, the only
        place any generator is touched, is simply never reached). Training-time (or real-
        collection-time) sampling adds truncated-Gaussian exploration noise around mu in the
        bounded [-1, 1] space, clamps back into [-1, 1], THEN rescales -- not the unbounded
        `tanh(mean + std*eps)` form.

        `generator` MUST be supplied by the caller (via DrQPolicyBinding, which threads through
        whichever stream belongs to ITS loop) whenever deterministic=False -- there is no
        actor-owned default here, precisely so that which stream gets consumed is always an
        explicit, per-loop choice rather than an accidentally-shared one (module docstring,
        RNG-isolation section).

        Reads the current exploration std but NEVER advances the schedule counter -- this
        method is called from every env_loop that uses this actor: imagined-rollout training
        collection, real-env train collection, real-env test collection, AND deterministic
        evaluation. If it advanced schedule_step itself, evaluation or real collection cadence
        would silently perturb subsequent TRAINING exploration. The schedule advances exactly
        once per collect_rollout() call instead -- see that method's docstring."""
        mu = dist_params
        if deterministic:
            canonical = mu
        else:
            assert generator is not None, (
                "sample_action(deterministic=False) requires an explicit generator -- pass one "
                "via DrQPolicyBinding (setup_training/make_collector_binding), never rely on an "
                "actor-owned default, see the module docstring's RNG-isolation section"
            )
            std = _noise_std_at(self.cfg.noise_schedule, self.exploration_state.schedule_step)
            eps = self._sample_noise(mu.shape, std, self.cfg.noise_schedule.clip, mu.device, generator)
            canonical = (mu + eps).clamp(-1.0, 1.0)
        action = self._rescale(canonical).detach()
        return action, None

    def _dense_final_obs(self, infos: List[dict], end: Tensor, trunc: Tensor, all_obs: Tensor) -> Tensor:
        """(num_envs, T, C, H, W): the TRUE final observation at each dead step, scattered into
        the dead rows only (live rows hold a harmless placeholder -- that step's ordinary
        all_obs -- never read for live rows, since the bootstrap-source selection in
        _compute_bootstrap_info only consults this at rows/steps it has already identified as
        dead)."""
        dense = all_obs.clone()
        T = end.size(1)
        for k in range(T):
            dead_k = torch.logical_or(end[:, k].bool(), trunc[:, k].bool())
            if dead_k.any() and "final_observation" in infos[k]:
                dense[dead_k, k] = infos[k]["final_observation"]
        return dense

    def _accumulate_rewards(self, rew: Tensor, end: Tensor, trunc: Tensor, gamma: float, n: int, usable: int) -> Tensor:
        """Reward accumulation does NOT need the end-vs-trunc distinction: the reward AT a dead
        step (whichever kind) is a real reward and always counts; rewards strictly AFTER a dead
        step never count, since that row's subsequent frames belong to a different (reset)
        episode regardless of why the reset happened. Only the bootstrap source/validity/
        DISCOUNT (_compute_bootstrap_info) depends on end vs trunc and on exactly how many
        transitions preceded the dead event."""
        num_envs = rew.size(0)
        returns = torch.zeros(num_envs, usable, device=rew.device, dtype=rew.dtype)
        alive = torch.ones(num_envs, usable, dtype=torch.bool, device=rew.device)
        for k in range(n):
            returns = returns + alive.to(rew.dtype) * (gamma ** k) * rew[:, k : k + usable]
            dead_this_step = torch.logical_or(end[:, k : k + usable].bool(), trunc[:, k : k + usable].bool())
            alive = alive & (~dead_this_step)
        return returns

    def _compute_bootstrap_info(
        self,
        all_hx: Tensor,
        all_obs: Tensor,
        end: Tensor,
        trunc: Tensor,
        infos: List[dict],
        gamma: float,
        n: int,
        usable: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """For each starting index t in [0, usable), walks k=0..n-1 to find the FIRST dead event
        (end or trunc) within [t, t+n-1], if any, producing:

          - `not_done`: (num_envs, usable) -- 0 wherever a TRUE termination (end=True) occurred
            at or before t+n-1 (matching this codebase's existing compute_lambda_returns
            convention in the original ActorCritic: end[i]=1 means bootstrapping from the state
            AFTER step i is invalid); 1 otherwise -- INCLUDING when the only dead event in the
            window was a truncation. Truncation-only windows still bootstrap: reaching
            WorldModelEnv's imagined-rollout horizon is not a true terminal env state.

          - `bootstrap_stack_flat`: (num_envs, usable, hx_dim) -- the K-frame stack to evaluate
            the target critic at.

          - `bootstrap_discount`: (num_envs, usable) -- gamma^m, where m is the ACTUAL number
            of reward transitions accumulated before the bootstrap point (NOT always gamma^n).
            m = n for a window that survives the full n steps; m = k+1 for a window whose only
            dead event is a truncation at offset k (0-indexed) within the window -- e.g. n=3
            with truncation after 2 transitions (k=1) must discount the bootstrap by gamma^2,
            not gamma^3, since the target is evaluated only 2 steps ahead, not 3. For a window
            containing a true termination this value is masked to 0 by not_done and never read,
            so it's left at whatever value the loop produced (harmless either way).

        Three cases per row/t, matching the above:
          * window survives fully (no end/trunc in [t, t+n-1]): bootstrap_stack_flat =
            all_hx[:, t+n] (the exact state n steps ahead, the same state action-time
            computation used there); bootstrap_discount = gamma^n.
          * window's only dead event is a truncation at t+k (k<n): bootstrap_stack_flat is the
            stack AT THE POINT OF TRUNCATION, built from all_hx[:, t+k] (the pre-truncation
            stack that produced the action taken at t+k) shifted with the TRUE final
            observation (_dense_final_obs[:, t+k]) -- NOT all_hx[:, t+n], which after
            WorldModelEnv's reset_dead() reflects an unrelated, freshly-reset episode's early
            frames; bootstrap_discount = gamma^(k+1).
          * window contains a true termination: not_done masks the bootstrap to 0 regardless of
            bootstrap_stack_flat/bootstrap_discount's values.

        End takes priority over trunc if a row's `end` and `trunc` are ever simultaneously true
        at the same step (shouldn't normally happen, but resolved deterministically either way).
        """
        num_envs = all_hx.size(0)
        hx_dim = all_hx.size(-1)
        dense_final_obs = self._dense_final_obs(infos, end, trunc, all_obs)

        not_done = torch.ones(num_envs, usable, dtype=all_hx.dtype, device=all_hx.device)
        bootstrap_stack_flat = torch.zeros(num_envs, usable, hx_dim, dtype=all_hx.dtype, device=all_hx.device)
        bootstrap_discount = torch.full((num_envs, usable), gamma ** n, dtype=all_hx.dtype, device=all_hx.device)

        for t in range(usable):
            resolved = torch.zeros(num_envs, dtype=torch.bool, device=all_hx.device)
            for k in range(n):
                step = t + k
                end_k = end[:, step].bool()
                trunc_k = trunc[:, step].bool()
                newly_end = end_k & ~resolved
                newly_trunc = trunc_k & ~end_k & ~resolved

                if newly_end.any():
                    not_done[newly_end, t] = 0.0
                    bootstrap_discount[newly_end, t] = gamma ** (k + 1)
                    resolved = resolved | newly_end

                if newly_trunc.any():
                    trunc_stack = self._shift_and_append(all_hx[:, step], dense_final_obs[:, step])
                    bootstrap_stack_flat[newly_trunc, t] = trunc_stack[newly_trunc]
                    bootstrap_discount[newly_trunc, t] = gamma ** (k + 1)
                    resolved = resolved | newly_trunc

            still_alive = ~resolved
            if still_alive.any():
                bootstrap_stack_flat[still_alive, t] = all_hx[still_alive, t + n]
                # bootstrap_discount already defaulted to gamma**n for these rows

        return not_done, bootstrap_stack_flat, bootstrap_discount

    def collect_rollout(self) -> None:
        """Runs ONE env_loop.send() -- a fresh imagined rollout, always from a WorldModelEnv
        (see setup_training's assertion) -- and caches everything critic_update/actor_update
        need. Every observation/action/reward here originates from WorldModelEnv.step(); real
        environment transitions never reach this method (they only ever pass through
        coroutines.collector's SEPARATE env_loop/DrQPolicyBinding, which trains nothing -- see
        make_collector_binding).

        Also advances the exploration-noise schedule exactly once per call -- see
        DrQExplorationState/sample_action's docstrings for why this, not sample_action itself,
        is the training cadence hook."""
        c = self.loss_cfg
        all_obs, act, rew, end, trunc, _dist_params, _val, _val_bootstrap, _z, all_hx, infos = self.env_loop.send(
            c.backup_every
        )

        if self.intrinsic_reward_fn is not None:
            rew = self.intrinsic_reward_fn(infos, rew)

        n = c.n_step
        T = all_obs.size(1)
        usable = T - n
        assert usable > 0, f"n_step={n} must be < backup_every={T} so s_(t+n) is available within the rollout"

        num_envs = all_obs.size(0)

        def chw(stack_flat: Tensor) -> Tensor:
            return stack_flat.view(-1, self.frame_stack * self.img_channels, self.img_size, self.img_size)

        s_t = torch.stack([chw(all_hx[:, t]) for t in range(usable)], dim=1)  # (n, usable, K*C, H, W)
        a_t = act[:, :usable]

        returns = self._accumulate_rewards(rew, end, trunc, c.gamma, n, usable)
        not_done, bootstrap_stack_flat, bootstrap_discount = self._compute_bootstrap_info(
            all_hx, all_obs, end, trunc, infos, c.gamma, n, usable
        )

        self._cached_rollout = {
            "s_t": s_t,
            "a_t": a_t,
            "returns": returns,
            "not_done": not_done,
            "bootstrap_stack_flat": bootstrap_stack_flat,
            "bootstrap_discount": bootstrap_discount,
            "num_envs": num_envs,
            "usable": usable,
        }
        self.exploration_state.schedule_step += 1

    def _set_critic_requires_grad(self, flag: bool) -> None:
        for p in self.critic.parameters():
            p.requires_grad_(flag)

    def critic_update(self, opt_critic: torch.optim.Optimizer) -> Dict[str, Any]:
        """Critic + encoder optimization step, using the rollout collect_rollout() most
        recently cached: encode s_t (grad-tracked -- this is what trains the encoder and the
        critic's OWN projection trunk, both part of self.critic/self.encoder's parameters, see
        module docstring point 4), compute the TD target under no_grad from the target critics
        ONLY (never val/val_bootstrap), Bellman loss, backward, step. The TD target uses a
        PER-SAMPLE bootstrap discount (gamma^m, m = actual transitions before bootstrap -- see
        _compute_bootstrap_info), not a single gamma^n for every sample: a window truncated
        after m < n transitions must discount its bootstrap by gamma^m, not gamma^n, or the
        target overweights states reached fewer steps away than the discount assumes.

        Target-critic soft update happens AFTER opt_critic.step() (not before) -- moving
        targets before the online critic's own update would make them lag the CURRENT step's
        update by an extra full step, silently tracking one step stale forever."""
        assert self._cached_rollout is not None, "call collect_rollout() before critic_update()"
        r = self._cached_rollout
        num_envs, usable = r["num_envs"], r["usable"]

        s_t_flat = r["s_t"].reshape(num_envs * usable, *r["s_t"].shape[2:])
        a_t_flat = r["a_t"].reshape(num_envs * usable, -1)
        bootstrap_flat = r["bootstrap_stack_flat"].reshape(num_envs * usable, -1)
        not_done_flat = r["not_done"].reshape(-1)
        returns_flat = r["returns"].reshape(-1)
        discount_flat = r["bootstrap_discount"].reshape(-1)

        s_t_aug = self.aug(s_t_flat)
        bootstrap_aug = self.aug(self._flat_to_chw(bootstrap_flat))

        features_t = self.encoder(s_t_aug)
        q1, q2 = self.critic(features_t, a_t_flat)

        with torch.no_grad():
            features_boot = self.encoder(bootstrap_aug)
            mu_boot = self.actor(features_boot)
            noise_std = _noise_std_at(self.cfg.noise_schedule, self.exploration_state.schedule_step)
            eps = self._sample_noise(
                mu_boot.shape, noise_std, self.loss_cfg.noise_clip, mu_boot.device, self.exploration_state.generator
            )
            a_boot = (mu_boot + eps).clamp(-1.0, 1.0)
            tq1, tq2 = self.target_critic(features_boot, a_boot)
            target_q = torch.min(tq1, tq2)
            td_target = returns_flat + discount_flat * not_done_flat * target_q

        loss_critic = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        opt_critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        _soft_update(self.target_critic, self.critic, self.loss_cfg.target_tau)

        # Stash the raw (un-augmented) s_t for actor_update's own, independent augmentation +
        # fresh encoder forward pass (at the just-updated encoder weights) -- see
        # actor_update's docstring for why it recomputes rather than reusing features_t as-is.
        self._cached_rollout["s_t_flat_for_actor"] = s_t_flat

        return {
            "loss_critic": loss_critic.detach(),
            "q1_mean": q1.detach().mean(),
            "q2_mean": q2.detach().mean(),
            "target_q_mean": target_q.detach().mean(),
        }

    def actor_update(self, opt_actor: torch.optim.Optimizer) -> Dict[str, Any]:
        """Actor optimization step: fresh encoder forward (at the encoder's POST-critic-update
        weights -- critic_update's own opt_critic.step() already ran), features detached before
        the actor even sees them (so no gradient from this step can reach the encoder), then
        the actor's OWN projection trunk + MLP (module docstring point 4 -- a separate trunk
        instance from the critic's own, so this step never touches any critic parameter through
        a SHARED trunk either). The critic's parameters are ALSO explicitly frozen
        (requires_grad_(False)) for the duration of this step's backward pass -- not merely
        relying on opt_actor containing only actor parameters, but actively preventing any
        gradient computation into critic parameters at all, so `loss_actor.backward()` cannot
        populate a stray `.grad` on them under any circumstance. Consumes (clears) the cached
        rollout afterward, so a stale rollout can never accidentally be reused by a subsequent
        call without an intervening collect_rollout()."""
        assert self._cached_rollout is not None and "s_t_flat_for_actor" in self._cached_rollout, (
            "call collect_rollout() then critic_update() before actor_update()"
        )
        s_t_flat = self._cached_rollout["s_t_flat_for_actor"]
        s_t_aug = self.aug(s_t_flat)

        features = self.encoder(s_t_aug).detach()
        mu = self.actor(features)

        self._set_critic_requires_grad(False)
        try:
            q1_pi, q2_pi = self.critic(features, mu)
            loss_actor = -torch.min(q1_pi, q2_pi).mean()

            opt_actor.zero_grad()
            loss_actor.backward()
            opt_actor.step()
        finally:
            self._set_critic_requires_grad(True)

        self._cached_rollout = None
        return {"loss_actor": loss_actor.detach()}
