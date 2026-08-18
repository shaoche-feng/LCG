from collections import namedtuple
from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import torch.nn.functional as F

from .blocks import Conv3x3, SmallResBlock
from coroutines.env_loop import make_env_loop
from envs import TorchEnv, WorldModelEnv
from utils import init_lstm, LossAndLogs


ActorCriticOutput = namedtuple("ActorCriticOutput", "logits_act val hx_cx")

# Squashed-Gaussian continuous policy: clamp log_std for numerical stability (standard practice,
# e.g. SAC), and keep tanh^-1 away from its +-1 singularities when inverting a replayed action.
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
ATANH_EPS = 1e-6


@dataclass
class ActorCriticLossConfig:
    backup_every: int
    gamma: float
    lambda_: float
    weight_value_loss: float
    weight_entropy_loss: float


@dataclass
class ActorCriticConfig:
    lstm_dim: int
    img_channels: int
    img_size: int
    channels: List[int]
    down: List[int]
    num_actions: Optional[int] = None  # discrete action count, e.g. Atari
    continuous_action_dim: Optional[int] = None  # continuous action dimension, e.g. DM Control
    action_low: Optional[List[float]] = None  # required iff continuous_action_dim is set
    action_high: Optional[List[float]] = None  # required iff continuous_action_dim is set
    continuous_reward: bool = False  # False: Atari-style reward-sign lambda returns; True: raw reward

    # NOTE: no `__post_init__` validation of num_actions/continuous_action_dim here, on purpose,
    # mirroring InnerModelConfig/RewEndModelConfig. `num_actions` is left unset (None) in
    # config/agent/default.yaml and patched onto an already-constructed ActorCriticConfig
    # instance by AgentConfig.__post_init__ (agent.py), i.e. *after* hydra's bottom-up
    # instantiate() has already built this dataclass. Validating "exactly one of the two is set"
    # here would fire during that intermediate state and break the existing Atari path. The
    # check instead lives in ActorCritic.__init__, which only runs once the config is populated.


class ActorCritic(nn.Module):
    def __init__(self, cfg: ActorCriticConfig) -> None:
        super().__init__()
        assert (cfg.num_actions is None) != (cfg.continuous_action_dim is None), (
            "ActorCriticConfig requires exactly one of `num_actions` (discrete action space) "
            "or `continuous_action_dim` (continuous action space) to be set."
        )
        self.encoder = ActorCriticEncoder(cfg)
        self.lstm_dim = cfg.lstm_dim
        input_dim_lstm = cfg.channels[-1] * (cfg.img_size // 2 ** (sum(cfg.down))) ** 2
        self.lstm = nn.LSTMCell(input_dim_lstm, cfg.lstm_dim)
        self.critic_linear = nn.Linear(cfg.lstm_dim, 1)

        self.continuous_action = cfg.continuous_action_dim is not None
        self.continuous_reward = cfg.continuous_reward

        if self.continuous_action:
            assert cfg.action_low is not None and cfg.action_high is not None, (
                "continuous_action_dim requires action_low and action_high to be set"
            )
            assert len(cfg.action_low) == len(cfg.action_high) == cfg.continuous_action_dim
            self.actor_linear = nn.Linear(cfg.lstm_dim, 2 * cfg.continuous_action_dim)  # (mean, log_std)
            self.register_buffer("action_low", torch.tensor(cfg.action_low, dtype=torch.float32))
            self.register_buffer("action_high", torch.tensor(cfg.action_high, dtype=torch.float32))
        else:
            self.actor_linear = nn.Linear(cfg.lstm_dim, cfg.num_actions)

        self.actor_linear.weight.data.fill_(0)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data.fill_(0)
        self.critic_linear.bias.data.fill_(0)
        init_lstm(self.lstm)

        self.env_loop = None
        self.loss_cfg = None
        self.intrinsic_reward_fn = None

    @property
    def device(self) -> torch.device:
        return self.lstm.weight_hh.device

    def setup_training(self, rl_env: Union[TorchEnv, WorldModelEnv], loss_cfg: ActorCriticLossConfig) -> None:
        assert self.env_loop is None and self.loss_cfg is None
        self.env_loop = make_env_loop(rl_env, self)
        self.loss_cfg = loss_cfg

    def set_intrinsic_reward_fn(self, fn) -> None:
        """Optional hook: fn(infos, env_rew) -> Tensor, same shape as env_rew, used in place
        of the environment/world-model reward when set. `infos` is the per-step list
        env_loop already yields (previously discarded); `env_rew` is the reward env_loop
        collected from `rl_env.step()` for that rollout. None (the default) reproduces
        forward()'s exact prior behavior -- this method exists only so that behavior can be
        opted into, never as a side effect of another call."""
        self.intrinsic_reward_fn = fn

    def predict_act_value(self, obs: Tensor, hx_cx: Tuple[Tensor, Tensor]) -> ActorCriticOutput:
        assert obs.ndim == 4
        x = self.encoder(obs)
        x = x.flatten(start_dim=1)
        hx, cx = self.lstm(x, hx_cx)
        return ActorCriticOutput(self.actor_linear(hx), self.critic_linear(hx).squeeze(dim=1), (hx, cx))

    def _split_dist_params(self, dist_params: Tensor) -> Tuple[Tensor, Tensor]:
        mean, log_std = dist_params.chunk(2, dim=-1)
        return mean, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def _squash_and_rescale(self, z: Tensor) -> Tensor:
        scale = 0.5 * (self.action_high - self.action_low)
        return self.action_low + (torch.tanh(z) + 1) * scale

    def sample_action(self, dist_params: Tensor, deterministic: bool = False) -> Tensor:
        """Given raw actor output (`logits_act`/`dist_params` from `predict_act_value`), produce
        an action. Not currently called by `env_loop` (which constructs `Categorical` itself for
        the discrete path) -- provided so continuous-policy sampling logic lives entirely inside
        `ActorCritic` for testing, ahead of the `env_loop` integration that will use it later.
        """
        if self.continuous_action:
            mean, log_std = self._split_dist_params(dist_params)
            if deterministic:
                z = mean
            else:
                # Reparameterized draw. DIAMOND's actor loss is a REINFORCE/score-function
                # estimator (-log_prob(act) * advantage.detach()) that recomputes log_prob from
                # the *replayed* action later (see log_prob_and_entropy), so it never needs a
                # pathwise gradient through the sampled action itself -- only through the
                # distribution parameters at replay time. We still draw via rsample() (identical
                # numerically to sample() for a Normal) as requested, but detach the action before
                # returning it: if left attached, replaying it through log_prob_and_entropy would
                # add a second, spurious gradient path back to these same parameters (via the
                # action's own dependency on them), double-counting/corrupting the REINFORCE
                # gradient. Detaching here matches Categorical.sample()'s implicit detachment.
                z = Normal(mean, log_std.exp()).rsample()
            return self._squash_and_rescale(z).detach()
        else:
            if deterministic:
                return dist_params.argmax(dim=-1)
            return Categorical(logits=dist_params).sample()

    def _tanh_affine_log_prob(self, mean: Tensor, std: Tensor, z: Tensor) -> Tensor:
        """log p_Y(y) for y = low + (tanh(z)+1)*scale, z ~ Normal(mean, std), reduced over the
        action dimension. Shared by the replay-based log_prob (log_prob_and_entropy) and the
        fresh-sample entropy estimate (_continuous_entropy_estimate) below -- the only difference
        between the two call sites is whether `z` came from inverting a detached replayed action
        or from a fresh, non-detached rsample()."""
        scale = 0.5 * (self.action_high - self.action_low)
        base_log_prob = Normal(mean, std).log_prob(z)
        # log(1 - tanh(z)^2), numerically stable form (same as torch.distributions.TanhTransform)
        tanh_log_abs_det = 2.0 * (math.log(2.0) - z - F.softplus(-2.0 * z))
        log_abs_det = tanh_log_abs_det + scale.log()
        return (base_log_prob - log_abs_det).sum(dim=-1)

    def _continuous_entropy_estimate(self, dist_params: Tensor) -> Tensor:
        """Reparameterized single-sample Monte Carlo estimate of the entropy of the *transformed*
        (tanh-squashed, affine-rescaled) action distribution Y = f_theta(Z), Z ~ Normal(mean,std).

        Derivation: H[Y] = E_Y[-log p_Y(Y)] = E_Z[-log p_Y(f(Z))] (change of variables under the
        expectation) = E_eps[-log p_Y(f_theta(g_theta(eps)))], eps ~ N(0,1) fixed, g_theta(eps) =
        mean + std*eps. Drawing one z = g_theta(eps) via rsample() (NOT detached) and evaluating
        -log p_Y(f(z)) -- via _tanh_affine_log_prob, which already accounts for the full tanh +
        affine Jacobian -- is exactly a single-sample MC estimate of H[Y], i.e. the entropy of the
        fully transformed action distribution, not of the base Normal Z (whose entropy has the
        simple closed form 0.5*log(2*pi*e*std^2) and ignores the squashing entirely -- that would
        be a different, incorrect quantity here).

        Because z is not detached, autograd differentiates -log p_Y(f(z)) w.r.t. (mean, std)
        through *two* paths: the explicit occurrence of (mean, std) inside the log-density formula,
        and z's own dependence on (mean, std) via the reparameterization. Averaging this estimator
        over a minibatch gives an unbiased, non-zero-in-expectation estimate of the true entropy
        gradient d(H[Y])/d(theta) -- unlike -log_prob evaluated at a *detached* sample, whose
        gradient under this same formula only captures the first path and has zero expectation
        (see the caller's derivation / conversation), which is why it is not reused here.
        """
        mean, log_std = self._split_dist_params(dist_params)
        std = log_std.exp()
        z = Normal(mean, std).rsample()  # fresh draw, independent of any replayed action, not detached
        return -self._tanh_affine_log_prob(mean, std, z)

    def log_prob_and_entropy(self, dist_params: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute (log_prob, entropy) for a rollout's stored `action`, reducing over the action
        dimension so the result has one scalar per sample (matching `Categorical.log_prob`/
        `.entropy`'s shape).

        Continuous case: `dist_params` is a fresh (differentiable) forward pass' (mean, log_std).
        `log_prob` is evaluated at the *replayed* `action` (produced by `sample_action`, which
        detaches it) by inverting its squash + affine-rescale to recover the pre-squash value --
        this is the quantity the REINFORCE actor loss needs (`-log_prob(act) * advantage.detach()`
        in `forward()`), and using a detached action here is correct and intentional for that
        purpose. `entropy` is a *separate* quantity computed by `_continuous_entropy_estimate`
        from a fresh, non-detached reparameterized sample -- see that method's docstring for why
        reusing `-log_prob(action)` here would be a zero-expectation, invalid gradient estimator
        for entropy specifically (even though it is a valid, unbiased *value* estimate of the
        entropy itself -- just not of its gradient).
        """
        if self.continuous_action:
            mean, log_std = self._split_dist_params(dist_params)
            std = log_std.exp()
            scale = 0.5 * (self.action_high - self.action_low)
            tanh_z = (action - self.action_low) / scale - 1.0
            z = torch.atanh(tanh_z.clamp(-1 + ATANH_EPS, 1 - ATANH_EPS))
            log_prob = self._tanh_affine_log_prob(mean, std, z)
            entropy = self._continuous_entropy_estimate(dist_params)
            return log_prob, entropy
        else:
            d = Categorical(logits=dist_params)
            return d.log_prob(action), d.entropy()

    def forward(self) -> LossAndLogs:
        c = self.loss_cfg
        _, act, rew, end, trunc, logits_act, val, val_bootstrap, infos = self.env_loop.send(c.backup_every)

        if self.intrinsic_reward_fn is not None:
            rew = self.intrinsic_reward_fn(infos, rew)

        log_prob, entropy_per_sample = self.log_prob_and_entropy(logits_act, act)
        entropy = entropy_per_sample.mean()

        lambda_returns = compute_lambda_returns(
            rew, end, trunc, val_bootstrap, c.gamma, c.lambda_, continuous_reward=self.continuous_reward
        )

        loss_actions = (-log_prob * (lambda_returns - val).detach()).mean()
        loss_values = c.weight_value_loss * F.mse_loss(val, lambda_returns)
        loss_entropy = -c.weight_entropy_loss * entropy

        loss = loss_actions + loss_entropy + loss_values

        metrics = {
            "policy_entropy": entropy.detach() / math.log(2),
            "loss_actions": loss_actions.detach(),
            "loss_entropy": loss_entropy.detach(),
            "loss_values": loss_values.detach(),
            "loss_total": loss.detach(),
        }

        return loss, metrics


class ActorCriticEncoder(nn.Module):
    def __init__(self, cfg: ActorCriticConfig) -> None:
        super().__init__()
        assert len(cfg.channels) == len(cfg.down)
        encoder_layers = [Conv3x3(cfg.img_channels, cfg.channels[0])]
        for i in range(len(cfg.channels)):
            encoder_layers.append(SmallResBlock(cfg.channels[max(0, i - 1)], cfg.channels[i]))
            if cfg.down[i]:
                encoder_layers.append(nn.MaxPool2d(2))
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


@torch.no_grad()
def compute_lambda_returns(
    rew: Tensor,
    end: Tensor,
    trunc: Tensor,
    val_bootstrap: Tensor,
    gamma: float,
    lambda_: float,
    continuous_reward: bool = False,
) -> Tensor:
    assert rew.ndim == 2 and rew.size() == end.size() == trunc.size() == val_bootstrap.size()

    if not continuous_reward:
        rew = rew.sign()  # clip reward (Atari); continuous-reward envs use the raw value as-is

    end_or_trunc = (end + trunc).clip(max=1)
    not_end = 1 - end
    not_trunc = 1 - trunc

    lambda_returns = rew + not_end * gamma * (not_trunc * (1 - lambda_) + trunc) * val_bootstrap

    if lambda_ == 0:
        return lambda_returns

    last = val_bootstrap[:, -1]
    for t in reversed(range(rew.size(1))):
        lambda_returns[:, t] += end_or_trunc[:, t].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, t]

    return lambda_returns
