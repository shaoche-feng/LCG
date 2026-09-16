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
# e.g. SAC). mean is also clamped: it's an unconstrained linear-layer output (tanh-squashing only
# bounds the *sampled action*, not mean itself), so nothing otherwise stops it drifting.
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
MEAN_ABS_MAX = 10.0
# ATANH_EPS previously bounded the atanh() used to reconstruct z from a replayed action for the
# log_prob computation. That reconstruction was the actual bug (see _tanh_affine_log_prob's
# docstring): whenever tanh(the true z) saturated in float32, atanh silently pinned the
# reconstructed z at atanh(1-ATANH_EPS)~=7.2477 regardless of z's true magnitude, producing a
# systematic (mean - recovered_z) gap that a small std divided into +-tens of thousands. Fixed by
# carrying the true sampled z through the rollout (sample_action -> env_loop -> forward) instead
# of reconstructing it -- log_prob_and_entropy no longer calls atanh() at all. Kept only as a
# documented historical constant / for the regression test that reproduces the old behavior.
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
    # Experimental (causal-isolation study for the epoch-20->30 return regression on
    # walker/walk): when True, critic_linear reads hx.detach() instead of hx, so the value
    # loss's gradient no longer reaches the shared encoder/LSTM trunk (policy and entropy
    # losses are unaffected -- they still backprop through the full, non-detached hx). This
    # changes ONLY the gradient graph: detach() never alters tensor values, so actions and
    # values are numerically identical to the default (False) behavior at all times, including
    # immediately after loading a checkpoint trained with the flag off. Default False preserves
    # exactly the pre-existing behavior; see tests/models/test_detach_value_trunk.py.
    detach_value_trunk: bool = False

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
        self.detach_value_trunk = cfg.detach_value_trunk

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
        # detach_value_trunk (default False, unchanged behavior): hx.detach() has the exact same
        # values as hx, so `val` is numerically identical either way -- only whether loss_values'
        # gradient reaches encoder/lstm changes. hx itself (returned below for the next recurrent
        # step, and used un-detached by actor_linear) is never affected by this local detach.
        value_input = hx.detach() if self.detach_value_trunk else hx
        return ActorCriticOutput(self.actor_linear(hx), self.critic_linear(value_input).squeeze(dim=1), (hx, cx))

    def _split_dist_params(self, dist_params: Tensor) -> Tuple[Tensor, Tensor]:
        mean, log_std = dist_params.chunk(2, dim=-1)
        return mean.clamp(-MEAN_ABS_MAX, MEAN_ABS_MAX), log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def _squash_and_rescale(self, z: Tensor) -> Tensor:
        scale = 0.5 * (self.action_high - self.action_low)
        return self.action_low + (torch.tanh(z) + 1) * scale

    def sample_action(self, dist_params: Tensor, deterministic: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        """Given raw actor output (`logits_act`/`dist_params` from `predict_act_value`), produce
        an action. Called by `env_loop` for both the discrete and continuous paths.

        Returns `(action, z)`. For continuous actions, `z` is the pre-tanh Gaussian sample that
        produced `action` (detached) -- `env_loop` carries it through the rollout so the actor
        loss can compute log_prob directly from the *true* sample instead of reconstructing an
        approximation of it via `atanh(action)`, which is lossy once `tanh(z)` saturates near
        +-1 (see `log_prob_and_entropy`). For discrete actions `z` is always `None`.
        """
        if self.continuous_action:
            mean, log_std = self._split_dist_params(dist_params)
            if deterministic:
                z = mean
            else:
                # Reparameterized draw. DIAMOND's actor loss is a REINFORCE/score-function
                # estimator (-log_prob(z) * advantage.detach()) that recomputes log_prob from
                # the *replayed*, detached z later (see log_prob_and_entropy), so it never needs
                # a pathwise gradient through the sampled z itself -- only through the
                # distribution parameters at replay time. We still draw via rsample() (identical
                # numerically to sample() for a Normal) as requested, but detach z before
                # returning it: if left attached, replaying it through log_prob_and_entropy would
                # add a second, spurious gradient path back to these same parameters (via z's own
                # dependency on them), double-counting/corrupting the REINFORCE gradient.
                # Detaching here matches Categorical.sample()'s implicit detachment.
                z = Normal(mean, log_std.exp()).rsample()
            action = self._squash_and_rescale(z).detach()
            return action, z.detach()
        else:
            if deterministic:
                return dist_params.argmax(dim=-1), None
            return Categorical(logits=dist_params).sample(), None

    def _tanh_affine_log_prob(self, mean: Tensor, std: Tensor, z: Tensor) -> Tensor:
        """log p_Y(y) for y = low + (tanh(z)+1)*scale, z ~ Normal(mean, std), reduced over the
        action dimension. Shared by the replay-based log_prob (log_prob_and_entropy) and the
        fresh-sample entropy estimate (_continuous_entropy_estimate) below -- both call sites now
        pass a genuine sample z (either the detached, *actually-sampled* z carried through the
        rollout by sample_action/env_loop, or a fresh non-detached rsample() for the entropy
        estimate); neither reconstructs z via atanh(action) any more. That reconstruction used to
        be the only place z came from for the replay-based call, and was found to silently pin
        the recovered z at atanh(1-ATANH_EPS)~=7.2477 whenever the true z exceeded that magnitude
        (i.e. whenever tanh(z) saturated in float32) -- producing a systematic, non-random gap
        between the (still-unclamped) mean and the recovered z that, divided by a small std,
        drove log_prob to +-tens of thousands. Passing the real z removes that failure mode at
        its source rather than bounding its symptom (see the removed LOG_PROB_CLAMP_MIN/MAX)."""
        scale = 0.5 * (self.action_high - self.action_low)
        base_log_prob = Normal(mean, std).log_prob(z)
        # log(1 - tanh(z)^2), numerically stable form (same as torch.distributions.TanhTransform)
        tanh_log_abs_det = 2.0 * (math.log(2.0) - z - F.softplus(-2.0 * z))
        log_abs_det = tanh_log_abs_det + scale.log()
        log_prob = (base_log_prob - log_abs_det).sum(dim=-1)
        if not torch.isfinite(log_prob).all():
            n_bad = int((~torch.isfinite(log_prob)).sum().item())
            raise FloatingPointError(
                f"_tanh_affine_log_prob produced {n_bad}/{log_prob.numel()} non-finite log_prob "
                f"value(s) (NaN or Inf) -- refusing to return a corrupted value that could reach "
                f"loss.backward()/optimizer.step(). Diagnostics: "
                f"log_prob[min={log_prob[torch.isfinite(log_prob)].min().item() if torch.isfinite(log_prob).any() else float('nan'):.3f}, "
                f"max_finite={log_prob[torch.isfinite(log_prob)].max().item() if torch.isfinite(log_prob).any() else float('nan'):.3f}] "
                f"mean[min={mean.detach().min().item():.3f}, max={mean.detach().max().item():.3f}, "
                f"abs_max={mean.detach().abs().max().item():.3f}] "
                f"std[min={std.detach().min().item():.6f}, max={std.detach().max().item():.6f}] "
                f"z[min={z.detach().min().item():.3f}, max={z.detach().max().item():.3f}, "
                f"abs_max={z.detach().abs().max().item():.3f}] "
                f"base_log_prob[min={base_log_prob.detach().min().item():.3f}, max={base_log_prob.detach().max().item():.3f}] "
                f"log_abs_det[min={log_abs_det.detach().min().item():.3f}, max={log_abs_det.detach().max().item():.3f}]"
            )
        return log_prob

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

    def log_prob_and_entropy(
        self, dist_params: Tensor, action: Tensor, z: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Compute (log_prob, entropy) for a rollout's stored step, reducing over the action
        dimension so the result has one scalar per sample (matching `Categorical.log_prob`/
        `.entropy`'s shape).

        Continuous case: `dist_params` is a fresh (differentiable) forward pass' (mean, log_std).
        `z` is the *true* pre-tanh sample that produced `action`, carried through the rollout by
        `sample_action`/`env_loop` (see their docstrings) -- `log_prob` is evaluated directly at
        this detached `z`, which is the quantity the REINFORCE actor loss needs
        (`-log_prob(z) * advantage.detach()` in `forward()`). `action` itself is unused in the
        continuous branch (kept as a parameter for interface parity with the discrete branch,
        which still needs it for `Categorical.log_prob`). `entropy` is a *separate* quantity
        computed by `_continuous_entropy_estimate` from a fresh, non-detached reparameterized
        sample -- see that method's docstring for why reusing `-log_prob(z)` here would be a
        zero-expectation, invalid gradient estimator for entropy specifically (even though it is
        a valid, unbiased *value* estimate of the entropy itself -- just not of its gradient).
        """
        if self.continuous_action:
            assert z is not None, "continuous_action requires the sampled z carried by env_loop"
            mean, log_std = self._split_dist_params(dist_params)
            std = log_std.exp()
            policy_z = z.detach()
            log_prob = self._tanh_affine_log_prob(mean, std, policy_z)
            entropy = self._continuous_entropy_estimate(dist_params)
            return log_prob, entropy
        else:
            d = Categorical(logits=dist_params)
            return d.log_prob(action), d.entropy()

    def forward(self) -> LossAndLogs:
        c = self.loss_cfg
        _, act, rew, end, trunc, logits_act, val, val_bootstrap, z, infos = self.env_loop.send(c.backup_every)

        if self.intrinsic_reward_fn is not None:
            rew = self.intrinsic_reward_fn(infos, rew)

        log_prob, entropy_per_sample = self.log_prob_and_entropy(logits_act, act, z)
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
