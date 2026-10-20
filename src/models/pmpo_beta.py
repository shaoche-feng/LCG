"""Sign-only PMPO on fresh DIAMOND experience; no differentiable imagination."""
from copy import deepcopy
from dataclasses import dataclass, replace
from contextlib import contextmanager
import random
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta, kl_divergence
from torch.nn import functional as F

from .actor_critic import ActorCriticEncoder, ActorCriticOutput, compute_lambda_returns
from coroutines.env_loop import make_env_loop
from envs import WorldModelEnv


@dataclass
class PMPOBetaConfig:
    img_channels: int
    img_size: int
    channels: List[int]
    down: List[int]
    # Hydra constructs component configs before AgentConfig supplies environment metadata.
    continuous_action_dim: Optional[int] = None
    action_low: Optional[List[float]] = None
    action_high: Optional[List[float]] = None
    frame_stack: int = 4
    actor_hidden_dims: tuple = (256, 256)
    value_hidden_dims: tuple = (256, 256)
    actor_lr: float = 1e-4
    value_lr: float = 1e-4
    gamma: float = 0.985
    lambda_: float = 0.95
    alpha_pmpo: float = 0.5
    beta_kl: float = 0.3
    # Entropy bonus coefficient (Stage 3 experiment): actor loss becomes
    # L_PMPO + beta_kl * KL - entropy_coef * H(pi). 0.0 (default) exactly
    # reproduces the frozen 100k baseline; no other term changes with it.
    entropy_coef: float = 0.0
    # Experiment (stage3-weighted-beta-seed1): raw PMPO loss weighting. "sign" (default) is the
    # original sign-only loss, unchanged. "abs_advantage" multiplies each log-probability by
    # |detached advantage| and keeps the original per-group sample-count denominators: no
    # normalization by the weight sum or batch std, no clipping, no dead zone. KL and entropy
    # coefficients are unchanged, so this also changes PMPO's strength relative to those terms.
    pmpo_weighting: str = "sign"
    # Matches config/agent/pmpo_beta.yaml; saved configs always state their own value.
    prior_refresh_interval: int = 500
    # Diagnosis-only knob (see docs/pmpo_beta/k4_implementation.md): when True, the prior is
    # snapshotted once at construction and never refreshed again, isolating the moving-prior
    # trust-region mechanism as the single variable under test. False (default) preserves the
    # existing every-prior_refresh_interval-updates moving prior exactly as before.
    fixed_prior: bool = False
    concentration_min: float = 1.0
    imagination_horizon: int = 15
    max_grad_norm: float = 10.0
    seed: int = 0
    log_diagnostics: bool = False


def require_finite(**tensors):
    for name, tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite PMPO {name}")


def summary_statistics(values, prefix):
    values = values.detach().flatten()
    quantiles = torch.quantile(values, values.new_tensor([0.1, 0.5, 0.9]))
    return {f"{prefix}_{name}": value for name, value in
            zip(("mean", "std", "min", "max", "p10", "median", "p90"),
                (values.mean(), values.std(unbiased=False), values.min(), values.max(), *quantiles))}


def pmpo_loss(log_prob, advantage, alpha=0.5, weighting="sign"):
    """Eq. 11: (1-alpha) mean_negative(log pi) - alpha mean_positive(log pi).

    Empty groups contribute zero, without renormalizing the surviving weight.
    Exactly zero belongs to the positive group. weighting="sign": no magnitude weighting
    (original). weighting="abs_advantage" (experiment): each log-probability is multiplied
    by |detached advantage| before the same per-group sums and sample-count denominators,
    so exactly zero advantages contribute zero.
    """
    if log_prob.shape != advantage.shape or not 0 <= alpha <= 1 or not log_prob.numel():
        raise ValueError("Invalid PMPO shapes, empty batch, or mixing coefficient")
    if weighting not in ("sign", "abs_advantage"):
        raise ValueError(f"Unknown PMPO weighting {weighting!r}")
    require_finite(log_prob=log_prob, advantage=advantage)
    positive = advantage.detach() >= 0
    negative = ~positive
    if weighting == "abs_advantage":
        log_prob = advantage.detach().abs() * log_prob
    pos = log_prob.masked_fill(~positive, 0).sum() / positive.sum().clamp_min(1)
    neg = log_prob.masked_fill(~negative, 0).sum() / negative.sum().clamp_min(1)
    return (1 - alpha) * neg - alpha * pos


class VisualHead(nn.Module):
    def __init__(self, cfg, hidden_dims, outputs):
        super().__init__()
        self.encoder = ActorCriticEncoder(cfg)
        width = cfg.channels[-1] * (cfg.img_size // 2 ** sum(cfg.down)) ** 2
        layers = [nn.Flatten()]
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.SiLU()])
            width = hidden
        layers.append(nn.Linear(width, outputs))
        self.head = nn.Sequential(*layers)
        # Small nonzero weights preserve initial state dependence.
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, obs):
        return self.head(self.encoder(obs))


class PMPOBeta(nn.Module):
    def __init__(self, cfg: PMPOBetaConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.frame_stack != 4:
            raise ValueError("This PMPO experiment requires exactly four frames")
        self.frame_stack = cfg.frame_stack
        seed = cfg.seed if cfg.seed is not None else 0
        if cfg.continuous_action_dim is None or cfg.continuous_action_dim < 1:
            raise ValueError("PMPOBeta requires an environment-derived continuous action space")
        low, high = torch.tensor(cfg.action_low), torch.tensor(cfg.action_high)
        if low.shape != (cfg.continuous_action_dim,) or high.shape != low.shape:
            raise ValueError("Invalid action bounds shape")
        require_finite(low=low, high=high)
        require_finite(action_width=high - low)
        if not (high > low).all():
            raise ValueError("Action bounds must have positive width")
        if not (0 <= cfg.gamma <= 1 and 0 <= cfg.lambda_ <= 1 and 0 <= cfg.alpha_pmpo <= 1):
            raise ValueError("Invalid discount, lambda, or PMPO mixing coefficient")
        if min(cfg.concentration_min, cfg.actor_lr, cfg.value_lr, cfg.max_grad_norm) <= 0:
            raise ValueError("Concentration floor, learning rates and gradient clip must be positive")
        if cfg.beta_kl < 0 or cfg.entropy_coef < 0 or cfg.prior_refresh_interval < 1 or cfg.imagination_horizon < 1:
            raise ValueError("Invalid KL/entropy coefficient or rollout/block length")
        if cfg.pmpo_weighting not in ("sign", "abs_advantage"):
            raise ValueError(f"Unknown PMPO weighting {cfg.pmpo_weighting!r}")
        self.register_buffer("action_low", low.float())
        self.register_buffer("action_high", high.float())
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))
        stacked_cfg = replace(cfg, img_channels=cfg.img_channels * cfg.frame_stack)
        self.actor = VisualHead(stacked_cfg, cfg.actor_hidden_dims, 2 * cfg.continuous_action_dim)
        self.value = VisualHead(stacked_cfg, cfg.value_hidden_dims, 1)
        self.prior_actor = deepcopy(self.actor).requires_grad_(False).eval()
        # Diagnostic only (see pmpo_diagnostic.py): a SEPARATE frozen snapshot from
        # prior_actor, taken once here and never refreshed, so KL(current || initial)
        # can distinguish local (per-block) trust-region movement from total cumulative
        # drift over the whole run. Never read by the loss.
        self.initial_actor = deepcopy(self.actor).requires_grad_(False).eval()
        self.intrinsic_reward_fn = None
        self.rl_env = None
        self._fixed_obs = None
        self._rng = dict(cpu=torch.Generator().manual_seed(seed).get_state(), cuda=None,
                         python=random.Random(seed).getstate(), numpy=np.random.RandomState(seed).get_state())

    @property
    def device(self):
        return self.action_low.device

    def get_extra_state(self):
        return deepcopy(dict(rng=self._rng, fixed_obs=self._fixed_obs))

    def set_extra_state(self, state):
        self._rng = deepcopy(state["rng"])
        self._fixed_obs = state["fixed_obs"]

    @contextmanager
    def imagination_rng(self):
        """Checkpointed private stream for policy, WM sampling and prompt selection."""
        py_state, np_state = random.getstate(), np.random.get_state()
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self._rng["cpu"].cpu())
            if devices:
                if self._rng["cuda"] is None:
                    gen = torch.Generator(device=self.device).manual_seed(self.cfg.seed or 0)
                    self._rng["cuda"] = gen.get_state()
                torch.cuda.set_rng_state(self._rng["cuda"].cpu(), self.device)
            random.setstate(self._rng["python"])
            np.random.set_state(self._rng["numpy"])
            try:
                yield
            finally:
                self._rng = dict(cpu=torch.get_rng_state(),
                                 cuda=torch.cuda.get_rng_state(self.device) if devices else None,
                                 python=random.getstate(), numpy=np.random.get_state())
                random.setstate(py_state)
                np.random.set_state(np_state)

    def distribution(self, raw):
        require_finite(raw_parameters=raw)
        a, b = (F.softplus(raw) + self.cfg.concentration_min).chunk(2, dim=-1)
        return Beta(a, b)

    def to_environment(self, unit):
        # Equivalent to canonical = 2*x-1; low + (canonical+1)*(high-low)/2.
        return self.action_low + unit * (self.action_high - self.action_low)

    def sample_action(self, raw, deterministic=False):
        dist = self.distribution(raw)
        unit = dist.mean if deterministic else dist.sample()
        return self.to_environment(unit).detach()

    def log_prob_and_entropy(self, raw, action):
        width = self.action_high - self.action_low
        unit = (action.detach() - self.action_low) / width
        if ((unit < 0) | (unit > 1)).any():
            raise ValueError("Stored action outside environment bounds")
        # Affine float32 roundoff can land exactly on an endpoint. Only move the
        # argument inside numerical support; never clamp log probabilities.
        eps = torch.finfo(unit.dtype).eps
        unit = unit.clamp(eps, 1 - eps)
        dist = self.distribution(raw)
        logp = (dist.log_prob(unit) - width.log()).sum(-1)
        entropy = (dist.entropy() + width.log()).sum(-1)
        require_finite(log_probability=logp, entropy=entropy)
        return logp, entropy

    def predict_act_value(self, obs):
        if obs.ndim != 4 or obs.shape[1] != self.cfg.img_channels * self.frame_stack:
            raise ValueError("PMPO requires the exact channel-concatenated four-frame state")
        return ActorCriticOutput(self.actor(obs), self.value(obs).squeeze(-1))

    def setup_training(self, rl_env):
        if not isinstance(rl_env, WorldModelEnv):
            raise ValueError("PMPO learns only from WorldModelEnv imagined transitions")
        if rl_env.data_loader.num_workers != 0:
            raise ValueError("PMPO prototype requires num_workers_data_loaders=0 for exact prompt RNG")
        self.rl_env = rl_env

    def set_intrinsic_reward_fn(self, fn):
        self.intrinsic_reward_fn = fn

    @torch.no_grad()
    def collect_imagination(self):
        if self.rl_env is None:
            raise RuntimeError("Call setup_training with WorldModelEnv first")
        # Each update starts a fresh prompt block: no hidden coroutine/prefetch
        # state survives checkpoint boundaries. Real prompts are conditioning only.
        with self.imagination_rng():
            self.rl_env.restart_initial_conditions()
            loop = make_env_loop(self.rl_env, self, store_policy_observations=True)
            try:
                return loop.send(self.cfg.imagination_horizon)
            finally:
                loop.close()

    def forward(self):
        c = self.cfg
        if not c.fixed_prior and self.updates.item() % c.prior_refresh_interval == 0:
            self.prior_actor.load_state_dict(self.actor.state_dict())
        self.prior_actor.eval()
        obs, act, rew, end, trunc, _, old_values, bootstrap, infos = self.collect_imagination()
        if self.intrinsic_reward_fn is not None:
            rew = self.intrinsic_reward_fn(infos, rew).detach()
        targets = compute_lambda_returns(rew, end, trunc, bootstrap, c.gamma, c.lambda_)
        advantage = targets - old_values
        flat_obs = obs.flatten(0, 1).detach()
        raw = self.actor(flat_obs)
        values = self.value(flat_obs).squeeze(-1)
        logp, entropy = self.log_prob_and_entropy(raw, act.flatten(0, 1))
        dist = self.distribution(raw)
        with torch.no_grad():
            prior = self.distribution(self.prior_actor(flat_obs))
            # Diagnostic only, see initial_actor's docstring above: never added to
            # actor_loss, computed fully detached so it cannot perturb the main graph.
            kl_initial = kl_divergence(self.distribution(raw.detach()), self.distribution(self.initial_actor(flat_obs))).sum(-1)
        kl = kl_divergence(dist, prior).sum(-1)
        pmpo_raw_loss = pmpo_loss(logp, advantage.flatten(), c.alpha_pmpo, c.pmpo_weighting)
        kl_loss = c.beta_kl * kl.mean()
        entropy_loss = -c.entropy_coef * entropy.mean()
        actor_loss = pmpo_raw_loss + kl_loss + entropy_loss
        value_loss = F.mse_loss(values, targets.flatten())
        require_finite(targets=targets, values=values, kl=kl, actor_loss=actor_loss, value_loss=value_loss)
        metrics = self.diagnostics(raw.detach(), act.flatten(0, 1), logp.detach(), entropy.detach())
        metrics.update(loss_actor=actor_loss.detach(), loss_value=value_loss.detach(), kl_prior=kl.mean().detach(),
                       kl_initial=kl_initial.mean().detach(),
                       loss_pmpo_raw=pmpo_raw_loss.detach(), loss_entropy=entropy_loss.detach(),
                       loss_actor_total=actor_loss.detach(),
                       value_mean=values.mean().detach(), value_std=values.std(unbiased=False).detach(),
                       return_mean=targets.mean(), return_std=targets.std(unbiased=False),
                       positive_fraction=(advantage >= 0).float().mean(), negative_fraction=(advantage < 0).float().mean(),
                       rollout_length=float(obs.shape[1]), all_finite=1.0)
        with torch.no_grad():
            metrics.update(summary_statistics(advantage, "advantage"))
            pos = advantage.flatten() >= 0
            pos_logp = logp.detach().masked_fill(~pos, 0).sum() / pos.sum().clamp_min(1)
            neg_logp = logp.detach().masked_fill(pos, 0).sum() / (~pos).sum().clamp_min(1)
            centered_value = values.detach() - values.detach().mean()
            centered_target = targets.flatten() - targets.mean()
            correlation = (centered_value * centered_target).mean() / (
                centered_value.square().mean() * centered_target.square().mean()).sqrt().clamp_min(1e-12)
            metrics.update(positive_log_probability=pos_logp, negative_log_probability=neg_logp,
                           loss_positive=-c.alpha_pmpo * pos_logp,
                           loss_negative=(1 - c.alpha_pmpo) * neg_logp,
                           loss_kl=c.beta_kl * kl.mean().detach(),
                           advantage_zero_fraction=(advantage == 0).float().mean(),
                           advantage_near_zero_fraction=(advantage.abs() <= 1e-6).float().mean(),
                           value_target_correlation=correlation,
                           value_bias=(values.detach() - targets.flatten()).mean(),
                           value_abs_max=values.detach().abs().max(), return_abs_max=targets.abs().max())
            width = self.action_high - self.action_low
            per_dim_entropy = dist.entropy().detach() + width.log()
            for d in range(per_dim_entropy.shape[-1]):
                metrics[f"policy_entropy_dim_{d}"] = per_dim_entropy[:, d].mean()
            frames = flat_obs.reshape(-1, self.frame_stack, self.cfg.img_channels, self.cfg.img_size, self.cfg.img_size)
            difference = (frames[:, 1:] - frames[:, :-1]).abs().flatten(1).mean(1)
            metrics.update(stack_temporal_difference_mean=difference.mean(),
                           stack_temporal_variation_fraction=(difference > 0).float().mean())
        with torch.no_grad():
            if self._fixed_obs is None:
                self._fixed_obs = flat_obs[:32].clone()
            fixed_mean = self.sample_action(self.actor(self._fixed_obs.to(self.device)), deterministic=True)
            for d, std in enumerate(fixed_mean.std(0, unbiased=False)):
                metrics[f"fixed_policy_mean_state_std_{d}"] = std
            metrics.update(self.temporal_diagnostics(self._fixed_obs.to(self.device), "fixed"))
            require_finite(**{k: torch.as_tensor(v) for k, v in metrics.items()})
        # Disjoint graphs: each optimizer owns exactly one encoder and head.
        return actor_loss + value_loss, metrics

    @torch.no_grad()
    def diagnostics(self, raw, actions, logp=None, entropy=None):
        dist = self.distribution(raw)
        unit = (actions - self.action_low) / (self.action_high - self.action_low)
        metrics = {}
        for name, param in (("alpha", dist.concentration1), ("beta", dist.concentration0)):
            metrics.update(summary_statistics(param, name))
            metrics[f"{name}_near_floor_fraction"] = (param < self.cfg.concentration_min + 0.01).float().mean()
        means = self.to_environment(dist.mean)
        # Per-state stochastic spread (instrumentation; analytic, consumes no RNG): each sample's
        # conditional Beta variance from its own (alpha, beta), in environment units, averaged
        # over states before the square root. Distinct from policy_mean_state_std_* (variation
        # of mean actions across states) and action_std_* (spread of sampled actions overall).
        a, b = dist.concentration1, dist.concentration0
        conditional_var = a * b / ((a + b) ** 2 * (a + b + 1)) * (self.action_high - self.action_low) ** 2
        for d in range(actions.shape[-1]):
            metrics[f"policy_conditional_std_{d}"] = conditional_var[:, d].mean().sqrt()
        for d in range(actions.shape[-1]):
            metrics[f"action_mean_{d}"] = actions[:, d].mean()
            metrics[f"action_std_{d}"] = actions[:, d].std(unbiased=False)
            metrics[f"policy_mean_{d}"] = means[:, d].mean()
            metrics[f"policy_mean_state_std_{d}"] = means[:, d].std(unbiased=False)
            metrics[f"policy_mean_min_{d}"] = means[:, d].min()
            metrics[f"policy_mean_max_{d}"] = means[:, d].max()
        metrics.update(near_lower_fraction=(unit < 0.01).float().mean(),
                       near_upper_fraction=(unit > 0.99).float().mean())
        if logp is not None:
            metrics.update(log_probability=logp.mean(), policy_entropy=entropy.mean(),
                           policy_entropy_std=entropy.std(unbiased=False))
        return metrics

    @torch.no_grad()
    def temporal_diagnostics(self, states, prefix="fixed"):
        frames = states.reshape(-1, self.frame_stack, self.cfg.img_channels, self.cfg.img_size, self.cfg.img_size)
        ordered = self.sample_action(self.actor(states), deterministic=True)
        repeated = self.sample_action(self.actor(frames[:, -1].repeat(1, self.frame_stack, 1, 1)), deterministic=True)
        reversed_ = self.sample_action(self.actor(frames.flip(1).flatten(1, 2)), deterministic=True)
        difference = (frames[:, 1:] - frames[:, :-1]).abs().flatten(1).mean(1)
        result = {f"{prefix}_temporal_difference_mean": difference.mean(),
                  f"{prefix}_temporal_variation_fraction": (difference > 0).float().mean(),
                  f"{prefix}_ordered_repeated_action_difference": (ordered - repeated).abs().mean(),
                  f"{prefix}_ordered_reversed_action_difference": (ordered - reversed_).abs().mean()}
        for d in range(ordered.shape[1]):
            result[f"{prefix}_policy_mean_state_std_{d}"] = ordered[:, d].std(unbiased=False)
        return result


class PMPOOptimizers:
    """Two Adams + two schedulers, serialized through Trainer's CommonTools."""
    def __init__(self, model, warmup_steps):
        from utils import get_lr_sched
        self.model = model
        self.actor = torch.optim.Adam(model.actor.parameters(), lr=model.cfg.actor_lr)
        self.value = torch.optim.Adam(model.value.parameters(), lr=model.cfg.value_lr)
        self.schedulers = [get_lr_sched(self.actor, warmup_steps), get_lr_sched(self.value, warmup_steps)]

    def zero_grad(self):
        self.actor.zero_grad(set_to_none=True)
        self.value.zero_grad(set_to_none=True)

    def step(self):
        metrics = {}
        for name in ("actor", "value"):
            norm = nn.utils.clip_grad_norm_(getattr(self.model, name).parameters(), self.model.cfg.max_grad_norm,
                                           error_if_nonfinite=True)
            metrics[f"{name}_grad_norm"] = norm.detach()
        self.actor.step()
        self.value.step()
        for name in ("actor", "value"):
            require_finite(**{f"{name}.{key}": p for key, p in getattr(self.model, name).named_parameters()})
        self.model.updates.add_(1)
        for scheduler in self.schedulers:
            scheduler.step()
        return metrics

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), value=self.value.state_dict(),
                    schedulers=[s.state_dict() for s in self.schedulers])

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.value.load_state_dict(state["value"])
        for scheduler, sd in zip(self.schedulers, state["schedulers"]):
            scheduler.load_state_dict(sd)
