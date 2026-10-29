"""Sign-only PMPO on fresh DIAMOND experience; no differentiable imagination."""
from copy import deepcopy
from dataclasses import dataclass, replace
from contextlib import contextmanager
import math
import random
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta, Normal, kl_divergence
from torch.nn import functional as F

from .actor_critic import ActorCriticEncoder, ActorCriticOutput, compute_lambda_returns
from coroutines.env_loop import make_env_loop
from envs import WorldModelEnv
from utils import RegionTimer, timed


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
    # Actor objective: "sign" (default, Eq. 11 without magnitude weighting) or
    # "advantage_weighted" (each log-probability weighted by min(cap, |A| / mean_group |A|)
    # within its sign group; see actor_sample_weights). KL, entropy and critic are unchanged.
    actor_objective: str = "sign"
    advantage_weight_cap: float = 5.0  # used only by advantage_weighted; 5 is not a validated optimum
    # Actor distribution (from the reference controller, stage3-gaussian-h10-emacritic). "beta"
    # (the dataclass default, so configs saved before this field stay Beta) is the Beta actor.
    # "tanh_gaussian" is the Gaussian-PMPO controller (not an exact DreamerV3 implementation): a
    # diagonal Gaussian over pre-tanh actions with std = 0.1 + 0.9 * sigmoid(raw_std), squashed by
    # tanh and mapped affinely to the action bounds. The std floor applies before tanh; it does not
    # guarantee a minimum spread of environment actions.
    actor_distribution: str = "beta"
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
    log_diagnostics: bool = False  # print every update's metrics as JSON on the terminal
    # Detailed per-update metrics (logged under actor_critic/train/debug/). False: the debug-only
    # quantities are not computed at all. Logging only: losses, updates and state are identical.
    debug_metrics: bool = False
    # Slow EMA critic regularization (reference controller), adapted to the scalar critic. With
    # slow_critic_coef > 0 an EMA copy of the critic (outside the optimizer, no gradients) is kept,
    # value_loss += slow_critic_coef * MSE(V(s), sg(V_slow(s))) on the same states and mean
    # reduction, and after every critic optimizer step slow = (1 - rate) * slow + rate * online.
    # The ONLINE critic still provides lambda-return bootstraps and advantages. 0.0 (default)
    # creates no EMA critic and leaves the learning path and state_dict unchanged.
    slow_critic_coef: float = 0.0
    slow_critic_rate: float = 0.02
    # Policy frame history at an imagined reset (reference controller convention). "repeat_anchor":
    # the policy's four-frame stack repeats the imagined start frame (the last replay prompt frame),
    # exactly as a real reset repeats its first observation; the world model is still conditioned
    # on the real replay prompt (frames and actions). "replay_prompt" (the dataclass default, so
    # configs saved before this field keep it): the stack starts from the latest prompt frames.
    # Real-environment histories always repeat the reset observation.
    imagination_history: str = "replay_prompt"


# Logged metric groups (actor_critic/train/<group>/<name>); every other metric is debug-only.
CORE_METRICS = ("loss_actor", "loss_value", "kl_prior", "policy_entropy", "return_mean", "value_mean",
                "value_target_correlation", "positive_fraction", "actor_grad_norm", "value_grad_norm",
                "num_batch_train_actor_critic")
EMA_METRICS = ("loss_value_target", "loss_slowreg", "slow_value_gap")
DISTRIBUTION_METRIC_PREFIXES = dict(tanh_gaussian=("pretanh_", "tanh_inverse_limited_fraction", "policy_conditional_std_"),
                                    beta=("alpha_", "beta_", "policy_conditional_std_"))
DISTRIBUTION_GROUPS = dict(tanh_gaussian="gaussian", beta="beta")
# Renamed in their groups: flat model name -> path within actor_critic/train/.
RENAMED_METRICS = {"weight_clipped_fraction_positive": "weights/clipped_fraction_positive",
                   "weight_clipped_fraction_negative": "weights/clipped_fraction_negative",
                   **{f"intrinsic_{kind}_reward_{stat}": f"intrinsic/{kind}_reward_{stat}"
                      for kind in ("raw", "scaled") for stat in ("mean", "max")}}


def metric_path(name, actor_distribution):
    """W&B path of a flat controller metric below actor_critic/train/."""
    if name in CORE_METRICS:
        return f"core/{name}"
    if name in EMA_METRICS:
        return f"EMA/{name}"
    if name in RENAMED_METRICS:
        return RENAMED_METRICS[name]
    if name.startswith(DISTRIBUTION_METRIC_PREFIXES[actor_distribution]):
        return f"{DISTRIBUTION_GROUPS[actor_distribution]}/{name}"
    return f"debug/{name}"


def flat_metric_name(path):
    """Inverse of metric_path: the flat controller name of a path below actor_critic/train/."""
    for name, renamed in RENAMED_METRICS.items():
        if path == renamed:
            return name
    return path.split("/", 1)[1]


ACTOR_DISTRIBUTIONS = ("beta", "tanh_gaussian")
IMAGINATION_HISTORIES = ("replay_prompt", "repeat_anchor")
GAUSSIAN_STD_MIN, GAUSSIAN_STD_SPAN = 0.1, 0.9
# Actions are stored as float32 in environment units, so tanh values closer to +-1 than this
# cannot be inverted faithfully (|pre-tanh| > ~8.4). Scoring evaluates such actions at the
# nearest invertible pre-image -- still with the full transformed density, never as an
# untransformed Gaussian -- and diagnostics log how often this happens.
TANH_INVERSE_LIMIT = 1 - 1e-7
GAUSS_HERMITE_NODES = 32
# numpy.polynomial.hermite.hermgauss(32), the reference's rule, as exact float64 constants: the
# same values on numpy 1.26.0 (kingplus0) and 1.26.4 (local), but computing them calls LAPACK,
# which aborts on Windows once torch has loaded its own OpenMP runtime (libiomp5md "already
# initialized"). A test checks these against hermgauss(32).
GAUSS_HERMITE_32 = (
    (-7.125813909830728, -6.409498149269661, -5.812225949515914, -5.2755509865158805, -4.777164503502596,
     -4.305547953351199, -3.853755485471445, -3.417167492818571, -2.992490825002374, -2.5772495377323175,
     -2.169499183606112, -1.7676541094632015, -1.3703764109528718, -0.9765004635896828, -0.5849787654359324,
     -0.19484074156939934, 0.19484074156939934, 0.5849787654359324, 0.9765004635896828, 1.3703764109528718,
     1.7676541094632015, 2.169499183606112, 2.5772495377323175, 2.992490825002374, 3.417167492818571,
     3.853755485471445, 4.305547953351199, 4.777164503502596, 5.2755509865158805, 5.812225949515914,
     6.409498149269661, 7.125813909830728),
    (7.310676427384096e-23, 9.231736536518258e-19, 1.1973440170928503e-15, 4.2150102113264155e-13,
     5.933291463396676e-11, 4.098832164770879e-09, 1.5741677925455882e-07, 3.650585129562378e-06,
     5.416584061819991e-05, 0.000536268365527972, 0.003654890326654426, 0.017553428831573438,
     0.06045813095591269, 0.15126973407664232, 0.27745814230252996, 0.37523835259280247,
     0.37523835259280247, 0.27745814230252996, 0.15126973407664232, 0.06045813095591269,
     0.017553428831573438, 0.003654890326654426, 0.000536268365527972, 5.416584061819991e-05,
     3.650585129562378e-06, 1.5741677925455882e-07, 4.098832164770879e-09, 5.933291463396676e-11,
     4.2150102113264155e-13, 1.1973440170928503e-15, 9.231736536518258e-19, 7.310676427384096e-23),
)


def log1m_tanh_sq(x):
    """log(1 - tanh(x)^2), numerically stable for large |x|."""
    return 2 * (math.log(2) - x - F.softplus(-2 * x))


def controller_settings(cfg):
    """Settings a controller checkpoint was trained with; restoring with other values would
    reinterpret its weights or change its learning path (checked before anything is loaded)."""
    slow = dict(coef=float(cfg.slow_critic_coef), rate=float(cfg.slow_critic_rate)) if cfg.slow_critic_coef > 0 else None
    return dict(actor_distribution=cfg.actor_distribution, slow_critic=slow,
                imagination_history=cfg.imagination_history, imagination_horizon=int(cfg.imagination_horizon))


# Controller state saved before the settings were recorded: the Beta actor, no slow critic and
# replay-prompt histories (the horizon was not recorded). Never reinterpreted as anything else.
LEGACY_CONTROLLER = dict(actor_distribution="beta", slow_critic=None, imagination_history="replay_prompt",
                         imagination_horizon=None)


def check_controller_state(extra_state, cfg, source="PMPO state"):
    """Refuse a controller state (its _extra_state) trained with other controller settings."""
    saved = (extra_state or {}).get("controller", LEGACY_CONTROLLER)
    expected = controller_settings(cfg)
    if saved["imagination_horizon"] is None:  # not recorded: the other settings must still match
        expected = dict(expected, imagination_horizon=None)
    if saved != expected:
        legacy = " (saved before controller settings were recorded: a Beta controller)" if saved is LEGACY_CONTROLLER else ""
        raise ValueError(f"{source} was trained with controller {saved}{legacy}; the config gives {expected}. "
                         "It cannot be loaded or resumed with a different controller; start a fresh controller.")


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


ACTOR_OBJECTIVES = ("sign", "advantage_weighted")


def pmpo_loss(log_prob, advantage, alpha=0.5, objective="sign", weight_cap=5.0):
    """Eq. 11: (1-alpha) mean_negative(log pi) - alpha mean_positive(log pi).

    Empty groups contribute zero, without renormalizing the surviving weight.
    Exactly zero belongs to the positive group. objective="sign": no magnitude weighting.
    objective="advantage_weighted": each log-probability is multiplied by its detached
    weight (actor_sample_weights) inside the same group means (sample-count denominators;
    no renormalization after clipping).
    """
    if log_prob.shape != advantage.shape or not 0 <= alpha <= 1 or not log_prob.numel():
        raise ValueError("Invalid PMPO shapes, empty batch, or mixing coefficient")
    if objective not in ACTOR_OBJECTIVES:
        raise ValueError(f"Unknown PMPO actor objective {objective!r}")
    require_finite(log_prob=log_prob, advantage=advantage)
    positive = advantage.detach() >= 0
    negative = ~positive
    if objective == "advantage_weighted":
        log_prob = actor_sample_weights(advantage, weight_cap) * log_prob
    pos = log_prob.masked_fill(~positive, 0).sum() / positive.sum().clamp_min(1)
    neg = log_prob.masked_fill(~negative, 0).sum() / negative.sum().clamp_min(1)
    return (1 - alpha) * neg - alpha * pos


def actor_sample_weights(advantage, weight_cap=5.0):
    """Detached weights w_i = min(cap, |A_i| / max(mean_{j in g} |A_j|, 1e-8)) within each
    sign group g (A >= 0 positive, A < 0 negative). All-zero advantages get weight 0."""
    magnitude = advantage.detach().abs()
    positive = advantage.detach() >= 0
    weights = torch.zeros_like(magnitude)
    for group in (positive, ~positive):
        if group.any():
            scale = magnitude[group].mean().clamp_min(1e-8)
            weights[group] = (magnitude[group] / scale).clamp(max=weight_cap)
    return weights


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
        if cfg.slow_critic_coef < 0 or not 0 < cfg.slow_critic_rate <= 1:
            raise ValueError("Invalid slow critic coefficient or EMA rate")
        if cfg.beta_kl < 0 or cfg.entropy_coef < 0 or cfg.prior_refresh_interval < 1 or cfg.imagination_horizon < 1:
            raise ValueError("Invalid KL/entropy coefficient or rollout/block length")
        if cfg.actor_objective not in ACTOR_OBJECTIVES:
            raise ValueError(f"actor_objective must be one of {ACTOR_OBJECTIVES}, got {cfg.actor_objective!r}")
        cap = cfg.advantage_weight_cap
        if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not np.isfinite(cap) or cap <= 0:
            raise ValueError(f"advantage_weight_cap must be a finite positive number, got {cap!r}")
        if cfg.actor_distribution not in ACTOR_DISTRIBUTIONS:
            raise ValueError(f"Unknown actor distribution {cfg.actor_distribution!r}")
        if cfg.imagination_history not in IMAGINATION_HISTORIES:
            raise ValueError(f"imagination_history must be one of {IMAGINATION_HISTORIES}, got {cfg.imagination_history!r}")
        # Gauss-Hermite rule for expectations under the pre-tanh Gaussian (entropy Jacobian term,
        # transformed action spread): deterministic, differentiable, consumes no RNG. Non-persistent,
        # so state_dicts and checkpoints are unchanged.
        nodes, weights = np.array(GAUSS_HERMITE_32[0]), np.array(GAUSS_HERMITE_32[1])
        self.register_buffer("_gh_nodes", torch.tensor(nodes * math.sqrt(2), dtype=torch.float32), persistent=False)
        self.register_buffer("_gh_weights", torch.tensor(weights / math.sqrt(math.pi), dtype=torch.float32), persistent=False)
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
        if cfg.slow_critic_coef > 0:
            # Exact copy of the freshly initialized online critic; never optimized (no gradients).
            self.slow_value = deepcopy(self.value).requires_grad_(False)
        # Logging only (not a controller setting): the PMPO diagnostic mode turns it on.
        self.debug_metrics = bool(cfg.debug_metrics)
        self.intrinsic_reward_fn = None
        self.scoring_timer: Optional[RegionTimer] = None  # monitoring only: time in intrinsic_reward_fn
        self.rl_env = None
        self._fixed_obs = None
        self._rng = dict(cpu=torch.Generator().manual_seed(seed).get_state(), cuda=None,
                         python=random.Random(seed).getstate(), numpy=np.random.RandomState(seed).get_state())

    @property
    def device(self):
        return self.action_low.device

    def objective_settings(self):
        """The actor objective a checkpoint was trained with; loading must not change it."""
        if self.cfg.actor_objective == "sign":
            return dict(mode="sign")
        return dict(mode=self.cfg.actor_objective, weight_cap=float(self.cfg.advantage_weight_cap))

    def get_extra_state(self):
        return deepcopy(dict(rng=self._rng, fixed_obs=self._fixed_obs, objective=self.objective_settings(),
                             controller=controller_settings(self.cfg)))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Controller settings first: a Beta/Gaussian, slow-critic or frame-history mismatch is
        # refused before any tensor is copied (the Beta and Gaussian heads have equal shapes).
        check_controller_state(state_dict.get("_extra_state"), self.cfg)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def set_extra_state(self, state):
        check_controller_state(state, self.cfg)  # also when loaded as part of a larger module
        saved = state.get("objective", dict(mode="sign"))  # saved before the setting existed: sign-only
        if saved != self.objective_settings():
            raise ValueError(f"PMPO state was trained with actor objective {saved}, the config gives "
                             f"{self.objective_settings()}; it cannot be loaded or resumed with a different objective")
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

    @property
    def gaussian(self):
        return self.cfg.actor_distribution == "tanh_gaussian"

    @property
    def repeats_imagined_anchor(self):
        """Imagined resets start the policy's frame stack from the repeated start frame."""
        return self.cfg.imagination_history == "repeat_anchor"

    def distribution(self, raw):
        """Beta over unit actions, or (tanh_gaussian) the Normal over PRE-tanh actions."""
        require_finite(raw_parameters=raw)
        if self.gaussian:
            mu, raw_std = raw.chunk(2, dim=-1)
            return Normal(mu, GAUSSIAN_STD_MIN + GAUSSIAN_STD_SPAN * torch.sigmoid(raw_std))
        a, b = (F.softplus(raw) + self.cfg.concentration_min).chunk(2, dim=-1)
        return Beta(a, b)

    def to_environment(self, unit):
        # Equivalent to canonical = 2*x-1; low + (canonical+1)*(high-low)/2.
        return self.action_low + unit * (self.action_high - self.action_low)

    def sample_action(self, raw, deterministic=False):
        dist = self.distribution(raw)
        if self.gaussian:
            # Deterministic evaluation rule: tanh(mu), the transformed Gaussian MEAN, mapped to the
            # bounds. This is a rule, not the expectation of the transformed distribution.
            pre_tanh = dist.mean if deterministic else dist.sample()
            unit = (torch.tanh(pre_tanh) + 1) / 2
        else:
            unit = dist.mean if deterministic else dist.sample()
        return self.to_environment(unit).detach()

    def pre_tanh_from_unit(self, unit):
        """Invert unit = (tanh(x) + 1) / 2 in float64 (see TANH_INVERSE_LIMIT)."""
        canonical = (2 * unit.double() - 1).clamp(-TANH_INVERSE_LIMIT, TANH_INVERSE_LIMIT)
        return torch.atanh(canonical).to(unit.dtype)

    def gauss_hermite_expectation(self, dist, fn):
        """E[fn(X)] per element for X ~ dist (Normal), by Gauss-Hermite quadrature; differentiable."""
        x = dist.mean.unsqueeze(-1) + dist.stddev.unsqueeze(-1) * self._gh_nodes
        return (fn(x) * self._gh_weights).sum(-1)

    def entropy_per_dim(self, dist):
        """Differential entropy of the ENVIRONMENT action, per dimension."""
        width = self.action_high - self.action_low
        if self.gaussian:
            # H(a) = H(x) + E[log |da/dx|], da/dx = (1 - tanh(x)^2) * width / 2.
            return dist.entropy() + self.gauss_hermite_expectation(dist, log1m_tanh_sq) + (width / 2).log()
        return dist.entropy() + width.log()

    def log_prob_and_entropy(self, raw, action):
        width = self.action_high - self.action_low
        unit = (action.detach() - self.action_low) / width
        if ((unit < 0) | (unit > 1)).any():
            raise ValueError("Stored action outside environment bounds")
        if self.gaussian:
            dist = self.distribution(raw)
            pre_tanh = self.pre_tanh_from_unit(unit)
            # Change of variables: log p(a) = log N(x) - log(1 - tanh(x)^2) - log(width / 2).
            logp = (dist.log_prob(pre_tanh) - log1m_tanh_sq(pre_tanh) - (width / 2).log()).sum(-1)
            entropy = self.entropy_per_dim(dist).sum(-1)
            require_finite(log_probability=logp, entropy=entropy)
            return logp, entropy
        # Affine float32 roundoff can land exactly on an endpoint. Only move the
        # argument inside numerical support; never clamp log probabilities.
        eps = torch.finfo(unit.dtype).eps
        unit = unit.clamp(eps, 1 - eps)
        dist = self.distribution(raw)
        logp = (dist.log_prob(unit) - width.log()).sum(-1)
        entropy = (dist.entropy() + width.log()).sum(-1)
        require_finite(log_probability=logp, entropy=entropy)
        return logp, entropy

    @torch.no_grad()
    def update_slow_critic(self):
        """slow = (1 - rate) * slow + rate * online, over every floating-point tensor of the critic."""
        rate = self.cfg.slow_critic_rate
        for slow, online in zip(self.slow_value.state_dict().values(), self.value.state_dict().values()):
            if slow.is_floating_point():
                slow.mul_(1 - rate).add_(online, alpha=rate)

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
        if self.intrinsic_reward_fn is None and self.rl_env.rew_end_model is None:
            # Never train on the placeholder reward of a world model without a reward model.
            raise RuntimeError("PMPO has no reward source: set an intrinsic reward hook or a reward model")
        if not c.fixed_prior and self.updates.item() % c.prior_refresh_interval == 0:
            self.prior_actor.load_state_dict(self.actor.state_dict())
        self.prior_actor.eval()
        obs, act, rew, end, trunc, _, old_values, bootstrap, infos = self.collect_imagination()
        if self.intrinsic_reward_fn is not None:
            with timed(self.scoring_timer):
                rew = self.intrinsic_reward_fn(infos, rew).detach()
        targets = compute_lambda_returns(rew, end, trunc, bootstrap, c.gamma, c.lambda_)
        advantage = targets - old_values
        flat_obs = obs.flatten(0, 1).detach()
        raw = self.actor(flat_obs)
        values = self.value(flat_obs).squeeze(-1)
        logp, entropy = self.log_prob_and_entropy(raw, act.flatten(0, 1))
        dist = self.distribution(raw)
        debug = self.debug_metrics
        with torch.no_grad():
            prior = self.distribution(self.prior_actor(flat_obs))
        kl = kl_divergence(dist, prior).sum(-1)
        pmpo_raw_loss = pmpo_loss(logp, advantage.flatten(), c.alpha_pmpo, c.actor_objective, c.advantage_weight_cap)
        kl_loss = c.beta_kl * kl.mean()
        entropy_loss = -c.entropy_coef * entropy.mean()
        actor_loss = pmpo_raw_loss + kl_loss + entropy_loss
        value_loss = F.mse_loss(values, targets.flatten())
        slow_metrics = {}
        if c.slow_critic_coef > 0:
            with torch.no_grad():
                slow_values = self.slow_value(flat_obs).squeeze(-1)
            slowreg_loss = F.mse_loss(values, slow_values)  # same states, same mean reduction
            slow_metrics = dict(loss_value_target=value_loss.detach(), loss_slowreg=slowreg_loss.detach(),
                                slow_value_gap=(values.detach() - slow_values).abs().mean())
            if debug:
                slow_metrics["slow_value_mean"] = slow_values.mean()
            value_loss = value_loss + c.slow_critic_coef * slowreg_loss
        require_finite(targets=targets, values=values, kl=kl, actor_loss=actor_loss, value_loss=value_loss)
        # Logged metrics (metric_path gives their W&B groups). Everything debug-only is computed
        # only with debug_metrics: no extra policy probes, initial-actor KL or parameter
        # distances otherwise. None of it enters the losses or consumes random numbers.
        with torch.no_grad():
            metrics = self.diagnostics(raw.detach(), act.flatten(0, 1), logp.detach(), entropy.detach(), debug=debug)
            centered_value = values.detach() - values.detach().mean()
            centered_target = targets.flatten() - targets.mean()
            correlation = (centered_value * centered_target).mean() / (
                centered_value.square().mean() * centered_target.square().mean()).sqrt().clamp_min(1e-12)
            metrics.update(loss_actor=actor_loss.detach(), loss_value=value_loss.detach(), kl_prior=kl.mean().detach(),
                           policy_entropy=entropy.mean(), return_mean=targets.mean(), value_mean=values.mean().detach(),
                           value_target_correlation=correlation, positive_fraction=(advantage >= 0).float().mean(),
                           **slow_metrics)
            metrics.update(self.objective_diagnostics(advantage.flatten(), debug=debug))
            if debug:
                # Diagnostic only, see initial_actor's docstring above: never added to actor_loss.
                kl_initial = kl_divergence(self.distribution(raw.detach()), self.distribution(self.initial_actor(flat_obs))).sum(-1)
                pos = advantage.flatten() >= 0
                pos_logp = logp.detach().masked_fill(~pos, 0).sum() / pos.sum().clamp_min(1)
                neg_logp = logp.detach().masked_fill(pos, 0).sum() / (~pos).sum().clamp_min(1)
                metrics.update(summary_statistics(advantage, "advantage"))
                metrics.update(kl_initial=kl_initial.mean(), loss_pmpo_raw=pmpo_raw_loss.detach(),
                               loss_entropy=entropy_loss.detach(), loss_kl=c.beta_kl * kl.mean().detach(),
                               positive_log_probability=pos_logp, negative_log_probability=neg_logp,
                               loss_positive=-c.alpha_pmpo * pos_logp, loss_negative=(1 - c.alpha_pmpo) * neg_logp,
                               value_std=values.std(unbiased=False).detach(), return_std=targets.std(unbiased=False),
                               negative_fraction=(advantage < 0).float().mean(),
                               advantage_zero_fraction=(advantage == 0).float().mean(),
                               advantage_near_zero_fraction=(advantage.abs() <= 1e-6).float().mean(),
                               value_bias=(values.detach() - targets.flatten()).mean(),
                               value_abs_max=values.detach().abs().max(), return_abs_max=targets.abs().max())
                per_dim_entropy = self.entropy_per_dim(dist).detach()
                for d in range(per_dim_entropy.shape[-1]):
                    metrics[f"policy_entropy_dim_{d}"] = per_dim_entropy[:, d].mean()
                frames = flat_obs.reshape(-1, self.frame_stack, self.cfg.img_channels, self.cfg.img_size, self.cfg.img_size)
                difference = (frames[:, 1:] - frames[:, :-1]).abs().flatten(1).mean(1)
                metrics.update(stack_temporal_difference_mean=difference.mean(),
                               stack_temporal_variation_fraction=(difference > 0).float().mean())
                # Fixed-state probes: the first 32 states seen while debug metrics are on (a
                # debug-only cache in the extra state; it never affects training).
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
    def objective_diagnostics(self, advantage, debug=True):
        """Per sign group: the fraction of actor sample weights clipped at the cap (0 for
        sign-only); with debug, also |A| mean/max, the weights (1 for sign-only) and the actor's
        movement from its construction-time snapshot."""
        metrics = {}
        positive = advantage >= 0
        weighted = self.cfg.actor_objective == "advantage_weighted"
        weights = actor_sample_weights(advantage, self.cfg.advantage_weight_cap) if weighted else torch.ones_like(advantage)
        for name, group in (("positive", positive), ("negative", ~positive)):
            count = group.sum()
            if count:
                a, w = advantage[group].abs(), weights[group]
                metrics[f"weight_clipped_fraction_{name}"] = (w >= self.cfg.advantage_weight_cap).float().mean()                     if weighted else w.new_zeros(())
                if debug:
                    metrics.update({f"advantage_abs_mean_{name}": a.mean(), f"advantage_abs_max_{name}": a.max(),
                                    f"weight_mean_{name}": w.mean(), f"weight_max_{name}": w.max(),
                                    f"weight_std_{name}": w.std(unbiased=False)})
            else:
                keys = ("weight_clipped_fraction",) + (("advantage_abs_mean", "advantage_abs_max", "weight_mean",
                                                        "weight_max", "weight_std") if debug else ())
                metrics.update({f"{k}_{name}": advantage.new_zeros(()) for k in keys})
        if debug:
            num = sum((p - q).square().sum() for p, q in zip(self.actor.parameters(), self.initial_actor.parameters()))
            den = sum(q.square().sum() for q in self.initial_actor.parameters())
            metrics["actor_rel_distance_from_init"] = (num.sqrt() / den.sqrt().clamp_min(1e-12))
        return metrics

    @torch.no_grad()
    def diagnostics(self, raw, actions, logp=None, entropy=None, debug=True):
        """Distribution parameters and per-dimension conditional spread; with debug (the default
        for direct callers such as the PMPO diagnostic), also per-dimension action and mean-action
        statistics, bound proximity and log-probability/entropy summaries."""
        dist = self.distribution(raw)
        width = self.action_high - self.action_low
        unit = (actions - self.action_low) / width
        metrics = {}
        # Per-state stochastic spread (instrumentation; analytic or quadrature, consumes no RNG):
        # each state's conditional action variance in environment units, averaged over states
        # before the square root. Distinct from policy_mean_state_std_* (variation of mean actions
        # across states) and action_std_* (spread of sampled actions overall).
        if self.gaussian:
            metrics.update(summary_statistics(dist.mean, "pretanh_mean"))
            metrics.update(summary_statistics(dist.stddev, "pretanh_std"))
            metrics["pretanh_std_near_floor_fraction"] = (dist.stddev < GAUSSIAN_STD_MIN + 0.01).float().mean()
            metrics["pretanh_mean_abs_gt_3_fraction"] = (dist.mean.abs() > 3).float().mean()
            metrics["tanh_inverse_limited_fraction"] = ((2 * unit.double() - 1).abs() > TANH_INVERSE_LIMIT).float().mean()
            m1 = self.gauss_hermite_expectation(dist, torch.tanh)
            m2 = self.gauss_hermite_expectation(dist, lambda x: torch.tanh(x) ** 2)
            conditional_var = (m2 - m1 ** 2).clamp_min(0) * (width / 2) ** 2
        else:
            for name, param in (("alpha", dist.concentration1), ("beta", dist.concentration0)):
                metrics.update(summary_statistics(param, name))
                metrics[f"{name}_near_floor_fraction"] = (param < self.cfg.concentration_min + 0.01).float().mean()
            a, b = dist.concentration1, dist.concentration0
            conditional_var = a * b / ((a + b) ** 2 * (a + b + 1)) * width ** 2
        for d in range(actions.shape[-1]):
            metrics[f"policy_conditional_std_{d}"] = conditional_var[:, d].mean().sqrt()
        if not debug:
            return metrics
        # Deterministic evaluation actions: tanh(mu) (Gaussian) or the Beta mean, in env units.
        means = self.to_environment((torch.tanh(dist.mean) + 1) / 2 if self.gaussian else dist.mean)
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
        if getattr(self.model, "slow_value", None) is not None:
            self.model.update_slow_critic()  # after the critic optimizer step
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
