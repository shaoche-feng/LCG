"""DreamerV3-style actor-critic controller adapted to the DIAMOND diffusion world model.

A DreamerV3-style ADAPTATION, not an exact DreamerV3 reproduction. Reference:
  Hafner et al., "Mastering Diverse Domains through World Models", arXiv:2301.04104v2 (17 Apr 2024);
  github.com/danijar/dreamerv3 at e01491fad6434b2245a3b8ca201dd7faedcc458c:
    dreamerv3/agent.py        imag_loss, repl_loss, lambda_return, Agent.loss / Agent.train
    dreamerv3/configs.yaml    imag_loss, repl_loss, slowvalue, retnorm, loss_scales, horizon, contdisc
    embodied/jax/utils.py     Normalize (impl=perc), SlowModel
    embodied/jax/outs.py      TwoHot (loss, symmetric pred)
    embodied/jax/heads.py     symexp_twohot bins, outscale
Mapping and every adaptation: docs/experiments/dreamerv3_controller.md.

Kept from the reviewed Gaussian controller (PMPOBeta, actor_distribution=tanh_gaussian): visual
encoder + MLP heads on the 4-frame stack, tanh-transformed diagonal Gaussian actor, action bounds,
deterministic evaluation rule, transformed-Gaussian log-density and entropy, private checkpointed
imagination RNG, repeated-final-frame imagined history (env_loop) and all diagnostics.
Replaced: the PMPO actor objective (sign groups + reference-policy KL; no prior actor exists here)
and the scalar MSE critic.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Optional

import torch
from torch import nn
from torch.distributions import kl_divergence

from data import Batch, BatchSampler, Dataset, collate_segments_to_batch
from .actor_critic import ActorCriticOutput, compute_lambda_returns
from .pmpo_beta import PMPOBeta, PMPOBetaConfig, PMPOOptimizers, VisualHead, require_finite, summary_statistics


# ----------------------------------------------------------------------------------------------
# Reference building blocks (pure functions; unit-tested against explicit calculations)
# ----------------------------------------------------------------------------------------------

def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
    return torch.sign(x) * torch.expm1(x.abs())


def symexp_bins(num_bins: int = 255) -> torch.Tensor:
    """heads.py symexp_twohot: symexp(linspace(-20, 0)) mirrored; raw-space bin locations."""
    if num_bins % 2 == 1:
        half = symexp(torch.linspace(-20, 0, (num_bins - 1) // 2 + 1, dtype=torch.float32))
        return torch.cat([half, -half[:-1].flip(0)])
    half = symexp(torch.linspace(-20, 0, num_bins // 2, dtype=torch.float32))
    return torch.cat([half, -half.flip(0)])


def twohot_encode(target: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """outs.TwoHot.loss target construction (no squash: targets live in raw space)."""
    target = target.detach().float()
    n = len(bins)
    below = (bins <= target[..., None]).long().sum(-1) - 1
    above = n - (bins > target[..., None]).long().sum(-1)
    below, above = below.clamp(0, n - 1), above.clamp(0, n - 1)
    equal = below == above
    one = torch.ones_like(target)
    dist_below = torch.where(equal, one, (bins[below] - target).abs())
    dist_above = torch.where(equal, one, (bins[above] - target).abs())
    total = dist_below + dist_above
    weight_below, weight_above = dist_above / total, dist_below / total
    return (nn.functional.one_hot(below, n) * weight_below[..., None]
            + nn.functional.one_hot(above, n) * weight_above[..., None])


def twohot_loss(logits: torch.Tensor, target: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against the two-hot target; per element (no reduction)."""
    log_pred = logits.float() - torch.logsumexp(logits.float(), -1, keepdim=True)
    return -(twohot_encode(target, bins) * log_pred).sum(-1)


def twohot_mean(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """outs.TwoHot.pred: expectation with the reference's symmetric summation (exactly 0 when
    the probabilities are uniform and the bins symmetric)."""
    probs = torch.softmax(logits.float(), -1)
    n = probs.shape[-1]
    if n % 2 == 1:
        m = (n - 1) // 2
        p1, p2, p3 = probs[..., :m], probs[..., m:m + 1], probs[..., m + 1:]
        b1, b2, b3 = bins[:m], bins[m:m + 1], bins[m + 1:]
        return (p2 * b2).sum(-1) + ((p1 * b1).flip(-1) + p3 * b3).sum(-1)
    p1, p2, b1, b2 = probs[..., :n // 2], probs[..., n // 2:], bins[:n // 2], bins[n // 2:]
    return ((p1 * b1).flip(-1) + p2 * b2).sum(-1)


def imagination_weights(end: torch.Tensor, gamma: float) -> torch.Tensor:
    """agent.py imag_loss with contdisc=True: weight = cumprod(con) over states 0..H-1, where the
    reference continuation con_i = (1 - 1/horizon) * P(continue at state i) and state 0 is the
    replay start. Here con_0 = gamma (the start is a real, non-terminal replay state) and
    con_{t+1} = gamma * (1 - end_t) with end_t the world model's sampled termination of
    transition t. Returns (B, H): weight_t = gamma^(t+1) * prod_{i<t} (1 - end_i)."""
    con = gamma * (1 - end.float())
    start = torch.full_like(con[:, :1], gamma)
    return torch.cumprod(torch.cat([start, con[:, :-1]], 1), 1)


def replay_lambda_return(rew, term, last, boot, gamma, lam):
    """agent.py lambda_return as used by repl_loss (val argument unused there), re-indexed to this
    codebase's transition convention. Replay states t = 0..K-1; rew[t], term[t], last[t] describe
    the transition leaving state t (reference rew/term/last at index t+1); boot[t] is the
    imagination lambda-return started at replay state t. Returns R_t for t = 0..K-2:
      R_{K-1} = boot_{K-1};  R_t = rew_t + live_t * ((1 - cont_t) * boot_{t+1} + cont_t * R_{t+1}),
      live_t = gamma * (1 - term_t),  cont_t = lam * (1 - last_t)."""
    K = boot.shape[1]
    live = gamma * (1 - term[:, :K - 1].float())
    cont = lam * (1 - last[:, :K - 1].float())
    rets = [boot[:, -1]]
    for t in reversed(range(K - 1)):
        rets.append(rew[:, t] + live[:, t] * ((1 - cont[:, t]) * boot[:, t + 1] + cont[:, t] * rets[-1]))
    return torch.stack(list(reversed(rets))[:-1], 1)


def dreamer_actor_loss(logp, entropy, advantage_normed, weight, entropy_coef):
    """agent.py imag_loss policy term, mean-reduced as Agent.loss does (mean over time, then batch,
    = mean over all elements): mean( sg(weight) * -(logp * sg(adv) + actent * entropy) )."""
    return (weight.detach() * -(logp * advantage_normed.detach() + entropy_coef * entropy)).mean()


# ----------------------------------------------------------------------------------------------
# Replay windows: consecutive imagination starts per replay segment (needed by repl_loss)
# ----------------------------------------------------------------------------------------------

class ReplayStartSampler(BatchSampler):
    """BatchSampler's own segment placement (padding only before an episode's start, same RNG
    draws), for segments of window + starts - 1 frames; the stop is raised to at least `starts`
    so that every imagination start frame is a real frame (never padding)."""

    def __init__(self, dataset, segments, window, starts, sample_weights=None):
        super().__init__(dataset, 0, 1, segments, window + starts - 1, sample_weights, can_sample_beyond_end=False)
        self.window, self.starts = window, starts

    def sample(self):
        ids = super().sample()
        out = []
        for i in ids:
            stop = max(i.stop, self.starts)
            if stop > self.dataset.lengths[i.episode_id]:
                raise ValueError("Replay episode shorter than the number of imagination starts")
            out.append(type(i)(i.episode_id, stop - self.seq_length, stop))
        return out


def unfold_windows(batch: Batch, window: int, starts: int) -> Batch:
    """(B, window+starts-1) segments -> (B*starts, window) prompts, segment-major: prompt b*starts+k
    ends at segment frame window-1+k (the k-th consecutive replay start of segment b)."""
    def cut(x):
        return torch.stack([x[:, k:k + window] for k in range(starts)], 1).flatten(0, 1)
    return Batch(cut(batch.obs), cut(batch.act), cut(batch.rew), cut(batch.end), cut(batch.trunc),
                 cut(batch.mask_padding), [i for i in batch.info for _ in range(starts)],
                 [s for s in batch.segment_ids for _ in range(starts)])


class ReplayWindowLoader:
    """Stands in for the actor-critic DataLoader inside WorldModelEnv. Each new iterator (one per
    PMPO-style prompt block, i.e. per controller update) yields unfolded prompt batches; the FIRST
    segment batch of each iterator -- the one whose 32 prompts WorldModelEnv.reset() consumes in
    order -- is kept as `replay` for the replay-value loss. Single-process, like the original."""

    num_workers = 0

    def __init__(self, dataset, segments, window, starts, sample_weights=None):
        self.dataset = dataset
        # WorldModelEnv read num_envs (= segments * starts prompts) from the original loader at
        # construction; this sampler draws `segments` segments per batch.
        self.batch_sampler = ReplayStartSampler(dataset, segments, window, starts, sample_weights)
        self.window, self.starts, self.segments = window, starts, segments
        self.generator = None  # set by WorldModelEnv.restart_initial_conditions; sampling uses numpy
        self.replay, self.blocks = None, 0

    def __iter__(self):
        first = True
        while True:
            ids = self.batch_sampler.sample()
            batch = collate_segments_to_batch([self.dataset[i] for i in ids])
            if first:
                self.replay, first = batch, False
                self.blocks += 1
            yield unfold_windows(batch, self.window, self.starts)


# ----------------------------------------------------------------------------------------------
# Controller
# ----------------------------------------------------------------------------------------------

@dataclass
class DreamerV3ControllerConfig(PMPOBetaConfig):
    actor_distribution: str = "tanh_gaussian"
    # configs.yaml horizon: 333 -> discount 1 - 1/333 = 0.997; imag_loss/repl_loss lam 0.95.
    gamma: float = 0.997
    lambda_: float = 0.95
    # imag_loss actent; reused field name so the inherited validation applies.
    entropy_coef: float = 3e-4
    value_bins: int = 255
    # retnorm: {impl: perc, rate: 0.01, limit: 1.0, perclo: 5.0, perchi: 95.0, debias: False}
    retnorm_rate: float = 0.01
    retnorm_limit: float = 1.0
    retnorm_perclo: float = 5.0
    retnorm_perchi: float = 95.0
    # slowvalue: {rate: 0.02, every: 1}; imag_loss/repl_loss slowreg 1.0, slowtar False.
    slow_value_rate: float = 0.02
    slow_value_every: int = 1
    slowreg: float = 1.0
    # loss_scales: policy 1.0, value 1.0, repval 0.3
    policy_loss_scale: float = 1.0
    value_loss_scale: float = 1.0
    repval_loss_scale: float = 0.3
    # Consecutive replay states imagined from per segment; prompts per update = segments * starts
    # (= actor_critic.training.batch_size). Adaptation of imag_last=0 (all replay states).
    replay_starts: int = 4


class DreamerV3Controller(PMPOBeta):
    def __init__(self, cfg: DreamerV3ControllerConfig):
        if not isinstance(cfg, DreamerV3ControllerConfig):
            raise TypeError("DreamerV3Controller requires DreamerV3ControllerConfig")
        if cfg.actor_distribution != "tanh_gaussian":
            raise ValueError("The DreamerV3-style controller keeps the tanh-Gaussian actor")
        if cfg.value_bins < 3 or not 0 < cfg.slow_value_rate < 0.5 or cfg.slow_value_every < 1:
            raise ValueError("Invalid value bins or slow-critic schedule")
        if not (0 < cfg.retnorm_rate <= 1 and cfg.retnorm_limit > 0 and 0 <= cfg.retnorm_perclo < cfg.retnorm_perchi <= 100):
            raise ValueError("Invalid return-normalization settings")
        if cfg.replay_starts < 2 or min(cfg.slowreg, cfg.policy_loss_scale, cfg.value_loss_scale, cfg.repval_loss_scale) < 0:
            raise ValueError("Invalid replay starts or loss scales")
        super().__init__(cfg)
        # PMPO's periodically refreshed reference actor does not exist in this controller.
        # (initial_actor stays: a frozen diagnostic snapshot for kl_initial, never in any loss.)
        del self.prior_actor
        stacked = replace(cfg, img_channels=cfg.img_channels * cfg.frame_stack)
        self.value = VisualHead(stacked, cfg.value_hidden_dims, cfg.value_bins)
        # heads.py value outscale 0.0: zero output layer -> uniform bins -> value exactly 0.
        nn.init.zeros_(self.value.head[-1].weight)
        nn.init.zeros_(self.value.head[-1].bias)
        # SlowModel initializes from the source on first use, i.e. an exact copy.
        self.slow_value = deepcopy(self.value).requires_grad_(False)
        self.register_buffer("value_bin_locations", symexp_bins(cfg.value_bins), persistent=False)
        self.register_buffer("retnorm_lo", torch.zeros(()))
        self.register_buffer("retnorm_hi", torch.zeros(()))
        self.register_buffer("slow_value_count", torch.zeros((), dtype=torch.long))
        self.replay_loader = None

    # -- value -------------------------------------------------------------------------------
    def value_mean(self, logits):
        return twohot_mean(logits, self.value_bin_locations)

    def predict_act_value(self, obs):
        if obs.ndim != 4 or obs.shape[1] != self.cfg.img_channels * self.frame_stack:
            raise ValueError("The controller requires the exact channel-concatenated four-frame state")
        return ActorCriticOutput(self.actor(obs), self.value_mean(self.value(obs)))

    @torch.no_grad()
    def update_slow_value(self):
        """SlowModel.update: mix = rate on every `every`-th call (count before increment)."""
        mix = self.cfg.slow_value_rate if self.slow_value_count.item() % self.cfg.slow_value_every == 0 else 0.0
        for slow, fast in zip(self.slow_value.state_dict().values(), self.value.state_dict().values()):
            if slow.is_floating_point():
                slow.mul_(1 - mix).add_(fast, alpha=mix)
        self.slow_value_count.add_(1)

    # -- return normalization -----------------------------------------------------------------
    @torch.no_grad()
    def update_return_norm(self, returns):
        """Normalize(impl=perc, debias=False).update: EMA of the 5th/95th percentiles (linear
        interpolation, as jnp.percentile), incorporating `rate` of each new estimate."""
        x = returns.detach().float().flatten()
        q = torch.quantile(x, x.new_tensor([self.cfg.retnorm_perclo, self.cfg.retnorm_perchi]) / 100)
        r = self.cfg.retnorm_rate
        self.retnorm_lo.mul_(1 - r).add_(r * q[0])
        self.retnorm_hi.mul_(1 - r).add_(r * q[1])

    def return_scale(self):
        """Normalize.stats (perc): max(limit, hi - lo); the offset is unused by the advantage."""
        return torch.clamp(self.retnorm_hi - self.retnorm_lo, min=self.cfg.retnorm_limit).detach()

    # -- training wiring ------------------------------------------------------------------------
    def setup_training(self, rl_env):
        super().setup_training(rl_env)
        loader = rl_env.data_loader
        dataset = getattr(loader, "dataset", None)
        if not isinstance(dataset, Dataset):
            raise ValueError("The replay-value loss needs the replay Dataset behind the actor-critic loader")
        prompts, window = rl_env.num_envs, loader.batch_sampler.seq_length
        if window < self.frame_stack:
            raise ValueError("Replay prompts must hold at least frame_stack frames")
        if prompts % self.cfg.replay_starts:
            raise ValueError("actor_critic.training.batch_size must be a multiple of replay_starts")
        self.replay_loader = ReplayWindowLoader(dataset, prompts // self.cfg.replay_starts, window,
                                                self.cfg.replay_starts, loader.batch_sampler.sample_weights)
        rl_env.data_loader = self.replay_loader

    def replay_start_rows(self, segments):
        """Imagination row of replay state t of segment s: s * replay_starts + t (unfold_windows order)."""
        return torch.arange(segments * self.cfg.replay_starts, device=self.device).view(segments, self.cfg.replay_starts)

    def replay_inputs(self, replay: Batch):
        """Controller input for replay states t = 0..K-2 of every segment, flattened segment-major:
        the state's own frame repeated frame_stack times. This is EXACTLY the input from which the
        imagination rollout started at that replay state produced its first prediction (the reviewed
        repeated-final-frame initialization in env_loop), so each replay critic prediction and the
        bootstrap generated from that state's rollout share one input representation. The world
        model's 4-frame real context (prompt) is untouched."""
        K, W = self.cfg.replay_starts, self.replay_loader.window
        starts = replay.obs[:, W - 1:W - 1 + K - 1].to(self.device)  # (S, K-1, C, H, W), never padding
        return starts.flatten(0, 1).repeat(1, self.frame_stack, 1, 1)

    def replay_value_loss(self, replay: Batch, boot: torch.Tensor, start_states: Optional[torch.Tensor] = None):
        """agent.py repl_loss on real replay states 0..K-2 of each segment, bootstrapped at every
        state by the imagination return started there (slowtar=False, valnorm none).
        start_states: the stored imagination states at step 0, (S*K, C*frame_stack, H, W); when given,
        the replay inputs must equal them element for element (guard against any index or history
        mismatch)."""
        c, K, W = self.cfg, self.cfg.replay_starts, self.replay_loader.window
        replay = replay.to(self.device)
        S = replay.obs.shape[0]
        boot = boot.detach().view(S, K)
        stacks = self.replay_inputs(replay)
        if start_states is not None:
            rows = self.replay_start_rows(S)[:, :K - 1].flatten()
            if not torch.equal(stacks, start_states[rows]):
                raise RuntimeError("Replay-value inputs differ from the imagination start states")
        sl = slice(W - 1, W - 1 + K)
        rew, term, trunc = replay.rew[:, sl].float(), replay.end[:, sl].float(), replay.trunc[:, sl].float()
        valid = replay.mask_padding[:, sl]
        last = torch.clamp(term + trunc, max=1)
        targets = replay_lambda_return(rew, term, last, boot, c.gamma, c.lambda_)  # (S, K-1)
        # A transition truncated by the time limit has no stored successor: no target (cf. ~is_last).
        weight = ((1 - trunc[:, :K - 1]) * valid[:, :K - 1].float() * valid[:, 1:].float()).detach()
        logits = self.value(stacks)
        with torch.no_grad():
            slow_pred = self.value_mean(self.slow_value(stacks))
        bins = self.value_bin_locations
        per = twohot_loss(logits, targets.flatten(), bins) + c.slowreg * twohot_loss(logits, slow_pred, bins)
        loss = (weight.flatten() * per).mean()
        stats = dict(replay_return_mean=targets.mean(), replay_boot_mean=boot.mean(),
                     replay_value_mean=self.value_mean(logits.detach()).mean(), replay_weight_mean=weight.mean(),
                     replay_reward_mean=rew[:, :K - 1].mean())
        return loss, stats

    # -- update ---------------------------------------------------------------------------------
    def forward(self):
        c = self.cfg
        blocks_before = self.replay_loader.blocks
        obs, act, rew, end, trunc, _, values_rollout, bootstrap, infos = self.collect_imagination()
        if self.replay_loader.blocks != blocks_before + 1:
            raise RuntimeError("Replay batch is not aligned with this update's imagination starts")
        replay = self.replay_loader.replay
        if self.intrinsic_reward_fn is not None:
            rew = self.intrinsic_reward_fn(infos, rew).detach()
        B, H = rew.shape
        # Lambda returns from the online critic's values (slowtar=False); horizon truncation
        # bootstraps from V(final observation), only a sampled end is terminal.
        targets = compute_lambda_returns(rew, end, trunc, bootstrap, c.gamma, c.lambda_)
        weight = imagination_weights(end, c.gamma)
        if self.training:
            self.update_return_norm(targets)
        scale = self.return_scale()
        advantage = (targets - values_rollout) / scale  # advnorm impl none
        flat_obs = obs.flatten(0, 1).detach()
        raw = self.actor(flat_obs)
        logp, entropy = self.log_prob_and_entropy(raw, act.flatten(0, 1))
        policy_loss = dreamer_actor_loss(logp, entropy, advantage.flatten(), weight.flatten(), c.entropy_coef)
        logits = self.value(flat_obs)
        bins = self.value_bin_locations
        with torch.no_grad():
            slow_pred = self.value_mean(self.slow_value(flat_obs))
        value_target_loss = twohot_loss(logits, targets.flatten(), bins)
        slowreg_loss = twohot_loss(logits, slow_pred, bins)
        value_loss = (weight.flatten().detach() * (value_target_loss + c.slowreg * slowreg_loss)).mean()
        repval_loss, replay_stats = self.replay_value_loss(replay, targets[:, 0], start_states=obs[:, 0])
        actor_total = c.policy_loss_scale * policy_loss
        critic_total = c.value_loss_scale * value_loss + c.repval_loss_scale * repval_loss
        values = self.value_mean(logits.detach())
        require_finite(targets=targets, values=values, actor_loss=actor_total, value_loss=critic_total)

        metrics = self.diagnostics(raw.detach(), act.flatten(0, 1), logp.detach(), entropy.detach())
        with torch.no_grad():
            kl_initial = kl_divergence(self.distribution(raw.detach()),
                                       self.distribution(self.initial_actor(flat_obs))).sum(-1)
            centered_value = values - values.mean()
            centered_target = targets.flatten() - targets.mean()
            correlation = (centered_value * centered_target).mean() / (
                centered_value.square().mean() * centered_target.square().mean()).sqrt().clamp_min(1e-12)
            metrics.update(
                loss_actor=actor_total.detach(), loss_actor_total=actor_total.detach(), loss_policy=policy_loss.detach(),
                loss_entropy=(weight.flatten() * -c.entropy_coef * entropy.detach()).mean(),
                loss_value=critic_total.detach(), loss_value_imag=value_loss.detach(), loss_repval=repval_loss.detach(),
                loss_value_target_ce=(weight.flatten() * value_target_loss.detach()).mean(),
                loss_slowreg_ce=(weight.flatten() * slowreg_loss.detach()).mean(),
                kl_initial=kl_initial.mean(),
                return_scale=scale, retnorm_lo=self.retnorm_lo.clone(), retnorm_hi=self.retnorm_hi.clone(),
                return_mean=targets.mean(), return_std=targets.std(unbiased=False),
                value_mean=values.mean(), value_std=values.std(unbiased=False), slow_value_mean=slow_pred.mean(),
                slow_value_gap=(values - slow_pred).abs().mean(),
                weight_mean=weight.mean(), weight_last=weight[:, -1].mean(), end_fraction=end.float().mean(),
                positive_fraction=(advantage >= 0).float().mean(), negative_fraction=(advantage < 0).float().mean(),
                advantage_normed_abs_mean=advantage.abs().mean(),
                value_target_correlation=correlation, value_bias=(values - targets.flatten()).mean(),
                value_abs_max=values.abs().max(), return_abs_max=targets.abs().max(),
                rollout_length=float(H), all_finite=1.0, **replay_stats)
            metrics.update(summary_statistics(advantage, "advantage"))
            metrics["advantage_zero_fraction"] = (advantage == 0).float().mean()
            metrics["advantage_near_zero_fraction"] = (advantage.abs() <= 1e-6).float().mean()
            per_dim_entropy = self.entropy_per_dim(self.distribution(raw.detach()))
            for d in range(per_dim_entropy.shape[-1]):
                metrics[f"policy_entropy_dim_{d}"] = per_dim_entropy[:, d].mean()
            frames = flat_obs.reshape(-1, self.frame_stack, self.cfg.img_channels, self.cfg.img_size, self.cfg.img_size)
            difference = (frames[:, 1:] - frames[:, :-1]).abs().flatten(1).mean(1)
            metrics.update(stack_temporal_difference_mean=difference.mean(),
                           stack_temporal_variation_fraction=(difference > 0).float().mean())
            if self._fixed_obs is None:
                self._fixed_obs = flat_obs[:32].clone()
            fixed_mean = self.sample_action(self.actor(self._fixed_obs.to(self.device)), deterministic=True)
            for d, std in enumerate(fixed_mean.std(0, unbiased=False)):
                metrics[f"fixed_policy_mean_state_std_{d}"] = std
            metrics.update(self.temporal_diagnostics(self._fixed_obs.to(self.device), "fixed"))
            require_finite(**{k: torch.as_tensor(v) for k, v in metrics.items()})
        # Disjoint graphs: the actor optimizer sees only actor parameters, the critic optimizer only
        # value parameters; the slow critic and all rollout tensors carry no gradient.
        return actor_total + critic_total, metrics


class DreamerV3Optimizers(PMPOOptimizers):
    """PMPO's two Adams + warm-up schedulers (unchanged), then the slow-critic EMA after every step
    (Agent.train: self.opt(...) then self.slowval.update())."""

    def step(self):
        metrics = super().step()
        self.model.update_slow_value()
        return metrics


DreamerV3Controller.optimizers_class = DreamerV3Optimizers
