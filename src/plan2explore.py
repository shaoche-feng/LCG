"""Plan2Explore-style lightweight ensemble disagreement on DIAMOND (an adaptation, not an exact
Plan2Explore reproduction; see docs/PLAN2EXPLORE_DESIGN.md).

One diffusion denoiser generates imagined frames. A frozen copy of its activations is the
feature extractor E: a frame o_t is encoded with its real preceding context (frames
o_{t-n..t-1}, actions a_{t-n..t-1}; positions before the episode start are the zero padding
the denoiser is trained with), a fixed reproducible Gaussian probe at feature_sigma, the
denoiser's normal corruption formula and EDM preconditioning, then selected UNet block outputs
pooled to pool_size x pool_size and concatenated. K small predictor heads learn
f_k(z_t, a_t) -> z_{t+1} on standardized features; the intrinsic reward of an imagined
transition (o_t, a_t) is the sample variance (K - 1 convention) of their predicted means,
averaged over feature coordinates. The heads are refitted from scratch after every denoiser
phase (deterministic per epoch/member seeds, episode-level Poisson(1) bootstrap) and frozen
during PMPO.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from data import Dataset
from data.milestone import state_sha256
from ensemble import episode_multiplicity
from models.diffusion.denoiser import apply_noise_from_samples, Denoiser
from utils import Progress

REWARD_SCALINGS = ("fixed", "none")
ACTIVATIONS = {"silu": nn.SiLU, "elu": nn.ELU, "relu": nn.ReLU}


@dataclass
class Plan2ExploreConfig:
    """All plan2explore_disagreement settings (config/intrinsic_reward/plan2explore_disagreement.yaml).
    The numbers are working placeholders, not tuned or validated experimental settings."""
    method: str = "plan2explore_disagreement"
    enabled: bool = True
    num_members: int = 5  # lightweight predictor heads (>= 2); the agent keeps one denoiser
    bootstrap: bool = True  # episode-level Poisson(1) multiplicities, redrawn per refresh and member
    # Feature extractor: InnerModel submodule names whose outputs are pooled and concatenated.
    feature_layers: List[str] = field(default_factory=lambda: ["unet.d_blocks.2", "unet.d_blocks.3"])
    pool_size: int = 4  # adaptive average pooling to pool_size x pool_size per selected map
    feature_sigma: float = 0.1  # diffusion noise level of the extraction probe (finite, > 0)
    probe_seed: int = 0  # seeds the fixed Gaussian probe (pixel and offset noise)
    feature_std_floor: float = 1e-3  # standardization: std = max(std, floor)
    encode_batch_size: int = 256  # frames per extractor call when encoding replay
    # Predictor heads: MLP (z_t, a_t) -> z_{t+1}, linear output.
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "silu"
    lr: float = 1e-3  # Adam
    batch_size: int = 256
    fit_updates: int = 500  # per member, after every denoiser phase
    reward_scaling: str = "fixed"  # fixed: raw / fixed_divisor; none: raw
    fixed_divisor: Optional[float] = 1.0

    def __post_init__(self) -> None:
        if self.method != "plan2explore_disagreement":
            raise ValueError(f"Plan2ExploreConfig.method must be plan2explore_disagreement, got {self.method!r}")
        self.feature_layers = list(self.feature_layers)
        self.hidden_dims = list(self.hidden_dims)
        if not is_int(self.num_members, minimum=2):
            raise ValueError(f"intrinsic_reward.num_members must be an integer >= 2 predictor heads, got {self.num_members!r}")
        if not isinstance(self.bootstrap, bool):
            raise ValueError(f"intrinsic_reward.bootstrap must be true or false, got {self.bootstrap!r}")
        if not self.feature_layers or not all(isinstance(x, str) and x for x in self.feature_layers) \
                or len(set(self.feature_layers)) != len(self.feature_layers):
            raise ValueError(f"intrinsic_reward.feature_layers must be distinct module names, got {self.feature_layers!r}")
        for name in ("pool_size", "encode_batch_size", "batch_size", "fit_updates", "probe_seed"):
            if not is_int(getattr(self, name), minimum=0 if name == "probe_seed" else 1):
                raise ValueError(f"intrinsic_reward.{name} must be a positive integer, got {getattr(self, name)!r}")
        for name in ("feature_sigma", "feature_std_floor", "lr"):
            if not is_positive_float(getattr(self, name)):
                raise ValueError(f"intrinsic_reward.{name} must be a finite positive number, got {getattr(self, name)!r}")
        if not self.hidden_dims or not all(is_int(h, minimum=1) for h in self.hidden_dims):
            raise ValueError(f"intrinsic_reward.hidden_dims must be positive integers, got {self.hidden_dims!r}")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"intrinsic_reward.activation must be one of {tuple(ACTIVATIONS)}, got {self.activation!r}")
        if self.reward_scaling not in REWARD_SCALINGS:
            raise ValueError(f"intrinsic_reward.reward_scaling must be one of {REWARD_SCALINGS}, got {self.reward_scaling!r}")
        if self.reward_scaling == "fixed" and not is_positive_float(self.fixed_divisor):
            raise ValueError(f"intrinsic_reward.fixed_divisor must be a finite positive number, got {self.fixed_divisor!r}")

    def reward_scaling_settings(self) -> Dict:
        """What turns predictor disagreement into policy reward; a resume must not change it."""
        settings = dict(method=self.method, mode=self.reward_scaling, variance="sample (K - 1), mean over features")
        if self.reward_scaling == "fixed":
            settings["fixed_divisor"] = float(self.fixed_divisor)
        return settings


def is_int(value, minimum: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


def is_positive_float(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def predictor_disagreement(predictions: Tensor) -> Tensor:
    """predictions: (K, batch, features) predicted next-feature means. Sample variance across
    the K heads (divisor K - 1), averaged over features -> (batch,)."""
    k = predictions.shape[0]
    if k < 2:
        raise ValueError("Disagreement needs at least two predictions")
    centered = predictions - predictions.mean(dim=0, keepdim=True)
    return centered.square().sum(dim=0).div(k - 1).mean(dim=-1)


@contextmanager
def eval_modes(module: nn.Module):
    """Eval mode inside, then every submodule's own previous mode (no blanket train())."""
    modes = {name: m.training for name, m in module.named_modules()}
    module.eval()
    try:
        yield
    finally:
        for name, m in module.named_modules():
            m.training = modes[name]


class _Captured(Exception):
    """Early exit: every selected activation has been captured."""


class FeatureExtractor:
    """E(o_t | o_{t-n..t-1}, a_{t-n..t-1}): the denoiser's normal corruption
    o_t + sigma * eps + sigma_offset_noise * eps_offset with one fixed probe (eps, eps_offset)
    shared by every frame, EDM preconditioning at feature_sigma, selected InnerModel activations
    (read through forward hooks registered only for the call; the forward pass stops once the
    last selected block has run), adaptive average pooling and concatenation (float32)."""

    def __init__(self, denoiser: Denoiser, layers: Sequence[str], pool_size: int, sigma: float, probe_seed: int,
                 img_size: int) -> None:
        self.denoiser = denoiser
        self.layers = list(layers)
        self.pool_size = pool_size
        self.sigma = float(sigma)
        inner = denoiser.cfg.inner_model
        self.num_conditioning = inner.num_steps_conditioning
        self.img_shape = (inner.img_channels, img_size, img_size)
        self.action_dim = inner.continuous_action_dim
        modules = dict(denoiser.inner_model.named_modules())
        missing = [name for name in self.layers if name not in modules or name == ""]
        if missing:
            available = [n for n in modules if n.count(".") <= 2 and n.startswith("unet.")]
            raise ValueError(f"intrinsic_reward.feature_layers {missing} are not InnerModel modules; e.g. {available}")
        self.modules = [modules[name] for name in self.layers]
        # The probe: drawn once from its own generator (no global RNG), identical on resume.
        generator = torch.Generator().manual_seed(int(probe_seed))
        c, h, w = self.img_shape
        self.probe_eps = torch.randn(c, h, w, generator=generator)
        self.probe_offset = torch.randn(c, 1, 1, generator=generator)
        digest = hashlib.sha256()
        for t in (self.probe_eps, self.probe_offset):
            digest.update(t.numpy().tobytes())
        self.probe_sha256 = digest.hexdigest()
        self.seconds = 0.0  # time in __call__ (monitoring)
        self.layer_shapes = self.check_shapes()
        self.feature_dim = sum(s[0] * pool_size ** 2 for s in self.layer_shapes.values())

    def check_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Model-only shape check on a zero input (no RNG, no data): every selected block must
        give a (channels, height, width) map at least pool_size wide."""
        c, h, w = self.img_shape
        device = self.denoiser.device
        frames = torch.zeros(1, c, h, w, device=device)
        context = torch.zeros(1, self.num_conditioning, c, h, w, device=device)
        actions = torch.zeros(1, self.num_conditioning, self.action_dim, device=device)
        maps = self.activations(frames, context, actions)
        shapes = {}
        for name, value in zip(self.layers, maps):
            if value.ndim != 4 or min(value.shape[2:]) < self.pool_size:
                raise ValueError(f"Feature layer {name} gives shape {tuple(value.shape[1:])}; a (channels, height, "
                                 f"width) map at least pool_size={self.pool_size} wide is required")
            shapes[name] = tuple(value.shape[1:])
        return shapes

    @torch.no_grad()
    def activations(self, frames: Tensor, context: Tensor, actions: Tensor, early_exit: bool = True) -> List[Tensor]:
        b = frames.shape[0]
        device = self.denoiser.device
        frames, context, actions = frames.to(device), context.to(device), actions.to(device)
        sigma = torch.full((b,), self.sigma, device=device)
        eps = self.probe_eps.to(device).expand(b, *self.img_shape)
        noisy = apply_noise_from_samples(frames, sigma, eps, self.probe_offset.to(device),
                                         self.denoiser.cfg.sigma_offset_noise)
        cs = self.denoiser.compute_conditioners(sigma)
        obs = context.reshape(b, -1, *self.img_shape[1:])
        captured: Dict[int, Tensor] = {}

        def hook(index):
            def capture(module, inputs, output):
                captured[index] = output[0] if isinstance(output, tuple) else output
                if early_exit and len(captured) == len(self.modules):
                    raise _Captured()
            return capture

        handles = [m.register_forward_hook(hook(i)) for i, m in enumerate(self.modules)]
        try:
            with eval_modes(self.denoiser):
                self.denoiser.compute_model_output(noisy, obs, actions, cs)
        except _Captured:
            pass
        finally:
            for handle in handles:
                handle.remove()
        return [captured[i] for i in range(len(self.modules))]

    @torch.no_grad()
    def __call__(self, frames: Tensor, context: Tensor, actions: Tensor) -> Tensor:
        """frames (B, c, h, w) to encode; context (B, n, c, h, w) = o_{t-n..t-1};
        actions (B, n, action_dim) = a_{t-n..t-1}. Returns (B, feature_dim) float32."""
        start = synchronized_time(self.denoiser.device)
        maps = self.activations(frames, context, actions)
        features = torch.cat([F.adaptive_avg_pool2d(m, self.pool_size).flatten(1) for m in maps], dim=1).float()
        self.seconds += synchronized_time(self.denoiser.device) - start
        return features

    def settings(self) -> Dict[str, Any]:
        return dict(layers=list(self.layers), layer_shapes={k: list(v) for k, v in self.layer_shapes.items()},
                    pool_size=self.pool_size, sigma=self.sigma, probe_sha256=self.probe_sha256,
                    num_conditioning=self.num_conditioning, img_shape=list(self.img_shape),
                    sigma_offset_noise=float(self.denoiser.cfg.sigma_offset_noise),
                    sigma_data=float(self.denoiser.cfg.sigma_data), feature_dim=self.feature_dim)


def check_feature_layers(cfg: Plan2ExploreConfig, denoiser_cfg, img_size: int) -> Dict[str, Tuple[int, ...]]:
    """Model-only check of the selectors and pooling before a run writes anything: a throwaway
    CPU denoiser of the configured architecture (built under a forked RNG, so no global random
    numbers are drawn). Feature shapes do not depend on the action dimension."""
    denoiser_cfg = deepcopy(denoiser_cfg)
    denoiser_cfg.inner_model.continuous_action_dim = 1
    with torch.random.fork_rng(devices=[]):
        denoiser = Denoiser(denoiser_cfg)
    return FeatureExtractor(denoiser, cfg.feature_layers, cfg.pool_size, cfg.feature_sigma, cfg.probe_seed,
                            img_size).layer_shapes


@dataclass
class FeatureCache:
    """Replay features of one (denoiser, replay, extraction settings) identity, on CPU.
    features: one row per encoded frame; transitions index current/target rows."""
    features: Tensor  # (frames, D) float32
    current: Tensor  # (M,) rows of z_t
    target: Tensor  # (M,) rows of z_{t+1}
    actions: Tensor  # (M, action_dim) a_t
    episode: Tensor  # (M,) episode id (bootstrap only)
    counts: Dict[str, int]
    identity: Dict[str, str]


def tensor_digest(digest, name: str, value: Tensor) -> None:
    value = value.detach().cpu().contiguous()
    digest.update(f"{name}:{value.dtype}:{tuple(value.shape)};".encode())
    digest.update(value.numpy().tobytes())


def episode_digest(episode) -> str:
    """Content hash of exactly the reward-free inputs features are built from: observations,
    actions, the end/truncation boundaries and the final observation. Rewards, task rewards and
    other metadata are not read."""
    digest = hashlib.sha256()
    for name in ("obs", "act", "end", "trunc"):
        tensor_digest(digest, name, getattr(episode, name))
    final = episode.info.get("final_observation") if isinstance(episode.info, dict) else None
    if final is not None:
        tensor_digest(digest, "final_observation", final.float())
    return digest.hexdigest()


def replay_identity(dataset: Dataset, successors: Dict[int, Tensor], memo: Optional[Dict] = None) -> str:
    """Identity of the replay content features depend on: every episode's content digest (in
    order), and the collector's successor observations. memo caches episode digests keyed by
    the episode version file's resolved path and its write-time SHA-256 (immutable versions);
    versions without a recorded checksum are always rehashed."""
    digest = hashlib.sha256()
    checksums = dataset.file_sha256 or [None] * dataset.num_episodes
    for episode_id, (path, length, sha) in enumerate(zip(dataset.episode_files(), dataset.lengths, checksums)):
        key = (str(Path(path).resolve()), sha) if sha is not None and memo is not None else None
        value = memo.get(key) if key is not None else None
        if value is None:
            value = episode_digest(dataset.load_episode(episode_id))
            if key is not None:
                memo[key] = value
        digest.update(f"{episode_id}:{int(length)}:{value};".encode())
    for episode_id in sorted(successors):
        tensor_digest(digest, f"successor_{episode_id}", successors[episode_id].float())
    return digest.hexdigest()


def settings_sha256(settings: Dict[str, Any]) -> str:
    return hashlib.sha256(repr(sorted(settings.items())).encode()).hexdigest()


def episode_frames(episode, successor: Optional[Tensor]) -> Tuple[Tensor, Tensor, bool]:
    """(frames o_0..o_{T-1}[, o_T], actions a_0..a_{T-1}, has successor). The successor of the
    last step is the episode's final observation (time limit) or the observation recorded at a
    collection cut; without either the last step has no verified next frame."""
    obs, act = episode.obs, episode.act
    final = episode.info.get("final_observation") if isinstance(episode.info, dict) else None
    nxt = final if final is not None else successor
    if nxt is None:
        return obs, act, False
    nxt = nxt.detach().to(device=obs.device, dtype=obs.dtype).reshape(1, *obs.shape[1:])
    return torch.cat((obs, nxt), dim=0), act, True


def context_window(frames: Tensor, actions: Tensor, j: int, n: int) -> Tuple[Tensor, Tensor, bool]:
    """Context of frame j: frames j-n..j-1 and actions j-n..j-1, zero-padded before the episode
    start (the denoiser's training padding). Returns (obs (n,c,h,w), act (n,a), padded)."""
    lo = j - n
    obs = frames.new_zeros(n, *frames.shape[1:])
    act = actions.new_zeros(n, actions.shape[1])
    s = max(lo, 0)
    obs[s - lo:] = frames[s:j]
    act[s - lo:] = actions[s:j]
    return obs, act, lo < 0


def build_feature_cache(extractor: FeatureExtractor, dataset: Dataset, successors: Dict[int, Tensor],
                        batch_size: int, identity: Dict[str, str]) -> FeatureCache:
    """Encode every frame of the replay once (CPU result, batched extractor calls) and index the
    eligible transitions (o_t, a_t) -> o_{t+1}: t+1 inside the episode, or t the last stored
    step with a final observation / cut successor. Reads only obs, act and the final observation
    of each episode; task rewards and other metadata are never read."""
    n = extractor.num_conditioning
    current, target, actions, episodes = [], [], [], []
    counts = dict(episodes=0, encoded_frames=0, eligible_transitions=0, excluded_no_successor=0,
                  successor_transitions=0, padded_context_transitions=0)
    pending: List[Tuple[Tensor, Tensor, Tensor]] = []
    out: List[Tensor] = []

    def flush():
        if pending:
            frames, ctx, ctx_act = (torch.stack(x) for x in zip(*pending))
            out.append(extractor(frames, ctx, ctx_act).cpu())
            pending.clear()

    row = 0
    for episode_id in Progress(range(dataset.num_episodes), desc="P2E replay encoding", unit="ep"):
        episode = dataset.load_episode(episode_id)
        frames, acts, has_next = episode_frames(episode, successors.get(episode_id))
        steps = len(episode)
        counts["episodes"] += 1
        for j in range(frames.shape[0]):
            ctx, ctx_act, _ = context_window(frames, acts, j, n)
            pending.append((frames[j], ctx, ctx_act))
            if len(pending) == batch_size:
                flush()
        for t in range(frames.shape[0] - 1):
            current.append(row + t)
            target.append(row + t + 1)
            actions.append(acts[t])
            episodes.append(episode_id)
            counts["padded_context_transitions"] += int(t < n)
        counts["successor_transitions"] += int(has_next)
        counts["excluded_no_successor"] += int(not has_next)
        row += frames.shape[0]
    flush()
    counts["encoded_frames"] = row
    counts["eligible_transitions"] = len(current)
    if not current:
        raise ValueError("Plan2Explore: the replay has no transition with a verified next observation")
    features = torch.cat(out) if out else torch.zeros(0, extractor.feature_dim)
    return FeatureCache(features, torch.tensor(current), torch.tensor(target), torch.stack(actions).float(),
                        torch.tensor(episodes), counts, identity)


class PredictorHead(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, hidden_dims: Sequence[int], activation: str) -> None:
        super().__init__()
        layers, width = [], feature_dim + action_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(width, hidden), ACTIVATIONS[activation]()]
            width = hidden
        layers.append(nn.Linear(width, feature_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor, action: Tensor) -> Tensor:
        return self.net(torch.cat((z, action), dim=-1))


class Plan2Explore:
    """Feature extractor, standardization, predictor heads and their refresh; the PMPO
    intrinsic-reward hook (__call__). Checkpointed by ExplorationTrainer (state_dict)."""

    metric_prefix = "p2e"

    def __init__(self, cfg: Plan2ExploreConfig, denoiser: Denoiser, img_size: int, action_low: Sequence[float],
                 action_high: Sequence[float], seed: int, device: torch.device) -> None:
        self.cfg, self.seed, self.device = cfg, int(seed), device
        self.extractor = FeatureExtractor(denoiser, cfg.feature_layers, cfg.pool_size, cfg.feature_sigma,
                                          cfg.probe_seed, img_size)
        self.feature_dim = self.extractor.feature_dim
        self.action_low = torch.tensor(action_low, dtype=torch.float32)
        self.action_high = torch.tensor(action_high, dtype=torch.float32)
        self.action_dim = len(action_low)
        self.heads: Optional[nn.ModuleList] = None
        self.mean: Optional[Tensor] = None  # standardization (CPU float32), fixed between refreshes
        self.std: Optional[Tensor] = None
        self.raw_std: Optional[Tensor] = None  # before the floor (monitoring)
        self.cache: Optional[FeatureCache] = None  # not checkpointed; rebuilt deterministically
        self.refreshes = 0
        self.last_refresh_epoch = None
        self.fit_updates = [0] * cfg.num_members
        self.empty_bootstrap_fallbacks = [0] * cfg.num_members
        self.cache_identity = None
        # Episode content digests by (version file, write-time SHA-256); not checkpointed.
        self._episode_digests: Dict[Tuple[str, str], str] = {}
        # Monitoring only.
        self.last_raw: Optional[Tensor] = None
        self.last_scaled: Optional[Tensor] = None
        self.scoring_seconds = 0.0
        self.num_head_parameters = sum(p.numel() for p in self.make_head(0).parameters()) * cfg.num_members

    # ---------------------------------------------------------------- settings and state

    def settings(self) -> Dict[str, Any]:
        """Everything that changes features, fitting or reward; a resume must not change it."""
        # encode_batch_size is included: batch composition changes features at the float
        # rounding level (~1e-7 on CPU), enough to change refitted heads.
        cfg = asdict(self.cfg)
        cfg.pop("enabled")
        return dict(config=cfg, extractor=self.extractor.settings(), action_dim=self.action_dim,
                    action_low=self.action_low.tolist(), action_high=self.action_high.tolist())

    def make_head(self, init_seed: int) -> PredictorHead:
        """A freshly initialized head from its own seed (CPU init, global RNG untouched). Only the
        CPU generator is seeded: torch.manual_seed would also reseed the CUDA generators, which
        fork_rng(devices=[]) does not restore."""
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(int(init_seed))
            head = PredictorHead(self.feature_dim, self.action_dim, self.cfg.hidden_dims, self.cfg.activation)
        return head

    def member_seeds(self, epoch: int, member: int) -> Tuple[int, int]:
        """(head initialization, transition sampling) seeds of one refresh and member."""
        init, sample = np.random.SeedSequence([self.seed, 11, int(epoch), int(member)]).generate_state(2)
        return int(init), int(sample)

    def bootstrap_seed(self, epoch: int) -> int:
        return int(np.random.SeedSequence([self.seed, 12, int(epoch)]).generate_state(1)[0])

    def multiplicities(self, epoch: int, member: int, num_episodes: int) -> np.ndarray:
        """Poisson(1) multiplicity of every episode for this refresh and member (the existing
        episode-level bootstrap function, seeded per refresh)."""
        seed = self.bootstrap_seed(epoch)
        return np.array([episode_multiplicity(seed, member, e) for e in range(num_episodes)], dtype=np.float64)

    def state_dict(self) -> Dict[str, Any]:
        return dict(settings=self.settings(), seed=self.seed,
                    heads=None if self.heads is None else {k: v.cpu() for k, v in self.heads.state_dict().items()},
                    mean=self.mean, std=self.std, refreshes=self.refreshes, last_refresh_epoch=self.last_refresh_epoch,
                    fit_updates=list(self.fit_updates), empty_bootstrap_fallbacks=list(self.empty_bootstrap_fallbacks),
                    cache_identity=self.cache_identity,
                    seed_spec="init/sample: SeedSequence([seed, 11, epoch, member]); "
                              "bootstrap: episode_multiplicity(SeedSequence([seed, 12, epoch]), member, episode)")

    def check_settings(self, state: Optional[Dict[str, Any]]) -> None:
        saved = None if state is None else state["settings"]
        if saved != self.settings() or (state is not None and state["seed"] != self.seed):
            raise ValueError(f"Checkpoint was trained with Plan2Explore settings {saved}, the config gives "
                             f"{self.settings()}; a resume must not change them")

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.check_settings(state)
        self.heads = None
        if state["heads"] is not None:
            heads = nn.ModuleList(self.make_head(0) for _ in range(self.cfg.num_members))
            heads.load_state_dict(state["heads"])
            self.heads = heads.to(self.device).eval().requires_grad_(False)
        self.mean, self.std = state["mean"], state["std"]
        self.refreshes, self.last_refresh_epoch = state["refreshes"], state["last_refresh_epoch"]
        self.fit_updates = list(state["fit_updates"])
        self.empty_bootstrap_fallbacks = list(state["empty_bootstrap_fallbacks"])
        self.cache_identity = state["cache_identity"]
        self.cache = None

    # ---------------------------------------------------------------- refresh

    def identity(self, denoiser: Denoiser, dataset: Dataset, successors: Dict[int, Tensor]) -> Dict[str, str]:
        return dict(denoiser_sha256=state_sha256(denoiser.state_dict()),
                    replay_sha256=replay_identity(dataset, successors, self._episode_digests),
                    extraction_sha256=settings_sha256(self.extractor.settings()))

    def refresh(self, denoiser: Denoiser, dataset: Dataset, successors: Dict[int, Tensor], epoch: int) -> List[Dict]:
        """After a denoiser phase: (re)encode the replay unless the cache has exactly this
        identity, fit standardization on it, then fit fresh heads. The denoiser is only read."""
        if denoiser is not self.extractor.denoiser:
            raise RuntimeError("Plan2Explore was built for a different denoiser instance")
        logs = {}
        t0 = synchronized_time(self.device)
        identity = self.identity(denoiser, dataset, successors)
        t1 = synchronized_time(self.device)
        reused = self.cache is not None and self.cache.identity == identity
        if not reused:
            self.cache = None  # a stale cache is never used, even if building fails
            cache = build_feature_cache(self.extractor, dataset, successors, self.cfg.encode_batch_size, identity)
            self.cache = cache
            self.fit_standardization()
        t2 = synchronized_time(self.device)
        logs.update(self.fit_heads(epoch))
        t3 = synchronized_time(self.device)
        self.refreshes += 1
        self.last_refresh_epoch = int(epoch)
        self.cache_identity = dict(identity)
        self.last_raw = self.last_scaled = None
        self.scoring_seconds = 0.0
        self.extractor.seconds = 0.0
        p = "plan2explore"
        cache = self.cache
        logs.update({f"{p}/{k}": v for k, v in cache.counts.items()})
        logs.update({f"{p}/feature_dim": self.feature_dim, f"{p}/head_parameters": self.num_head_parameters,
                     f"{p}/cache_reused": int(reused), f"{p}/refreshes": self.refreshes,
                     f"{p}/cache_mb": cache.features.numel() * 4 / 2 ** 20,
                     f"{p}/feature_std_min": float(self.raw_std.min()), f"{p}/feature_std_mean": float(self.raw_std.mean()),
                     f"{p}/feature_std_max": float(self.raw_std.max()),
                     f"{p}/feature_mean_abs_mean": float(self.mean.abs().mean()),
                     f"{p}/near_constant_coordinates": int((self.raw_std < self.cfg.feature_std_floor).sum()),
                     "timing/p2e_identity_seconds": t1 - t0,
                     "timing/p2e_feature_preparation_seconds": t2 - t1, "timing/p2e_fit_seconds": t3 - t2})
        return [logs]

    def fit_standardization(self) -> None:
        """Coordinate-wise mean and std over the encoded frames used by eligible training
        transitions (current or target), std floored at feature_std_floor."""
        cache = self.cache
        used = torch.unique(torch.cat((cache.current, cache.target)))
        values = cache.features[used].double()
        self.mean = values.mean(0).float()
        self.raw_std = values.std(0, unbiased=False).float()
        self.std = self.raw_std.clamp_min(self.cfg.feature_std_floor)

    def standardize(self, features: Tensor) -> Tensor:
        return (features - self.mean.to(features.device)) / self.std.to(features.device)

    def canonical_action(self, action: Tensor) -> Tensor:
        low, high = self.action_low.to(action.device), self.action_high.to(action.device)
        return 2 * (action - low) / (high - low) - 1

    def sampling_probabilities(self, epoch: int, member: int, num_episodes: int) -> Tuple[np.ndarray, Optional[float], bool]:
        """Per-transition sampling probabilities of one member: proportional to its episode's
        bootstrap multiplicity (uniform within the episode), or uniform without bootstrap or
        when every eligible episode has multiplicity 0 (the logged fallback). Returns
        (probabilities, fraction of episodes with multiplicity 0 or None, fallback used)."""
        weights = np.ones(len(self.cache.current))
        zero_fraction, fallback = None, False
        if self.cfg.bootstrap:
            mult = self.multiplicities(epoch, member, num_episodes)
            boot = mult[self.cache.episode.numpy()]
            zero_fraction = float((mult == 0).mean())
            if boot.sum() > 0:
                weights = boot
            else:
                fallback = True
        return weights / weights.sum(), zero_fraction, fallback

    def fit_heads(self, epoch: int) -> Dict[str, Any]:
        """Fresh heads, each with its own init and sampling seeds and bootstrap weights; Adam on
        MSE in standardized coordinates; only head parameters are optimized. Frozen afterwards."""
        cache, cfg = self.cache, self.cfg
        num_episodes = int(cache.episode.max()) + 1
        heads, logs = [], {}
        pbar = Progress(total=cfg.num_members * cfg.fit_updates, desc=f"P2E head fitting ({cfg.num_members} heads)",
                        unit="updates")
        for k in range(cfg.num_members):
            init_seed, sample_seed = self.member_seeds(epoch, k)
            head = self.make_head(init_seed).to(self.device).train()
            opt = torch.optim.Adam(head.parameters(), lr=cfg.lr)
            p, zero_fraction, fallback = self.sampling_probabilities(epoch, k, num_episodes)
            if zero_fraction is not None:
                logs[f"plan2explore/member_{k}_bootstrap_zero_episode_fraction"] = zero_fraction
            self.empty_bootstrap_fallbacks[k] += int(fallback)
            rng = np.random.default_rng(sample_seed)
            losses = []
            for _ in range(cfg.fit_updates):
                idx = torch.from_numpy(rng.choice(len(p), size=cfg.batch_size, replace=True, p=p))
                z = self.standardize(cache.features[cache.current[idx]].to(self.device))
                target = self.standardize(cache.features[cache.target[idx]].to(self.device))
                action = self.canonical_action(cache.actions[idx].to(self.device))
                loss = F.mse_loss(head(z, action), target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(loss.detach())
                self.fit_updates[k] += 1
                pbar.update(1)
            losses = torch.stack(losses).float().cpu()
            if not torch.isfinite(losses).all():
                raise FloatingPointError(f"Non-finite Plan2Explore predictor loss (member {k})")
            logs.update({f"plan2explore/member_{k}_loss_first": float(losses[0]),
                         f"plan2explore/member_{k}_loss_last": float(losses[-1]),
                         f"plan2explore/member_{k}_loss_last10_mean": float(losses[-10:].mean()),
                         f"plan2explore/member_{k}_fit_updates": self.fit_updates[k],
                         f"plan2explore/member_{k}_empty_bootstrap_fallbacks": self.empty_bootstrap_fallbacks[k]})
            heads.append(head.eval().requires_grad_(False))
        pbar.close()
        self.heads = nn.ModuleList(heads)
        return logs

    # ---------------------------------------------------------------- scoring

    @torch.no_grad()
    def predict(self, features: Tensor, actions: Tensor) -> Tensor:
        """(K, batch, D) predicted standardized next features of raw features and env actions."""
        z, a = self.standardize(features), self.canonical_action(actions)
        return torch.stack([head(z, a) for head in self.heads])

    @torch.no_grad()
    def disagreement(self, features: Tensor, actions: Tensor) -> Tensor:
        """Raw disagreement of the frozen heads for raw features (B, D) and env actions (B, a):
        the PMPO hook's score, with no monitoring state changed (evaluation)."""
        if self.heads is None:
            raise RuntimeError("Plan2Explore heads are not fitted")
        return predictor_disagreement(self.predict(features.float(), actions.float()))

    def scale(self, raw: Tensor) -> Tensor:
        return raw / self.cfg.fixed_divisor if self.cfg.reward_scaling == "fixed" else raw

    @torch.no_grad()
    def __call__(self, infos: List[Dict], env_rew: Tensor) -> Tensor:
        """PMPO hook: per imagined step, the disagreement of the frozen heads on the current
        frame's feature and the taken action (WorldModelEnv info), scaled. (envs, steps)."""
        if self.heads is None:
            raise RuntimeError("Plan2Explore heads are not fitted")
        start = synchronized_time(self.device)
        features = torch.stack([info["plan2explore_feature"] for info in infos], dim=1)
        actions = torch.stack([info["plan2explore_action"] for info in infos], dim=1)
        b, s, d = features.shape
        raw = predictor_disagreement(self.predict(features.flatten(0, 1).float(), actions.flatten(0, 1).float()))
        raw = raw.view(b, s).to(device=env_rew.device, dtype=env_rew.dtype)
        if not torch.isfinite(raw).all():
            raise FloatingPointError("Non-finite Plan2Explore disagreement")
        reward = raw / self.cfg.fixed_divisor if self.cfg.reward_scaling == "fixed" else raw
        self.last_raw, self.last_scaled = raw.detach(), reward.detach()
        self.scoring_seconds += synchronized_time(self.device) - start
        return reward


def synchronized_time(device: torch.device) -> float:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()
