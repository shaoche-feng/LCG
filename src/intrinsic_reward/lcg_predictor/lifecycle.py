"""Predictor-based LCG: lifecycle, configuration and integration (design: docs/implement/LCG_PREDICTOR_DESIGN.md).

A diagonal GGN-based Laplace approximation around the fitted predictor: the posterior over all predictor
parameters is approximated as N(psi_hat, diag(h_D)^{-1}) with psi_hat the fitted weights. A fixed optimization
budget does not guarantee that psi_hat is the MAP. The candidate score, tr(J diag(h_D)^{-1} J^T) / tau^2, is an
information-gain surrogate (the small-variance limit of the expected information gain about psi from observing
z_{t+1}), not a calibrated uncertainty.

Refresh is transactional. After every denoiser phase, with the denoiser frozen:
1. Encode the replay with Plan2Explore's frozen-denoiser feature extractor (same layers, probe, pooling,
   history/action indexing and replay cache) unless the cache has exactly this denoiser/replay/settings
   identity, and refit the replay-only feature standardization.
2. Fit ONE predictor f_psi(z_t, a_t) -> z_{t+1} by explicit MAP (training.py): Gaussian feature likelihood
   (variance tau^2), Gaussian prior (precision lambda), uniform minibatches, no bootstrap.
3. Historical precision h_D = lambda + s * diag(sum_i J_i^T J_i) / tau^2, exact GGN (curvature.py), by default
   over every valid replay transition (s = 1); optionally over a uniform without-replacement subset of B_H
   transitions (s = N / B_H for full-replay scaling). Recomputed from scratch every refresh.
4. The round's scoring bank v_m = h_D^{-1/2} * eta_m (jvp_bank backend only; the default exact_trace
   backend uses no bank).
The new round is built in local state and installed only once every step succeeded; a failure leaves the
previous round untouched. Extractor, standardization, predictor, h_D and bank stay frozen through PMPO and
intrinsic evaluation.
"""
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from data import Dataset
from data.milestone import state_sha256
import intrinsic_reward.plan2explore as p2e
from intrinsic_reward.plan2explore import (ACTIVATIONS, FeatureExtractor, is_int, is_positive_float, PredictorHead,
                                           replay_identity, settings_sha256, synchronized_time)
from models.diffusion.denoiser import Denoiser
from utils import Progress
from .curvature import exact_trace, ggn_diagonal_autograd, ggn_diagonal_closed_form, mlp_structure
from .scoring import jvp_squared_norms, matmul_tf32, ParameterLayout, probe_bank
from .training import fit_map, full_objective, OPTIMIZERS

INIT_POLICIES = ("reinitialize", "warm_start")
CURVATURE_BACKENDS = ("closed_form", "autograd")
CURVATURE_SCALINGS = ("full_replay", "none")
SCORER_BACKENDS = ("jvp_bank", "exact_trace")
SCORE_DTYPES = {"float32": torch.float32, "float64": torch.float64}
REWARD_SCALINGS = ("data_scaled", "fixed", "none")
SCHEMA_VERSION = 1
MILESTONE_FILES = ("lcg_predictor.pt", "lcg_predictor_precision.pt", "lcg_predictor_bank.pt",
                   "lcg_predictor_normalization.pt", "lcg_predictor_scorer.pt")
SEED_SPEC = ("predictor init/sample: SeedSequence([seed, 31, round]); curvature subset: SeedSequence([seed, 32, round]); "
             "bank: SeedSequence([seed, 33, bank_seed, round]); round = the refresh epoch")


@dataclass
class LCGPredictorConfig:
    """All lcg_predictor settings (config/intrinsic_reward/LCG_predictor.yaml). Working placeholders, not
    tuned or validated experimental settings."""
    method: str = "lcg_predictor"
    enabled: bool = True
    # Features: the Plan2Explore extractor and standardization (same meaning as plan2explore_disagreement).
    feature_layers: List[str] = field(default_factory=lambda: ["unet.d_blocks.2", "unet.d_blocks.3"])
    pool_size: int = 4
    feature_sigma: float = 0.1
    probe_seed: int = 0
    feature_std_floor: float = 1e-3
    encode_batch_size: int = 256
    # Predictor f_psi(z_t, a_t) -> z_{t+1}: MLP, input feature_dim + action_dim, linear output feature_dim.
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "silu"
    init_policy: str = "reinitialize"  # reinitialize: fresh init every refresh; warm_start: previous weights
    optimizer: str = "adam"  # adam or sgd (no momentum); never weight decay
    lr: float = 1e-3
    adam_betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    adam_eps: float = 1e-8
    grad_clip_norm: Optional[float] = None
    batch_size: int = 256
    fit_updates: int = 500
    # Gaussian feature likelihood (variance, standardized-feature units) and parameter prior (precision).
    tau_squared: float = 1.0
    prior_precision: float = 1.0
    # Historical precision.
    curvature_backend: str = "closed_form"  # closed_form (MLP, exact) or autograd (generic per-sample VJPs, exact)
    curvature_num_transitions: Optional[int] = None  # B_H; None: every valid transition (exact full-replay sum)
    curvature_block_size: int = 8192  # transitions moved to the device per block (memory only)
    curvature_scaling: str = "full_replay"  # full_replay: N / B_H; none: the subset's plain sum
    curvature_chunk_size: int = 64  # closed form: samples per chunk (memory only)
    autograd_output_chunk: int = 128  # autograd: output basis vectors per vmap (memory only)
    autograd_sample_chunk: int = 4
    # Candidate scoring.
    scorer_backend: str = "exact_trace"  # exact_trace (closed form, no bank) or jvp_bank (num_probes forward JVPs)
    num_probes: int = 5  # M (jvp_bank only)
    bank_seed: int = 0
    score_dtype: str = "float32"  # float32 or float64 (parameters, bank and inputs of the scoring computation)
    allow_tf32: bool = False  # TF32 float32 matmuls in fitting, curvature and scoring
    candidate_chunk_size: int = 1024
    # Policy reward: data_scaled raw / (fixed_divisor * data_reference_steps / replay steps); fixed raw /
    # fixed_divisor; none raw. The divisor is frozen per round.
    reward_scaling: str = "data_scaled"
    fixed_divisor: Optional[float] = 500.0
    data_reference_steps: Optional[int] = 5500

    def __post_init__(self) -> None:
        if self.method != "lcg_predictor":
            raise ValueError(f"LCGPredictorConfig.method must be lcg_predictor, got {self.method!r}")
        self.feature_layers, self.hidden_dims, self.adam_betas = (list(self.feature_layers), list(self.hidden_dims),
                                                                  list(self.adam_betas))
        if not self.feature_layers or not all(isinstance(x, str) and x for x in self.feature_layers) \
                or len(set(self.feature_layers)) != len(self.feature_layers):
            raise ValueError(f"intrinsic_reward.feature_layers must be distinct module names, got {self.feature_layers!r}")
        if self.curvature_num_transitions is not None and not is_int(self.curvature_num_transitions, minimum=1):
            raise ValueError("intrinsic_reward.curvature_num_transitions must be null (full replay) or a positive "
                             f"integer, got {self.curvature_num_transitions!r}")
        for name in ("pool_size", "encode_batch_size", "batch_size", "fit_updates", "curvature_block_size",
                     "curvature_chunk_size", "autograd_output_chunk", "autograd_sample_chunk", "num_probes",
                     "candidate_chunk_size"):
            if not is_int(getattr(self, name), minimum=1):
                raise ValueError(f"intrinsic_reward.{name} must be a positive integer, got {getattr(self, name)!r}")
        for name in ("probe_seed", "bank_seed"):
            if not is_int(getattr(self, name), minimum=0):
                raise ValueError(f"intrinsic_reward.{name} must be a non-negative integer, got {getattr(self, name)!r}")
        for name in ("feature_sigma", "feature_std_floor", "lr", "adam_eps", "tau_squared", "prior_precision"):
            if not is_positive_float(getattr(self, name)):
                raise ValueError(f"intrinsic_reward.{name} must be a finite positive number, got {getattr(self, name)!r}")
        if len(self.adam_betas) != 2 or not all(isinstance(b, (int, float)) and not isinstance(b, bool) and 0 <= b < 1
                                                for b in self.adam_betas):
            raise ValueError(f"intrinsic_reward.adam_betas must be two numbers in [0, 1), got {self.adam_betas!r}")
        if self.grad_clip_norm is not None and not is_positive_float(self.grad_clip_norm):
            raise ValueError(f"intrinsic_reward.grad_clip_norm must be null or a finite positive number, got "
                             f"{self.grad_clip_norm!r}")
        if not self.hidden_dims or not all(is_int(h, minimum=1) for h in self.hidden_dims):
            raise ValueError(f"intrinsic_reward.hidden_dims must be positive integers, got {self.hidden_dims!r}")
        for name, allowed in (("activation", tuple(ACTIVATIONS)), ("init_policy", INIT_POLICIES), ("optimizer", OPTIMIZERS),
                              ("curvature_backend", CURVATURE_BACKENDS), ("curvature_scaling", CURVATURE_SCALINGS),
                              ("scorer_backend", SCORER_BACKENDS), ("score_dtype", tuple(SCORE_DTYPES)),
                              ("reward_scaling", REWARD_SCALINGS)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"intrinsic_reward.{name} must be one of {allowed}, got {getattr(self, name)!r}")
        if not isinstance(self.allow_tf32, bool):
            raise ValueError(f"intrinsic_reward.allow_tf32 must be true or false, got {self.allow_tf32!r}")
        if self.reward_scaling in ("fixed", "data_scaled"):
            if not is_positive_float(self.fixed_divisor):
                raise ValueError(f"intrinsic_reward.fixed_divisor must be a finite positive number, got {self.fixed_divisor!r}")
        elif self.fixed_divisor is not None:
            raise ValueError(f"intrinsic_reward.fixed_divisor is used only with reward_scaling=fixed or data_scaled, "
                             f"got {self.fixed_divisor!r}")
        if self.reward_scaling == "data_scaled":
            if not is_int(self.data_reference_steps, minimum=1):
                raise ValueError(f"intrinsic_reward.data_reference_steps must be a positive integer with "
                                 f"reward_scaling=data_scaled, got {self.data_reference_steps!r}")
        elif self.data_reference_steps is not None:
            raise ValueError(f"intrinsic_reward.data_reference_steps is used only with reward_scaling=data_scaled, "
                             f"got {self.data_reference_steps!r}")

    def reward_scaling_settings(self) -> Dict:
        """What turns raw predictor-LCG scores into policy reward; a resume must not change it."""
        settings = dict(method=self.method, mode=self.reward_scaling)
        if self.reward_scaling in ("fixed", "data_scaled"):
            settings["fixed_divisor"] = float(self.fixed_divisor)
        if self.reward_scaling == "data_scaled":
            settings["data_reference_steps"] = int(self.data_reference_steps)
        return settings


@dataclass
class RoundFeatures:
    """The replay features and standardization one round is fitted on."""
    cache: "p2e.FeatureCache"
    mean: Tensor
    std: Tensor
    raw_std: Tensor


def standardization(cache, floor: float) -> Tuple[Tensor, Tensor, Tensor]:
    """Plan2Explore's standardization: coordinate-wise mean and std over the encoded frames used by valid
    transitions (current or target), std floored at feature_std_floor. Returns (mean, std, raw_std), float32."""
    used = torch.unique(torch.cat((cache.current, cache.target)))
    values = cache.features[used].double()
    mean = values.mean(0).float()
    raw_std = values.std(0, unbiased=False).float()
    return mean, raw_std.clamp_min(floor), raw_std


class LCGPredictor:
    """Feature extractor, standardization, the MAP predictor, its diagonal precision and the round's scoring
    bank; refresh after every denoiser phase; the PMPO intrinsic-reward hook (__call__). Checkpointed by
    ExplorationTrainer (state_dict) and saved with every milestone (milestone_files)."""

    def __init__(self, cfg: LCGPredictorConfig, denoiser: Denoiser, img_size: int, action_low: Sequence[float],
                 action_high: Sequence[float], seed: int, device: torch.device) -> None:
        self.cfg, self.seed, self.device = cfg, int(seed), torch.device(device)
        self.extractor = FeatureExtractor(denoiser, cfg.feature_layers, cfg.pool_size, cfg.feature_sigma,
                                          cfg.probe_seed, img_size)
        self.feature_dim = self.extractor.feature_dim
        self.action_low = torch.tensor(action_low, dtype=torch.float32)
        self.action_high = torch.tensor(action_high, dtype=torch.float32)
        self.action_dim = len(action_low)
        self.dtype = SCORE_DTYPES[cfg.score_dtype]
        template = self.make_predictor(0)
        mlp_structure(template, cfg.activation)  # the closed forms' layout assumptions, checked once
        self.layout = ParameterLayout(template)
        self.num_parameters = self.layout.size
        # Frozen round state (None before the first refresh).
        self.predictor: Optional[PredictorHead] = None
        self.mean: Optional[Tensor] = None  # standardization (CPU float32)
        self.std: Optional[Tensor] = None
        self.raw_std: Optional[Tensor] = None  # before the floor (monitoring)
        self.h_D: Optional[Tensor] = None  # (P,) float32, layout order
        self.bank: Optional[Tensor] = None  # (M, P) float32 tangents; None with exact_trace
        self.round_id: Optional[int] = None
        self.reward_divisor: Optional[float] = None
        self.replay_steps: Optional[int] = None
        self.num_transitions: Optional[int] = None
        self.curvature_info: Optional[Dict[str, Any]] = None
        self.fit_info: Optional[Dict[str, Any]] = None
        self.predictor_sha256: Optional[str] = None
        self.cache_identity: Optional[Dict[str, str]] = None
        self.refreshes = 0
        self.fit_updates_total = 0
        self.cache: Optional[p2e.FeatureCache] = None  # not checkpointed; rebuilt deterministically
        self._episode_digests: Dict[Tuple[str, str], str] = {}
        self._score_params = self._tangents = self._inv_h = None  # scoring views (score dtype)
        # Monitoring only.
        self.last_raw: Optional[Tensor] = None
        self.last_scaled: Optional[Tensor] = None
        self.scoring_seconds = 0.0

    # ---------------------------------------------------------------- settings, seeds and state

    def settings(self) -> Dict[str, Any]:
        """Everything that changes features, fitting, precision, bank or reward; a resume must not change it."""
        cfg = asdict(self.cfg)
        cfg.pop("enabled")
        return dict(config=cfg, extractor=self.extractor.settings(), action_dim=self.action_dim,
                    action_low=self.action_low.tolist(), action_high=self.action_high.tolist(),
                    parameter_layout=self.layout.describe(), num_parameters=self.num_parameters)

    def make_predictor(self, init_seed: int) -> PredictorHead:
        """A freshly initialized predictor from its own seed (CPU init; global and CUDA RNG untouched)."""
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(int(init_seed))
            head = PredictorHead(self.feature_dim, self.action_dim, self.cfg.hidden_dims, self.cfg.activation)
        return head

    def fit_seeds(self, round_id: int) -> Tuple[int, int]:
        init, sample = np.random.SeedSequence([self.seed, 31, int(round_id)]).generate_state(2)
        return int(init), int(sample)

    def curvature_seed(self, round_id: int) -> int:
        return int(np.random.SeedSequence([self.seed, 32, int(round_id)]).generate_state(1)[0])

    def bank_seed(self, round_id: int) -> int:
        return int(np.random.SeedSequence([self.seed, 33, self.cfg.bank_seed, int(round_id)]).generate_state(1)[0])

    def state_dict(self) -> Dict[str, Any]:
        cpu = lambda t: None if t is None else t.detach().cpu()
        return dict(schema=SCHEMA_VERSION, settings=self.settings(), seed=self.seed,
                    predictor=None if self.predictor is None else {k: v.cpu() for k, v in self.predictor.state_dict().items()},
                    predictor_sha256=self.predictor_sha256, mean=cpu(self.mean), std=cpu(self.std),
                    raw_std=cpu(self.raw_std), h_D=cpu(self.h_D), bank=cpu(self.bank), round_id=self.round_id,
                    reward_divisor=self.reward_divisor, replay_steps=self.replay_steps,
                    num_transitions=self.num_transitions, curvature_info=self.curvature_info, fit_info=self.fit_info,
                    cache_identity=self.cache_identity, refreshes=self.refreshes,
                    fit_updates_total=self.fit_updates_total, seed_spec=SEED_SPEC)

    def check_settings(self, state: Optional[Dict[str, Any]]) -> None:
        saved = None if state is None else state["settings"]
        if saved != self.settings() or (state is not None and state["seed"] != self.seed):
            raise ValueError(f"Checkpoint was trained with LCG-predictor settings {saved}, the config gives "
                             f"{self.settings()}; a resume must not change them")

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore a saved round exactly: nothing is refitted, recomputed or redrawn."""
        self.check_settings(state)
        if state.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"LCG-predictor state schema {state.get('schema')!r}, expected {SCHEMA_VERSION}")
        self.predictor = None
        if state["predictor"] is not None:
            head = self.make_predictor(0)
            head.load_state_dict(state["predictor"])
            self.predictor = head.to(self.device).eval().requires_grad_(False)
            if state_sha256(self.predictor.state_dict()) != state["predictor_sha256"]:
                raise ValueError("Saved LCG predictor weights do not match their recorded SHA-256")
        dev = lambda t: None if t is None else t.to(self.device)
        self.mean, self.std, self.raw_std = state["mean"], state["std"], state["raw_std"]
        self.h_D, self.bank = dev(state["h_D"]), dev(state["bank"])
        if self.predictor is not None:
            if self.h_D is None or tuple(self.h_D.shape) != (self.num_parameters,):
                raise ValueError("Saved LCG-predictor precision does not match the predictor's parameters")
            expected_bank = (self.cfg.num_probes, self.num_parameters) if self.cfg.scorer_backend == "jvp_bank" else None
            if (None if self.bank is None else tuple(self.bank.shape)) != expected_bank:
                raise ValueError(f"Saved LCG-predictor bank has shape {None if self.bank is None else tuple(self.bank.shape)}, "
                                 f"the {self.cfg.scorer_backend} backend needs {expected_bank}")
            divisor = state["reward_divisor"]
            if (self.cfg.reward_scaling == "none") != (divisor is None) or (
                    divisor is not None and not (math.isfinite(divisor) and divisor > 0)):
                raise ValueError(f"Saved LCG-predictor reward divisor {divisor!r} does not fit "
                                 f"reward_scaling={self.cfg.reward_scaling}")
        for name in ("round_id", "reward_divisor", "replay_steps", "num_transitions", "curvature_info", "fit_info",
                     "predictor_sha256", "cache_identity", "refreshes", "fit_updates_total"):
            setattr(self, name, state[name])
        self.cache = None
        self._prepare_scoring()

    def check_denoiser(self) -> None:
        """The extractor's denoiser must be the one this round was refreshed on (its weights loaded)."""
        if self.predictor is None:
            return
        recorded = (self.cache_identity or {}).get("denoiser_sha256")
        actual = state_sha256(self.extractor.denoiser.state_dict())
        if recorded != actual:
            raise ValueError(f"The LCG predictor was refreshed on denoiser {recorded}, the loaded denoiser is {actual}")

    def verify_frozen(self) -> None:
        """The predictor weights are still exactly the round's fitted weights."""
        if self.predictor is not None and state_sha256(self.predictor.state_dict()) != self.predictor_sha256:
            raise RuntimeError("The frozen LCG predictor changed after its refresh")

    # ---------------------------------------------------------------- refresh

    def identity(self, denoiser: Denoiser, dataset: Dataset, successors: Dict[int, Tensor]) -> Dict[str, str]:
        return dict(denoiser_sha256=state_sha256(denoiser.state_dict()),
                    replay_sha256=replay_identity(dataset, successors, self._episode_digests),
                    extraction_sha256=settings_sha256(self.extractor.settings()))

    def refresh(self, denoiser: Denoiser, dataset: Dataset, successors: Dict[int, Tensor], epoch: int) -> List[Dict]:
        """After a denoiser phase: features and standardization (replay cache reused only for an identical
        identity), MAP predictor, historical precision and the round's bank. The denoiser is only read; the
        new round replaces the old one only once every part has been computed."""
        if denoiser is not self.extractor.denoiser:
            raise RuntimeError("LCGPredictor was built for a different denoiser instance")
        cfg, round_id = self.cfg, int(epoch)
        # Transactional: the new round (cache, standardization, predictor, precision, bank, divisor) is built
        # in local state and installed only after every step succeeded. A failure anywhere leaves the previous
        # round (and its cache) exactly as it was, still usable for scoring.
        t0 = synchronized_time(self.device)
        identity = self.identity(denoiser, dataset, successors)
        reused = self.cache is not None and self.cache.identity == identity
        if reused:
            features = RoundFeatures(self.cache, self.mean, self.std, self.raw_std)
        else:
            cache = p2e.build_feature_cache(self.extractor, dataset, successors, cfg.encode_batch_size, identity,
                                            desc="LCG-predictor replay encoding")
            features = RoundFeatures(cache, *standardization(cache, cfg.feature_std_floor))
        t1 = synchronized_time(self.device)
        n = len(features.cache.current)
        with matmul_tf32(cfg.allow_tf32):
            predictor, fit_info = self.fit_predictor(round_id, n, features)
            t2 = synchronized_time(self.device)
            h_D, curvature_info = self.fit_precision(predictor, round_id, n, features)
            t3 = synchronized_time(self.device)
            bank = None if cfg.scorer_backend == "exact_trace" else probe_bank(h_D, cfg.num_probes, self.bank_seed(round_id))
            if bank is not None and not torch.isfinite(bank).all():
                raise FloatingPointError("Non-finite LCG-predictor scoring bank")
        t4 = synchronized_time(self.device)
        replay_steps = int(dataset.num_steps)
        divisor = self.divisor(replay_steps)
        predictor_sha256 = state_sha256(predictor.state_dict())
        # ---- install (no step below can fail on valid inputs)
        self.cache, self.mean, self.std, self.raw_std = features.cache, features.mean, features.std, features.raw_std
        self.predictor, self.h_D, self.bank = predictor, h_D, bank
        self.fit_info, self.curvature_info = fit_info, dict(curvature_info, replay_steps=replay_steps)
        self.round_id, self.replay_steps, self.num_transitions = round_id, replay_steps, n
        self.reward_divisor = divisor
        self.cache_identity = dict(identity)
        self.predictor_sha256 = predictor_sha256
        self.refreshes += 1
        self.fit_updates_total += cfg.fit_updates
        self._prepare_scoring()
        self.last_raw = self.last_scaled = None
        self.scoring_seconds = 0.0
        self.extractor.seconds = 0.0
        p = "lcg_predictor"
        logs = {f"{p}/{k}": v for k, v in self.cache.counts.items()}
        logs.update({f"{p}/fit_{k}": v for k, v in fit_info.items() if isinstance(v, (int, float))})
        logs.update({f"{p}/curvature_{k}": v for k, v in curvature_info.items() if isinstance(v, (int, float))})
        logs.update({f"{p}/cache_reused": int(reused), f"{p}/refreshes": self.refreshes,
                     f"{p}/num_parameters": self.num_parameters, f"{p}/reward_divisor": self.reward_divisor,
                     "timing/lcgp_feature_preparation_seconds": t1 - t0, "timing/lcgp_fit_seconds": t2 - t1,
                     "timing/lcgp_curvature_seconds": t3 - t2, "timing/lcgp_bank_seconds": t4 - t3})
        return [logs]

    def fit_standardization(self) -> None:
        """Install the standardization of the current cache (standardization())."""
        self.mean, self.std, self.raw_std = standardization(self.cache, self.cfg.feature_std_floor)

    def standardize(self, features: Tensor, mean: Optional[Tensor] = None, std: Optional[Tensor] = None) -> Tensor:
        """With the installed round's statistics unless others are given (a round being built)."""
        mean = self.mean if mean is None else mean
        std = self.std if std is None else std
        return (features - mean.to(features.device)) / std.to(features.device)

    def canonical_action(self, action: Tensor) -> Tensor:
        low, high = self.action_low.to(action.device), self.action_high.to(action.device)
        return 2 * (action - low) / (high - low) - 1

    def transition_batch(self, idx: Tensor, features: Optional["RoundFeatures"] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """(standardized z_t, canonical a_t, standardized z_{t+1}) of valid transitions idx, on the device; of
        the installed round unless the features of a round being built are given."""
        f = features or RoundFeatures(self.cache, self.mean, self.std, self.raw_std)
        c = f.cache
        z = self.standardize(c.features[c.current[idx]].to(self.device), f.mean, f.std)
        y = self.standardize(c.features[c.target[idx]].to(self.device), f.mean, f.std)
        return z, self.canonical_action(c.actions[idx].to(self.device)), y

    def fit_predictor(self, round_id: int, n: int, features: Optional["RoundFeatures"] = None
                      ) -> Tuple[PredictorHead, Dict[str, Any]]:
        cfg = self.cfg
        init_seed, sample_seed = self.fit_seeds(round_id)
        head = self.make_predictor(init_seed)
        init = "fresh"
        if cfg.init_policy == "warm_start" and self.predictor is not None:
            head.load_state_dict({k: v.cpu() for k, v in self.predictor.state_dict().items()})
            init = f"warm_start (round {self.round_id})"
        head = head.to(self.device).requires_grad_(True)
        batch = lambda idx: self.transition_batch(idx, features)
        before = full_objective(head, batch, n, cfg.tau_squared, cfg.prior_precision)
        pbar = Progress(total=cfg.fit_updates, desc="LCG-predictor MAP fitting", unit="updates")
        logs = fit_map(head, batch, n, cfg, sample_seed, on_update=lambda: pbar.update(1))
        pbar.close()
        head.eval().requires_grad_(False)
        after = full_objective(head, batch, n, cfg.tau_squared, cfg.prior_precision)
        info = dict(round=round_id, init=init, init_seed=init_seed, sample_seed=sample_seed, num_transitions=n,
                    updates=cfg.fit_updates, batch_size=cfg.batch_size,
                    sampling="uniform with replacement over the N valid transitions (no bootstrap)", **logs,
                    **{f"before_{k}": v for k, v in before.items()}, **{f"after_{k}": v for k, v in after.items()})
        return head, info

    def fit_precision(self, head: PredictorHead, round_id: int, n: int, features: Optional["RoundFeatures"] = None
                      ) -> Tuple[Tensor, Dict[str, Any]]:
        """h_D = lambda + s * sum_{i in subset} diag(J_i^T J_i) / tau^2, from scratch (no earlier round's
        curvature is reused, so no replay is counted twice)."""
        cfg = self.cfg
        full = cfg.curvature_num_transitions is None or cfg.curvature_num_transitions >= n
        if full:
            b_h, seed, idx = n, None, torch.arange(n)
            sampling = "every valid transition (full replay, no sampling)"
        else:
            b_h, seed = cfg.curvature_num_transitions, self.curvature_seed(round_id)
            idx = torch.from_numpy(np.sort(np.random.default_rng(seed).choice(n, size=b_h, replace=False)))
            sampling = "uniform without replacement over the N valid transitions (private generator)"
        ggn = None
        for start in range(0, b_h, cfg.curvature_block_size):  # float64 accumulation across blocks
            z, a, _ = self.transition_batch(idx[start:start + cfg.curvature_block_size], features)
            u = torch.cat((z, a), dim=-1)
            if cfg.curvature_backend == "closed_form":
                part = ggn_diagonal_closed_form(head, cfg.activation, u, cfg.curvature_chunk_size)
            else:
                part = ggn_diagonal_autograd(head, u, cfg.autograd_output_chunk, cfg.autograd_sample_chunk)
            ggn = part if ggn is None else ggn + part
            del z, a, u
        scale = n / b_h if cfg.curvature_scaling == "full_replay" else 1.0
        h_D = (cfg.prior_precision + scale * ggn / cfg.tau_squared).float()
        if not torch.isfinite(h_D).all() or (h_D <= 0).any():
            raise FloatingPointError("Non-finite or non-positive LCG-predictor precision")
        info = dict(round=round_id, backend=cfg.curvature_backend, replay_transitions=n, num_samples=b_h,
                    sampling=sampling, sample_seed=seed, scaling=cfg.curvature_scaling, scale_factor=float(scale),
                    tau_squared=float(cfg.tau_squared), prior_precision=float(cfg.prior_precision),
                    ggn_sum=float(ggn.sum()), h_min=float(h_D.min()), h_median=float(h_D.median()),
                    h_max=float(h_D.max()),
                    h_at_prior_fraction=float((h_D <= cfg.prior_precision * (1 + 1e-6)).float().mean()),
                    recomputed="from scratch at every refresh; no accumulation across rounds")
        return h_D, info

    def divisor(self, replay_steps: int) -> Optional[float]:
        c = self.cfg
        if c.reward_scaling == "fixed":
            return float(c.fixed_divisor)
        if c.reward_scaling == "data_scaled":
            d = float(c.fixed_divisor) * c.data_reference_steps / replay_steps
            if not (math.isfinite(d) and d > 0):
                raise ValueError(f"data_scaled produced an invalid reward divisor {d!r}")
            return d
        return None

    def _prepare_scoring(self) -> None:
        """Score-dtype views of the frozen parameters and bank (or 1/h_D for the exact backend)."""
        self._score_params = self._tangents = self._inv_h = None
        if self.predictor is None:
            return
        self._score_params = {k: v.detach().to(self.dtype) for k, v in self.predictor.named_parameters()}
        if self.bank is not None:
            self._tangents = [self.layout.split(row.to(self.dtype)) for row in self.bank]
        else:
            self._inv_h = self.layout.split(self.h_D.to(self.dtype).reciprocal())

    # ---------------------------------------------------------------- scoring

    def inputs(self, features: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        return (self.standardize(features.float()).to(self.dtype),
                self.canonical_action(actions.float().to(features.device)).to(self.dtype))

    @torch.no_grad()
    def score_features(self, features: Tensor, actions: Tensor) -> Tensor:
        """Raw score (B,) float32 of raw features (B, D) and env actions (B, a) with the frozen round (the
        configured backend); no monitoring state changes (evaluation)."""
        if self.predictor is None:
            raise RuntimeError("The LCG predictor is not fitted")
        z, a = self.inputs(features, actions)
        cfg, out = self.cfg, []
        with matmul_tf32(cfg.allow_tf32):
            for i in range(0, z.shape[0], cfg.candidate_chunk_size):
                zc, ac = z[i:i + cfg.candidate_chunk_size], a[i:i + cfg.candidate_chunk_size]
                if self._tangents is not None:
                    s = jvp_squared_norms(self.predictor, self._score_params, zc, ac, self._tangents) / len(self._tangents)
                else:
                    s = exact_trace(self.predictor, cfg.activation, torch.cat((zc, ac), -1), self._inv_h, self._score_params)
                out.append(s / cfg.tau_squared)
        raw = torch.cat(out).float() if out else z.new_zeros(0, dtype=torch.float32)
        if not torch.isfinite(raw).all() or (raw < 0).any():
            raise FloatingPointError("Non-finite or negative LCG-predictor score")
        return raw

    @torch.no_grad()
    def exact_scores(self, features: Tensor, actions: Tensor) -> Tensor:
        """The closed-form diagonal-posterior trace (any backend; cross-checks)."""
        z, a = self.inputs(features, actions)
        inv_h = self.layout.split(self.h_D.to(self.dtype).reciprocal())
        with matmul_tf32(self.cfg.allow_tf32):
            r = exact_trace(self.predictor, self.cfg.activation, torch.cat((z, a), -1), inv_h, self._score_params)
        return (r / self.cfg.tau_squared).float()

    def scale(self, raw: Tensor) -> Tensor:
        """Policy reward of raw scores with the round's frozen divisor."""
        return raw if self.reward_divisor is None else raw / self.reward_divisor

    @torch.no_grad()
    def __call__(self, infos: List[Dict], env_rew: Tensor) -> Tensor:
        """PMPO hook: per imagined step, the frozen score of the current frame's feature and the taken action
        (WorldModelEnv info), scaled. (envs, steps)."""
        if self.predictor is None:
            raise RuntimeError("The LCG predictor is not fitted")
        start = synchronized_time(self.device)
        features = torch.stack([info["plan2explore_feature"] for info in infos], dim=1)
        actions = torch.stack([info["plan2explore_action"] for info in infos], dim=1)
        b, s, _ = features.shape
        raw = self.score_features(features.flatten(0, 1), actions.flatten(0, 1))
        raw = raw.view(b, s).to(device=env_rew.device, dtype=env_rew.dtype)
        reward = self.scale(raw)
        self.last_raw, self.last_scaled = raw.detach(), reward.detach()
        self.scoring_seconds += synchronized_time(self.device) - start
        return reward

    # ---------------------------------------------------------------- milestones

    def milestone_files(self) -> Dict[str, Any]:
        """The frozen round, file name -> object (docs/analysis/restore_scorer.md); {} before a refresh."""
        if self.predictor is None:
            return {}
        self.verify_frozen()
        s = self.state_dict()
        order = self.layout.describe()
        common = dict(schema=SCHEMA_VERSION, round_id=self.round_id)
        return {
            "lcg_predictor.pt": dict(
                common, state_dict=s["predictor"], predictor_sha256=self.predictor_sha256, parameter_order=order,
                num_parameters=self.num_parameters,
                architecture=dict(input_dim=self.feature_dim + self.action_dim, feature_dim=self.feature_dim,
                                  action_dim=self.action_dim, hidden_dims=list(self.cfg.hidden_dims),
                                  activation=self.cfg.activation, output="linear, standardized z_{t+1}",
                                  input="concat(standardized z_t, canonical a_t in [-1, 1])")),
            "lcg_predictor_precision.pt": dict(
                common, h_D=s["h_D"], parameter_order=order, curvature=self.curvature_info,
                tau_squared=float(self.cfg.tau_squared), prior_precision=float(self.cfg.prior_precision),
                representation="diagonal precision, flat in parameter_order: lambda + scale * diag(sum J^T J) / tau^2"),
            "lcg_predictor_bank.pt": dict(
                common, backend=self.cfg.scorer_backend, tangents=s["bank"], num_probes=self.cfg.num_probes,
                bank_seed=self.bank_seed(self.round_id) if s["bank"] is not None else None, seed_spec=SEED_SPEC,
                generator="torch.Generator() on CPU, torch.randn(num_probes, num_parameters, float32)",
                representation=("v_m = h_D^{-1/2} * eta_m; rows in probe order, columns in parameter_order"
                                if s["bank"] is not None else "exact_trace backend: no scoring bank is used")),
            "lcg_predictor_normalization.pt": dict(
                common, mean=s["mean"], std=s["std"], raw_std=s["raw_std"],
                feature_std_floor=float(self.cfg.feature_std_floor), extractor=self.extractor.settings(),
                probe_sha256=self.extractor.probe_sha256,
                use="standardize(features) = (features - mean) / std, std already floored; raw_std is monitoring only"),
            "lcg_predictor_scorer.pt": dict(
                common, settings=s["settings"], seed=self.seed, denoiser_identity=self.cache_identity,
                reward=dict(self.cfg.reward_scaling_settings(), reward_divisor=self.reward_divisor,
                            replay_steps=self.replay_steps),
                likelihood=dict(tau_squared=float(self.cfg.tau_squared)),
                prior=dict(prior_precision=float(self.cfg.prior_precision)),
                fit=self.fit_info, num_transitions=self.num_transitions, refreshes=self.refreshes,
                fit_updates_total=self.fit_updates_total, scorer_backend=self.cfg.scorer_backend,
                score_dtype=self.cfg.score_dtype, allow_tf32=self.cfg.allow_tf32, seed_spec=SEED_SPEC),
        }

    def load_milestone_files(self, path: Path) -> None:
        """Restore a milestone's round from its files (the denoiser weights already loaded): the saved
        predictor, h_D, bank, standardization and divisor are used as they are; nothing is refitted or redrawn."""
        path = Path(path)
        load = lambda name: torch.load(path / name, map_location="cpu", weights_only=False)
        pred, prec, bank, norm, scorer = (load(name) for name in MILESTONE_FILES)
        rounds = {f["round_id"] for f in (pred, prec, bank, norm, scorer)}
        if len(rounds) != 1 or any(f["schema"] != SCHEMA_VERSION for f in (pred, prec, bank, norm, scorer)):
            raise ValueError(f"Milestone {path}: LCG-predictor files of different rounds or schemas ({sorted(rounds)})")
        if norm["probe_sha256"] != self.extractor.probe_sha256:
            raise ValueError(f"Milestone {path}: the feature probe differs from the one the run recorded")
        if pred["parameter_order"] != self.layout.describe():
            raise ValueError(f"Milestone {path}: lcg_predictor.pt parameter order differs from the config's predictor")
        round_id = pred["round_id"]
        if bank["tangents"] is not None and bank["bank_seed"] != self.bank_seed(round_id):
            raise ValueError(f"Milestone {path}: the bank seed differs from this run's seed and round")
        self.load_state_dict(dict(
            schema=SCHEMA_VERSION, settings=scorer["settings"], seed=scorer["seed"], predictor=pred["state_dict"],
            predictor_sha256=pred["predictor_sha256"], mean=norm["mean"], std=norm["std"], raw_std=norm["raw_std"],
            h_D=prec["h_D"], bank=bank["tangents"], round_id=round_id, reward_divisor=scorer["reward"]["reward_divisor"],
            replay_steps=scorer["reward"]["replay_steps"], num_transitions=scorer["num_transitions"],
            curvature_info=prec["curvature"], fit_info=scorer["fit"], cache_identity=scorer["denoiser_identity"],
            refreshes=scorer["refreshes"], fit_updates_total=scorer["fit_updates_total"]))
        self.check_denoiser()
