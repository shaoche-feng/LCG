#! /usr/bin/env python
"""
Shared setup/utilities for the LCG backward/VJP candidate-score variance diagnostic
(stochastic-source decomposition, independent-MC vs CRN, K sweep). Imported by the Part
A/B/C/D/E/F scripts; also runnable standalone to do the one-time setup (compute+cache
h_D_full, generate+freeze+verify the 480 imagined candidates).

Does NOT modify src/lcg/*.py, src/models/actor_critic.py, or src/trainer.py. Reuses
lcg.batched_vjp.compute_vjp_batched (Stage 4.5, unmodified) as the sole VJP primitive:
compute_vjp_batched already supports BOTH the CRN case (sigma shape (1,), eps/xi shape
(1,C,H,W), broadcast across the whole chunk via ordinary broadcasting) and the fully
independent case (sigma shape (B,), eps/xi shape (B,C,H,W), one independent draw per
candidate) with zero code changes -- Denoiser.compute_conditioners already broadcasts
per-example sigma tensors (see models/diffusion/denoiser.py), so "independent" sampling
here is implemented purely by feeding compute_vjp_batched differently-shaped (but
already-supported) sigma/eps/xi tensors, never by editing it.

Usage (standalone setup):
    python scripts/diagnose_lcg_backward_variance_setup.py
"""
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from data import BatchSampler, Dataset, collate_segments_to_batch
from envs import WorldModelEnv, WorldModelEnvConfig
from envs.dm_control_env import make_dm_control_env
from lcg.batched_scoring import imagined_candidates_from_batch
from lcg.batched_vjp import compute_vjp_batched
from lcg.sigma_strata import sample_sigma_stratum
from lcg.theta_s import selected_parameters
from models.actor_critic import ActorCritic, ActorCriticConfig, ActorCriticLossConfig
from models.diffusion import Denoiser, DenoiserConfig, DiffusionSamplerConfig, SigmaDistributionConfig
from models.diffusion.inner_model import InnerModelConfig
from models.rew_end_model import RewEndModel, RewEndModelConfig

import diagnose_lcg_precision_full_vs_subset as precision_diag

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_BASE = Path(
    r"C:\Users\jerry\AppData\Local\Temp\claude\c--Users-jerry-Project-LCG"
    r"\974bd621-91fa-4766-822e-267b547a92c5\scratchpad"
)
CHECKPOINT_PATH = SCRATCH_BASE / "lcg_diag_denoiser_converged.pt"
TRAIN_DATASET_DIR = SCRATCH_BASE / "lcg_diag_train_dataset"
DIAG_DIR = SCRATCH_BASE / "lcg_backward_variance_diag"
DIAG_DIR.mkdir(parents=True, exist_ok=True)
H_D_FULL_PATH = DIAG_DIR / "h_D_full.pt"
CANDIDATES_PATH = DIAG_DIR / "frozen_candidates_480.pt"

DOMAIN_NAME, TASK_NAME = "cheetah", "run"
ENV_KWARGS = dict(size=64, camera_id=0, action_repeat=2, time_limit=1.0)
NUM_STEPS_CONDITIONING = 4
SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)
SAMPLER_CFG = DiffusionSamplerConfig(num_steps_denoising=3, sigma_min=2e-3, sigma_max=5.0, rho=7, order=1)
NUM_STRATA = 3
DAMPING = 1e-4
BETA = 1.0
CHUNK_SIZE = 4

B_AC = 32
HORIZON = 15
NUM_CANDIDATES = B_AC * HORIZON  # 480


# --------------------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------------------


def load_converged_denoiser(device: torch.device) -> Tuple[Denoiser, int]:
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    action_dim = ckpt["action_dim"]
    inner_cfg = InnerModelConfig(
        img_channels=3, num_steps_conditioning=NUM_STEPS_CONDITIONING, cond_channels=256,
        depths=[2, 2, 2, 2], channels=[64, 64, 64, 64], attn_depths=[0, 0, 0, 0],
        continuous_action_dim=action_dim,
    )
    cfg = DenoiserConfig(inner_model=inner_cfg, sigma_data=0.5, sigma_offset_noise=0.3)
    denoiser = Denoiser(cfg).to(device)
    denoiser.load_state_dict(ckpt["denoiser"])
    denoiser.eval()
    print(f"Loaded converged diagnostic checkpoint (total_step={ckpt['step']}) from {CHECKPOINT_PATH}", flush=True)
    return denoiser, action_dim


def build_rew_end_model(device: torch.device, action_dim: int) -> RewEndModel:
    cfg = RewEndModelConfig(
        lstm_dim=512, img_channels=3, img_size=64, cond_channels=128,
        depths=[2, 2, 2, 2], channels=[32, 32, 32, 32], attn_depths=[0, 0, 0, 0],
        continuous_action_dim=action_dim, continuous_reward=True,
    )
    model = RewEndModel(cfg).to(device)
    model.eval()
    return model


def load_train_dataset() -> Dataset:
    dataset = Dataset(TRAIN_DATASET_DIR, "lcg_diag_train", cache_in_ram=True)
    dataset.load_from_default_path()
    return dataset


# --------------------------------------------------------------------------------------
# h_D_full: reuse (not recompute-and-diverge) the exact deterministic full-enumeration
# procedure validated in the historical-precision diagnostic. Cached once, reused by every
# script in this diagnostic (this experiment must not introduce historical-subset noise).
# --------------------------------------------------------------------------------------


def get_or_compute_h_D_full(denoiser: Denoiser, params, dataset: Dataset, device: torch.device) -> Tensor:
    if H_D_FULL_PATH.is_file():
        h_D_full = torch.load(H_D_FULL_PATH, map_location=device, weights_only=True)
        print(f"Loaded cached h_D_full from {H_D_FULL_PATH} (shape={tuple(h_D_full.shape)}).", flush=True)
        return h_D_full

    print("No cached h_D_full found -- computing via the validated deterministic "
          "full-enumeration procedure (scripts/diagnose_lcg_precision_full_vs_subset.py).", flush=True)
    N = dataset.num_steps
    all_segment_ids = precision_diag.enumerate_all_transitions(dataset, NUM_STEPS_CONDITIONING)
    assert len(all_segment_ids) == N
    d_S = sum(p.numel() for p in params)
    sum_full = torch.zeros(d_S, device=device)
    t0 = time.time()
    for i, segment_id in enumerate(all_segment_ids):
        g_i = precision_diag.compute_g_i(
            denoiser, params, dataset, segment_id, NUM_STRATA, SIGMA_CFG,
            precision_diag.BASE_SALT, device,
        )
        sum_full += g_i
        if (i + 1) % 200 == 0 or (i + 1) == N:
            print(f"  h_D_full [{i + 1}/{N}] elapsed={time.time() - t0:.1f}s", flush=True)
    h_D_full = DAMPING + BETA * sum_full
    torch.save(h_D_full.detach().cpu(), H_D_FULL_PATH)
    print(f"h_D_full computed and cached to {H_D_FULL_PATH}.", flush=True)
    return h_D_full.to(device)


# --------------------------------------------------------------------------------------
# Candidate generation, freezing, verification
# --------------------------------------------------------------------------------------


def build_world_model_env(denoiser, rew_end_model, dataset, num_envs, horizon):
    cfg = WorldModelEnvConfig(horizon=horizon, num_batches_to_preload=4, diffusion_sampler=SAMPLER_CFG)
    bs = BatchSampler(dataset, 0, 1, num_envs, NUM_STEPS_CONDITIONING, sample_weights=None)
    loader = DataLoader(dataset=dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch, num_workers=0)
    return WorldModelEnv(denoiser, rew_end_model, loader, cfg, return_imagined_candidate=True)


def build_actor_critic(device, action_dim, action_low, action_high):
    cfg = ActorCriticConfig(
        lstm_dim=512, img_channels=3, img_size=64, channels=[32, 32, 64, 64], down=[1, 1, 1, 1],
        continuous_action_dim=action_dim, action_low=action_low, action_high=action_high, continuous_reward=True,
    )
    return ActorCritic(cfg).to(device)


def generate_frozen_candidates(denoiser, rew_end_model, dataset, device, action_dim, seed=0):
    """One ActorCritic-sized imagined rollout (B_AC=32, H=15), generated via the exact
    normal DIAMOND path (WorldModelEnv + ActorCritic.forward()/env_loop, Stages 4-5,
    unmodified) -- NOT hand-rolled. backup_every == horizon == 15 so a single ac() call
    produces the complete rollout in one shot. Returns exactly B_AC*horizon=480 candidates,
    concatenated across steps in (step-major, env-minor) order via imagined_candidates_from_batch.
    """
    torch.manual_seed(seed)
    probe_env = make_dm_control_env(domain_name=DOMAIN_NAME, task_name=TASK_NAME, **ENV_KWARGS)
    action_low, action_high = probe_env.action_low.tolist(), probe_env.action_high.tolist()

    wm_env = build_world_model_env(denoiser, rew_end_model, dataset, B_AC, HORIZON)
    ac = build_actor_critic(device, action_dim, action_low, action_high)
    loss_cfg = ActorCriticLossConfig(
        backup_every=HORIZON, gamma=0.985, lambda_=0.95, weight_value_loss=1.0, weight_entropy_loss=0.001
    )
    ac.setup_training(wm_env, loss_cfg)

    with torch.no_grad():
        _, act, rew, end, trunc, logits_act, val, val_bootstrap, infos = ac.env_loop.send(loss_cfg.backup_every)

    assert len(infos) == HORIZON, f"expected {HORIZON} steps of imagined_candidate infos, got {len(infos)}"
    candidates = []
    for t in range(HORIZON):
        batch = infos[t]["imagined_candidate"]
        candidates.extend(imagined_candidates_from_batch(batch.x_obs, batch.x_act, batch.y_star))
    assert len(candidates) == NUM_CANDIDATES
    return candidates


def save_candidates(candidates: List[Tuple[Tensor, Tensor, Tensor]], path: Path = CANDIDATES_PATH) -> None:
    payload = [(c[0].detach().cpu(), c[1].detach().cpu(), c[2].detach().cpu()) for c in candidates]
    torch.save(payload, path)


def load_frozen_candidates(device: torch.device, path: Path = CANDIDATES_PATH) -> List[Tuple[Tensor, Tensor, Tensor]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    return [(c[0].to(device), c[1].to(device), c[2].to(device)) for c in payload]


def verify_candidates(candidates: List[Tuple[Tensor, Tensor, Tensor]]) -> None:
    assert len(candidates) == NUM_CANDIDATES, f"expected {NUM_CANDIDATES} candidates, got {len(candidates)}"
    for i, (obs, act, y) in enumerate(candidates):
        assert obs.shape == (1, NUM_STEPS_CONDITIONING * 3, 64, 64), f"candidate {i} obs shape {obs.shape}"
        assert act.shape[0] == 1 and act.shape[1] == NUM_STEPS_CONDITIONING, f"candidate {i} act shape {act.shape}"
        assert y.shape == (1, 3, 64, 64), f"candidate {i} y shape {y.shape}"
        assert torch.isfinite(obs).all() and torch.isfinite(act).all() and torch.isfinite(y).all(), \
            f"candidate {i} contains non-finite values"
    print(f"verify_candidates: all {len(candidates)} candidates finite, correctly shaped.", flush=True)


# --------------------------------------------------------------------------------------
# Independent-sampling "bank": per-candidate (not shared) sigma/eps/xi, one draw per
# stratum per candidate. Mirrors lcg.crn.CRNBank's structure exactly except each tensor
# carries a leading candidate dimension instead of being a single broadcastable sample.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class IndependentBank:
    sigmas: Tuple[Tensor, ...]     # each (N,)
    epsilons: Tuple[Tensor, ...]   # each (N, C, H, W)
    xis: Tuple[Tensor, ...]        # each (N, C, H, W)

    def __post_init__(self) -> None:
        assert len(self.sigmas) == len(self.epsilons) == len(self.xis)

    @property
    def num_strata(self) -> int:
        return len(self.sigmas)


def make_independent_bank(
    sigma_cfg: SigmaDistributionConfig, num_candidates: int, single_shape: Tuple[int, int, int],
    device: torch.device, num_strata: int = NUM_STRATA, seed: Optional[int] = None,
) -> IndependentBank:
    if seed is not None:
        torch.manual_seed(seed)
    sigmas, epsilons, xis = [], [], []
    for m in range(num_strata):
        sigmas.append(sample_sigma_stratum(sigma_cfg, m, num_strata, num_candidates, device).detach())
        epsilons.append(torch.randn((num_candidates,) + single_shape, device=device).detach())
        xis.append(torch.randn((num_candidates,) + single_shape, device=device).detach())
    return IndependentBank(tuple(sigmas), tuple(epsilons), tuple(xis))


def score_one_bank(denoiser, params, h_D, bank, candidates, chunk_size, independent: bool) -> Tensor:
    """Per-bank (not averaged across banks) score for every candidate, using the SAME
    compute_vjp_batched primitive and the SAME formula as lcg.batched_vjp.score_candidates_batched
    with num_banks=1 -- this function only decomposes that reduction so the K individual
    per-bank scalars can be cached and averaged offline (Part B/C's required optimization),
    never altering the per-bank computation itself. independent=True slices bank tensors
    per-chunk (each candidate gets its own row); independent=False broadcasts the bank's
    single (1,...)-shaped tensors across the whole chunk (CRN)."""
    device = h_D.device
    num_strata = bank.num_strata
    all_scores = []
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        obs_batch = torch.cat([c[0] for c in chunk], dim=0)
        act_batch = torch.cat([c[1] for c in chunk], dim=0)
        y_batch = torch.cat([c[2] for c in chunk], dim=0)
        B = len(chunk)
        score_accum = torch.zeros(B, device=device)
        for m in range(num_strata):
            if independent:
                sigma = bank.sigmas[m][start : start + B]
                eps = bank.epsilons[m][start : start + B]
                xi = bank.xis[m][start : start + B]
            else:
                sigma = bank.sigmas[m]
                eps = bank.epsilons[m]
                xi = bank.xis[m]
            y_sigma_batch = (y_batch + sigma.view(-1, 1, 1, 1) * eps).detach()
            v_batch = compute_vjp_batched(denoiser, params, y_sigma_batch, sigma, obs_batch, act_batch, xi)
            contribution = (v_batch.square() / h_D.unsqueeze(0)).sum(dim=1)
            score_accum = score_accum + contribution / num_strata
            del v_batch, contribution
        all_scores.append(score_accum.detach().cpu())
    return torch.cat(all_scores)


# --------------------------------------------------------------------------------------
# Metrics (self-contained, no scipy -- avoids the Windows/conda libiomp5md.dll conflict
# found in the historical-precision diagnostic)
# --------------------------------------------------------------------------------------


def pearson_corr(a: Tensor, b: Tensor) -> float:
    a, b = a.double(), b.double()
    a, b = a - a.mean(), b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-30)).item()


def _rank(x: Tensor) -> Tensor:
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(x.numel(), dtype=torch.float64, device=x.device)
    return ranks


def spearman_corr(a: Tensor, b: Tensor) -> float:
    return pearson_corr(_rank(a), _rank(b))


def top_q_overlap(a: Tensor, b: Tensor, q: float) -> float:
    n = a.numel()
    k = max(1, int(round(q * n)))
    top_a = set(torch.topk(a, k).indices.tolist())
    top_b = set(torch.topk(b, k).indices.tolist())
    return len(top_a & top_b) / len(top_a)


def cv(x: Tensor, dim=None) -> Tensor:
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=True)
    return std / (mean.abs() + 1e-30)


# --------------------------------------------------------------------------------------


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    denoiser, action_dim = load_converged_denoiser(device)
    params = selected_parameters(denoiser)
    rew_end_model = build_rew_end_model(device, action_dim)
    dataset = load_train_dataset()
    print(f"dataset: {dataset.num_episodes} episodes, N={dataset.num_steps} steps", flush=True)

    h_D_full = get_or_compute_h_D_full(denoiser, params, dataset, device)
    d_S = sum(p.numel() for p in params)
    assert h_D_full.shape == (d_S,)
    print(f"h_D_full ready: shape={tuple(h_D_full.shape)}, min={h_D_full.min().item():.4g}, "
          f"max={h_D_full.max().item():.4g}", flush=True)

    if CANDIDATES_PATH.is_file():
        print(f"Frozen candidates already exist at {CANDIDATES_PATH}; loading instead of regenerating.", flush=True)
        candidates = load_frozen_candidates(device)
    else:
        print(f"Generating {NUM_CANDIDATES} frozen candidates (B_AC={B_AC}, H={HORIZON})...", flush=True)
        candidates = generate_frozen_candidates(denoiser, rew_end_model, dataset, device, action_dim, seed=0)
        save_candidates(candidates)
        print(f"Saved frozen candidates to {CANDIDATES_PATH}.", flush=True)

    verify_candidates(candidates)

    # Bit-identical-on-reload check
    reloaded = load_frozen_candidates(device)
    all_equal = all(
        torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
        for a, b in zip(candidates, reloaded)
    )
    print(f"bit-identical on reload: {all_equal}", flush=True)
    assert all_equal

    print("\nSetup complete.", flush=True)


if __name__ == "__main__":
    main()
