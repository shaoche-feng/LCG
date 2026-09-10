#! /usr/bin/env python
"""
Historical precision B-scaling diagnostic, N~5000 slice: does the absolute historical
subset size B that suffices at N=1540 still suffice at a much larger replay, or does the
required B grow with N? Companion to
scripts/historical_precision_diagnostic/diagnose_historical_B_scaling.py (the N=1540 Stage A
run) -- same estimators, same metrics, same nested-subset design, run against the new matched
N~5000 checkpoint/replay pair produced by build_replay_N5000.py + train_checkpoint_N5000.py.

Diagnostic only -- does not modify production historical_precision(), LCGConfig, or any
estimator. Candidate scoring is economized per the task spec: the h_D-level comparison (cheap,
pure cached-summation) covers the full B grid, but candidate scoring (expensive) only covers
B={40,80,160,320,640,h_D_full} plus B=1280 IF B=640 is not yet effectively converged.

Usage:
    python scripts/historical_precision_diagnostic/diagnose_historical_B_scaling_N5000.py
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while not ((p / "src").is_dir() and (p / "scripts").is_dir()):
        if p.parent == p:
            raise RuntimeError("could not locate LCG repo root")
        p = p.parent
    return p


_REPO_ROOT = _find_repo_root(Path(__file__).parent)
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import torch

from data import Dataset
from lcg.forward_jvp import frozen_named_parameters, make_jvp_bank, score_one_jvp_bank, selected_named_parameters
from lcg.gauss_newton import compute_vjp
from lcg.precision import load_transition, sample_uniform_historical_transitions
from models.diffusion import Denoiser, DenoiserConfig, DiffusionSampler, DiffusionSamplerConfig, SigmaDistributionConfig
from models.diffusion.denoiser import apply_noise_from_samples, sample_sigma_training_distribution
from models.diffusion.inner_model import InnerModelConfig

N5000_DIR = _REPO_ROOT.parent / "docs" / "lcg_diagnostic" / "historical_precision_B_scaling" / "N5000"
DENOISER_PATH = N5000_DIR / "denoiser_converged.pt"
TRAIN_DATASET_DIR = N5000_DIR / "train_dataset"

N1540_DIR = _REPO_ROOT.parent / "docs" / "lcg_diagnostic" / "historical_precision_B_scaling"
N1540_RESULTS_PATH = N1540_DIR / "results_summary.json"

OUT_DIR = N5000_DIR  # results saved alongside the checkpoint/dataset for this N slice
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_STEPS_CONDITIONING = 4
SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)
DAMPING = 1e-4
BETA = 1.0
M_H = 3  # production historical-precision simple-MC budget

B_GRID = [40, 80, 160, 320, 640, 1280]
NUM_SUBSET_SEEDS = 8
CONTRIBUTION_SEED = 12345

CANDIDATE_SUBSET_SIZE = 160
CANDIDATE_M = 12
CANDIDATE_CHUNK_SIZE = 16
CANDIDATE_BANK_SEED = 424242
CANDIDATE_GENERATION_SEED = 777  # for drawing fresh candidates from the N5000 model

# Economical candidate scoring: always score these plus h_D_full; add 1280 only if 640 isn't
# already effectively converged (per the task's "don't blindly spend compute" instruction).
CANDIDATE_B_MINIMUM = [40, 80, 160, 320, 640]
CONVERGENCE_SPEARMAN_THRESHOLD = 0.998
CONVERGENCE_RMS_RATIO_TOLERANCE = 0.02


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pearson(a, b):
    a, b = a.double(), b.double()
    a, b = a - a.mean(), b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-12)).item()


def spearman(a, b):
    ra = torch.argsort(torch.argsort(a)).double()
    rb = torch.argsort(torch.argsort(b)).double()
    return pearson(ra, rb)


def top_q_overlap(a, b, q):
    k = max(1, int(round(q * a.numel())))
    sa = set(torch.topk(a, k).indices.tolist())
    sb = set(torch.topk(b, k).indices.tolist())
    return len(sa & sb) / k


def h_D_comparison(h_test, h_ref):
    inv_test, inv_ref = h_test.rsqrt(), h_ref.rsqrt()
    return dict(
        raw_pearson=pearson(h_test, h_ref), raw_spearman=spearman(h_test, h_ref),
        raw_rel_l2=((h_test - h_ref).norm() / h_ref.norm()).item(),
        raw_norm_ratio=(h_test.norm() / h_ref.norm()).item(),
        inv_sqrt_pearson=pearson(inv_test, inv_ref), inv_sqrt_spearman=spearman(inv_test, inv_ref),
        inv_sqrt_rel_l2=((inv_test - inv_ref).norm() / inv_ref.norm()).item(),
    )


def candidate_metrics(scores_test, scores_ref):
    return dict(
        spearman=spearman(scores_test, scores_ref), pearson=pearson(scores_test, scores_ref),
        top10=top_q_overlap(scores_test, scores_ref, 0.10), top20=top_q_overlap(scores_test, scores_ref, 0.20),
        rms_ratio=(scores_test.pow(2).mean().sqrt() / scores_ref.pow(2).mean().sqrt()).item(),
    )


def load_denoiser(device):
    ckpt = torch.load(DENOISER_PATH, map_location=device, weights_only=False)
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
    return denoiser, action_dim, ckpt


CANDIDATE_SAMPLER_CFG = DiffusionSamplerConfig(
    num_steps_denoising=3, sigma_min=2e-3, sigma_max=5.0, rho=7, order=1,
    s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0,
)  # matches config/trainer.yaml's world_model_env.diffusion_sampler exactly


def generate_candidates_from_dataset(dataset, denoiser, device, num_candidates, seed):
    """Genuine model-imagined candidates, matching production's actual candidate semantics
    (lcg.intrinsic_reward.imagined_candidates_from_batch: x_obs/x_act = a real conditioning
    window, y_star = the DENOISER'S OWN diffusion-sampled continuation, not the real next
    frame) -- production candidates always come from WorldModelEnv.step(), whose y_star is
    produced by exactly this DiffusionSampler.sample() call on the real (obs_buffer,
    act_buffer) conditioning window. This reuses DiffusionSampler directly (no RewEndModel /
    full WorldModelEnv needed, since only y_star is required here, not reward/termination),
    with the same DiffusionSamplerConfig production actually uses (config/trainer.yaml).
    Conditioning windows are drawn uniformly from the N5000 replay via the same
    RNG-isolated sampler historical_precision itself uses."""
    segment_ids = sample_uniform_historical_transitions(dataset, num_candidates, NUM_STEPS_CONDITIONING, seed=seed, replace=False)
    n = NUM_STEPS_CONDITIONING
    obs_windows, act_windows = [], []
    for sid in segment_ids:
        segment = dataset[sid]
        obs_windows.append(segment.obs[:n])
        act_windows.append(segment.act[:n])
    prev_obs = torch.stack(obs_windows).to(device)  # (B, n, C, H, W)
    prev_act = torch.stack(act_windows).to(device)  # (B, n, action_dim)

    sampler = DiffusionSampler(denoiser, CANDIDATE_SAMPLER_CFG)
    y_star, _ = sampler.sample(prev_obs, prev_act)  # (B, C, H, W), no_grad internally

    candidates = []
    for i in range(len(segment_ids)):
        obs_flat = prev_obs[i : i + 1].reshape(1, -1, prev_obs.shape[-2], prev_obs.shape[-1])
        candidates.append((obs_flat, prev_act[i : i + 1], y_star[i : i + 1].detach()))
    return candidates


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    assert DENOISER_PATH.is_file(), f"missing {DENOISER_PATH} -- run train_checkpoint_N5000.py first"
    denoiser, action_dim, ckpt = load_denoiser(device)
    denoiser_sha256 = sha256_of(DENOISER_PATH)
    print(f"checkpoint: {DENOISER_PATH} (total_step={ckpt['step']}) sha256={denoiser_sha256}", flush=True)

    dataset = Dataset(TRAIN_DATASET_DIR, "lcg_diag_train_N5000", cache_in_ram=True)
    dataset.load_from_default_path()
    N = dataset.num_steps
    assert N > 0, f"empty dataset at {TRAIN_DATASET_DIR} -- run build_replay_N5000.py first"

    theta_s_named = selected_named_parameters(denoiser)
    params = list(theta_s_named.values())
    d_S = sum(p.numel() for p in params)
    print(f"N={N}  episodes={dataset.num_episodes}  d_S={d_S}", flush=True)

    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True).stdout.strip()

    manifest = dict(
        git_commit=git_commit, checkpoint_path=str(DENOISER_PATH), checkpoint_sha256=denoiser_sha256,
        checkpoint_total_step=ckpt["step"], checkpoint_continued_from_step=ckpt.get("continued_from_step"),
        checkpoint_stop_reason=ckpt.get("stop_reason"), dataset_path=str(TRAIN_DATASET_DIR), dataset_N=N,
        dataset_num_episodes=dataset.num_episodes, theta_s_dim=d_S, M_h=M_H, beta=BETA, damping=DAMPING,
        sigma_cfg=dict(loc=SIGMA_CFG.loc, scale=SIGMA_CFG.scale, sigma_min=SIGMA_CFG.sigma_min, sigma_max=SIGMA_CFG.sigma_max),
        sigma_offset_noise=denoiser.cfg.sigma_offset_noise, contribution_seed=CONTRIBUTION_SEED,
        B_grid=B_GRID, num_subset_seeds=NUM_SUBSET_SEEDS, candidate_subset_size=CANDIDATE_SUBSET_SIZE,
        candidate_M=CANDIDATE_M, candidate_chunk_size=CANDIDATE_CHUNK_SIZE, candidate_bank_seed=CANDIDATE_BANK_SEED,
        candidate_generation_seed=CANDIDATE_GENERATION_SEED, device=str(device),
    )

    # ------------------------------------------------------------------------------
    # Per-transition contributions g_i, computed ONCE for every transition
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88 + "\nComputing per-transition contributions g_i (production M_h=3, once per transition)\n" + "=" * 88, flush=True)
    t0 = time.perf_counter()
    all_segment_ids = sample_uniform_historical_transitions(dataset, N, NUM_STEPS_CONDITIONING, seed=0, replace=False)

    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(CONTRIBUTION_SEED)

    # g_all is cached on CPU, not GPU: at N~5000, an (N, d_S) float32 GPU tensor is
    # ~11.7GB, which exceeds this GPU's 8GB VRAM (this exact bug caused a CUDA OOM crash
    # and, before that, silently forced the whole contribution loop into Windows'
    # much slower shared-GPU-memory fallback -- the root cause of the ~4x per-VJP
    # slowdown observed relative to the N=1540 run, which comfortably fit in VRAM at
    # ~4GB). VJP computation itself is unaffected (still fully on GPU per transition);
    # only the persistent accumulator moves to host memory.
    g_all = torch.zeros(N, d_S)
    for i, segment_id in enumerate(all_segment_ids):
        obs, act, y = load_transition(dataset, segment_id, NUM_STEPS_CONDITIONING, device)
        g_i = torch.zeros(d_S, device=device)
        for _ in range(M_H):
            sigma = sample_sigma_training_distribution(SIGMA_CFG, 1, device, generator=torch_gen)
            eps = torch.randn(y.shape, dtype=y.dtype, device=y.device, generator=torch_gen)
            eps_offset = torch.randn(y.shape[0], y.shape[1], 1, 1, device=device, generator=torch_gen)
            y_sigma = apply_noise_from_samples(y, sigma, eps, eps_offset, denoiser.cfg.sigma_offset_noise).detach()
            v, _ = compute_vjp(denoiser, params, y_sigma, sigma, obs, act, generator=torch_gen)
            g_i = g_i + (v * v) / M_H
        g_all[i] = g_i.cpu()
        if (i + 1) % 1000 == 0 or (i + 1) == N:
            print(f"  contributions: {i + 1}/{N}  elapsed={time.perf_counter() - t0:.1f}s", flush=True)

    t_contributions = time.perf_counter() - t0
    manifest["t_contributions_s"] = t_contributions
    print(f"\ncontribution computation: {t_contributions:.1f}s total, {1000 * t_contributions / (N * M_H):.2f} ms/VJP", flush=True)

    h_D_full_cpu = DAMPING * torch.ones(d_S) + BETA * g_all.sum(dim=0)
    h_D_full = h_D_full_cpu.to(device)
    torch.save(dict(h_D_full=h_D_full_cpu, g_all_sha256=hashlib.sha256(g_all.numpy().tobytes()).hexdigest()),
               OUT_DIR / "reference_h_D_full.pt")
    print(f"h_D_full: shape={tuple(h_D_full.shape)} finite={torch.isfinite(h_D_full).all().item()} "
          f"positive={(h_D_full > 0).all().item()} min={h_D_full.min().item():.4g} max={h_D_full.max().item():.4g}", flush=True)

    # ------------------------------------------------------------------------------
    # Nested B subsets (cheap, full grid, matches N=1540 design exactly). Uses a
    # cumulative running sum along each seed's permutation (O(d_S) memory) rather than
    # gathering g_all[idx] for each B (which would allocate an (up to 1280, d_S) ~3.35GB
    # temporary per call, on top of g_all itself -- the same class of bug that caused the
    # OOM above). All summation happens on CPU; only the small (d_S,) per-B h_D vector is
    # moved to GPU, once, for downstream scoring.
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88 + "\nNested B-subset construction (cheap: cumulative CPU summation of cached g_i)\n" + "=" * 88, flush=True)
    t0 = time.perf_counter()
    valid_B = sorted(B for B in B_GRID if B <= N)
    skipped_B = [B for B in B_GRID if B > N]
    if skipped_B:
        print(f"  skipping B>N: {skipped_B} (N={N})")
    max_B = max(valid_B)

    h_D_by_B_seed = {B: [] for B in valid_B}
    for seed_idx in range(NUM_SUBSET_SEEDS):
        rng = np.random.default_rng(seed_idx + 500_000)
        perm = rng.permutation(N)[:max_B]
        running_sum = torch.zeros(d_S)
        b_iter = iter(valid_B)
        next_B = next(b_iter)
        for i, idx in enumerate(perm, start=1):
            running_sum += g_all[idx]
            if i == next_B:
                h_D_B = (DAMPING * torch.ones(d_S) + BETA * (N / next_B) * running_sum).to(device)
                h_D_by_B_seed[next_B].append(h_D_B)
                next_B = next(b_iter, None)
                if next_B is None:
                    break
    t_subsets = time.perf_counter() - t0
    manifest["t_subset_construction_s"] = t_subsets
    print(f"subset construction (all {len(valid_B)} B values x {NUM_SUBSET_SEEDS} seeds): {t_subsets:.3f}s", flush=True)

    print(f"\n{'B':>5} {'B/N':>7} {'raw_pearson':>12} {'raw_spearman':>13} {'raw_rel_l2':>11} "
          f"{'inv_sqrt_pearson':>17} {'inv_sqrt_spearman':>18} {'inv_sqrt_rel_l2':>16}")
    h_D_metrics_summary = {}
    for B in valid_B:
        metrics_list = [h_D_comparison(h, h_D_full) for h in h_D_by_B_seed[B]]
        summary = {k: (float(np.mean([m[k] for m in metrics_list])), float(np.std([m[k] for m in metrics_list]))) for k in metrics_list[0]}
        h_D_metrics_summary[B] = summary
        print(f"{B:>5} {B / N:>7.4f} {summary['raw_pearson'][0]:>7.4f}+-{summary['raw_pearson'][1]:<4.4f} "
              f"{summary['raw_spearman'][0]:>8.4f}+-{summary['raw_spearman'][1]:<4.4f} "
              f"{summary['raw_rel_l2'][0]:>6.4f}+-{summary['raw_rel_l2'][1]:<4.4f} "
              f"{summary['inv_sqrt_pearson'][0]:>12.4f}+-{summary['inv_sqrt_pearson'][1]:<4.4f} "
              f"{summary['inv_sqrt_spearman'][0]:>13.4f}+-{summary['inv_sqrt_spearman'][1]:<4.4f} "
              f"{summary['inv_sqrt_rel_l2'][0]:>11.4f}+-{summary['inv_sqrt_rel_l2'][1]:<4.4f}")

    # ------------------------------------------------------------------------------
    # Downstream candidate-score comparison -- economical: minimum set + adaptive 1280
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88 + "\nDownstream candidate-score comparison (economical)\n" + "=" * 88, flush=True)
    t0 = time.perf_counter()
    candidates = generate_candidates_from_dataset(dataset, denoiser, device, CANDIDATE_SUBSET_SIZE, CANDIDATE_GENERATION_SEED)
    print(f"  generated {len(candidates)} candidates from the N5000 replay (seed={CANDIDATE_GENERATION_SEED})", flush=True)

    frozen_named = frozen_named_parameters(denoiser, theta_s_named)
    bank = make_jvp_bank(SIGMA_CFG, torch.Size([1, 3, 64, 64]), d_S, device, num_samples=CANDIDATE_M, seed=CANDIDATE_BANK_SEED)
    print(f"  fixed candidate CRN bank built once (M={CANDIDATE_M}, seed={CANDIDATE_BANK_SEED}), reused for every h_D", flush=True)

    def score_with(h_D_variant):
        return score_one_jvp_bank(denoiser, theta_s_named, frozen_named, theta_s_named, h_D_variant,
                                   h_D_variant.rsqrt(), bank, candidates, CANDIDATE_CHUNK_SIZE)

    scores_full = score_with(h_D_full)
    print(f"  reference (h_D_full) scores: finite={torch.isfinite(scores_full).all().item()} "
          f"nonneg={(scores_full >= 0).all().item()} mean={scores_full.mean().item():.3f}", flush=True)

    candidate_metrics_summary = {}
    scored_B = [B for B in CANDIDATE_B_MINIMUM if B in valid_B]
    print(f"\n{'B':>5} {'spearman':>16} {'pearson':>16} {'top10':>16} {'top20':>16} {'rms_ratio':>16}")
    for B in scored_B:
        per_seed_metrics = [candidate_metrics(score_with(h), scores_full) for h in h_D_by_B_seed[B]]
        summary = {k: (float(np.mean([m[k] for m in per_seed_metrics])), float(np.std([m[k] for m in per_seed_metrics]))) for k in per_seed_metrics[0]}
        candidate_metrics_summary[B] = summary
        print(f"{B:>5} " + " ".join(f"{summary[k][0]:>7.4f}+-{summary[k][1]:<6.4f}" for k in ["spearman", "pearson", "top10", "top20", "rms_ratio"]))

    # Adaptive: only spend compute on B=1280 if B=640 is not already effectively converged.
    b640_converged = False
    if 640 in candidate_metrics_summary:
        s640 = candidate_metrics_summary[640]
        b640_converged = (s640["spearman"][0] >= CONVERGENCE_SPEARMAN_THRESHOLD and
                           abs(s640["rms_ratio"][0] - 1.0) <= CONVERGENCE_RMS_RATIO_TOLERANCE)
    print(f"\nB=640 effectively converged (spearman>={CONVERGENCE_SPEARMAN_THRESHOLD}, "
          f"|rms_ratio-1|<={CONVERGENCE_RMS_RATIO_TOLERANCE}): {b640_converged}", flush=True)

    if not b640_converged and 1280 in valid_B:
        print("  -> scoring B=1280 as well (640 not yet converged)", flush=True)
        per_seed_metrics = [candidate_metrics(score_with(h), scores_full) for h in h_D_by_B_seed[1280]]
        summary = {k: (float(np.mean([m[k] for m in per_seed_metrics])), float(np.std([m[k] for m in per_seed_metrics]))) for k in per_seed_metrics[0]}
        candidate_metrics_summary[1280] = summary
        print(f"{1280:>5} " + " ".join(f"{summary[k][0]:>7.4f}+-{summary[k][1]:<6.4f}" for k in ["spearman", "pearson", "top10", "top20", "rms_ratio"]))
    else:
        print("  -> skipping B=1280 candidate scoring (640 already effectively converged, or 1280>N)", flush=True)

    t_candidate_scoring = time.perf_counter() - t0
    manifest["t_candidate_scoring_s"] = t_candidate_scoring
    manifest["candidate_scored_B_values"] = list(candidate_metrics_summary.keys())
    print(f"\ncandidate-scoring stage: {t_candidate_scoring:.1f}s", flush=True)
    if device.type == "cuda":
        manifest["peak_cuda_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1e6
        print(f"peak CUDA memory: {manifest['peak_cuda_memory_mb']:.1f}MB")

    # ------------------------------------------------------------------------------
    # Direct N=1540 vs N~5000 comparison
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88 + f"\nDirect comparison: N=1540 vs N={N}\n" + "=" * 88, flush=True)
    comparison_table = None
    if N1540_RESULTS_PATH.is_file():
        with open(N1540_RESULTS_PATH) as f:
            n1540_results = json.load(f)
        n1540_N = 1540
        common_B = [B for B in [80, 160, 320, 640] if str(B) in n1540_results["h_D_metrics_by_B"] and B in h_D_metrics_summary]
        print(f"\n{'B':>5} {'B/N (1540)':>11} {'B/N (N5000)':>12} {'inv_sqrt_pear@1540':>19} {'inv_sqrt_pear@N5000':>20} "
              f"{'cand_spearman@1540':>19} {'cand_spearman@N5000':>20}")
        comparison_table = []
        for B in common_B:
            n1540_h = n1540_results["h_D_metrics_by_B"][str(B)]["inv_sqrt_pearson"][0]
            n5000_h = h_D_metrics_summary[B]["inv_sqrt_pearson"][0]
            n1540_c = n1540_results["candidate_metrics_by_B"][str(B)]["spearman"][0] if str(B) in n1540_results["candidate_metrics_by_B"] else None
            n5000_c = candidate_metrics_summary[B]["spearman"][0] if B in candidate_metrics_summary else None
            row = dict(B=B, B_over_N_1540=B / n1540_N, B_over_N_N5000=B / N, inv_sqrt_pearson_1540=n1540_h,
                       inv_sqrt_pearson_N5000=n5000_h, candidate_spearman_1540=n1540_c, candidate_spearman_N5000=n5000_c)
            comparison_table.append(row)
            n1540_c_str = f"{n1540_c:.4f}" if n1540_c is not None else "n/a"
            n5000_c_str = f"{n5000_c:.4f}" if n5000_c is not None else "n/a"
            print(f"{B:>5} {B / n1540_N:>11.4f} {B / N:>12.4f} {n1540_h:>19.4f} {n5000_h:>20.4f} "
                  f"{n1540_c_str:>19} {n5000_c_str:>20}")
    else:
        print(f"  WARNING: {N1540_RESULTS_PATH} not found -- cannot build direct comparison table.", flush=True)

    # ------------------------------------------------------------------------------
    # Decision-gate classification (Case A / B / C)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88 + "\nDecision-gate classification (informal, no fixed threshold)\n" + "=" * 88, flush=True)
    decision_notes = []
    if comparison_table:
        degradations = [row["inv_sqrt_pearson_1540"] - row["inv_sqrt_pearson_N5000"] for row in comparison_table]
        cand_degradations = [
            row["candidate_spearman_1540"] - row["candidate_spearman_N5000"]
            for row in comparison_table if row["candidate_spearman_1540"] is not None and row["candidate_spearman_N5000"] is not None
        ]
        max_h_degradation = max(degradations) if degradations else 0.0
        max_cand_degradation = max(cand_degradations) if cand_degradations else 0.0
        print(f"  max h_D^-1/2 Pearson degradation at matched B (1540 -> N5000): {max_h_degradation:+.4f}")
        if cand_degradations:
            print(f"  max candidate-Spearman degradation at matched B (1540 -> N5000): {max_cand_degradation:+.4f}")
        decision_notes.append(f"max_h_D_inv_sqrt_pearson_degradation={max_h_degradation:.4f}")
        decision_notes.append(f"max_candidate_spearman_degradation={max_cand_degradation:.4f}")
    print("\nSee the accompanying report for the Case A/B/C classification and N~15000 recommendation.", flush=True)

    # ------------------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------------------
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    results_summary = dict(
        N=N, h_D_metrics_by_B={str(B): {k: list(v) for k, v in s.items()} for B, s in h_D_metrics_summary.items()},
        candidate_metrics_by_B={str(B): {k: list(v) for k, v in s.items()} for B, s in candidate_metrics_summary.items()},
        comparison_table=comparison_table, decision_notes=decision_notes,
    )
    with open(OUT_DIR / "results_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2, default=str)

    print(f"\nStage N5000 diagnostic complete. Results saved under {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
