#! /usr/bin/env python
"""
Simple-MC vs 3-stratum diagnostic -- Section 13: compute and memory profiling.

Dedicated, clean profiling runs (fresh torch.cuda.reset_peak_memory_stats() per case,
CUDA-synchronized timing) for M in {1,3,12} x 3 conditions = 9 cases. MC samples are
generated and consumed ONE AT A TIME inside the timed loop (never accumulated into a list)
-- learned directly from the backward/VJP variance diagnostic's Part F, whose first version
pre-built all K bank objects up front and thereby measured an artificial ~3.6x memory
"leak" that was actually just its own list-materialization, not the estimator's real
scaling. This script avoids that from the start.

Usage:
    python scripts/diagnose_lcg_simple_mc_profile.py
"""
import sys
import time
from pathlib import Path

def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while not ((p / "src").is_dir() and (p / "scripts").is_dir()):
        if p.parent == p:
            raise RuntimeError("could not locate LCG repo root (no ancestor has both src/ and scripts/)")
        p = p.parent
    return p


_REPO_ROOT = _find_repo_root(Path(__file__).parent)
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "backward_VJP" / "3-stratum"))

import torch

import diagnose_lcg_backward_variance_setup as setup
import diagnose_lcg_simple_mc_common as mc
from lcg.gauss_newton import differentiable_denoise
from lcg.theta_s import selected_parameters

RESULTS_PATH = setup.DIAG_DIR / "simple_mc_profiling.pt"
M_CASES = [1, 3, 12]


def score_profiled(denoiser, params, h_D, condition, M, candidates, chunk_size, device):
    total_scores = torch.zeros(len(candidates), device=device)
    t_forward = t_backward = t_reduction = 0.0
    num_vjps = 0

    for m in range(M):
        sample = mc.make_mc_sample(condition, setup.SIGMA_CFG, setup.NUM_CANDIDATES, device, seed=700000 + m)
        for start in range(0, len(candidates), chunk_size):
            chunk = candidates[start : start + chunk_size]
            obs_batch = torch.cat([c[0] for c in chunk], dim=0)
            act_batch = torch.cat([c[1] for c in chunk], dim=0)
            y_batch = torch.cat([c[2] for c in chunk], dim=0)
            B = len(chunk)

            sigma = sample.sigma if sample.sigma.shape[0] == 1 else sample.sigma[start : start + B]
            eps = sample.eps if sample.eps.shape[0] == 1 else sample.eps[start : start + B]
            xi = sample.xi if sample.xi.shape[0] == 1 else sample.xi[start : start + B]
            y_sigma_batch = (y_batch + sigma.view(-1, 1, 1, 1) * eps).detach()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            d_theta, w = differentiable_denoise(denoiser, y_sigma_batch, sigma, obs_batch, act_batch)
            torch.cuda.synchronize()
            t_forward += time.perf_counter() - t0

            Bc = d_theta.size(0)
            per_example = (torch.sqrt(2 * w) * xi * d_theta).reshape(Bc, -1).sum(dim=1)
            grad_outputs = torch.eye(Bc, device=per_example.device, dtype=per_example.dtype)
            t0 = time.perf_counter()
            grads = torch.autograd.grad(
                per_example, params, grad_outputs=grad_outputs, is_grads_batched=True,
                retain_graph=False, create_graph=False,
            )
            torch.cuda.synchronize()
            t_backward += time.perf_counter() - t0
            num_vjps += Bc

            t0 = time.perf_counter()
            v_batch = torch.cat([g.reshape(Bc, -1) for g in grads], dim=1)
            contribution = (v_batch.square() / h_D.unsqueeze(0)).sum(dim=1)
            torch.cuda.synchronize()
            t_reduction += time.perf_counter() - t0

            total_scores[start : start + B] += contribution / M
            del v_batch, contribution, grads, d_theta, w

    return total_scores, dict(t_forward=t_forward, t_backward=t_backward, t_reduction=t_reduction, num_vjps=num_vjps)


def profile_case(name, denoiser, params, h_D, candidates, condition, M, device):
    print(f"\n--- profiling: {name} (condition={condition}, M={M}) ---", flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    scores, timing = score_profiled(denoiser, params, h_D, condition, M, candidates, setup.CHUNK_SIZE, device)
    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    peak_alloc = torch.cuda.max_memory_allocated(device) / 1e6
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1e6

    num_candidates = len(candidates)
    total_vjps = M * num_candidates
    result = dict(
        name=name, condition=condition, M=M,
        total_time_s=total_time, ms_per_candidate=1000 * total_time / num_candidates,
        ms_per_vjp=1000 * total_time / total_vjps,
        t_forward_s=timing["t_forward"], t_backward_s=timing["t_backward"], t_reduction_s=timing["t_reduction"],
        total_vjps=total_vjps, peak_alloc_mb=peak_alloc, peak_reserved_mb=peak_reserved,
        score_finite=bool(torch.isfinite(scores).all().item()),
    )
    print(f"    total_time={total_time:.2f}s  forward={timing['t_forward']:.2f}s  "
          f"backward={timing['t_backward']:.2f}s  reduction={timing['t_reduction']:.2f}s")
    print(f"    total_vjps={total_vjps}  ms/candidate={result['ms_per_candidate']:.3f}  "
          f"ms/vjp={result['ms_per_vjp']:.3f}")
    print(f"    peak_alloc={peak_alloc:.1f}MB  peak_reserved={peak_reserved:.1f}MB  "
          f"scores_finite={result['score_finite']}")
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    denoiser, action_dim = setup.load_converged_denoiser(device)
    params = selected_parameters(denoiser)
    h_D_full = torch.load(setup.H_D_FULL_PATH, map_location=device, weights_only=True)
    candidates = setup.load_frozen_candidates(device)
    setup.verify_candidates(candidates)

    print("\n" + "=" * 88)
    print("SECTION 13 -- compute and memory profiling")
    print("=" * 88)

    results = []
    for condition in mc.CONDITIONS:
        for M in M_CASES:
            results.append(profile_case(f"{condition}_M{M}", denoiser, params, h_D_full, candidates, condition, M, device))

    torch.save(results, RESULTS_PATH)

    print("\n" + "=" * 88)
    print("SECTION 13 -- summary table")
    print("=" * 88)
    print(f"\n{'case':<20} {'M':>3} {'total_s':>9} {'fwd_s':>8} {'bwd_s':>8} {'ms/cand':>9} "
          f"{'ms/vjp':>8} {'peak_alloc_MB':>14} {'peak_reserved_MB':>17}")
    for r in results:
        print(f"{r['name']:<20} {r['M']:>3} {r['total_time_s']:>9.2f} {r['t_forward_s']:>8.2f} "
              f"{r['t_backward_s']:>8.2f} {r['ms_per_candidate']:>9.3f} {r['ms_per_vjp']:>8.4f} "
              f"{r['peak_alloc_mb']:>14.1f} {r['peak_reserved_mb']:>17.1f}")

    print(f"\nMemory scaling check (M should primarily increase RUNTIME, not peak memory):")
    for condition in mc.CONDITIONS:
        m1 = next(r for r in results if r["condition"] == condition and r["M"] == 1)
        m12 = next(r for r in results if r["condition"] == condition and r["M"] == 12)
        print(f"  {condition:<12}: M1 peak_alloc={m1['peak_alloc_mb']:.1f}MB -> M12 peak_alloc={m12['peak_alloc_mb']:.1f}MB "
              f"(ratio={m12['peak_alloc_mb'] / m1['peak_alloc_mb']:.2f}x, expect ~1x)  "
              f"runtime ratio={m12['total_time_s'] / m1['total_time_s']:.2f}x (expect ~12x)")

    print("\nSection 13 complete.", flush=True)


if __name__ == "__main__":
    main()
