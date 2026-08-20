#! /usr/bin/env python
"""
LCG backward/VJP variance diagnostic -- Part F: compute and memory profiling.

Dedicated, clean profiling runs (fresh torch.cuda.reset_peak_memory_stats() per case,
CUDA-synchronized timing) for exactly the 4 requested cases: Independent K=1, Independent
K=16, CRN K=1, CRN K=16. Not reused from Part B/C's cache, so each measurement is isolated
(no risk of a later config's activity polluting an earlier one's memory/timing reading).

Reimplements compute_vjp_batched's forward/backward/reduction split ONLY for timing
instrumentation -- same math, same torch.autograd.grad(..., is_grads_batched=True) call,
same broadcasting rules for independent vs CRN sigma/eps/xi shapes as
diagnose_lcg_backward_variance_setup.score_one_bank. No src/lcg/*.py changes.

Usage:
    python scripts/diagnose_lcg_backward_variance_partF.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import diagnose_lcg_backward_variance_setup as setup
from lcg.crn import make_crn_bank_set
from lcg.gauss_newton import differentiable_denoise
from lcg.theta_s import selected_parameters

SINGLE_SHAPE = (3, 64, 64)
RESULTS_PATH = setup.DIAG_DIR / "partF_profiling.pt"


def score_profiled(denoiser, params, h_D, bank_factory, K, candidates, chunk_size, independent, device):
    """bank_factory(k) -> one bank (CRNBank or IndependentBank), called and consumed ONE AT
    A TIME inside this loop (never accumulated into a list) -- matches
    diagnose_lcg_backward_variance_partBC.py's compute_condition exactly, so peak memory
    reflects realistic sequential-bank usage rather than an artifact of pre-materializing
    all K banks (which independent banks are large enough, at K=16, to make a real
    difference: see the first Part F run's independent_K1->K16 3.64x memory-growth finding,
    caused by exactly that pre-materialization bug in an earlier version of this script)."""
    num_strata = setup.NUM_STRATA
    total_scores = torch.zeros(len(candidates), device=device)
    t_forward = t_backward = t_reduction = 0.0
    num_vjps = 0

    for k in range(K):
        bank = bank_factory(k)
        for start in range(0, len(candidates), chunk_size):
            chunk = candidates[start : start + chunk_size]
            obs_batch = torch.cat([c[0] for c in chunk], dim=0)
            act_batch = torch.cat([c[1] for c in chunk], dim=0)
            y_batch = torch.cat([c[2] for c in chunk], dim=0)
            B = len(chunk)
            chunk_score = torch.zeros(B, device=device)
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
                chunk_score = chunk_score + contribution / num_strata
                torch.cuda.synchronize()
                t_reduction += time.perf_counter() - t0
                del v_batch, contribution, grads, d_theta, w

            total_scores[start : start + B] += chunk_score / K

    return total_scores, dict(t_forward=t_forward, t_backward=t_backward, t_reduction=t_reduction, num_vjps=num_vjps)


def profile_case(name, denoiser, params, h_D, candidates, K, independent, device):
    print(f"\n--- profiling: {name} (K={K}) ---", flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    if independent:
        def bank_factory(k):
            return setup.make_independent_bank(setup.SIGMA_CFG, setup.NUM_CANDIDATES, SINGLE_SHAPE, device,
                                                num_strata=setup.NUM_STRATA, seed=900000 + k)
    else:
        crn_banks = make_crn_bank_set(setup.SIGMA_CFG, torch.Size([1] + list(SINGLE_SHAPE)), device,
                                       num_crn_banks=K, num_strata=setup.NUM_STRATA, seed=800000)

        def bank_factory(k):
            return crn_banks[k]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    scores, timing = score_profiled(denoiser, params, h_D, bank_factory, K, candidates, setup.CHUNK_SIZE, independent, device)
    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    peak_alloc = torch.cuda.max_memory_allocated(device) / 1e6
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1e6

    num_candidates = len(candidates)
    total_vjps = 3 * K * num_candidates
    result = dict(
        name=name, K=K, independent=independent,
        total_time_s=total_time, ms_per_candidate=1000 * total_time / num_candidates,
        ms_per_vjp=1000 * total_time / total_vjps,
        t_forward_s=timing["t_forward"], t_backward_s=timing["t_backward"], t_reduction_s=timing["t_reduction"],
        total_vjps=total_vjps, num_vjps_measured=timing["num_vjps"],
        peak_alloc_mb=peak_alloc, peak_reserved_mb=peak_reserved,
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
    dataset = setup.load_train_dataset()
    h_D_full = setup.get_or_compute_h_D_full(denoiser, params, dataset, device)
    candidates = setup.load_frozen_candidates(device)
    setup.verify_candidates(candidates)

    print("\n" + "=" * 88)
    print("PART F -- compute and memory profiling")
    print("=" * 88)

    results = []
    results.append(profile_case("independent_K1", denoiser, params, h_D_full, candidates, 1, True, device))
    results.append(profile_case("independent_K16", denoiser, params, h_D_full, candidates, 16, True, device))
    results.append(profile_case("crn_K1", denoiser, params, h_D_full, candidates, 1, False, device))
    results.append(profile_case("crn_K16", denoiser, params, h_D_full, candidates, 16, False, device))

    torch.save(results, RESULTS_PATH)

    print("\n" + "=" * 88)
    print("PART F -- summary table")
    print("=" * 88)
    print(f"\n{'case':<18} {'K':>3} {'total_s':>9} {'fwd_s':>8} {'bwd_s':>8} {'ms/cand':>9} "
          f"{'ms/vjp':>8} {'peak_alloc_MB':>14} {'peak_reserved_MB':>17}")
    for r in results:
        print(f"{r['name']:<18} {r['K']:>3} {r['total_time_s']:>9.2f} {r['t_forward_s']:>8.2f} "
              f"{r['t_backward_s']:>8.2f} {r['ms_per_candidate']:>9.3f} {r['ms_per_vjp']:>8.4f} "
              f"{r['peak_alloc_mb']:>14.1f} {r['peak_reserved_mb']:>17.1f}")

    # memory-leak / accumulation check: K=16 peak memory should NOT be ~16x K=1's peak
    ind1 = next(r for r in results if r["name"] == "independent_K1")
    ind16 = next(r for r in results if r["name"] == "independent_K16")
    crn1 = next(r for r in results if r["name"] == "crn_K1")
    crn16 = next(r for r in results if r["name"] == "crn_K16")
    print(f"\nMemory scaling check (K should primarily increase RUNTIME, not peak memory):")
    print(f"  independent: K1 peak_alloc={ind1['peak_alloc_mb']:.1f}MB -> K16 peak_alloc={ind16['peak_alloc_mb']:.1f}MB "
          f"(ratio={ind16['peak_alloc_mb'] / ind1['peak_alloc_mb']:.2f}x, expect ~1x)")
    print(f"  crn:         K1 peak_alloc={crn1['peak_alloc_mb']:.1f}MB -> K16 peak_alloc={crn16['peak_alloc_mb']:.1f}MB "
          f"(ratio={crn16['peak_alloc_mb'] / crn1['peak_alloc_mb']:.2f}x, expect ~1x)")
    print(f"  runtime scaling: independent K1->K16 ratio={ind16['total_time_s'] / ind1['total_time_s']:.2f}x (expect ~16x)")
    print(f"  runtime scaling: crn K1->K16 ratio={crn16['total_time_s'] / crn1['total_time_s']:.2f}x (expect ~16x)")

    print("\nPart F complete.", flush=True)


if __name__ == "__main__":
    main()
