#! /usr/bin/env python
"""
LCG forward/JVP diagnostic -- Stage F6: compute and memory profiling, forward JVP vs
backward VJP, same 480-candidate workload, same 3-stratum Full-CRN sampling design.

Both paths generate/consume one bank at a time (never a pre-materialized list) --
following the lesson learned (and fixed) in the backward/VJP variance diagnostic's Part F,
whose first version measured an artificial memory "leak" caused only by its own bank-list
pre-materialization, not the estimator's real scaling.

Confirms genuine forward-mode execution: dual-tensor JVP outputs must have grad_fn=None
(no reverse-mode graph built), unlike a hypothetical reverse-over-reverse implementation.

Usage:
    python scripts/diagnose_lcg_forward_jvp_F6_profile.py
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
import torch.func as func

import diagnose_lcg_backward_variance_setup as setup
import diagnose_lcg_forward_jvp_common as jvp_common
from lcg.crn import make_crn_bank_set
from lcg.gauss_newton import differentiable_denoise
from lcg.theta_s import selected_parameters

CHUNK_SIZE = 4
K_CASES = [1, 2, 16]
RESULTS_PATH = setup.DIAG_DIR / "forward_jvp_f6_profiling.pt"


# --------------------------------------------------------------------------------------
# Forward JVP profiling (primal-forward time vs JVP/tangent time separated)
# --------------------------------------------------------------------------------------


def score_jvp_profiled(denoiser, theta_s_named, frozen_named, h_D, h_D_inv_sqrt, K, candidates, chunk_size, device):
    num_strata = setup.NUM_STRATA
    total_scores = torch.zeros(len(candidates), device=device)
    t_primal_and_jvp = 0.0  # torch.func.jvp computes both together -- cannot cleanly separate
    t_reduction = 0.0
    num_jvps = 0
    grad_fn_seen = False

    for k in range(K):
        bank = jvp_common.make_jvp_bank(setup.SIGMA_CFG, torch.Size([1, 3, 64, 64]), h_D.numel(), device,
                                         num_strata=num_strata, seed=600000 + k)
        for start in range(0, len(candidates), chunk_size):
            chunk = candidates[start : start + chunk_size]
            obs_batch = torch.cat([c[0] for c in chunk], dim=0)
            act_batch = torch.cat([c[1] for c in chunk], dim=0)
            y_batch = torch.cat([c[2] for c in chunk], dim=0)
            B = len(chunk)
            chunk_score = torch.zeros(B, device=device)

            for m in range(num_strata):
                sigma, eps, eta = bank.sigmas[m], bank.epsilons[m], bank.etas[m]
                with torch.no_grad():
                    z_flat = h_D_inv_sqrt * eta
                tangent_named = jvp_common.unflatten_to_dict(z_flat, theta_s_named)
                y_sigma_batch = (y_batch + sigma.view(-1, 1, 1, 1) * eps).detach()

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                primal_out, jvp_out = jvp_common.jvp_through_F(
                    denoiser, theta_s_named, frozen_named, tangent_named, y_sigma_batch, sigma, obs_batch, act_batch
                )
                torch.cuda.synchronize()
                t_primal_and_jvp += time.perf_counter() - t0
                num_jvps += B
                if not grad_fn_seen:
                    grad_fn_seen = True
                    print(f"    [genuine-forward-mode check] primal_out.grad_fn={primal_out.grad_fn}  "
                          f"jvp_out.grad_fn={jvp_out.grad_fn}  (expect None for both -- no reverse-mode graph)")

                t0 = time.perf_counter()
                contribution = 2.0 * jvp_out.reshape(B, -1).square().sum(dim=1)
                chunk_score = chunk_score + contribution / num_strata
                torch.cuda.synchronize()
                t_reduction += time.perf_counter() - t0
                del primal_out, jvp_out, contribution

            total_scores[start : start + B] += chunk_score / K

    return total_scores, dict(t_primal_and_jvp=t_primal_and_jvp, t_reduction=t_reduction, num_jvps=num_jvps)


# --------------------------------------------------------------------------------------
# Backward VJP profiling (forward vs backward split, same as the earlier diagnostic's F6)
# --------------------------------------------------------------------------------------


def score_vjp_profiled(denoiser, params, h_D, K, candidates, chunk_size, device):
    num_strata = setup.NUM_STRATA
    total_scores = torch.zeros(len(candidates), device=device)
    t_forward = t_backward = t_reduction = 0.0
    num_vjps = 0

    for k in range(K):
        banks = make_crn_bank_set(setup.SIGMA_CFG, torch.Size([1, 3, 64, 64]), device,
                                   num_crn_banks=1, num_strata=num_strata, seed=650000 + k)
        bank = banks[0]
        for start in range(0, len(candidates), chunk_size):
            chunk = candidates[start : start + chunk_size]
            obs_batch = torch.cat([c[0] for c in chunk], dim=0)
            act_batch = torch.cat([c[1] for c in chunk], dim=0)
            y_batch = torch.cat([c[2] for c in chunk], dim=0)
            B = len(chunk)
            chunk_score = torch.zeros(B, device=device)

            for m in range(num_strata):
                sigma, eps, xi = bank.sigmas[m], bank.epsilons[m], bank.xis[m]
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


def profile_forward(name, denoiser, theta_s_named, frozen_named, h_D, h_D_inv_sqrt, K, candidates, device):
    print(f"\n--- profiling FORWARD JVP: {name} (K={K}) ---", flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    scores, timing = score_jvp_profiled(denoiser, theta_s_named, frozen_named, h_D, h_D_inv_sqrt, K, candidates, CHUNK_SIZE, device)
    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    peak_alloc = torch.cuda.max_memory_allocated(device) / 1e6
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1e6
    num_candidates = len(candidates)
    total_jvps = 3 * K * num_candidates
    result = dict(
        name=name, mode="forward_jvp", K=K, total_time_s=total_time,
        ms_per_candidate=1000 * total_time / num_candidates, ms_per_deriv=1000 * total_time / total_jvps,
        t_primal_and_jvp_s=timing["t_primal_and_jvp"], t_reduction_s=timing["t_reduction"],
        total_derivs=total_jvps, peak_alloc_mb=peak_alloc, peak_reserved_mb=peak_reserved,
        score_finite=bool(torch.isfinite(scores).all().item()),
    )
    print(f"    total_time={total_time:.2f}s  primal+jvp={timing['t_primal_and_jvp']:.2f}s  "
          f"reduction={timing['t_reduction']:.2f}s")
    print(f"    total_jvps={total_jvps}  ms/candidate={result['ms_per_candidate']:.3f}  ms/jvp={result['ms_per_deriv']:.4f}")
    print(f"    peak_alloc={peak_alloc:.1f}MB  peak_reserved={peak_reserved:.1f}MB  scores_finite={result['score_finite']}")
    return result


def profile_backward(name, denoiser, params, h_D, K, candidates, device):
    print(f"\n--- profiling BACKWARD VJP: {name} (K={K}) ---", flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    scores, timing = score_vjp_profiled(denoiser, params, h_D, K, candidates, CHUNK_SIZE, device)
    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    peak_alloc = torch.cuda.max_memory_allocated(device) / 1e6
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1e6
    num_candidates = len(candidates)
    total_vjps = 3 * K * num_candidates
    result = dict(
        name=name, mode="backward_vjp", K=K, total_time_s=total_time,
        ms_per_candidate=1000 * total_time / num_candidates, ms_per_deriv=1000 * total_time / total_vjps,
        t_forward_s=timing["t_forward"], t_backward_s=timing["t_backward"], t_reduction_s=timing["t_reduction"],
        total_derivs=total_vjps, peak_alloc_mb=peak_alloc, peak_reserved_mb=peak_reserved,
        score_finite=bool(torch.isfinite(scores).all().item()),
    )
    print(f"    total_time={total_time:.2f}s  forward={timing['t_forward']:.2f}s  backward={timing['t_backward']:.2f}s  "
          f"reduction={timing['t_reduction']:.2f}s")
    print(f"    total_vjps={total_vjps}  ms/candidate={result['ms_per_candidate']:.3f}  ms/vjp={result['ms_per_deriv']:.4f}")
    print(f"    peak_alloc={peak_alloc:.1f}MB  peak_reserved={peak_reserved:.1f}MB  scores_finite={result['score_finite']}")
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    denoiser, action_dim = setup.load_converged_denoiser(device)
    h_D_full = torch.load(setup.H_D_FULL_PATH, map_location=device, weights_only=True)
    candidates = setup.load_frozen_candidates(device)
    setup.verify_candidates(candidates)

    theta_s_named = jvp_common.selected_named_parameters(denoiser)
    frozen_named = jvp_common.frozen_named_parameters(denoiser, theta_s_named)
    h_D_inv_sqrt = h_D_full.rsqrt()
    params = selected_parameters(denoiser)

    print("\n" + "=" * 88)
    print("STAGE F6 -- compute and memory profiling: forward JVP vs backward VJP")
    print("=" * 88)

    results = []
    for K in K_CASES:
        results.append(profile_forward(f"forward_K{K}", denoiser, theta_s_named, frozen_named, h_D_full, h_D_inv_sqrt, K, candidates, device))
    for K in K_CASES:
        results.append(profile_backward(f"backward_K{K}", denoiser, params, h_D_full, K, candidates, device))

    torch.save(results, RESULTS_PATH)

    print("\n" + "=" * 88)
    print("STAGE F6 -- summary table")
    print("=" * 88)
    print(f"\n{'case':<16} {'mode':<14} {'K':>3} {'total_s':>9} {'ms/cand':>9} {'ms/deriv':>9} "
          f"{'peak_alloc_MB':>14} {'peak_reserved_MB':>17}")
    for r in results:
        print(f"{r['name']:<16} {r['mode']:<14} {r['K']:>3} {r['total_time_s']:>9.2f} "
              f"{r['ms_per_candidate']:>9.3f} {r['ms_per_deriv']:>9.4f} {r['peak_alloc_mb']:>14.1f} "
              f"{r['peak_reserved_mb']:>17.1f}")

    print(f"\nDirect ms/derivative comparison (JVP vs VJP), same K:")
    for K in K_CASES:
        fwd = next(r for r in results if r["mode"] == "forward_jvp" and r["K"] == K)
        bwd = next(r for r in results if r["mode"] == "backward_vjp" and r["K"] == K)
        ratio = fwd["ms_per_deriv"] / bwd["ms_per_deriv"]
        print(f"  K={K:>2}: forward={fwd['ms_per_deriv']:.4f}ms/jvp  backward={bwd['ms_per_deriv']:.4f}ms/vjp  "
              f"ratio(fwd/bwd)={ratio:.3f}  ({'forward faster' if ratio < 1 else 'backward faster'})")

    print(f"\nMemory scaling check (K should primarily increase RUNTIME, not peak memory):")
    for mode in ["forward_jvp", "backward_vjp"]:
        k1 = next(r for r in results if r["mode"] == mode and r["K"] == 1)
        k16 = next(r for r in results if r["mode"] == mode and r["K"] == 16)
        print(f"  {mode:<14}: K1 peak_alloc={k1['peak_alloc_mb']:.1f}MB -> K16 peak_alloc={k16['peak_alloc_mb']:.1f}MB "
              f"(ratio={k16['peak_alloc_mb'] / k1['peak_alloc_mb']:.2f}x, expect ~1x)  "
              f"runtime ratio={k16['total_time_s'] / k1['total_time_s']:.2f}x (expect ~16x)")

    print("\nStage F6 complete.", flush=True)


if __name__ == "__main__":
    main()
