#! /usr/bin/env python
"""
Forward-JVP sampling diagnostic -- Section 12: runtime/memory check, simple MC M=24 vs
stratified K=8 (both cost exactly 24 JVPs/candidate over 480 candidates).

Reuses the exact same score_one_jvp_bank path as the compute scripts (num_strata=1 banks
x24 for simple MC, num_strata=3 banks x8 for stratified) -- not a separate profiling
implementation, so the timings reflect the real scoring path. Banks/samples generated one
at a time (never pre-materialized), matching the lesson already applied in Stage F6.

Usage:
    python scripts/diagnose_lcg_forward_simple_mc_runtime.py
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
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "forward_JVP" / "3-stratum_CRN"))

import torch

import diagnose_lcg_backward_variance_setup as setup
import diagnose_lcg_forward_jvp_common as jvp_common

CHUNK_SIZE = 4
RESULTS_PATH = setup.DIAG_DIR / "forward_simple_mc_runtime.pt"


def profile_case(name, denoiser, theta_s_named, frozen_named, h_D, h_D_inv_sqrt, num_strata, num_banks, candidates, device):
    print(f"\n--- profiling: {name} (num_strata={num_strata}, num_banks={num_banks}, "
          f"total_jvps/candidate={num_strata * num_banks}) ---", flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    total_scores = torch.zeros(len(candidates), device=device)
    t0 = time.perf_counter()
    for b in range(num_banks):
        bank = jvp_common.make_jvp_bank(setup.SIGMA_CFG, torch.Size([1, 3, 64, 64]), h_D.numel(), device,
                                         num_strata=num_strata, seed=500000 + b)
        s = jvp_common.score_one_jvp_bank(
            denoiser, theta_s_named, frozen_named, theta_s_named, h_D, h_D_inv_sqrt, bank, candidates, CHUNK_SIZE
        )
        total_scores += s.to(device) / num_banks
    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    peak_alloc = torch.cuda.max_memory_allocated(device) / 1e6
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1e6
    num_candidates = len(candidates)
    total_jvps = num_strata * num_banks * num_candidates
    result = dict(
        name=name, total_time_s=total_time, ms_per_candidate=1000 * total_time / num_candidates,
        ms_per_jvp=1000 * total_time / total_jvps, total_jvps=total_jvps,
        peak_alloc_mb=peak_alloc, peak_reserved_mb=peak_reserved,
        score_finite=bool(torch.isfinite(total_scores).all().item()),
    )
    print(f"    total_time={total_time:.2f}s  total_jvps={total_jvps}  "
          f"ms/candidate={result['ms_per_candidate']:.3f}  ms/jvp={result['ms_per_jvp']:.4f}")
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

    print("\n" + "=" * 88)
    print("SECTION 12 -- runtime/memory: simple MC M=24 vs stratified K=8 (both 24 JVPs/candidate)")
    print("=" * 88)

    results = []
    results.append(profile_case("simple_MC_M24", denoiser, theta_s_named, frozen_named, h_D_full, h_D_inv_sqrt,
                                 num_strata=1, num_banks=24, candidates=candidates, device=device))
    results.append(profile_case("stratified_K8", denoiser, theta_s_named, frozen_named, h_D_full, h_D_inv_sqrt,
                                 num_strata=3, num_banks=8, candidates=candidates, device=device))

    torch.save(results, RESULTS_PATH)

    print("\n" + "=" * 88)
    print("SECTION 12 -- summary")
    print("=" * 88)
    mc, strat = results[0], results[1]
    print(f"\n{'case':<16} {'total_s':>9} {'ms/cand':>9} {'ms/jvp':>8} {'peak_alloc_MB':>14} {'peak_reserved_MB':>17}")
    for r in results:
        print(f"{r['name']:<16} {r['total_time_s']:>9.2f} {r['ms_per_candidate']:>9.3f} {r['ms_per_jvp']:>8.4f} "
              f"{r['peak_alloc_mb']:>14.1f} {r['peak_reserved_mb']:>17.1f}")
    print(f"\nruntime ratio (simple_MC/stratified): {mc['total_time_s'] / strat['total_time_s']:.3f} (expect ~1.0, same total JVP count)")

    print("\nSection 12 complete.", flush=True)


if __name__ == "__main__":
    main()
