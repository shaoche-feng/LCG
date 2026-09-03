#! /usr/bin/env python
"""
Stage 6 Part 11: runtime/memory sanity check for the production forward-JVP candidate
scorer (src/lcg/forward_jvp.py + src/lcg/intrinsic_reward.py's
make_lcg_forward_jvp_intrinsic_reward_fn), on the SAME 480-candidate workload used
throughout the forward-JVP diagnostics. This calls the actual production functions
(not a re-implementation) with M=candidate_num_mc=24, Full CRN, chunk_size=4 -- the
exact configuration LCGLifecycle wires up by default -- to confirm the production
integration is not dramatically slower/heavier than the already-profiled diagnostic
path (M=24 took ~78.9s / ~283MB peak_alloc there; the legacy backward/VJP K=16
3-stratum scorer took ~241.3s / ~666.5MB peak_alloc for comparison).

Usage:
    python scripts/forward_JVP/integration/profile_production_scorer.py
"""
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
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "backward_VJP" / "3-stratum"))

import torch

import diagnose_lcg_backward_variance_setup as setup
from lcg.forward_jvp import (
    frozen_named_parameters,
    make_forward_jvp_simple_mc_bank,
    score_one_jvp_bank,
    selected_named_parameters,
)

CHUNK_SIZE = 4
CANDIDATE_NUM_MC = 24
NUM_REPEATS = 3


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    denoiser, action_dim = setup.load_converged_denoiser(device)
    h_D_full = torch.load(setup.H_D_FULL_PATH, map_location=device, weights_only=True)
    candidates = setup.load_frozen_candidates(device)
    setup.verify_candidates(candidates)
    num_candidates = len(candidates)

    theta_s_named = selected_named_parameters(denoiser)
    frozen_named = frozen_named_parameters(denoiser, theta_s_named)
    h_D_inv_sqrt = h_D_full.rsqrt()

    print("\n" + "=" * 88)
    print(f"PRODUCTION SCORER -- lcg.forward_jvp.score_one_jvp_bank, M={CANDIDATE_NUM_MC}, "
          f"Full CRN, chunk_size={CHUNK_SIZE}, {num_candidates} candidates")
    print("=" * 88)

    times = []
    for rep in range(NUM_REPEATS):
        bank = make_forward_jvp_simple_mc_bank(
            setup.SIGMA_CFG, torch.Size([1, 3, 64, 64]), h_D_full.numel(), device,
            num_samples=CANDIDATE_NUM_MC, seed=555000 + rep,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        scores = score_one_jvp_bank(
            denoiser, theta_s_named, frozen_named, theta_s_named, h_D_full, h_D_inv_sqrt, bank, candidates, CHUNK_SIZE
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        peak_alloc = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else float("nan")
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1e6 if device.type == "cuda" else float("nan")
        finite_ok = torch.isfinite(scores).all().item()
        nonneg_ok = (scores >= 0).all().item()
        print(f"  rep {rep}: elapsed={elapsed:.2f}s  ms/candidate={1000*elapsed/num_candidates:.3f}  "
              f"peak_alloc={peak_alloc:.1f}MB  peak_reserved={peak_reserved:.1f}MB  "
              f"scores finite={finite_ok} nonneg={nonneg_ok}  "
              f"mean={scores.mean().item():.3f} min={scores.min().item():.3f} max={scores.max().item():.3f}")

    print(f"\nmean elapsed over {NUM_REPEATS} reps: {sum(times)/len(times):.2f}s  min={min(times):.2f}s  max={max(times):.2f}s")
    print("\nReference comparison points (measured earlier in this diagnostic effort, same hardware):")
    print("  forward-JVP diagnostic M=24 (profile_m24_to_m60.py):  78.92s total,  283.1MB peak_alloc, 327.2MB peak_reserved")
    print("  backward/VJP 3-stratum K=16 (partF_profiling):       241.3s total,  666.5MB peak_alloc")
    ratio_vs_diag = (sum(times) / len(times)) / 78.92
    print(f"\nproduction/diagnostic runtime ratio: {ratio_vs_diag:.3f} (expect close to 1.0 -- same underlying code path)")
    print("\nProfiling complete.", flush=True)


if __name__ == "__main__":
    main()
