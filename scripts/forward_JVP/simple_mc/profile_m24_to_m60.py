#! /usr/bin/env python
"""
Forward simple-MC JVP: continuous runtime profiling from M=1 to M=60 in a single
uninterrupted pass (one CUDA peak-memory reset at the start, cumulative wall-clock time
recorded after every sample) -- gives real measured runtime at every M in this range,
including M=24 (reference), M_0.90=51, and everything in between, rather than
extrapolating from a couple of discrete profiled points. Same genuine JVP scoring path as
all prior profiling (diagnose_lcg_forward_jvp_common.score_one_jvp_bank, banks generated
one at a time, never pre-materialized). This is a dedicated timed run, independent of the
already-cached forward_simple_mc_raw_scores.pt (whose per-sample timings are fragmented
across many separate process launches with per-launch model-loading overhead, not
representative of true steady-state per-JVP cost).

Usage:
    python scripts/forward_JVP/simple_mc/profile_m24_to_m60.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "backward_VJP" / "3-stratum"))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "forward_JVP" / "3-stratum_CRN"))

import torch

import diagnose_lcg_backward_variance_setup as setup
import diagnose_lcg_forward_jvp_common as jvp_common

CHUNK_SIZE = 4
M_MAX = 60
RESULTS_PATH = setup.DIAG_DIR / "forward_simple_mc_m24_to_m60_profiling.pt"


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
    num_candidates = len(candidates)

    print("\n" + "=" * 88)
    print(f"PART 7 -- continuous runtime profiling, M=1..{M_MAX} in one pass")
    print("=" * 88)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    total_scores = torch.zeros(num_candidates, device=device)
    cumulative_times = []
    t_start = time.perf_counter()
    for m in range(M_MAX):
        bank = jvp_common.make_jvp_bank(setup.SIGMA_CFG, torch.Size([1, 3, 64, 64]), h_D_full.numel(), device,
                                         num_strata=1, seed=910000 + m)
        s = jvp_common.score_one_jvp_bank(
            denoiser, theta_s_named, frozen_named, theta_s_named, h_D_full, h_D_inv_sqrt, bank, candidates, CHUNK_SIZE
        )
        total_scores += s.to(device)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
        cumulative_times.append(elapsed)
        M = m + 1
        if M in (1, 24, 30, 36, 42, 48, 51, 54, 60) or M % 10 == 0:
            print(f"  M={M:>3}  cumulative_time={elapsed:>8.2f}s  ms/candidate={1000*elapsed/num_candidates:>9.3f}  "
                  f"ms/jvp={1000*elapsed/(M*num_candidates):>7.4f}", flush=True)

    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated(device) / 1e6
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1e6

    print(f"\nfinite check: {torch.isfinite(total_scores).all().item()}")
    print(f"peak_alloc={peak_alloc:.1f}MB  peak_reserved={peak_reserved:.1f}MB  "
          f"(one reset at start, so this is the peak over the WHOLE M=1..{M_MAX} run)")

    torch.save(dict(cumulative_times=cumulative_times, peak_alloc_mb=peak_alloc, peak_reserved_mb=peak_reserved,
                     num_candidates=num_candidates), RESULTS_PATH)

    print("\n" + "=" * 88)
    print("PART 7 -- key checkpoints")
    print("=" * 88)
    print(f"\n{'M':>4} {'total_s':>9} {'ms/candidate':>13} {'ms/jvp':>8}")
    for M in [24, 51, 60]:
        t = cumulative_times[M - 1]
        print(f"{M:>4} {t:>9.2f} {1000*t/num_candidates:>13.3f} {1000*t/(M*num_candidates):>8.4f}")
    print(f"\nbackward_K16 (existing, not rerun): 241.3s total, 502.756 ms/candidate, 10.4741 ms/vjp, "
          f"666.5MB peak_alloc")

    t51 = cumulative_times[50]
    print(f"\nruntime(M_0.90=51) = {t51:.2f}s  vs  backward K16 = 241.3s  "
          f"({'FASTER' if t51 < 241.3 else 'SLOWER'}, ratio={t51/241.3:.3f})")

    print("\nPart 7 complete.", flush=True)


if __name__ == "__main__":
    main()
