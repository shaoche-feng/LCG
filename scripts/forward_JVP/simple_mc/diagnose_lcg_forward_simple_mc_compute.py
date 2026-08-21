#! /usr/bin/env python
"""
Forward-JVP sampling diagnostic -- simple MC vs 3-stratum, compute pass.

Simple-MC forward estimator: for each MC sample m, one UNSTRATIFIED (sigma, eps, eta)
triple, drawn straight from p_train(sigma) (not restricted to a stratum quantile window),
shared (Full CRN) across all 480 candidates -- s_m(x) = 2||J_F(x;sigma_m,eps_m) z_m||^2,
z_m = H_D^{-1/2} eta_m. Exactly one JVP/candidate per sample.

Reuses diagnose_lcg_forward_jvp_common.py's make_jvp_bank/score_one_jvp_bank completely
unmodified: sample_sigma_stratum(cfg, 0, 1, n, device) already collapses to the full
unstratified p_train(sigma) distribution when num_strata=1 (its quantile window becomes
[eps, 1-eps] = the whole range) -- this was already established and used in the backward
simple-MC diagnostic, and holds identically here. So a "JVPBank" with num_strata=1 IS one
simple-MC sample; no new sampling code needed, no changes to the validated forward JVP
scorer or torch.no_grad() protection.

8 seeds x 24 samples, checkpointed after every seed. M in {3,6,12,24} reconstructed offline
via cumulative averaging of the first M cached samples per seed.

Usage:
    python scripts/diagnose_lcg_forward_simple_mc_compute.py
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

NUM_OUTER_SEEDS = 8
NUM_SAMPLES = 24
CHUNK_SIZE = 4
RESULTS_PATH = setup.DIAG_DIR / "forward_simple_mc_raw_scores.pt"


def main():
    target_m = NUM_SAMPLES
    for arg in sys.argv[1:]:
        if arg.startswith("--target-m="):
            target_m = int(arg.split("=", 1)[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  num_samples={target_m}  chunk_size={CHUNK_SIZE}", flush=True)

    denoiser, action_dim = setup.load_converged_denoiser(device)
    h_D_full = torch.load(setup.H_D_FULL_PATH, map_location=device, weights_only=True)
    candidates = setup.load_frozen_candidates(device)
    setup.verify_candidates(candidates)

    theta_s_named = jvp_common.selected_named_parameters(denoiser)
    frozen_named = jvp_common.frozen_named_parameters(denoiser, theta_s_named)
    d_S = sum(p.numel() for p in theta_s_named.values())
    jvp_common.assert_setup_valid(theta_s_named, h_D_full, d_S)
    h_D_inv_sqrt = h_D_full.rsqrt()

    cache = {}
    if RESULTS_PATH.is_file():
        cache = torch.load(RESULTS_PATH, map_location="cpu", weights_only=True)
        print(f"Loaded existing cache with {len(cache)} seed entries from {RESULTS_PATH}.", flush=True)

    t_start = time.time()
    for seed in range(NUM_OUTER_SEEDS):
        existing = cache.get(seed)
        if existing is not None and existing.shape[0] >= target_m:
            print(f"[seed {seed}] already have {existing.shape[0]} samples, skipping.", flush=True)
            continue

        start_m = 0 if existing is None else existing.shape[0]
        rows = [] if existing is None else [existing]
        t0 = time.time()
        for m in range(start_m, target_m):
            # num_strata=1 -> one unstratified (sigma, eps, eta) triple, Full CRN
            bank = jvp_common.make_jvp_bank(
                setup.SIGMA_CFG, torch.Size([1, 3, 64, 64]), d_S, device,
                num_strata=1, seed=seed * 200000 + m,
            )
            s = jvp_common.score_one_jvp_bank(
                denoiser, theta_s_named, frozen_named, theta_s_named, h_D_full, h_D_inv_sqrt,
                bank, candidates, CHUNK_SIZE,
            )
            rows.append(s.unsqueeze(0))
        seed_rows = torch.cat(rows, dim=0)
        cache[seed] = seed_rows
        torch.save(cache, RESULTS_PATH)

        elapsed = time.time() - t0
        total_elapsed = time.time() - t_start
        print(f"[seed {seed}] samples={seed_rows.shape[0]:>3}  seed_time={elapsed:>7.1f}s  "
              f"total_elapsed={total_elapsed:>7.1f}s (checkpointed)", flush=True)

    print(f"\nAll {NUM_OUTER_SEEDS} seeds complete. Cache: {RESULTS_PATH}", flush=True)
    all_finite = all(torch.isfinite(v).all().item() for v in cache.values())
    all_nonneg = all((v >= 0).all().item() for v in cache.values())
    print(f"all cached scores finite: {all_finite}  all nonneg: {all_nonneg}", flush=True)
    assert all_finite and all_nonneg

    print("\nForward simple-MC compute complete.", flush=True)


if __name__ == "__main__":
    main()
