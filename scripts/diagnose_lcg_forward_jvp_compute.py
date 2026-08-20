#! /usr/bin/env python
"""
LCG forward/JVP diagnostic -- Stage F4 compute pass: 3-stratum Full-CRN forward-mode JVP
scorer, 8 outer seeds x 16 banks, over the same 480 frozen candidates used throughout.

Genuine forward-mode AD via diagnose_lcg_forward_jvp_common.score_one_jvp_bank
(torch.func.jvp + functional_call, EDM-cancelled through F_theta) -- validated against an
explicit Jacobian and against the backward Hutchinson estimator's exact-trace target in
Stage F2 before this script was ever run.

Checkpointed after every seed (16 banks each) so a crash loses at most one seed's compute.
K in {1,2,4,8,16} is reconstructed offline by averaging the first K cached banks per seed --
no extra JVPs are spent on the K sweep.

Usage:
    python scripts/diagnose_lcg_forward_jvp_compute.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import diagnose_lcg_backward_variance_setup as setup
import diagnose_lcg_forward_jvp_common as jvp_common

NUM_OUTER_SEEDS = 8
NUM_BANKS = 16
CHUNK_SIZE = 4
RESULTS_PATH = setup.DIAG_DIR / "forward_jvp_raw_scores.pt"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  num_banks={NUM_BANKS}  chunk_size={CHUNK_SIZE}", flush=True)

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
        if existing is not None and existing.shape[0] >= NUM_BANKS:
            print(f"[seed {seed}] already have {existing.shape[0]} banks, skipping.", flush=True)
            continue

        start_k = 0 if existing is None else existing.shape[0]
        rows = [] if existing is None else [existing]
        t0 = time.time()
        for k in range(start_k, NUM_BANKS):
            bank = jvp_common.make_jvp_bank(
                setup.SIGMA_CFG, torch.Size([1, 3, 64, 64]), d_S, device,
                num_strata=setup.NUM_STRATA, seed=seed * 100000 + k,
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
        print(f"[seed {seed}] banks={seed_rows.shape[0]:>3}  seed_time={elapsed:>7.1f}s  "
              f"total_elapsed={total_elapsed:>7.1f}s (checkpointed)", flush=True)

    print(f"\nAll {NUM_OUTER_SEEDS} seeds complete. Cache: {RESULTS_PATH}", flush=True)
    all_finite = all(torch.isfinite(v).all().item() for v in cache.values())
    all_nonneg = all((v >= 0).all().item() for v in cache.values())
    print(f"all cached scores finite: {all_finite}  all nonneg: {all_nonneg}", flush=True)
    assert all_finite and all_nonneg

    print("\nForward-JVP compute complete.", flush=True)


if __name__ == "__main__":
    main()
