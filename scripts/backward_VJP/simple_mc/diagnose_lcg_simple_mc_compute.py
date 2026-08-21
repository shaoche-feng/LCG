#! /usr/bin/env python
"""
Simple-MC vs 3-stratum diagnostic -- main compute pass.

For each of 8 outer seeds and each of 3 conditions (independent, xi_crn, full_crn),
computes and caches NUM_SAMPLES=12 individual MC scalar scores for every one of the 480
frozen candidates (a (NUM_SAMPLES, 480) tensor per (seed, condition)). M=1..12 metrics are
all derivable OFFLINE from this cache (r_M = mean of the first M samples) -- the same
cumulative-averaging optimization used in the backward/VJP variance diagnostic, and no
different from what's requested here (Section 5).

Checkpointed after every (seed, condition) combo so a crash loses at most one combo, and
reruns resume. Supports extending an existing 12-sample cache to 24 in place (Section 5's
optional adaptive extension) without recomputing or discarding the first 12.

Usage:
    python scripts/diagnose_lcg_simple_mc_compute.py [--extend-to-24]
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
from lcg.theta_s import selected_parameters

NUM_OUTER_SEEDS = 8
NUM_SAMPLES = 12
RESULTS_PATH = setup.DIAG_DIR / "simple_mc_raw_scores.pt"

# Fixed (not Python's randomized hash()) per-condition seed offset, so resumed/extended
# runs in a fresh process reproduce the exact same seed sequence as the original run.
CONDITION_SEED_OFFSET = {"independent": 0, "xi_crn": 11, "full_crn": 22}


def compute_condition(denoiser, params, h_D, candidates, outer_seed, condition, num_samples, existing_rows=None):
    start_m = 0 if existing_rows is None else existing_rows.shape[0]
    if start_m >= num_samples:
        return existing_rows[:num_samples]

    device = h_D.device
    base_seed = outer_seed * 1_000_000 + CONDITION_SEED_OFFSET[condition]
    rows = []
    for m in range(start_m, num_samples):
        sample = mc.make_mc_sample(condition, setup.SIGMA_CFG, setup.NUM_CANDIDATES, device, seed=base_seed + m)
        s = mc.score_mc_sample(denoiser, params, h_D, sample, candidates, setup.CHUNK_SIZE, device)
        rows.append(s)

    new_rows = torch.stack(rows, dim=0)
    if existing_rows is not None:
        return torch.cat([existing_rows, new_rows], dim=0)
    return new_rows


def main():
    extend_to_24 = "--extend-to-24" in sys.argv
    num_samples = 24 if extend_to_24 else NUM_SAMPLES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  num_samples={num_samples}  extend_to_24={extend_to_24}", flush=True)

    denoiser, action_dim = setup.load_converged_denoiser(device)
    params = selected_parameters(denoiser)
    h_D_full = torch.load(setup.H_D_FULL_PATH, map_location=device, weights_only=True)
    candidates = setup.load_frozen_candidates(device)
    setup.verify_candidates(candidates)

    cache = {}
    if RESULTS_PATH.is_file():
        cache = torch.load(RESULTS_PATH, map_location="cpu", weights_only=True)
        print(f"Loaded existing cache with {len(cache)} (seed,condition) entries from {RESULTS_PATH}.", flush=True)

    combos = [(s, c) for s in range(NUM_OUTER_SEEDS) for c in mc.CONDITIONS]
    t_start = time.time()
    for combo_i, (seed, condition) in enumerate(combos):
        key = (seed, condition)
        existing = cache.get(key)
        if existing is not None and existing.shape[0] >= num_samples:
            print(f"[{combo_i + 1}/{len(combos)}] seed={seed} condition={condition}: "
                  f"already have {existing.shape[0]} samples, skipping.", flush=True)
            continue

        t0 = time.time()
        rows = compute_condition(denoiser, params, h_D_full, candidates, seed, condition, num_samples, existing)
        elapsed = time.time() - t0
        cache[key] = rows
        torch.save(cache, RESULTS_PATH)
        total_elapsed = time.time() - t_start
        print(f"[{combo_i + 1}/{len(combos)}] seed={seed:>2} condition={condition:<12} "
              f"samples={rows.shape[0]:>3}  combo_time={elapsed:>7.1f}s  total_elapsed={total_elapsed:>7.1f}s "
              f"(checkpointed)", flush=True)

    print(f"\nAll {len(combos)} (seed, condition) combos complete. Cache: {RESULTS_PATH}", flush=True)
    all_finite = all(torch.isfinite(v).all().item() for v in cache.values())
    print(f"all cached scores finite: {all_finite}", flush=True)
    assert all_finite

    print("\nSimple-MC compute complete.", flush=True)


if __name__ == "__main__":
    main()
