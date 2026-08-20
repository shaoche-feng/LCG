#! /usr/bin/env python
"""
LCG backward/VJP variance diagnostic -- Part B (independent MC) + Part C (CRN), combined
compute pass.

For each of 8 outer seeds and each of {independent, CRN} sampling strategies, computes and
caches NUM_BANKS=16 individual per-bank scalar scores for every one of the 480 frozen
candidates (a (NUM_BANKS, 480) tensor per (seed, condition)). K=1..16 metrics are then all
derivable OFFLINE from this single cache (r_K = mean of the first K banks) -- exactly the
implementation optimization requested, and exactly what Parts B/C/D/E's analysis script
consumes. No extra VJPs are spent recomputing lower K values.

Independent: bank k = one IndependentBank drawn fresh for all 480 candidates (each
candidate gets its own (sigma,eps,xi) per stratum). CRN: bank k = one CRNBank shared
(broadcast) across all 480 candidates. Both go through the same, unmodified
score_one_bank/compute_vjp_batched primitive from the setup module -- the ONLY difference
is the shape of the sigma/eps/xi tensors fed in, never the estimator's math.

Checkpointed after every (seed, condition) combo (16 combos total) so a crash partway
through the ~50-60 min run loses at most one combo's compute, and reruns resume rather than
restart. Supports extending an existing 16-bank cache to 32 banks in place (adaptive K=32
rule) without recomputing or discarding the first 16 (same seeds -> bit-identical reuse).

Usage:
    python scripts/diagnose_lcg_backward_variance_partBC.py [--extend-to-32]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import diagnose_lcg_backward_variance_setup as setup
from lcg.crn import make_crn_bank_set
from lcg.theta_s import selected_parameters

NUM_OUTER_SEEDS = 8
NUM_BANKS = 16
RAW_SCORES_PATH = setup.DIAG_DIR / "partBC_raw_scores.pt"
SINGLE_SHAPE = (3, 64, 64)


def compute_condition(denoiser, params, h_D, candidates, outer_seed, condition, num_banks, existing_rows=None):
    """Returns a (num_banks, 480) tensor of per-bank scores for this (outer_seed, condition).
    If existing_rows is given (K_existing, 480), only computes banks K_existing..num_banks-1
    and concatenates -- the K=32 extension path."""
    start_k = 0 if existing_rows is None else existing_rows.shape[0]
    if start_k >= num_banks:
        return existing_rows[:num_banks]

    rows = []
    device = h_D.device
    if condition == "crn":
        base_seed = outer_seed * 100000
        banks = make_crn_bank_set(setup.SIGMA_CFG, torch.Size([1] + list(SINGLE_SHAPE)), device,
                                   num_crn_banks=num_banks, num_strata=setup.NUM_STRATA, seed=base_seed)
        for k in range(start_k, num_banks):
            s = setup.score_one_bank(denoiser, params, h_D, banks[k], candidates, setup.CHUNK_SIZE, independent=False)
            rows.append(s)
    else:
        assert condition == "independent"
        base_seed = outer_seed * 100000 + 50000
        for k in range(start_k, num_banks):
            bank = setup.make_independent_bank(
                setup.SIGMA_CFG, setup.NUM_CANDIDATES, SINGLE_SHAPE, device,
                num_strata=setup.NUM_STRATA, seed=base_seed + k,
            )
            s = setup.score_one_bank(denoiser, params, h_D, bank, candidates, setup.CHUNK_SIZE, independent=True)
            rows.append(s)

    new_rows = torch.stack(rows, dim=0)  # (num_banks - start_k, 480)
    if existing_rows is not None:
        return torch.cat([existing_rows, new_rows], dim=0)
    return new_rows


def main():
    extend_to_32 = "--extend-to-32" in sys.argv
    num_banks = 32 if extend_to_32 else NUM_BANKS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  num_banks={num_banks}  extend_to_32={extend_to_32}", flush=True)

    denoiser, action_dim = setup.load_converged_denoiser(device)
    params = selected_parameters(denoiser)
    dataset = setup.load_train_dataset()
    h_D_full = setup.get_or_compute_h_D_full(denoiser, params, dataset, device)
    candidates = setup.load_frozen_candidates(device)
    setup.verify_candidates(candidates)

    cache = {}
    if RAW_SCORES_PATH.is_file():
        cache = torch.load(RAW_SCORES_PATH, map_location="cpu", weights_only=True)
        print(f"Loaded existing cache with {len(cache)} (seed,condition) entries from {RAW_SCORES_PATH}.", flush=True)

    combos = [(s, c) for s in range(NUM_OUTER_SEEDS) for c in ["independent", "crn"]]
    t_start = time.time()
    for combo_i, (seed, condition) in enumerate(combos):
        key = (seed, condition)
        existing = cache.get(key)
        if existing is not None and existing.shape[0] >= num_banks:
            print(f"[{combo_i + 1}/{len(combos)}] seed={seed} condition={condition}: "
                  f"already have {existing.shape[0]} banks, skipping.", flush=True)
            continue

        t0 = time.time()
        rows = compute_condition(denoiser, params, h_D_full, candidates, seed, condition, num_banks, existing)
        elapsed = time.time() - t0
        cache[key] = rows
        torch.save(cache, RAW_SCORES_PATH)
        total_elapsed = time.time() - t_start
        print(f"[{combo_i + 1}/{len(combos)}] seed={seed:>2} condition={condition:<12} "
              f"banks={rows.shape[0]:>3}  combo_time={elapsed:>7.1f}s  total_elapsed={total_elapsed:>7.1f}s "
              f"(checkpointed)", flush=True)

    print(f"\nAll {len(combos)} (seed, condition) combos complete. Cache: {RAW_SCORES_PATH}", flush=True)

    # sanity: finite check
    all_finite = all(torch.isfinite(v).all().item() for v in cache.values())
    print(f"all cached scores finite: {all_finite}", flush=True)
    assert all_finite

    print("\nPart B/C compute complete.", flush=True)


if __name__ == "__main__":
    main()
