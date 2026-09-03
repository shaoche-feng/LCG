#! /usr/bin/env python
"""
Stage 7 continued: probe-batching benchmark (Part 4/5), fixed candidate chunk C=16 --
the empirical winner from the completed candidate-chunk sweep (C in {4..240}, see
docs/lcg_diagnostic/forward_JVP/integration/optimize_scorer_M12.txt): C=16 was fastest
(22.9s) and used the least memory among the near-optimal band (C=16..128 all within
~2s of each other); C=240 hit a genuine GPU-memory cliff (117.8s, Windows shared-memory
fallback) so C=480 was not worth testing.

This script tests probe batching via torch.func.vmap(jvp), P in {1,2,3,4,6,12}, then a
small 2D (candidate_chunk, probe_chunk) grid if vmap helps. Every print flushes
immediately (fixing the buffering issue from the first run of this benchmark).

Usage:
    python scripts/forward_JVP/integration/optimize_scorer_M12_probes.py
"""
import gc
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

import numpy as np
import torch
import torch.func as func

import diagnose_lcg_backward_variance_setup as setup
from lcg.forward_jvp import (
    frozen_named_parameters,
    make_forward_jvp_simple_mc_bank,
    score_one_jvp_bank,
    selected_named_parameters,
    unflatten_to_dict,
)

M = 12
BANK_SEED = 424242  # SAME seed as the candidate-chunk sweep, for a directly comparable bank
FASTEST_C = 16  # empirical winner from the completed candidate-chunk sweep
PROBE_CHUNKS = [1, 2, 3, 4, 6, 12]
NUM_TIMED_REPS = 3
REFERENCE_CHUNK_SIZE = 4


def gpu_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gpu_sync()
        torch.cuda.reset_peak_memory_stats(device)


def peak_stats(device):
    if device.type != "cuda":
        return float("nan"), float("nan")
    return torch.cuda.max_memory_allocated(device) / 1e6, torch.cuda.max_memory_reserved(device) / 1e6


def build_chunks(candidates, chunk_size):
    chunks = []
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        obs_batch = torch.cat([c[0] for c in chunk], dim=0)
        act_batch = torch.cat([c[1] for c in chunk], dim=0)
        y_batch = torch.cat([c[2] for c in chunk], dim=0)
        chunks.append((start, obs_batch, act_batch, y_batch, len(chunk)))
    return chunks


def score_vmap_probes(denoiser, theta_s_named, frozen_named, template, h_D, h_D_inv_sqrt, bank, candidates, chunk_size, probe_chunk):
    device = h_D.device
    num_entries = bank.num_strata
    num_candidates = len(candidates)
    total_scores = torch.zeros(num_candidates, device=device)
    chunks = build_chunks(candidates, chunk_size)
    num_jvp_calls = 0
    sigma_data = denoiser.cfg.sigma_data
    last_jvp_out = None

    def single_probe_jvp(sigma, eps, eta, obs_batch, act_batch, y_batch):
        cs = denoiser.compute_conditioners(sigma)
        rescaled_obs = obs_batch / sigma_data
        y_sigma_batch = (y_batch + sigma.view(-1, 1, 1, 1) * eps).detach()
        rescaled_noise = y_sigma_batch * cs.c_in
        c_noise = cs.c_noise
        z_flat = h_D_inv_sqrt * eta
        tangent_named = unflatten_to_dict(z_flat, template)

        def f(theta_s_dict):
            full_params = {**frozen_named, **theta_s_dict}
            return func.functional_call(denoiser.inner_model, full_params, (rescaled_noise, c_noise, rescaled_obs, act_batch))

        _, jvp_out = func.jvp(f, (theta_s_named,), (tangent_named,))
        return jvp_out

    batched_probe_jvp = func.vmap(single_probe_jvp, in_dims=(0, 0, 0, None, None, None))

    for pstart in range(0, num_entries, probe_chunk):
        pend = min(pstart + probe_chunk, num_entries)
        P = pend - pstart
        sigmas_P = torch.cat([bank.sigmas[m] for m in range(pstart, pend)], dim=0)
        epsilons_P = torch.stack([bank.epsilons[m].squeeze(0) for m in range(pstart, pend)], dim=0)
        etas_P = torch.stack([bank.etas[m] for m in range(pstart, pend)], dim=0)

        for start, obs_batch, act_batch, y_batch, B in chunks:
            with torch.no_grad():
                jvp_out_all = batched_probe_jvp(sigmas_P, epsilons_P, etas_P, obs_batch, act_batch, y_batch)
            last_jvp_out = jvp_out_all
            num_jvp_calls += 1
            contribution = 2.0 * jvp_out_all.reshape(P, B, -1).square().sum(dim=2)
            total_scores[start : start + B] += contribution.sum(dim=0) / num_entries

    return total_scores.detach().cpu(), num_jvp_calls, last_jvp_out


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

    bank = make_forward_jvp_simple_mc_bank(
        setup.SIGMA_CFG, torch.Size([1, 3, 64, 64]), h_D_full.numel(), device, num_samples=M, seed=BANK_SEED
    )
    print(f"fixed M={M} Full-CRN bank built (seed={BANK_SEED}), same as the candidate-chunk sweep.", flush=True)

    reset_peak(device)
    r_reference = None
    for rep in range(2):  # 1 warmup + 1 timed, just need a stable reference (already timed in the earlier run)
        r_reference = score_one_jvp_bank(
            denoiser, theta_s_named, frozen_named, theta_s_named, h_D_full, h_D_inv_sqrt, bank, candidates, REFERENCE_CHUNK_SIZE
        )
    print("reference (chunk_size=4, sequential probes) scores computed.", flush=True)

    print("\n" + "=" * 88, flush=True)
    print(f"PART 4 -- probe batching via torch.func.vmap(jvp), P in {PROBE_CHUNKS}, candidate chunk C={FASTEST_C}", flush=True)
    print("=" * 88, flush=True)
    probe_results = {}
    print(f"\n{'P':>3} {'n_vmap_calls':>12} {'mean_s':>8} {'std_s':>7} {'peak_alloc_MB':>13} {'peak_res_MB':>12} "
          f"{'grad_fn_None':>12} {'max_abs_err':>12} {'rel_err':>9}", flush=True)
    for P in PROBE_CHUNKS:
        reset_peak(device)
        try:
            times = []
            r_new = None
            grad_fn_ok = None
            for rep in range(NUM_TIMED_REPS + 1):
                gpu_sync()
                t0 = time.perf_counter()
                r_new, n_calls, last_jvp_out = score_vmap_probes(
                    denoiser, theta_s_named, frozen_named, theta_s_named, h_D_full, h_D_inv_sqrt, bank, candidates, FASTEST_C, P
                )
                gpu_sync()
                if rep > 0:
                    times.append(time.perf_counter() - t0)
                if grad_fn_ok is None and last_jvp_out is not None:
                    grad_fn_ok = last_jvp_out.grad_fn is None
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{P:>3}  OOM -- stopping probe-chunk sweep here.", flush=True)
                torch.cuda.empty_cache()
                gc.collect()
                break
            raise

        peak_alloc, peak_reserved = peak_stats(device)
        max_abs_err = (r_new - r_reference).abs().max().item()
        rel_err = max_abs_err / r_reference.abs().max().item()
        probe_results[P] = dict(mean_s=np.mean(times), std_s=np.std(times), n_calls=n_calls,
                                 peak_alloc=peak_alloc, peak_reserved=peak_reserved, grad_fn_ok=grad_fn_ok,
                                 max_abs_err=max_abs_err, rel_err=rel_err)
        print(f"{P:>3} {n_calls:>12} {np.mean(times):>8.3f} {np.std(times):>7.3f} "
              f"{peak_alloc:>13.1f} {peak_reserved:>12.1f} {str(grad_fn_ok):>12} "
              f"{max_abs_err:>12.3e} {rel_err:>9.3e}", flush=True)

    if probe_results:
        best_vmap_P = min(probe_results, key=lambda P: probe_results[P]["mean_s"])
        best_vmap_time = probe_results[best_vmap_P]["mean_s"]
        seq_time = probe_results[1]["mean_s"] if 1 in probe_results else None
        print(f"\n  sequential probes (P=1) at C={FASTEST_C}: {seq_time:.3f}s" if seq_time else "", flush=True)
        print(f"  best vmap probe-batching: P={best_vmap_P}, {best_vmap_time:.3f}s", flush=True)
        vmap_helps = seq_time is not None and best_vmap_time < seq_time * 0.95
        print(f"  vmap probe-batching helps (>5% faster than sequential, P=1): {vmap_helps}", flush=True)
    else:
        vmap_helps = False
        best_vmap_P = None
        print("\n  no probe-batching configuration completed -- keeping sequential probes.", flush=True)

    print("\n" + "=" * 88, flush=True)
    print("PART 5 -- 2D (candidate_chunk, probe_chunk) search", flush=True)
    print("=" * 88, flush=True)
    if vmap_helps:
        grid = [(FASTEST_C, best_vmap_P), (FASTEST_C // 2, best_vmap_P), (FASTEST_C * 2, best_vmap_P), (FASTEST_C, 1)]
        grid = sorted(set(c for c in grid if c[0] >= 1))
        print(f"  testing combinations: {grid}", flush=True)
        grid_results = {}
        for C, P in grid:
            reset_peak(device)
            try:
                times = []
                for rep in range(NUM_TIMED_REPS + 1):
                    gpu_sync()
                    t0 = time.perf_counter()
                    r_new, n_calls, _ = score_vmap_probes(denoiser, theta_s_named, frozen_named, theta_s_named, h_D_full, h_D_inv_sqrt, bank, candidates, C, P)
                    gpu_sync()
                    if rep > 0:
                        times.append(time.perf_counter() - t0)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  (C={C}, P={P})  OOM", flush=True)
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise
            peak_alloc, peak_reserved = peak_stats(device)
            grid_results[(C, P)] = (np.mean(times), peak_alloc, peak_reserved)
            print(f"  (C={C:>4}, P={P:>2}): {np.mean(times):.3f}s  peak_alloc={peak_alloc:.1f}MB  peak_reserved={peak_reserved:.1f}MB", flush=True)
        if grid_results:
            best_combo = min(grid_results, key=lambda k: grid_results[k][0])
            print(f"\n  fastest isolated (C,P) combination: {best_combo} -> {grid_results[best_combo][0]:.3f}s, "
                  f"peak_alloc={grid_results[best_combo][1]:.1f}MB", flush=True)
    else:
        print("  skipped: vmap probe-batching did not help, so C=16 sequential probes (already benchmarked) remains best.", flush=True)

    print("\nProbe-batching benchmark complete.", flush=True)


if __name__ == "__main__":
    main()
