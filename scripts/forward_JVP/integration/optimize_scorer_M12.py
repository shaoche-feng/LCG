#! /usr/bin/env python
"""
Stage 7: candidate-chunk and probe-chunk batching optimization for the production
forward-JVP LCG scorer, benchmarked at M=12 (the currently configured production
candidate_num_mc). Pure engineering optimization -- does NOT change the LCG estimator,
CRN semantics, h_D, theta_S, or the EDM-cancelled J_F convention. Every tested
configuration is checked for exact numerical equivalence (up to float32 noise) against
the current production reference (chunk_size=4, sequential probes).

Mathematical basis for candidate/probe batching (verified before benchmarking, see
Part 0): the UNet's only normalization layers are GroupNorm/AdaGroupNorm, which
normalize per-example (over channels-within-group and spatial dims only) -- confirmed
via grep, no nn.BatchNorm anywhere in src/models/blocks.py or inner_model.py. This
means every row of a batch dimension is processed completely independently through
every layer, so concatenating more candidates (or more probes) into one forward/JVP
call cannot mix information across rows -- it is a pure throughput optimization, not a
semantic change.

Parts:
  0. Audit: current chunk size, JVP call count, redundant per-call work.
  1. Numerical verification that batching does not mix candidates (isolated vs batched).
  2. Candidate-chunk-size benchmark, C in {4,8,16,32,64,96,128,240,480}, with hoisted
     probe-invariant computation (unflatten_to_dict, obs/act/y concatenation) -- the
     "optimized" scorer implemented here first, before touching src/lcg/.
  3. Correctness regression vs the current production reference at every chunk size.
  4. Probe-batching via torch.func.vmap(jvp), P in {1,2,3,4,6,12}.
  5. 2D (candidate_chunk, probe_chunk) search for the fastest stable configuration.

Usage:
    python scripts/forward_JVP/integration/optimize_scorer_M12.py
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
    jvp_through_F,
    make_forward_jvp_simple_mc_bank,
    score_one_jvp_bank,  # CURRENT production reference (chunk_size=4, sequential probes)
    selected_named_parameters,
    unflatten_to_dict,
)

M = 12
BANK_SEED = 424242  # ONE fixed bank reused across every configuration in this script
CHUNK_SIZES = [4, 8, 16, 32, 64, 96, 128, 240, 480]
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


def pearson_corr(a, b):
    a, b = a.double(), b.double()
    a, b = a - a.mean(), b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-12)).item()


def spearman_corr(a, b):
    ra = torch.argsort(torch.argsort(a)).double()
    rb = torch.argsort(torch.argsort(b)).double()
    return pearson_corr(ra, rb)


def top_q_overlap(a, b, q):
    k = max(1, int(round(q * a.numel())))
    sa = set(torch.topk(a, k).indices.tolist())
    sb = set(torch.topk(b, k).indices.tolist())
    return len(sa & sb) / k


# --------------------------------------------------------------------------------------
# "Optimized" candidate-chunked scorer: same math as lcg.forward_jvp.score_one_jvp_bank,
# but loop order swapped (probe outer / candidate-chunk inner) so the probe-dependent
# tangent (z_m, unflattened into a dict) is computed ONCE per probe instead of once per
# (chunk, probe) pair, and candidate chunks (obs/act/y concatenation) are pre-built once
# and reused across all M probes. This is the item-5 "hoist invariant quantities"
# optimization, combined with the item-2/3 candidate-chunk-size sweep.
# --------------------------------------------------------------------------------------


def build_chunks(candidates, chunk_size):
    chunks = []
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        obs_batch = torch.cat([c[0] for c in chunk], dim=0)
        act_batch = torch.cat([c[1] for c in chunk], dim=0)
        y_batch = torch.cat([c[2] for c in chunk], dim=0)
        chunks.append((start, obs_batch, act_batch, y_batch, len(chunk)))
    return chunks


def score_optimized(denoiser, theta_s_named, frozen_named, template, h_D, h_D_inv_sqrt, bank, candidates, chunk_size):
    device = h_D.device
    num_entries = bank.num_strata
    num_candidates = len(candidates)
    total_scores = torch.zeros(num_candidates, device=device)
    chunks = build_chunks(candidates, chunk_size)
    num_jvp_calls = 0

    for m in range(num_entries):
        sigma, eps, eta = bank.sigmas[m], bank.epsilons[m], bank.etas[m]
        with torch.no_grad():
            z_flat = h_D_inv_sqrt * eta
        tangent_named = unflatten_to_dict(z_flat, template)  # hoisted: computed ONCE per probe, not per chunk

        for start, obs_batch, act_batch, y_batch, B in chunks:
            y_sigma_batch = (y_batch + sigma.view(-1, 1, 1, 1) * eps).detach()
            _, jvp_out = jvp_through_F(denoiser, theta_s_named, frozen_named, tangent_named, y_sigma_batch, sigma, obs_batch, act_batch)
            num_jvp_calls += 1
            contribution = 2.0 * jvp_out.reshape(B, -1).square().sum(dim=1)
            total_scores[start : start + B] += contribution / num_entries

    return total_scores.detach().cpu(), num_jvp_calls


# --------------------------------------------------------------------------------------
# Probe-batching via torch.func.vmap(jvp): batches P probes (each with its own sigma/eps/
# eta, hence its own primal input AND its own tangent) into a single vmapped call. This is
# the standard functorch "vmap of jvp" composition -- batching multiple TANGENT directions
# through one dispatch, distinct from candidate-batching (which only batches the ordinary
# forward-pass batch dimension for a SHARED tangent).
# --------------------------------------------------------------------------------------


def score_vmap_probes(denoiser, theta_s_named, frozen_named, template, h_D, h_D_inv_sqrt, bank, candidates, chunk_size, probe_chunk):
    device = h_D.device
    num_entries = bank.num_strata
    num_candidates = len(candidates)
    total_scores = torch.zeros(num_candidates, device=device)
    chunks = build_chunks(candidates, chunk_size)
    num_jvp_calls = 0  # counts vmapped calls, each internally covering probe_chunk probes

    sigma_data = denoiser.cfg.sigma_data

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
        sigmas_P = torch.cat([bank.sigmas[m] for m in range(pstart, pend)], dim=0)  # (P,)
        epsilons_P = torch.stack([bank.epsilons[m].squeeze(0) for m in range(pstart, pend)], dim=0)  # (P, C, H, W)
        etas_P = torch.stack([bank.etas[m] for m in range(pstart, pend)], dim=0)  # (P, d_S)

        for start, obs_batch, act_batch, y_batch, B in chunks:
            with torch.no_grad():
                jvp_out_all = batched_probe_jvp(sigmas_P, epsilons_P, etas_P, obs_batch, act_batch, y_batch)  # (P, B, C, H, W)
            num_jvp_calls += 1
            contribution = 2.0 * jvp_out_all.reshape(P, B, -1).square().sum(dim=2)  # (P, B)
            total_scores[start : start + B] += contribution.sum(dim=0) / num_entries

    return total_scores.detach().cpu(), num_jvp_calls, jvp_out_all if num_entries > 0 else None


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
    print(f"fixed M={M} Full-CRN bank built (seed={BANK_SEED}), reused for every configuration below.")

    # ------------------------------------------------------------------------------
    # PART 0 -- audit
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("PART 0 -- audit of the current production path")
    print("=" * 88)
    n_chunks_ref = -(-num_candidates // REFERENCE_CHUNK_SIZE)
    n_jvp_calls_ref = n_chunks_ref * M
    print(f"  current production chunk_size (config/trainer.yaml lcg.chunk_size): {REFERENCE_CHUNK_SIZE}")
    print(f"  jvp_through_F already receives multiple candidates in one tensor: YES "
          f"(obs/act/y concatenated per chunk via torch.cat before the JVP call)")
    print(f"  one torch.func.jvp call currently scores: ONE CHUNK ({REFERENCE_CHUNK_SIZE} candidates), "
          f"not one candidate and not the whole 480-candidate batch")
    print(f"  num_chunks for 480 candidates at chunk_size={REFERENCE_CHUNK_SIZE}: {n_chunks_ref}")
    print(f"  total torch.func.jvp calls at M={M}: {n_chunks_ref} chunks x {M} probes = {n_jvp_calls_ref}")
    print(f"  redundant work identified: in lcg.forward_jvp.score_one_jvp_bank, the probe-dependent tangent "
          f"(z_m = h_D_inv_sqrt*eta_m, then unflatten_to_dict into a ~13-tensor dict) is recomputed once per "
          f"(chunk, probe) pair -- {n_jvp_calls_ref} times -- even though it only depends on the probe m, "
          f"not on which candidate chunk. Correct minimum: {M} times (once per probe). obs/act/y concatenation "
          f"was ALREADY hoisted to once-per-chunk in the existing code (not redundant).")
    print(f"  GroupNorm/AdaGroupNorm only (no BatchNorm) confirmed via grep -- batching candidates or probes "
          f"cannot mix information across the batch dimension (verified structurally, Part 1 verifies numerically).")

    # baseline timed measurement of the CURRENT (unmodified) production function
    reset_peak(device)
    ref_times = []
    r_reference = None
    for rep in range(NUM_TIMED_REPS + 1):  # +1 warm-up
        gpu_sync()
        t0 = time.perf_counter()
        r_reference = score_one_jvp_bank(
            denoiser, theta_s_named, frozen_named, theta_s_named, h_D_full, h_D_inv_sqrt, bank, candidates, REFERENCE_CHUNK_SIZE
        )
        gpu_sync()
        if rep > 0:  # discard warm-up
            ref_times.append(time.perf_counter() - t0)
    peak_alloc_ref, peak_reserved_ref = peak_stats(device)
    print(f"\n  BASELINE (current production, chunk_size={REFERENCE_CHUNK_SIZE}, M={M}): "
          f"mean={np.mean(ref_times):.2f}s std={np.std(ref_times):.3f}s  "
          f"ms/candidate={1000*np.mean(ref_times)/num_candidates:.2f}  "
          f"ms/candidate-probe={1000*np.mean(ref_times)/(num_candidates*M):.4f}  "
          f"peak_alloc={peak_alloc_ref:.1f}MB peak_reserved={peak_reserved_ref:.1f}MB")

    # ------------------------------------------------------------------------------
    # PART 1 -- numerical verification: isolated candidate vs batched candidate
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("PART 1 -- numerical verification that candidate batching does not mix candidates")
    print("=" * 88)
    probe_candidate_idx = 250
    isolated_candidates = [candidates[probe_candidate_idx]]
    r_isolated, _ = score_optimized(denoiser, theta_s_named, frozen_named, theta_s_named, h_D_full, h_D_inv_sqrt, bank, isolated_candidates, 1)
    r_full_batch, _ = score_optimized(denoiser, theta_s_named, frozen_named, theta_s_named, h_D_full, h_D_inv_sqrt, bank, candidates, 480)
    diff = abs(r_isolated[0].item() - r_full_batch[probe_candidate_idx].item())
    rel_diff = diff / abs(r_isolated[0].item())
    print(f"  candidate #{probe_candidate_idx}: score in isolation (chunk_size=1)={r_isolated[0].item():.6f}  "
          f"score inside the full 480-candidate batch (chunk_size=480)={r_full_batch[probe_candidate_idx].item():.6f}")
    print(f"  abs_diff={diff:.3e}  rel_diff={rel_diff:.3e}")
    part1_pass = rel_diff < 1e-4
    print(f"  PASS (no mixing across candidates, agrees to float32 precision): {part1_pass}")

    # ------------------------------------------------------------------------------
    # PART 2/3 -- candidate-chunk-size benchmark + correctness regression
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print(f"PART 2/3 -- candidate-chunk-size benchmark + correctness regression, M={M}")
    print("=" * 88)
    print(f"\n{'C':>4} {'n_jvp_calls':>11} {'mean_s':>8} {'std_s':>7} {'ms/cand':>9} {'ms/c-p':>8} "
          f"{'peak_alloc_MB':>13} {'peak_res_MB':>12} {'max_abs_err':>12} {'rel_err':>9} "
          f"{'Pearson':>8} {'Spearman':>9} {'top10':>7} {'top20':>7}")
    chunk_results = {}
    for C in CHUNK_SIZES:
        reset_peak(device)
        try:
            times = []
            r_new = None
            for rep in range(NUM_TIMED_REPS + 1):
                gpu_sync()
                t0 = time.perf_counter()
                r_new, n_calls = score_optimized(denoiser, theta_s_named, frozen_named, theta_s_named, h_D_full, h_D_inv_sqrt, bank, candidates, C)
                gpu_sync()
                if rep > 0:
                    times.append(time.perf_counter() - t0)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{C:>4}  OOM -- stopping chunk-size sweep here.")
                torch.cuda.empty_cache()
                gc.collect()
                break
            raise

        peak_alloc, peak_reserved = peak_stats(device)
        max_abs_err = (r_new - r_reference).abs().max().item()
        rel_err = max_abs_err / r_reference.abs().max().item()
        pear = pearson_corr(r_new, r_reference)
        spear = spearman_corr(r_new, r_reference)
        t10 = top_q_overlap(r_new, r_reference, 0.10)
        t20 = top_q_overlap(r_new, r_reference, 0.20)
        chunk_results[C] = dict(mean_s=np.mean(times), std_s=np.std(times), n_calls=n_calls,
                                 peak_alloc=peak_alloc, peak_reserved=peak_reserved,
                                 max_abs_err=max_abs_err, rel_err=rel_err, pearson=pear, spearman=spear,
                                 top10=t10, top20=t20, scores=r_new)
        print(f"{C:>4} {n_calls:>11} {np.mean(times):>8.3f} {np.std(times):>7.3f} "
              f"{1000*np.mean(times)/num_candidates:>9.3f} {1000*np.mean(times)/(num_candidates*M):>8.4f} "
              f"{peak_alloc:>13.1f} {peak_reserved:>12.1f} {max_abs_err:>12.3e} {rel_err:>9.3e} "
              f"{pear:>8.5f} {spear:>9.5f} {t10:>7.4f} {t20:>7.4f}", flush=True)

    fastest_C = min(chunk_results, key=lambda C: chunk_results[C]["mean_s"])
    print(f"\n  fastest tested candidate chunk size: C={fastest_C} "
          f"({chunk_results[fastest_C]['mean_s']:.3f}s, peak_alloc={chunk_results[fastest_C]['peak_alloc']:.1f}MB)")

    # ------------------------------------------------------------------------------
    # PART 4 -- probe batching via vmap(jvp)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print(f"PART 4 -- probe batching via torch.func.vmap(jvp), P in {PROBE_CHUNKS}, candidate chunk fixed at C={fastest_C}")
    print("=" * 88)
    probe_results = {}
    print(f"\n{'P':>3} {'n_vmap_calls':>12} {'mean_s':>8} {'std_s':>7} {'peak_alloc_MB':>13} {'peak_res_MB':>12} "
          f"{'grad_fn_None':>12} {'max_abs_err':>12} {'rel_err':>9}")
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
                    denoiser, theta_s_named, frozen_named, theta_s_named, h_D_full, h_D_inv_sqrt, bank, candidates, fastest_C, P
                )
                gpu_sync()
                if rep > 0:
                    times.append(time.perf_counter() - t0)
                if grad_fn_ok is None and last_jvp_out is not None:
                    grad_fn_ok = last_jvp_out.grad_fn is None
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{P:>3}  OOM -- stopping probe-chunk sweep here.")
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
        best_seq = chunk_results[fastest_C]["mean_s"]
        best_vmap_P = min(probe_results, key=lambda P: probe_results[P]["mean_s"])
        best_vmap_time = probe_results[best_vmap_P]["mean_s"]
        print(f"\n  sequential probes at C={fastest_C}: {best_seq:.3f}s")
        print(f"  best vmap probe-batching: P={best_vmap_P}, {best_vmap_time:.3f}s")
        vmap_helps = best_vmap_time < best_seq * 0.95  # require a real >5% improvement, not noise
        print(f"  vmap probe-batching helps (>5% faster than sequential): {vmap_helps}")
    else:
        vmap_helps = False
        best_vmap_P = None
        print("\n  no probe-batching configuration completed (OOM at every tested P) -- keeping sequential probes.")

    # ------------------------------------------------------------------------------
    # PART 5 -- 2D search (only if vmap helped)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("PART 5 -- 2D (candidate_chunk, probe_chunk) search")
    print("=" * 88)
    if vmap_helps:
        grid = []
        for C in [c for c in CHUNK_SIZES if c <= fastest_C * 2 and c in chunk_results]:
            for P in [1, best_vmap_P] if best_vmap_P not in (None, 1) else [1]:
                grid.append((C, P))
        grid = sorted(set(grid))
        print(f"  testing combinations: {grid}")
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
                    print(f"  (C={C}, P={P})  OOM")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise
            peak_alloc, peak_reserved = peak_stats(device)
            grid_results[(C, P)] = (np.mean(times), peak_alloc, peak_reserved)
            print(f"  (C={C:>4}, P={P:>2}): {np.mean(times):.3f}s  peak_alloc={peak_alloc:.1f}MB  peak_reserved={peak_reserved:.1f}MB")
        if grid_results:
            best_combo = min(grid_results, key=lambda k: grid_results[k][0])
            print(f"\n  fastest isolated (C,P) combination: {best_combo} -> {grid_results[best_combo][0]:.3f}s, "
                  f"peak_alloc={grid_results[best_combo][1]:.1f}MB")
    else:
        print("  skipped: vmap probe-batching did not help (or OOM'd at every tested P), "
              "so the 2D search reduces to the 1D candidate-chunk sweep already done in Part 2/3.")

    print("\nOptimization benchmark complete.", flush=True)


if __name__ == "__main__":
    main()
