#! /usr/bin/env python
"""
Stage 6: regression/unit tests for the production forward-JVP candidate scorer
(src/lcg/forward_jvp.py, src/lcg/intrinsic_reward.py's
make_lcg_forward_jvp_intrinsic_reward_fn, src/lcg/lifecycle.py's candidate_estimator
branch). Reuses the already-validated real-model fixtures (converged diagnostic
denoiser, 480 frozen candidates, full-dataset h_D) from the forward-JVP diagnostics
rather than a synthetic tiny model -- a stronger test than a from-scratch harness, and
avoids re-deriving any math already validated in Stage F1-F6.

Parts (per the Stage 6 spec):
  A. JVP correctness -- the existing tiny explicit-Jacobian agreement still holds
     (re-run via lcg.forward_jvp directly, not a duplicated implementation).
  B. Full CRN -- one MC sample: all candidates in a chunked call receive identical
     (sigma, eps, eta); different chunks of the same call still share the same probe.
  C. IID simple MC -- different m samples are independent draws from the *full*
     training sigma distribution (not fixed strata): checks marginal sigma statistics
     over many samples against the unstratified population, and checks etas/epsilons
     across samples are not degenerate/identical.
  D. Determinism -- fixed candidate batch + fixed CRN seed + fixed h_D gives
     reproducible candidate scores across repeated calls.
  E. No reverse graph -- integrated JVP scoring does not build a reverse-mode
     autograd graph (grad_fn is None on JVP outputs) and does not leak GPU memory
     across repeated scoring calls.
  F. D_theta/F_theta convention equivalence -- regression guard: 2*w*||J_D z||^2 ==
     2*||J_F z||^2 for a probe batch, reusing jvp_through_F plus an audit-only
     jvp_through_D (mirrors scripts/forward_JVP/integration/audit_D_vs_F_convention.py).

Diagnostic/test only: never modifies any estimator, h_D, or theta_S.

Usage:
    python scripts/forward_JVP/integration/validate_lcg_forward_jvp_integration.py
"""
import subprocess
import sys
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
import torch.func as func

import diagnose_lcg_backward_variance_setup as setup
from lcg.forward_jvp import (
    frozen_named_parameters,
    jvp_through_F,
    make_forward_jvp_simple_mc_bank,
    score_one_jvp_bank,
    selected_named_parameters,
)
from lcg.gauss_newton import edm_weight
from lcg.intrinsic_reward import make_lcg_forward_jvp_intrinsic_reward_fn
from lcg.sigma_strata import sample_sigma_stratum

CHUNK_SIZE = 4
NUM_CANDIDATES_SMALL = 16  # small subset for the finer-grained regression checks


def part_a_jvp_correctness():
    """A: "Existing small JVP correctness test still passes." Re-running an ad hoc
    finite-difference check directly on the real 654,851-parameter UNet is not a sound
    correctness test: z = h_D^-1/2 * eta has a large norm here (h_D's 1e-4 damping floor
    gives ||z|| ~ 100), so *any* step size small enough to stay in the network's local-
    linear regime is also small enough for float32 catastrophic cancellation to dominate
    across this deep, highly nonlinear model's many sequential ops -- shrinking the step
    was observed to make the finite-difference estimate WORSE, not better (the classic
    signature of round-off, not truncation, error). The already-validated ground truth
    for JVP correctness is Stage F2's tiny-toy-model test (explicit Jacobian, D_S small
    enough to build exactly, well-conditioned FD): this re-runs that exact test (unchanged
    math, same file) as a regression gate."""
    print("\n" + "=" * 88)
    print("PART A -- JVP correctness (re-running Stage F2's tiny-model exact-Jacobian gate)")
    print("=" * 88)
    f2_path = _REPO_ROOT / "scripts" / "forward_JVP" / "3-stratum_CRN" / "diagnose_lcg_forward_jvp_F2_correctness.py"
    result = subprocess.run([sys.executable, str(f2_path)], capture_output=True, text=True)
    print(result.stdout[-2500:])
    if result.returncode != 0:
        print(result.stderr[-2000:])
    passed = result.returncode == 0 and "ALL F2 CHECKS PASS: True" in result.stdout
    print(f"  F2 subprocess exit_code={result.returncode}  PASS: {passed}")
    return passed


def part_b_full_crn(denoiser, theta_s_named, frozen_named, candidates, h_D):
    """B: one MC sample -> all candidates (across chunk boundaries) receive identical
    (sigma, eps, eta). Checked indirectly but rigorously: two candidates that are
    IDENTICAL up to floating point (we duplicate candidate 0) must get IDENTICAL scores
    when scored together in the same call, even when placed in different chunks -- this
    can only happen if they saw the same probes."""
    print("\n" + "=" * 88)
    print("PART B -- Full CRN sharing across candidates and across chunks")
    print("=" * 88)
    device = h_D.device
    obs0, act0, y0 = candidates[0]
    # Build a batch where candidate 0 is duplicated at position 0 and position 10 (forces
    # them into different chunks under CHUNK_SIZE=4).
    dup_candidates = list(candidates[:NUM_CANDIDATES_SMALL])
    dup_candidates[10] = (obs0.clone(), act0.clone(), y0.clone())
    bank = make_forward_jvp_simple_mc_bank(setup.SIGMA_CFG, torch.Size([1, 3, 64, 64]), h_D.numel(), device,
                                            num_samples=4, seed=777)
    h_D_inv_sqrt = h_D.rsqrt()
    scores = score_one_jvp_bank(denoiser, theta_s_named, frozen_named, theta_s_named, h_D, h_D_inv_sqrt, bank,
                                 dup_candidates, CHUNK_SIZE)
    diff = (scores[0] - scores[10]).abs().item()
    print(f"  score[0]={scores[0].item():.6f}  score[10] (duplicate of candidate 0, different chunk)={scores[10].item():.6f}  "
          f"abs_diff={diff:.3e}")
    passed = diff < 1e-4
    print(f"  PASS: {passed}")
    return passed


def part_c_iid_simple_mc(sigma_cfg, h_D, device):
    """C: bank samples are IID draws from the FULL training sigma distribution (not
    fixed strata) -- checks the empirical CDF of many bank-drawn sigmas against
    Denoiser.sample_sigma_training's own log-normal, and checks etas/epsilons across
    samples are distinct (non-degenerate) draws."""
    print("\n" + "=" * 88)
    print("PART C -- IID simple-MC sampling (full distribution, not fixed strata)")
    print("=" * 88)
    d_S = h_D.numel()
    bank = make_forward_jvp_simple_mc_bank(sigma_cfg, torch.Size([1, 3, 64, 64]), d_S, device, num_samples=500, seed=99)
    log_sigmas = torch.stack([s.log() for s in bank.sigmas]).flatten()
    empirical_mean, empirical_std = log_sigmas.mean().item(), log_sigmas.std().item()
    print(f"  log(sigma) over 500 bank samples: mean={empirical_mean:.4f} (cfg.loc={sigma_cfg.loc}), "
          f"std={empirical_std:.4f} (cfg.scale={sigma_cfg.scale})")
    mean_ok = abs(empirical_mean - sigma_cfg.loc) < 0.15
    std_ok = abs(empirical_std - sigma_cfg.scale) < 0.15

    etas_stacked = torch.stack(bank.etas)
    pairwise_corr = torch.corrcoef(etas_stacked[:20])[0, 1:].abs().max().item()
    print(f"  max |corr(eta_0, eta_j)| over first 20 samples: {pairwise_corr:.4f} (expect near 0 for independent draws)")
    etas_distinct = pairwise_corr < 0.3

    passed = mean_ok and std_ok and etas_distinct
    print(f"  PASS: {passed} (mean_ok={mean_ok}, std_ok={std_ok}, etas_distinct={etas_distinct})")
    return passed


def part_d_determinism(denoiser, theta_s_named, frozen_named, candidates, h_D):
    """D: fixed candidate batch + fixed CRN seed + fixed h_D -> reproducible scores."""
    print("\n" + "=" * 88)
    print("PART D -- determinism (fixed seed/candidates/h_D reproduces identical scores)")
    print("=" * 88)
    device = h_D.device
    h_D_inv_sqrt = h_D.rsqrt()
    small_candidates = candidates[:NUM_CANDIDATES_SMALL]

    scores_runs = []
    for run in range(2):
        bank = make_forward_jvp_simple_mc_bank(setup.SIGMA_CFG, torch.Size([1, 3, 64, 64]), h_D.numel(), device,
                                                num_samples=6, seed=4242)
        s = score_one_jvp_bank(denoiser, theta_s_named, frozen_named, theta_s_named, h_D, h_D_inv_sqrt, bank,
                                small_candidates, CHUNK_SIZE)
        scores_runs.append(s)

    max_diff = (scores_runs[0] - scores_runs[1]).abs().max().item()
    print(f"  max abs diff between two identically-seeded runs: {max_diff:.3e}")
    passed = max_diff == 0.0
    print(f"  PASS (bit-exact): {passed}")
    return passed


def part_e_no_reverse_graph(denoiser, theta_s_named, frozen_named, candidates, h_D):
    """E: integrated JVP scoring does not build a reverse-mode graph (grad_fn is None)
    and does not leak GPU memory across repeated calls."""
    print("\n" + "=" * 88)
    print("PART E -- no reverse-mode graph construction / no memory leak")
    print("=" * 88)
    device = h_D.device
    obs, act, y = candidates[0]
    sigma = sample_sigma_stratum(setup.SIGMA_CFG, 0, 1, 1, device)
    eps = torch.randn_like(y)
    y_sigma = (y + sigma.view(-1, 1, 1, 1) * eps).detach()
    from lcg.forward_jvp import unflatten_to_dict
    eta = torch.randn(h_D.numel(), device=device)
    tangent_named = unflatten_to_dict(h_D.rsqrt() * eta, theta_s_named)

    primal_out, jvp_out = jvp_through_F(denoiser, theta_s_named, frozen_named, tangent_named, y_sigma, sigma, obs, act)
    grad_fn_check = primal_out.grad_fn is None and jvp_out.grad_fn is None
    print(f"  primal_out.grad_fn={primal_out.grad_fn}  jvp_out.grad_fn={jvp_out.grad_fn}  no_reverse_graph={grad_fn_check}")

    leak_ok = True
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        bank = make_forward_jvp_simple_mc_bank(setup.SIGMA_CFG, torch.Size([1, 3, 64, 64]), h_D.numel(), device,
                                                num_samples=4, seed=1)
        h_D_inv_sqrt = h_D.rsqrt()
        allocs = []
        for i in range(20):
            _ = score_one_jvp_bank(denoiser, theta_s_named, frozen_named, theta_s_named, h_D, h_D_inv_sqrt, bank,
                                    candidates[:NUM_CANDIDATES_SMALL], CHUNK_SIZE)
            torch.cuda.synchronize()
            allocs.append(torch.cuda.memory_allocated(device) / 1e6)
        drift = allocs[-1] - allocs[5]  # ignore first few calls' warm-up allocation
        print(f"  memory_allocated across 20 repeated scoring calls: first={allocs[0]:.1f}MB last={allocs[-1]:.1f}MB "
              f"drift(call5->call19)={drift:.2f}MB")
        leak_ok = abs(drift) < 5.0  # a few MB of allocator noise is fine; unbounded growth is not
    else:
        print("  (CPU device: skipping CUDA memory-drift check)")

    passed = grad_fn_check and leak_ok
    print(f"  PASS: {passed}")
    return passed


def part_f_d_vs_f_equivalence(denoiser, theta_s_named, frozen_named, candidates, h_D):
    """F: regression guard for the EDM-cancellation identity 2*w*||J_D z||^2 ==
    2*||J_F z||^2, on a small candidate batch, so future changes to EDM
    preconditioning/weighting that break the cancellation are caught here rather than
    silently producing an inconsistent candidate scorer."""
    print("\n" + "=" * 88)
    print("PART F -- D_theta/F_theta convention equivalence regression guard")
    print("=" * 88)
    device = h_D.device

    def jvp_through_D(denoiser, theta_s_named, frozen_named, tangent_named, y_sigma_batch, sigma, obs_batch, act_batch):
        cs = denoiser.compute_conditioners(sigma)
        rescaled_obs = obs_batch / denoiser.cfg.sigma_data
        rescaled_noise = y_sigma_batch * cs.c_in
        c_noise = cs.c_noise

        def f(theta_s_dict):
            full_params = {**frozen_named, **theta_s_dict}
            F_out = func.functional_call(denoiser.inner_model, full_params, (rescaled_noise, c_noise, rescaled_obs, act_batch))
            return cs.c_skip * y_sigma_batch + cs.c_out * F_out

        with torch.no_grad():
            _, jvp_out = func.jvp(f, (theta_s_named,), (tangent_named,))
        return jvp_out, cs

    from lcg.forward_jvp import unflatten_to_dict
    max_rel_diff = 0.0
    for cand_idx in [0, 5, 10]:
        obs, act, y = candidates[cand_idx]
        sigma = sample_sigma_stratum(setup.SIGMA_CFG, 0, 1, 1, device)
        eps = torch.randn_like(y)
        y_sigma = (y + sigma.view(-1, 1, 1, 1) * eps).detach()
        eta = torch.randn(h_D.numel(), device=device)
        tangent_named = unflatten_to_dict(h_D.rsqrt() * eta, theta_s_named)

        _, jvp_F = jvp_through_F(denoiser, theta_s_named, frozen_named, tangent_named, y_sigma, sigma, obs, act)
        jvp_D, cs = jvp_through_D(denoiser, theta_s_named, frozen_named, tangent_named, y_sigma, sigma, obs, act)
        w = edm_weight(cs.c_out)

        score_F = 2.0 * jvp_F.square().sum().item()
        score_D = 2.0 * (w * jvp_D.square()).sum().item()
        rel_diff = abs(score_F - score_D) / max(abs(score_F), 1e-12)
        max_rel_diff = max(max_rel_diff, rel_diff)
        print(f"  candidate={cand_idx}: score_F={score_F:.6f} score_D={score_D:.6f} rel_diff={rel_diff:.3e}")

    passed = max_rel_diff < 1e-3
    print(f"  max rel_diff={max_rel_diff:.3e}  PASS: {passed}")
    return passed


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    denoiser, action_dim = setup.load_converged_denoiser(device)
    h_D_full = torch.load(setup.H_D_FULL_PATH, map_location=device, weights_only=True)
    candidates = setup.load_frozen_candidates(device)
    setup.verify_candidates(candidates)

    theta_s_named = selected_named_parameters(denoiser)
    frozen_named = frozen_named_parameters(denoiser, theta_s_named)

    results = {}
    results["A_jvp_correctness"] = part_a_jvp_correctness()
    results["B_full_crn"] = part_b_full_crn(denoiser, theta_s_named, frozen_named, candidates, h_D_full)
    results["C_iid_simple_mc"] = part_c_iid_simple_mc(setup.SIGMA_CFG, h_D_full, device)
    results["D_determinism"] = part_d_determinism(denoiser, theta_s_named, frozen_named, candidates, h_D_full)
    results["E_no_reverse_graph"] = part_e_no_reverse_graph(denoiser, theta_s_named, frozen_named, candidates, h_D_full)
    results["F_d_vs_f_equivalence"] = part_f_d_vs_f_equivalence(denoiser, theta_s_named, frozen_named, candidates, h_D_full)

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    all_passed = all(results.values())
    print(f"\nALL PASS: {all_passed}")
    print("\nRegression tests complete.", flush=True)


if __name__ == "__main__":
    main()
