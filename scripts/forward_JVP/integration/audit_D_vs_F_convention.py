#! /usr/bin/env python
"""
Stage 6 Phase 0: numerical audit of the D_theta vs F_theta convention.

The backward/VJP path (lcg.gauss_newton.compute_vjp, used by both
lcg.precision.historical_precision and the legacy lcg.batched_vjp/candidate_score
scorers) differentiates D_theta = c_skip*y_sigma + c_out*F_theta directly, with an
explicit sqrt(2*w(sigma)) prefactor -- i.e. it computes v = sqrt(2w) J_D^T xi, so
v^2 = 2w (J_D^T xi)^2 (convention "A").

The forward-JVP path (lcg.forward_jvp.jvp_through_F, used by the new production
candidate scorer) instead takes the JVP through the *raw* inner-model output F_theta,
never multiplying by w explicitly, relying on the EDM-cancellation identity
w(sigma)*c_out(sigma)^2 = 1 (lcg.gauss_newton.edm_weight is literally defined as
c_out^-2, so this holds by construction, not merely approximately) to make
2*w*||J_D z||^2 == 2*||J_F z||^2 (convention "B").

This script checks that identity directly and numerically, on the real converged
diagnostic model, for the same candidate/sigma/eps/z: it builds a second, audit-only JVP
function `jvp_through_D` (JVP through the full preconditioned D_theta, exists ONLY in
this script -- not added to src/lcg/, since production code only ever needs one
convention) and compares 2*w*||JVP_D||^2 against 2*||JVP_F||^2 for several
candidates/sigma/eta draws. Read-only / diagnostic: does not modify any estimator.

Usage:
    python scripts/forward_JVP/integration/audit_D_vs_F_convention.py
"""
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
    selected_named_parameters,
    unflatten_to_dict,
)
from lcg.gauss_newton import edm_weight
from lcg.sigma_strata import sample_sigma_stratum


def jvp_through_D(denoiser, theta_s_named, frozen_named, tangent_named, y_sigma_batch, sigma, obs_batch, act_batch):
    """Audit-only: JVP through the FULL preconditioned D_theta = c_skip*y_sigma +
    c_out*F_theta (convention A), for direct comparison against jvp_through_F
    (convention B). Not part of production code -- exists only to verify the
    cancellation identity numerically."""
    cs = denoiser.compute_conditioners(sigma)
    rescaled_obs = obs_batch / denoiser.cfg.sigma_data
    rescaled_noise = y_sigma_batch * cs.c_in
    c_noise = cs.c_noise

    def f(theta_s_dict):
        full_params = {**frozen_named, **theta_s_dict}
        F_out = func.functional_call(denoiser.inner_model, full_params, (rescaled_noise, c_noise, rescaled_obs, act_batch))
        return cs.c_skip * y_sigma_batch + cs.c_out * F_out

    with torch.no_grad():
        primal_out, jvp_out = func.jvp(f, (theta_s_named,), (tangent_named,))
    return primal_out, jvp_out, cs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    denoiser, action_dim = setup.load_converged_denoiser(device)
    h_D_full = torch.load(setup.H_D_FULL_PATH, map_location=device, weights_only=True)
    candidates = setup.load_frozen_candidates(device)
    setup.verify_candidates(candidates)

    theta_s_named = selected_named_parameters(denoiser)
    frozen_named = frozen_named_parameters(denoiser, theta_s_named)
    d_S = h_D_full.numel()

    print("\n" + "=" * 88)
    print("PHASE 0 AUDIT -- 2*w*||J_D z||^2  vs  2*||J_F z||^2, same candidate/sigma/eps/z")
    print("=" * 88)

    torch.manual_seed(12345)
    test_candidate_idxs = [0, 17, 123, 240, 479]
    rel_diffs = []
    d_recon_diffs = []

    for i, cand_idx in enumerate(test_candidate_idxs):
        obs, act, y = candidates[cand_idx]
        sigma = sample_sigma_stratum(setup.SIGMA_CFG, 0, 1, 1, device)
        eps = torch.randn_like(y)
        y_sigma = (y + sigma.view(-1, 1, 1, 1) * eps).detach()
        eta = torch.randn(d_S, device=device)
        z_flat = h_D_full.rsqrt() * eta
        tangent_named = unflatten_to_dict(z_flat, theta_s_named)

        primal_F, jvp_F = jvp_through_F(denoiser, theta_s_named, frozen_named, tangent_named, y_sigma, sigma, obs, act)
        primal_D, jvp_D, cs = jvp_through_D(denoiser, theta_s_named, frozen_named, tangent_named, y_sigma, sigma, obs, act)

        w = edm_weight(cs.c_out)
        score_F = 2.0 * jvp_F.square().sum().item()
        score_D = 2.0 * (w * jvp_D.square()).sum().item()
        rel_diff = abs(score_F - score_D) / max(abs(score_F), 1e-12)
        rel_diffs.append(rel_diff)

        # sanity: does the D-primal we reconstructed here actually match c_skip*y_sigma + c_out*F_primal?
        d_recon = cs.c_skip * y_sigma + cs.c_out * primal_F
        d_recon_diff = (d_recon - primal_D).abs().max().item()
        d_recon_diffs.append(d_recon_diff)

        print(f"  candidate={cand_idx:>4}  sigma={sigma.item():.5f}  w={w.item():.6e}  "
              f"score_F(2||J_F z||^2)={score_F:.6f}  score_D(2w||J_D z||^2)={score_D:.6f}  "
              f"rel_diff={rel_diff:.3e}  primal_D_reconstruction_maxdiff={d_recon_diff:.3e}", flush=True)

    max_rel_diff = max(rel_diffs)
    max_d_recon_diff = max(d_recon_diffs)
    print(f"\nmax rel_diff over {len(test_candidate_idxs)} probes: {max_rel_diff:.3e}")
    print(f"max primal_D reconstruction diff: {max_d_recon_diff:.3e}")
    print(f"PASS (near float32 precision, <1e-3 relative): {max_rel_diff < 1e-3}")
    print("\nAudit complete.", flush=True)


if __name__ == "__main__":
    main()
