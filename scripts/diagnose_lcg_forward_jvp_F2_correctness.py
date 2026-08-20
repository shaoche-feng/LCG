#! /usr/bin/env python
"""
LCG forward/JVP diagnostic -- Stage F2: numerical correctness validation, on a tiny toy
model where the Jacobian can be constructed explicitly. Must pass before Stage F3 (real
model) proceeds.

F2.1: torch.func.jvp(f, params, z) vs an explicit-Jacobian J @ z, on the SAME
      functional_call + torch.func.jvp machinery used by the real pipeline
      (diagnose_lcg_forward_jvp_common.jvp_through_F's general pattern, specialized to a
      tiny model here so the reference Jacobian is cheap to build).
F2.2: exact trace r_exact = 2*tr(H_D^-1 J^T J) vs Monte Carlo forward (eta probes, via the
      SAME production jvp machinery) and Monte Carlo backward (xi probes, via real
      torch.autograd.grad -- an independent AD mode, not matrix math on both sides).
F2.3: finite-difference directional derivative sanity check.

Usage:
    python scripts/diagnose_lcg_forward_jvp_F2_correctness.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.func as func
import torch.nn as nn

torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------------------
# Tiny toy model: entire model is "theta_S_tiny" (small enough to build J explicitly)
# --------------------------------------------------------------------------------------

D_IN, D_HIDDEN, D_OUT = 4, 8, 5


def build_tiny_model():
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(D_IN, D_HIDDEN), nn.SiLU(), nn.Linear(D_HIDDEN, D_OUT)).to(DEVICE)
    return model


def named_params_dict(model):
    return dict(model.named_parameters())


def flatten_dict(named):
    return torch.cat([t.reshape(-1) for t in named.values()])


def unflatten_to_dict(flat, template):
    result, offset = {}, 0
    for name, p in template.items():
        n = p.numel()
        result[name] = flat[offset : offset + n].reshape(p.shape)
        offset += n
    return result


def build_explicit_jacobian(model, x):
    """J[k, :] = d F(theta)[k] / d theta, via real torch.autograd.grad, one row per output
    unit -- an independent reference computation from the JVP path being tested."""
    params = list(model.parameters())
    F_out = model(x)  # (d_y,)
    d_y = F_out.numel()
    rows = []
    for k in range(d_y):
        grads = torch.autograd.grad(F_out[k], params, retain_graph=True, create_graph=False)
        rows.append(torch.cat([g.reshape(-1) for g in grads]))
    J = torch.stack(rows, dim=0)  # (d_y, d_S)
    return J, F_out.detach()


def jvp_via_functional_call(model, template, z_flat, x):
    """The SAME general mechanism as the real pipeline's jvp_through_F: functional_call +
    torch.func.jvp, genuine forward-mode AD."""
    tangent = unflatten_to_dict(z_flat, template)

    def f(params_dict):
        return func.functional_call(model, params_dict, (x,))

    primal, jvp_out = func.jvp(f, (template,), (tangent,))
    return primal, jvp_out


def vjp_via_autograd(model, xi, x):
    """Real backward-mode VJP: v = J^T xi, via torch.autograd.grad -- independent AD mode
    from the forward path, used as the toy system's backward-estimator analog."""
    params = list(model.parameters())
    F_out = model(x)
    scalar = (xi * F_out).sum()
    grads = torch.autograd.grad(scalar, params, retain_graph=False, create_graph=False)
    return torch.cat([g.reshape(-1) for g in grads])


def main():
    print(f"device={DEVICE}", flush=True)
    model = build_tiny_model()
    template = named_params_dict(model)
    d_S = sum(p.numel() for p in template.values())
    x = torch.randn(D_IN, device=DEVICE)

    print(f"tiny model: d_in={D_IN} d_hidden={D_HIDDEN} d_out={D_OUT} d_S={d_S}", flush=True)

    J, F_out = build_explicit_jacobian(model, x)
    print(f"explicit Jacobian J: shape={tuple(J.shape)}", flush=True)

    # ------------------------------------------------------------------------------
    # F2.1: JVP(F, z) vs explicit J @ z
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("F2.1 -- torch.func.jvp vs explicit J @ z")
    print("=" * 88)
    torch.manual_seed(42)
    max_abs_err, max_rel_err = 0.0, 0.0
    for trial in range(10):
        z = torch.randn(d_S, device=DEVICE)
        Jz_explicit = J @ z
        primal, jvp_out = jvp_via_functional_call(model, template, z, x)
        abs_err = (jvp_out - Jz_explicit).abs()
        rel_err = abs_err / (Jz_explicit.abs() + 1e-12)
        max_abs_err = max(max_abs_err, abs_err.max().item())
        max_rel_err = max(max_rel_err, rel_err.max().item())
        primal_diff = (primal - F_out).abs().max().item()
        print(f"  trial {trial}: max_abs_err={abs_err.max().item():.3e}  max_rel_err={rel_err.max().item():.3e}  "
              f"primal_vs_explicit_diff={primal_diff:.3e}")
    print(f"\n  OVERALL: max_abs_error={max_abs_err:.3e}  max_rel_error={max_rel_err:.3e}")
    f21_pass = max_abs_err < 1e-4 and max_rel_err < 1e-3
    print(f"  F2.1 PASS: {f21_pass} (target: near float32 tolerance)")

    # ------------------------------------------------------------------------------
    # F2.2: exact trace vs forward MC (via production jvp path) vs backward MC (via
    # real autograd VJP)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("F2.2 -- exact trace vs forward-MC (eta) and backward-MC (xi) convergence")
    print("=" * 88)
    torch.manual_seed(7)
    h_D = torch.rand(d_S, device=DEVICE) * 2.0 + 0.5  # arbitrary positive diagonal, fixed
    h_D_inv_sqrt = h_D.rsqrt()

    G_diag = (J * J).sum(dim=0)  # diag(J^T J), shape (d_S,)
    r_exact = 2.0 * (G_diag / h_D).sum().item()
    print(f"  r_exact = 2*tr(H_D^-1 J^T J) = {r_exact:.6f}", flush=True)

    M = 4000
    torch.manual_seed(100)
    fwd_samples, bwd_samples = [], []
    for m in range(M):
        eta = torch.randn(d_S, device=DEVICE)
        z = h_D_inv_sqrt * eta
        _, jvp_out = jvp_via_functional_call(model, template, z, x)
        fwd_samples.append(2.0 * jvp_out.square().sum().item())

        xi = torch.randn(D_OUT, device=DEVICE)
        v = vjp_via_autograd(model, xi, x)
        bwd_samples.append(2.0 * (h_D_inv_sqrt * v).square().sum().item())

    fwd_t = torch.tensor(fwd_samples)
    bwd_t = torch.tensor(bwd_samples)
    checkpoints = [10, 50, 200, 1000, M]
    print(f"\n  {'M':>6} {'forward_MC_mean':>16} {'fwd_rel_err':>12} {'backward_MC_mean':>17} {'bwd_rel_err':>12}")
    for cp in checkpoints:
        fwd_mean = fwd_t[:cp].mean().item()
        bwd_mean = bwd_t[:cp].mean().item()
        print(f"  {cp:>6} {fwd_mean:>16.4f} {abs(fwd_mean - r_exact) / r_exact:>12.4f} "
              f"{bwd_mean:>17.4f} {abs(bwd_mean - r_exact) / r_exact:>12.4f}")

    final_fwd_relerr = abs(fwd_t.mean().item() - r_exact) / r_exact
    final_bwd_relerr = abs(bwd_t.mean().item() - r_exact) / r_exact
    f22_pass = final_fwd_relerr < 0.05 and final_bwd_relerr < 0.05
    print(f"\n  final forward MC mean={fwd_t.mean().item():.4f} (rel_err={final_fwd_relerr:.4f})")
    print(f"  final backward MC mean={bwd_t.mean().item():.4f} (rel_err={final_bwd_relerr:.4f})")
    print(f"  F2.2 PASS: {f22_pass} (both converge to the same exact trace within 5% at M={M})")

    # ------------------------------------------------------------------------------
    # F2.3: finite-difference directional derivative
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("F2.3 -- finite-difference directional derivative check")
    print("=" * 88)
    torch.manual_seed(55)
    fd_errs = []
    for trial in range(5):
        z = torch.randn(d_S, device=DEVICE)
        z = z / z.norm()
        _, jvp_out = jvp_via_functional_call(model, template, z, x)

        for delta in [1e-3, 1e-4]:
            theta_plus = unflatten_to_dict(flatten_dict(template) + delta * z, template)
            theta_minus = unflatten_to_dict(flatten_dict(template) - delta * z, template)
            with torch.no_grad():
                F_plus = func.functional_call(model, theta_plus, (x,))
                F_minus = func.functional_call(model, theta_minus, (x,))
            fd = (F_plus - F_minus) / (2 * delta)
            err = (fd - jvp_out).abs().max().item()
            fd_errs.append(err)
            print(f"  trial {trial} delta={delta:.0e}: max_abs_diff(FD, JVP)={err:.3e}")

    f23_pass = min(fd_errs) < 1e-2  # the smaller-delta trials should show FD -> JVP
    print(f"\n  F2.3 PASS: {f23_pass} (finite-difference approaches the JVP value as delta shrinks)")

    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("F2 SUMMARY")
    print("=" * 88)
    print(f"  F2.1 (explicit Jacobian match): {'PASS' if f21_pass else 'FAIL'}")
    print(f"  F2.2 (exact trace convergence):  {'PASS' if f22_pass else 'FAIL'}")
    print(f"  F2.3 (finite-difference check):  {'PASS' if f23_pass else 'FAIL'}")
    all_pass = f21_pass and f22_pass and f23_pass
    print(f"\n  ALL F2 CHECKS PASS: {all_pass}")
    if not all_pass:
        print("\n  STOPPING per instructions: F2 correctness gate failed, not proceeding to F3.")
        sys.exit(1)

    print("\nF2 complete.")


if __name__ == "__main__":
    main()
