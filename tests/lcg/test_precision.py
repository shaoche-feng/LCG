"""
Permanent regression tests for the historical backward-VJP probe
(lcg.precision._backward_vjp_probe). Uses tiny_denoiser (a real, tiny Denoiser) so an
explicit Jacobian is cheap to build via a loop of torch.autograd.grad calls -- an
independent reference computation from the VJP path being tested.

Production computes v = sqrt(2) * J_F^T xi directly (differentiating the raw inner-model
output F_theta). This was simplified from an earlier D_theta-based form,
v = sqrt(2*w(sigma)) * J_D^T xi with D_theta = c_skip*y_sigma + c_out*F_theta -- an EXACT
algebraic simplification (see lcg.precision._backward_vjp_probe's docstring for the
derivation), not a different estimator. test_backward_vjp_D_form_equals_F_form below is
the permanent numerical proof of that equivalence: it explicitly reconstructs D_theta
locally, purely to document/verify the identity -- production itself never does this.
"""
import torch
from lcg.precision import _backward_vjp_probe
from lcg.theta_s import selected_parameters
from models.diffusion.denoiser import apply_noise_from_samples


def _explicit_jacobian(out_flat, params):
    rows = []
    for k in range(out_flat.numel()):
        grads = torch.autograd.grad(out_flat[k], params, retain_graph=True)
        rows.append(torch.cat([g.reshape(-1) for g in grads]))
    return torch.stack(rows)  # (d_out, d_S)


def _explicit_jacobian_of_F(denoiser, params, y_sigma, sigma, obs, act):
    cs = denoiser.compute_conditioners(sigma)
    F_out = denoiser.compute_model_output(y_sigma, obs, act, cs)
    J = _explicit_jacobian(F_out.reshape(-1), params)
    return J, F_out


def test_backward_vjp_matches_explicit_jacobian_of_F(tiny_denoiser, tiny_transition):
    """Production's actual current formula, v = sqrt(2) * J_F^T xi, checked against an
    independently-built explicit Jacobian of the raw inner-model output F_theta."""
    denoiser = tiny_denoiser
    obs, act, y = tiny_transition
    params = selected_parameters(denoiser)

    sigma = torch.tensor([1.0])
    eps = torch.randn_like(y)
    eps_offset = torch.randn(1, y.shape[1], 1, 1)
    y_sigma = apply_noise_from_samples(y, sigma, eps, eps_offset, denoiser.cfg.sigma_offset_noise).detach()

    J, F_out = _explicit_jacobian_of_F(denoiser, params, y_sigma, sigma, obs, act)

    torch.manual_seed(42)
    xi = torch.randn_like(F_out)
    v_expected = (2.0**0.5) * xi.reshape(-1) @ J

    v_actual = _backward_vjp_probe(denoiser, params, y_sigma, sigma, obs, act, xi=xi)

    assert torch.allclose(v_actual, v_expected, atol=1e-4, rtol=1e-3)
    assert v_actual.shape == (sum(p.numel() for p in params),)
    assert torch.isfinite(v_actual).all()


def test_backward_vjp_D_form_equals_F_form(tiny_denoiser, tiny_transition):
    """Permanent numerical proof of the D-vs-F equivalence that justifies production's
    simplified formula: for the SAME (y_sigma, sigma, xi, theta_S), the original
    D_theta-based probe v_D = sqrt(2*w)*J_D^T xi (reconstructed locally here, off the
    production path, purely to document the identity) must equal production's
    v_F = sqrt(2)*J_F^T xi to numerical floating-point tolerance. Reports max abs diff,
    max/mean relative diff, cosine similarity, and relative L2 (asserted, not just printed)."""
    denoiser = tiny_denoiser
    obs, act, y = tiny_transition
    params = selected_parameters(denoiser)

    sigma = torch.tensor([0.9])
    eps = torch.randn_like(y)
    eps_offset = torch.randn(1, y.shape[1], 1, 1)
    y_sigma = apply_noise_from_samples(y, sigma, eps, eps_offset, denoiser.cfg.sigma_offset_noise).detach()

    cs = denoiser.compute_conditioners(sigma)
    model_output = denoiser.compute_model_output(y_sigma, obs, act, cs)
    torch.manual_seed(5)
    xi = torch.randn_like(model_output)

    # -- old formulation, reconstructed locally for validation only --
    d_theta = cs.c_skip * y_sigma + cs.c_out * model_output
    w = cs.c_out.pow(-2)  # edm_weight(c_out), inlined -- production no longer needs this
    scalar_D = (torch.sqrt(2 * w) * xi * d_theta).sum()
    grads_D = torch.autograd.grad(scalar_D, params, retain_graph=False, create_graph=False)
    v_D = torch.cat([g.reshape(-1) for g in grads_D])

    # -- new (production) formulation --
    v_F = _backward_vjp_probe(denoiser, params, y_sigma, sigma, obs, act, xi=xi)

    diff = (v_D - v_F).abs()
    max_abs_diff = diff.max().item()
    rel_diff = diff / (v_D.abs() + 1e-12)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(v_D.unsqueeze(0), v_F.unsqueeze(0)).item()
    rel_l2 = ((v_D - v_F).norm() / v_D.norm()).item()

    assert max_abs_diff < 1e-4
    assert max_rel_diff < 1e-2
    assert mean_rel_diff < 1e-4
    assert cos_sim > 1 - 1e-5
    assert rel_l2 < 1e-4
    assert torch.allclose(v_D, v_F, atol=1e-4, rtol=1e-3)


def test_hutchinson_diagonal_estimate_matches_exact_gauss_newton_diagonal(tiny_denoiser, tiny_transition):
    """E_xi[(J_F^T xi)^2] == diag(J_F^T J_F), the Gauss-Newton diagonal identity underlying
    historical_precision -- checked against the EXPLICIT diag(J_F^T J_F) from the same
    tiny model, not a separately-hardcoded expected value. (G(x) = 2*J_F^T J_F is the
    quantity historical_precision estimates the diagonal of; the factor of 2 is constant
    across draws and cancels out of this ratio check, so it's omitted here.)"""
    denoiser = tiny_denoiser
    obs, act, y = tiny_transition
    params = selected_parameters(denoiser)

    sigma = torch.tensor([0.8])
    eps = torch.randn_like(y)
    eps_offset = torch.randn(1, y.shape[1], 1, 1)
    y_sigma = apply_noise_from_samples(y, sigma, eps, eps_offset, denoiser.cfg.sigma_offset_noise).detach()

    J, F_out = _explicit_jacobian_of_F(denoiser, params, y_sigma, sigma, obs, act)  # (d_out, d_S)
    diag_exact = (J * J).sum(dim=0)  # diag(J_F^T J_F)

    torch.manual_seed(7)
    M = 300
    d_S = diag_exact.numel()
    accum = torch.zeros(d_S)
    for _ in range(M):
        xi = torch.randn(J.shape[0])
        v = J.T @ xi  # (d_S,) == J_F^T xi
        accum += v * v
    diag_mc = accum / M

    rel_err = ((diag_mc - diag_exact).norm() / diag_exact.norm()).item()
    assert rel_err < 0.3  # M=300 on a small d_S is a coarse but real MC convergence check
    assert diag_mc.shape == diag_exact.shape
    assert torch.isfinite(diag_mc).all()
    assert (diag_mc >= 0).all()
    assert (diag_exact >= 0).all()  # diag(J_F^T J_F) is a sum of squares, always nonnegative
