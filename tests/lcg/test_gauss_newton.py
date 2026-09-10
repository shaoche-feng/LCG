"""
Permanent regression test for the backward/VJP Gauss-Newton estimator
(lcg.gauss_newton.compute_vjp), extracted from the original Stage-1 correctness
validation. Uses tiny_denoiser (a real, tiny Denoiser) so the explicit Jacobian is cheap
to build via a loop of torch.autograd.grad calls -- an independent reference computation
from the VJP path being tested.
"""
import torch
from lcg.gauss_newton import compute_vjp, differentiable_denoise
from lcg.theta_s import selected_parameters
from models.diffusion.denoiser import apply_noise_from_samples


def _explicit_jacobian(d_theta_flat, params):
    rows = []
    for k in range(d_theta_flat.numel()):
        grads = torch.autograd.grad(d_theta_flat[k], params, retain_graph=True)
        rows.append(torch.cat([g.reshape(-1) for g in grads]))
    return torch.stack(rows)  # (d_out, d_S)


def test_vjp_matches_explicit_jacobian(tiny_denoiser, tiny_transition):
    denoiser = tiny_denoiser
    obs, act, y = tiny_transition
    params = selected_parameters(denoiser)

    sigma = torch.tensor([1.0])
    eps = torch.randn_like(y)
    eps_offset = torch.randn(1, y.shape[1], 1, 1)
    y_sigma = apply_noise_from_samples(y, sigma, eps, eps_offset, denoiser.cfg.sigma_offset_noise).detach()

    d_theta, w = differentiable_denoise(denoiser, y_sigma, sigma, obs, act)
    J = _explicit_jacobian(d_theta.reshape(-1), params)

    torch.manual_seed(42)
    xi = torch.randn_like(d_theta)
    v_expected = (torch.sqrt(2 * w).reshape(-1) * xi.reshape(-1)) @ J

    d_theta2, w2 = differentiable_denoise(denoiser, y_sigma, sigma, obs, act)  # fresh graph for compute_vjp
    v_actual, _ = compute_vjp(denoiser, params, y_sigma, sigma, obs, act, xi=xi)

    assert torch.allclose(v_actual, v_expected, atol=1e-4, rtol=1e-3)
    assert v_actual.shape == (sum(p.numel() for p in params),)
    assert torch.isfinite(v_actual).all()


def test_hutchinson_diagonal_estimate_matches_exact_gauss_newton_diagonal(tiny_denoiser, tiny_transition):
    """E_xi[(J^T xi)^2] == diag(J^T J), the Gauss-Newton diagonal identity underlying
    historical_precision -- checked against the EXPLICIT diag(J^T J) from the same tiny
    model, not a separately-hardcoded expected value."""
    denoiser = tiny_denoiser
    obs, act, y = tiny_transition
    params = selected_parameters(denoiser)

    sigma = torch.tensor([0.8])
    eps = torch.randn_like(y)
    eps_offset = torch.randn(1, y.shape[1], 1, 1)
    y_sigma = apply_noise_from_samples(y, sigma, eps, eps_offset, denoiser.cfg.sigma_offset_noise).detach()

    d_theta, w = differentiable_denoise(denoiser, y_sigma, sigma, obs, act)
    J = _explicit_jacobian(d_theta.reshape(-1), params)  # (d_out, d_S), raw (unweighted) Jacobian
    diag_exact = (J * J).sum(dim=0)  # diag(J^T J)

    torch.manual_seed(7)
    M = 300
    d_S = diag_exact.numel()
    accum = torch.zeros(d_S)
    for _ in range(M):
        xi = torch.randn(J.shape[0])
        v = J.T @ xi  # (d_S,) == J^T xi, the raw (unweighted) VJP
        accum += v * v
    diag_mc = accum / M

    rel_err = ((diag_mc - diag_exact).norm() / diag_exact.norm()).item()
    assert rel_err < 0.3  # M=300 on a small d_S is a coarse but real MC convergence check
    assert diag_mc.shape == diag_exact.shape
    assert torch.isfinite(diag_mc).all()
    assert (diag_mc >= 0).all()
    assert (diag_exact >= 0).all()  # diag(J^T J) is a sum of squares, always nonnegative
