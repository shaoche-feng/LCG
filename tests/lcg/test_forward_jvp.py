"""
Permanent regression tests for the production forward-JVP scorer (src/lcg/forward_jvp.py),
extracted from scripts/forward_JVP/integration/{audit_D_vs_F_convention,
validate_lcg_forward_jvp_integration}.py. Tiny real Denoiser (see conftest.py) so explicit
Jacobians are cheap to build on CPU.
"""
import torch
import torch.func as func
from lcg.forward_jvp import (
    JVPBank,
    frozen_named_parameters,
    jvp_through_F,
    make_forward_jvp_simple_mc_bank,
    score_one_jvp_bank,
    selected_named_parameters,
    unflatten_to_dict,
)
from lcg.gauss_newton import edm_weight
from lcg.theta_s import selected_parameters
from models.diffusion import SigmaDistributionConfig
from models.diffusion.denoiser import apply_noise_from_samples

# Matches conftest.py's tiny_denoiser fixture dimensions -- duplicated here (not imported
# via a relative import) to avoid depending on pytest's test-file import mode.
TINY_IMG_CHANNELS = 3
TINY_IMG_SIZE = 8
TINY_SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)


def _explicit_jacobian_of_F(denoiser, params, y_sigma, sigma, obs, act):
    cs = denoiser.compute_conditioners(sigma)
    F_out = denoiser.compute_model_output(y_sigma, obs, act, cs)
    F_flat = F_out.reshape(-1)
    rows = []
    for k in range(F_flat.numel()):
        grads = torch.autograd.grad(F_flat[k], params, retain_graph=True)
        rows.append(torch.cat([g.reshape(-1) for g in grads]))
    return torch.stack(rows), F_out  # (d_out, d_S), F_out


def _tiny_setup(tiny_denoiser, tiny_transition):
    denoiser = tiny_denoiser
    obs, act, y = tiny_transition
    theta_s_named = selected_named_parameters(denoiser)
    frozen_named = frozen_named_parameters(denoiser, theta_s_named)
    params = list(theta_s_named.values())
    d_S = sum(p.numel() for p in params)

    sigma = torch.tensor([1.3])
    eps = torch.randn_like(y)
    eps_offset = torch.randn(1, y.shape[1], 1, 1)
    y_sigma = apply_noise_from_samples(y, sigma, eps, eps_offset, denoiser.cfg.sigma_offset_noise).detach()
    return denoiser, obs, act, y, theta_s_named, frozen_named, params, d_S, sigma, eps, eps_offset, y_sigma


def test_jvp_matches_explicit_jacobian(tiny_denoiser, tiny_transition):
    denoiser, obs, act, y, theta_s_named, frozen_named, params, d_S, sigma, eps, eps_offset, y_sigma = _tiny_setup(
        tiny_denoiser, tiny_transition
    )
    J, F_out = _explicit_jacobian_of_F(denoiser, params, y_sigma, sigma, obs, act)

    torch.manual_seed(1)
    z_flat = torch.randn(d_S)
    tangent_named = unflatten_to_dict(z_flat, theta_s_named)
    primal_out, jvp_out = jvp_through_F(denoiser, theta_s_named, frozen_named, tangent_named, y_sigma, sigma, obs, act)

    jvp_expected = (J @ z_flat).reshape(F_out.shape)
    assert torch.allclose(jvp_out, jvp_expected, atol=1e-4, rtol=1e-3)
    assert torch.allclose(primal_out, F_out.detach(), atol=1e-5)


def test_parameter_ordering_matches_theta_s(tiny_denoiser):
    theta_s_named = selected_named_parameters(tiny_denoiser)
    ref = selected_parameters(tiny_denoiser)
    assert len(theta_s_named) == len(ref)
    assert all(a is b for a, b in zip(theta_s_named.values(), ref))


def test_score_formula_matches_explicit_jacobian(tiny_denoiser, tiny_transition):
    denoiser, obs, act, y, theta_s_named, frozen_named, params, d_S, sigma, eps, eps_offset, y_sigma = _tiny_setup(
        tiny_denoiser, tiny_transition
    )
    J, F_out = _explicit_jacobian_of_F(denoiser, params, y_sigma, sigma, obs, act)

    torch.manual_seed(2)
    h_D = torch.rand(d_S) + 0.5
    h_D_inv_sqrt = h_D.rsqrt()
    eta = torch.randn(d_S)
    z = h_D_inv_sqrt * eta

    jvp_expected = (J @ z).reshape(F_out.shape)
    r_expected = 2.0 * jvp_expected.square().sum().item()

    bank = JVPBank((sigma,), (eps,), (eps_offset,), (eta,))
    r_actual = score_one_jvp_bank(
        denoiser, theta_s_named, frozen_named, theta_s_named, h_D, h_D_inv_sqrt, bank, [(obs, act, y)], chunk_size=1
    ).item()
    assert abs(r_actual - r_expected) < 1e-3 * max(abs(r_expected), 1.0)


def test_no_reverse_graph(tiny_denoiser, tiny_transition):
    denoiser, obs, act, y, theta_s_named, frozen_named, params, d_S, sigma, eps, eps_offset, y_sigma = _tiny_setup(
        tiny_denoiser, tiny_transition
    )
    torch.manual_seed(3)
    tangent_named = unflatten_to_dict(torch.randn(d_S), theta_s_named)
    primal_out, jvp_out = jvp_through_F(denoiser, theta_s_named, frozen_named, tangent_named, y_sigma, sigma, obs, act)
    assert primal_out.grad_fn is None
    assert jvp_out.grad_fn is None


def test_score_finite_nonneg_correct_shape(tiny_denoiser, tiny_transition):
    denoiser, obs, act, y, theta_s_named, frozen_named, params, d_S, *_ = _tiny_setup(tiny_denoiser, tiny_transition)
    h_D = torch.rand(d_S) + 0.5
    h_D_inv_sqrt = h_D.rsqrt()
    bank = make_forward_jvp_simple_mc_bank(
        TINY_SIGMA_CFG, torch.Size([1, TINY_IMG_CHANNELS, TINY_IMG_SIZE, TINY_IMG_SIZE]), d_S,
        device=torch.device("cpu"), num_samples=3, seed=0,
    )
    candidates = [(obs, act, y), (obs, act, y)]
    scores = score_one_jvp_bank(denoiser, theta_s_named, frozen_named, theta_s_named, h_D, h_D_inv_sqrt, bank, candidates, chunk_size=2)
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
    assert (scores >= 0).all()


def test_d_vs_f_equivalence_at_corrected_corruption(tiny_denoiser, tiny_transition):
    """2*w*||J_D z||^2 == 2*||J_F z||^2 must still hold exactly at the CORRECTED
    (offset-noise-included) y_sigma -- the EDM cancellation w=c_out^-2 is an algebraic
    identity independent of how y_sigma was constructed, but this is the permanent
    regression guard against a future change to EDM preconditioning/weighting breaking it."""
    denoiser, obs, act, y, theta_s_named, frozen_named, params, d_S, sigma, eps, eps_offset, y_sigma = _tiny_setup(
        tiny_denoiser, tiny_transition
    )

    def jvp_through_D(tangent_named):
        cs = denoiser.compute_conditioners(sigma)
        rescaled_obs = obs / denoiser.cfg.sigma_data
        rescaled_noise = y_sigma * cs.c_in

        def f(theta_s_dict):
            full_params = {**frozen_named, **theta_s_dict}
            F_out = func.functional_call(denoiser.inner_model, full_params, (rescaled_noise, cs.c_noise, rescaled_obs, act))
            return cs.c_skip * y_sigma + cs.c_out * F_out

        with torch.no_grad():
            _, jvp_out = func.jvp(f, (theta_s_named,), (tangent_named,))
        return jvp_out, cs

    torch.manual_seed(4)
    tangent_named = unflatten_to_dict(torch.randn(d_S), theta_s_named)
    _, jvp_F = jvp_through_F(denoiser, theta_s_named, frozen_named, tangent_named, y_sigma, sigma, obs, act)
    jvp_D, cs = jvp_through_D(tangent_named)
    w = edm_weight(cs.c_out)

    score_F = 2.0 * jvp_F.square().sum().item()
    score_D = 2.0 * (w * jvp_D.square()).sum().item()
    rel_diff = abs(score_F - score_D) / max(abs(score_F), 1e-12)
    assert rel_diff < 1e-3
