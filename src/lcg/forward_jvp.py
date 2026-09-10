from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.func as func
from torch import Tensor

from models.diffusion.denoiser import (
    Denoiser,
    SigmaDistributionConfig,
    apply_noise_from_samples,
    sample_sigma_training_distribution,
)

from .theta_s import selected_parameters

Candidate = Tuple[Tensor, Tensor, Tensor]  # (x_obs_flat, x_act, y_star), each batch size 1


# --------------------------------------------------------------------------------------
# theta_S <-> named-dict plumbing (functional_call needs dotted names; ordering must match
# selected_parameters()/h_D exactly). Moved verbatim from the validated forward-JVP
# diagnostic (scripts/forward_JVP/3-stratum_CRN/diagnose_lcg_forward_jvp_common.py, Stage
# F1-F6) -- no re-derivation of the JVP mathematics here.
# --------------------------------------------------------------------------------------


def selected_named_parameters(denoiser: Denoiser) -> Dict[str, torch.nn.Parameter]:
    inner = denoiser.inner_model
    last_idx = len(inner.unet.u_blocks) - 1
    prefixed_modules = [
        (f"unet.u_blocks.{last_idx}", inner.unet.u_blocks[-1]),
        ("norm_out", inner.norm_out),
        ("conv_out", inner.conv_out),
    ]
    named = {}
    for prefix, module in prefixed_modules:
        for name, p in module.named_parameters():
            named[f"{prefix}.{name}"] = p

    ref = selected_parameters(denoiser)
    assert len(named) == len(ref) and all(a is b for a, b in zip(named.values(), ref)), (
        "selected_named_parameters ordering/membership does not match lcg.theta_s.selected_parameters()"
    )
    return named


def frozen_named_parameters(denoiser: Denoiser, theta_s_named: Dict[str, torch.nn.Parameter]) -> Dict[str, Tensor]:
    inner = denoiser.inner_model
    selected_ids = {id(p) for p in theta_s_named.values()}
    return {name: p for name, p in inner.named_parameters() if id(p) not in selected_ids}


def flatten_dict(named: Dict[str, Tensor]) -> Tensor:
    return torch.cat([t.reshape(-1) for t in named.values()])


def unflatten_to_dict(flat: Tensor, template: Dict[str, Tensor]) -> Dict[str, Tensor]:
    result = {}
    offset = 0
    for name, p in template.items():
        n = p.numel()
        result[name] = flat[offset : offset + n].reshape(p.shape)
        offset += n
    assert offset == flat.numel(), f"flat tensor size {flat.numel()} does not match template total {offset}"
    return result


# --------------------------------------------------------------------------------------
# Genuine forward-mode JVP through F_theta (EDM-cancelled, no c_out/c_skip), validated in
# Stage F1-F6: w(sigma)*c_out(sigma)^2 = 1 identically (edm_weight is defined as
# c_out^-2, see lcg.gauss_newton), so 2*w*||J_D z||^2 = 2*||J_F z||^2 exactly, and the JVP
# can be taken through the raw (un-preconditioned) inner-model output directly.
# --------------------------------------------------------------------------------------


def jvp_through_F(
    denoiser: Denoiser,
    theta_s_named: Dict[str, Tensor],
    frozen_named: Dict[str, Tensor],
    tangent_named: Dict[str, Tensor],
    y_sigma_batch: Tensor,
    sigma: Tensor,
    obs_batch: Tensor,
    act_batch: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Returns (F_theta_primal, JVP_output), both shape matching y_sigma_batch. Uses
    torch.func.jvp + torch.func.functional_call exclusively -- true forward-mode AD, never
    autograd.grad/reverse-over-reverse/an explicit Jacobian.

    Wrapped in torch.no_grad(): frozen_named's tensors are real nn.Parameters with
    requires_grad=True, so without this, functional_call's forward pass also builds a
    reverse-mode autograd graph (visible as primal_out.grad_fn/jvp_out.grad_fn being
    non-None) IN ADDITION TO the forward-mode dual computation -- pure waste, since nothing
    here ever calls .backward(), and it would otherwise retain the entire inner_model
    activation graph per call, growing without bound across repeated scoring calls.
    torch.no_grad() only disables reverse-mode graph construction; forward-mode AD
    (torch.autograd.forward_ad, which torch.func.jvp is built on) is an orthogonal
    mechanism and is unaffected -- this was confirmed empirically in the forward-JVP
    diagnostic (Stage F6): a real CUDA OOM from unbounded reverse-graph accumulation,
    fixed by this torch.no_grad() wrap, reverified against tiny-model correctness checks
    (bit-identical primal/JVP values, grad_fn now None on both outputs)."""
    cs = denoiser.compute_conditioners(sigma)
    rescaled_obs = obs_batch / denoiser.cfg.sigma_data
    rescaled_noise = y_sigma_batch * cs.c_in
    c_noise = cs.c_noise

    def f(theta_s_dict):
        full_params = {**frozen_named, **theta_s_dict}
        return func.functional_call(denoiser.inner_model, full_params, (rescaled_noise, c_noise, rescaled_obs, act_batch))

    with torch.no_grad():
        primal_out, jvp_out = func.jvp(f, (theta_s_named,), (tangent_named,))
    return primal_out, jvp_out


@dataclass(frozen=True)
class JVPBank:
    """A set of shared forward-JVP probes: one (sigma, eps, eps_offset, eta) quadruple per
    entry, reused across every candidate (Full CRN). epsilons_offset: one (1, C, 1, 1)
    tensor per entry -- the DIAMOND offset-noise draw
    (models.diffusion.denoiser.apply_noise_from_samples), broadcast across the whole
    candidate batch (batch dim 1), NOT one independent draw per candidate. etas: one flat
    (d_S,) tensor per entry (parameter-space probe).

    Entries are ordinary IID Monte Carlo samples (sigma_m ~ p_train(sigma), see
    make_jvp_bank) -- score_one_jvp_bank treats the bank as a plain average over however
    many entries it holds."""

    sigmas: Tuple[Tensor, ...]
    epsilons: Tuple[Tensor, ...]
    epsilons_offset: Tuple[Tensor, ...]
    etas: Tuple[Tensor, ...]

    def __post_init__(self) -> None:
        assert len(self.sigmas) == len(self.epsilons) == len(self.epsilons_offset) == len(self.etas)

    @property
    def num_samples(self) -> int:
        return len(self.sigmas)


def make_jvp_bank(
    sigma_cfg: SigmaDistributionConfig,
    y_shape: torch.Size,
    d_S: int,
    device: torch.device,
    num_samples: int = 24,
    seed: Optional[int] = None,
) -> JVPBank:
    """Production Full-CRN probe bank: num_samples IID (sigma, eps, eps_offset, eta)
    quadruples, each sigma drawn from the complete training distribution p_train(sigma)
    (sample_sigma_training_distribution, the same authoritative distribution DIAMOND
    training itself samples from). eps_offset has shape (1, C, 1, 1) -- one draw per MC
    sample, shared across every candidate via broadcasting (Full CRN), NOT independent
    per candidate. All candidates scored against one bank instance share the identical
    num_samples quadruples, including across computational chunks -- the bank is built
    once and passed unchanged into every chunk's score_one_jvp_bank call.

    A single generator seeded once up front (not one reseed per sample) is sufficient for
    determinism: the whole num_samples-long draw sequence is then a deterministic
    function of seed. RNG isolation: uses a local, device-matched torch.Generator rather
    than global torch.manual_seed, so building a bank never mutates the global torch RNG
    stream DIAMOND's own code observes. Same seed -> same bank; different seeds ->
    different banks (unchanged)."""
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    c = y_shape[1]
    sigmas, epsilons, epsilons_offset, etas = [], [], [], []
    for _ in range(num_samples):
        sigmas.append(sample_sigma_training_distribution(sigma_cfg, 1, device, generator=gen).detach())
        epsilons.append(torch.randn(y_shape, device=device, generator=gen).detach())
        epsilons_offset.append(torch.randn(1, c, 1, 1, device=device, generator=gen).detach())
        etas.append(torch.randn(d_S, device=device, generator=gen).detach())
    return JVPBank(tuple(sigmas), tuple(epsilons), tuple(epsilons_offset), tuple(etas))


def score_one_jvp_bank(
    denoiser: Denoiser,
    theta_s_named: Dict[str, Tensor],
    frozen_named: Dict[str, Tensor],
    template: Dict[str, Tensor],
    h_D: Tensor,
    h_D_inv_sqrt: Tensor,
    bank: JVPBank,
    candidates: List[Candidate],
    chunk_size: int,
) -> Tensor:
    """r_hat(x_j) = (1/len(bank)) * sum_s 2*||JVP_F(x_j; y_sigma_s; z_s)||^2, for every
    candidate, where y_sigma_s = apply_noise_from_samples(x_j, sigma_s, eps_s,
    eps_offset_s, denoiser.cfg.sigma_offset_noise) is the exact DIAMOND training
    corruption law (fixed: previously used y + sigma*eps only, omitting the offset-noise
    term DIAMOND training actually applies). z_s = h_D_inv_sqrt * eta_s (elementwise,
    H_D diagonal, no_grad). Processes candidates in chunks of chunk_size; one probe (bank
    entry) at a time across all chunks, so at most one (chunk_size, d_S)-shaped JVP
    output is alive per step regardless of bank size.

    Loop order (probe outer, candidate-chunk inner) and candidate-chunk pre-batching
    (obs/act/y concatenated once per chunk and reused across every probe) were chosen
    per the Stage 7 batching optimization: the probe-dependent tangent
    (z_m = h_D_inv_sqrt*eta_m, then unflattened into theta_S's parameter-dict shape) is
    now computed once per probe instead of once per (chunk, probe) pair -- a pure
    reordering/hoisting of the exact same terms, mathematically identical to summing in
    any other order (verified to float32 precision against the original chunk-outer/
    probe-inner implementation; see scripts/forward_JVP/integration/
    optimize_scorer_M12.py Parts 1-3). Candidate-chunk size is the primary throughput
    lever (larger chunks amortize per-call kernel-launch/dispatch overhead across more
    candidates); probe batching via torch.func.vmap was benchmarked separately and found
    NOT to help (vmap's own dispatch overhead outweighs the reduced call count at this
    problem size, and it uses substantially more memory for no net speedup) -- kept
    sequential, per Stage 7's explicit "measure, don't assume" guidance."""
    device = h_D.device
    num_entries = bank.num_samples
    num_candidates = len(candidates)
    total_scores = torch.zeros(num_candidates, device=device)

    chunks = []
    for start in range(0, num_candidates, chunk_size):
        chunk = candidates[start : start + chunk_size]
        obs_batch = torch.cat([c[0] for c in chunk], dim=0)
        act_batch = torch.cat([c[1] for c in chunk], dim=0)
        y_batch = torch.cat([c[2] for c in chunk], dim=0)
        chunks.append((start, obs_batch, act_batch, y_batch, len(chunk)))

    for m in range(num_entries):
        sigma, eps, eps_offset, eta = bank.sigmas[m], bank.epsilons[m], bank.epsilons_offset[m], bank.etas[m]
        with torch.no_grad():
            z_flat = h_D_inv_sqrt * eta
        tangent_named = unflatten_to_dict(z_flat, template)  # hoisted: once per probe, not once per (chunk, probe)

        for start, obs_batch, act_batch, y_batch, B in chunks:
            y_sigma_batch = apply_noise_from_samples(y_batch, sigma, eps, eps_offset, denoiser.cfg.sigma_offset_noise).detach()
            _, jvp_out = jvp_through_F(
                denoiser, theta_s_named, frozen_named, tangent_named, y_sigma_batch, sigma, obs_batch, act_batch
            )
            contribution = 2.0 * jvp_out.reshape(B, -1).square().sum(dim=1)
            total_scores[start : start + B] += contribution / num_entries

    return total_scores.detach().cpu()


def assert_setup_valid(theta_s_named: Dict[str, Tensor], h_D: Tensor, d_S: int) -> None:
    flat = flatten_dict(theta_s_named)
    assert flat.numel() == d_S, f"theta_S flat dim {flat.numel()} != d_S {d_S}"
    assert h_D.numel() == d_S, f"h_D dim {h_D.numel()} != d_S {d_S}"
    assert torch.isfinite(h_D).all(), "h_D contains non-finite values"
    assert (h_D > 0).all(), "h_D is not strictly positive"
    assert not h_D.requires_grad, "h_D must not require grad"
    print(f"assert_setup_valid: dim(theta_S)=dim(h_D)={d_S}, h_D finite and >0, h_D.requires_grad=False -- OK")
