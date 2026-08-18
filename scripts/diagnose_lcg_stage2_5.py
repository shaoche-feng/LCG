#! /usr/bin/env python
"""
LCG Stage 2.5 diagnostic: source of the heavy-tailed historical precision found in Stage 2.

Does NOT modify src/lcg/*.py. Reuses lcg.gauss_newton.compute_vjp, lcg.precision.*,
lcg.sigma_strata.sample_sigma_stratum, lcg.theta_s.* exactly as validated in Stages 1-2, and
reuses the exact real-checkpoint / real-(supplemented)-dataset setup from
scripts/validate_lcg_stage2.py so results are directly comparable.

Verifies, then diagnoses:
  1. The claimed analytic cancellation sqrt(2 w(sigma)) * J_D = sqrt(2) * J_F (since
     D_theta = c_skip*y_sigma + c_out*F_theta, c_skip/c_out independent of theta, and
     w(sigma) = 1/c_out(sigma)^2) -- checked directly against the current implementation,
     at low/mid/high sigma.
  2. Curvature by sigma stratum: is low-sigma actually bigger after the cancellation?
  3. Module-wise concentration of accumulated curvature across theta_S.
  4. Damping-floor domination: exactly-zero vs tiny-but-nonzero vs sparse-across-samples.
  5. Checkpoint-maturity context (how far zero-initialized layers have moved from init).

Usage:
    python scripts/diagnose_lcg_stage2_5.py
"""
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from data import Dataset, Episode
from envs.dm_control_env import make_dm_control_env
from lcg.gauss_newton import compute_vjp
from lcg.precision import load_transition, sample_valid_transitions
from lcg.sigma_strata import sample_sigma_stratum
from lcg.theta_s import selected_parameters
from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.diffusion.inner_model import InnerModelConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = REPO_ROOT / "outputs" / "2026-08-17" / "15-35-46"
SCRATCH_DIR = Path(
    r"C:\Users\jerry\AppData\Local\Temp\claude\c--Users-jerry-Project-LCG"
    r"\974bd621-91fa-4766-822e-267b547a92c5\scratchpad\lcg_stage2_5_dataset"
)

DOMAIN_NAME, TASK_NAME = "cheetah", "run"
ENV_KWARGS = dict(size=64, camera_id=0, action_repeat=2, time_limit=1.0)
SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)

B = 40
NUM_STRATA = 3
DAMPING = 1e-4
BETA = 1.0
TARGET_N = 400
SEED = 0


# --------------------------------------------------------------------------------------
# Setup: identical recipe to scripts/validate_lcg_stage2.py
# --------------------------------------------------------------------------------------


def build_real_denoiser(device: torch.device) -> Denoiser:
    sd = torch.load(RUN_DIR / "checkpoints" / "state.pt", map_location=device, weights_only=False)
    agent_sd = sd["agent"]
    denoiser_sd = {k[len("denoiser."):]: v for k, v in agent_sd.items() if k.startswith("denoiser.")}
    action_dim = int(denoiser_sd["inner_model.act_emb.0.weight"].shape[1])
    inner_cfg = InnerModelConfig(
        img_channels=3, num_steps_conditioning=4, cond_channels=256,
        depths=[2, 2, 2, 2], channels=[64, 64, 64, 64], attn_depths=[0, 0, 0, 0],
        continuous_action_dim=action_dim,
    )
    cfg = DenoiserConfig(inner_model=inner_cfg, sigma_data=0.5, sigma_offset_noise=0.3)
    denoiser = Denoiser(cfg).to(device)
    denoiser.load_state_dict(denoiser_sd)
    denoiser.eval()
    return denoiser


def collect_episode(env, rng: np.random.Generator) -> Episode:
    obs_frames, act_list, rew_list, end_list, trunc_list = [], [], [], [], []
    raw_obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    done = False
    while not done:
        action = rng.uniform(env.action_low, env.action_high).astype(np.float32)
        next_raw_obs, rew, terminated, truncated, _ = env.step(action)
        obs_frames.append(raw_obs)
        act_list.append(action)
        rew_list.append(rew)
        end_list.append(int(terminated))
        trunc_list.append(int(truncated))
        raw_obs = next_raw_obs
        done = terminated or truncated
    obs = torch.from_numpy(np.stack(obs_frames)).float().div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous()
    act = torch.from_numpy(np.stack(act_list)).float()
    rew = torch.tensor(rew_list, dtype=torch.float32)
    end = torch.tensor(end_list, dtype=torch.uint8)
    trunc = torch.tensor(trunc_list, dtype=torch.uint8)
    return Episode(obs=obs, act=act, rew=rew, end=end, trunc=trunc, info={})


def build_static_dataset(target_num_steps: int = TARGET_N, seed: int = SEED) -> Dataset:
    if SCRATCH_DIR.exists():
        shutil.rmtree(SCRATCH_DIR)
    dataset = Dataset(SCRATCH_DIR, "lcg_stage2_5", cache_in_ram=True)
    stray_path = RUN_DIR / "dataset" / "train" / "000" / "00" / "0" / "0.pt"
    if stray_path.is_file():
        dataset.add_episode(Episode.load(stray_path))
    env = make_dm_control_env(domain_name=DOMAIN_NAME, task_name=TASK_NAME, **ENV_KWARGS)
    rng = np.random.default_rng(seed)
    while dataset.num_steps < target_num_steps:
        dataset.add_episode(collect_episode(env, rng))
    dataset.save_to_default_path()
    return dataset


def theta_s_groups(denoiser: Denoiser):
    final_level = denoiser.inner_model.unet.u_blocks[-1]
    groups = [(f"resblock_{i}", list(rb.parameters())) for i, rb in enumerate(final_level.resblocks)]
    groups.append(("norm_out", list(denoiser.inner_model.norm_out.parameters())))
    groups.append(("conv_out", list(denoiser.inner_model.conv_out.parameters())))
    flat_check = [p for _, plist in groups for p in plist]
    real = selected_parameters(denoiser)
    assert len(flat_check) == len(real) and all(a is b for a, b in zip(flat_check, real)), (
        "theta_s_groups ordering does not match selected_parameters() -- offsets would be wrong"
    )
    return groups


def group_slices(denoiser: Denoiser):
    offset = 0
    slices = []
    for name, plist in theta_s_groups(denoiser):
        n = sum(p.numel() for p in plist)
        slices.append((name, offset, offset + n))
        offset += n
    return slices


# --------------------------------------------------------------------------------------
# Diagnostic 1: numerical cancellation
# --------------------------------------------------------------------------------------


def diagnostic_1(denoiser, params, obs, act, y, sigma_values):
    print("\n" + "=" * 88)
    print("DIAGNOSTIC 1: numerical cancellation of sqrt(2 w(sigma)) * J_D vs sqrt(2) * J_F")
    print("=" * 88)
    sqrt2 = math.sqrt(2.0)
    for label, sigma_val in sigma_values:
        sigma = torch.tensor([sigma_val], device=obs.device)
        eps = torch.randn_like(y)
        y_sigma = (y + sigma.view(-1, 1, 1, 1) * eps).detach()
        xi = torch.randn_like(y)

        # (a) current implementation: sqrt(2 w(sigma)) * J_D^T xi
        v_D, _ = compute_vjp(denoiser, params, y_sigma, sigma, obs, act, xi=xi.clone())

        # (b) directly through the preconditioned network output: sqrt(2) * J_F^T xi
        cs = denoiser.compute_conditioners(sigma)
        model_output = denoiser.compute_model_output(y_sigma, obs, act, cs)
        scalar_F = (sqrt2 * xi * model_output).sum()
        grads_F = torch.autograd.grad(scalar_F, params, retain_graph=False, create_graph=False)
        v_F = torch.cat([g.reshape(-1) for g in grads_F])

        rel_diff = ((v_D - v_F).norm() / (v_F.norm() + 1e-30)).item()
        max_abs_diff = (v_D - v_F).abs().max().item()
        c_out = cs.c_out.reshape(()).item()
        print(f"  sigma={sigma_val:<10.5g} ({label:<6})  c_out={c_out:.6g}  "
              f"||v_D||={v_D.norm().item():.6g}  ||v_F||={v_F.norm().item():.6g}  "
              f"rel_diff={rel_diff:.3e}  max_abs_diff={max_abs_diff:.3e}")


# --------------------------------------------------------------------------------------
# Combined collection for diagnostics 2, 3, 4: one pass, B transitions x num_strata strata
# --------------------------------------------------------------------------------------


def collect_v_squared(denoiser, params, dataset, B, num_strata, seed):
    torch.manual_seed(seed)
    d_S = sum(p.numel() for p in params)
    num_steps_conditioning = denoiser.cfg.inner_model.num_steps_conditioning
    segment_ids = sample_valid_transitions(dataset, B, num_steps_conditioning, seed=seed)

    v2 = torch.zeros(B, num_strata, d_S)  # CPU accumulator, avoids holding it all on GPU
    sigmas = torch.zeros(B, num_strata)

    for i, segment_id in enumerate(segment_ids):
        obs, act, y = load_transition(dataset, segment_id, num_steps_conditioning, denoiser.device)
        for m in range(num_strata):
            sigma = sample_sigma_stratum(SIGMA_CFG, m, num_strata, 1, denoiser.device)
            eps = torch.randn_like(y)
            y_sigma = (y + sigma.view(-1, 1, 1, 1) * eps).detach()
            v, _ = compute_vjp(denoiser, params, y_sigma, sigma, obs, act)
            v2[i, m] = (v * v).detach().cpu()
            sigmas[i, m] = sigma.item()

    return v2, sigmas


# --------------------------------------------------------------------------------------
# Diagnostic 2: curvature by sigma stratum
# --------------------------------------------------------------------------------------


def diagnostic_2(v2: torch.Tensor, sigmas: torch.Tensor, N: int, B: int, beta: float, num_strata: int):
    print("\n" + "=" * 88)
    print("DIAGNOSTIC 2: curvature by sigma stratum (after the c_out cancellation)")
    print("=" * 88)
    scale = beta * (N / B)
    total_mass = (scale / num_strata) * v2.sum().item()

    for m in range(num_strata):
        v2_m = v2[:, m, :]  # (B, d_S)
        sig_m = sigmas[:, m]
        per_draw_sq_norm = v2_m.sum(dim=1)  # (B,) = ||v||^2 per transition-draw
        per_coord_mean = v2_m.mean(dim=0)  # (d_S,) average v^2 per coordinate in this stratum

        q_coord = torch.quantile(per_coord_mean, torch.tensor([0.5, 0.9, 0.99]))
        q_draw = torch.quantile(per_draw_sq_norm, torch.tensor([0.5, 0.9, 0.99]))

        frac_nonzero = (per_coord_mean > 0).float().mean().item()
        frac_above_1e8 = (per_coord_mean > 1e-8).float().mean().item()
        frac_above_damping = (per_coord_mean > DAMPING).float().mean().item()

        stratum_mass = (scale / num_strata) * v2_m.sum().item()
        mass_fraction = stratum_mass / total_mass

        print(f"\n  --- stratum {m} ---")
        print(f"    sigma range: [{sig_m.min().item():.5g}, {sig_m.max().item():.5g}]  "
              f"median sigma: {sig_m.median().item():.5g}")
        print(f"    ||v||^2 per transition-draw: mean={per_draw_sq_norm.mean().item():.6g} "
              f"median={q_draw[0].item():.6g} p90={q_draw[1].item():.6g} p99={q_draw[2].item():.6g}")
        print(f"    per-coordinate mean v^2 (across {B} transitions): "
              f"mean={per_coord_mean.mean().item():.6g} median={q_coord[0].item():.6g} "
              f"p90={q_coord[1].item():.6g} p99={q_coord[2].item():.6g} max={per_coord_mean.max().item():.6g}")
        print(f"    fraction of coordinates: nonzero={frac_nonzero:.4f}  "
              f">1e-8={frac_above_1e8:.4f}  >damping(1e-4)={frac_above_damping:.4f}")
        print(f"    contribution to total (undamped) h_D mass: {mass_fraction:.4f}")


# --------------------------------------------------------------------------------------
# Diagnostic 3: module-wise concentration
# --------------------------------------------------------------------------------------


def diagnostic_3(denoiser, v2: torch.Tensor, N: int, B: int, beta: float, num_strata: int, damping: float):
    print("\n" + "=" * 88)
    print("DIAGNOSTIC 3: module-wise concentration of accumulated curvature")
    print("=" * 88)
    scale = beta * (N / B)
    h_D = damping + (scale / num_strata) * v2.sum(dim=(0, 1))  # (d_S,), matches historical_precision exactly
    excess_total = (h_D - damping).sum().item()

    for name, start, stop in group_slices(denoiser):
        h_group = h_D[start:stop]
        excess = (h_group - damping).sum().item()
        q = torch.quantile(h_group, torch.tensor([0.5, 0.99]))
        print(f"\n  --- {name} ---")
        print(f"    param count: {stop - start}")
        print(f"    sum(h_D) [incl. damping]: {h_group.sum().item():.6g}")
        print(f"    sum(h_D - damping) [pure accumulated curvature]: {excess:.6g}")
        print(f"    median={q[0].item():.6g}  p99={q[1].item():.6g}  max={h_group.max().item():.6g}")
        print(f"    fraction of total (undamped) precision mass: {excess / excess_total:.4f}")

    return h_D


# --------------------------------------------------------------------------------------
# Diagnostic 4: damping-floor domination
# --------------------------------------------------------------------------------------


def diagnostic_4(v2: torch.Tensor, h_D: torch.Tensor, damping: float):
    print("\n" + "=" * 88)
    print("DIAGNOSTIC 4: damping-floor domination -- exactly zero vs tiny vs sparse")
    print("=" * 88)
    B, num_strata, d_S = v2.shape
    flat = v2.reshape(B * num_strata, d_S)  # (B*num_strata, d_S)

    excess = h_D - damping  # raw accumulated sum before damping, per coordinate
    exactly_zero = excess == 0
    frac_exactly_zero = exactly_zero.float().mean().item()

    at_floor = h_D <= damping * 1.01
    frac_at_floor = at_floor.float().mean().item()

    tiny_but_nonzero = at_floor & (~exactly_zero)
    frac_tiny_nonzero = tiny_but_nonzero.float().mean().item()

    nnz_count_per_coord = (flat > 0).sum(dim=0)  # (d_S,), how many of the B*num_strata draws were nonzero
    print(f"  total coordinates: {d_S}")
    print(f"  fraction at damping floor (h_D <= 1.01*damping): {frac_at_floor:.4f}")
    print(f"  fraction EXACTLY zero accumulated curvature (before damping): {frac_exactly_zero:.4f}")
    print(f"  fraction at floor but nonzero (tiny relative to damping): {frac_tiny_nonzero:.4f}")

    print(f"\n  nnz draws per coordinate (out of {B * num_strata} total draws), overall:")
    print(f"    min={nnz_count_per_coord.min().item()}  median={nnz_count_per_coord.float().median().item():.1f}  "
          f"max={nnz_count_per_coord.max().item()}")

    if frac_at_floor > 0:
        floor_idx = at_floor.nonzero(as_tuple=True)[0]
        sample_idx = floor_idx[torch.randperm(len(floor_idx))[: min(5000, len(floor_idx))]]
        nnz_floor = nnz_count_per_coord[sample_idx]
        print(f"\n  among a random sample of {len(sample_idx)} floor-dominated coordinates:")
        print(f"    nnz draws/coord: min={nnz_floor.min().item()}  median={nnz_floor.float().median().item():.1f}  "
              f"max={nnz_floor.max().item()}  (out of {B * num_strata})")
        frac_all_active = (nnz_floor == B * num_strata).float().mean().item()
        frac_sparse = (nnz_floor < (B * num_strata) // 2).float().mean().item()
        print(f"    fraction with EVERY draw active (uniformly tiny): {frac_all_active:.4f}")
        print(f"    fraction active in <50% of draws (sparse activation): {frac_sparse:.4f}")

    if (~at_floor).any():
        nonfloor_idx = (~at_floor).nonzero(as_tuple=True)[0]
        sample_idx = nonfloor_idx[torch.randperm(len(nonfloor_idx))[: min(2000, len(nonfloor_idx))]]
        nnz_nonfloor = nnz_count_per_coord[sample_idx]
        print(f"\n  among a random sample of {len(sample_idx)} NON-floor coordinates:")
        print(f"    nnz draws/coord: min={nnz_nonfloor.min().item()}  median={nnz_nonfloor.float().median().item():.1f}  "
              f"max={nnz_nonfloor.max().item()}  (out of {B * num_strata})")


# --------------------------------------------------------------------------------------
# Diagnostic 5: checkpoint maturity context
# --------------------------------------------------------------------------------------


def diagnostic_5(denoiser: Denoiser):
    print("\n" + "=" * 88)
    print("DIAGNOSTIC 5: checkpoint-maturity context")
    print("=" * 88)
    print("  DIAMOND zero-initializes InnerModel.conv_out.weight and every ResBlock.conv2.weight")
    print("  at construction (see inner_model.py / blocks.py). Their current norm is a proxy for")
    print("  how far training has moved them from a fresh, untrained network:")
    final_level = denoiser.inner_model.unet.u_blocks[-1]
    zero_init_tensors = [("conv_out.weight", denoiser.inner_model.conv_out.weight)]
    for i, rb in enumerate(final_level.resblocks):
        zero_init_tensors.append((f"u_blocks[-1].resblocks.{i}.conv2.weight", rb.conv2.weight))
    for name, p in zero_init_tensors:
        print(f"    {name}: norm={p.norm().item():.6g}, mean_abs={p.abs().mean().item():.6g}, shape={tuple(p.shape)}")

    other_weight = final_level.resblocks[0].conv1.weight
    print(f"    (for reference, a non-zero-init sibling) resblocks.0.conv1.weight: "
          f"norm={other_weight.norm().item():.6g}, mean_abs={other_weight.abs().mean().item():.6g}")
    print("\n  Interpretation: if the zero-init tensors' norms are still small relative to the")
    print("  non-zero-init sibling above, this checkpoint is close to initialization, and any")
    print("  heterogeneity in accumulated curvature involving those tensors specifically should")
    print("  be treated as a checkpoint-maturity artifact, not a property of a trained LCG model.")


# --------------------------------------------------------------------------------------


def main():
    print("from diagnose_lcg_stage2_5.py: running Stage 2.5 diagnostic checks...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    denoiser = build_real_denoiser(device)
    params = selected_parameters(denoiser)
    dataset = build_static_dataset()
    N = dataset.num_steps
    print(f"Real denoiser loaded (device={device}). Static dataset: {dataset.num_episodes} episodes, N={N} steps.")

    # ---- Diagnostic 1 ----
    num_steps_conditioning = denoiser.cfg.inner_model.num_steps_conditioning
    seg0 = sample_valid_transitions(dataset, 1, num_steps_conditioning, seed=SEED)[0]
    obs0, act0, y0 = load_transition(dataset, seg0, num_steps_conditioning, device)
    sigma_values = [("low", SIGMA_CFG.sigma_min), ("mid", math.exp(SIGMA_CFG.loc)), ("high", 15.0)]
    diagnostic_1(denoiser, params, obs0, act0, y0, sigma_values)

    # ---- Shared collection for diagnostics 2-4 ----
    v2, sigmas = collect_v_squared(denoiser, params, dataset, B, NUM_STRATA, SEED)

    diagnostic_2(v2, sigmas, N, B, BETA, NUM_STRATA)
    h_D = diagnostic_3(denoiser, v2, N, B, BETA, NUM_STRATA, DAMPING)
    diagnostic_4(v2, h_D, DAMPING)
    diagnostic_5(denoiser)

    print("\nStage 2.5 diagnostic complete.")


if __name__ == "__main__":
    main()
