#! /usr/bin/env python
"""
LCG Stage 2 validation: offline diagonal Laplace/Gauss-Newton precision estimation
(Algorithm 2), against the REAL DIAMOND dm_control denoiser architecture and weights --
not the Stage 1 toy model.

No fully-trained-checkpoint + complete-static-dataset pair currently exists in this repo:
`outputs/2026-08-17/15-35-46` is a short smoke-test run (config overrides show
steps_first_epoch=1 etc.) whose embedded `train_dataset` bookkeeping was never
checkpointed past epoch 0, even though its collector did write one real 40-step episode to
disk. This script therefore:
  - loads the REAL denoiser architecture and REAL (if minimally trained) weights from that
    run's checkpoint (config: 4-level 64-channel UNet, cond_channels=256, dm_control
    cheetah/run, action_dim=6 -- action_dim is read off the checkpoint's own
    act_emb.0.weight shape, not hard-coded);
  - recovers that one real episode, then supplements it with freshly-collected episodes
    from the *same* env config (cheetah/run, action_repeat=2, time_limit=1.0) via a random
    policy (clearly not a trained collection policy, but real dm_control rollouts through
    the real Episode/Dataset pipeline) to reach a usable N;
  - freezes both, and treats them as "frozen world model + static replay data" for the
    remainder of the script.

Runs Algorithm 2 (`lcg.precision.historical_precision`) and the 5 requested diagnostics:
basic validity, batch-size scaling (B vs 2B), seed stability, stratification
(num_strata=1 vs 3), and cost.

Usage:
    python scripts/validate_lcg_stage2.py
"""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from data import Dataset, Episode
from envs.dm_control_env import make_dm_control_env
from lcg.precision import historical_precision
from lcg.theta_s import selected_dim, selected_parameters, selected_submodules
from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.diffusion.inner_model import InnerModelConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = REPO_ROOT / "outputs" / "2026-08-17" / "15-35-46"
SCRATCH_DIR = Path(
    r"C:\Users\jerry\AppData\Local\Temp\claude\c--Users-jerry-Project-LCG"
    r"\974bd621-91fa-4766-822e-267b547a92c5\scratchpad\lcg_stage2_dataset"
)

DOMAIN_NAME, TASK_NAME = "cheetah", "run"
ENV_KWARGS = dict(size=64, camera_id=0, action_repeat=2, time_limit=1.0)
SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)

B_BASE = 40
DAMPING = 1e-4
BETA = 1.0


# --------------------------------------------------------------------------------------
# Real model + real (supplemented) static dataset
# --------------------------------------------------------------------------------------


def build_real_denoiser(device: torch.device) -> Denoiser:
    sd = torch.load(RUN_DIR / "checkpoints" / "state.pt", map_location=device, weights_only=False)
    agent_sd = sd["agent"]
    denoiser_sd = {k[len("denoiser."):]: v for k, v in agent_sd.items() if k.startswith("denoiser.")}
    action_dim = int(denoiser_sd["inner_model.act_emb.0.weight"].shape[1])

    inner_cfg = InnerModelConfig(
        img_channels=3,
        num_steps_conditioning=4,
        cond_channels=256,
        depths=[2, 2, 2, 2],
        channels=[64, 64, 64, 64],
        attn_depths=[0, 0, 0, 0],
        continuous_action_dim=action_dim,
    )
    cfg = DenoiserConfig(inner_model=inner_cfg, sigma_data=0.5, sigma_offset_noise=0.3)
    denoiser = Denoiser(cfg).to(device)
    denoiser.load_state_dict(denoiser_sd)
    denoiser.eval()
    print(f"Loaded real denoiser from {RUN_DIR.relative_to(REPO_ROOT)} (action_dim={action_dim}, device={device}).")
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


def build_static_dataset(target_num_steps: int = 400, seed: int = 0) -> Dataset:
    if SCRATCH_DIR.exists():
        shutil.rmtree(SCRATCH_DIR)
    dataset = Dataset(SCRATCH_DIR, "lcg_stage2", cache_in_ram=True)

    stray_path = RUN_DIR / "dataset" / "train" / "000" / "00" / "0" / "0.pt"
    if stray_path.is_file():
        episode = Episode.load(stray_path)
        dataset.add_episode(episode)
        print(f"Recovered 1 real episode from the original run ({len(episode)} steps).")

    env = make_dm_control_env(domain_name=DOMAIN_NAME, task_name=TASK_NAME, **ENV_KWARGS)
    rng = np.random.default_rng(seed)
    num_collected = 0
    while dataset.num_steps < target_num_steps:
        dataset.add_episode(collect_episode(env, rng))
        num_collected += 1
    print(f"Collected {num_collected} additional random-policy episodes from {DOMAIN_NAME}/{TASK_NAME}.")
    print(f"Static dataset: {dataset.num_episodes} episodes, {dataset.num_steps} steps (N).")
    dataset.save_to_default_path()
    return dataset


def report_theta_s(denoiser: Denoiser) -> list:
    modules = selected_submodules(denoiser)
    labels = ["unet.u_blocks[-1] (final decoder level, 3 ResBlocks)", "norm_out", "conv_out"]
    print("\n=== theta_S: selected modules ===")
    for label, module in zip(labels, modules):
        print(f"  {label}: {type(module).__name__}")
        for pname, p in module.named_parameters():
            print(f"    {pname}: {tuple(p.shape)} ({p.numel()} params)")
    params = selected_parameters(denoiser)
    d_s = selected_dim(denoiser)
    print(f"  num selected parameter tensors: {len(params)}")
    print(f"  d_S (real flattened dim) = {d_s}")
    print(f"  memory for one float32 h_D vector: {d_s * 4 / 1e6:.3f} MB")
    return params


# --------------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------------


def state_dict_snapshot(denoiser: Denoiser) -> dict:
    return {k: v.detach().clone() for k, v in denoiser.state_dict().items()}


def assert_denoiser_unchanged(denoiser: Denoiser, before: dict) -> None:
    for k, v in denoiser.state_dict().items():
        assert torch.equal(before[k], v), f"denoiser parameter {k} changed during precision estimation!"
    print("[integrity] denoiser parameters unchanged before vs after estimation: OK")


def report_basic_validity(h: torch.Tensor, damping: float, label: str) -> dict:
    finite = torch.isfinite(h).all().item()
    nonneg = (h >= 0).all().item()
    at_least_damping = (h >= damping - 1e-9).all().item()
    frac_damped = (h <= damping * 1.01).float().mean().item()
    q50, q90, q99 = torch.quantile(h, torch.tensor([0.5, 0.9, 0.99], device=h.device)).tolist()
    stats = dict(
        shape=tuple(h.shape), finite=finite, nonneg=nonneg, at_least_damping=at_least_damping,
        frac_damped=frac_damped, min=h.min().item(), median=q50, mean=h.mean().item(),
        p90=q90, p99=q99, max=h.max().item(),
    )
    print(f"\n=== basic validity: {label} ===")
    print(f"  shape={stats['shape']}  finite={finite}  nonneg={nonneg}  h[k]>=damping={at_least_damping}")
    print(f"  min={stats['min']:.6g}  median={stats['median']:.6g}  mean={stats['mean']:.6g}  "
          f"p90={stats['p90']:.6g}  p99={stats['p99']:.6g}  max={stats['max']:.6g}")
    print(f"  fraction of coordinates within 1% of the damping floor: {frac_damped:.4f}")
    assert finite and nonneg and at_least_damping
    return stats


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a - a.mean(), b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-12)).item()


def run_estimate(denoiser, params, dataset, B, num_strata, seed) -> torch.Tensor:
    return historical_precision(
        denoiser, params, dataset, SIGMA_CFG, B=B, num_strata=num_strata,
        beta=BETA, damping=DAMPING, seed=seed,
    )


def summarize_group(hs: list, label: str) -> None:
    means = torch.tensor([h.mean().item() for h in hs])
    medians = torch.tensor([torch.median(h).item() for h in hs])
    print(f"  [{label}] mean(h) across {len(hs)} draws: {means.tolist()} (std={means.std().item():.6g})")
    print(f"  [{label}] median(h) across {len(hs)} draws: {medians.tolist()} (std={medians.std().item():.6g})")


def pairwise_stats(hs: list) -> dict:
    corrs, rel_l2 = [], []
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            corrs.append(pearson_corr(hs[i], hs[j]))
            rel_l2.append(((hs[i] - hs[j]).norm() / hs[j].norm()).item())
    return dict(mean_corr=float(np.mean(corrs)), mean_rel_l2=float(np.mean(rel_l2)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    denoiser = build_real_denoiser(device)
    params = report_theta_s(denoiser)
    dataset = build_static_dataset(target_num_steps=400, seed=0)
    N = dataset.num_steps

    print("from validate_lcg_stage2.py: running Stage 2 validation checks...")
    # ---- Diagnostic 1: basic validity + parameter-integrity check ----
    before = state_dict_snapshot(denoiser)
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    h_main = run_estimate(denoiser, params, dataset, B=B_BASE, num_strata=3, seed=0)
    t_main = time.perf_counter() - t0
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None
    assert_denoiser_unchanged(denoiser, before)
    report_basic_validity(h_main, DAMPING, f"main estimate (B={B_BASE}, num_strata=3, N={N})")

    # ---- Diagnostic 2: batch-size scaling (B vs 2B), N/B-corrected ----
    print("\n=== batch-size scaling: standard vs B vs 2B (N/B-corrected) ===")
    seeds_B = [0, 1, 2, 3]
    seeds_2B = [10, 11, 12, 13]
    #hs_standard = [run_estimate(denoiser, params, dataset, B=dataset.num_steps, num_strata=3, seed=s) for s in seeds_B]    
    hs_B = [run_estimate(denoiser, params, dataset, B=B_BASE, num_strata=3, seed=s) for s in seeds_B]
    hs_2B = [run_estimate(denoiser, params, dataset, B=2 * B_BASE, num_strata=3, seed=s) for s in seeds_2B]
    #summarize_group(hs_standard, "B=1")
    summarize_group(hs_B, f"B={B_BASE}")
    summarize_group(hs_2B, f"B={2 * B_BASE}")
    #norm_standard = torch.stack([(h - DAMPING).norm() for h in hs_standard])
    norm_B = torch.stack([(h - DAMPING).norm() for h in hs_B])
    norm_2B = torch.stack([(h - DAMPING).norm() for h in hs_2B])
    print(f"  ||h - damping|| : B={B_BASE} -> mean={norm_B.mean():.4g} std={norm_B.std():.4g} | "
              f"B={2 * B_BASE} -> mean={norm_2B.mean():.4g} std={norm_2B.std():.4g}")
    #print(f"  ||h - damping|| : B=1 -> mean={norm_standard.mean():.4g} std={norm_standard.std():.4g} | " 
     #     f"B={B_BASE} -> mean={norm_B.mean():.4g} std={norm_B.std():.4g} | "
      #    f"B={2 * B_BASE} -> mean={norm_2B.mean():.4g} std={norm_2B.std():.4g}")
    cross_corr = pearson_corr(hs_B[0], hs_2B[0])
    print(f"  coordinate-wise correlation, one B draw vs one 2B draw: {cross_corr:.4f}")
    scale_ratio = (norm_2B.mean() / norm_B.mean()).item()
    print(f"  ||h-damping|| ratio (2B / B): {scale_ratio:.4f}  (expect ~1.0, i.e. no systematic scale drift; "
          f"std should shrink with B)")

    # ---- Diagnostic 3: seed stability (reusing the B_BASE group above) ----
    print(f"\n=== seed stability (B={B_BASE}, num_strata=3, {len(hs_B)} seeds) ===")
    stab_B = pairwise_stats(hs_B)
    print(f"  mean pairwise correlation: {stab_B['mean_corr']:.4f}")
    print(f"  mean pairwise relative L2 difference: {stab_B['mean_rel_l2']:.4f}")

    # ---- Diagnostic 4: stratification, num_strata=1 vs 3 ----
    print("\n=== stratification: num_strata=1 vs num_strata=3 ===")
    seeds_strat = [0, 1, 2, 3]
    # Same B, same seeds as hs_B already computed for num_strata=3 -> reuse directly.
    hs_strata3_sameB = hs_B
    hs_strata1_sameB = [run_estimate(denoiser, params, dataset, B=B_BASE, num_strata=1, seed=s) for s in seeds_strat]
    # Compute-matched: num_strata=1 with 3x the transitions, same total VJP budget as num_strata=3 @ B_BASE.
    hs_strata1_matched = [
        run_estimate(denoiser, params, dataset, B=3 * B_BASE, num_strata=1, seed=100 + s) for s in seeds_strat
    ]
    stab_strata1_sameB = pairwise_stats(hs_strata1_sameB)
    stab_strata1_matched = pairwise_stats(hs_strata1_matched)
    print(f"  num_strata=3, B={B_BASE}      (same VJP budget as below): "
          f"mean_corr={stab_B['mean_corr']:.4f}  mean_rel_l2={stab_B['mean_rel_l2']:.4f}")
    print(f"  num_strata=1, B={3 * B_BASE}  (VJP-budget-matched):        "
          f"mean_corr={stab_strata1_matched['mean_corr']:.4f}  mean_rel_l2={stab_strata1_matched['mean_rel_l2']:.4f}")
    print(f"  num_strata=1, B={B_BASE}      (same B, 3x fewer VJPs):     "
          f"mean_corr={stab_strata1_sameB['mean_corr']:.4f}  mean_rel_l2={stab_strata1_sameB['mean_rel_l2']:.4f}")
    more_stable = stab_B["mean_rel_l2"] < stab_strata1_matched["mean_rel_l2"]
    print(f"  stratified (num_strata=3) more stable than compute-matched num_strata=1: {more_stable}")

    # ---- Diagnostic 5: cost ----
    print("\n=== cost ===")
    print(f"  VJP/backward passes per transition (main run): 3 (num_strata)")
    print(f"  total VJPs (main run, B={B_BASE}): {B_BASE * 3}")
    print(f"  runtime (main run): {t_main:.2f} s ({t_main / (B_BASE * 3) * 1000:.1f} ms/VJP)")
    if peak_mem_mb is not None:
        print(f"  peak GPU memory (main run): {peak_mem_mb:.1f} MB")
    else:
        print("  peak GPU memory: n/a (running on CPU)")
    print(f"  real-model d_S: {sum(p.numel() for p in params)}")

    print("\nStage 2 validation complete.")


if __name__ == "__main__":
    main()
