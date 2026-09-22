"""Phase 11 -- LCG on IMAGINED context + IMAGINED target; only the ACTIONS come from
the held-out episode.

Phase 8 replaced only the target y with the model's own sample y*; the context frames were
still the real held-out frames. In production (WorldModelEnv) the context is the model's own
rolling imagined buffer -- real frames appear only at reset -- and the target is the model's
own sample. This script tests that setting directly.

For a candidate at held-out transition t and imagination depth d:
  1. start from the REAL n_cond context frames that end d steps before t
     (frames t-d-n .. t-d-1);
  2. roll the world model forward d steps: each step conditions on the last n frames (which
     become fully imagined once d >= n) and on the REAL held-out action for that step, samples
     one frame with the production DiffusionSampler (single draw, like production), and appends
     it to the context;
  3. the scored candidate is (imagined context, real action window, y*) where y* is the model's
     sample for frame t. LCG then scores it exactly as in Phases 5/8 (same h_D, theta_S,
     candidate bank, M).
d=0 is exactly Phase 8's self-y setting (real context). Production's imagination horizon is 15,
so depths up to 14 are production-reachable; d>=4 means the context is entirely imagined.

Every depth is scored on the SAME candidate set (transitions t >= max_depth + n_cond, which also
drops the t=0 blank-padding artifact), alongside a real-y reference on that same set, so depths are
directly comparable. Also saved per candidate: MSE of y* against the real frame at t, as a drift
measure.

No retraining, no production-code changes.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase11_imagined_context_lcg.py
    python scripts/lcg_diagnostic/pretrained_collectors/phase11_imagined_context_lcg.py --smoke hopper
"""
from __future__ import annotations

from hopper_naming import cond_dir, slot_dir

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

_THIS_DIR = Path(__file__).resolve().parent
_LCG_ROOT = _THIS_DIR.parent.parent.parent
_PROJECT_ROOT = _LCG_ROOT.parent
sys.path.insert(0, str(_LCG_ROOT / "src"))
sys.path.insert(0, str(_THIS_DIR))

from lcg.forward_jvp import make_jvp_bank, score_one_jvp_bank  # noqa: E402
from lcg.theta_s import selected_named_parameters, frozen_named_parameters  # noqa: E402
from lcg.precision import historical_precision, assert_setup_valid  # noqa: E402
from models.diffusion.diffusion_sampler import DiffusionSampler  # noqa: E402
from data import Dataset  # noqa: E402

from phase5_lcg_scoring import (  # noqa: E402
    load_agent, evenly_spaced_indices, HELD_OUT_EPISODE_IDS, EPISODE_LEN, N_TRANS_PER_EPISODE,
    PRECISION_REFERENCE_SIZE, PRECISION_NUM_MC, BETA, DAMPING, CANDIDATE_NUM_MC, CANDIDATE_CHUNK_SIZE,
    PRECISION_SEED, BANK_SEED, THETA_S, MIXTURES, SOURCE_POOLS,
)
from phase6_full_ensemble import checkpoint_path  # noqa: E402
from phase8c_multiseed_selfy_check import summarize  # noqa: E402

DOMAINS = ["walker", "quadruped", "hopper"]
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]
SEEDS = ["A", "B", "C"]
DEPTHS = [0, 4, 8, 14]
PROBE_TASK = {"walker": "walk", "quadruped": "walk", "hopper": "stand"}
SAMPLE_SEED = 24681  # same family of documented seed as Phase 8's Y_SAMPLE_SEED
ROLLOUT_CHUNK = 96

OUT_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "phase11_imagined_context_lcg"


def select_candidates(pools: dict, n_cond: int, smoke: bool) -> list[dict]:
    dmax = max(DEPTHS)
    indices = [int(t) for t in evenly_spaced_indices(EPISODE_LEN, N_TRANS_PER_EPISODE) if t >= dmax + n_cond]
    if smoke:
        indices = indices[:: max(1, len(indices) // 6)][:6]
    cands = []
    for behavior in ("walk", "run"):
        for eid in HELD_OUT_EPISODE_IDS:
            ep = pools[behavior].load_episode(eid)
            for t in indices:
                cands.append({"behavior": behavior, "episode_id": int(eid), "t": t, "obs": ep.obs, "act": ep.act})
    return cands


def gather(cands: list[dict], start_of, length: int, key: str, device) -> torch.Tensor:
    """(B, length, ...) window of each candidate's episode tensor, starting at start_of(c)."""
    return torch.stack([c[key][start_of(c): start_of(c) + length] for c in cands]).to(device)


@torch.no_grad()
def build_candidates(cands: list[dict], depth: int, n_cond: int, sampler: DiffusionSampler, device):
    """Returns (ctx (B,n,c,h,w), act (B,n,adim), y_star (B,c,h,w)) for the given imagination depth."""
    ctx_all, act_all, y_all = [], [], []
    for lo in range(0, len(cands), ROLLOUT_CHUNK):
        chunk = cands[lo: lo + ROLLOUT_CHUNK]
        torch.manual_seed(SAMPLE_SEED + depth * 100_003 + lo)
        ctx = gather(chunk, lambda c: c["t"] - depth - n_cond, n_cond, "obs", device)          # real start context
        acts = gather(chunk, lambda c: c["t"] - depth - n_cond, n_cond + depth, "act", device)  # real actions
        for j in range(depth):
            new_frame, _ = sampler.sample(ctx, acts[:, j: j + n_cond])
            ctx = torch.cat([ctx[:, 1:], new_frame.unsqueeze(1)], dim=1)
        act_win = acts[:, depth: depth + n_cond]
        y_star, _ = sampler.sample(ctx, act_win)
        ctx_all.append(ctx); act_all.append(act_win); y_all.append(y_star)
    return torch.cat(ctx_all), torch.cat(act_all), torch.cat(y_all)


def to_scoring_list(ctx: torch.Tensor, act: torch.Tensor, y: torch.Tensor) -> list:
    b, n, c, h, w = ctx.shape
    flat = ctx.reshape(b, n * c, h, w)
    return [(flat[i: i + 1], act[i: i + 1], y[i: i + 1].float()) for i in range(b)]


def process(domain: str, condition: str, seed: str, pools: dict, smoke: bool) -> dict:
    ckpt = checkpoint_path(domain, condition, seed)
    assert ckpt.exists(), f"missing checkpoint: {ckpt}"
    agent, sigma_cfg, _ = load_agent(domain, ckpt, probe_task=PROBE_TASK[domain])
    denoiser = agent.denoiser
    device = denoiser.device
    n_cond = denoiser.cfg.inner_model.num_steps_conditioning

    with initialize_config_dir(version_base="1.3", config_dir=str(_LCG_ROOT / "config")):
        cfg = compose(config_name="trainer", overrides=["env=dm_control"])
    diffusion_cfg = instantiate(cfg.world_model_env.diffusion_sampler)
    assert diffusion_cfg.s_churn == 0.0
    sampler = DiffusionSampler(denoiser, diffusion_cfg)

    cands = select_candidates(pools, n_cond, smoke)
    behaviors = np.array([c["behavior"] for c in cands])
    episode_ids = np.array([c["episode_id"] for c in cands])
    tidx = np.array([c["t"] for c in cands])
    print(f"  {len(cands)} candidates (t >= {max(DEPTHS) + n_cond}), depths={DEPTHS}", flush=True)

    theta_s_named = selected_named_parameters(denoiser, THETA_S)
    frozen_named = frozen_named_parameters(denoiser, theta_s_named)
    d_S = sum(p.numel() for p in theta_s_named.values())
    train_dataset = Dataset(MIXTURES / domain / cond_dir(domain, condition) / "dataset",
                             name=f"{domain}_{condition}_train", cache_in_ram=True)
    train_dataset.load_from_default_path()
    assert train_dataset.num_steps == 5000
    t0 = time.time()
    h_D = historical_precision(
        denoiser, list(theta_s_named.values()), train_dataset, sigma_cfg,
        B=PRECISION_REFERENCE_SIZE, N=train_dataset.num_steps, num_mc=PRECISION_NUM_MC,
        beta=BETA, damping=DAMPING, seed=PRECISION_SEED,
    )
    assert_setup_valid(theta_s_named, h_D, d_S=d_S)
    h_D_inv_sqrt = h_D.rsqrt()
    print(f"  h_D built in {time.time() - t0:.1f}s", flush=True)
    bank = make_jvp_bank(sigma_cfg, torch.Size([1, denoiser.cfg.inner_model.img_channels, 64, 64]), d_S, device,
                          num_samples=CANDIDATE_NUM_MC, seed=BANK_SEED)

    def score(cand_list) -> np.ndarray:
        s = score_one_jvp_bank(denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, bank, cand_list,
                               CANDIDATE_CHUNK_SIZE).cpu().numpy()
        assert np.isfinite(s).all()
        return s

    real_ctx = gather(cands, lambda c: c["t"] - n_cond, n_cond, "obs", device)
    real_act = gather(cands, lambda c: c["t"] - n_cond, n_cond, "act", device)
    real_y = torch.stack([c["obs"][c["t"]] for c in cands]).to(device)

    columns = {}
    summaries = {}
    t0 = time.time()
    columns["real_y"] = score(to_scoring_list(real_ctx, real_act, real_y))
    summaries["real_y"] = summarize(columns["real_y"], behaviors, episode_ids, "real_y")
    print(f"  real-y scored in {time.time() - t0:.1f}s", flush=True)

    mse = {}
    for d in DEPTHS:
        t0 = time.time()
        ctx, act, y_star = build_candidates(cands, d, n_cond, sampler, device)
        mse[d] = ((y_star - real_y) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
        ctx_mse = ((ctx[:, -1] - real_ctx[:, -1]) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
        columns[f"selfy_d{d}"] = score(to_scoring_list(ctx, act, y_star))
        summaries[f"selfy_d{d}"] = summarize(columns[f"selfy_d{d}"], behaviors, episode_ids, f"imagined d={d}")
        summaries[f"selfy_d{d}"]["mean_mse_y_star_vs_real"] = float(mse[d].mean())
        summaries[f"selfy_d{d}"]["mean_mse_last_context_vs_real"] = float(ctx_mse.mean())
        print(f"  depth {d}: built+scored in {time.time() - t0:.1f}s  mse(y*,real)={mse[d].mean():.5f}", flush=True)
        del ctx, act, y_star
        torch.cuda.empty_cache()

    out_dir = OUT_ROOT / ("smoke" if smoke else "") / domain / seed / cond_dir(domain, condition)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["domain", "seed", "behavior", "episode_id", "transition_index", "real_y_lcg_score",
              *[f"selfy_d{d}_lcg_score" for d in DEPTHS], *[f"mse_y_star_d{d}" for d in DEPTHS]]
    with open(out_dir / "per_transition_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(len(cands)):
            row = {"domain": domain, "seed": seed, "behavior": behaviors[i], "episode_id": int(episode_ids[i]),
                   "transition_index": int(tidx[i]), "real_y_lcg_score": float(columns["real_y"][i])}
            for d in DEPTHS:
                row[f"selfy_d{d}_lcg_score"] = float(columns[f"selfy_d{d}"][i])
                row[f"mse_y_star_d{d}"] = float(mse[d][i])
            w.writerow(row)

    del agent
    torch.cuda.empty_cache()
    return {"n_candidates": len(cands), "depths": DEPTHS, "summaries": summaries}


def main() -> None:
    smoke = "--smoke" in sys.argv
    domains = [sys.argv[sys.argv.index("--smoke") + 1]] if smoke and len(sys.argv) > sys.argv.index("--smoke") + 1 else DOMAINS
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results_path = OUT_ROOT / ("smoke_summary.json" if smoke else "phase11_summary.json")
    all_results = json.loads(results_path.read_text()) if results_path.exists() and not smoke else {}

    for domain in domains:
        pools = {}
        for beh in ("walk", "run"):
            ds = Dataset(SOURCE_POOLS / domain / slot_dir(domain, beh) / "dataset", name=f"{domain}_{beh}_pool", cache_in_ram=True)
            ds.load_from_default_path()
            pools[beh] = ds
        for condition in (CONDITIONS[:1] if smoke else CONDITIONS):
            for seed in SEEDS:
                key = f"{domain}/{seed}/{condition}"
                if key in all_results:
                    print(f">>> SKIPPING {key} (already done)", flush=True)
                    continue
                print(f"\n{'=' * 70}\n{key}\n{'=' * 70}", flush=True)
                all_results[key] = process(domain, condition, seed, pools, smoke)
                results_path.write_text(json.dumps(all_results, indent=2, default=str))

    print("\n" + "=" * 100)
    print("PHASE 11 SUMMARY: run-minus-walk LCG delta by imagination depth (real_y reference first)")
    print("=" * 100)
    for key, r in all_results.items():
        s = r["summaries"]
        cells = "  ".join(f"{k}={v['delta_mean']:+8.2f}{'*' if v['significant'] else ' '}" for k, v in s.items())
        print(f"{key:<32} {cells}")
    print(f"\nSaved: {results_path}   (* = episode-level bootstrap CI excludes 0)")


if __name__ == "__main__":
    main()
