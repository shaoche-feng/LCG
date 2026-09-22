"""Full six-ensemble disagreement diagnostic (all domain x coverage-condition
combinations), scaling the Phase 6 pilot from 20 candidates to the full
500-Walk + 500-Run held-out candidate set used by Phase 5's LCG scoring, so
per-candidate ensemble disagreement (ENS_5) can be directly compared against
the already-saved per-candidate LCG scores.

Ensemble members per domain/condition are the SAME Seed A/B/C denoiser
checkpoints already trained (see train_multiseed.py) -- no retraining, no
production-code changes. The production DiffusionSampler math is reused via
phase6_pilot_ensemble.py's sample_with_given_noise()/load_transition_5d(),
which were verified in the pilot to be bit-identical to production given
s_churn=0 (the only randomness is the initial noise tensor). Initial noise is
shared across models A/B/C AND across the three coverage conditions within a
domain (same candidate set, same architecture), exactly as in the pilot but
now for the full candidate set.

Held-out candidates use the EXACT same construction as phase5_lcg_scoring.py
(episodes {7,8,9,10,11} x 100 evenly-spaced transitions/episode, walk-then-run
concatenation order) so per-candidate LCG scores can be joined by
(behavior, episode_id, transition_index) key.

For memory/runtime reasons, only SUMMARY statistics (ENS_1..5, within-model
variance, prediction error) are computed for all 1000 candidates per
domain/condition; full generated images are re-sampled (cheap, same shared
noise, models still resident) only for a handful of representative candidates
chosen after the summary pass, for visualization.

Correlation statistics against LCG (Pearson/Spearman) and the S=1..5
stability table are deferred to phase6c_full_analysis.py, torch-free, for the
same OMP-conflict reason documented in phase6_pilot_ensemble.py.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase6_full_ensemble.py
"""
from __future__ import annotations

from hopper_naming import cond_dir, slot_dir

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from phase5_lcg_scoring import (  # noqa: E402
    load_agent, evenly_spaced_indices, HELD_OUT_EPISODE_IDS, EPISODE_LEN, N_TRANS_PER_EPISODE,
    SOURCE_POOLS, MODELS_ROOT,
)
from phase6_pilot_ensemble import load_transition_5d, sample_with_given_noise  # noqa: E402
from models.diffusion.diffusion_sampler import build_sigmas  # noqa: E402
from data import Dataset, SegmentId  # noqa: E402

DOMAINS = ["walker", "quadruped"]
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]
SEED_SUBDIRS = {"A": None, "B": "seed43", "C": "seed44"}  # None -> base MODELS_ROOT/domain/condition
NUM_SAMPLES = 5
NOISE_SEED = 24680  # distinct namespace from the pilot's 13579; documented fixed seed for the full run

_PROJECT_ROOT = _THIS_DIR.parent.parent.parent.parent
OUT_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "phase6_full_ensemble"
LCG_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "phase5_lcg_scoring"


def checkpoint_path(domain: str, condition: str, seed_label: str) -> Path:
    sub = SEED_SUBDIRS[seed_label]
    base = MODELS_ROOT if sub is None else (MODELS_ROOT / "multiseed" / sub)
    return base / domain / cond_dir(domain, condition) / "checkpoints" / "agent_versions" / "agent_epoch_00001.pt"


def load_lcg_scores(domain: str, condition: str) -> dict:
    path = LCG_ROOT / domain / cond_dir(domain, condition) / "per_transition_scores.csv"
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["behavior"], int(row["episode_id"]), int(row["transition_index"]))
            out[key] = float(row["lcg_score"])
    return out


def build_domain_candidates(domain: str, n_cond: int, device) -> list:
    indices = evenly_spaced_indices(EPISODE_LEN, N_TRANS_PER_EPISODE)
    walk_pool = Dataset(SOURCE_POOLS / domain / slot_dir(domain, "walk") / "dataset", name=f"{domain}_walk_pool", cache_in_ram=True)
    walk_pool.load_from_default_path()
    run_pool = Dataset(SOURCE_POOLS / domain / slot_dir(domain, "run") / "dataset", name=f"{domain}_run_pool", cache_in_ram=True)
    run_pool.load_from_default_path()
    pools = {"walk": walk_pool, "run": run_pool}

    candidates = []
    for behavior in ("walk", "run"):  # matches phase5_lcg_scoring's walk-then-run concatenation order
        for eid in HELD_OUT_EPISODE_IDS:
            for t in indices:
                t = int(t)
                seg = SegmentId(int(eid), t + 1 - (n_cond + 1), t + 1)
                obs5d, act, y = load_transition_5d(pools[behavior], seg, n_cond, device)
                candidates.append({"behavior": behavior, "episode_id": int(eid), "transition_index": t,
                                    "obs5d": obs5d, "act": act, "y": y})
    assert len(candidates) == 1000
    assert sum(1 for c in candidates if c["behavior"] == "walk") == 500
    assert sum(1 for c in candidates if c["behavior"] == "run") == 500
    return candidates


def build_domain_noise(num_candidates: int, device) -> list:
    noises = []
    for ci in range(num_candidates):
        gen = torch.Generator(device=device)
        gen.manual_seed(NOISE_SEED + ci)
        noises.append([torch.randn(1, 3, 64, 64, device=device, generator=gen) for _ in range(NUM_SAMPLES)])
    return noises


def to_img(t: torch.Tensor) -> np.ndarray:
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip((arr + 1) / 2, 0, 1)


def save_representative_figure(out_path: Path, tag: str, c: dict, y_hat: dict, mus: dict, mu_bar, var_map,
                                y_true, lcg_score: float, ens5: float, err_ensemble: float) -> None:
    fig, axes = plt.subplots(4, 6, figsize=(15, 10.5))
    for row, label in enumerate(("A", "B", "C")):
        for s in range(NUM_SAMPLES):
            axes[row, s].imshow(to_img(y_hat[label][s]))
            axes[row, s].set_title(f"{label}{s + 1}", fontsize=8)
            axes[row, s].axis("off")
        axes[row, 5].imshow(to_img(mus[label]))
        axes[row, 5].set_title(f"mu_{label}", fontsize=8, fontweight="bold")
        axes[row, 5].axis("off")
    axes[3, 0].imshow(to_img(y_true)); axes[3, 0].set_title("ground truth", fontsize=8); axes[3, 0].axis("off")
    axes[3, 1].imshow(to_img(mu_bar)); axes[3, 1].set_title("ensemble mean", fontsize=8); axes[3, 1].axis("off")
    im = axes[3, 2].imshow(var_map, cmap="inferno"); axes[3, 2].set_title("disagreement heatmap", fontsize=8); axes[3, 2].axis("off")
    fig.colorbar(im, ax=axes[3, 2], fraction=0.046)
    for j in range(3, 6):
        axes[3, j].axis("off")
    lcg_str = f"{lcg_score:.2f}" if np.isfinite(lcg_score) else "N/A"
    fig.suptitle(f"{tag}: {c['behavior']} ep{c['episode_id']} t{c['transition_index']}  "
                 f"LCG={lcg_str}  ENS_5={ens5:.5f}  ensemble_pred_err={err_ensemble:.5f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def process_domain_condition(domain: str, condition: str, candidates: list, noises: list, lcg_scores: dict,
                              out_dir: Path, probe_task: str = "walk") -> dict:
    print(f"\n{'=' * 70}\n{domain}/{condition}\n{'=' * 70}", flush=True)
    agents = {}
    for label in ("A", "B", "C"):
        ckpt = checkpoint_path(domain, condition, label)
        assert ckpt.exists(), f"missing checkpoint: {ckpt}"
        agent, _, _ = load_agent(domain, ckpt, probe_task=probe_task)
        agents[label] = agent

    device = agents["A"].denoiser.device
    with initialize_config_dir(version_base="1.3", config_dir=str(_THIS_DIR.parent.parent.parent / "config")):
        cfg = compose(config_name="trainer", overrides=["env=dm_control"])
    diffusion_cfg = instantiate(cfg.world_model_env.diffusion_sampler)
    assert diffusion_cfg.s_churn == 0.0, "full-ensemble run assumes production s_churn=0 (see pilot docstring)"
    sigmas = build_sigmas(diffusion_cfg.num_steps_denoising, diffusion_cfg.sigma_min,
                           diffusion_cfg.sigma_max, diffusion_cfg.rho, device)

    P = 3 * 64 * 64
    per_candidate = []
    t0_total = time.time()
    n_samples_done = 0
    with torch.no_grad():
        for ci, c in enumerate(candidates):
            y_hat = {label: [] for label in ("A", "B", "C")}
            for label, agent in agents.items():
                for s in range(NUM_SAMPLES):
                    y = sample_with_given_noise(agent.denoiser, sigmas, diffusion_cfg.order,
                                                 c["obs5d"], c["act"], noises[ci][s])
                    y_hat[label].append(y.float())
                    n_samples_done += 1

            cum_means = {}
            for label in ("A", "B", "C"):
                stack = torch.stack(y_hat[label], dim=0)
                cumsum = torch.cumsum(stack, dim=0)
                counts = torch.arange(1, NUM_SAMPLES + 1, device=device, dtype=torch.float32).view(-1, 1, 1, 1, 1)
                cum_means[label] = cumsum / counts

            ens_per_S = []
            for S in range(NUM_SAMPLES):
                mus_S = [cum_means[label][S] for label in ("A", "B", "C")]
                mu_bar_S = sum(mus_S) / 3
                ens = sum(((mu - mu_bar_S) ** 2).sum().item() for mu in mus_S) / (3 * P)
                ens_per_S.append(ens)

            within_var_k = {}
            error_k = {}
            for label in ("A", "B", "C"):
                mu5 = cum_means[label][NUM_SAMPLES - 1]
                per_sample_sq = [((y_hat[label][s] - mu5) ** 2).sum().item() / P for s in range(NUM_SAMPLES)]
                within_var_k[label] = float(np.mean(per_sample_sq))
                error_k[label] = float(((mu5 - c["y"]) ** 2).sum().item() / P)
            within_sample_variance = float(np.mean(list(within_var_k.values())))

            mu_bar_5 = sum(cum_means[label][NUM_SAMPLES - 1] for label in ("A", "B", "C")) / 3
            error_ensemble = float(((mu_bar_5 - c["y"]) ** 2).sum().item() / P)

            key = (c["behavior"], c["episode_id"], c["transition_index"])
            lcg_score = lcg_scores.get(key)

            per_candidate.append({
                "behavior": c["behavior"], "episode_id": c["episode_id"], "transition_index": c["transition_index"],
                "ens": ens_per_S, "within_var_by_model": within_var_k, "within_sample_variance": within_sample_variance,
                "error_by_model": error_k, "error_ensemble": error_ensemble, "lcg_score": lcg_score,
            })

            if (ci + 1) % 100 == 0:
                print(f"  [{domain}/{condition}] candidate {ci + 1}/1000", flush=True)

    n_missing_lcg = sum(1 for r in per_candidate if r["lcg_score"] is None)
    assert n_missing_lcg == 0, f"{n_missing_lcg} candidates had no matching LCG score (key mismatch)"

    elapsed = time.time() - t0_total
    ms_per_sample = elapsed / n_samples_done * 1000
    print(f"[{domain}/{condition}] done: {elapsed:.1f}s total, {ms_per_sample:.2f} ms/sample, n_samples={n_samples_done}")

    # --- representative candidates for visualization ---
    ens5 = np.array([r["ens"][-1] for r in per_candidate])
    lcg_arr = np.array([r["lcg_score"] for r in per_candidate])
    idx_highest = int(np.argmax(ens5))
    idx_lowest = int(np.argmin(ens5))
    idx_median = int(np.argmin(np.abs(ens5 - np.median(ens5))))
    reps = {"highest_ens": idx_highest, "median_ens": idx_median, "lowest_ens": idx_lowest}

    lcg_p90 = np.percentile(lcg_arr, 90)
    ens_p10 = np.percentile(ens5, 10)
    mask = (lcg_arr >= lcg_p90) & (ens5 <= ens_p10)
    if mask.any():
        masked_lcg = np.where(mask, lcg_arr, -np.inf)
        reps["high_lcg_low_ens"] = int(np.argmax(masked_lcg))
    else:
        print(f"[{domain}/{condition}] no high-LCG/low-ENS candidate found (p90 LCG={lcg_p90:.2f}, p10 ENS={ens_p10:.6f})")

    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for tag, ci in reps.items():
            c = candidates[ci]
            y_hat = {label: [] for label in ("A", "B", "C")}
            for label, agent in agents.items():
                for s in range(NUM_SAMPLES):
                    y_hat[label].append(sample_with_given_noise(agent.denoiser, sigmas, diffusion_cfg.order,
                                                                  c["obs5d"], c["act"], noises[ci][s]).float())
            mus = {label: torch.stack(y_hat[label]).mean(dim=0) for label in ("A", "B", "C")}
            mu_bar = sum(mus.values()) / 3
            var_map = (sum((mus[label] - mu_bar) ** 2 for label in ("A", "B", "C"))
                       .squeeze(0).mean(dim=0).cpu().numpy() / 3)
            save_representative_figure(
                out_dir / f"rep_{tag}.png", tag, c, y_hat, mus, mu_bar, var_map, c["y"],
                lcg_scores.get((c["behavior"], c["episode_id"], c["transition_index"]), float("nan")),
                per_candidate[ci]["ens"][-1], per_candidate[ci]["error_ensemble"],
            )
            print(f"Saved rep_{tag}.png ({domain}/{condition}, {c['behavior']} ep{c['episode_id']} t{c['transition_index']})")

    (out_dir / "ens_summary.json").write_text(json.dumps({
        "domain": domain, "condition": condition, "num_candidates": len(candidates),
        "runtime_seconds": elapsed, "ms_per_sample": ms_per_sample, "n_samples": n_samples_done,
        "representative_candidates": {
            tag: {"behavior": candidates[ci]["behavior"], "episode_id": candidates[ci]["episode_id"],
                  "transition_index": candidates[ci]["transition_index"]}
            for tag, ci in reps.items()
        },
        "per_candidate": per_candidate,
    }, indent=2, default=str))
    print(f"Saved: {out_dir / 'ens_summary.json'}")

    del agents
    torch.cuda.empty_cache()
    return {"elapsed_seconds": elapsed, "ms_per_sample": ms_per_sample, "n_samples": n_samples_done}


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    runtime_log = {}
    for domain in DOMAINS:
        print(f"\n{'#' * 70}\nDOMAIN: {domain}\n{'#' * 70}", flush=True)
        bootstrap_ckpt = checkpoint_path(domain, "run_scarce", "A")
        bootstrap_agent, _, _ = load_agent(domain, bootstrap_ckpt)
        n_cond = bootstrap_agent.denoiser.cfg.inner_model.num_steps_conditioning
        device = bootstrap_agent.denoiser.device
        del bootstrap_agent
        torch.cuda.empty_cache()

        candidates = build_domain_candidates(domain, n_cond, device)
        noises = build_domain_noise(len(candidates), device)
        print(f"[{domain}] built {len(candidates)} candidates (500 walk + 500 run), n_cond={n_cond}, "
              f"noise shared across models A/B/C and across {CONDITIONS}")

        for condition in CONDITIONS:
            lcg_scores = load_lcg_scores(domain, condition)
            out_dir = OUT_ROOT / domain / cond_dir(domain, condition)
            r = process_domain_condition(domain, condition, candidates, noises, lcg_scores, out_dir)
            runtime_log[f"{domain}/{condition}"] = r

        del candidates, noises
        torch.cuda.empty_cache()

    (OUT_ROOT / "runtime_log.json").write_text(json.dumps(runtime_log, indent=2))
    print("\nALL_6_ENSEMBLES_DONE")


if __name__ == "__main__":
    main()
