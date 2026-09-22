"""Renders full sample grids + disagreement heatmaps (reusing phase6_full_ensemble.py's
save_representative_figure) for the highest-ENS_5 WALK-behavior outliers under
walker/run_scarce -- the case where the ensemble's domain-fixed bias (walk always
scores higher than run in the walker domain, regardless of coverage condition)
runs directly counter to what's wanted (run is scarce here, so walk should NOT be
favored). Purpose: check whether these outliers are genuine epistemic uncertainty
(the model hasn't learned this content well, so disagreement is spatially
structured around the moving parts) or a boundary/data artifact.

Reuses the EXACT candidate ordering, shared noise (NOISE_SEED + candidate index),
and figure code from phase6_full_ensemble.py, so the images are bit-identical to
what actually produced the already-saved ens_S5 values -- no retraining, no
production-code changes, no new randomness.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase6e_outlier_inspection.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from phase5_lcg_scoring import load_agent  # noqa: E402
from phase6_full_ensemble import (  # noqa: E402
    checkpoint_path, build_domain_candidates, build_domain_noise, save_representative_figure, OUT_ROOT,
)
from phase6_pilot_ensemble import sample_with_given_noise  # noqa: E402
from models.diffusion.diffusion_sampler import build_sigmas  # noqa: E402

DOMAIN = "walker"
CONDITION = "run_scarce"
NUM_SAMPLES = 5
TOP_N = 5

_PROJECT_ROOT = _THIS_DIR.parent.parent.parent.parent
LCG_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "phase5_lcg_scoring"


def load_lcg_scores(domain: str, condition: str) -> dict:
    path = LCG_ROOT / domain / condition / "per_transition_scores.csv"
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["behavior"], int(row["episode_id"]), int(row["transition_index"]))
            out[key] = float(row["lcg_score"])
    return out


def main() -> None:
    csv_path = OUT_ROOT / DOMAIN / CONDITION / "per_transition_scores.csv"
    rows = list(csv.DictReader(open(csv_path, newline="")))
    walk_rows = [r for r in rows if r["behavior"] == "walk"]
    walk_rows_nonzero = [r for r in walk_rows if int(r["transition_index"]) != 0]

    walk_rows_sorted_desc = sorted(walk_rows_nonzero, key=lambda r: -float(r["ens_S5"]))
    outliers = walk_rows_sorted_desc[:TOP_N]

    walk_rows_sorted_asc = sorted(walk_rows, key=lambda r: float(r["ens_S5"]))
    median_row = walk_rows_sorted_asc[len(walk_rows_sorted_asc) // 2]

    targets = [(f"walk_outlier_nonzero_rank{i + 1}", r) for i, r in enumerate(outliers)]
    targets.append(("walk_median", median_row))

    print(f"Targets to render ({DOMAIN}/{CONDITION}):")
    for tag, r in targets:
        print(f"  {tag}: ep{r['episode_id']} t{r['transition_index']} ens_S5={float(r['ens_S5']):.6f} "
              f"lcg={float(r['lcg_score']):.2f}")

    bootstrap_ckpt = checkpoint_path(DOMAIN, CONDITION, "A")
    bootstrap_agent, _, _ = load_agent(DOMAIN, bootstrap_ckpt)
    n_cond = bootstrap_agent.denoiser.cfg.inner_model.num_steps_conditioning
    device = bootstrap_agent.denoiser.device
    del bootstrap_agent
    torch.cuda.empty_cache()

    candidates = build_domain_candidates(DOMAIN, n_cond, device)
    noises = build_domain_noise(len(candidates), device)
    index_by_key = {(c["behavior"], c["episode_id"], c["transition_index"]): ci for ci, c in enumerate(candidates)}

    lcg_scores = load_lcg_scores(DOMAIN, CONDITION)

    agents = {}
    for label in ("A", "B", "C"):
        ckpt = checkpoint_path(DOMAIN, CONDITION, label)
        agent, _, _ = load_agent(DOMAIN, ckpt)
        agents[label] = agent

    with initialize_config_dir(version_base="1.3", config_dir=str(_THIS_DIR.parent.parent.parent / "config")):
        cfg = compose(config_name="trainer", overrides=["env=dm_control"])
    diffusion_cfg = instantiate(cfg.world_model_env.diffusion_sampler)
    assert diffusion_cfg.s_churn == 0.0
    sigmas = build_sigmas(diffusion_cfg.num_steps_denoising, diffusion_cfg.sigma_min,
                           diffusion_cfg.sigma_max, diffusion_cfg.rho, device)

    out_dir = OUT_ROOT / DOMAIN / CONDITION / "outlier_inspection"
    out_dir.mkdir(parents=True, exist_ok=True)

    P = 3 * 64 * 64
    with torch.no_grad():
        for tag, r in targets:
            key = (r["behavior"], int(r["episode_id"]), int(r["transition_index"]))
            ci = index_by_key[key]
            c = candidates[ci]

            y_hat = {label: [] for label in ("A", "B", "C")}
            for label, agent in agents.items():
                for s in range(NUM_SAMPLES):
                    y_s = sample_with_given_noise(agent.denoiser, sigmas, diffusion_cfg.order,
                                                   c["obs5d"], c["act"], noises[ci][s])
                    y_hat[label].append(y_s.float())

            mus = {label: torch.stack(y_hat[label]).mean(dim=0) for label in ("A", "B", "C")}
            mu_bar = sum(mus.values()) / 3
            var_map = (sum((mus[label] - mu_bar) ** 2 for label in ("A", "B", "C"))
                       .squeeze(0).mean(dim=0).cpu().numpy() / 3)
            ens5 = sum(((mus[label] - mu_bar) ** 2).sum().item() for label in ("A", "B", "C")) / (3 * P)
            err_ensemble = float(((mu_bar - c["y"]) ** 2).sum().item() / P)

            fname = f"{tag}_ep{c['episode_id']}_t{c['transition_index']}.png"
            save_representative_figure(
                out_dir / fname, tag, c, y_hat, mus, mu_bar, var_map, c["y"],
                lcg_scores.get(key, float("nan")), ens5, err_ensemble,
            )
            print(f"Saved {fname}  (ens5_recomputed={ens5:.6f} vs csv={float(r['ens_S5']):.6f})")

    del agents
    torch.cuda.empty_cache()
    print(f"\nDone. Saved {len(targets)} figures to {out_dir}")


if __name__ == "__main__":
    main()
