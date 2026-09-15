"""Final multi-seed robustness analysis: combines Seed A (existing) with
Seeds B/C (newly trained) into the full domain x seed x condition table,
per-seed monotonic-ordering checks, cross-seed Delta statistics, per-seed
scarcity interactions, balanced normalized-position, and the 3-seeds-per-
domain figure. Uses ONLY already-saved summary.json files -- no rescoring.
Deliberately avoids importing torch (see phase5e's docstring for why).

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase5h_multiseed_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent.parent
OUT_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "phase5_lcg_scoring"
MULTISEED_SCORING_ROOT = OUT_ROOT / "multiseed"

DOMAINS = ["walker", "quadruped"]
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]
SEED_LABELS = {"A": "Seed A (0/42)", "B": "Seed B (43)", "C": "Seed C (44)"}


def load_seed_a(domain: str, condition: str) -> dict:
    return json.loads((OUT_ROOT / domain / condition / "summary.json").read_text())


def load_seed_bc(seed: int, domain: str, condition: str) -> dict:
    return json.loads((MULTISEED_SCORING_ROOT / f"seed{seed}" / domain / condition / "summary.json").read_text())


def main() -> None:
    summaries = {}
    for domain in DOMAINS:
        for condition in CONDITIONS:
            summaries[("A", domain, condition)] = load_seed_a(domain, condition)
            summaries[("B", domain, condition)] = load_seed_bc(43, domain, condition)
            summaries[("C", domain, condition)] = load_seed_bc(44, domain, condition)

    print("=== TABLE 1: domain | seed | Delta_run_scarce | Delta_balanced | Delta_walk_scarce | ordered? ===")
    deltas = {}
    ordered_flags = {}
    for domain in DOMAINS:
        for seed in ("A", "B", "C"):
            d = {c: summaries[(seed, domain, c)]["delta_mean"] for c in CONDITIONS}
            deltas[(domain, seed)] = d
            ordered = d["run_scarce"] > d["balanced"] > d["walk_scarce"]
            ordered_flags[(domain, seed)] = ordered
            print(f"{domain:<10} {seed:<3} run_scarce={d['run_scarce']:+8.4f}  balanced={d['balanced']:+8.4f}  "
                  f"walk_scarce={d['walk_scarce']:+8.4f}  ordered={ordered}")

    print("\n=== TABLE 2: domain | condition | mean_Delta_across_seeds | std | min | max ===")
    cross_seed_stats = {}
    for domain in DOMAINS:
        for condition in CONDITIONS:
            vals = np.array([deltas[(domain, s)][condition] for s in ("A", "B", "C")])
            stats = {"mean": float(vals.mean()), "std": float(vals.std()), "min": float(vals.min()), "max": float(vals.max()),
                     "values": vals.tolist()}
            cross_seed_stats[(domain, condition)] = stats
            print(f"{domain:<10} {condition:<12} mean={stats['mean']:+8.4f}  std={stats['std']:7.4f}  "
                  f"min={stats['min']:+8.4f}  max={stats['max']:+8.4f}")

    print("\n=== TABLE 3: domain | seed | scarcity_interaction (Delta_rs - Delta_ws) ===")
    interactions = {}
    for domain in DOMAINS:
        for seed in ("A", "B", "C"):
            interaction = deltas[(domain, seed)]["run_scarce"] - deltas[(domain, seed)]["walk_scarce"]
            interactions[(domain, seed)] = interaction
            print(f"{domain:<10} {seed:<3} interaction={interaction:+8.4f}")

    print("\n=== TABLE 4: domain | mean_interaction_across_seeds | std_interaction ===")
    interaction_stats = {}
    for domain in DOMAINS:
        vals = np.array([interactions[(domain, s)] for s in ("A", "B", "C")])
        stats = {"mean": float(vals.mean()), "std": float(vals.std()), "values": vals.tolist()}
        interaction_stats[domain] = stats
        print(f"{domain:<10} mean_interaction={stats['mean']:+8.4f}  std={stats['std']:7.4f}  values={stats['values']}")

    print("\n=== Balanced normalized position: (Delta_bal - Delta_ws) / (Delta_rs - Delta_ws) ===")
    norm_pos = {}
    for domain in DOMAINS:
        for seed in ("A", "B", "C"):
            d = deltas[(domain, seed)]
            denom = d["run_scarce"] - d["walk_scarce"]
            pos = (d["balanced"] - d["walk_scarce"]) / denom if denom != 0 else float("nan")
            norm_pos[(domain, seed)] = pos
            print(f"{domain:<10} {seed:<3} normalized_position={pos:.4f}")

    print("\n=== Secondary: held-out denoising loss trend across seeds ===")
    denoising = {}
    for domain in DOMAINS:
        for condition in CONDITIONS:
            for seed in ("A", "B", "C"):
                s = summaries[(seed, domain, condition)]
                denoising[(domain, condition, seed)] = {
                    "walk": s["walk_denoising_loss_mean"], "run": s["run_denoising_loss_mean"],
                }
        print(f"{domain}:")
        for condition in CONDITIONS:
            vals = [denoising[(domain, condition, s)] for s in ("A", "B", "C")]
            print(f"  {condition}: " + "  ".join(f"{s}(walk={v['walk']:.5f},run={v['run']:.5f})" for s, v in zip("ABC", vals)))

    # --- figure: one per domain, 3 thin seed lines + bold mean line ---
    colors = {"A": "#888888", "B": "#2E86AB", "C": "#C73E1D"}
    for domain in DOMAINS:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        x = [0, 1, 2]
        for seed in ("A", "B", "C"):
            y = [deltas[(domain, seed)][c] for c in CONDITIONS]
            ax.plot(x, y, marker="o", markersize=7, linewidth=1.6, alpha=0.8,
                    label=SEED_LABELS[seed], color=colors[seed])
        mean_y = [cross_seed_stats[(domain, c)]["mean"] for c in CONDITIONS]
        ax.plot(x, mean_y, marker="s", markersize=10, linewidth=3.2, color="black", label="Mean across seeds", zorder=5)
        for xi, yi in zip(x, mean_y):
            ax.annotate(f"{yi:+.2f}", (xi, yi), textcoords="offset points", xytext=(8, 8), fontsize=9, fontweight="bold")
        ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
        ax.set_xticks(x); ax.set_xticklabels(["run_scarce", "balanced", "walk_scarce"])
        ax.set_xlim(-0.3, 2.3)
        ax.set_ylabel(r"$\Delta$ = mean LCG(run) $-$ mean LCG(walk)")
        ax.set_title(f"{domain}: LCG coverage response across 3 training seeds")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        fig.tight_layout()
        out_fig = MULTISEED_SCORING_ROOT / f"{domain}_multiseed_trend.png"
        fig.savefig(out_fig, dpi=150)
        print(f"\nSaved figure: {out_fig}")

    final = {
        "deltas": {f"{d}/{s}": v for (d, s), v in deltas.items()},
        "ordered_flags": {f"{d}/{s}": v for (d, s), v in ordered_flags.items()},
        "cross_seed_stats": {f"{d}/{c}": v for (d, c), v in cross_seed_stats.items()},
        "interactions": {f"{d}/{s}": v for (d, s), v in interactions.items()},
        "interaction_stats": interaction_stats,
        "normalized_position": {f"{d}/{s}": v for (d, s), v in norm_pos.items()},
    }
    out_json = MULTISEED_SCORING_ROOT / "multiseed_final_summary.json"
    out_json.write_text(json.dumps(final, indent=2, default=str))
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
