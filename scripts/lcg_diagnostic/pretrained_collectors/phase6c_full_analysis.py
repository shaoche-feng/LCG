"""Final analysis for the full six-ensemble disagreement diagnostic: loads the
six ens_summary.json files saved by phase6_full_ensemble.py (each candidate
row already carries its joined LCG score) and computes every correlation,
overlap, stability, and episode-level statistic against LCG. Deliberately
torch-free (same OMP-conflict reason as phase5e/phase5g/phase6b) -- does NOT
import phase5_lcg_scoring.py (which pulls in torch transitively); the tiny
numpy-only helpers it would have reused are reimplemented locally instead.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase6c_full_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent.parent
ENS_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "phase6_full_ensemble"
LCG_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "phase5_lcg_scoring"

DOMAINS = ["walker", "quadruped"]
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]
NUM_SAMPLES = 5


def describe(x: np.ndarray) -> dict:
    return {
        "n": int(len(x)), "mean": float(np.mean(x)), "std": float(np.std(x)), "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)), "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)), "p90": float(np.percentile(x, 90)),
        "min": float(np.min(x)), "max": float(np.max(x)),
    }


def episode_level_means(scores: np.ndarray, episode_ids: np.ndarray) -> dict:
    out = {}
    for eid in sorted(set(episode_ids.tolist())):
        mask = episode_ids == eid
        out[int(eid)] = {"mean": float(scores[mask].mean()), "n": int(mask.sum())}
    return out


def bootstrap_ci(walk_ep_means: np.ndarray, run_ep_means: np.ndarray, num_resamples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_w, n_r = len(walk_ep_means), len(run_ep_means)
    diffs = np.empty(num_resamples)
    for i in range(num_resamples):
        w = rng.choice(walk_ep_means, size=n_w, replace=True)
        r = rng.choice(run_ep_means, size=n_r, replace=True)
        diffs[i] = r.mean() - w.mean()
    return {
        "point_estimate": float(run_ep_means.mean() - walk_ep_means.mean()),
        "ci_2.5": float(np.percentile(diffs, 2.5)), "ci_97.5": float(np.percentile(diffs, 97.5)),
        "num_resamples": num_resamples, "seed": seed,
    }


def topk_overlap(a: np.ndarray, b: np.ndarray, frac: float) -> float:
    k = int(round(len(a) * frac))
    top_a = set(np.argsort(a)[-k:].tolist())
    top_b = set(np.argsort(b)[-k:].tolist())
    return len(top_a & top_b) / k


BOOTSTRAP_SEED = 999
NUM_BOOTSTRAP = 10_000


def analyze_domain_condition(domain: str, condition: str) -> dict:
    data = json.loads((ENS_ROOT / domain / condition / "ens_summary.json").read_text())
    rows = data["per_candidate"]
    behavior = np.array([r["behavior"] for r in rows])
    episode_id = np.array([r["episode_id"] for r in rows])
    ens_matrix = np.array([r["ens"] for r in rows])  # (1000, 5)
    ens5 = ens_matrix[:, -1]
    lcg = np.array([r["lcg_score"] for r in rows])
    within_var = np.array([r["within_sample_variance"] for r in rows])
    err_ensemble = np.array([r["error_ensemble"] for r in rows])

    walk_mask, run_mask = behavior == "walk", behavior == "run"

    # --- Table A: ENS_5 descriptive stats by behavior ---
    table_a = {
        "walk": describe(ens5[walk_mask]),
        "run": describe(ens5[run_mask]),
    }
    delta_ens = table_a["run"]["mean"] - table_a["walk"]["mean"]
    delta_lcg = float(lcg[run_mask].mean() - lcg[walk_mask].mean())

    # --- Table D: correlation / overlap between LCG and ENS_5 ---
    pear = float(pearsonr(lcg, ens5)[0])
    spear = float(spearmanr(lcg, ens5)[0])
    top10 = topk_overlap(lcg, ens5, 0.10)
    top20 = topk_overlap(lcg, ens5, 0.20)
    tau = float(kendalltau(lcg, ens5)[0])
    frac_reversed = (1 - tau) / 2
    table_d = {"pearson": pear, "spearman": spear, "top10_overlap": top10, "top20_overlap": top20,
               "kendall_tau": tau, "fraction_pair_order_reversals": frac_reversed}

    # --- Table E: prediction error / within-sample variance by behavior ---
    table_e = {
        "walk": {"ensemble_prediction_error": float(err_ensemble[walk_mask].mean()),
                 "within_sample_variance": float(within_var[walk_mask].mean())},
        "run": {"ensemble_prediction_error": float(err_ensemble[run_mask].mean()),
                "within_sample_variance": float(within_var[run_mask].mean())},
    }
    mean_within = float(within_var.mean())
    mean_between_s5 = float(ens5.mean())
    ratio_between_within = mean_between_s5 / mean_within

    # --- Table F: S=1..4 stability vs S=5 ---
    table_f = {}
    for S in range(1, NUM_SAMPLES):
        a, b = ens_matrix[:, S - 1], ens_matrix[:, NUM_SAMPLES - 1]
        table_f[S] = {
            "spearman_vs_S5": float(spearmanr(a, b)[0]), "pearson_vs_S5": float(pearsonr(a, b)[0]),
            "mean_abs_relative_diff": float(np.mean(np.abs(a - b) / np.maximum(np.abs(b), 1e-8))),
            "top10_overlap_vs_S5": topk_overlap(a, b, 0.10), "top20_overlap_vs_S5": topk_overlap(a, b, 0.20),
        }

    # --- Section 11: episode-level analysis ---
    walk_ep = episode_level_means(ens5[walk_mask], episode_id[walk_mask])
    run_ep = episode_level_means(ens5[run_mask], episode_id[run_mask])
    walk_ep_means = np.array([v["mean"] for v in walk_ep.values()])
    run_ep_means = np.array([v["mean"] for v in run_ep.values()])
    boot = bootstrap_ci(walk_ep_means, run_ep_means, NUM_BOOTSTRAP, BOOTSTRAP_SEED)

    # --- prediction-error secondary check on the "low ENS but is it genuinely easy" question ---
    err_by_behavior_model = {}
    for label in ("A", "B", "C"):
        err_by_behavior_model[label] = {
            "walk": float(np.mean([r["error_by_model"][label] for r in rows if r["behavior"] == "walk"])),
            "run": float(np.mean([r["error_by_model"][label] for r in rows if r["behavior"] == "run"])),
        }

    return {
        "domain": domain, "condition": condition, "n_candidates": len(rows),
        "table_a_ens5_by_behavior": table_a, "delta_ens5": delta_ens, "delta_lcg": delta_lcg,
        "table_d_lcg_vs_ens": table_d,
        "table_e_error_and_within_var": table_e,
        "mean_within_sample_variance": mean_within, "mean_between_model_disagreement_S5": mean_between_s5,
        "ratio_between_within": ratio_between_within,
        "table_f_stability_vs_S5": table_f,
        "episode_level": {"walk_episode_means": walk_ep, "run_episode_means": run_ep, "bootstrap": boot},
        "error_by_model_and_behavior": err_by_behavior_model,
        "runtime": {"ms_per_sample": data["ms_per_sample"], "runtime_seconds": data["runtime_seconds"]},
    }


def main() -> None:
    results = {}
    for domain in DOMAINS:
        for condition in CONDITIONS:
            print(f"analyzing {domain}/{condition} ...", flush=True)
            results[f"{domain}/{condition}"] = analyze_domain_condition(domain, condition)

    print("\n" + "=" * 100)
    print("TABLE A: domain | condition | behavior | mean_ENS5 | median | std | p10 | p90")
    print("=" * 100)
    for key, r in results.items():
        for beh in ("walk", "run"):
            d = r["table_a_ens5_by_behavior"][beh]
            print(f"{key:<28} {beh:<5} mean={d['mean']:.6f} median={d['median']:.6f} std={d['std']:.6f} "
                  f"p10={d['p10']:.6f} p90={d['p90']:.6f}")

    print("\n" + "=" * 100)
    print("TABLE B: domain | condition | Delta_LCG | Delta_ENS5")
    print("=" * 100)
    for key, r in results.items():
        print(f"{key:<28} Delta_LCG={r['delta_lcg']:+10.4f}   Delta_ENS5={r['delta_ens5']:+10.6f}")

    print("\n" + "=" * 100)
    print("TABLE C: domain | method | Delta_RS | Delta_BAL | Delta_WS | monotonic?")
    print("=" * 100)
    table_c = {}
    for domain in DOMAINS:
        for method, key in (("LCG", "delta_lcg"), ("ENS5", "delta_ens5")):
            vals = {c: results[f"{domain}/{c}"][key] for c in CONDITIONS}
            ordered = vals["run_scarce"] > vals["balanced"] > vals["walk_scarce"]
            table_c[f"{domain}/{method}"] = {**vals, "monotonic": ordered}
            print(f"{domain:<10} {method:<5} RS={vals['run_scarce']:+10.4f}  BAL={vals['balanced']:+10.4f}  "
                  f"WS={vals['walk_scarce']:+10.4f}  monotonic={ordered}")

    print("\n" + "=" * 100)
    print("TABLE D: domain | condition | Pearson_LCG_ENS | Spearman | top10 | top20 | frac_pair_reversals")
    print("=" * 100)
    for key, r in results.items():
        d = r["table_d_lcg_vs_ens"]
        print(f"{key:<28} pearson={d['pearson']:+.4f}  spearman={d['spearman']:+.4f}  "
              f"top10={d['top10_overlap']:.4f}  top20={d['top20_overlap']:.4f}  "
              f"frac_pair_rev={d['fraction_pair_order_reversals']:.4f}")

    print("\n" + "=" * 100)
    print("TABLE E: domain | condition | behavior | ensemble_pred_error | within_sample_var")
    print("=" * 100)
    for key, r in results.items():
        for beh in ("walk", "run"):
            e = r["table_e_error_and_within_var"][beh]
            print(f"{key:<28} {beh:<5} ensemble_pred_error={e['ensemble_prediction_error']:.6f}  "
                  f"within_sample_var={e['within_sample_variance']:.6f}")
        print(f"{key:<28} mean_within={r['mean_within_sample_variance']:.6f}  "
              f"mean_between_S5={r['mean_between_model_disagreement_S5']:.6f}  "
              f"ratio={r['ratio_between_within']:.4f}")

    print("\n" + "=" * 100)
    print("TABLE F: domain | condition | S | Spearman_vs_S5 | mean_abs_rel_diff | top10 | top20")
    print("=" * 100)
    for key, r in results.items():
        for S, d in r["table_f_stability_vs_S5"].items():
            print(f"{key:<28} S={S}  spearman={d['spearman_vs_S5']:.4f}  rel_diff={d['mean_abs_relative_diff']:.4f}  "
                  f"top10={d['top10_overlap_vs_S5']:.4f}  top20={d['top20_overlap_vs_S5']:.4f}")

    print("\n" + "=" * 100)
    print("EPISODE-LEVEL: domain | condition | Delta_ENS5 (episode-level) | bootstrap 95% CI")
    print("=" * 100)
    for key, r in results.items():
        b = r["episode_level"]["bootstrap"]
        print(f"{key:<28} point={b['point_estimate']:+.6f}  CI=[{b['ci_2.5']:+.6f}, {b['ci_97.5']:+.6f}]")

    print("\n" + "=" * 100)
    print("SECONDARY: per-model prediction error by behavior (common-mode bias check)")
    print("=" * 100)
    for key, r in results.items():
        print(f"{key}:")
        for label, e in r["error_by_model_and_behavior"].items():
            print(f"  model {label}: walk_err={e['walk']:.6f}  run_err={e['run']:.6f}")

    # === figure: Delta_LCG vs Delta_ENS5 across conditions, per domain ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(3)
    width = 0.35
    for ax, domain in zip(axes, DOMAINS):
        lcg_vals = [results[f"{domain}/{c}"]["delta_lcg"] for c in CONDITIONS]
        ens_vals = [results[f"{domain}/{c}"]["delta_ens5"] for c in CONDITIONS]
        ax2 = ax.twinx()
        ax.bar(x - width / 2, lcg_vals, width, color="#C4622D", label="Delta_LCG")
        ax2.bar(x + width / 2, ens_vals, width, color="#2B6CB0", label="Delta_ENS5")
        ax.axhline(0, color="#999999", linewidth=1)
        ax2.axhline(0, color="#999999", linewidth=0.6, linestyle=":")
        ax.set_xticks(x); ax.set_xticklabels(CONDITIONS)
        ax.set_title(domain)
        ax.set_ylabel("Delta_LCG (mean run - mean walk)", color="#C4622D")
        ax2.set_ylabel("Delta_ENS5 (mean run - mean walk)", color="#2B6CB0")
    fig.suptitle("LCG vs. ensemble-disagreement coverage response, by domain/condition", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(ENS_ROOT / "lcg_vs_ens_delta_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved figure: {ENS_ROOT / 'lcg_vs_ens_delta_comparison.png'}")

    out = {"per_domain_condition": results, "table_c_monotonicity": table_c}
    (ENS_ROOT / "full_analysis_summary.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {ENS_ROOT / 'full_analysis_summary.json'}")


if __name__ == "__main__":
    main()
