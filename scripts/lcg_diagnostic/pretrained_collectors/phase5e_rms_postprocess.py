"""Finishes the RMS diagnostic (Tables A-D, probe-CV diagnostic, figure,
final JSON) using ONLY the already-saved q_values.npy/metadata.json from
phase5d_rms_diagnostic.py -- no GPU, no torch, no model loading, no
rescoring. Deliberately avoids importing torch in this process: the
original run crashed at Table D (scipy pearsonr/spearmanr) with an
`OMP: Error #15` DLL conflict between torch's and scipy's bundled OpenMP
runtimes. Since everything needed from here on is plain numpy arrays
already on disk, running this as a torch-free process sidesteps the
conflict entirely rather than papering over it with KMP_DUPLICATE_LIB_OK.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase5e_rms_postprocess.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent.parent
OUT_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "phase5_lcg_scoring"
RMS_OUT_ROOT = OUT_ROOT / "rms_diagnostic"

DOMAINS = ["walker", "quadruped"]
CONDITIONS = ["run_scarce", "walk_scarce"]


def paired_bootstrap_interaction(d_run: np.ndarray, d_walk: np.ndarray, num_resamples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_run, n_walk = len(d_run), len(d_walk)
    draws = np.empty(num_resamples)
    for b in range(num_resamples):
        run_idx = rng.integers(0, n_run, size=n_run)
        walk_idx = rng.integers(0, n_walk, size=n_walk)
        draws[b] = d_run[run_idx].mean() - d_walk[walk_idx].mean()
    point = float(d_run.mean() - d_walk.mean())
    return {"point_estimate": point, "ci_2.5": float(np.percentile(draws, 2.5)),
            "ci_97.5": float(np.percentile(draws, 97.5)), "num_resamples": num_resamples, "seed": seed}


def exact_paired_sign_flip_test(d_run: np.ndarray, d_walk: np.ndarray) -> dict:
    import itertools
    n_run, n_walk = len(d_run), len(d_walk)
    observed = d_run.mean() - d_walk.mean()
    perm_stats = []
    for s_run in itertools.product([1, -1], repeat=n_run):
        fr = (np.array(s_run) * d_run).mean()
        for s_walk in itertools.product([1, -1], repeat=n_walk):
            fw = (np.array(s_walk) * d_walk).mean()
            perm_stats.append(fr - fw)
    perm_stats = np.array(perm_stats)
    return {"observed": float(observed), "num_sign_patterns": len(perm_stats),
            "p_value_one_sided_interaction_gt_0": float((perm_stats >= observed).mean()),
            "p_value_two_sided": float((np.abs(perm_stats) >= abs(observed)).mean())}


def load_result(domain: str, condition: str) -> dict:
    d = RMS_OUT_ROOT / domain / condition
    q_values = np.load(d / "q_values.npy")
    meta = json.loads((d / "metadata.json").read_text())
    behaviors = np.array([m["behavior"] for m in meta["candidate_order"]])
    episode_ids = np.array([m["episode_id"] for m in meta["candidate_order"]])

    lcg_mean = q_values.mean(axis=1)
    lcg_rms = np.sqrt(np.mean(q_values ** 2, axis=1))
    probe_variance = np.mean((q_values - lcg_mean[:, None]) ** 2, axis=1)
    probe_std = np.sqrt(probe_variance)
    probe_cv = np.where(lcg_mean > 1e-8, probe_std / np.maximum(lcg_mean, 1e-8), np.nan)
    identity_discrepancy = np.abs(lcg_rms ** 2 - (lcg_mean ** 2 + probe_variance))

    return {
        "domain": domain, "condition": condition, "meta": meta, "q_values": q_values,
        "lcg_mean": lcg_mean, "lcg_rms": lcg_rms, "probe_std": probe_std, "probe_cv": probe_cv,
        "behaviors": behaviors, "episode_ids": episode_ids,
        "identity_max_discrepancy": float(identity_discrepancy.max()),
    }


def summarize_behavior(r: dict, behavior: str) -> dict:
    mask = r["behaviors"] == behavior
    return {
        "mean_LCG_mean": float(r["lcg_mean"][mask].mean()),
        "mean_LCG_RMS": float(r["lcg_rms"][mask].mean()),
        "median_LCG_RMS": float(np.median(r["lcg_rms"][mask])),
        "mean_probe_std": float(r["probe_std"][mask].mean()),
        "mean_probe_CV": float(np.nanmean(r["probe_cv"][mask])),
        "mean_RMS_over_mean_ratio": float(np.mean(r["lcg_rms"][mask] / r["lcg_mean"][mask])),
    }


def episode_means_by_behavior(r: dict, behavior: str, metric: str) -> dict:
    mask = r["behaviors"] == behavior
    vals, eids = r[metric][mask], r["episode_ids"][mask]
    return {int(eid): float(vals[eids == eid].mean()) for eid in sorted(set(eids.tolist()))}


def ranking_comparison(r: dict) -> dict:
    mean_, rms_ = r["lcg_mean"], r["lcg_rms"]
    pear = float(pearsonr(mean_, rms_)[0])
    spear = float(spearmanr(mean_, rms_)[0])
    n = len(mean_)
    k10, k20 = max(1, n // 10), max(1, n // 5)
    top_mean_10 = set(np.argsort(-mean_)[:k10].tolist())
    top_rms_10 = set(np.argsort(-rms_)[:k10].tolist())
    top_mean_20 = set(np.argsort(-mean_)[:k20].tolist())
    top_rms_20 = set(np.argsort(-rms_)[:k20].tolist())
    overlap10 = len(top_mean_10 & top_rms_10) / k10
    overlap20 = len(top_mean_20 & top_rms_20) / k20

    rng = np.random.default_rng(0)
    n_pairs = 20000
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    valid = i != j
    i, j = i[valid], j[valid]
    sign_mean = np.sign(mean_[i] - mean_[j])
    sign_rms = np.sign(rms_[i] - rms_[j])
    frac_order_changed = float(np.mean(sign_mean != sign_rms))

    return {"pearson": pear, "spearman": spear, "top10_overlap": overlap10, "top20_overlap": overlap20,
            "fraction_pairs_order_changed_sampled": frac_order_changed, "n_pairs_sampled": int(valid.sum())}


def main() -> None:
    raw = {(d, c): load_result(d, c) for d in DOMAINS for c in CONDITIONS}

    print("=== Identity check RMS^2 = mean^2 + variance (max abs discrepancy) ===")
    for (d, c), r in raw.items():
        print(f"{d}/{c}: {r['identity_max_discrepancy']:.6e}")

    print("\n=== TABLE A: mean_LCG / mean_RMS / probe stats per domain/condition/behavior ===")
    table_a = {}
    for (d, c), r in raw.items():
        for behavior in ("walk", "run"):
            key = f"{d}/{c}/{behavior}"
            table_a[key] = summarize_behavior(r, behavior)
            s = table_a[key]
            print(f"{key:<27} mean_LCG={s['mean_LCG_mean']:>9.4f} mean_RMS={s['mean_LCG_RMS']:>9.4f} "
                  f"median_RMS={s['median_LCG_RMS']:>9.4f} probe_std={s['mean_probe_std']:>8.4f} "
                  f"probe_CV={s['mean_probe_CV']:>7.4f} RMS/mean={s['mean_RMS_over_mean_ratio']:>6.4f}")

    print("\n=== TABLE B: Delta_mean vs Delta_RMS ===")
    table_b = {}
    for (d, c), r in raw.items():
        dm = table_a[f"{d}/{c}/run"]["mean_LCG_mean"] - table_a[f"{d}/{c}/walk"]["mean_LCG_mean"]
        drms = table_a[f"{d}/{c}/run"]["mean_LCG_RMS"] - table_a[f"{d}/{c}/walk"]["mean_LCG_RMS"]
        same_sign = (dm > 0) == (drms > 0)
        table_b[f"{d}/{c}"] = {"delta_mean": dm, "delta_rms": drms, "same_sign": same_sign}
        print(f"{d:<10} {c:<12} Delta_mean={dm:>+9.4f} Delta_RMS={drms:>+9.4f} same_sign={same_sign}")

    print("\n=== TABLE C: scarcity interaction (mean vs RMS), paired episode-level bootstrap ===")
    table_c = {}
    for d in DOMAINS:
        rs, ws = raw[(d, "run_scarce")], raw[(d, "walk_scarce")]

        def interaction_for_metric(metric: str) -> dict:
            rs_walk = episode_means_by_behavior(rs, "walk", metric)
            rs_run = episode_means_by_behavior(rs, "run", metric)
            ws_walk = episode_means_by_behavior(ws, "walk", metric)
            ws_run = episode_means_by_behavior(ws, "run", metric)
            run_ids = sorted(rs_run.keys())
            walk_ids = sorted(rs_walk.keys())
            d_run = np.array([rs_run[i] - ws_run[i] for i in run_ids])
            d_walk = np.array([rs_walk[i] - ws_walk[i] for i in walk_ids])
            boot = paired_bootstrap_interaction(d_run, d_walk, 10_000, 999)
            perm = exact_paired_sign_flip_test(d_run, d_walk)
            return {"interaction": boot["point_estimate"], "ci": [boot["ci_2.5"], boot["ci_97.5"]],
                    "p_one_sided": perm["p_value_one_sided_interaction_gt_0"], "p_two_sided": perm["p_value_two_sided"]}

        int_mean = interaction_for_metric("lcg_mean")
        int_rms = interaction_for_metric("lcg_rms")
        table_c[d] = {"interaction_mean": int_mean, "interaction_rms": int_rms}
        print(f"{d}: Interaction_mean={int_mean['interaction']:+.4f} CI={int_mean['ci']} p1s={int_mean['p_one_sided']:.4f} | "
              f"Interaction_RMS={int_rms['interaction']:+.4f} CI={int_rms['ci']} p1s={int_rms['p_one_sided']:.4f}")

    print("\n=== TABLE D: ranking comparison (mean vs RMS) ===")
    table_d = {}
    for (d, c), r in raw.items():
        rc = ranking_comparison(r)
        table_d[f"{d}/{c}"] = rc
        print(f"{d:<10} {c:<12} Pearson={rc['pearson']:.6f} Spearman={rc['spearman']:.6f} "
              f"top10_overlap={rc['top10_overlap']:.3f} top20_overlap={rc['top20_overlap']:.3f} "
              f"frac_pairs_flipped={rc['fraction_pairs_order_changed_sampled']:.4f}")

    print("\n=== Probe variability (CV) diagnostic: scarce vs abundant ===")
    cv_diag = {}
    for d in DOMAINS:
        cv_run_scarce = {"scarce(run)_CV": table_a[f"{d}/run_scarce/run"]["mean_probe_CV"],
                          "abundant(walk)_CV": table_a[f"{d}/run_scarce/walk"]["mean_probe_CV"]}
        cv_walk_scarce = {"scarce(walk)_CV": table_a[f"{d}/walk_scarce/walk"]["mean_probe_CV"],
                           "abundant(run)_CV": table_a[f"{d}/walk_scarce/run"]["mean_probe_CV"]}
        cv_diag[d] = {"run_scarce": cv_run_scarce, "walk_scarce": cv_walk_scarce}
        print(f"{d} run_scarce: {cv_run_scarce}")
        print(f"{d} walk_scarce: {cv_walk_scarce}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    colors = {"walker": "#2E86AB", "quadruped": "#C73E1D"}
    for ax, metric_key, title in [(axes[0], "delta_mean", "Original mean/trace LCG"), (axes[1], "delta_rms", "RMS LCG")]:
        for d in DOMAINS:
            y = [table_b[f"{d}/run_scarce"][metric_key], table_b[f"{d}/walk_scarce"][metric_key]]
            ax.plot([0, 1], y, marker="o", markersize=9, linewidth=2.5, label=d, color=colors[d])
            for xi, yi in zip([0, 1], y):
                ax.annotate(f"{yi:+.2f}", (xi, yi), textcoords="offset points", xytext=(8, 6), fontsize=9)
        ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["run_scarce", "walk_scarce"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylabel(r"$\Delta$ = score(run) $-$ score(walk)")
        ax.set_title(title)
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RMS_OUT_ROOT / "mean_vs_rms_sign_reversal.png", dpi=150)
    print(f"\nSaved figure: {RMS_OUT_ROOT / 'mean_vs_rms_sign_reversal.png'}")

    final = {
        "identity_check_max_discrepancy": {f"{d}/{c}": r["identity_max_discrepancy"] for (d, c), r in raw.items()},
        "table_a": table_a, "table_b": table_b,
        "table_c": {d: {"interaction_mean": v["interaction_mean"], "interaction_rms": v["interaction_rms"]}
                    for d, v in table_c.items()},
        "table_d": table_d, "probe_cv_diagnostic": cv_diag,
    }
    (RMS_OUT_ROOT / "rms_diagnostic_summary.json").write_text(json.dumps(final, indent=2, default=str))
    print(f"\nSaved: {RMS_OUT_ROOT / 'rms_diagnostic_summary.json'}")


if __name__ == "__main__":
    main()
