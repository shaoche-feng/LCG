"""Post-hoc RMS diagnostic: recovers the per-probe q_m(x) quantities that
production `score_one_jvp_bank` (src/lcg/forward_jvp.py, UNMODIFIED) computes
internally but discards after folding into the running mean, by re-running
the exact same JVP scoring with a local diagnostic-only wrapper that stores
q_m instead of only accumulating mean_m(q_m).

Mirrors score_one_jvp_bank as closely as possible: only `jvp_through_F` and
`unflatten_to_dict` (both unmodified production functions) are reused; the
only intentional behavioral difference is storage vs. discard of q_m.

Reproduces the EXACT original Phase 5 experimental state (same checkpoints,
h_D seed, bank seed, theta_S, held-out episodes/indices/ordering) by
importing setup code directly from phase5_lcg_scoring.py rather than
reimplementing it, to eliminate any chance of drift.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase5d_rms_diagnostic.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

_THIS_DIR = Path(__file__).resolve().parent
_LCG_ROOT = _THIS_DIR.parent.parent.parent
_PROJECT_ROOT = _LCG_ROOT.parent
sys.path.insert(0, str(_LCG_ROOT / "src"))
sys.path.insert(0, str(_THIS_DIR))

from lcg.forward_jvp import jvp_through_F, unflatten_to_dict  # noqa: E402 -- unmodified production functions
from lcg.theta_s import selected_named_parameters, frozen_named_parameters  # noqa: E402
from lcg.precision import historical_precision, assert_setup_valid  # noqa: E402
from lcg.forward_jvp import make_jvp_bank  # noqa: E402
from models.diffusion.denoiser import apply_noise_from_samples  # noqa: E402
from data import Dataset  # noqa: E402

from phase5_lcg_scoring import (  # noqa: E402 -- reuse Phase 5's exact setup code, no reimplementation
    load_agent, evenly_spaced_indices, build_candidates, describe, episode_level_means,
    PRECISION_REFERENCE_SIZE, PRECISION_NUM_MC, BETA, DAMPING, CANDIDATE_NUM_MC, CANDIDATE_CHUNK_SIZE,
    THETA_S, PRECISION_SEED, BANK_SEED, HELD_OUT_EPISODE_IDS, N_TRANS_PER_EPISODE, EPISODE_LEN,
    MODELS_ROOT, MIXTURES, SOURCE_POOLS, OUT_ROOT,
)
from phase5c_interaction_analysis import paired_bootstrap_interaction, exact_paired_sign_flip_test  # noqa: E402

DOMAINS = ["walker", "quadruped"]
CONDITIONS = ["run_scarce", "walk_scarce"]
RMS_OUT_ROOT = OUT_ROOT / "rms_diagnostic"
REPRO_RTOL = 1e-4
REPRO_ATOL = 1e-3


def score_one_jvp_bank_with_probes(denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, bank, candidates, chunk_size):
    """Mirrors lcg.forward_jvp.score_one_jvp_bank EXACTLY, except storing every
    per-probe q_m(x) = 2*||J_F H_D^{-1/2} eta_m||^2 instead of only its running
    mean. Returns q_values: (num_candidates, M)."""
    device = h_D_inv_sqrt.device
    d_S = sum(p.numel() for p in theta_s_named.values())
    assert h_D_inv_sqrt.numel() == d_S

    num_entries = bank.num_samples
    num_candidates = len(candidates)
    q_values = torch.zeros(num_candidates, num_entries, device=device)

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
        tangent_named = unflatten_to_dict(z_flat, theta_s_named)

        for start, obs_batch, act_batch, y_batch, B in chunks:
            y_sigma_batch = apply_noise_from_samples(y_batch, sigma, eps, eps_offset, denoiser.cfg.sigma_offset_noise).detach()
            _, jvp_out = jvp_through_F(
                denoiser, theta_s_named, frozen_named, tangent_named, y_sigma_batch, sigma, obs_batch, act_batch
            )
            contribution = 2.0 * jvp_out.reshape(B, -1).square().sum(dim=1)
            q_values[start : start + B, m] = contribution  # STORE (production discards this after +=)

    return q_values.detach()


def rescore_domain_condition(domain: str, condition: str) -> dict:
    print(f"\n{'=' * 70}\n{domain}/{condition}\n{'=' * 70}")
    checkpoint_path = MODELS_ROOT / domain / condition / "checkpoints" / "agent_versions" / "agent_epoch_00001.pt"
    agent, sigma_cfg, action_dim = load_agent(domain, checkpoint_path)
    denoiser = agent.denoiser
    device = denoiser.device
    num_steps_conditioning = denoiser.cfg.inner_model.num_steps_conditioning

    theta_s_named = selected_named_parameters(denoiser, THETA_S)
    frozen_named = frozen_named_parameters(denoiser, theta_s_named)
    d_S = sum(p.numel() for p in theta_s_named.values())

    train_dataset = Dataset(MIXTURES / domain / condition / "dataset", name=f"{domain}_{condition}_train", cache_in_ram=True)
    train_dataset.load_from_default_path()
    h_D = historical_precision(
        denoiser, list(theta_s_named.values()), train_dataset, sigma_cfg,
        B=PRECISION_REFERENCE_SIZE, N=train_dataset.num_steps, num_mc=PRECISION_NUM_MC,
        beta=BETA, damping=DAMPING, seed=PRECISION_SEED,
    )
    assert_setup_valid(theta_s_named, h_D, d_S=d_S)
    h_D_inv_sqrt = h_D.rsqrt()

    y_shape = torch.Size([1, denoiser.cfg.inner_model.img_channels, 64, 64])
    bank = make_jvp_bank(sigma_cfg, y_shape, d_S, device, num_samples=CANDIDATE_NUM_MC, seed=BANK_SEED)

    indices = evenly_spaced_indices(EPISODE_LEN, N_TRANS_PER_EPISODE)
    walk_pool = Dataset(SOURCE_POOLS / domain / "walk" / "dataset", name=f"{domain}_walk_pool", cache_in_ram=True)
    walk_pool.load_from_default_path()
    run_pool = Dataset(SOURCE_POOLS / domain / "run" / "dataset", name=f"{domain}_run_pool", cache_in_ram=True)
    run_pool.load_from_default_path()
    walk_candidates, walk_meta = build_candidates(walk_pool, HELD_OUT_EPISODE_IDS, indices, num_steps_conditioning, device)
    run_candidates, run_meta = build_candidates(run_pool, HELD_OUT_EPISODE_IDS, indices, num_steps_conditioning, device)

    all_candidates = walk_candidates + run_candidates
    all_meta = [{"behavior": "walk", **m} for m in walk_meta] + [{"behavior": "run", **m} for m in run_meta]

    q_values = score_one_jvp_bank_with_probes(
        denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, bank, all_candidates, CANDIDATE_CHUNK_SIZE
    ).cpu().numpy()
    assert q_values.shape == (1000, CANDIDATE_NUM_MC), f"unexpected q_values shape {q_values.shape}"
    assert np.isfinite(q_values).all(), "non-finite q_m values"

    # --- reproduction gate against the ORIGINAL saved per_transition_scores.csv ---
    import csv
    orig_path = OUT_ROOT / domain / condition / "per_transition_scores.csv"
    orig_scores = {}
    with open(orig_path) as f:
        for row in csv.DictReader(f):
            key = (row["behavior"], int(row["episode_id"]), int(row["transition_index"]))
            orig_scores[key] = float(row["lcg_score"])

    reproduced_mean = q_values.mean(axis=1)
    orig_ordered = np.array([orig_scores[(m["behavior"], m["episode_id"], m["transition_index"])] for m in all_meta])

    abs_err = np.abs(reproduced_mean - orig_ordered)
    rel_err = abs_err / np.maximum(np.abs(orig_ordered), 1e-8)
    pearson_repro = float(pearsonr(reproduced_mean, orig_ordered)[0])
    allclose = bool(np.allclose(reproduced_mean, orig_ordered, rtol=REPRO_RTOL, atol=REPRO_ATOL))

    repro_gate = {
        "max_abs_error": float(abs_err.max()), "mean_abs_error": float(abs_err.mean()),
        "max_rel_error": float(rel_err.max()), "pearson": pearson_repro, "allclose": allclose,
        "rtol": REPRO_RTOL, "atol": REPRO_ATOL,
    }
    print(f"REPRODUCTION GATE: max_abs_err={repro_gate['max_abs_error']:.6e} "
          f"max_rel_err={repro_gate['max_rel_error']:.6e} pearson={pearson_repro:.8f} allclose={allclose}")

    if not allclose:
        print("  *** REPRODUCTION GATE FAILED -- STOPPING before RMS interpretation for this domain/condition ***")
        return {"domain": domain, "condition": condition, "reproduction_gate": repro_gate, "gate_passed": False}

    # --- save q_values permanently ---
    save_dir = RMS_OUT_ROOT / domain / condition
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "q_values.npy", q_values)
    meta_out = {
        "domain": domain, "condition": condition, "M": CANDIDATE_NUM_MC,
        "candidate_order": all_meta, "checkpoint_path": str(checkpoint_path),
        "precision_seed": PRECISION_SEED, "bank_seed": BANK_SEED,
        "theta_s_dim": d_S, "h_D_sum": float(h_D.sum().item()),
        "lcg_config": {
            "precision_reference_size": PRECISION_REFERENCE_SIZE, "precision_num_mc": PRECISION_NUM_MC,
            "beta": BETA, "damping": DAMPING, "candidate_num_mc": CANDIDATE_NUM_MC,
            "candidate_chunk_size": CANDIDATE_CHUNK_SIZE,
        },
        "reproduction_gate": repro_gate,
    }
    (save_dir / "metadata.json").write_text(json.dumps(meta_out, indent=2))

    # --- compute LCG_mean, LCG_RMS, probe stats ---
    lcg_mean = q_values.mean(axis=1)
    lcg_rms = np.sqrt(np.mean(q_values ** 2, axis=1))
    probe_variance = np.mean((q_values - lcg_mean[:, None]) ** 2, axis=1)  # population variance
    probe_std = np.sqrt(probe_variance)
    probe_cv = np.where(lcg_mean > 1e-8, probe_std / np.maximum(lcg_mean, 1e-8), np.nan)

    identity_check = np.abs(lcg_rms ** 2 - (lcg_mean ** 2 + probe_variance))
    print(f"RMS^2 = mean^2 + variance identity: max abs discrepancy = {identity_check.max():.6e}")

    behaviors = np.array([m["behavior"] for m in all_meta])
    episode_ids = np.array([m["episode_id"] for m in all_meta])

    return {
        "domain": domain, "condition": condition, "gate_passed": True, "reproduction_gate": repro_gate,
        "q_values": q_values, "lcg_mean": lcg_mean, "lcg_rms": lcg_rms,
        "probe_std": probe_std, "probe_cv": probe_cv, "behaviors": behaviors, "episode_ids": episode_ids,
        "identity_max_discrepancy": float(identity_check.max()),
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

    # fraction of pairs whose relative order flips (sampled if n large; here n=1000 -> ~500k pairs, still cheap)
    rank_mean = np.argsort(np.argsort(mean_))
    rank_rms = np.argsort(np.argsort(rms_))
    # Kendall-tau-like discordance via rank difference sign changes on a random pair sample
    rng = np.random.default_rng(0)
    n_pairs = 20000
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    valid = i != j
    i, j = i[valid], j[valid]
    sign_mean = np.sign(mean_[i] - mean_[j])
    sign_rms = np.sign(rms_[i] - rms_[j])
    frac_order_changed = float(np.mean(sign_mean != sign_rms))

    return {
        "pearson": pear, "spearman": spear, "top10_overlap": overlap10, "top20_overlap": overlap20,
        "fraction_pairs_order_changed_sampled": frac_order_changed, "n_pairs_sampled": int(valid.sum()),
    }


def main() -> None:
    RMS_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    raw = {}
    for domain in DOMAINS:
        for condition in CONDITIONS:
            raw[(domain, condition)] = rescore_domain_condition(domain, condition)

    if not all(r["gate_passed"] for r in raw.values()):
        print("\n*** ONE OR MORE REPRODUCTION GATES FAILED. STOPPING BEFORE RMS INTERPRETATION. ***")
        failed = {f"{d}/{c}": r["reproduction_gate"] for (d, c), r in raw.items() if not r["gate_passed"]}
        (RMS_OUT_ROOT / "FAILED_reproduction_gates.json").write_text(json.dumps(failed, indent=2))
        return

    print("\n\n=== TABLE: REPRODUCTION GATE ===")
    print(f"{'domain':<10} {'condition':<12} {'max_abs_err':>12} {'max_rel_err':>12} {'allclose':>9}")
    for (d, c), r in raw.items():
        g = r["reproduction_gate"]
        print(f"{d:<10} {c:<12} {g['max_abs_error']:>12.6e} {g['max_rel_error']:>12.6e} {str(g['allclose']):>9}")

    print("\n=== TABLE A: mean_LCG / mean_RMS / probe stats per domain/condition/behavior ===")
    table_a = {}
    for (d, c), r in raw.items():
        for behavior in ("walk", "run"):
            key = f"{d}/{c}/{behavior}"
            table_a[key] = summarize_behavior(r, behavior)
            s = table_a[key]
            print(f"{key:<25} mean_LCG={s['mean_LCG_mean']:>9.4f} mean_RMS={s['mean_LCG_RMS']:>9.4f} "
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
        rs, ws = raw[(d, "run_scarce")], raw[(d, "walk_scarce")]
        cv_run_scarce = {"scarce(run)_CV": table_a[f"{d}/run_scarce/run"]["mean_probe_CV"],
                          "abundant(walk)_CV": table_a[f"{d}/run_scarce/walk"]["mean_probe_CV"]}
        cv_walk_scarce = {"scarce(walk)_CV": table_a[f"{d}/walk_scarce/walk"]["mean_probe_CV"],
                           "abundant(run)_CV": table_a[f"{d}/walk_scarce/run"]["mean_probe_CV"]}
        cv_diag[d] = {"run_scarce": cv_run_scarce, "walk_scarce": cv_walk_scarce}
        print(f"{d} run_scarce: {cv_run_scarce}")
        print(f"{d} walk_scarce: {cv_walk_scarce}")

    # --- figures ---
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
        "reproduction_gate": {f"{d}/{c}": r["reproduction_gate"] for (d, c), r in raw.items()},
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
