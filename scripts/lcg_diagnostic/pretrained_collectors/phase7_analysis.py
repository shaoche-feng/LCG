"""Phase 7 -- theta_S x M Robustness of the Undersampling Diagnostic: final
analysis. Loads the q_values.npy (1000, 12) + metadata.json files saved by
phase7_thetaS_M_scoring.py for all 8 theta_S x 18 models (2 domains x 3
conditions x 3 seeds), derives score_M(x) for M in {5,6,7,8,9,12} as POST-HOC
nested-prefix means of the same 12 stored probes (never resampled), and
computes every table/figure required by the Phase 7 spec. Torch-free (uses
scipy.stats freely -- this script never imports torch, so there is no
OMP-conflict risk here, unlike the GPU scoring script).

Timing reference note: only M=12 is MEASURED by this experiment (one pass per
model/theta_S, per the spec -- no independent per-M reruns). M=5..9 scoring
costs are ESTIMATED via linear interpolation of the M vs ms/candidate grid
already measured by the earlier resblock_ablation_diagnostic.py run (same 8
theta_S names, same production score_one_jvp_bank path, different checkpoint/
dataset) -- reported as an empirical reference, clearly labeled, never mixed
with this experiment's own MEASURED M=12 numbers.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase7_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent.parent
OUT_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "phase7_thetaS_M_robustness"

DOMAINS = ["walker", "quadruped"]
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]
SEEDS = ["A", "B", "C"]
THETA_S_ORDER = ["R1", "R2", "R3", "R1+R2", "R1+R3", "R2+R3", "3R", "full"]
M_LIST = [5, 6, 7, 8, 9, 12]
FOCUS_THETA = ["R1", "R1+R2", "full"]

EXPECTED_D_S = {"R1": 217664, "R2": 217664, "R3": 217664, "R1+R2": 435328, "R1+R3": 435328,
                 "R2+R3": 435328, "3R": 652992, "full": 654851}

# Reference-only M vs ms/candidate grid, from docs/lcg_diagnostic/resblock_ablation/pareto_summary.csv
# (a DIFFERENT checkpoint/dataset -- used only to ESTIMATE M=5..9 relative cost via interpolation).
REFERENCE_MS_PER_CAND = {
    "R1": {1: 3.840, 2: 7.645, 3: 11.443, 4: 15.251, 5: 19.068, 6: 22.870, 8: 30.489, 10: 38.080, 12: 45.695, 16: 60.908},
    "R2": {1: 3.358, 2: 6.464, 3: 9.663, 4: 12.990, 5: 16.161, 6: 19.322, 8: 25.744, 10: 32.204, 12: 38.770, 16: 51.671},
    "R3": {1: 2.635, 2: 4.785, 3: 7.269, 4: 9.630, 5: 11.906, 6: 14.290, 8: 18.978, 10: 26.195, 12: 34.738, 16: 47.300},
    "R1+R2": {1: 3.999, 2: 7.803, 3: 11.786, 4: 15.599, 5: 19.548, 6: 23.446, 8: 31.189, 10: 39.130, 12: 46.645, 16: 62.498},
    "R1+R3": {1: 3.946, 2: 7.791, 3: 11.727, 4: 15.648, 5: 19.505, 6: 23.397, 8: 31.150, 10: 38.903, 12: 46.730, 16: 62.615},
    "R2+R3": {1: 3.145, 2: 6.297, 3: 10.280, 4: 12.720, 5: 15.952, 6: 19.271, 8: 25.657, 10: 31.830, 12: 38.146, 16: 51.174},
    "3R": {1: 3.944, 2: 7.840, 3: 11.760, 4: 15.644, 5: 19.572, 6: 23.478, 8: 31.297, 10: 39.092, 12: 46.993, 16: 63.398},
    "full": {1: 3.958, 2: 7.908, 3: 11.932, 4: 15.879, 5: 19.815, 6: 23.872, 8: 31.786, 10: 39.650, 12: 47.465, 16: 63.503},
}


def interp_ms_per_cand(theta_name: str, M: int) -> float:
    grid = REFERENCE_MS_PER_CAND[theta_name]
    if M in grid:
        return grid[M]
    xs = sorted(grid.keys())
    lo = max(x for x in xs if x < M)
    hi = min(x for x in xs if x > M)
    frac = (M - lo) / (hi - lo)
    return grid[lo] + frac * (grid[hi] - grid[lo])


def topk_overlap(a: np.ndarray, b: np.ndarray, frac: float) -> float:
    k = int(round(len(a) * frac))
    top_a = set(np.argsort(a)[-k:].tolist())
    top_b = set(np.argsort(b)[-k:].tolist())
    return len(top_a & top_b) / k


def pairwise_reversal_rate(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_pairs: int = 20000) -> float:
    n = len(a)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    sign_a = np.sign(a[i] - a[j])
    sign_b = np.sign(b[i] - b[j])
    valid = sign_b != 0
    return float((sign_a[valid] != sign_b[valid]).mean())


def load_all() -> tuple[dict, dict]:
    """Returns (q_by_key, meta_by_key) keyed by (domain, seed, condition, theta_name), plus
    candidate behavior masks per domain."""
    q_by_key, meta_by_key = {}, {}
    for domain in DOMAINS:
        for seed in SEEDS:
            for condition in CONDITIONS:
                for theta_name in THETA_S_ORDER:
                    d = OUT_ROOT / domain / seed / condition / theta_name
                    q_path, m_path = d / "q_values.npy", d / "metadata.json"
                    if not (q_path.exists() and m_path.exists()):
                        continue
                    key = (domain, seed, condition, theta_name)
                    q_by_key[key] = np.load(q_path)
                    meta_by_key[key] = json.loads(m_path.read_text())
    return q_by_key, meta_by_key


def load_behavior_masks() -> dict:
    masks = {}
    for domain in DOMAINS:
        meta = json.loads((OUT_ROOT / domain / "candidates_meta.json").read_text())
        behavior = np.array([c["behavior"] for c in meta["candidates"]])
        masks[domain] = {"walk": behavior == "walk", "run": behavior == "run"}
    return masks


def main() -> None:
    q_by_key, meta_by_key = load_all()
    masks = load_behavior_masks()
    n_expected = len(DOMAINS) * len(SEEDS) * len(CONDITIONS) * len(THETA_S_ORDER)
    print(f"Loaded {len(q_by_key)}/{n_expected} (domain,seed,condition,theta_S) evaluations")
    if len(q_by_key) < n_expected:
        missing = [k for k in [(d, s, c, t) for d in DOMAINS for s in SEEDS for c in CONDITIONS for t in THETA_S_ORDER]
                   if k not in q_by_key]
        print(f"WARNING: {len(missing)} evaluations still missing (scoring may still be running): {missing[:10]}"
              f"{' ...' if len(missing) > 10 else ''}")
        print("Proceeding with partial data -- tables below reflect only completed evaluations.")

    # === score_M(x), Delta, Interaction for every (theta_S, M, domain, seed, condition) ===
    # score_M[key][M] -> np.ndarray shape (1000,)
    score_M = {key: {M: q[:, :M].mean(axis=1) for M in M_LIST} for key, q in q_by_key.items()}

    delta = {}  # (theta_S, M, domain, seed, condition) -> float
    for key, per_M in score_M.items():
        domain, seed, condition, theta_name = key
        for M, scores in per_M.items():
            d = scores[masks[domain]["run"]].mean() - scores[masks[domain]["walk"]].mean()
            delta[(theta_name, M, domain, seed, condition)] = float(d)

    interaction = {}  # (theta_S, M, domain, seed) -> float
    for theta_name in THETA_S_ORDER:
        for M in M_LIST:
            for domain in DOMAINS:
                for seed in SEEDS:
                    rs = delta.get((theta_name, M, domain, seed, "run_scarce"))
                    ws = delta.get((theta_name, M, domain, seed, "walk_scarce"))
                    if rs is not None and ws is not None:
                        interaction[(theta_name, M, domain, seed)] = rs - ws

    def ordered_cases(theta_name, M):
        n = 0
        for domain in DOMAINS:
            for seed in SEEDS:
                rs = delta.get((theta_name, M, domain, seed, "run_scarce"))
                bal = delta.get((theta_name, M, domain, seed, "balanced"))
                ws = delta.get((theta_name, M, domain, seed, "walk_scarce"))
                if rs is not None and bal is not None and ws is not None and rs > bal > ws:
                    n += 1
        return n

    def positive_interaction_cases(theta_name, M):
        n = 0
        for domain in DOMAINS:
            for seed in SEEDS:
                v = interaction.get((theta_name, M, domain, seed))
                if v is not None and v > 0:
                    n += 1
        return n

    # === Table 2 / 3: theta_S x M matrices ===
    matrix_ordered = {t: {M: ordered_cases(t, M) for M in M_LIST} for t in THETA_S_ORDER}
    matrix_interaction = {t: {M: positive_interaction_cases(t, M) for M in M_LIST} for t in THETA_S_ORDER}

    print("\n" + "=" * 100)
    print("TABLE 2: ordered_cases / 6  (Delta_RS > Delta_BAL > Delta_WS)")
    print("=" * 100)
    header = f"{'theta_S':<8}" + "".join(f"{'M' + str(M):>8}" for M in M_LIST)
    print(header)
    for t in THETA_S_ORDER:
        print(f"{t:<8}" + "".join(f"{matrix_ordered[t][M]}/6".rjust(8) for M in M_LIST))

    print("\n" + "=" * 100)
    print("TABLE 3: positive_interaction_cases / 6  (Delta_RS - Delta_WS > 0)")
    print("=" * 100)
    print(header)
    for t in THETA_S_ORDER:
        print(f"{t:<8}" + "".join(f"{matrix_interaction[t][M]}/6".rjust(8) for M in M_LIST))

    # === Table 4: smallest M with 6/6 ===
    print("\n" + "=" * 100)
    print("TABLE 4: theta_S | smallest_M_with_6/6_ordering | smallest_M_with_6/6_positive_interaction")
    print("=" * 100)
    table4 = {}
    for t in THETA_S_ORDER:
        m_ord = next((M for M in M_LIST if matrix_ordered[t][M] == 6), None)
        m_int = next((M for M in M_LIST if matrix_interaction[t][M] == 6), None)
        table4[t] = {"smallest_M_ordered_6of6": m_ord, "smallest_M_interaction_6of6": m_int}
        print(f"{t:<8} ordered_6of6_at_M={m_ord}   interaction_6of6_at_M={m_int}")

    # === Table 1: M=12 summary + measured timing ===
    print("\n" + "=" * 100)
    print("TABLE 1: theta_S | d_S | ordered_M12/6 | positive_interaction_M12/6 | measured_hD_s | measured_M12_score_s")
    print("=" * 100)
    table1 = {}
    for t in THETA_S_ORDER:
        hD_times = [meta_by_key[k]["h_D_construction_seconds"] for k in meta_by_key if k[3] == t]
        score_times = [meta_by_key[k]["candidate_scoring_seconds"] for k in meta_by_key if k[3] == t]
        row = {
            "d_S": EXPECTED_D_S[t], "ordered_M12": matrix_ordered[t][12], "positive_interaction_M12": matrix_interaction[t][12],
            "measured_hD_seconds_mean": float(np.mean(hD_times)) if hD_times else None,
            "measured_M12_scoring_seconds_mean": float(np.mean(score_times)) if score_times else None,
        }
        table1[t] = row
        print(f"{t:<8} d_S={row['d_S']:>7}  ordered={row['ordered_M12']}/6  interaction+={row['positive_interaction_M12']}/6  "
              f"hD={row['measured_hD_seconds_mean']:.1f}s  M12_score={row['measured_M12_scoring_seconds_mean']:.1f}s"
              if hD_times else f"{t:<8} (incomplete)")

    # === Table 5: stability vs M=12 (Delta/Interaction correlation + candidate-level ranking) ===
    print("\n" + "=" * 100)
    print("TABLE 5: theta_S | M | Delta_corr_vs_M12 | Interaction_corr_vs_M12 | cand_Spearman | top10 | top20 | pair_reversal")
    print("=" * 100)
    table5 = {}
    rng = np.random.default_rng(2024)
    for t in THETA_S_ORDER:
        for M in [m for m in M_LIST if m != 12]:
            delta_m = np.array([delta[(t, M, d, s, c)] for d in DOMAINS for s in SEEDS for c in CONDITIONS
                                 if (t, M, d, s, c) in delta and (t, 12, d, s, c) in delta])
            delta_12 = np.array([delta[(t, 12, d, s, c)] for d in DOMAINS for s in SEEDS for c in CONDITIONS
                                  if (t, M, d, s, c) in delta and (t, 12, d, s, c) in delta])
            inter_m = np.array([interaction[(t, M, d, s)] for d in DOMAINS for s in SEEDS
                                 if (t, M, d, s) in interaction and (t, 12, d, s) in interaction])
            inter_12 = np.array([interaction[(t, 12, d, s)] for d in DOMAINS for s in SEEDS
                                  if (t, M, d, s) in interaction and (t, 12, d, s) in interaction])
            delta_corr = float(pearsonr(delta_m, delta_12)[0]) if len(delta_m) > 2 else float("nan")
            inter_corr = float(pearsonr(inter_m, inter_12)[0]) if len(inter_m) > 2 else float("nan")

            sp_list, t10_list, t20_list, rev_list = [], [], [], []
            for key, q in q_by_key.items():
                if key[3] != t:
                    continue
                a, b = q[:, :M].mean(axis=1), q[:, :12].mean(axis=1)
                sp_list.append(spearmanr(a, b)[0])
                t10_list.append(topk_overlap(a, b, 0.10))
                t20_list.append(topk_overlap(a, b, 0.20))
                rev_list.append(pairwise_reversal_rate(a, b, rng))
            row = {
                "delta_corr_vs_M12": delta_corr, "interaction_corr_vs_M12": inter_corr,
                "candidate_spearman_vs_M12": float(np.mean(sp_list)) if sp_list else float("nan"),
                "top10_vs_M12": float(np.mean(t10_list)) if t10_list else float("nan"),
                "top20_vs_M12": float(np.mean(t20_list)) if t20_list else float("nan"),
                "pairwise_reversal_vs_M12": float(np.mean(rev_list)) if rev_list else float("nan"),
            }
            table5[(t, M)] = row
            print(f"{t:<8} M={M:<3} delta_corr={row['delta_corr_vs_M12']:.4f}  inter_corr={row['interaction_corr_vs_M12']:.4f}  "
                  f"cand_spearman={row['candidate_spearman_vs_M12']:.4f}  top10={row['top10_vs_M12']:.4f}  "
                  f"top20={row['top20_vs_M12']:.4f}  pair_rev={row['pairwise_reversal_vs_M12']:.4f}")

    # === Supplementary: reduced theta_S vs FULL (at M=12) ===
    print("\n" + "=" * 100)
    print("SUPPLEMENTARY: reduced theta_S vs FULL at M=12 (Delta/Interaction correlation)")
    print("=" * 100)
    vs_full = {}
    for t in [x for x in THETA_S_ORDER if x != "full"]:
        delta_t = np.array([delta[(t, 12, d, s, c)] for d in DOMAINS for s in SEEDS for c in CONDITIONS
                             if (t, 12, d, s, c) in delta and ("full", 12, d, s, c) in delta])
        delta_full = np.array([delta[("full", 12, d, s, c)] for d in DOMAINS for s in SEEDS for c in CONDITIONS
                                if (t, 12, d, s, c) in delta and ("full", 12, d, s, c) in delta])
        inter_t = np.array([interaction[(t, 12, d, s)] for d in DOMAINS for s in SEEDS
                             if (t, 12, d, s) in interaction and ("full", 12, d, s) in interaction])
        inter_full = np.array([interaction[("full", 12, d, s)] for d in DOMAINS for s in SEEDS
                                if (t, 12, d, s) in interaction and ("full", 12, d, s) in interaction])
        row = {
            "delta_pearson_vs_full": float(pearsonr(delta_t, delta_full)[0]) if len(delta_t) > 2 else float("nan"),
            "delta_spearman_vs_full": float(spearmanr(delta_t, delta_full)[0]) if len(delta_t) > 2 else float("nan"),
            "interaction_pearson_vs_full": float(pearsonr(inter_t, inter_full)[0]) if len(inter_t) > 2 else float("nan"),
            "interaction_spearman_vs_full": float(spearmanr(inter_t, inter_full)[0]) if len(inter_t) > 2 else float("nan"),
        }
        vs_full[t] = row
        print(f"{t:<8} Delta: pearson={row['delta_pearson_vs_full']:.4f} spearman={row['delta_spearman_vs_full']:.4f}  "
              f"Interaction: pearson={row['interaction_pearson_vs_full']:.4f} spearman={row['interaction_spearman_vs_full']:.4f}")

    # === Table 6: timing (measured M12, estimated M5-9) + robustness ===
    print("\n" + "=" * 100)
    print("TABLE 6: theta_S | M | time (MEASURED/ESTIMATED) | coverage_robustness (ordered/6)")
    print("=" * 100)
    table6 = {}
    for t in THETA_S_ORDER:
        for M in M_LIST:
            if M == 12 and t in table1 and table1[t]["measured_M12_scoring_seconds_mean"] is not None:
                time_s, label = table1[t]["measured_M12_scoring_seconds_mean"], "MEASURED"
            else:
                time_s, label = interp_ms_per_cand(t, M) * 1000 / 1000.0, "ESTIMATED"  # ms/cand * 1000 cand / 1000 = seconds
            row = {"time_seconds": time_s, "label": label, "coverage_robustness": matrix_ordered[t][M] / 6.0}
            table6[(t, M)] = row
            print(f"{t:<8} M={M:<3} time={time_s:>6.2f}s [{label:<9}]  robustness={row['coverage_robustness']:.3f}")

    # === Figures ===
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    theta_idx = {t: i for i, t in enumerate(THETA_S_ORDER)}

    def heatmap(matrix, title, fname, vmax=6):
        fig, ax = plt.subplots(figsize=(8, 6))
        arr = np.array([[matrix[t][M] for M in M_LIST] for t in THETA_S_ORDER])
        im = ax.imshow(arr, cmap="RdYlGn", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(M_LIST))); ax.set_xticklabels([f"M={m}" for m in M_LIST])
        ax.set_yticks(range(len(THETA_S_ORDER))); ax.set_yticklabels(THETA_S_ORDER)
        for i in range(len(THETA_S_ORDER)):
            for j in range(len(M_LIST)):
                ax.text(j, i, f"{arr[i, j]}/{vmax}", ha="center", va="center", fontsize=9,
                        color="white" if arr[i, j] < vmax * 0.4 or arr[i, j] > vmax * 0.85 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(OUT_ROOT / fname, dpi=150)
        plt.close(fig)

    heatmap(matrix_ordered, "Phase 7: coverage-ordering robustness (ordered_cases / 6)", "figA_ordering_heatmap.png")
    heatmap(matrix_interaction, "Phase 7: scarcity-interaction robustness (positive_interaction / 6)", "figB_interaction_heatmap.png")

    # C: mean interaction heatmap, by domain
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, domain in zip(axes, DOMAINS):
        arr = np.full((len(THETA_S_ORDER), len(M_LIST)), np.nan)
        for i, t in enumerate(THETA_S_ORDER):
            for j, M in enumerate(M_LIST):
                vals = [interaction[(t, M, domain, s)] for s in SEEDS if (t, M, domain, s) in interaction]
                if vals:
                    arr[i, j] = np.mean(vals)
        im = ax.imshow(arr, cmap="RdBu_r", vmin=-np.nanmax(np.abs(arr)), vmax=np.nanmax(np.abs(arr)), aspect="auto")
        ax.set_xticks(range(len(M_LIST))); ax.set_xticklabels([f"M={m}" for m in M_LIST])
        ax.set_yticks(range(len(THETA_S_ORDER))); ax.set_yticklabels(THETA_S_ORDER)
        for i in range(len(THETA_S_ORDER)):
            for j in range(len(M_LIST)):
                if np.isfinite(arr[i, j]):
                    ax.text(j, i, f"{arr[i, j]:.1f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"{domain}: mean interaction (across 3 seeds)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "figC_mean_interaction_heatmap.png", dpi=150)
    plt.close(fig)

    # D: trend plots for focus configs, M=12
    colors = {"A": "#888888", "B": "#2E86AB", "C": "#C73E1D"}
    for domain in DOMAINS:
        fig, axes = plt.subplots(1, len(FOCUS_THETA), figsize=(5 * len(FOCUS_THETA), 5), sharey=False)
        for ax, t in zip(axes, FOCUS_THETA):
            xs = [0, 1, 2]
            seed_vals = {}
            for s in SEEDS:
                vals = [delta.get((t, 12, domain, s, c)) for c in CONDITIONS]
                if all(v is not None for v in vals):
                    seed_vals[s] = vals
                    ax.plot(xs, vals, marker="o", linewidth=1.5, alpha=0.8, color=colors[s], label=f"Seed {s}")
            if seed_vals:
                mean_y = [np.mean([seed_vals[s][i] for s in seed_vals]) for i in range(3)]
                ax.plot(xs, mean_y, marker="s", markersize=9, linewidth=3, color="black", label="mean", zorder=5)
            ax.axhline(0, color="gray", linewidth=1, linestyle="--")
            ax.set_xticks(xs); ax.set_xticklabels(["run_scarce", "balanced", "walk_scarce"], rotation=15)
            ax.set_title(f"{t}  (d_S={EXPECTED_D_S[t]:,})")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(r"$\Delta$ = mean score(run) $-$ mean score(walk), M=12")
        fig.suptitle(f"{domain}: coverage response for focus theta_S configs (M=12)", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(OUT_ROOT / f"figD_{domain}_focus_trends.png", dpi=150)
        plt.close(fig)

    # E: compute/robustness Pareto
    fig, ax = plt.subplots(figsize=(9, 6.5))
    cmap = plt.get_cmap("tab10")
    for i, t in enumerate(THETA_S_ORDER):
        xs = [table6[(t, M)]["time_seconds"] for M in M_LIST]
        ys = [table6[(t, M)]["coverage_robustness"] for M in M_LIST]
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.6, color=cmap(i), label=t)
        for M, x, y in zip(M_LIST, xs, ys):
            ax.annotate(f"M{M}", (x, y), textcoords="offset points", xytext=(4, 3), fontsize=7, color=cmap(i))
    ax.set_xlabel("scoring time per model, seconds (MEASURED at M=12, ESTIMATED at M<12)")
    ax.set_ylabel("coverage robustness (ordered_cases / 6)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Phase 7: compute/robustness Pareto")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "figE_pareto.png", dpi=150)
    plt.close(fig)

    print("\nSaved figures: figA_ordering_heatmap.png, figB_interaction_heatmap.png, "
          "figC_mean_interaction_heatmap.png, figD_{domain}_focus_trends.png x2, figE_pareto.png")

    # === Save everything ===
    out = {
        "table1": table1, "table4": table4,
        "table2_ordered_matrix": {t: matrix_ordered[t] for t in THETA_S_ORDER},
        "table3_interaction_matrix": {t: matrix_interaction[t] for t in THETA_S_ORDER},
        "table5": {f"{t}/M{M}": v for (t, M), v in table5.items()},
        "table6": {f"{t}/M{M}": v for (t, M), v in table6.items()},
        "supplementary_vs_full_M12": vs_full,
    }
    (OUT_ROOT / "phase7_analysis_summary.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {OUT_ROOT / 'phase7_analysis_summary.json'}")


if __name__ == "__main__":
    main()
