#! /usr/bin/env python
"""
LCG backward/VJP variance diagnostic -- Part A: stochastic-source decomposition.

4x4x4=64 crossed design over independently-generated sigma/epsilon/xi "sets" (each set =
one shared value per stratum, applied identically to all 480 frozen candidates -- a CRN-
style broadcast per combination, letting the ANOVA-style decomposition below attribute
variability to sigma, epsilon, xi, and their interactions cleanly, since every candidate
sees the exact same perturbation within one combination). No K-averaging, no per-candidate
independent probes -- see Part B/C for that.

Uses only lcg.batched_vjp.score_candidates_batched (Stage 4.5, unmodified) with a single
CRNBank per combination (num_banks=1). Reuses the frozen 480 candidates and cached
h_D_full from diagnose_lcg_backward_variance_setup.py; does not regenerate either.

Usage:
    python scripts/diagnose_lcg_backward_variance_partA.py
"""
import itertools
import sys
import time
from pathlib import Path

def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while not ((p / "src").is_dir() and (p / "scripts").is_dir()):
        if p.parent == p:
            raise RuntimeError("could not locate LCG repo root (no ancestor has both src/ and scripts/)")
        p = p.parent
    return p


_REPO_ROOT = _find_repo_root(Path(__file__).parent)
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "backward_VJP" / "3-stratum"))

import numpy as np
import torch

import diagnose_lcg_backward_variance_setup as setup
from lcg.batched_vjp import score_candidates_batched
from lcg.crn import CRNBank
from lcg.sigma_strata import sample_sigma_stratum
from lcg.theta_s import selected_parameters

NUM_LEVELS = 4
RESULTS_PATH = setup.DIAG_DIR / "partA_scores_4x4x4.pt"


def make_sigma_set(seed, device):
    torch.manual_seed(seed)
    return tuple(sample_sigma_stratum(setup.SIGMA_CFG, m, setup.NUM_STRATA, 1, device).detach() for m in range(setup.NUM_STRATA))


def make_eps_or_xi_set(seed, single_shape, device):
    torch.manual_seed(seed)
    return tuple(torch.randn((1,) + single_shape, device=device).detach() for _ in range(setup.NUM_STRATA))


def anova_ss_decomposition(y: np.ndarray) -> dict:
    """y: (4,4,4) array for one candidate. Standard single-replicate 3-way factorial
    sum-of-squares decomposition (main effects A=sigma, B=epsilon, C=xi; 2-way
    interactions; 3-way interaction folded into 'residual' since there is no replication
    within a cell)."""
    mu = y.mean()
    m_a = y.mean(axis=(1, 2))  # (4,) marginal over b,c
    m_b = y.mean(axis=(0, 2))
    m_c = y.mean(axis=(0, 1))
    alpha = m_a - mu
    beta = m_b - mu
    gamma = m_c - mu

    m_ab = y.mean(axis=2)  # (4,4)
    m_ac = y.mean(axis=1)
    m_bc = y.mean(axis=0)
    ab_eff = m_ab - mu - alpha[:, None] - beta[None, :]
    ac_eff = m_ac - mu - alpha[:, None] - gamma[None, :]
    bc_eff = m_bc - mu - beta[:, None] - gamma[None, :]

    ss_total = ((y - mu) ** 2).sum()
    ss_a = 16 * (alpha ** 2).sum()
    ss_b = 16 * (beta ** 2).sum()
    ss_c = 16 * (gamma ** 2).sum()
    ss_ab = 4 * (ab_eff ** 2).sum()
    ss_ac = 4 * (ac_eff ** 2).sum()
    ss_bc = 4 * (bc_eff ** 2).sum()
    ss_resid = ss_total - ss_a - ss_b - ss_c - ss_ab - ss_ac - ss_bc
    return dict(total=ss_total, sigma=ss_a, epsilon=ss_b, xi=ss_c,
                sigma_x_epsilon=ss_ab, sigma_x_xi=ss_ac, epsilon_x_xi=ss_bc, residual_3way=ss_resid)


def summarize(values, name):
    arr = np.array(values, dtype=np.float64)
    print(f"    {name:<28} mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
          f"p90={np.percentile(arr, 90):.4f}  max={arr.max():.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    denoiser, action_dim = setup.load_converged_denoiser(device)
    params = selected_parameters(denoiser)
    dataset = setup.load_train_dataset()
    h_D_full = setup.get_or_compute_h_D_full(denoiser, params, dataset, device)
    candidates = setup.load_frozen_candidates(device)
    setup.verify_candidates(candidates)

    single_shape = (3, 64, 64)

    if RESULTS_PATH.is_file():
        print(f"Loading cached Part A scores from {RESULTS_PATH}.", flush=True)
        scores = torch.load(RESULTS_PATH, map_location="cpu", weights_only=True)
    else:
        sigma_sets = [make_sigma_set(1000 + i, device) for i in range(NUM_LEVELS)]
        eps_sets = [make_eps_or_xi_set(2000 + j, single_shape, device) for j in range(NUM_LEVELS)]
        xi_sets = [make_eps_or_xi_set(3000 + k, single_shape, device) for k in range(NUM_LEVELS)]

        scores = torch.zeros(NUM_LEVELS, NUM_LEVELS, NUM_LEVELS, setup.NUM_CANDIDATES)
        t0 = time.time()
        combo_idx = 0
        for i, j, k in itertools.product(range(NUM_LEVELS), range(NUM_LEVELS), range(NUM_LEVELS)):
            bank = CRNBank(sigmas=sigma_sets[i], epsilons=eps_sets[j], xis=xi_sets[k])
            s = score_candidates_batched(denoiser, params, h_D_full, (bank,), candidates, setup.CHUNK_SIZE)
            scores[i, j, k] = s
            combo_idx += 1
            if combo_idx % 8 == 0 or combo_idx == NUM_LEVELS ** 3:
                elapsed = time.time() - t0
                rate = combo_idx / elapsed
                eta = (NUM_LEVELS ** 3 - combo_idx) / rate if rate > 0 else float("nan")
                print(f"  [{combo_idx:>3}/{NUM_LEVELS ** 3}] elapsed={elapsed:>7.1f}s  eta={eta:>6.1f}s", flush=True)
        torch.save(scores, RESULTS_PATH)
        print(f"Saved Part A scores to {RESULTS_PATH}.", flush=True)

    assert torch.isfinite(scores).all(), "non-finite scores in Part A crossed design!"
    print(f"\nAll {scores.numel()} scores finite. Score range: min={scores.min().item():.6g} "
          f"max={scores.max().item():.6g}", flush=True)

    # ------------------------------------------------------------------------------
    # Score-level: simple marginal CV per candidate, per factor
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("PART A -- score-level variability (marginal CV per candidate)")
    print("=" * 88)
    y = scores.numpy()  # (4,4,4,480)
    m_a = y.mean(axis=(1, 2))  # (4,480) marginal over eps,xi -> varying sigma
    m_b = y.mean(axis=(0, 2))  # (4,480) marginal over sigma,xi -> varying eps
    m_c = y.mean(axis=(0, 1))  # (4,480) marginal over sigma,eps -> varying xi
    all_flat = y.reshape(-1, y.shape[-1])  # (64,480)

    def cv_over_axis0(m):
        return m.std(axis=0, ddof=1) / (np.abs(m.mean(axis=0)) + 1e-30)

    cv_sigma = cv_over_axis0(m_a)
    cv_eps = cv_over_axis0(m_b)
    cv_xi = cv_over_axis0(m_c)
    cv_total = cv_over_axis0(all_flat)
    print("  marginal CV (std/mean of the 4 marginal means, per candidate), across 480 candidates:")
    summarize(cv_sigma, "CV_sigma (marginal)")
    summarize(cv_eps, "CV_epsilon (marginal)")
    summarize(cv_xi, "CV_xi (marginal)")
    summarize(cv_total, "CV_total (all 64 combos)")

    log_y = np.log(np.clip(y, 1e-30, None))
    log_m_a = log_y.mean(axis=(1, 2))
    log_m_b = log_y.mean(axis=(0, 2))
    log_m_c = log_y.mean(axis=(0, 1))
    log_all = log_y.reshape(-1, log_y.shape[-1])
    print("\n  log-score CV (raw scores are heavy-tailed; reporting log-space variability too):")
    summarize(cv_over_axis0(log_m_a), "CV_sigma (log-space)")
    summarize(cv_over_axis0(log_m_b), "CV_epsilon (log-space)")
    summarize(cv_over_axis0(log_m_c), "CV_xi (log-space)")
    summarize(cv_over_axis0(log_all), "CV_total (log-space)")

    # ------------------------------------------------------------------------------
    # ANOVA sum-of-squares decomposition, per candidate
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("PART A -- ANOVA sum-of-squares decomposition (fraction of SS_total), per candidate")
    print("=" * 88)
    components = ["sigma", "epsilon", "xi", "sigma_x_epsilon", "sigma_x_xi", "epsilon_x_xi", "residual_3way"]
    frac = {c: [] for c in components}
    for j in range(setup.NUM_CANDIDATES):
        ss = anova_ss_decomposition(y[:, :, :, j])
        total = ss["total"] if ss["total"] > 0 else 1e-30
        for c in components:
            frac[c].append(ss[c] / total)
    for c in components:
        summarize(frac[c], f"SS_{c} / SS_total")

    # ------------------------------------------------------------------------------
    # Ranking-level: pairwise Spearman / top-10% / top-20% overlap, varying one factor
    # while holding the other two fixed (averaged over all fixed settings + all pairs
    # of the varying factor's levels)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("PART A -- ranking-level variability (pairwise, holding other two factors fixed)")
    print("=" * 88)

    def pairwise_ranking_stats(get_slice, held_ranges):
        spearmans, top10s, top20s = [], [], []
        for held in itertools.product(*held_ranges):
            for l1, l2 in itertools.combinations(range(NUM_LEVELS), 2):
                a = torch.from_numpy(get_slice(l1, held))
                b = torch.from_numpy(get_slice(l2, held))
                spearmans.append(setup.spearman_corr(a, b))
                top10s.append(setup.top_q_overlap(a, b, 0.10))
                top20s.append(setup.top_q_overlap(a, b, 0.20))
        return spearmans, top10s, top20s

    sp_s, t10_s, t20_s = pairwise_ranking_stats(lambda l, held: y[l, held[0], held[1], :], [range(NUM_LEVELS), range(NUM_LEVELS)])
    sp_e, t10_e, t20_e = pairwise_ranking_stats(lambda l, held: y[held[0], l, held[1], :], [range(NUM_LEVELS), range(NUM_LEVELS)])
    sp_x, t10_x, t20_x = pairwise_ranking_stats(lambda l, held: y[held[0], held[1], l, :], [range(NUM_LEVELS), range(NUM_LEVELS)])

    # baseline: vary everything (all 64 combos, all C(64,2) pairs)
    all_combo_scores = [y[i, j, k, :] for i, j, k in itertools.product(range(NUM_LEVELS), repeat=3)]
    sp_all, t10_all, t20_all = [], [], []
    for a_idx, b_idx in itertools.combinations(range(len(all_combo_scores)), 2):
        a = torch.from_numpy(all_combo_scores[a_idx])
        b = torch.from_numpy(all_combo_scores[b_idx])
        sp_all.append(setup.spearman_corr(a, b))
        t10_all.append(setup.top_q_overlap(a, b, 0.10))
        t20_all.append(setup.top_q_overlap(a, b, 0.20))

    def report_ranking(name, sp, t10, t20):
        sp, t10, t20 = np.array(sp), np.array(t10), np.array(t20)
        print(f"  vary {name:<26} (n_pairs={len(sp):>5})  Spearman mean={sp.mean():.4f} std={sp.std():.4f}  "
              f"top10={t10.mean():.4f}  top20={t20.mean():.4f}")

    report_ranking("sigma only", sp_s, t10_s, t20_s)
    report_ranking("epsilon only", sp_e, t10_e, t20_e)
    report_ranking("xi only", sp_x, t10_x, t20_x)
    report_ranking("everything (baseline)", sp_all, t10_all, t20_all)

    # sigma x epsilon interaction check: does the "sigma pairwise Spearman" (holding
    # eps,xi fixed) itself vary depending on WHICH eps was used? -> interaction on ranking
    print("\n  sigma x epsilon interaction (ranking-level):")
    sigma_spearman_by_eps = {b: [] for b in range(NUM_LEVELS)}
    for (i_held, j_held), sp in zip(itertools.product(range(NUM_LEVELS), range(NUM_LEVELS)),
                                     [None] * 0):  # placeholder, recomputed properly below
        pass
    # recompute directly: for each (eps=b, xi=c), mean pairwise sigma-Spearman over the 6 sigma pairs
    per_bc_mean_sigma_spearman = []
    for b in range(NUM_LEVELS):
        for c in range(NUM_LEVELS):
            pair_sps = []
            for l1, l2 in itertools.combinations(range(NUM_LEVELS), 2):
                a_ = torch.from_numpy(y[l1, b, c, :])
                b_ = torch.from_numpy(y[l2, b, c, :])
                pair_sps.append(setup.spearman_corr(a_, b_))
            per_bc_mean_sigma_spearman.append(np.mean(pair_sps))
    per_bc_mean_sigma_spearman = np.array(per_bc_mean_sigma_spearman)
    print(f"    mean sigma-pairwise-Spearman across the 16 (epsilon,xi) settings: "
          f"mean={per_bc_mean_sigma_spearman.mean():.4f}  std={per_bc_mean_sigma_spearman.std():.4f}  "
          f"min={per_bc_mean_sigma_spearman.min():.4f}  max={per_bc_mean_sigma_spearman.max():.4f}")
    print("    (large std here => the ranking impact of varying sigma itself depends on which "
          "epsilon/xi setting is active, i.e. a sigma x {epsilon,xi} interaction on ranking stability)")

    # ------------------------------------------------------------------------------
    # Explicit answer
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("PART A -- explicit answer")
    print("=" * 88)
    mean_frac = {c: float(np.mean(frac[c])) for c in components}
    ranked = sorted(mean_frac.items(), key=lambda kv: -kv[1])
    print("  Mean SS fraction ranking (score-level ANOVA):")
    for name, val in ranked:
        print(f"    {name:<20} {val:.4f}")
    ranking_disruption = {
        "sigma": 1 - np.mean(sp_s), "epsilon": 1 - np.mean(sp_e), "xi": 1 - np.mean(sp_x),
    }
    ranked_disruption = sorted(ranking_disruption.items(), key=lambda kv: -kv[1])
    print("\n  Ranking disruption (1 - mean pairwise Spearman) when varying one factor alone:")
    for name, val in ranked_disruption:
        print(f"    {name:<20} {val:.4f}")

    print("\nPart A complete.", flush=True)


if __name__ == "__main__":
    main()
