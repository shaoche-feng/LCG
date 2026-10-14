"""Analysis-only (no env/model needed): quantify perturbation response, autocorrelation/
dominant period, cross-seed waveform similarity, and low-vs-high oscillation
characteristics from the already-saved episodes.json.
"""
import json

import numpy as np
from scipy.signal import correlate
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # scripts/controller_diagnostic/<file>.py -> repo root

OUT_DIR = f"{REPO_ROOT}/outputs/gait_diagnostic"
HIGH_SEEDS = [1016, 1009, 1006]
LOW_SEEDS = [1027, 1019, 1022]


def perturbation_response(normal, perturbed, perturb_step):
    actions_n = np.array(normal["actions"])
    actions_p = np.array(perturbed["actions"])
    n = min(len(actions_n), len(actions_p))
    diff = np.abs(actions_n[:n] - actions_p[:n])  # (T, 6)
    windows = {"immediate_+1": perturb_step + 1, "+10": perturb_step + 10,
              "+25": perturb_step + 25, "+50": perturb_step + 50}
    result = {}
    for name, step in windows.items():
        if step < n:
            result[name] = diff[step].tolist()
            result[f"{name}_mean_abs_diff"] = float(diff[step].mean())
    # window-mean over [perturb_step+1, perturb_step+50] to see divergence vs. recovery trend
    pre = diff[max(0, perturb_step - 10):perturb_step].mean() if perturb_step >= 10 else None
    post_windows = {}
    for lo, hi in [(1, 10), (10, 25), (25, 50), (50, 100)]:
        a, b = perturb_step + lo, min(perturb_step + hi, n)
        if a < n:
            post_windows[f"mean_abs_diff_{lo}_{hi}"] = float(diff[a:b].mean())
    result["pre_perturbation_baseline_diff"] = float(pre) if pre is not None else None
    result["post_windows"] = post_windows
    # reward trajectory: cumulative at 50/100/250/final relative to perturb_step
    rew_n, rew_p = np.array(normal["rewards"]), np.array(perturbed["rewards"])
    cum_n, cum_p = np.cumsum(rew_n), np.cumsum(rew_p)
    result["cumulative_return_normal"] = {str(k): float(cum_n[min(k, len(cum_n) - 1)]) for k in [50, 100, 250, len(cum_n) - 1]}
    result["cumulative_return_perturbed"] = {str(k): float(cum_p[min(k, len(cum_p) - 1)]) for k in [50, 100, 250, len(cum_p) - 1]}
    return result


def autocorr_dominant_period(x):
    x = np.asarray(x) - np.mean(x)
    ac = correlate(x, x, mode="full")
    ac = ac[len(ac) // 2:]
    ac = ac / (ac[0] + 1e-12)
    # dominant period: first local max after lag 0 (skip until ac drops then rises)
    period = None
    for lag in range(2, min(len(ac), 100)):
        if ac[lag - 1] < ac[lag] > ac[lag + 1] if lag + 1 < len(ac) else False:
            period = lag
            break
    return ac[:60].tolist(), period


def best_phase_cross_correlation(x, y, max_lag=50):
    x, y = np.asarray(x) - np.mean(x), np.asarray(y) - np.mean(y)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    cc = correlate(x, y, mode="full")
    lags = np.arange(-len(y) + 1, len(x))
    mask = np.abs(lags) <= max_lag
    cc, lags = cc[mask], lags[mask]
    norm = np.sqrt((x ** 2).sum() * (y ** 2).sum()) + 1e-12
    cc_norm = cc / norm
    best_idx = np.argmax(np.abs(cc_norm))
    return float(cc_norm[best_idx]), int(lags[best_idx])


def main():
    with open(f"{OUT_DIR}/episodes.json") as f:
        data = json.load(f)

    results = {}

    # 4-5: perturbation response
    perturb_step = data["_meta"]["perturb_step"]
    results["perturbation_response"] = {}
    for seed in HIGH_SEEDS:
        normal, perturbed = data[f"{seed}_normal"], data[f"{seed}_perturbed"]
        results["perturbation_response"][seed] = perturbation_response(normal, perturbed, perturb_step)

    # 6: autocorrelation / dominant period per high seed, per action dim (using NORMAL episode)
    results["autocorrelation"] = {}
    for seed in HIGH_SEEDS:
        actions = np.array(data[f"{seed}_normal"]["actions"])
        results["autocorrelation"][seed] = {}
        for dim in range(actions.shape[1]):
            ac, period = autocorr_dominant_period(actions[:, dim])
            results["autocorrelation"][seed][f"dim_{dim}"] = dict(dominant_period=period, ac_lag1=ac[1], ac_lag5=ac[5] if len(ac) > 5 else None)

    # cross-seed waveform similarity (high seeds, pairwise, per action dim)
    results["cross_seed_similarity"] = {}
    for i, s1 in enumerate(HIGH_SEEDS):
        for s2 in HIGH_SEEDS[i + 1:]:
            a1 = np.array(data[f"{s1}_normal"]["actions"])
            a2 = np.array(data[f"{s2}_normal"]["actions"])
            key = f"{s1}_vs_{s2}"
            results["cross_seed_similarity"][key] = {}
            for dim in range(a1.shape[1]):
                corr, lag = best_phase_cross_correlation(a1[:, dim], a2[:, dim])
                results["cross_seed_similarity"][key][f"dim_{dim}"] = dict(best_corr=corr, best_lag=lag)

    # 9: low-vs-high action oscillation amplitude/std comparison
    results["low_vs_high_action_stats"] = {}
    for seed in HIGH_SEEDS + LOW_SEEDS:
        actions = np.array(data[f"{seed}_normal"]["actions"])
        label = "high" if seed in HIGH_SEEDS else "low"
        results["low_vs_high_action_stats"][seed] = dict(
            label=label, std_per_dim=actions.std(0).tolist(), mean_per_dim=actions.mean(0).tolist(),
            range_per_dim=(actions.max(0) - actions.min(0)).tolist(),
        )

    with open(f"{OUT_DIR}/analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved analysis.json")
    print(json.dumps({"perturbation_response_summary": {s: results["perturbation_response"][s].get("post_windows") for s in HIGH_SEEDS}}, indent=2))


if __name__ == "__main__":
    main()
