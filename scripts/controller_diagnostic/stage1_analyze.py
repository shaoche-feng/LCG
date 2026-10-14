import json, statistics as st
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # scripts/controller_diagnostic/<file>.py -> repo root

with open(f"{REPO_ROOT}/outputs/stage1_det_vs_stoch/results.json") as f:
    data = json.load(f)


def pct(sorted_vals, p):
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    idx = p / 100 * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def summarize(returns, label):
    s = sorted(returns)
    n = len(s)
    frac = lambda thresh: sum(1 for x in s if x > thresh) / n
    print(f"  {label:28s} n={n:4d} mean={st.mean(s):8.2f} median={st.median(s):8.2f} std={st.pstdev(s):8.2f} "
          f"min={s[0]:8.2f} max={s[-1]:8.2f} p10={pct(s,10):8.2f} p25={pct(s,25):8.2f} p75={pct(s,75):8.2f} p90={pct(s,90):8.2f} "
          f">200={frac(200)*100:5.1f}% >400={frac(400)*100:5.1f}% >600={frac(600)*100:5.1f}% >800={frac(800)*100:5.1f}%")


for ckpt_name, d in data.items():
    print(f"\n=== {ckpt_name} (epoch={d['epoch']} update={d['update']}) ===")
    det = d["deterministic"]
    stoch = d["stochastic"]

    det_returns = [r["total_return"] for r in det]
    stoch_returns = [r["total_return"] for r in stoch]
    summarize(det_returns, "deterministic (n=30)")
    summarize(stoch_returns, "stochastic (n=90, 3 reps/seed)")

    # per-seed stochastic mean (averaging over the 3 repeats) to compare directly against deterministic per seed
    by_seed = {}
    for r in stoch:
        by_seed.setdefault(r["seed"], []).append(r["total_return"])
    stoch_seed_mean = {seed: st.mean(vals) for seed, vals in by_seed.items()}
    summarize(list(stoch_seed_mean.values()), "stochastic (per-seed mean of 3, n=30)")

    # within-seed stochastic std (sampling noise magnitude)
    within_seed_std = [st.pstdev(vals) if len(vals) > 1 else 0.0 for vals in by_seed.values()]
    print(f"  within-seed stochastic std (sampling noise): mean={st.mean(within_seed_std):.2f} median={st.median(within_seed_std):.2f} max={max(within_seed_std):.2f}")

    # paired per-seed diff: stochastic(mean of 3) - deterministic
    det_by_seed = {r["seed"]: r["total_return"] for r in det}
    diffs = []
    improve = degrade = same = 0
    for seed in det_by_seed:
        d_ = stoch_seed_mean[seed] - det_by_seed[seed]
        diffs.append(d_)
        if d_ > 1e-6:
            improve += 1
        elif d_ < -1e-6:
            degrade += 1
        else:
            same += 1
    print(f"  paired diff (stoch_mean - det): mean={st.mean(diffs):.2f} median={st.median(diffs):.2f} std={st.pstdev(diffs):.2f} "
          f"min={min(diffs):.2f} max={max(diffs):.2f}  improve={improve} degrade={degrade} tie={same}")
    print(f"  sorted diffs: {[round(x,1) for x in sorted(diffs)]}")
