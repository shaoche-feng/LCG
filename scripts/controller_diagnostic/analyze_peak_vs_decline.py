import json, statistics as st
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # scripts/controller_diagnostic/<file>.py -> repo root

with open(f"{REPO_ROOT}/outputs/stage3_peak_vs_decline_30seed/results.json") as f:
    data = json.load(f)


def pct(sorted_vals, p):
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    idx = p / 100 * (n - 1)
    lo = int(idx); hi = min(lo + 1, n - 1); frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def summarize(returns, label):
    s = sorted(returns)
    n = len(s)
    frac = lambda thresh: sum(1 for x in s if x > thresh) / n
    band = lambda lo, hi: sum(1 for x in s if lo <= x < hi) / n
    print(f"{label}")
    print(f"  n={n} mean={st.mean(s):.2f} median={st.median(s):.2f} std={st.pstdev(s):.2f} min={s[0]:.2f} max={s[-1]:.2f}")
    print(f"  p10={pct(s,10):.2f} p25={pct(s,25):.2f} p75={pct(s,75):.2f} p90={pct(s,90):.2f}")
    print(f"  <200={band(0,200)*100:.1f}%  200-400={band(200,400)*100:.1f}%  400-600={band(400,600)*100:.1f}%  >600={frac(600)*100:.1f}%")
    print(f"  fraction >200={frac(200)*100:.1f}%  >400={frac(400)*100:.1f}%  >500={frac(500)*100:.1f}%  >600={frac(600)*100:.1f}%  >800={frac(800)*100:.1f}%")


order = ["context_epoch103", "peak_epoch108", "decline_epoch118"]
by_seed = {}
for name in order:
    d = data[name]
    print(f"\n=== {name} (epoch={d['epoch']} update={d['update']}) ===")
    returns = [r["total_return"] for r in d["results"]]
    summarize(returns, name)
    by_seed[name] = {r["seed"]: r["total_return"] for r in d["results"]}
    h = d["controller_health"]
    print(f"  controller_health: alpha_mean={h['alpha_mean']:.2f} alpha_median={h['alpha_median']:.2f} alpha_max={h['alpha_max']:.2f} "
          f"beta_mean={h['beta_mean']:.2f} beta_median={h['beta_median']:.2f} beta_max={h['beta_max']:.2f} "
          f"entropy={h['policy_entropy']:.3f} near_lower={h['near_lower_fraction']*100:.2f}% near_upper={h['near_upper_fraction']*100:.2f}% "
          f"fixed_state_std={h['fixed_state_policy_mean_std']:.4f}")

print("\n=== PAIRED: decline - peak ===")
seeds = sorted(by_seed["peak_epoch108"].keys())
diffs = []
improve = degrade = same = 0
for seed in seeds:
    d_ = by_seed["decline_epoch118"][seed] - by_seed["peak_epoch108"][seed]
    diffs.append((seed, d_))
    if d_ > 20:
        degrade += 1
    elif d_ < -20:
        degrade += 1  # placeholder, fixed below
print("(recompute with correct sign convention below)")

diffs2 = []
improved = degraded = unchanged = 0
for seed in seeds:
    change = by_seed["decline_epoch118"][seed] - by_seed["peak_epoch108"][seed]
    diffs2.append(change)
    if change > 20:
        improved += 1
    elif change < -20:
        degraded += 1
    else:
        unchanged += 1
print(f"n={len(diffs2)}  improved(+20)={improved}  degraded(-20)={degraded}  unchanged(within +-20)={unchanged}")
print(f"mean paired change = {st.mean(diffs2):.2f}   median paired change = {st.median(diffs2):.2f}")

print("\nper-seed peak -> decline (sorted by peak return):")
for seed in sorted(seeds, key=lambda s: by_seed["peak_epoch108"][s]):
    p = by_seed["peak_epoch108"][seed]
    dcl = by_seed["decline_epoch118"][seed]
    basin_p = "HIGH" if p > 350 else "normal"
    basin_d = "HIGH" if dcl > 350 else "normal"
    migration = "" if basin_p == basin_d else f"  <-- MIGRATED {basin_p}->{basin_d}"
    print(f"  seed={seed}  peak={p:7.2f} ({basin_p:6s})  decline={dcl:7.2f} ({basin_d:6s})  change={dcl-p:+8.2f}{migration}")

# basin membership stability
peak_high = {s for s in seeds if by_seed["peak_epoch108"][s] > 350}
decline_high = {s for s in seeds if by_seed["decline_epoch118"][s] > 350}
print(f"\npeak checkpoint: {len(peak_high)}/{len(seeds)} seeds in HIGH basin (>350)")
print(f"decline checkpoint: {len(decline_high)}/{len(seeds)} seeds in HIGH basin (>350)")
print(f"seeds HIGH at peak that stayed HIGH at decline: {len(peak_high & decline_high)}")
print(f"seeds HIGH at peak that dropped to normal at decline: {len(peak_high - decline_high)}")
print(f"seeds normal at peak that became HIGH at decline: {len(decline_high - peak_high)}")
