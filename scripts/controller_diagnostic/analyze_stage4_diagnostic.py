import json
import numpy as np
from scipy import stats
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # scripts/controller_diagnostic/<file>.py -> repo root

with open(f"{REPO_ROOT}/outputs/stage4_advantage_magnitude_diagnostic/raw_records.json") as f:
    data = json.load(f)

ORDER = ["early_epoch34", "mid_early_epoch55", "strong_learning_epoch80", "peak_epoch108", "late_epoch118"]
HORIZONS = [15, 50]


def pct(a, p):
    return float(np.percentile(a, p))


def per_checkpoint_frame(records):
    by_state = {}
    for r in records:
        by_state.setdefault(r["state_idx"], []).append(r)
    return by_state


summary_rows = []
full_report = {}

for name in ORDER:
    d = data[name]
    recs = d["candidates"]
    by_state = per_checkpoint_frame(recs)
    n_states = len(by_state)

    adv = np.array([r["imagined_advantage_mean"] for r in recs])

    result = dict(epoch=d["epoch"], update=d["update"], n_states=n_states, n_candidates=len(recs))

    print(f"\n{'='*90}\n{name}  (epoch={d['epoch']} update={d['update']}, n_states={n_states}, n_candidates={len(recs)})\n{'='*90}")

    # advantage distribution
    print(f"advantage distribution: mean={adv.mean():.3f} std={adv.std():.3f} median={np.median(adv):.3f} "
          f"min={adv.min():.3f} max={adv.max():.3f}")
    print(f"  p10={pct(adv,10):.3f} p25={pct(adv,25):.3f} p75={pct(adv,75):.3f} p90={pct(adv,90):.3f} "
          f"p95={pct(adv,95):.3f} p99={pct(adv,99):.3f}")
    result["advantage_dist"] = dict(mean=float(adv.mean()), std=float(adv.std()), median=float(np.median(adv)),
                                    min=float(adv.min()), max=float(adv.max()), p10=pct(adv,10), p25=pct(adv,25),
                                    p75=pct(adv,75), p90=pct(adv,90), p95=pct(adv,95), p99=pct(adv,99))

    for h in HORIZONS:
        key = f"real_h{h}"
        # within-state relative return
        rel = []
        adv_all = []
        for sid, group in by_state.items():
            raw = np.array([r[key] for r in group])
            mean_ = raw.mean()
            rel_ = raw - mean_
            rel.extend(rel_.tolist())
            adv_all.extend([r["imagined_advantage_mean"] for r in group])
        rel = np.array(rel); adv_all = np.array(adv_all)

        pearson_r, pearson_p = stats.pearsonr(adv_all, rel)
        spearman_r, spearman_p = stats.spearmanr(adv_all, rel)

        # pairwise ranking accuracy (within state): fraction of within-state candidate pairs
        # where sign(adv_i - adv_j) == sign(real_i - real_j)
        concordant = 0; total_pairs = 0
        for sid, group in by_state.items():
            n = len(group)
            a = [r["imagined_advantage_mean"] for r in group]
            b = [r[key] for r in group]
            for i in range(n):
                for j in range(i + 1, n):
                    da = a[i] - a[j]; db = b[i] - b[j]
                    if da == 0 or db == 0:
                        continue
                    total_pairs += 1
                    if (da > 0) == (db > 0):
                        concordant += 1
        pairwise_acc = concordant / total_pairs if total_pairs else float("nan")

        # within-state spearman distribution
        within_spearman = []
        for sid, group in by_state.items():
            a = [r["imagined_advantage_mean"] for r in group]
            b = [r[key] for r in group]
            if len(set(a)) < 2 or len(set(b)) < 2:
                continue
            rho, _ = stats.spearmanr(a, b)
            if not np.isnan(rho):
                within_spearman.append(rho)
        within_spearman = np.array(within_spearman)

        # sign-only vs magnitude: positive-only and negative-only subsets
        pos_mask = adv_all > 0
        neg_mask = adv_all < 0
        pos_pearson = stats.pearsonr(adv_all[pos_mask], rel[pos_mask]) if pos_mask.sum() > 2 else (float("nan"), float("nan"))
        pos_spearman = stats.spearmanr(adv_all[pos_mask], rel[pos_mask]) if pos_mask.sum() > 2 else (float("nan"), float("nan"))
        neg_pearson = stats.pearsonr(adv_all[neg_mask], rel[neg_mask]) if neg_mask.sum() > 2 else (float("nan"), float("nan"))
        neg_spearman = stats.spearmanr(adv_all[neg_mask], rel[neg_mask]) if neg_mask.sum() > 2 else (float("nan"), float("nan"))

        # advantage quintile bins
        quintiles = np.percentile(adv_all, [20, 40, 60, 80])
        bin_idx = np.digitize(adv_all, quintiles)
        bin_stats = []
        for b in range(5):
            m = bin_idx == b
            if m.sum() == 0:
                bin_stats.append(None)
                continue
            frac_above = (rel[m] > 0).mean()
            bin_stats.append(dict(n=int(m.sum()), mean_adv=float(adv_all[m].mean()), mean_rel_return=float(rel[m].mean()),
                                  frac_above_state_avg=float(frac_above)))

        # top-vs-bottom within state
        top1_vals, bot1_vals, topq_vals, botq_vals, mid_vals = [], [], [], [], []
        for sid, group in by_state.items():
            srt = sorted(group, key=lambda r: r["imagined_advantage_mean"])
            n = len(srt)
            q = max(1, n // 4)
            bot1_vals.append(srt[0][key]); top1_vals.append(srt[-1][key])
            botq_vals.extend([r[key] for r in srt[:q]])
            topq_vals.extend([r[key] for r in srt[-q:]])
            mid_vals.extend([r[key] for r in srt[q:n - q]])
        top1 = float(np.mean(top1_vals)); bot1 = float(np.mean(bot1_vals))
        topq = float(np.mean(topq_vals)); botq = float(np.mean(botq_vals)); midv = float(np.mean(mid_vals))

        print(f"\n--- horizon {h} ---")
        print(f"  global Pearson={pearson_r:.3f} (p={pearson_p:.2e})  Spearman={spearman_r:.3f} (p={spearman_p:.2e})")
        print(f"  within-state Spearman: mean={within_spearman.mean():.3f} median={np.median(within_spearman):.3f} "
              f"std={within_spearman.std():.3f} p25={pct(within_spearman,25):.3f} p75={pct(within_spearman,75):.3f} "
              f"frac>0={float((within_spearman>0).mean()):.2f} frac>0.2={float((within_spearman>0.2).mean()):.2f} "
              f"frac>0.4={float((within_spearman>0.4).mean()):.2f}  (n_states_used={len(within_spearman)})")
        print(f"  pairwise ranking accuracy={pairwise_acc:.3f} (n_pairs={total_pairs})")
        print(f"  positive-only (n={int(pos_mask.sum())}): Pearson={pos_pearson[0]:.3f} Spearman={pos_spearman[0]:.3f}")
        print(f"  negative-only (n={int(neg_mask.sum())}): Pearson={neg_pearson[0]:.3f} Spearman={neg_spearman[0]:.3f}")
        print(f"  advantage bins (Q1 low -> Q5 high):")
        for bi, bs in enumerate(bin_stats):
            if bs:
                print(f"    Q{bi+1}: n={bs['n']:3d} mean_adv={bs['mean_adv']:7.3f} mean_rel_return={bs['mean_rel_return']:8.2f} "
                      f"frac_above_avg={bs['frac_above_state_avg']:.2f}")
        print(f"  top-vs-bottom real return: top1={top1:.2f} bot1={bot1:.2f} gap={top1-bot1:.2f} | "
              f"topQ={topq:.2f} mid={midv:.2f} botQ={botq:.2f} gap(topQ-botQ)={topq-botq:.2f}")

        result[f"h{h}"] = dict(
            global_pearson=float(pearson_r), global_pearson_p=float(pearson_p),
            global_spearman=float(spearman_r), global_spearman_p=float(spearman_p),
            within_state_spearman_mean=float(within_spearman.mean()), within_state_spearman_median=float(np.median(within_spearman)),
            within_state_spearman_std=float(within_spearman.std()), within_state_spearman_p25=pct(within_spearman,25),
            within_state_spearman_p75=pct(within_spearman,75),
            frac_positive=float((within_spearman > 0).mean()), frac_gt_0p2=float((within_spearman > 0.2).mean()),
            frac_gt_0p4=float((within_spearman > 0.4).mean()),
            pairwise_ranking_accuracy=float(pairwise_acc), n_pairs=total_pairs,
            positive_only_pearson=float(pos_pearson[0]), positive_only_spearman=float(pos_spearman[0]), n_positive=int(pos_mask.sum()),
            negative_only_pearson=float(neg_pearson[0]), negative_only_spearman=float(neg_spearman[0]), n_negative=int(neg_mask.sum()),
            bins=bin_stats,
            top1=top1, bot1=bot1, top1_minus_bot1=top1 - bot1,
            topq=topq, botq=botq, mid=midv, topq_minus_botq=topq - botq,
        )

    full_report[name] = result
    summary_rows.append(dict(
        name=name, epoch=d["epoch"], update=d["update"],
        within_spearman_h15=result["h15"]["within_state_spearman_mean"],
        within_spearman_h50=result["h50"]["within_state_spearman_mean"],
        global_spearman_h15=result["h15"]["global_spearman"],
        global_spearman_h50=result["h50"]["global_spearman"],
        pos_spearman_h15=result["h15"]["positive_only_spearman"],
        neg_spearman_h15=result["h15"]["negative_only_spearman"],
        pairwise_acc_h15=result["h15"]["pairwise_ranking_accuracy"],
        topbot_gap_h15=result["h15"]["top1_minus_bot1"],
        topbot_gap_h50=result["h50"]["top1_minus_bot1"],
    ))

print(f"\n\n{'='*110}\nCROSS-CHECKPOINT SUMMARY\n{'='*110}")
header = f"{'checkpoint':25s} {'epoch':>6s} {'wSpear15':>9s} {'wSpear50':>9s} {'gSpear15':>9s} {'gSpear50':>9s} {'pos15':>7s} {'neg15':>7s} {'pairAcc15':>10s} {'top-bot15':>10s} {'top-bot50':>10s}"
print(header)
for row in summary_rows:
    print(f"{row['name']:25s} {row['epoch']:6d} {row['within_spearman_h15']:9.3f} {row['within_spearman_h50']:9.3f} "
          f"{row['global_spearman_h15']:9.3f} {row['global_spearman_h50']:9.3f} {row['pos_spearman_h15']:7.3f} "
          f"{row['neg_spearman_h15']:7.3f} {row['pairwise_acc_h15']:10.3f} {row['topbot_gap_h15']:10.2f} {row['topbot_gap_h50']:10.2f}")

with open(f"{REPO_ROOT}/outputs/stage4_advantage_magnitude_diagnostic/analysis_summary.json", "w") as f:
    json.dump(dict(per_checkpoint=full_report, summary_rows=summary_rows), f, indent=2)
print("\nSaved analysis_summary.json")
