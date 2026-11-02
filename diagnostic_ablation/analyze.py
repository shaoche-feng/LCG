"""Analysis of the replay-value ablation (CPU, read-only). Outputs results/*.json, results/*.md and plots/*.png under
the ablation folder.
  1. evaluations: 30-seed at updates 0 (shared) and 2,000; four fixed seeds at 0, 500, 1,000, 2,000; paired B-A, C-B
  2. learning curves: 100-update window means of the controller signals, per arm
  3. fixed-input values: online and EMA critic values on fixed real states (real-history / repeated-frame inputs)
  4. sampled-index checks: layout, recency and episode diversity of the imagination starts
  5. behaviour and real-return calibration at update 2,000 (stage-5 tools; update 0 = the source run's epoch-190 traces)
usage: python analyze.py ABLATION_DIR"""
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

A = Path(sys.argv[1])
sys.path.insert(0, "/home/bigking/scfeng/LCG-replayvalue-diagnostic-20261027/scripts")
from s5_behavior_analysis import CATS, T_MAX_CAL, analyse_episode  # noqa: E402  (stage-5 tools, unchanged)

ARMS = "ABC"
LABEL = dict(A="A independent, no replay", B="B grouped, no replay", C="C grouped + replay 0.3")
THR = (200, 400, 500, 600)
RES, PLOTS = A / "results", A / "plots"
RES.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)
FIXED = [3001, 3004, 3016, 3021]
REF_TRACES = Path("/home/bigking/scfeng/LCG-replayvalue-diagnostic-20261027/data/behavior/rv1")


def stats(r):
    r = list(r)
    return dict(n=len(r), mean=st.mean(r), median=st.median(r), **{f"frac_gt_{t}": sum(x > t for x in r) / len(r) for t in THR})


def paired(x, y):
    d = np.array(x) - np.array(y)
    return dict(mean_diff=float(d.mean()), median_diff=float(np.median(d)), se_mean_diff=float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None,
                wins=int((d > 0).sum()), losses=int((d < 0).sum()), ties=int((d == 0).sum()), n=len(d))


def label(ep):  # s11_rv_summary.py rules, unchanged
    tf = ep["time_frac"]
    up = tf["upright"] + tf["crouched"]
    fallen = tf["fallen_fwd"] + tf["fallen_back"]
    v = ep["fwd_vel_mean"]
    if up >= 0.8 and len(ep["falls"]) <= 2 and v >= 0.5:
        return "sustained_walking"
    if up >= 0.5 and len(ep["falls"]) >= 8 and v >= 0.5:
        return "dive_recover_gait"
    if tf["seated"] >= 0.5 and v >= 0.5:
        return "seated_movement"
    if fallen >= 0.5 or abs(v) < 0.2:
        return "static_or_lying"
    return "mixed"


def evaluations():
    e0 = json.loads((A / "eval_shared/eval_eval30_update_00000.json").read_text())
    ev = {a: {u: json.loads((A / f"arms/{a}/eval/eval_{k}_update_{u:05d}.json").read_text()) for u, k in ((500, "eval4"), (1000, "eval4"), (2000, "eval30"))} for a in ARMS}
    seeds = e0["seeds"]
    r0 = dict(zip(seeds, e0["returns"]))
    r2000 = {a: dict(zip(ev[a][2000]["seeds"], ev[a][2000]["returns"])) for a in ARMS}
    out = dict(update0=dict(stats(e0["returns"]), reproduces_saved_epoch190=e0["reproduces_saved_reference"]),
               update2000={a: stats(r2000[a].values()) for a in ARMS})
    out["paired_2000"] = {"B-A": paired([r2000["B"][s] for s in seeds], [r2000["A"][s] for s in seeds]),
                          "C-B": paired([r2000["C"][s] for s in seeds], [r2000["B"][s] for s in seeds]),
                          "C-A": paired([r2000["C"][s] for s in seeds], [r2000["A"][s] for s in seeds]),
                          **{f"{a}-update0": paired([r2000[a][s] for s in seeds], [r0[s] for s in seeds]) for a in ARMS}}
    four = {a: {0: {s: r0[s] for s in FIXED}, 500: dict(zip(ev[a][500]["seeds"], ev[a][500]["returns"])),
                1000: dict(zip(ev[a][1000]["seeds"], ev[a][1000]["returns"])), 2000: {s: r2000[a][s] for s in FIXED}} for a in ARMS}
    out["fixed4"] = {a: {u: dict(per_seed=four[a][u], **stats(four[a][u].values())) for u in four[a]} for a in ARMS}
    out["fixed4_paired"] = {u: {"B-A": paired([four["B"][u][s] for s in FIXED], [four["A"][u][s] for s in FIXED]),
                                "C-B": paired([four["C"][u][s] for s in FIXED], [four["B"][u][s] for s in FIXED])} for u in (500, 1000, 2000)}
    out["per_seed"] = {s: dict(update0=r0[s], **{a: r2000[a][s] for a in ARMS}) for s in seeds}
    return out


CURVE_KEYS = ["return_mean", "lambda_return_dist_p10", "lambda_return_dist_p90", "value_mean", "value_dist_p10", "value_dist_p90",
              "value_bias", "loss_value_target", "value_target_correlation", "slow_value_gap", "positive_fraction", "advantage_std",
              "advantage_p10", "advantage_p90", "actor_grad_norm", "value_grad_norm", "actor_grad_clipped", "value_grad_clipped",
              "policy_entropy", "kl_prior", "kl_initial", "near_lower_fraction", "near_upper_fraction", "tanh_inverse_limited_fraction",
              "pretanh_mean_abs_gt_3_fraction", "pretanh_std_mean", "loss_value", "loss_value_imag", "loss_slowreg", "loss_actor",
              "replay_pred_mean", "replay_target_mean", "replay_bias", "replay_rmse", "loss_repval", "loss_repval_target", "loss_repval_slowreg",
              "loss_repval_weighted", "replay_slow_gap", "replay_target_reward_part_mean", "replay_target_boot_part_mean", "replay_boot_mean",
              "replay_weight_mean"]


def curves():
    rows = {a: [json.loads(l) for l in open(A / f"arms/{a}/updates.jsonl")] for a in ARMS}
    out = {}
    for a in ARMS:
        out[a] = {}
        for w in range(0, len(rows[a]), 100):
            chunk = rows[a][w:w + 100]
            d = {k: float(np.mean([r[k] for r in chunk])) for k in CURVE_KEYS if k in chunk[0]}
            d["value_target_rmse"] = float(np.mean([math.sqrt(r["loss_value_target"]) for r in chunk]))
            if "replay_target_mean" in chunk[0]:
                d["replay_reward_share"] = d["replay_target_reward_part_mean"] / d["replay_target_mean"]
            out[a][chunk[-1]["diagnostic_update"]] = d
    return out, rows


def fixed_inputs():
    fx = torch.load(A / "fixed_inputs.pt", weights_only=False)
    src = np.array(fx["source"])
    post = np.array(fx["posture"])
    g = np.array(fx["g_trunc"])
    bound = np.array(fx["bound"])
    groups = {f"source_{s}": src == s for s in ("rv190", "rv240", "base240")}
    groups.update(posture_upright_crouched=np.isin(post, ["upright", "crouched"]), posture_seated=post == "seated",
                  posture_fallen=np.isin(post, ["fallen_fwd", "fallen_back"]), all=np.ones(len(src), bool))
    out = {}
    for a in ARMS:
        out[a] = {}
        for f in sorted((A / f"arms/{a}/checkpoints").glob("fixed_values_*.pt")):
            u = int(f.stem.split("_")[-1])
            fv = torch.load(f, weights_only=False)
            d = {}
            for kind in ("real_history", "repeated_frame"):
                vo, ve = fv[kind]["v_online"].numpy(), fv[kind]["v_ema"].numpy()
                for gname, m in groups.items():
                    d[f"{kind}/{gname}/online"] = float(vo[m].mean())
                    d[f"{kind}/{gname}/ema"] = float(ve[m].mean())
                    d[f"{kind}/{gname}/online_minus_ema"] = float((vo[m] - ve[m]).mean())
                # rv190 states: discounted real return of the source (update-0) policy; policy-mismatched after update 0
                m = (src == "rv190") & (np.array(fx["t"]) <= T_MAX_CAL)
                d[f"{kind}/rv190_t<=300/online_minus_Gtrunc"] = float((vo[m] - g[m]).mean())
                d[f"{kind}/rv190_t<=300/ema_minus_Gtrunc"] = float((ve[m] - g[m]).mean())
                d[f"{kind}/rv190_t<=300/residual_bound_mean"] = float(bound[m].mean())
            a0 = fv["real_history"]["action"].numpy()
            d["real_history/action_abs_gt_0.95_fraction"] = float((np.abs(a0) > 0.95).mean())
            out[a][u] = d
    return out


def index_checks():
    out = {}
    lengths = None
    for a in ARMS:
        idx = [json.loads(l) for l in open(A / f"arms/{a}/indices.jsonl")]
        prov = json.loads((A / f"arms/{a}/provenance.json").read_text())
        n_ep = prov["dataset_num_episodes"]
        starts = np.array([s for r in idx for s in r["starts"]])
        group = np.minimum(starts[:, 0] // (n_ep // 4), 3)
        uniq = [len(set(e for e, _ in r["starts"])) for r in idx]
        out[a] = dict(updates=len(idx), starts=len(starts), start_frames_checked=sum(1 for r in idx if r.get("start_frames_checked")),
                      start_frames_all_match=all(r["start_frames_match"] for r in idx if r.get("start_frames_checked")),
                      episode_group_fractions=[float((group == k).mean()) for k in range(4)], mean_start_t=float(starts[:, 1].mean()),
                      unique_episodes_per_update_mean=float(np.mean(uniq)), unique_episodes_per_update_range=[int(min(uniq)), int(max(uniq))],
                      batches_per_update=sorted(set(r["n_batches"] for r in idx)),
                      unique_start_states_per_update_mean=float(np.mean([len(set(map(tuple, r["starts"]))) for r in idx])))
    ib = [json.loads(l) for l in open(A / "arms/B/indices.jsonl")]
    ic = [json.loads(l) for l in open(A / "arms/C/indices.jsonl")]
    out["B_C_first_batch_identical_updates"] = sum(b["batches"][0] == c["batches"][0] for b, c in zip(ib, ic))
    out["B_C_all_batches_identical_updates"] = sum(b["batches"] == c["batches"] for b, c in zip(ib, ic))
    return out


def behaviour():
    out, calib, per_ep = {}, {}, {}
    sources = {"update0": (REF_TRACES, "ep190", json.loads((REF_TRACES / "summary.json").read_text())["episodes"])}
    for a in ARMS:
        d = A / f"arms/{a}/behavior"
        sources[a] = (d, "ep2000", json.loads((d / "summary.json").read_text())["episodes"])
    for key, (d, prefix, summ) in sources.items():
        out[key], pooled = {}, {}
        for seed in FIXED:
            name = f"{prefix}_{seed}"
            z = np.load(d / f"{name}.npz")
            e = analyse_episode(z, 0.985)
            ep = dict(seed=seed, total_return=e["total_return"], reproduces=summ[name]["reproduces"], time_frac=e["time_frac"],
                      n_falls=len(e["falls"]), n_recoveries=len(e["recoveries"]), fwd_vel_mean=e["fwd_vel_mean"],
                      stationary_stretches=len(e["stationary_starts"]), near_bound_frac=e["near_bound_frac"], falls=e["falls"])
            ep["label"] = label(ep)
            del ep["falls"]
            out[key][seed] = ep
            per_ep[f"{key}_{seed}"] = dict(cat=e["cat"].tolist())
            sel = np.arange(e["T"]) <= T_MAX_CAL
            v_ema = z["v_ema_real"].astype(float) if "v_ema_real" in z else None
            for c in CATS + ["all"]:
                m = sel & ((e["cat"] == c) if c != "all" else True)
                if not m.any():
                    continue
                p = pooled.setdefault(c, dict(err=[], err_ema=[], bound=[], v=[], g=[], boot=[], vdiff=[]))
                p["err"] += list(e["v"][m] - e["g_trunc"][m]); p["bound"] += list(e["bound"][m]); p["v"] += list(e["v"][m]); p["g"] += list(e["g_trunc"][m])
                p["boot"] += list(e["v"][m] - e["g_boot"][m]); p["vdiff"] += list(e["v_rep"][m] - e["v"][m])
                if v_ema is not None:
                    p["err_ema"] += list(v_ema[m] - e["g_trunc"][m])
        calib[key] = {c: dict(n=len(p["err"]), v_mean=float(np.mean(p["v"])), g_trunc_mean=float(np.mean(p["g"])),
                              signed_err_trunc=float(np.mean(p["err"])), signed_err_lower_bound=float(np.mean(p["err"]) - np.mean(p["bound"])),
                              residual_bound_max=float(np.max(p["bound"])), signed_err_boot=float(np.mean(p["boot"])),
                              abs_err_trunc=float(np.mean(np.abs(p["err"]))), v_rep_minus_v_real=float(np.mean(p["vdiff"])),
                              ema_signed_err_trunc=float(np.mean(p["err_ema"])) if p["err_ema"] else None) for c, p in pooled.items()}
    return out, calib, per_ep


def plots(curv, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = dict(A="#1f77b4", B="#ff7f0e", C="#2ca02c")
    panels = [("return_mean", "imagined λ-return mean"), ("value_mean", "critic value mean (imagined states)"),
              ("value_bias", "value − λ-target (bias)"), ("value_target_rmse", "value-target RMSE"),
              ("value_target_correlation", "value-target correlation"), ("slow_value_gap", "|online − EMA| (imagined)"),
              ("positive_fraction", "advantage ≥ 0 fraction"), ("advantage_std", "advantage SD"),
              ("value_grad_norm", "critic grad norm (pre-clip)"), ("actor_grad_norm", "actor grad norm (pre-clip)"),
              ("policy_entropy", "policy entropy"), ("kl_prior", "KL to prior"),
              ("near_upper_fraction", "action > 0.99 of range"), ("near_lower_fraction", "action < 0.01 of range"),
              ("replay_target_mean", "replay target mean (B: monitor)"), ("replay_rmse", "replay RMSE (B: monitor)")]
    fig, axes = plt.subplots(4, 4, figsize=(18, 13))
    for ax, (k, title) in zip(axes.flat, panels):
        for a in ARMS:
            xs = [u for u in curv[a] if k in curv[a][u]]
            if xs:
                ax.plot(xs, [curv[a][u][k] for u in xs], color=col[a], label=LABEL[a], marker="o", ms=3)
        ax.set_title(title, fontsize=9); ax.set_xlabel("diagnostic update", fontsize=8); ax.tick_params(labelsize=7)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Replay-value ablation from epoch-190 checkpoint: 100-update window means (frozen world/reward models)", fontsize=11)
    fig.tight_layout(); fig.savefig(PLOTS / "learning_curves.png", dpi=80); plt.close(fig)


def sheets(per_ep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = dict(upright="#2ca02c", crouched="#98df8a", seated="#ff7f0e", tilted="#9467bd", fallen_fwd="#d62728", fallen_back="#8c564b")
    for a in ARMS:
        d = A / f"arms/{a}/behavior"
        fig = plt.figure(figsize=(20, 2.55 * 4))
        for i, seed in enumerate(FIXED):
            z = np.load(d / f"ep2000_{seed}.npz")
            rend = z["render"]
            idx = list(range(0, len(rend), 25))[:10]
            ax = fig.add_axes([0.01, 1 - (i + 1) / 4 + 0.03 / 4, 0.98, 0.7 / 4])
            ax.imshow(np.concatenate([rend[j] for j in idx], axis=1)); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"arm {a} update 2000 seed {seed}: return {float(z['reward'].sum()):.0f}; frames at t = " + ", ".join(str(2 * j) for j in idx), fontsize=8, loc="left")
            ax2 = fig.add_axes([0.01, 1 - (i + 1) / 4 + 0.005 / 4, 0.98, 0.025 / 4])
            ax2.imshow(np.array([[matplotlib.colors.to_rgb(col[c]) for c in per_ep[f"{a}_{seed}"]["cat"]]]), aspect="auto")
            ax2.set_xticks([]); ax2.set_yticks([])
        fig.legend(handles=[matplotlib.patches.Patch(color=c, label=k) for k, c in col.items()], loc="lower right", ncol=6, fontsize=8)
        fig.savefig(PLOTS / f"sheet_{a}_update2000.png", dpi=55); plt.close(fig)


def fmt_stats(s):
    return f"{s['mean']:.1f} / {s['median']:.1f} / " + " / ".join(f"{s[f'frac_gt_{t}']:.2f}" for t in THR)


def main():
    ev = evaluations()
    curv, rows = curves()
    fx = fixed_inputs()
    ix = index_checks()
    beh, cal, per_ep = behaviour()
    json.dump(dict(evaluations=ev, curves=curv, fixed_inputs=fx, index_checks=ix, behaviour=beh, calibration=cal),
              open(RES / "ablation_summary.json", "w"), indent=1, default=str)
    plots(curv, rows)
    sheets(per_ep)
    md = ["# Replay-value ablation: tables", "", "## 30-seed evaluation (mean / median / frac > 200 / 400 / 500 / 600)", "",
          f"- update 0 (shared, epoch-190 checkpoint): {fmt_stats(ev['update0'])}; reproduces saved epoch-190 eval30: {ev['update0']['reproduces_saved_epoch190']}"]
    for a in ARMS:
        md.append(f"- update 2000, {LABEL[a]}: {fmt_stats(ev['update2000'][a])}")
    md += ["", "| paired (update 2000) | mean diff | median diff | SE of mean diff | W / L / T |", "|---|---|---|---|---|"]
    for k, p in ev["paired_2000"].items():
        md.append(f"| {k} | {p['mean_diff']:+.1f} | {p['median_diff']:+.1f} | {p['se_mean_diff']:.1f} | {p['wins']} / {p['losses']} / {p['ties']} |")
    md += ["", "## Four fixed seeds (3001, 3004, 3016, 3021)", "", "| arm | update | 3001 | 3004 | 3016 | 3021 | mean |", "|---|---|---|---|---|---|---|"]
    for a in ARMS:
        for u, d in ev["fixed4"][a].items():
            md.append(f"| {a} | {u} | " + " | ".join(f"{d['per_seed'][s]:.0f}" for s in FIXED) + f" | {d['mean']:.1f} |")
    md += ["", "| paired, four seeds | B − A mean (W/L) | C − B mean (W/L) |", "|---|---|---|"]
    for u, d in ev["fixed4_paired"].items():
        md.append(f"| {u} | {d['B-A']['mean_diff']:+.1f} ({d['B-A']['wins']}/{d['B-A']['losses']}) | {d['C-B']['mean_diff']:+.1f} ({d['C-B']['wins']}/{d['C-B']['losses']}) |")
    md += ["", "## Per-seed 30-seed returns", "", "| seed | update 0 | A 2000 | B 2000 | C 2000 |", "|---|---|---|---|---|"]
    for s, d in ev["per_seed"].items():
        md.append(f"| {s} | {d['update0']:.1f} | {d['A']:.1f} | {d['B']:.1f} | {d['C']:.1f} |")
    md += ["", "## Learning curves (100-update window means, window ending at the listed update)", ""]
    ends = sorted(curv["A"])
    show = [e for e in ends if e in (100, 300, 500, 800, 1000, 1300, 1500, 1800, 2000)]
    for k in ["return_mean", "value_mean", "value_bias", "value_target_rmse", "value_target_correlation", "slow_value_gap", "positive_fraction",
              "advantage_std", "actor_grad_norm", "value_grad_norm", "actor_grad_clipped", "value_grad_clipped", "policy_entropy", "kl_prior",
              "near_lower_fraction", "near_upper_fraction", "replay_pred_mean", "replay_target_mean", "replay_bias", "replay_rmse",
              "loss_repval_target", "loss_repval_slowreg", "loss_repval_weighted", "loss_value_imag", "replay_target_reward_part_mean",
              "replay_target_boot_part_mean", "replay_reward_share"]:
        md.append(f"**{k}**: " + "; ".join(f"{a}: " + ", ".join(f"{curv[a][e][k]:.3g}" for e in show) for a in ARMS if k in curv[a][show[0]]))
        md.append("")
    md.insert(md.index("## Learning curves (100-update window means, window ending at the listed update)") + 2, f"Columns: updates {show}.")
    md += ["## Fixed-input critic values (mean over states)", "", "| arm | update | input | rv190 online / EMA | rv240 online / EMA | base240 online / EMA | upright+crouched online | seated online | fallen online | rv190 t≤300: online − G_trunc |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        for u, d in sorted(fx[a].items()):
            for kind in ("real_history", "repeated_frame"):
                md.append(f"| {a} | {u} | {kind} | " + " | ".join(f"{d[f'{kind}/source_{s}/online']:.1f} / {d[f'{kind}/source_{s}/ema']:.1f}" for s in ("rv190", "rv240", "base240"))
                          + " | " + " | ".join(f"{d[f'{kind}/posture_{p}/online']:.1f}" for p in ("upright_crouched", "seated", "fallen"))
                          + f" | {d[f'{kind}/rv190_t<=300/online_minus_Gtrunc']:+.1f} |")
    md += ["", "## Sampled-index checks", ""]
    for a in ARMS:
        x = ix[a]
        md.append(f"- {a}: {x['updates']} updates, {x['starts']} starts; start frames checked at {x['start_frames_checked']} updates, all match: {x['start_frames_all_match']}; "
                  f"episode-group fractions (oldest→newest quarter) {[round(v, 3) for v in x['episode_group_fractions']]}; mean start t {x['mean_start_t']:.1f}; "
                  f"unique episodes per update {x['unique_episodes_per_update_mean']:.1f} (range {x['unique_episodes_per_update_range']}); unique start states per update {x['unique_start_states_per_update_mean']:.1f}; "
                  f"prompt batches per update {x['batches_per_update']}")
    md.append(f"- B and C: first (replay) batch identical at {ix['B_C_first_batch_identical_updates']} / 2000 updates; all batches identical at {ix['B_C_all_batches_identical_updates']} / 2000")
    md += ["", "## Behaviour (four fixed seeds; deterministic evaluation policy)", "", "| source | seed | return | label | upright | crouched | seated | tilted | fallen fwd | fallen back | falls | recov | fwd vel |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, d in beh.items():
        for seed, e in d.items():
            tf = e["time_frac"]
            md.append(f"| {key} | {seed} | {e['total_return']:.0f} | {e['label']} | " + " | ".join(f"{tf[c]:.2f}" for c in CATS) + f" | {e['n_falls']} | {e['n_recoveries']} | {e['fwd_vel_mean']:.2f} |")
    md += ["", "## Real-return calibration, states t ≤ 300 (V − G_trunc; G_trunc is a lower bound, residual ≤ bound max)", "",
           "| source | posture | n | V mean | G_trunc mean | online signed | lower bound | EMA signed | boot signed | V_rep − V_real |", "|---|---|---|---|---|---|---|---|---|---|"]
    for key, d in cal.items():
        for c in ["all"] + CATS:
            if c in d:
                x = d[c]
                ema = f"{x['ema_signed_err_trunc']:+.1f}" if x["ema_signed_err_trunc"] is not None else "n/a"
                md.append(f"| {key} | {c} | {x['n']} | {x['v_mean']:.1f} | {x['g_trunc_mean']:.1f} | {x['signed_err_trunc']:+.1f} | {x['signed_err_lower_bound']:+.1f} | {ema} | {x['signed_err_boot']:+.1f} | {x['v_rep_minus_v_real']:+.2f} |")
    (RES / "ablation_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:40]))


if __name__ == "__main__":
    main()
