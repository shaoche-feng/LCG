"""Analysis of the controller x model/data swap (CPU, read-only). Sxy = controller bundle of seed x, model/data bundle of
seed y, all from the epoch-90 (50k) checkpoints. Same diagnostics as the replay-value ablation's analyze.py.
Outputs results/swap_summary.json, results/swap_tables.md, plots/*.png. usage: python analyze_swap.py SWAP_DIR"""
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

O = Path(sys.argv[1])
sys.path.insert(0, "/home/bigking/scfeng/LCG-replayvalue-diagnostic-20261027/scripts")
from s5_behavior_analysis import CATS, T_MAX_CAL, analyse_episode  # noqa: E402

ARMS = ["S00", "S01", "S10", "S11"]
LABEL = dict(S00="S00 ctrl0 + model/data0", S01="S01 ctrl0 + model/data1", S10="S10 ctrl1 + model/data0", S11="S11 ctrl1 + model/data1")
THR, FIXED = (200, 400, 500, 600), [3001, 3004, 3016, 3021]
RES, PLOTS = O / "results", O / "plots"
RES.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)
CURVE_KEYS = ["return_mean", "lambda_return_dist_p10", "lambda_return_dist_p90", "value_mean", "value_dist_p10", "value_dist_p90", "value_bias",
              "loss_value_target", "value_target_correlation", "slow_value_gap", "positive_fraction", "advantage_std", "advantage_p10", "advantage_p90",
              "actor_grad_norm", "value_grad_norm", "actor_grad_clipped", "value_grad_clipped", "policy_entropy", "kl_prior", "kl_initial",
              "near_lower_fraction", "near_upper_fraction", "tanh_inverse_limited_fraction", "pretanh_std_mean", "pretanh_mean_abs_gt_3_fraction",
              "policy_mean_state_std_0", "action_std_0", "loss_actor", "loss_value"]


def stats(r):
    r = list(r)
    return dict(n=len(r), mean=st.mean(r), median=st.median(r), **{f"frac_gt_{t}": sum(x > t for x in r) / len(r) for t in THR})


def paired(x, y):
    d = np.array(x) - np.array(y)
    return dict(mean_diff=float(d.mean()), median_diff=float(np.median(d)), se_mean_diff=float(d.std(ddof=1) / math.sqrt(len(d))),
                wins=int((d > 0).sum()), losses=int((d < 0).sum()), ties=int((d == 0).sum()), n=len(d))


def label(ep):  # s11_rv_summary.py rules
    tf = ep["time_frac"]; up = tf["upright"] + tf["crouched"]; fallen = tf["fallen_fwd"] + tf["fallen_back"]; v = ep["fwd_vel_mean"]
    if up >= 0.8 and ep["n_falls"] <= 2 and v >= 0.5:
        return "sustained_walking"
    if up >= 0.5 and ep["n_falls"] >= 8 and v >= 0.5:
        return "dive_recover_gait"
    if tf["seated"] >= 0.5 and v >= 0.5:
        return "seated_movement"
    if fallen >= 0.5 or abs(v) < 0.2:
        return "static_or_lying"
    return "mixed"


def load(p):
    return json.loads(Path(p).read_text())


def evaluations():
    ev = {a: {u: load(O / f"arms/{a}/eval/eval_{k}_update_{u:05d}.json") for u, k in ((0, "eval30"), (500, "eval4"), (1000, "eval4"), (2000, "eval30"))} for a in ARMS}
    R = {a: {u: dict(zip(ev[a][u]["seeds"], ev[a][u]["returns"])) for u in ev[a]} for a in ARMS}
    seeds = ev["S00"][0]["seeds"]
    out = dict(update0={a: dict(stats(R[a][0].values()), reproduces_saved_50k=ev[a][0]["reproduces_saved_reference"]) for a in ARMS},
               update2000={a: stats(R[a][2000].values()) for a in ARMS},
               change_from_own_start={a: paired([R[a][2000][s] for s in seeds], [R[a][0][s] for s in seeds]) for a in ARMS})
    pair = lambda x, y: paired([R[x][2000][s] for s in seeds], [R[y][2000][s] for s in seeds])
    chg = lambda a: [R[a][2000][s] - R[a][0][s] for s in seeds]
    out["comparisons_2000"] = {"S01-S00 (model/data, ctrl0)": pair("S01", "S00"), "S11-S10 (model/data, ctrl1)": pair("S11", "S10"),
                               "S10-S00 (controller, model/data0)": pair("S10", "S00"), "S11-S01 (controller, model/data1)": pair("S11", "S01"),
                               "original pairings (S00+S11) - swapped (S01+S10), per seed": paired(
                                   [R["S00"][2000][s] + R["S11"][2000][s] for s in seeds], [R["S01"][2000][s] + R["S10"][2000][s] for s in seeds])}
    out["interaction_2000_mean"] = (out["update2000"]["S11"]["mean"] - out["update2000"]["S10"]["mean"]) - (out["update2000"]["S01"]["mean"] - out["update2000"]["S00"]["mean"])
    out["change_comparisons"] = {"S01-S00": paired(chg("S01"), chg("S00")), "S11-S10": paired(chg("S11"), chg("S10"))}
    out["fixed4"] = {a: {u: dict(per_seed={s: R[a][u][s] for s in FIXED}, **stats([R[a][u][s] for s in FIXED])) for u in (0, 500, 1000, 2000)} for a in ARMS}
    out["per_seed"] = {s: {a: dict(u0=R[a][0][s], u2000=R[a][2000][s]) for a in ARMS} for s in seeds}
    return out


def curves():
    out = {}
    for a in ARMS:
        rows = [json.loads(l) for l in open(O / f"arms/{a}/updates.jsonl")]
        out[a] = {}
        for w in range(0, len(rows), 100):
            chunk = rows[w:w + 100]
            d = {k: float(np.mean([r[k] for r in chunk])) for k in CURVE_KEYS if k in chunk[0]}
            d["value_target_rmse"] = float(np.mean([math.sqrt(r["loss_value_target"]) for r in chunk]))
            d["action_std_mean"] = float(np.mean([np.mean([r[f"action_std_{i}"] for i in range(6)]) for r in chunk]))
            out[a][chunk[-1]["diagnostic_update"]] = d
    return out


def fixed_inputs():
    fx = torch.load(O / "fixed_inputs_swap.pt", weights_only=False)
    src, post, t = np.array(fx["source"]), np.array(fx["posture"]), np.array(fx["t"])
    g = np.array(fx["g_trunc"])
    groups = {f"source_{s}": src == s for s in ("ctrl0_ep90", "ctrl1_ep90")}
    groups.update(upright_crouched=np.isin(post, ["upright", "crouched"]), seated=post == "seated", fallen=np.isin(post, ["fallen_fwd", "fallen_back"]))
    out = {}
    for a in ARMS:
        out[a] = {}
        own = "ctrl0_ep90" if a[1] == "0" else "ctrl1_ep90"
        for f in sorted((O / f"arms/{a}/checkpoints").glob("fixed_values_*.pt")):
            u = int(f.stem.split("_")[-1])
            fv = torch.load(f, weights_only=False)
            d = {}
            for kind in ("real_history", "repeated_frame"):
                vo, ve = fv[kind]["v_online"].numpy(), fv[kind]["v_ema"].numpy()
                for gname, m in groups.items():
                    d[f"{kind}/{gname}/online"] = float(vo[m].mean()); d[f"{kind}/{gname}/ema"] = float(ve[m].mean())
                m = (src == own) & (t <= T_MAX_CAL)
                d[f"{kind}/own_start_t<=300/online_minus_Gtrunc"] = float((vo[m] - g[m]).mean())
            out[a][u] = d
    return out


def index_checks():
    out = {}
    for a in ARMS:
        idx = [json.loads(l) for l in open(O / f"arms/{a}/indices.jsonl")]
        starts = np.array([s for r in idx for s in r["starts"]])
        group = np.minimum(starts[:, 0] // 25, 3)
        out[a] = dict(updates=len(idx), start_frames_checked=sum(1 for r in idx if r.get("start_frames_checked")),
                      start_frames_all_match=all(r["start_frames_match"] for r in idx if r.get("start_frames_checked")),
                      max_episode_id=int(starts[:, 0].max()), episode_group_fractions=[float((group == k).mean()) for k in range(4)],
                      unique_episodes_per_update_mean=float(np.mean([len(set(e for e, _ in r["starts"])) for r in idx])))
    for x, y in (("S00", "S10"), ("S01", "S11")):
        ix = [json.loads(l)["batches"] for l in open(O / f"arms/{x}/indices.jsonl")]
        iy = [json.loads(l)["batches"] for l in open(O / f"arms/{y}/indices.jsonl")]
        out[f"{x}_{y}_first_batch_identical_updates"] = sum(p[0] == q[0] for p, q in zip(ix, iy))
        out[f"{x}_{y}_all_batches_identical_updates"] = sum(p == q for p, q in zip(ix, iy))
    return out


def behaviour():
    sources = {"ctrl0_update0": (O / "behavior_update0/ctrl0", "ep90"), "ctrl1_update0": (O / "behavior_update0/ctrl1", "ep90")}
    sources.update({f"{a}_update2000": (O / f"arms/{a}/behavior", "ep2000") for a in ARMS})
    beh, cal, per_ep = {}, {}, {}
    for key, (d, prefix) in sources.items():
        summ = load(d / "summary.json")["episodes"]
        beh[key], pooled = {}, {}
        for seed in FIXED:
            name = f"{prefix}_{seed}"
            z = np.load(d / f"{name}.npz")
            e = analyse_episode(z, 0.985)
            ep = dict(seed=seed, total_return=e["total_return"], reproduces=summ[name]["reproduces"], time_frac=e["time_frac"], n_falls=len(e["falls"]),
                      n_recoveries=len(e["recoveries"]), fwd_vel_mean=e["fwd_vel_mean"], stationary_stretches=len(e["stationary_starts"]))
            ep["label"] = label(ep)
            beh[key][seed] = ep
            per_ep[f"{key}_{seed}"] = e["cat"].tolist()
            sel = np.arange(e["T"]) <= T_MAX_CAL
            v_ema = z["v_ema_real"].astype(float)
            for c in CATS + ["all"]:
                m = sel & ((e["cat"] == c) if c != "all" else True)
                if not m.any():
                    continue
                p = pooled.setdefault(c, dict(err=[], err_ema=[], bound=[], v=[], g=[], vdiff=[]))
                p["err"] += list(e["v"][m] - e["g_trunc"][m]); p["err_ema"] += list(v_ema[m] - e["g_trunc"][m]); p["bound"] += list(e["bound"][m])
                p["v"] += list(e["v"][m]); p["g"] += list(e["g_trunc"][m]); p["vdiff"] += list(e["v_rep"][m] - e["v"][m])
        cal[key] = {c: dict(n=len(p["err"]), v_mean=float(np.mean(p["v"])), g_trunc_mean=float(np.mean(p["g"])), signed_err_trunc=float(np.mean(p["err"])),
                            signed_err_lower_bound=float(np.mean(p["err"]) - np.mean(p["bound"])), residual_bound_max=float(np.max(p["bound"])),
                            ema_signed_err_trunc=float(np.mean(p["err_ema"])), v_rep_minus_v_real=float(np.mean(p["vdiff"]))) for c, p in pooled.items()}
    return beh, cal, per_ep


def plots(curv, per_ep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = dict(S00="#d62728", S01="#ff9896", S10="#1f77b4", S11="#aec7e8")
    ls = dict(S00="-", S01="--", S10="-", S11="--")
    panels = [("return_mean", "imagined λ-return mean"), ("value_mean", "critic value mean (imagined)"), ("value_bias", "value − λ-target"),
              ("value_target_rmse", "value-target RMSE"), ("value_target_correlation", "value-target correlation"), ("slow_value_gap", "|online − EMA|"),
              ("positive_fraction", "advantage ≥ 0 fraction"), ("advantage_std", "advantage SD"), ("value_grad_norm", "critic grad norm (pre-clip)"),
              ("actor_grad_norm", "actor grad norm (pre-clip)"), ("policy_entropy", "policy entropy"), ("kl_prior", "KL to prior"),
              ("action_std_mean", "sampled action SD (mean over dims)"), ("near_upper_fraction", "action > 0.99 of range"),
              ("near_lower_fraction", "action < 0.01 of range"), ("pretanh_std_mean", "pre-tanh std mean")]
    fig, axes = plt.subplots(4, 4, figsize=(18, 13))
    for ax, (k, title) in zip(axes.flat, panels):
        for a in ARMS:
            xs = [u for u in curv[a] if k in curv[a][u]]
            ax.plot(xs, [curv[a][u][k] for u in xs], color=col[a], ls=ls[a], label=LABEL[a], marker="o", ms=3)
        ax.set_title(title, fontsize=9); ax.set_xlabel("diagnostic update", fontsize=8); ax.tick_params(labelsize=7)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Controller × model/data swap from the 50k (epoch-90) checkpoints: 100-update window means (frozen world/reward models)", fontsize=11)
    fig.tight_layout(); fig.savefig(PLOTS / "learning_curves.png", dpi=80); plt.close(fig)
    colp = dict(upright="#2ca02c", crouched="#98df8a", seated="#ff7f0e", tilted="#9467bd", fallen_fwd="#d62728", fallen_back="#8c564b")
    sheets = [("ctrl0_update0", O / "behavior_update0/ctrl0", "ep90"), ("ctrl1_update0", O / "behavior_update0/ctrl1", "ep90")] + \
             [(f"{a}_update2000", O / f"arms/{a}/behavior", "ep2000") for a in ARMS]
    for key, d, prefix in sheets:
        fig = plt.figure(figsize=(20, 2.55 * 4))
        for i, seed in enumerate(FIXED):
            z = np.load(d / f"{prefix}_{seed}.npz")
            rend = z["render"]
            idx = list(range(0, len(rend), 25))[:10]
            ax = fig.add_axes([0.01, 1 - (i + 1) / 4 + 0.03 / 4, 0.98, 0.7 / 4])
            ax.imshow(np.concatenate([rend[j] for j in idx], axis=1)); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{key} seed {seed}: return {float(z['reward'].sum()):.0f}; frames at t = " + ", ".join(str(2 * j) for j in idx), fontsize=8, loc="left")
            ax2 = fig.add_axes([0.01, 1 - (i + 1) / 4 + 0.005 / 4, 0.98, 0.025 / 4])
            ax2.imshow(np.array([[matplotlib.colors.to_rgb(colp[c]) for c in per_ep[f"{key}_{seed}"]]]), aspect="auto"); ax2.set_xticks([]); ax2.set_yticks([])
        fig.legend(handles=[matplotlib.patches.Patch(color=c, label=k) for k, c in colp.items()], loc="lower right", ncol=6, fontsize=8)
        fig.savefig(PLOTS / f"sheet_{key}.png", dpi=55); plt.close(fig)


def main():
    ev, curv, fx, ix = evaluations(), curves(), fixed_inputs(), index_checks()
    beh, cal, per_ep = behaviour()
    json.dump(dict(evaluations=ev, curves=curv, fixed_inputs=fx, index_checks=ix, behaviour=beh, calibration=cal), open(RES / "swap_summary.json", "w"), indent=1, default=str)
    plots(curv, per_ep)
    f = lambda s: f"{s['mean']:.1f} / {s['median']:.1f} / " + " / ".join(f"{s[f'frac_gt_{t}']:.2f}" for t in THR)
    p = lambda x: f"{x['mean_diff']:+.1f} (median {x['median_diff']:+.1f}; SE {x['se_mean_diff']:.1f}; W/L {x['wins']}/{x['losses']})"
    md = ["# Controller × model/data swap (50k, epoch 90): tables", "", "## Deterministic 30-seed evaluation (mean / median / frac > 200 / 400 / 500 / 600)", "",
          "| arm | update 0 | update 2000 | change from own start (paired) |", "|---|---|---|---|"]
    for a in ARMS:
        md.append(f"| {LABEL[a]} | {f(ev['update0'][a])} | {f(ev['update2000'][a])} | {p(ev['change_from_own_start'][a])} |")
    md += ["", f"Update-0 evaluations reproduce the saved 50k eval30 of their controller: {[ev['update0'][a]['reproduces_saved_50k'] for a in ARMS]}", "",
           "| comparison at update 2000 (paired by seed) | difference |", "|---|---|"]
    for k, v in ev["comparisons_2000"].items():
        md.append(f"| {k} | {p(v)} |")
    md += [f"| interaction (S11 − S10) − (S01 − S00), means | {ev['interaction_2000_mean']:+.1f} |"]
    for k, v in ev["change_comparisons"].items():
        md.append(f"| change-from-start {k} | {p(v)} |")
    md += ["", "## 2 × 2 means at update 2000 (rows controller, columns model/data)", "", "| | model/data seed 0 | model/data seed 1 |", "|---|---|---|",
           f"| controller seed 0 | {ev['update2000']['S00']['mean']:.1f} | {ev['update2000']['S01']['mean']:.1f} |",
           f"| controller seed 1 | {ev['update2000']['S10']['mean']:.1f} | {ev['update2000']['S11']['mean']:.1f} |",
           "", "## Four fixed seeds (3001, 3004, 3016, 3021)", "", "| arm | update | 3001 | 3004 | 3016 | 3021 | mean |", "|---|---|---|---|---|---|---|"]
    for a in ARMS:
        for u, d in ev["fixed4"][a].items():
            md.append(f"| {a} | {u} | " + " | ".join(f"{d['per_seed'][s]:.0f}" for s in FIXED) + f" | {d['mean']:.1f} |")
    md += ["", "## Per-seed 30-seed returns", "", "| seed | ctrl0 u0 | S00 | S01 | ctrl1 u0 | S10 | S11 |", "|---|---|---|---|---|---|---|"]
    for s, d in ev["per_seed"].items():
        md.append(f"| {s} | {d['S00']['u0']:.1f} | {d['S00']['u2000']:.1f} | {d['S01']['u2000']:.1f} | {d['S10']['u0']:.1f} | {d['S10']['u2000']:.1f} | {d['S11']['u2000']:.1f} |")
    ends = [e for e in sorted(curv["S00"]) if e in (100, 300, 500, 800, 1000, 1300, 1500, 1800, 2000)]
    md += ["", f"## Learning curves (100-update window means at updates {ends})", ""]
    for k in ["return_mean", "value_mean", "value_bias", "value_target_rmse", "value_target_correlation", "slow_value_gap", "positive_fraction", "advantage_std",
              "actor_grad_norm", "value_grad_norm", "actor_grad_clipped", "value_grad_clipped", "policy_entropy", "kl_prior", "action_std_mean",
              "near_lower_fraction", "near_upper_fraction", "pretanh_std_mean"]:
        md.append(f"**{k}**: " + "; ".join(f"{a}: " + ", ".join(f"{curv[a][e][k]:.3g}" for e in ends) for a in ARMS))
        md.append("")
    md += ["## Fixed-input critic values (online / EMA)", "", "| arm | update | input | ctrl0 ep90 states | ctrl1 ep90 states | upright+crouched | seated | fallen | own-start t≤300: V − G_trunc |",
           "|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        for u, d in sorted(fx[a].items()):
            for kind in ("real_history", "repeated_frame"):
                md.append(f"| {a} | {u} | {kind} | " + " | ".join(f"{d[f'{kind}/source_{s}/online']:.1f} / {d[f'{kind}/source_{s}/ema']:.1f}" for s in ("ctrl0_ep90", "ctrl1_ep90"))
                          + " | " + " | ".join(f"{d[f'{kind}/{q}/online']:.1f}" for q in ("upright_crouched", "seated", "fallen")) + f" | {d[f'{kind}/own_start_t<=300/online_minus_Gtrunc']:+.1f} |")
    md += ["", "## Sampled-index checks", ""] + [f"- {k}: {json.dumps(v)}" for k, v in ix.items()]
    md += ["", "## Behaviour (four fixed seeds; deterministic evaluation policy)", "",
           "| source | seed | return | label | upright | crouched | seated | tilted | fallen fwd | fallen back | falls | recov | fwd vel |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, d in beh.items():
        for seed, e in d.items():
            tf = e["time_frac"]
            md.append(f"| {key} | {seed} | {e['total_return']:.0f} | {e['label']} | " + " | ".join(f"{tf[c]:.2f}" for c in CATS) + f" | {e['n_falls']} | {e['n_recoveries']} | {e['fwd_vel_mean']:.2f} |")
    md += ["", "## Real-return calibration, states t ≤ 300 (V − G_trunc; G_trunc lower bound; deterministic evaluation policy)", "",
           "| source | posture | n | V mean | G_trunc mean | online signed | lower bound | EMA signed | V_rep − V_real |", "|---|---|---|---|---|---|---|---|---|"]
    for key, d in cal.items():
        for c in ["all"] + CATS:
            if c in d:
                x = d[c]
                md.append(f"| {key} | {c} | {x['n']} | {x['v_mean']:.1f} | {x['g_trunc_mean']:.1f} | {x['signed_err_trunc']:+.1f} | {x['signed_err_lower_bound']:+.1f} | {x['ema_signed_err_trunc']:+.1f} | {x['v_rep_minus_v_real']:+.2f} |")
    (RES / "swap_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
