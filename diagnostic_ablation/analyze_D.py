"""Analysis of baseline follow-up D, with arm A for reference (CPU, read-only). Same diagnostics as analyze.py.
D starts from the H10 scalar-EMA baseline's epoch-190 state; A from the replay-value run's. The two arms share the
controller settings, update schedule and prompt-seeding convention, but not the source policy, models or dataset:
the change-vs-change comparison contrasts two learned starting states, not a replay-loss effect.
Outputs baseline_D/results/{D_summary.json, D_tables.md}, baseline_D/plots/*.png.
usage: python analyze_D.py ABLATION_DIR"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

AB = Path(sys.argv[1])
B = AB / "baseline_D"
sys.path.insert(0, "/home/bigking/scfeng/LCG-replayvalue-diagnostic-20261027/scripts")
sys.path.insert(0, str(Path(__file__).parent))
from s5_behavior_analysis import CATS, T_MAX_CAL, analyse_episode  # noqa: E402
sys.argv = [sys.argv[0], str(AB)]
import analyze as base  # noqa: E402  (stats, paired, label, CURVE_KEYS; module-level paths point at AB)

FIXED, THR = base.FIXED, base.THR
RES, PLOTS = B / "results", B / "plots"
RES.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)
TRACE_BASE = Path("/home/bigking/scfeng/LCG-replayvalue-diagnostic-20261027/data/behavior/h10s1")


def load(p):
    return json.loads(Path(p).read_text())


def evaluations():
    d0, d2 = load(B / "eval_update0/eval_eval30_update_00000.json"), load(B / "D/eval/eval_eval30_update_02000.json")
    a0, a2 = load(AB / "eval_shared/eval_eval30_update_00000.json"), load(AB / "arms/A/eval/eval_eval30_update_02000.json")
    seeds = d0["seeds"]
    R = {k: dict(zip(v["seeds"], v["returns"])) for k, v in dict(d0=d0, d2=d2, a0=a0, a2=a2).items()}
    dD = [R["d2"][s] - R["d0"][s] for s in seeds]
    dA = [R["a2"][s] - R["a0"][s] for s in seeds]
    out = dict(D_update0=dict(base.stats(d0["returns"]), reproduces_saved_epoch190=d0["reproduces_saved_reference"],
                              max_abs_diff_vs_saved=d0["max_abs_diff_vs_saved_reference"]),
               D_update2000=base.stats(d2["returns"]), A_update0=base.stats(a0["returns"]), A_update2000=base.stats(a2["returns"]),
               D_change=base.paired([R["d2"][s] for s in seeds], [R["d0"][s] for s in seeds]),
               A_change=base.paired([R["a2"][s] for s in seeds], [R["a0"][s] for s in seeds]),
               change_D_minus_change_A=base.paired(dD, dA),
               relative_change=dict(D=float(np.mean(dD) / np.mean(list(R["d0"].values()))), A=float(np.mean(dA) / np.mean(list(R["a0"].values())))),
               threshold_changes={f"frac_gt_{t}": dict(D=base.stats(d2["returns"])[f"frac_gt_{t}"] - base.stats(d0["returns"])[f"frac_gt_{t}"],
                                                       A=base.stats(a2["returns"])[f"frac_gt_{t}"] - base.stats(a0["returns"])[f"frac_gt_{t}"]) for t in THR},
               per_seed={s: dict(D0=R["d0"][s], D2000=R["d2"][s], A0=R["a0"][s], A2000=R["a2"][s]) for s in seeds})
    four = {"D": {0: {s: R["d0"][s] for s in FIXED}, 2000: {s: R["d2"][s] for s in FIXED}},
            "A": {0: {s: R["a0"][s] for s in FIXED}, 2000: {s: R["a2"][s] for s in FIXED}}}
    for arm, root in (("D", B / "D/eval"), ("A", AB / "arms/A/eval")):
        for u in (500, 1000):
            e = load(root / f"eval_eval4_update_{u:05d}.json")
            four[arm][u] = dict(zip(e["seeds"], e["returns"]))
    out["fixed4"] = {arm: {u: dict(per_seed=four[arm][u], **base.stats(four[arm][u].values())) for u in sorted(four[arm])} for arm in four}
    sto = {u: load(B / f"D/eval/eval_stochastic30x3_update_{u:05d}.json") for u in (0, 2000)}
    out["stochastic"] = {u: dict(all_episodes=s["all_episodes"], per_env_seed_mean_stats=s["per_env_seed_mean_stats"],
                                 action_seed_spread_mean_sd=s["action_seed_spread_mean_sd"], noise_seed_rule=s["noise_seed_rule"]) for u, s in sto.items()}
    pe = {u: {int(k): v for k, v in sto[u]["per_env_seed_mean"].items()} for u in sto}
    out["stochastic_change"] = base.paired([pe[2000][s] for s in seeds], [pe[0][s] for s in seeds])
    out["stochastic_minus_deterministic"] = {u: base.paired([pe[u][s] for s in seeds], [R["d0" if u == 0 else "d2"][s] for s in seeds]) for u in (0, 2000)}
    out["stochastic_per_seed"] = {s: {u: pe[u][s] for u in pe} for s in seeds}
    return out


def curves():
    out = {}
    for arm, path in (("D", B / "D/updates.jsonl"), ("A", AB / "arms/A/updates.jsonl")):
        rows = [json.loads(l) for l in open(path)]
        out[arm] = {}
        for w in range(0, len(rows), 100):
            chunk = rows[w:w + 100]
            d = {k: float(np.mean([r[k] for r in chunk])) for k in base.CURVE_KEYS if k in chunk[0]}
            d["value_target_rmse"] = float(np.mean([math.sqrt(r["loss_value_target"]) for r in chunk]))
            out[arm][chunk[-1]["diagnostic_update"]] = d
    return out


def fixed_inputs():
    fx = torch.load(B / "fixed_inputs_D.pt", weights_only=False)
    src, post, t = np.array(fx["source"]), np.array(fx["posture"]), np.array(fx["t"])
    g, bound = np.array(fx["g_trunc"]), np.array(fx["bound"])
    n_shared = 600
    groups = {f"source_{s}": src == s for s in ("rv190", "rv240", "base240", "base190")}
    groups.update(posture_upright_crouched=np.isin(post, ["upright", "crouched"]), posture_seated=post == "seated",
                  posture_fallen=np.isin(post, ["fallen_fwd", "fallen_back"]))
    out = {}
    for arm, root in (("D", B / "D/checkpoints"), ("A", AB / "arms/A/checkpoints")):
        out[arm] = {}
        for f in sorted(root.glob("fixed_values_*.pt")):
            u = int(f.stem.split("_")[-1])
            fv = torch.load(f, weights_only=False)
            d = {}
            for kind in ("real_history", "repeated_frame"):
                vo, ve = fv[kind]["v_online"].numpy(), fv[kind]["v_ema"].numpy()
                n = len(vo)
                for gname, m in groups.items():
                    m = m[:n]
                    if m.any():
                        d[f"{kind}/{gname}/online"] = float(vo[m].mean()); d[f"{kind}/{gname}/ema"] = float(ve[m].mean())
                shared = np.arange(n) < n_shared
                for p in ("posture_upright_crouched", "posture_seated", "posture_fallen"):
                    m = groups[p][:n] & shared
                    d[f"{kind}/shared600_{p}/online"] = float(vo[m].mean())
                for s in ("rv190", "base190"):  # own-start calibration states (G_trunc from that source's own policy)
                    m = (src[:n] == s) & (t[:n] <= T_MAX_CAL)
                    if m.any():
                        d[f"{kind}/{s}_t<=300/online_minus_Gtrunc"] = float((vo[m] - g[:n][m]).mean())
            out[arm][u] = d
    return out


def index_checks():
    idx = [json.loads(l) for l in open(B / "D/indices.jsonl")]
    prov = load(B / "D/provenance.json")
    starts = np.array([s for r in idx for s in r["starts"]])
    group = np.minimum(starts[:, 0] // (prov["dataset_num_episodes"] // 4), 3)
    uniq = [len(set(e for e, _ in r["starts"])) for r in idx]
    return dict(updates=len(idx), starts=len(starts), start_frames_checked=sum(1 for r in idx if r.get("start_frames_checked")),
                start_frames_all_match=all(r["start_frames_match"] for r in idx if r.get("start_frames_checked")),
                episode_group_fractions=[float((group == k).mean()) for k in range(4)], mean_start_t=float(starts[:, 1].mean()),
                unique_episodes_per_update_mean=float(np.mean(uniq)), batches_per_update=sorted(set(r["n_batches"] for r in idx)),
                dataset_num_episodes=prov["dataset_num_episodes"], dataset_num_steps=prov["dataset_num_steps"])


def behaviour():
    sources = {"D_update0": (TRACE_BASE, "ep190", load(TRACE_BASE / "summary.json")["episodes"]),
               "D_update2000": (B / "D/behavior", "ep2000", load(B / "D/behavior/summary.json")["episodes"])}
    beh, cal, per_ep = {}, {}, {}
    for key, (d, prefix, summ) in sources.items():
        beh[key], pooled = {}, {}
        for seed in FIXED:
            name = f"{prefix}_{seed}"
            z = np.load(d / f"{name}.npz")
            e = analyse_episode(z, 0.985)
            ep = dict(seed=seed, total_return=e["total_return"], reproduces=summ[name]["reproduces"], time_frac=e["time_frac"],
                      n_falls=len(e["falls"]), n_recoveries=len(e["recoveries"]), fwd_vel_mean=e["fwd_vel_mean"], falls=e["falls"])
            ep["label"] = base.label(ep)
            del ep["falls"]
            beh[key][seed] = ep
            per_ep[f"{key}_{seed}"] = e["cat"].tolist()
            sel = np.arange(e["T"]) <= T_MAX_CAL
            v_ema = z["v_ema_real"].astype(float) if "v_ema_real" in z else None
            for c in CATS + ["all"]:
                m = sel & ((e["cat"] == c) if c != "all" else True)
                if not m.any():
                    continue
                p = pooled.setdefault(c, dict(err=[], err_ema=[], bound=[], v=[], g=[], vdiff=[]))
                p["err"] += list(e["v"][m] - e["g_trunc"][m]); p["bound"] += list(e["bound"][m]); p["v"] += list(e["v"][m])
                p["g"] += list(e["g_trunc"][m]); p["vdiff"] += list(e["v_rep"][m] - e["v"][m])
                if v_ema is not None:
                    p["err_ema"] += list(v_ema[m] - e["g_trunc"][m])
        cal[key] = {c: dict(n=len(p["err"]), v_mean=float(np.mean(p["v"])), g_trunc_mean=float(np.mean(p["g"])),
                            signed_err_trunc=float(np.mean(p["err"])), signed_err_lower_bound=float(np.mean(p["err"]) - np.mean(p["bound"])),
                            residual_bound_max=float(np.max(p["bound"])), v_rep_minus_v_real=float(np.mean(p["vdiff"])),
                            ema_signed_err_trunc=float(np.mean(p["err_ema"])) if p["err_ema"] else None) for c, p in pooled.items()}
    prev = load(AB / "results/ablation_summary.json")
    beh["A_update0"], beh["A_update2000"] = prev["behaviour"]["update0"], prev["behaviour"]["A"]
    cal["A_update0"], cal["A_update2000"] = prev["calibration"]["update0"], prev["calibration"]["A"]
    return beh, cal, per_ep


def plots(curv, per_ep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = dict(D="#9467bd", A="#1f77b4")
    lab = dict(D="D baseline ep190, independent, no replay", A="A replay-value ep190, independent, no replay")
    panels = [("return_mean", "imagined λ-return mean"), ("value_mean", "critic value mean (imagined)"), ("value_bias", "value − λ-target"),
              ("value_target_rmse", "value-target RMSE"), ("value_target_correlation", "value-target correlation"), ("slow_value_gap", "|online − EMA|"),
              ("positive_fraction", "advantage ≥ 0 fraction"), ("advantage_std", "advantage SD"), ("value_grad_norm", "critic grad norm (pre-clip)"),
              ("actor_grad_norm", "actor grad norm (pre-clip)"), ("policy_entropy", "policy entropy"), ("kl_prior", "KL to prior"),
              ("near_upper_fraction", "action > 0.99 of range"), ("near_lower_fraction", "action < 0.01 of range"),
              ("value_grad_clipped", "critic clip frequency"), ("pretanh_std_mean", "pre-tanh std mean")]
    fig, axes = plt.subplots(4, 4, figsize=(18, 13))
    for ax, (k, title) in zip(axes.flat, panels):
        for arm in ("D", "A"):
            xs = [u for u in curv[arm] if k in curv[arm][u]]
            ax.plot(xs, [curv[arm][u][k] for u in xs], color=col[arm], label=lab[arm], marker="o", ms=3)
        ax.set_title(title, fontsize=9); ax.set_xlabel("diagnostic update", fontsize=8); ax.tick_params(labelsize=7)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Baseline follow-up D vs arm A: 100-update window means (frozen world/reward models; different source states)", fontsize=11)
    fig.tight_layout(); fig.savefig(PLOTS / "learning_curves_D_vs_A.png", dpi=80); plt.close(fig)
    colp = dict(upright="#2ca02c", crouched="#98df8a", seated="#ff7f0e", tilted="#9467bd", fallen_fwd="#d62728", fallen_back="#8c564b")
    for key, d, prefix in (("D_update2000", B / "D/behavior", "ep2000"), ("D_update0", TRACE_BASE, "ep190")):
        fig = plt.figure(figsize=(20, 2.55 * 4))
        for i, seed in enumerate(FIXED):
            z = np.load(d / f"{prefix}_{seed}.npz")
            rend = z["render"]
            idx = list(range(0, len(rend), 25))[:10]
            ax = fig.add_axes([0.01, 1 - (i + 1) / 4 + 0.03 / 4, 0.98, 0.7 / 4])
            ax.imshow(np.concatenate([rend[j] for j in idx], axis=1)); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{key} seed {seed}: return {float(z['reward'].sum()):.0f}; frames at t = " + ", ".join(str(2 * j) for j in idx), fontsize=8, loc="left")
            ax2 = fig.add_axes([0.01, 1 - (i + 1) / 4 + 0.005 / 4, 0.98, 0.025 / 4])
            ax2.imshow(np.array([[matplotlib.colors.to_rgb(colp[c]) for c in per_ep[f"{key}_{seed}"]]]), aspect="auto")
            ax2.set_xticks([]); ax2.set_yticks([])
        fig.legend(handles=[matplotlib.patches.Patch(color=c, label=k) for k, c in colp.items()], loc="lower right", ncol=6, fontsize=8)
        fig.savefig(PLOTS / f"sheet_{key}.png", dpi=55); plt.close(fig)


def main():
    ev = evaluations()
    curv = curves()
    fx = fixed_inputs()
    ix = index_checks()
    beh, cal, per_ep = behaviour()
    json.dump(dict(evaluations=ev, curves=curv, fixed_inputs=fx, index_checks=ix, behaviour=beh, calibration=cal), open(RES / "D_summary.json", "w"), indent=1, default=str)
    plots(curv, per_ep)
    f = lambda s: f"{s['mean']:.1f} / {s['median']:.1f} / " + " / ".join(f"{s[f'frac_gt_{t}']:.2f}" for t in THR)
    p = lambda x: f"{x['mean_diff']:+.1f} (median {x['median_diff']:+.1f}; SE {x['se_mean_diff']:.1f}; W/L {x['wins']}/{x['losses']})"
    md = ["# Baseline follow-up D: tables", "", "## Deterministic 30-seed evaluation (mean / median / frac > 200 / 400 / 500 / 600)", "",
          f"- D update 0 (baseline epoch 190): {f(ev['D_update0'])}; reproduces saved epoch-190 eval30: {ev['D_update0']['reproduces_saved_epoch190']} (max abs diff {ev['D_update0']['max_abs_diff_vs_saved']})",
          f"- D update 2000: {f(ev['D_update2000'])}",
          f"- A update 0 (replay-value epoch 190): {f(ev['A_update0'])}", f"- A update 2000: {f(ev['A_update2000'])}", "",
          f"- D change (2000 − 0, paired by seed): {p(ev['D_change'])}; relative {100 * ev['relative_change']['D']:+.1f}%",
          f"- A change (2000 − 0, paired by seed): {p(ev['A_change'])}; relative {100 * ev['relative_change']['A']:+.1f}%",
          f"- change(D) − change(A), paired by environment seed: {p(ev['change_D_minus_change_A'])}",
          "- threshold-fraction changes (D / A): " + "; ".join(f">{t}: {ev['threshold_changes'][f'frac_gt_{t}']['D']:+.2f} / {ev['threshold_changes'][f'frac_gt_{t}']['A']:+.2f}" for t in THR),
          "", "## Stochastic evaluation, D (30 environment seeds × 3 action seeds; noise " + ev["stochastic"][0]["noise_seed_rule"] + ")", ""]
    for u in (0, 2000):
        s = ev["stochastic"][u]
        md.append(f"- update {u}: all 90 episodes {f(s['all_episodes'])}; per-environment-seed means {f(s['per_env_seed_mean_stats'])}; mean SD across action seeds {s['action_seed_spread_mean_sd']:.1f}")
    md += [f"- stochastic change (per-seed means, 2000 − 0): {p(ev['stochastic_change'])}",
           f"- stochastic − deterministic, update 0: {p(ev['stochastic_minus_deterministic'][0])}; update 2000: {p(ev['stochastic_minus_deterministic'][2000])}",
           "", "## Four fixed seeds", "", "| arm | update | 3001 | 3004 | 3016 | 3021 | mean |", "|---|---|---|---|---|---|---|"]
    for arm in ("D", "A"):
        for u, d in ev["fixed4"][arm].items():
            md.append(f"| {arm} | {u} | " + " | ".join(f"{d['per_seed'][s]:.0f}" for s in FIXED) + f" | {d['mean']:.1f} |")
    md += ["", "## Per-seed deterministic returns (and D stochastic per-seed means)", "", "| seed | D 0 | D 2000 | A 0 | A 2000 | D stoch 0 | D stoch 2000 |", "|---|---|---|---|---|---|---|"]
    for s, d in ev["per_seed"].items():
        sp = ev["stochastic_per_seed"][s]
        md.append(f"| {s} | {d['D0']:.1f} | {d['D2000']:.1f} | {d['A0']:.1f} | {d['A2000']:.1f} | {sp[0]:.1f} | {sp[2000]:.1f} |")
    ends = [e for e in sorted(curv["D"]) if e in (100, 300, 500, 800, 1000, 1300, 1500, 1800, 2000)]
    md += ["", f"## Learning curves (100-update window means at updates {ends})", ""]
    for k in ["return_mean", "value_mean", "value_bias", "value_target_rmse", "value_target_correlation", "slow_value_gap", "positive_fraction",
              "advantage_std", "actor_grad_norm", "value_grad_norm", "actor_grad_clipped", "value_grad_clipped", "policy_entropy", "kl_prior",
              "near_lower_fraction", "near_upper_fraction", "pretanh_std_mean"]:
        md.append(f"**{k}**: " + "; ".join(f"{arm}: " + ", ".join(f"{curv[arm][e][k]:.3g}" for e in ends) for arm in ("D", "A")))
        md.append("")
    md += ["## Fixed-input critic values (online / EMA, mean over states)", "",
           "| arm | update | input | rv190 | rv240 | base240 | base190 | shared: upright+crouched | shared: seated | shared: fallen | own-start t≤300: V − G_trunc |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ("D", "A"):
        own = "base190" if arm == "D" else "rv190"
        for u, d in sorted(fx[arm].items()):
            for kind in ("real_history", "repeated_frame"):
                cells = [f"{d[f'{kind}/source_{s}/online']:.1f} / {d[f'{kind}/source_{s}/ema']:.1f}" if f"{kind}/source_{s}/online" in d else "—"
                         for s in ("rv190", "rv240", "base240", "base190")]
                md.append(f"| {arm} | {u} | {kind} | " + " | ".join(cells) + " | " + " | ".join(f"{d[f'{kind}/shared600_{q}/online']:.1f}" for q in ("posture_upright_crouched", "posture_seated", "posture_fallen"))
                          + f" | {d[f'{kind}/{own}_t<=300/online_minus_Gtrunc']:+.1f} ({own}) |")
    md += ["", "## Sampled-index checks (D)", "", "- " + json.dumps(ix), "",
           "## Behaviour (four fixed seeds; deterministic evaluation policy)", "",
           "| source | seed | return | label | upright | crouched | seated | tilted | fallen fwd | fallen back | falls | recov | fwd vel |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, d in beh.items():
        for seed, e in d.items():
            tf = e["time_frac"]
            md.append(f"| {key} | {seed} | {e['total_return']:.0f} | {e['label']} | " + " | ".join(f"{tf[c]:.2f}" for c in CATS) + f" | {e['n_falls']} | {e['n_recoveries']} | {e['fwd_vel_mean']:.2f} |")
    md += ["", "## Real-return calibration, states t ≤ 300 (V − G_trunc; deterministic evaluation policy)", "",
           "| source | posture | n | V mean | G_trunc mean | online signed | lower bound | EMA signed | V_rep − V_real |", "|---|---|---|---|---|---|---|---|---|"]
    for key, d in cal.items():
        for c in ["all"] + CATS:
            if c in d:
                x = d[c]
                ema = f"{x['ema_signed_err_trunc']:+.1f}" if x.get("ema_signed_err_trunc") is not None else "n/a"
                md.append(f"| {key} | {c} | {x['n']} | {x['v_mean']:.1f} | {x['g_trunc_mean']:.1f} | {x['signed_err_trunc']:+.1f} | {x['signed_err_lower_bound']:+.1f} | {ema} | {x['v_rep_minus_v_real']:+.2f} |")
    (RES / "D_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
