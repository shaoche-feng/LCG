"""Suitability of the epoch-90 (50k) starting point: saved eval30 milestones of both source runs around 40-60k and the
posture make-up of each controller's epoch-90 behaviour traces (stage-5 categories, s11 labels).
usage: python suitability.py SWAP_DIR"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/bigking/scfeng/LCG-replayvalue-diagnostic-20261027/scripts")
from s5_behavior_analysis import CATS, analyse_episode  # noqa: E402

O = Path(sys.argv[1])
E = Path("/home/bigking/scfeng/LCG-stage3-gaussian-h10-emacritic/outputs")
RUNS = dict(seed0=E / "k4-walker-100k-entropy-stage3-seed0-gaussian-h10-emacritic-20261022", seed1=E / "k4-walker-100k-entropy-stage3-seed1-gaussian-h10-emacritic-20261021")


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


res = {"milestones": {}, "behaviour_epoch90": {}}
for k, R in RUNS.items():
    res["milestones"][k] = {}
    for m in ("30k", "40k", "50k", "60k", "70k", "100k_final"):
        d = json.loads((R / f"eval30/eval30_{m}.json").read_text())
        res["milestones"][k][m] = {x: round(d[x], 3) for x in ("mean", "median", "frac_gt_200", "frac_gt_400", "frac_gt_500", "frac_gt_600")}
    d = O / f"behavior_update0/ctrl{k[-1]}"
    summ = json.loads((d / "summary.json").read_text())["episodes"]
    eps = {}
    for name, info in summ.items():
        e = analyse_episode(np.load(d / f"{name}.npz"), 0.985)
        ep = dict(seed=info["seed"], total_return=round(e["total_return"], 1), reproduces_saved_eval30=info["reproduces"],
                  time_frac={c: round(e["time_frac"][c], 2) for c in CATS}, n_falls=len(e["falls"]), fwd_vel_mean=round(e["fwd_vel_mean"], 2))
        ep["label"] = label(ep)
        eps[info["seed"]] = ep
    res["behaviour_epoch90"][k] = dict(episodes=eps, mean_time_frac={c: round(float(np.mean([e["time_frac"][c] for e in eps.values()])), 2) for c in CATS})
(O / "records/suitability_epoch90.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
