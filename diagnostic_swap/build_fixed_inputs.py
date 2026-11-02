"""Fixed real-state inputs for the ablation, from existing deterministic behaviour traces (read-only):
  rv190   replay-value run h10emarvs1f62ae16d, epoch 190 (the shared starting checkpoint's own behaviour)
  rv240   replay-value run, epoch 240 (late seated/lying states)
  base240 H10 scalar-EMA baseline h10ema5bc11033, epoch 240 (dive-and-recover gait states)
Fixed seeds 3001, 3004, 3016, 3021; states t = 0, 10, ..., 490. For each state two inputs:
  real_history    the production stack at t: frames max(0, t-3) .. t (FrameHistory: a reset repeats its first frame)
  repeated_frame  the frame at t repeated four times (the input imagination starts from)
stored as uint8 (converted exactly like evaluation: /255*2-1). Also stored: posture category (stage-5 rules), the
trace's saved V_real/V_rep (identity check at update 0 for rv190), and the truncated discounted real return
G_trunc(t) of the TRACE policy with its residual bound. usage: python build_fixed_inputs.py OUT.pt"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

DIAG = Path("/home/bigking/scfeng/LCG-replayvalue-diagnostic-20261027/data/behavior")
SOURCES = dict(rv190=("rv1", 190), rv240=("rv1", 240), base240=("h10s1", 240))
# Optional (baseline follow-up D): append base190 = baseline epoch 190, the D starting checkpoint's own behaviour.
if len(sys.argv) > 2 and sys.argv[2] == "with-base190":
    SOURCES["base190"] = ("h10s1", 190)
SEEDS = [3001, 3004, 3016, 3021]
GAMMA, R_MAX, STRIDE = 0.985, 2.0, 10


def categorize(h, u, p):
    if u >= 0.7:
        return "upright" if h >= 0.9 else "crouched" if h >= 0.6 else "seated"
    if h >= 0.6:
        return "tilted"
    return "fallen_fwd" if p > 0 else "fallen_back"


def main():
    out = Path(sys.argv[1])
    real, rep, meta = [], [], dict(source=[], seed=[], t=[], posture=[], v_real_trace=[], v_rep_trace=[], g_trunc=[], bound=[])
    files = {}
    for src, (key, epoch) in SOURCES.items():
        for seed in SEEDS:
            f = DIAG / key / f"ep{epoch}_{seed}.npz"
            files[f"{src}/{seed}"] = dict(path=str(f), sha256=hashlib.sha256(f.read_bytes()).hexdigest())
            z = np.load(f)
            obs = torch.as_tensor(z["obs"]).permute(0, 3, 1, 2)  # (T, 3, 64, 64) uint8
            r = z["reward"].astype(np.float64)
            T = len(r)
            disc = GAMMA ** np.arange(T)
            for t in range(0, T, STRIDE):
                idx = [max(0, t - 3 + j) for j in range(4)]
                real.append(obs[idx].flatten(0, 1))
                rep.append(obs[t].repeat(4, 1, 1))
                h, u, pitch, _ = z["phys"][t]
                meta["source"].append(src); meta["seed"].append(seed); meta["t"].append(t)
                meta["posture"].append(categorize(h, u, pitch))
                meta["v_real_trace"].append(float(z["v_real"][t])); meta["v_rep_trace"].append(float(z["v_rep"][t]))
                meta["g_trunc"].append(float(np.sum(r[t:] * disc[:T - t]))); meta["bound"].append(float(GAMMA ** (T - t) * R_MAX / (1 - GAMMA)))
    data = dict(real_history=torch.stack(real), repeated_frame=torch.stack(rep), files=files, gamma=GAMMA, stride=STRIDE,
                note="uint8 stacks (N, 12, 64, 64); convert x.float()/255*2-1", **meta)
    torch.save(data, out)
    print(json.dumps(dict(n=len(real), sources={s: meta["source"].count(s) for s in SOURCES},
                          postures={c: meta["posture"].count(c) for c in sorted(set(meta["posture"]))}), indent=1))


if __name__ == "__main__":
    main()
