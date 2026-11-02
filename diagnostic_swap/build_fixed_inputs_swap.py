"""Fixed real-state inputs for the component swap, from deterministic behaviour traces of the two source controllers at
their epoch-90 checkpoints (recorded before learning by behavior_arm.py; the same stage-4 recording as before):
  ctrl0_ep90  seed-0 controller h10emas04689e641 at epoch 90      ctrl1_ep90  seed-1 controller h10ema5bc11033 at epoch 90
Fixed seeds 3001, 3004, 3016, 3021; states t = 0, 10, ..., 490; real-history and repeated-frame inputs (uint8; convert
/255*2-1), posture (stage-5 rules), the trace's own V_real/V_rep (update-0 identity check), and G_trunc of the trace
policy with its residual bound. Same construction as build_fixed_inputs.py.
usage: python build_fixed_inputs_swap.py OUT.pt NAME=DIR:PREFIX [NAME=DIR:PREFIX ...]"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

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
    sources = {}
    for spec in sys.argv[2:]:
        name, rest = spec.split("=", 1)
        d, prefix = rest.rsplit(":", 1)
        sources[name] = (Path(d), prefix)
    real, rep, files = [], [], {}
    meta = dict(source=[], seed=[], t=[], posture=[], v_real_trace=[], v_rep_trace=[], g_trunc=[], bound=[])
    for src, (d, prefix) in sources.items():
        for seed in SEEDS:
            f = d / f"{prefix}_{seed}.npz"
            files[f"{src}/{seed}"] = dict(path=str(f), sha256=hashlib.sha256(f.read_bytes()).hexdigest())
            z = np.load(f)
            obs = torch.as_tensor(z["obs"]).permute(0, 3, 1, 2)
            r = z["reward"].astype(np.float64)
            T = len(r)
            disc = GAMMA ** np.arange(T)
            for t in range(0, T, STRIDE):
                real.append(obs[[max(0, t - 3 + j) for j in range(4)]].flatten(0, 1))
                rep.append(obs[t].repeat(4, 1, 1))
                h, u, pitch, _ = z["phys"][t]
                meta["source"].append(src); meta["seed"].append(seed); meta["t"].append(t); meta["posture"].append(categorize(h, u, pitch))
                meta["v_real_trace"].append(float(z["v_real"][t])); meta["v_rep_trace"].append(float(z["v_rep"][t]))
                meta["g_trunc"].append(float(np.sum(r[t:] * disc[:T - t]))); meta["bound"].append(float(GAMMA ** (T - t) * R_MAX / (1 - GAMMA)))
    torch.save(dict(real_history=torch.stack(real), repeated_frame=torch.stack(rep), files=files, gamma=GAMMA, stride=STRIDE,
                    note="uint8 stacks (N, 12, 64, 64); convert x.float()/255*2-1", **meta), out)
    print(json.dumps(dict(n=len(real), sources={s: meta["source"].count(s) for s in sources},
                          postures={c: meta["posture"].count(c) for c in sorted(set(meta["posture"]))}), indent=1))


if __name__ == "__main__":
    main()
