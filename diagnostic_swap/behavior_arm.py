"""Behaviour traces at a diagnostic checkpoint: the 2026-10-27 stage-4 tool (s4_behavior.py) with the checkpoint loader
changed to ablation checkpoints and the EMA critic's values recorded alongside the online critic's. Deterministic
actions, 500 steps, fixed seeds 3001, 3004, 3016, 3021; the episode recording is otherwise unchanged (reward, action,
deterministic action on the repeated-frame input, V on the real-history and repeated-frame stacks, physics rows and
state, 64x64 observations, 128x128 render every 2 steps). Each episode's return must equal the arm's eval30 return
for that seed at the same checkpoint.
usage: python behavior_arm.py --source-run DIR --checkpoint CKPT --eval30 JSON --out DIR --label NAME"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402
from eval_arm import load_model  # noqa: E402

FIXED = [3001, 3004, 3016, 3021]


def physics_row(env):
    ph = env._dm_env.physics  # live handle
    return [float(ph.torso_height()), float(ph.torso_upright()), float(ph.named.data.qpos["rooty"]), float(ph.horizontal_velocity())]


@torch.no_grad()
def episode(model, cfg, seed, device):
    from coroutines.frame_history import FrameHistory
    from envs.dm_control_env import DMControlEnv
    kwargs = {k: v for k, v in cfg.env.test.items() if k != "type"}
    env = DMControlEnv(**kwargs, seed=int(seed))
    rec = {k: [] for k in ("reward", "action", "action_rep", "v_real", "v_rep", "v_ema_real", "v_ema_rep", "phys", "state", "obs", "render")}
    try:
        image, _ = env.reset(seed=int(seed))

        def observation(img):
            return torch.as_tensor(img.copy(), device=device).permute(2, 0, 1).float().div(255).mul(2).sub(1).unsqueeze(0)

        o = observation(image)
        history = FrameHistory(o, model.frame_stack)
        for step in range(500):
            ph = env._dm_env.physics
            rec["state"].append(ph.get_state().copy())
            rec["phys"].append(physics_row(env))
            rec["obs"].append(image.copy())
            if step % 2 == 0:
                rec["render"].append(ph.render(height=128, width=128, camera_id=env.camera_id).copy())
            state = history.state
            raw, v = model.predict_act_value(state)
            action = model.sample_action(raw, deterministic=True)
            rep = o.repeat(1, model.frame_stack, 1, 1)
            raw_r, v_r = model.predict_act_value(rep)
            a_rep = model.sample_action(raw_r, deterministic=True)
            rec["v_real"].append(float(v[0]))
            rec["v_rep"].append(float(v_r[0]))
            rec["v_ema_real"].append(float(model.slow_value(state).squeeze(-1)[0]))
            rec["v_ema_rep"].append(float(model.slow_value(rep).squeeze(-1)[0]))
            rec["action"].append(action[0].cpu().numpy())
            rec["action_rep"].append(a_rep[0].cpu().numpy())
            image, reward, end, trunc, _ = env.step(action[0].cpu().numpy())
            rec["reward"].append(float(reward))
            if end or trunc:
                break
            o = observation(image)
            history.advance(o, torch.tensor([False], device=device))
        ph = env._dm_env.physics
        rec["final_state"] = ph.get_state().copy()
        rec["final_obs"] = image.copy()
        rec["final_phys"] = physics_row(env)
        o = observation(image)
        rec["v_final_real"] = float(model.predict_act_value(history.successor(o))[1][0])
        rec["v_ema_final_real"] = float(model.slow_value(history.successor(o)).squeeze(-1)[0])
    finally:
        env.close()
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-run", required=True, type=Path)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval30", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--label", required=True)
    args = p.parse_args()
    device = torch.device("cuda:0")
    cfg, model = load_model(args.source_run, device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck["actor_critic"], strict=True)
    saved = json.loads(args.eval30.read_text())
    saved_ret = dict(zip(saved["seeds"], saved["returns"]))
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"label": args.label, "episodes": {}}
    summary.update(checkpoint=args.checkpoint, diagnostic_update=ck["relative_update"], gamma=float(cfg.agent.actor_critic.gamma), seeds=FIXED)
    for seed in FIXED:
        name = f"{args.label}_{seed}"
        if (args.out / f"{name}.npz").exists() and name in summary["episodes"]:
            continue
        rec = episode(model, cfg, seed, device)
        total = float(np.sum(rec["reward"]))
        np.savez_compressed(args.out / f"{name}.npz", **{k: np.array(rec[k], np.float32) for k in ("reward", "action", "action_rep", "v_real", "v_rep", "v_ema_real", "v_ema_rep", "phys")},
                            state=np.array(rec["state"], np.float64), obs=np.array(rec["obs"], np.uint8), render=np.array(rec["render"], np.uint8),
                            final_state=rec["final_state"], final_obs=rec["final_obs"], final_phys=np.array(rec["final_phys"], np.float32),
                            v_final_real=rec["v_final_real"], v_ema_final_real=rec["v_ema_final_real"])
        summary["episodes"][name] = dict(epoch=ck["relative_update"], seed=seed, kind="fixed", checkpoint=args.checkpoint, total_return=total,
                                         saved_eval30_return=saved_ret[seed], reproduces=bool(abs(total - saved_ret[seed]) < 1e-6), length=len(rec["reward"]))
        summary_path.write_text(json.dumps(summary, indent=1))
        print(f"{name} return {total:.2f} eval30 {saved_ret[seed]:.2f} reproduces {abs(total - saved_ret[seed]) < 1e-6}", flush=True)


if __name__ == "__main__":
    main()
