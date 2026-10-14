"""Leg-mirror exploration-barrier diagnostic for Stage 3's dive/recover gait.

Tests whether the opposite-leg coordinated action is an extreme low-probability
region of the learned Beta policy (exploration barrier) vs. reachable-but-not-
favored (optimization/reward) vs. a horizon/world-model artifact vs. hypothesis
unsupported. Read-only: no training, no checkpoint writes, no dataset writes.
Runs on cuda:0 (GPU0), never touches GPU1 (the ongoing zero-shot run).
"""
import json
import os
import sys
import pathlib

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, "/home/bigking/scfeng/LCG-pmpo-beta-k4-20260923/src")

import numpy as np
import torch
from torch.distributions.categorical import Categorical

WM_CHECKPOINT = "/home/bigking/scfeng/LCG-pmpo-beta-k4-20260923/outputs/k4-walker-100k-entropy-stage3/pmpo_diagnostic/agent_pmpo_update_100600_epoch.pt"
OUT_DIR = "/home/bigking/scfeng/LCG-pmpo-beta-k4-20260923/outputs/leg_mirror_diagnostic"
SEED = 3013  # known high-return (634.0) dive/recover gait
HORIZON = 15
M_SAMPLES = 4
N_STOCHASTIC_REF = 200

RIGHT = slice(0, 3)   # action/leg-qpos dims: right_hip, right_knee, right_ankle
LEFT = slice(3, 6)    # left_hip, left_knee, left_ankle
QPOS_RIGHT = slice(3, 6)  # qpos: rootz,rootx,rooty,right(3),left(3)
QPOS_LEFT = slice(6, 9)


def mirror_action(a):
    m = a.copy()
    m[RIGHT], m[LEFT] = a[LEFT].copy(), a[RIGHT].copy()
    return m


def mirror_qpos_legs(q):
    m = q.copy()
    m[QPOS_RIGHT], m[QPOS_LEFT] = q[QPOS_LEFT].copy(), q[QPOS_RIGHT].copy()
    return m


def build_agent(cfg, device):
    from agent import Agent, get_action_space_kwargs
    from envs import make_dm_control_env
    from hydra.utils import instantiate
    test_env = make_dm_control_env(
        num_envs=1, device=device,
        domain_name=cfg.env.test.domain_name, task_name=cfg.env.test.task_name,
        size=cfg.env.test.size, camera_id=cfg.env.test.camera_id,
        action_repeat=cfg.env.test.action_repeat, time_limit=cfg.env.test.time_limit,
    )
    action_kwargs = get_action_space_kwargs(test_env)
    agent = Agent(instantiate(cfg.agent, **action_kwargs)).to(device)
    agent.eval()
    return agent


def load_checkpoint(agent, path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    agent.load_state_dict(payload["trainer_state"]["agent"], strict=True)
    return payload["epoch"], payload["update"]


def observation(img, device):
    return torch.as_tensor(img.copy(), device=device).permute(2, 0, 1).float().div(255).mul(2).sub(1).unsqueeze(0)


class ImaginationStepper:
    def __init__(self, sampler, rew_end_model, horizon):
        self.sampler = sampler
        self.rew_end_model = rew_end_model
        self.horizon = horizon

    def set_initial(self, obs_buffer, act_buffer, hx, cx):
        self.obs_buffer = obs_buffer.clone()
        self.act_buffer = act_buffer.clone()
        self.hx_rew_end = hx.clone()
        self.cx_rew_end = cx.clone()
        self.ep_len = torch.zeros(1, dtype=torch.long, device=obs_buffer.device)

    @torch.no_grad()
    def step(self, act):
        self.act_buffer[:, -1] = act
        next_obs, _ = self.sampler.sample(self.obs_buffer, self.act_buffer)
        rew_out, logits_end, (self.hx_rew_end, self.cx_rew_end) = self.rew_end_model.predict_rew_end(
            self.obs_buffer[:, -1:], self.act_buffer[:, -1:], next_obs.unsqueeze(1),
            (self.hx_rew_end, self.cx_rew_end))
        rew = rew_out.squeeze(1)
        end = Categorical(logits=logits_end).sample().squeeze(1)
        self.ep_len += 1
        trunc = (self.ep_len >= self.horizon).long()
        self.obs_buffer = self.obs_buffer.roll(-1, dims=1)
        self.act_buffer = self.act_buffer.roll(-1, dims=1)
        self.obs_buffer[:, -1] = next_obs
        return next_obs, rew, end.float(), trunc.float()


@torch.no_grad()
def collect_trajectory(agent, cfg, device, seed, horizon=500):
    from coroutines.frame_history import FrameHistory
    from envs.dm_control_env import DMControlEnv
    kwargs = {k: v for k, v in cfg.env.test.items() if k != "type"}
    env = DMControlEnv(**kwargs, seed=int(seed))
    image, _ = env.reset(seed=int(seed))
    physics = env._dm_env.physics  # captured AFTER reset

    model = agent.actor_critic
    frame_list = [observation(image, device)]
    steps = []
    total = 0.0
    for t in range(horizon):
        policy_obs = torch.cat(frame_list[-4:] if len(frame_list) >= 4 else [frame_list[0]] * (4 - len(frame_list)) + frame_list[-4:], dim=1)
        raw = model.actor(policy_obs)
        dist = model.distribution(raw)
        action = model.sample_action(raw, deterministic=True)
        action_np = action[0].cpu().numpy()
        qpos = physics.data.qpos.copy()
        torso_height = float(physics.named.data.xpos["torso", "z"])  # Cartesian height, matches the
        # earlier validated reward-decomposition diagnostic -- NOT the same scale as qpos[0] (rootz
        # joint coordinate), which has a body-frame offset and is unsuitable as a height threshold.
        phys_state = physics.get_state().copy()

        img, rew, end, trunc, _ = env.step(action_np)
        total += float(rew)
        new_frame = observation(img, device)

        steps.append(dict(
            t=t, qpos=qpos.tolist(), torso_height=torso_height, action=action_np.tolist(), reward=float(rew),
            alpha=dist.concentration1[0].cpu().numpy().tolist(),
            beta=dist.concentration0[0].cpu().numpy().tolist(),
            phys_state=phys_state.tolist(),
            frames_window=[f.clone() for f in (frame_list[-4:] if len(frame_list) >= 4 else [frame_list[0]] * (4 - len(frame_list)) + frame_list[-4:])],
        ))
        frame_list.append(new_frame)
        if len(frame_list) > 5:
            frame_list.pop(0)
        if end or trunc:
            break
    env.close()
    return steps, total


def find_collapse_cycles(steps, height_thresh=0.9):
    heights = [s["torso_height"] for s in steps]
    runs = []
    in_run = False
    start = None
    for i, h in enumerate(heights):
        if h < height_thresh and not in_run:
            in_run = True
            start = i
        elif h >= height_thresh and in_run:
            in_run = False
            runs.append((start, i - 1))
    if in_run:
        runs.append((start, len(heights) - 1))
    return runs


def mirror_distance(qpos_a, qpos_b):
    qa = np.array(qpos_a)
    qb_mirrored = mirror_qpos_legs(np.array(qpos_b))
    # compare only the 6 leg DOF (indices 3:9); root pose (0:3) may legitimately
    # differ in absolute x-position/time even at the "same" gait phase.
    return float(np.mean((qa[3:9] - qb_mirrored[3:9]) ** 2))


def unmirrored_distance(qpos_a, qpos_b):
    qa = np.array(qpos_a)
    qb = np.array(qpos_b)
    return float(np.mean((qa[3:9] - qb[3:9]) ** 2))


def main():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from models.diffusion import DiffusionSampler
    from hydra.utils import instantiate as hydra_instantiate
    from models.actor_critic import compute_lambda_returns

    OmegaConf.register_new_resolver("eval", eval)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda:0")

    with initialize_config_dir(version_base="1.3", config_dir="/home/bigking/scfeng/LCG-pmpo-beta-k4-20260923/config"):
        cfg = compose(config_name="trainer", overrides=[
            "agent=pmpo_beta", "env=dm_control", "env.train.domain_name=walker", "env.train.task_name=walk",
            "agent.actor_critic.prior_refresh_interval=500",
            "agent.actor_critic.entropy_coef=0.01",
        ])

    agent = build_agent(cfg, device)
    epoch, update = load_checkpoint(agent, WM_CHECKPOINT, device)
    print(f"Loaded Stage-3 final checkpoint: epoch={epoch} update={update}", flush=True)
    model = agent.actor_critic

    print(f"\n=== STEP 1: collecting seed={SEED} trajectory ===", flush=True)
    steps, total_return = collect_trajectory(agent, cfg, device, SEED)
    print(f"total_return={total_return:.2f} (n_steps={len(steps)})", flush=True)

    print(f"\n=== STEP 2: mirror-alternation hypothesis test ===", flush=True)
    cycles = find_collapse_cycles(steps)
    print(f"detected {len(cycles)} collapse cycles: {cycles}", flush=True)

    mirror_dists = []
    unmirror_dists = []
    cycle_starts = [c[0] for c in cycles]
    for i in range(len(cycle_starts) - 1):
        s_a, s_b = cycle_starts[i], cycle_starts[i + 1]
        period = s_b - s_a
        for p in range(min(period, len(steps) - s_b)):
            if s_a + p >= len(steps) or s_b + p >= len(steps):
                continue
            qa = steps[s_a + p]["qpos"]
            qb = steps[s_b + p]["qpos"]
            mirror_dists.append(mirror_distance(qa, qb))
            unmirror_dists.append(unmirrored_distance(qa, qb))

    mean_mirror = float(np.mean(mirror_dists)) if mirror_dists else float("nan")
    mean_unmirror = float(np.mean(unmirror_dists)) if unmirror_dists else float("nan")
    alternation_supported = mean_mirror < mean_unmirror * 0.5  # mirrored much closer than unmirrored
    print(f"mean mirrored-distance={mean_mirror:.5f}  mean unmirrored-distance={mean_unmirror:.5f}  "
          f"n_pairs={len(mirror_dists)}", flush=True)
    print(f"ALTERNATION HYPOTHESIS {'SUPPORTED' if alternation_supported else 'NOT SUPPORTED'} "
          f"(mirrored {'<<' if alternation_supported else '>~'} unmirrored)", flush=True)

    results = dict(
        checkpoint=WM_CHECKPOINT, epoch=epoch, update=update, seed=SEED, total_return=total_return,
        n_steps=len(steps), collapse_cycles=cycles,
        mirror_test=dict(mean_mirror_distance=mean_mirror, mean_unmirror_distance=mean_unmirror,
                         n_pairs=len(mirror_dists), alternation_supported=alternation_supported),
    )

    print("\nProceeding regardless of the alternation verdict, per instruction -- match quality will be "
          "explicitly verified per-state below before any candidate is trusted as a genuine opposite-leg "
          "action.", flush=True)

    # Calibration baseline: how far apart are TYPICAL unrelated state pairs, mirrored and unmirrored?
    # Used to judge whether a given "best match" is meaningfully better than chance, not just "the best
    # of a bad lot".
    rng = np.random.RandomState(0)
    n_states = len(steps)
    random_pairs = [(rng.randint(0, n_states), rng.randint(0, n_states)) for _ in range(2000)]
    random_pairs = [(a, b) for a, b in random_pairs if abs(a - b) > 15]
    calib_mirror = [mirror_distance(steps[a]["qpos"], steps[b]["qpos"]) for a, b in random_pairs]
    calib_unmirror = [unmirrored_distance(steps[a]["qpos"], steps[b]["qpos"]) for a, b in random_pairs]
    calib_mirror_median = float(np.median(calib_mirror))
    calib_unmirror_median = float(np.median(calib_unmirror))
    print(f"\ncalibration (n={len(random_pairs)} random non-adjacent pairs): "
          f"median mirrored-distance={calib_mirror_median:.5f}  median unmirrored-distance={calib_unmirror_median:.5f}",
          flush=True)
    results["mirror_test"]["calibration_random_pair_median_mirrored"] = calib_mirror_median
    results["mirror_test"]["calibration_random_pair_median_unmirrored"] = calib_unmirror_median
    results["mirror_test"]["same_phase_repeat_median_unmirrored"] = mean_unmirror  # already computed above (cross-cycle, same phase)

    print(f"\n=== STEP 3-4: representative states + opposite-leg candidates + Beta analysis ===", flush=True)
    # pick a middle cycle, well clear of episode start/end
    mid_idx = len(cycles) // 2
    cyc_start = cycle_starts[mid_idx]
    next_cyc_start = cycle_starts[mid_idx + 1] if mid_idx + 1 < len(cycle_starts) else len(steps) - 1
    period = next_cyc_start - cyc_start
    dip_start, dip_end = cycles[mid_idx]

    phase_offsets = sorted(set([
        max(0, dip_start - cyc_start - 3),   # pre-dip ("pushing off")
        (dip_start + dip_end) // 2 - cyc_start,  # mid-dip ("low point")
        min(period - 1, dip_end - cyc_start + 4),  # post-dip ("recovering")
    ]))

    per_state_results = []
    sampler = DiffusionSampler(agent.denoiser, hydra_instantiate(cfg.world_model_env.diffusion_sampler))
    stepper = ImaginationStepper(sampler, agent.rew_end_model, HORIZON)
    gamma = cfg.agent.actor_critic.gamma
    lambda_ = cfg.agent.actor_critic.lambda_

    from torch.distributions import Beta
    action_low = agent.actor_critic.action_low
    action_high = agent.actor_critic.action_high
    width = action_high - action_low

    def make_logprob_fn(dist):
        def logprob_env(a_np):
            a = torch.tensor(a_np, device=device, dtype=torch.float32)
            unit = (a - action_low) / width
            eps = torch.finfo(unit.dtype).eps
            unit = unit.clamp(eps, 1 - eps)
            per_dim = dist.log_prob(unit) - width.log()
            return per_dim.cpu().numpy(), float(per_dim.sum().item())
        return logprob_env

    for phase_i, p in enumerate(phase_offsets):
        i = cyc_start + p
        if i >= len(steps):
            continue

        # GLOBAL search for the best empirical mirror match anywhere in the trajectory,
        # excluding a +-15 step temporal neighborhood around i (a near-identical adjacent
        # state would trivially "match" by physical continuity/inertia, not because it
        # represents a genuinely distinct opposite-leg configuration).
        best_j, best_dist = None, float("inf")
        for j in range(len(steps)):
            if abs(j - i) <= 15:
                continue
            d = mirror_distance(steps[i]["qpos"], steps[j]["qpos"])
            if d < best_dist:
                best_dist, best_j = d, j
        if best_j is None:
            continue

        # Explicit match-quality verification (this is the check the user asked for,
        # BEFORE trusting any symmetry-transformed candidate as a genuine opposite-leg action):
        # is best_dist meaningfully smaller than what we'd see for typical/random pairs, and
        # smaller than the same-phase repeat distance (how close a plain, unmirrored repeat
        # of the SAME leg pattern already is)?
        match_is_good = (best_dist < 0.3 * calib_mirror_median) and (best_dist < mean_unmirror)
        print(f"\nphase {phase_i} (step {i}): global best mirror match = step {best_j}, "
              f"mirror_dist={best_dist:.5f}  (calibration median={calib_mirror_median:.5f}, "
              f"same-phase-repeat unmirrored median={mean_unmirror:.5f})  "
              f"-> MATCH QUALITY: {'GOOD -- treat as a genuine opposite-leg reference' if match_is_good else 'POOR -- NOT a genuine opposite-leg state, do not trust the symmetry-transformed candidate as data-grounded'}",
              flush=True)

        real_action = np.array(steps[i]["action"], dtype=np.float32)
        raw_adjacent_action = np.array(steps[best_j]["action"], dtype=np.float32)          # untransformed action actually taken at the matched state
        symmetry_transformed_action = mirror_action(raw_adjacent_action)                    # that action, remapped into state i's leg-labeling
        self_mirror_action = mirror_action(real_action)                                     # same-timestep self-mirror fallback (no cross-referenced state at all)

        alpha = torch.tensor(steps[i]["alpha"], device=device)
        beta = torch.tensor(steps[i]["beta"], device=device)
        dist = Beta(alpha, beta)
        logprob_env = make_logprob_fn(dist)

        candidates = dict(
            real_action=real_action,
            raw_adjacent_action=raw_adjacent_action,
            symmetry_transformed_action=symmetry_transformed_action,
            self_mirror_action=self_mirror_action,
        )
        cand_stats = {}
        mean_action_unit = dist.mean
        mean_action_env = (action_low + mean_action_unit * width).cpu().numpy()
        std_unit = dist.variance.sqrt()
        std_env = (std_unit * width).cpu().numpy()
        mean_unit_np = mean_action_unit.cpu().numpy()
        std_unit_np = std_unit.cpu().numpy()

        # empirical reference distribution: normal stochastic samples from this exact state
        stoch_logprobs = []
        for _ in range(N_STOCHASTIC_REF):
            sample_unit = dist.sample()
            per_dim = dist.log_prob(sample_unit) - width.log()
            stoch_logprobs.append(float(per_dim.sum().item()))
        stoch_logprobs = np.array(stoch_logprobs)

        for name, a_np in candidates.items():
            per_dim_lp, joint_lp = logprob_env(a_np)
            a_unit = ((torch.tensor(a_np, device=device) - action_low) / width).cpu().numpy()
            z_scores = (a_unit - mean_unit_np) / np.clip(std_unit_np, 1e-8, None)
            percentile = float((stoch_logprobs < joint_lp).mean() * 100)
            cand_stats[name] = dict(
                action=a_np.tolist(), per_dim_logprob=per_dim_lp.tolist(), joint_logprob=joint_lp,
                z_scores_per_dim=z_scores.tolist(), dist_from_policy_mean=float(np.linalg.norm(a_unit - mean_unit_np)),
                dist_from_real_action=float(np.linalg.norm(a_np - real_action)),
                percentile_vs_stochastic_samples=percentile,
            )
            print(f"    [{name}] joint_logprob={joint_lp:7.2f}  percentile_vs_samples={percentile:5.1f}%  "
                  f"z_scores={np.round(z_scores, 2)}", flush=True)

        state_result = dict(
            phase=phase_i, step_i=i, matched_step_j=best_j, mirror_match_distance=best_dist,
            match_is_good=match_is_good,
            alpha=steps[i]["alpha"], beta=steps[i]["beta"],
            mean_action_env=mean_action_env.tolist(), std_action_env=std_env.tolist(),
            candidates=cand_stats,
            stochastic_logprob_mean=float(stoch_logprobs.mean()), stochastic_logprob_std=float(stoch_logprobs.std()),
        )

        # STEP 5: world-model branch test
        if i >= 3:
            obs_buffer0 = torch.stack(steps[i]["frames_window"], dim=1)
            act_hist = [torch.tensor(steps[k]["action"], device=device, dtype=torch.float32).unsqueeze(0)
                       for k in range(max(0, i - 3), i)]
            while len(act_hist) < 3:
                act_hist.insert(0, act_hist[0].clone())
            placeholder = torch.zeros(1, 6, device=device)

            def rollout(forced_first_action_np, base_seed):
                act_buffer0 = torch.stack(act_hist + [placeholder], dim=1)
                hx, cx = agent.rew_end_model.predict_rew_end(obs_buffer0[:, :-1], act_buffer0[:, :-1], obs_buffer0[:, 1:])[2]
                targets_list = []
                for m in range(M_SAMPLES):
                    torch.manual_seed(base_seed * 1000 + m)
                    stepper.set_initial(obs_buffer0, act_buffer0, hx, cx)
                    frame_list_ = [obs_buffer0[:, k] for k in range(obs_buffer0.shape[1])]

                    def policy_state():
                        return torch.cat(frame_list_[-4:], dim=1)

                    rews, ends, truncs, vnexts = [], [], [], []
                    act = torch.tensor(forced_first_action_np, device=device, dtype=torch.float32).unsqueeze(0)
                    dead = False
                    for t in range(HORIZON):
                        if dead:
                            rews.append(torch.zeros(1, device=device)); ends.append(torch.ones(1, device=device))
                            truncs.append(torch.zeros(1, device=device)); vnexts.append(torch.zeros(1, device=device))
                            continue
                        next_obs, rew, end, trunc = stepper.step(act)
                        rews.append(rew); ends.append(end); truncs.append(trunc)
                        frame_list_.append(next_obs)
                        if len(frame_list_) > 4:
                            frame_list_.pop(0)
                        v_next = model.value(policy_state()).squeeze(-1)
                        vnexts.append(v_next)
                        if bool(end.item()):
                            dead = True
                        elif t < HORIZON - 1:
                            raw = model.actor(policy_state())
                            act = model.sample_action(raw, deterministic=False)
                    rew_t = torch.stack(rews, dim=1); end_t = torch.stack(ends, dim=1)
                    trunc_t = torch.stack(truncs, dim=1); vboot_t = torch.stack(vnexts, dim=1)
                    targets = compute_lambda_returns(rew_t, end_t, trunc_t, vboot_t, gamma, lambda_)
                    targets_list.append((targets[0, 0].item(), [r.item() for r in rews]))
                return targets_list

            base_seed_for_pair = i * 7919 + phase_i
            branch_test = {}
            for name in ("real_action", "symmetry_transformed_action", "self_mirror_action"):
                rollouts = rollout(candidates[name], base_seed_for_pair)
                returns = [r[0] for r in rollouts]
                branch_test[name] = dict(
                    R_lambda_mean=float(np.mean(returns)), R_lambda_samples=returns,
                    reward_trajectory_example=rollouts[0][1],
                )
            state_result["world_model_branch_test"] = branch_test
            print(f"  world-model branch (H={HORIZON}, shared RNG across conditions): "
                  f"normal={branch_test['real_action']['R_lambda_mean']:.2f}  "
                  f"symmetry_transformed(match_is_good={match_is_good})={branch_test['symmetry_transformed_action']['R_lambda_mean']:.2f}  "
                  f"self_mirror={branch_test['self_mirror_action']['R_lambda_mean']:.2f}", flush=True)

        per_state_results.append(state_result)

    results["per_state"] = per_state_results
    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {OUT_DIR}/results.json", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
