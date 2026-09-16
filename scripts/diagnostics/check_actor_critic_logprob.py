"""Reusable runtime diagnostic for the continuous-action actor-critic's log-probability.

Loads a checkpoint, builds the *actual* imagined WorldModelEnv actor-training path (the exact
code path `forward()` uses -- not a reimplementation), runs several `forward()` calls WITHOUT
ever calling `optimizer.step()`, and reports the policy-loss and entropy-estimate log_prob
distributions separately, plus gradient norms split by head (mean vs log_std) and by loss term
(policy vs entropy).

This is the script used to (a) originally diagnose the log_prob explosion (see the actor_critic.py
module docstrings/comments for the root cause), and (b) validate the fix that carries the true
sampled `z` through the rollout instead of reconstructing it via `atanh(action)`. Kept in the repo
as a reusable tool rather than a one-off throwaway, per the investigation this accompanies.

Usage (from the repo root, with the `lcg` conda env active):
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<n> python scripts/diagnostics/check_actor_critic_logprob.py \\
        <path_to_checkpoint.pt> <path_to_dataset/train> [--n-forward-calls 34]

No training config is modified and no optimizer step is ever taken -- this only reads a
checkpoint and computes forward/backward passes for inspection.
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

OmegaConf.register_new_resolver("eval", eval, replace=True)

from agent import Agent, get_action_space_kwargs  # noqa: E402
from data import BatchSampler, collate_segments_to_batch, Dataset  # noqa: E402
from envs import WorldModelEnv, make_dm_control_env  # noqa: E402


def summarize(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    if arr.size == 0:
        print(f"{name}: n=0 (no samples captured)")
        return
    print(
        f"{name}: n={len(arr)} min={arr.min():.4f} p1={np.percentile(arr, 1):.4f} "
        f"median={np.median(arr):.4f} p99={np.percentile(arr, 99):.4f} max={arr.max():.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("--n-forward-calls", type=int, default=34)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with initialize(version_base="1.3", config_path="../../config"):
        cfg = compose(config_name="trainer", overrides=[
            "env=dm_control", "env.train.domain_name=walker", "env.train.task_name=walk",
            "training.compile_wm=False",
        ])

    env_kwargs = {k: v for k, v in cfg.env.train.items() if k != "type"}
    probe_env = make_dm_control_env(num_envs=1, device=device, **env_kwargs)
    action_kwargs = get_action_space_kwargs(probe_env)

    agent = Agent(instantiate(cfg.agent, **action_kwargs)).to(device)
    agent.load(args.ckpt_path)
    agent.eval()
    print(f"loaded {args.ckpt_path}", flush=True)

    ac = agent.actor_critic

    # ---- instrumentation: tag each _tanh_affine_log_prob call as "policy" or "entropy",
    # wrapping (not reimplementing) the real methods, and confirm atanh() is never called. ----
    import torch as _torch

    atanh_call_count = {"n": 0}
    _orig_atanh = _torch.atanh

    def counting_atanh(*a, **kw):
        atanh_call_count["n"] += 1
        return _orig_atanh(*a, **kw)

    _torch.atanh = counting_atanh

    records = {"policy": [], "entropy": []}
    _orig_tanh_affine = ac._tanh_affine_log_prob
    _orig_entropy_estimate = ac._continuous_entropy_estimate
    _context = {"tag": "policy"}

    def patched_tanh_affine(mean, std, z):
        out = _orig_tanh_affine(mean, std, z)
        records[_context["tag"]].append(out.detach().cpu().numpy().flatten())
        return out

    def patched_entropy_estimate(dist_params):
        _context["tag"] = "entropy"
        out = _orig_entropy_estimate(dist_params)
        _context["tag"] = "policy"
        return out

    ac._tanh_affine_log_prob = patched_tanh_affine
    ac._continuous_entropy_estimate = patched_entropy_estimate

    # ---- additionally capture raw (pre-clamp) mean/log_std, clamped std, and sampled actions,
    # so we can report std/log_std percentiles, raw_log_std<-5 fraction, and action saturation. ----
    raw_mean_all, raw_logstd_all, clamped_std_all = [], [], []
    _orig_split = ac._split_dist_params

    def patched_split(dist_params):
        raw_mean, raw_log_std = dist_params.chunk(2, dim=-1)
        raw_mean_all.append(raw_mean.detach().cpu().numpy().flatten())
        raw_logstd_all.append(raw_log_std.detach().cpu().numpy().flatten())
        mean, log_std = _orig_split(dist_params)
        clamped_std_all.append(log_std.exp().detach().cpu().numpy().flatten())
        return mean, log_std

    ac._split_dist_params = patched_split

    action_saturation_all = []
    _orig_sample_action = ac.sample_action

    def patched_sample_action(dist_params, deterministic=False):
        action, z = _orig_sample_action(dist_params, deterministic)
        if action is not None:
            scale = 0.5 * (ac.action_high - ac.action_low)
            normalized = (action - ac.action_low) / scale - 1.0  # in [-1, 1]
            action_saturation_all.append(normalized.detach().abs().cpu().numpy().flatten())
        return action, z

    ac.sample_action = patched_sample_action

    # ---- build the actual imagined WorldModelEnv actor-training path ----
    dataset = Dataset(args.dataset_path, "train_dataset", cache_in_ram=True)
    dataset.load_from_default_path()
    print(f"dataset loaded: {dataset.num_episodes} episodes, {dataset.num_steps} steps", flush=True)

    c_ac = cfg.actor_critic.training
    seq_len_conditioning = cfg.agent.denoiser.inner_model.num_steps_conditioning
    bs = BatchSampler(dataset, 0, 1, c_ac.batch_size, seq_len_conditioning, list(c_ac.sample_weights))
    dl = DataLoader(dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch)

    wm_cfg = instantiate(cfg.world_model_env)
    wm_env = WorldModelEnv(agent.denoiser, agent.rew_end_model, dl, wm_cfg)

    sigma_distribution_cfg = instantiate(cfg.denoiser.sigma_distribution)
    actor_critic_loss_cfg = instantiate(cfg.actor_critic.actor_critic_loss)
    agent.setup_training(sigma_distribution_cfg, actor_critic_loss_cfg, wm_env)

    from models.actor_critic import compute_lambda_returns

    def global_grad_norm(params):
        sq = None
        for p in params:
            if p.grad is None:
                continue
            s = p.grad.detach().float().pow(2).sum()
            sq = s if sq is None else sq + s
        return (sq.sqrt().item()) if sq is not None else 0.0

    max_grad_norm = c_ac.max_grad_norm  # cfg.actor_critic.training.max_grad_norm, the actual value
    # trainer.py's clip_grad_norm_ call uses -- production clips ac.parameters() as a whole, not
    # any single head, so that's what's reproduced here.

    grad_norms_mean_head, grad_norms_logstd_head = [], []
    pre_clip_norms, post_clip_norms, clip_activated = [], [], []

    for _ in range(args.n_forward_calls):
        ac.zero_grad(set_to_none=True)
        loss, metrics = ac.forward()  # NEVER followed by optimizer.step() -- inspection only.
        loss.backward()

        W = ac.actor_linear.weight.grad
        if W is not None:
            n_out = W.shape[0]
            grad_norms_mean_head.append(W[: n_out // 2].norm().item())
            grad_norms_logstd_head.append(W[n_out // 2 :].norm().item())

        # Exactly trainer.py's clipping call (torch.nn.utils.clip_grad_norm_(model.parameters(),
        # cfg.max_grad_norm)): this clips ac's gradients IN PLACE and returns the pre-clip norm.
        pre = torch.nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
        pre_clip_norms.append(float(pre))
        clip_activated.append(float(pre) > max_grad_norm)
        post_clip_norms.append(global_grad_norm(ac.parameters()))  # measured directly, not assumed

    # Policy-loss vs entropy-loss (unweighted and weighted), each via its OWN fresh, independent
    # forward pass -- so none of these three measurements share a rollout or computation graph.
    grad_norms_policy_loss = []
    grad_norms_entropy_unweighted, grad_norms_entropy_weighted = [], []
    head_norms_policy_mean, head_norms_policy_logstd = [], []
    head_norms_entropy_mean, head_norms_entropy_logstd = [], []
    c = ac.loss_cfg

    def head_norms(W):
        n_out = W.shape[0]
        return W[: n_out // 2].norm().item(), W[n_out // 2 :].norm().item()

    for _ in range(min(10, args.n_forward_calls)):
        ac.zero_grad(set_to_none=True)
        _, act, rew, end, trunc, logits_act, val, val_bootstrap, z, infos = ac.env_loop.send(c.backup_every)
        log_prob, entropy_per_sample = ac.log_prob_and_entropy(logits_act, act, z)
        lambda_returns = compute_lambda_returns(
            rew, end, trunc, val_bootstrap, c.gamma, c.lambda_, continuous_reward=ac.continuous_reward
        )
        loss_actions = (-log_prob * (lambda_returns - val).detach()).mean()
        loss_actions.backward()
        grad_norms_policy_loss.append(global_grad_norm(ac.parameters()))
        if ac.actor_linear.weight.grad is not None:
            m, l = head_norms(ac.actor_linear.weight.grad)
            head_norms_policy_mean.append(m)
            head_norms_policy_logstd.append(l)

        ac.zero_grad(set_to_none=True)
        _, act, rew, end, trunc, logits_act, val, val_bootstrap, z, infos = ac.env_loop.send(c.backup_every)
        log_prob, entropy_per_sample = ac.log_prob_and_entropy(logits_act, act, z)
        entropy_unweighted = entropy_per_sample.mean()
        entropy_unweighted.backward()
        grad_norms_entropy_unweighted.append(global_grad_norm(ac.parameters()))
        if ac.actor_linear.weight.grad is not None:
            m, l = head_norms(ac.actor_linear.weight.grad)
            head_norms_entropy_mean.append(m)
            head_norms_entropy_logstd.append(l)

        ac.zero_grad(set_to_none=True)
        _, act, rew, end, trunc, logits_act, val, val_bootstrap, z, infos = ac.env_loop.send(c.backup_every)
        log_prob, entropy_per_sample = ac.log_prob_and_entropy(logits_act, act, z)
        loss_entropy = -c.weight_entropy_loss * entropy_per_sample.mean()
        loss_entropy.backward()
        grad_norms_entropy_weighted.append(global_grad_norm(ac.parameters()))

    policy_raw = np.concatenate(records["policy"]) if records["policy"] else np.array([])
    entropy_raw = np.concatenate(records["entropy"]) if records["entropy"] else np.array([])

    print(f"\n=== Direct policy log_prob (n={len(policy_raw)}) ===")
    summarize("policy log_prob", policy_raw)
    if len(policy_raw) > 0:
        print(f"fraction non-finite: {(~np.isfinite(policy_raw)).mean():.6f}")
        print(f"fraction that would previously have been < -50: {(policy_raw < -50).mean():.4f}")
        print(f"fraction that would previously have been > +20: {(policy_raw > 20).mean():.4f}")

    print(f"\n=== Direct entropy log_prob (n={len(entropy_raw)}) ===")
    summarize("entropy log_prob", entropy_raw)
    if len(entropy_raw) > 0:
        print(f"fraction non-finite: {(~np.isfinite(entropy_raw)).mean():.6f}")
        print(f"fraction that would previously have been < -50: {(entropy_raw < -50).mean():.4f}")
        print(f"fraction that would previously have been > +20: {(entropy_raw > 20).mean():.4f}")

    print(f"\nmean-head grad norm (actor_linear rows only): mean={np.mean(grad_norms_mean_head):.4f}  max={np.max(grad_norms_mean_head):.4f}")
    print(f"log_std-head grad norm (actor_linear rows only): mean={np.mean(grad_norms_logstd_head):.4f}  max={np.max(grad_norms_logstd_head):.4f}")

    print(f"\n=== Gradient clipping (matches trainer.py's clip_grad_norm_(ac.parameters(), max_grad_norm)) ===")
    print(f"configured max_grad_norm (cfg.actor_critic.training.max_grad_norm): {max_grad_norm}")
    pre = np.array(pre_clip_norms)
    post = np.array(post_clip_norms)
    print(f"pre-clip global grad norm:  mean={pre.mean():.4f} median={np.median(pre):.4f} "
          f"p90={np.percentile(pre, 90):.4f} max={pre.max():.4f}")
    print(f"post-clip global grad norm: mean={post.mean():.4f} median={np.median(post):.4f} "
          f"p90={np.percentile(post, 90):.4f} max={post.max():.4f}")
    print(f"fraction of batches where clipping activated: {np.mean(clip_activated):.4f} "
          f"({sum(clip_activated)}/{len(clip_activated)})")

    print(f"\n=== Policy vs entropy gradient norms (global, ac.parameters(), fresh independent forward passes) ===")
    print(f"weight_entropy_loss (cfg value): {c.weight_entropy_loss}")
    pol = np.array(grad_norms_policy_loss)
    ent_u = np.array(grad_norms_entropy_unweighted)
    ent_w = np.array(grad_norms_entropy_weighted)
    print(f"policy-loss grad norm:              mean={pol.mean():.4f}  max={pol.max():.4f}")
    print(f"unweighted entropy grad norm:        mean={ent_u.mean():.4f}  max={ent_u.max():.4f}")
    print(f"weighted entropy grad norm:          mean={ent_w.mean():.4f}  max={ent_w.max():.4f}")
    ratio = ent_w.mean() / ent_u.mean() if ent_u.mean() > 0 else float("nan")
    print(f"weighted/unweighted ratio: {ratio:.6f}  (expected ~= weight_entropy_loss = {c.weight_entropy_loss}, "
          f"since gradient scales linearly with a scalar loss multiplier)")

    print(f"\n=== Policy vs entropy gradients, split by actor_linear head (mean vs log_std rows) ===")
    print(f"policy-loss  -> mean head:    mean={np.mean(head_norms_policy_mean):.4f}  max={np.max(head_norms_policy_mean):.4f}")
    print(f"policy-loss  -> log_std head: mean={np.mean(head_norms_policy_logstd):.4f}  max={np.max(head_norms_policy_logstd):.4f}")
    print(f"entropy(unweighted) -> mean head:    mean={np.mean(head_norms_entropy_mean):.4f}  max={np.max(head_norms_entropy_mean):.4f}")
    print(f"entropy(unweighted) -> log_std head: mean={np.mean(head_norms_entropy_logstd):.4f}  max={np.max(head_norms_entropy_logstd):.4f}")

    print(f"\ntotal torch.atanh() calls during the entire run: {atanh_call_count['n']} "
          f"(must be 0 for loss_actions to be confirmed atanh-free)")

    print(f"\n=== std / log_std / action saturation ===")
    raw_logstd = np.concatenate(raw_logstd_all) if raw_logstd_all else np.array([])
    clamped_std = np.concatenate(clamped_std_all) if clamped_std_all else np.array([])
    raw_mean = np.concatenate(raw_mean_all) if raw_mean_all else np.array([])
    action_sat = np.concatenate(action_saturation_all) if action_saturation_all else np.array([])
    summarize("std (post-clamp)", clamped_std)
    summarize("raw log_std (pre-clamp)", raw_logstd)
    summarize("raw mean (pre-clamp)", raw_mean)
    if len(raw_logstd) > 0:
        print(f"fraction of raw log_std < -5 (LOG_STD_MIN, the dead-zone threshold): "
              f"{(raw_logstd < -5).mean():.4f}")
    if len(action_sat) > 0:
        print(f"action saturation |normalized action| (0=center, 1=at bound): "
              f"mean={action_sat.mean():.4f} median={np.median(action_sat):.4f} "
              f"p99={np.percentile(action_sat, 99):.4f} max={action_sat.max():.4f}")
        print(f"fraction of actions with |normalized| > 0.99 (near-saturated): "
              f"{(action_sat > 0.99).mean():.4f}")

    print(f"\n=== Finiteness confirmation ===")
    all_finite = (
        np.isfinite(policy_raw).all() and np.isfinite(entropy_raw).all()
        and np.isfinite(clamped_std).all() and np.isfinite(raw_mean).all()
        and np.isfinite(action_sat).all() if len(action_sat) else True
    )
    print(f"ALL captured values finite: {bool(all_finite)}")

    _torch.atanh = _orig_atanh  # restore, in case this module is imported rather than run standalone


if __name__ == "__main__":
    main()
