"""Opt-in bounded-run observation/evaluation only; never changes a learning loss."""
from contextlib import contextmanager
import json
from pathlib import Path
import random

import numpy as np
import torch

from coroutines.frame_history import FrameHistory
from envs.dm_control_env import DMControlEnv


@contextmanager
def preserve_rng(device):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


class DiagnosticStop(RuntimeError):
    pass


class PMPORunDiagnostic:
    def __init__(self, cfg, model, env_cfg):
        self.cfg, self.model, self.env_cfg = cfg, model, env_cfg
        self.path = Path(cfg.output_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.fixed_real = None
        self.boundary_streak = self.constant_streak = self.one_sided_epochs = 0
        self.previous_entropy_per_dim = None

    def write(self, name, record):
        with (self.path / name).open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, allow_nan=False) + "\n")

    def check_update(self, metrics, epoch, update):
        row = {k: float(v) for k, v in metrics.items()}
        row.update(epoch=epoch, update=update)
        if not all(np.isfinite(v) for v in row.values()):
            raise DiagnosticStop("Non-finite controller diagnostic")
        self.write("updates.jsonl", row)
        reason = None
        if max(row["alpha_max"], row["beta_max"]) > 1e4:
            reason = "Beta concentration exceeded 10000"
        if row["kl_prior"] > 10:
            reason = "Policy KL exceeded 10 nats"
        if max(row["value_abs_max"], row["return_abs_max"]) > 1e6:
            reason = "Value or lambda-return magnitude exceeded 1e6"
        boundary = row["near_lower_fraction"] + row["near_upper_fraction"]
        self.boundary_streak = self.boundary_streak + 1 if boundary > 0.95 else 0
        fixed_std = max(v for k, v in row.items() if k.startswith("fixed_policy_mean_state_std_"))
        self.constant_streak = self.constant_streak + 1 if fixed_std < 1e-6 else 0
        if self.boundary_streak >= 50:
            reason = "Over 95% near-bound actions for 50 consecutive updates"
        if self.constant_streak >= 100:
            reason = "Fixed-state actions effectively constant for 100 consecutive updates"
        # Differential entropy can legitimately be negative. Observe canonical
        # entropy per dimension, with an abrupt-drop/concentration condition.
        width_log = (self.model.action_high - self.model.action_low).log().sum().item()
        entropy = (row["policy_entropy"] - width_log) / len(self.model.action_low)
        if entropy < -5:
            reason = "Canonical differential entropy per dimension below -5 nats"
        if (self.previous_entropy_per_dim is not None and entropy < self.previous_entropy_per_dim - 2
                and max(row["alpha_mean"], row["beta_mean"]) > 100):
            reason = "Abrupt entropy collapse with mean concentration above 100"
        self.previous_entropy_per_dim = entropy
        if reason:
            self.write("stop.jsonl", dict(epoch=epoch, update=update, reason=reason))
            raise DiagnosticStop(reason)

    @torch.no_grad()
    def evaluate(self):
        model = self.model
        returns, lengths, fixed = [], [], []
        kwargs = {k: v for k, v in self.env_cfg.items() if k != "type"}
        was_training = model.training
        model.eval()
        with preserve_rng(model.device):
            try:
                for seed in self.cfg.evaluation_seeds:
                    env = DMControlEnv(**kwargs, seed=int(seed))
                    try:
                        image, _ = env.reset(seed=int(seed))
                        def observation(image):
                            return torch.as_tensor(image.copy(), device=model.device).permute(2, 0, 1).float().div(255).mul(2).sub(1).unsqueeze(0)
                        history = FrameHistory(observation(image), model.frame_stack)
                        total = 0.0
                        for step in range(10000):
                            state = history.state
                            if self.fixed_real is None and len(fixed) < 128 and step % 4 == 0:
                                fixed.append(state.cpu().clone())
                            action = model.sample_action(model.actor(state), deterministic=True)
                            image, reward, end, trunc, _ = env.step(action[0].cpu().numpy())
                            total += float(reward)
                            if end or trunc:
                                returns.append(total)
                                lengths.append(step + 1)
                                break
                            history.advance(observation(image), torch.tensor([False], device=model.device))
                        else:
                            raise DiagnosticStop("Evaluation exceeded 10000 control steps without episode end")
                    finally:
                        env.close()
                if self.fixed_real is None:
                    self.fixed_real = torch.cat(fixed)
                    torch.save(self.fixed_real, self.path / "fixed_real_stacks.pt")
                states = self.fixed_real.to(model.device)
                temporal = {k: float(v) for k, v in model.temporal_diagnostics(states, "real_fixed").items()}
                raw = model.actor(states)
                mean_actions = model.sample_action(raw, deterministic=True)
                logp, entropy = model.log_prob_and_entropy(raw, mean_actions)
                distribution = {"real_fixed_" + k: float(v) for k, v in model.diagnostics(raw, mean_actions, logp, entropy).items()}
                distribution.update(temporal)
            finally:
                model.train(was_training)
        return dict(eval_return_mean=float(np.mean(returns)), eval_return_median=float(np.median(returns)),
                    eval_return_std=float(np.std(returns)), eval_return_min=float(np.min(returns)),
                    eval_return_max=float(np.max(returns)), eval_returns=returns, eval_lengths=lengths,
                    evaluation_seeds=list(self.cfg.evaluation_seeds), **distribution)

    def finish_epoch(self, epoch, logs, real_steps):
        rows = [{k.removeprefix("actor_critic/train/"): float(v) for k, v in log.items()}
                for log in logs if "actor_critic/train/loss_actor" in log]
        if not rows:
            raise DiagnosticStop("Epoch had no PMPO updates")
        aggregate = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        for name in ("alpha", "beta"):
            aggregate[f"{name}_min"] = min(r[f"{name}_min"] for r in rows)
            aggregate[f"{name}_max"] = max(r[f"{name}_max"] for r in rows)
        row = dict(epoch=epoch, real_training_steps=real_steps, controller_updates=len(rows), **aggregate, **self.evaluate())
        one_sided = min(row["positive_fraction"], row["negative_fraction"]) < 0.01
        self.one_sided_epochs = self.one_sided_epochs + 1 if one_sided else 0
        row["one_sided_epochs"] = self.one_sided_epochs
        row["prolonged_one_sided_flag"] = self.one_sided_epochs >= 3
        self.write("epochs.jsonl", row)
        print("PMPO_EPOCH " + json.dumps(row, allow_nan=False), flush=True)
        return row
