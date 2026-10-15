"""Opt-in bounded-run observation/evaluation only; never changes a learning loss."""
from contextlib import contextmanager
import json
from pathlib import Path
import random

import numpy as np
import torch
import wandb

from coroutines.frame_history import FrameHistory
from envs.dm_control_env import DMControlEnv

# Clean wandb namespace for the main PMPO monitoring metrics (see check_update);
# mirrored only when a wandb run is already active (wandb.mode != "disabled"), so
# training behaves identically with or without it. Keys are read from the SAME row
# dict written to updates.jsonl, which remains the authoritative record either way.
_WANDB_UPDATE_KEYS = (
    "alpha_mean", "beta_mean", "policy_entropy", "kl_prior", "kl_initial", "loss_actor",
    "loss_value", "actor_grad_norm", "value_grad_norm", "near_lower_fraction",
    "near_upper_fraction", "positive_fraction",
)


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


def normalize_pixels(pixels):
    """uint8 RGB -> [-1, 1] float32; evaluation preprocessing and checkpoint decoding."""
    return pixels.float().div(255).mul(2).sub(1)


# normalize_pixels rounds differently by device, so each uint8 encoding names the
# arithmetic that reproduces the stored states; all decoding runs on CPU.
_RECIPROCAL_255 = torch.tensor(1.0) / torch.tensor(255.0)
PIXEL_DECODERS = {
    # CPU evaluation: true division.
    "uint8_pixels_div": lambda pixels: pixels.float().div(255).mul(2).sub(1),
    # CUDA evaluation: PyTorch divides by a scalar as a multiply by its float32 reciprocal.
    "uint8_pixels_mul_reciprocal": lambda pixels: pixels.float().mul(_RECIPROCAL_255).mul(2).sub(1),
}


def encode_fixed_real(fixed_real):
    """Checkpoint form of the fixed states: uint8 pixels when a decoder reproduces them
    bit for bit, else the float32 tensor itself."""
    if fixed_real is None:
        return None
    fixed_real = fixed_real.cpu()
    pixels = fixed_real.add(1).div(2).mul(255).round().clamp(0, 255).to(torch.uint8)
    for encoding, decode in PIXEL_DECODERS.items():
        if torch.equal(decode(pixels), fixed_real):
            return dict(encoding=encoding, values=pixels)
    return dict(encoding="float32", values=fixed_real)


def decode_fixed_real(stored):
    if stored is None:
        return None
    if torch.is_tensor(stored):  # Earlier diagnostic states stored the float32 tensor itself.
        return stored.cpu()
    values = stored["values"].cpu()
    if stored["encoding"] in PIXEL_DECODERS:
        return PIXEL_DECODERS[stored["encoding"]](values)
    if stored["encoding"] == "float32":
        return values
    raise ValueError(f"Unknown fixed_real encoding {stored['encoding']!r}")


class PMPORunDiagnostic:
    def __init__(self, cfg, model, env_cfg, optimizers=None, trainer_state_fn=None):
        self.cfg, self.model, self.env_cfg, self.optimizers = cfg, model, env_cfg, optimizers
        # trainer_state_fn (if given): a zero-arg callback returning the FULL Trainer
        # state_dict (agent incl. denoiser/rew_end_model/actor_critic, all three
        # optimizers, LR schedulers, epoch counter -- exactly what Trainer's own
        # epoch-boundary save_checkpoint() already relies on for resumability). Used to
        # save a genuinely resumable full-loop checkpoint every prior_refresh_interval
        # PMPO updates, not just the lightweight actor/value/prior-only snapshot below.
        self.trainer_state_fn = trainer_state_fn
        self.path = Path(cfg.output_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        # Not read here: a fresh run records its own reference set, and a restored run
        # takes its observations from the checkpoint (see load_state_dict).
        self.path_fixed_real = self.path / "fixed_real_stacks.pt"
        self.fixed_real = None
        self.boundary_streak = self.constant_streak = self.one_sided_epochs = 0
        self.previous_entropy_per_dim = None
        # Completed PMPO updates whose rows were processed, including non-finite rows,
        # which are kept out of the stop-rule history. With the diagnostic active from
        # the start, this equals PMPOBeta.updates in every checkpoint saving it.
        self.updates_observed = 0
        self.nonfinite_updates = 0
        self.checkpoint_updates = set(getattr(cfg, "checkpoint_updates", ()))

    def state_dict(self):
        return dict(updates_observed=self.updates_observed, nonfinite_updates=self.nonfinite_updates,
                    boundary_streak=self.boundary_streak, constant_streak=self.constant_streak,
                    one_sided_epochs=self.one_sided_epochs, previous_entropy_per_dim=self.previous_entropy_per_dim,
                    fixed_real=encode_fixed_real(self.fixed_real))

    def load_state_dict(self, state):
        """Call after the model state is restored: the history must describe the same
        moment as the restored PMPOBeta.updates."""
        model_updates = self.model.updates.item()
        if state is None:
            print("PMPO diagnostic: checkpoint has no diagnostic history; stop-rule streaks restart", flush=True)
            self.updates_observed = model_updates
            self.load_fixed_real_file()
            return
        if state["updates_observed"] != model_updates:
            raise ValueError(f"PMPO diagnostic history observed {state['updates_observed']} updates, "
                             f"but the restored model completed {model_updates}")
        self.updates_observed = state["updates_observed"]
        self.nonfinite_updates = state["nonfinite_updates"]
        self.boundary_streak = state["boundary_streak"]
        self.constant_streak = state["constant_streak"]
        self.one_sided_epochs = state["one_sided_epochs"]
        self.previous_entropy_per_dim = state["previous_entropy_per_dim"]
        if "fixed_real" not in state:
            self.load_fixed_real_file()
            return
        # None: saved before the first evaluation; the next evaluation records a new set.
        self.fixed_real = decode_fixed_real(state["fixed_real"])
        if self.fixed_real is None:
            return
        if not self.path_fixed_real.exists():
            torch.save(self.fixed_real, self.path_fixed_real)
            return
        try:
            on_disk = torch.load(self.path_fixed_real, map_location="cpu", weights_only=True)
            matches = torch.is_tensor(on_disk) and torch.equal(on_disk, self.fixed_real)
        except Exception:
            matches = False
        if not matches:
            print("PMPO diagnostic: fixed_real_stacks.pt is unreadable or differs from the checkpoint; "
                  "using the checkpoint's observations and leaving the file unchanged", flush=True)

    def load_fixed_real_file(self):
        """Fallback for checkpoints without observations: use fixed_real_stacks.pt only if
        it holds valid reference stacks for this model, and report which case applies."""
        if not self.path_fixed_real.is_file():
            print("PMPO diagnostic: no fixed_real_stacks.pt; the next evaluation records a new reference set",
                  flush=True)
            return
        cfg = self.model.cfg
        shape = (cfg.img_channels * self.model.frame_stack, cfg.img_size, cfg.img_size)
        try:
            fixed_real = torch.load(self.path_fixed_real, map_location="cpu", weights_only=True)
        except Exception as error:
            problem = f"unreadable ({type(error).__name__})"
        else:
            if not torch.is_tensor(fixed_real) or fixed_real.dtype != torch.float32:
                problem = "not a float32 tensor"
            elif fixed_real.ndim != 4 or tuple(fixed_real.shape[1:]) != shape or not 1 <= len(fixed_real) <= 128:
                problem = f"shape {tuple(fixed_real.shape)} is not (1..128, *{shape})"
            elif not torch.isfinite(fixed_real).all() or fixed_real.abs().max() > 1:
                problem = "values are non-finite or outside [-1, 1]"
            else:
                problem = None
        if problem:
            print(f"PMPO diagnostic: fixed_real_stacks.pt is not usable ({problem}); it is kept, "
                  "and the next evaluation records a new reference set", flush=True)
            return
        print("PMPO diagnostic: checkpoint has no fixed observations; using fixed_real_stacks.pt", flush=True)
        self.fixed_real = fixed_real

    def save_fixed_real_file(self):
        # A new reference set never overwrites an earlier file (stale, unreadable or rejected).
        if self.path_fixed_real.exists():
            n = 1
            while (preserved := self.path / f"fixed_real_stacks.preserved_{n}.pt").exists():
                n += 1
            self.path_fixed_real.rename(preserved)
            print(f"PMPO diagnostic: kept the earlier fixed_real_stacks.pt as {preserved.name}", flush=True)
        torch.save(self.fixed_real, self.path_fixed_real)

    def write(self, name, record):
        with (self.path / name).open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, allow_nan=False) + "\n")

    def save_diagnostic_checkpoint(self, update, tag=None):
        # Observation only: a plain torch.save taken between optimizer steps (never
        # mid-backward), so it never perturbs training state or semantics. model_state
        # is the FULL PMPOBeta state_dict (actor/value/prior_actor/updates-counter/
        # action bounds, exactly what checkpoint/resume already relies on); optimizers
        # (if provided) is PMPOOptimizers.state_dict() (actor+value Adam + both LR
        # schedulers); rng is the model's own imagination-RNG snapshot dict (the only
        # RNG state this branch tracks -- there is no separate global torch/numpy/python
        # RNG checkpoint here, unlike the richer DrQ branch); config is the model's own
        # PMPOBetaConfig (picklable dataclass) for exact hyperparameter provenance.
        payload = {"model_state": self.model.state_dict(),
                   "optimizers": self.optimizers.state_dict() if self.optimizers is not None else None,
                   "rng": self.model._rng, "config": self.model.cfg, "update": update}
        name = f"checkpoint_update_{update:05d}" + (f"_{tag}" if tag else "")
        torch.save(payload, self.path / f"{name}.pt")

    def save_full_training_checkpoint(self, epoch, update, tag=None):
        # The full Trainer state dict: denoiser, reward/end model, PMPO actor/value/prior,
        # all three optimizers, LR schedulers, counters, this diagnostic's history, and
        # (via PMPOBeta's get_extra_state/set_extra_state hook) the imagination-RNG
        # snapshot, at a finer (prior-refresh-interval) cadence than Trainer's own
        # epoch-boundary save_checkpoint(). The model and diagnostic history describe
        # the same completed update. It is not a supported mid-epoch resume point:
        # Trainer only resumes from checkpoints/state.pt; inside a refresh or stop save
        # Trainer's PMPO batch counter still lags PMPOBeta.updates by one; Trainer.epoch
        # is the epoch in progress; and the real-environment datasets are not
        # re-persisted here.
        if self.trainer_state_fn is None:
            return
        payload = {"trainer_state": self.trainer_state_fn(), "epoch": epoch, "update": update}
        # agent_pmpo_update_NNNNNN[_tag].pt -- epoch is still inside the payload for
        # provenance, just not in the filename (the update count is what matters for
        # matching this checkpoint to a prior-refresh block boundary).
        name = f"agent_pmpo_update_{update:06d}" + (f"_{tag}" if tag else "")
        path = self.path / f"{name}.pt"
        torch.save(payload, path)
        self.write("checkpoints.jsonl", dict(epoch=epoch, update=update, tag=tag, path=str(path)))

    def check_update(self, metrics, epoch, update, real_steps=None):
        row = {k: float(v) for k, v in metrics.items()}
        interval = self.model.cfg.prior_refresh_interval
        row.update(epoch=epoch, update=update, real_steps=real_steps,
                   prior_refresh_interval=interval, prior_refresh_event=(update % interval == 0),
                   prior_block_position=update % interval)
        finite_row = {k: v for k, v in row.items() if isinstance(v, (int, float))}
        self.updates_observed += 1
        if not all(np.isfinite(v) for v in finite_row.values()):
            # The update completed, but its values never enter the stop-rule history.
            self.nonfinite_updates += 1
            self.save_diagnostic_checkpoint(update, tag="stop")
            self.save_full_training_checkpoint(epoch, update, tag="stop")
            raise DiagnosticStop("Non-finite controller diagnostic")
        self.write("updates.jsonl", row)
        if wandb.run is not None:
            wandb.log({"pmpo/actor_update": update, "pmpo/real_env_steps": real_steps,
                       **{f"pmpo/{k}": row[k] for k in _WANDB_UPDATE_KEYS if k in row}})
        # The model already completed this update, so every snapshot below must include it.
        reason = self.observe(row)
        if update in self.checkpoint_updates:
            self.save_diagnostic_checkpoint(update)
        if update > 0 and update % interval == 0:
            self.save_full_training_checkpoint(epoch, update)
        if reason:
            self.save_diagnostic_checkpoint(update, tag="stop")
            self.save_full_training_checkpoint(epoch, update, tag="stop")
            self.write("stop.jsonl", dict(epoch=epoch, update=update, reason=reason))
            raise DiagnosticStop(reason)

    def observe(self, row):
        """Fold one finite row into the stop-rule history; return the stop reason, if any."""
        reason = None
        if max(row["alpha_max"], row["beta_max"]) > 1e4:
            reason = "Beta concentration exceeded 10000"
        if row["kl_prior"] > 10:
            reason = "Policy KL exceeded 10 nats"
        if max(row["value_abs_max"], row["return_abs_max"]) > 1e6:
            reason = "Value or lambda-return magnitude exceeded 1e6"
        boundary = row["near_lower_fraction"] + row["near_upper_fraction"]
        boundary_streak = self.boundary_streak + 1 if boundary > 0.95 else 0
        fixed_std = max(v for k, v in row.items() if k.startswith("fixed_policy_mean_state_std_"))
        constant_streak = self.constant_streak + 1 if fixed_std < 1e-6 else 0
        if boundary_streak >= 50:
            reason = "Over 95% near-bound actions for 50 consecutive updates"
        if constant_streak >= 100:
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
        self.boundary_streak, self.constant_streak = boundary_streak, constant_streak
        self.previous_entropy_per_dim = entropy
        return reason

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
                            return normalize_pixels(torch.as_tensor(image.copy(), device=model.device).permute(2, 0, 1)).unsqueeze(0)
                        history = FrameHistory(observation(image), model.frame_stack)
                        total = 0.0
                        for step in range(10000):
                            state = history.state
                            if self.fixed_real is None and len(fixed) < 128 and (step < 4 or step % 4 == 0):
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
                    self.save_fixed_real_file()
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

    def world_model_loss_summary(self, logs):
        # Denoiser/reward-end-model train (and test, when that epoch runs evaluation)
        # losses, straight from Trainer's own to_log rows -- observation only, never
        # read back into any loss. Epochs where start_after_epochs skips a component
        # (e.g. a warm-started diagnostic epoch 1) simply produce no matching rows.
        summary = {}
        for component in ("denoiser", "rew_end_model"):
            for split in ("train", "test"):
                prefix = f"{component}/{split}/"
                values = {}
                for log in logs:
                    for k, v in log.items():
                        if k.startswith(prefix):
                            values.setdefault(k[len(prefix):], []).append(float(v))
                for key, vals in values.items():
                    summary[f"{component}_{split}_{key}_mean"] = float(np.mean(vals))
                loss_key = "loss_denoising" if component == "denoiser" else "loss_total"
                summary[f"{component}_{split}_steps"] = len(values.get(loss_key, values.get("loss", [])))
        return summary

    def finish_epoch(self, epoch, logs, real_steps):
        rows = [{k.removeprefix("actor_critic/train/"): float(v) for k, v in log.items()}
                for log in logs if "actor_critic/train/loss_actor" in log]
        if not rows:
            raise DiagnosticStop("Epoch had no PMPO updates")
        aggregate = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        for name in ("alpha", "beta"):
            aggregate[f"{name}_min"] = min(r[f"{name}_min"] for r in rows)
            aggregate[f"{name}_max"] = max(r[f"{name}_max"] for r in rows)
        global_update = self.model.updates.item()
        row = dict(epoch=epoch, real_training_steps=real_steps, controller_updates=len(rows),
                   global_pmpo_update=global_update, **aggregate,
                   **self.world_model_loss_summary(logs), **self.evaluate())
        if wandb.run is not None:
            wandb.log({"pmpo/actor_update": global_update, "pmpo/real_env_steps": real_steps,
                       "eval/return_mean": row["eval_return_mean"], "eval/return_median": row["eval_return_median"],
                       "eval/return_std": row["eval_return_std"], "eval/return_min": row["eval_return_min"],
                       "eval/return_max": row["eval_return_max"]})
        one_sided = min(row["positive_fraction"], row["negative_fraction"]) < 0.01
        self.one_sided_epochs = self.one_sided_epochs + 1 if one_sided else 0
        row["one_sided_epochs"] = self.one_sided_epochs
        row["prolonged_one_sided_flag"] = self.one_sided_epochs >= 3
        self.write("epochs.jsonl", row)
        print("PMPO_EPOCH " + json.dumps(row, allow_nan=False), flush=True)
        self.save_full_training_checkpoint(epoch, global_update, tag="epoch")
        return row
