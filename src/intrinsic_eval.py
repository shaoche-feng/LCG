"""Real-environment intrinsic evaluation: score recorded test trajectories with the current frozen
intrinsic scorer, using each method's own scoring definition on the real history and action (and,
for LCG, the real successor). Nothing is refreshed, redrawn or refitted, and no normalization,
last-batch or timing state of the PMPO hook changes. Intrinsic returns have method-specific units:
they cannot rank methods against each other.

Alignment (step t of a recorded episode, frames o_0..o_{T-1}, actions a_0..a_{T-1}, final
observation o_T):
- Sampler window (LCG, full ensemble, Laplace): frames o_{t-n+1..t} and actions a_{t-n+1..t}, zero
  padded before the episode start like the denoiser's training segments; WorldModelEnv.step holds
  exactly this (its last action slot is the action just taken).
- LCG: the candidate (window, real o_{t+1}); steps without a real successor are not scored.
- Full ensemble / Laplace: shared-noise member predictions from the window (noise from the caller's
  RNG, the diagnostic stream); population pixel variance averaged over noise samples.
- Plan2Explore: feature of o_t with context o_{t-n..t-1}, a_{t-n..t-1} (plan2explore.context_window)
  and the action a_t, scored by the frozen heads.
"""
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from envs.world_model_env import ensemble_disagreement
from plan2explore import context_window


def sampler_windows(episode, n: int) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """(obs (T, n, c, h, w), act (T, n, a), successors (T, c, h, w) or (T - 1, ...)): the
    conditioning window of every step and the real next frames (the final observation completes
    the last step when the episode has one)."""
    obs, act = episode.obs, episode.act
    T = len(episode)
    padded_obs = torch.cat((obs.new_zeros(n - 1, *obs.shape[1:]), obs))
    padded_act = torch.cat((act.new_zeros(n - 1, *act.shape[1:]), act))
    windows_obs = torch.stack([padded_obs[t:t + n] for t in range(T)])
    windows_act = torch.stack([padded_act[t:t + n] for t in range(T)])
    final = episode.info.get("final_observation") if isinstance(episode.info, dict) else None
    successors = obs[1:]
    if final is not None:
        # The final observation may still be on the collection device.
        successors = torch.cat((successors, final.reshape(1, *obs.shape[1:]).to(device=obs.device, dtype=obs.dtype)))
    return windows_obs, windows_act, successors


@contextmanager
def preserved(obj, names):
    """Restore monitoring attributes (timings, call counters) changed by evaluation calls."""
    saved = {name: getattr(obj, name) for name in names}
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(obj, name, value)


class IntrinsicEvaluator:
    """Scores real episodes with one method's frozen scorer. method: lcg, ensemble_disagreement,
    laplace_disagreement or plan2explore_disagreement."""

    def __init__(self, method: str, num_conditioning: int, device: torch.device, chunk_size: int = 64,
                 lcg=None, ensemble_samplers=None, ensemble_scorer=None, laplace=None, sampler=None,
                 plan2explore=None, noise_samples: int = 1) -> None:
        self.method, self.n, self.device, self.chunk = method, num_conditioning, device, chunk_size
        self.lcg, self.ensemble_samplers, self.ensemble_scorer = lcg, ensemble_samplers, ensemble_scorer
        self.laplace, self.sampler, self.plan2explore, self.noise_samples = laplace, sampler, plan2explore, noise_samples

    def unavailable_reason(self) -> Optional[str]:
        """None when the frozen scorer exists; else why it cannot score (e.g. not refreshed yet)."""
        if self.method == "lcg" and not self.lcg.ready:
            return "LCG has not been refreshed in this process (no precision/bank yet)"
        if self.method == "laplace_disagreement" and self.laplace.perturbations is None:
            return "the Laplace sample bank has not been drawn"
        if self.method == "plan2explore_disagreement" and self.plan2explore.heads is None:
            return "the Plan2Explore heads are not fitted"
        return None

    def settings(self) -> Dict[str, Any]:
        s = dict(method=self.method, num_conditioning=self.n, noise_samples=self.noise_samples)
        if self.method == "lcg":
            s.update(round=self.lcg.round_id, scaling=self.lcg.cfg.reward_scaling, candidate_num_mc=self.lcg.cfg.candidate_num_mc,
                     scored_steps="steps with a real successor")
        elif self.method == "laplace_disagreement":
            s.update(refresh_epoch=self.laplace.refresh_epoch, members=self.laplace.cfg.num_members,
                     temperature=self.laplace.cfg.posterior_temperature, scaling=self.laplace.cfg.reward_scaling)
        elif self.method == "plan2explore_disagreement":
            s.update(refresh_epoch=self.plan2explore.last_refresh_epoch, heads=self.plan2explore.cfg.num_members,
                     scaling=self.plan2explore.cfg.reward_scaling)
        else:
            s.update(members=len(self.ensemble_samplers), scaling=self.ensemble_scorer.cfg.reward_scaling)
        return s

    @torch.no_grad()
    def score_episode(self, episode) -> Tuple[Tensor, Optional[Tensor]]:
        """(raw (S,), scaled (S,) or None when no frozen scale exists yet), on CPU."""
        if self.method == "lcg":
            raw = self._lcg(episode)
            scaled = self.lcg.scale_frozen(raw)
        elif self.method == "plan2explore_disagreement":
            raw = self._plan2explore(episode)
            scaled = self.plan2explore.scale(raw)
        else:
            raw = self._members(episode)
            scaled = self.ensemble_scorer.scale_frozen(raw)
        return raw.float().cpu(), None if scaled is None else scaled.float().cpu()

    def _lcg(self, episode) -> Tensor:
        obs, act, successors = sampler_windows(episode, self.n)
        candidates = []
        for t in range(successors.shape[0]):
            o = obs[t:t + 1].to(self.device)
            candidates.append((o.reshape(1, -1, *o.shape[-2:]), act[t:t + 1].to(self.device),
                               successors[t:t + 1].to(self.device)))
        return self.lcg.score_candidates(candidates)

    def _members(self, episode) -> Tensor:
        obs, act, _ = sampler_windows(episode, self.n)
        scores = []
        for i in range(0, obs.shape[0], self.chunk):
            o, a = obs[i:i + self.chunk].to(self.device), act[i:i + self.chunk].to(self.device)
            b, _, c, h, w = o.shape
            per_noise = []
            for _ in range(self.noise_samples):
                noise = torch.randn(b, c, h, w, device=self.device)
                if self.laplace is not None:
                    with preserved(self.laplace, ("prediction_seconds", "prediction_calls", "prefix_evaluations")):
                        predictions = self.laplace.predict_members(self.sampler, o, a, noise)
                else:
                    predictions = torch.stack([s.sample(o, a, noise)[0] for s in self.ensemble_samplers])
                per_noise.append(ensemble_disagreement(predictions))
            scores.append(torch.stack(per_noise).mean(dim=0))
        return torch.cat(scores)

    def _plan2explore(self, episode) -> Tensor:
        p2e = self.plan2explore
        frames, acts = episode.obs, episode.act
        scores = []
        with preserved(p2e.extractor, ("seconds",)):
            for i in range(0, len(episode), self.chunk):
                steps = range(i, min(i + self.chunk, len(episode)))
                context = [context_window(frames, acts, j, self.n) for j in steps]
                ctx = torch.stack([c[0] for c in context])
                ctx_act = torch.stack([c[1] for c in context])
                features = p2e.extractor(frames[i:steps.stop], ctx, ctx_act)
                scores.append(p2e.disagreement(features, acts[i:steps.stop].to(features.device)))
        return torch.cat(scores)


def episode_record(raw: Optional[Tensor], scaled: Optional[Tensor]) -> Dict[str, Any]:
    if raw is None:
        return {}
    steps = int(raw.numel())
    record = dict(intrinsic_scored_steps=steps, intrinsic_raw_return=float(raw.double().sum()),
                  intrinsic_raw_per_step=float(raw.double().mean()) if steps else float("nan"))
    if scaled is not None:
        record.update(intrinsic_scaled_return=float(scaled.double().sum()),
                      intrinsic_scaled_per_step=float(scaled.double().mean()) if steps else float("nan"))
    return record


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Mean/std (population) over episodes of each per-episode quantity present in all of them."""
    import numpy as np
    out = {}
    keys = [k for k in records[0] if all(k in r for r in records) and isinstance(records[0][k], (int, float))
            and k not in ("episode", "seed")] if records else []
    for k in keys:
        values = np.array([r[k] for r in records], dtype=np.float64)
        out[f"{k}_mean"], out[f"{k}_std"] = float(values.mean()), float(values.std())
    return out
