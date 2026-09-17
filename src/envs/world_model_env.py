from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader

from coroutines import coroutine
from data import collate_segments_to_batch, SegmentId
from models.diffusion import Denoiser, DiffusionSampler, DiffusionSamplerConfig
from models.rew_end_model import RewEndModel

ResetOutput = Tuple[torch.FloatTensor, Dict[str, Any]]
StepOutput = Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]
InitialCondition = Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]
# One preload cycle's segment ids: num_batches_to_preload batches, each batch_size SegmentIds.
PreloadCycle = List[List[SegmentId]]


@dataclass
class ImaginedCandidate:
    """The exact conditioning window x* = (obs_buffer, act_buffer) the diffusion sampler
    used, and the exact y* it sampled, for one WorldModelEnv.step() call. x_obs/x_act keep
    their natural (num_envs, t, ...) shape (matching DiffusionSampler.sample's own input
    convention); y_star is the unmodified tensor returned by the sampler, not reconstructed
    or independently resampled. Populated only when WorldModelEnv is constructed with
    `return_imagined_candidate=True` (default False, existing behavior unchanged)."""

    x_obs: Tensor
    x_act: Tensor
    y_star: Tensor


@dataclass
class WorldModelEnvConfig:
    horizon: int
    num_batches_to_preload: int
    diffusion_sampler: DiffusionSamplerConfig


class WorldModelEnv:
    def __init__(
        self,
        denoiser: Denoiser,
        rew_end_model: RewEndModel,
        data_loader: DataLoader,
        cfg: WorldModelEnvConfig,
        return_denoising_trajectory: bool = False,
        return_imagined_candidate: bool = False,
    ) -> None:
        self.sampler = DiffusionSampler(denoiser, cfg.diffusion_sampler)
        self.rew_end_model = rew_end_model
        self.horizon = cfg.horizon
        self.return_denoising_trajectory = return_denoising_trajectory
        self.return_imagined_candidate = return_imagined_candidate
        self.num_envs = data_loader.batch_sampler.batch_size
        self.dataset = data_loader.dataset

        # Resume-fidelity state (see class docstring section below for the full design).
        # self._preload_cycle/_preload_cursor track what make_generator_init's *current*
        # (possibly still in-use) 256-batch preload block actually contains, so a checkpoint
        # taken at ANY point mid-block can reproduce it exactly on resume, without redrawing
        # from the BatchSampler. self._pending_resume_preload, when set (by
        # load_preload_state_dict, called by Trainer.load_state_checkpoint BEFORE this
        # coroutine is ever exercised in the new process), makes the *next* preload cycle
        # replay that saved block instead of drawing a fresh one -- consuming zero new random
        # values -- after which normal fresh-drawing behavior resumes for all following cycles.
        self._preload_cycle: PreloadCycle = []
        self._preload_cursor: int = 0
        self._pending_resume_preload: Optional[Tuple[PreloadCycle, int]] = None

        self.generator_init = self.make_generator_init(data_loader, cfg.num_batches_to_preload)

    @property
    def device(self) -> torch.device:
        return self.sampler.denoiser.device

    # -------------------------------------------------------------------------------------
    # Resume-fidelity state: the live rollout buffers (obs_buffer/act_buffer/hx_rew_end/
    # cx_rew_end/ep_len, all set by reset()/reset_dead() -- absent until the first real use)
    # and the preload-cycle replay state (see make_generator_init). Both are opt-in, read by
    # Trainer via getattr with a default so a WorldModelEnv that was never exercised, or an
    # old Trainer that never calls these, behaves exactly as before.
    # -------------------------------------------------------------------------------------

    def rollout_state_dict(self) -> Optional[Dict[str, Any]]:
        """None if reset() has never been called yet in this process (nothing to save)."""
        if not hasattr(self, "obs_buffer"):
            return None
        return {
            "obs_buffer": self.obs_buffer,
            "act_buffer": self.act_buffer,
            "hx_rew_end": self.hx_rew_end,
            "cx_rew_end": self.cx_rew_end,
            "ep_len": self.ep_len,
        }

    def load_rollout_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Directly installs the live rollout buffers, bypassing reset() entirely -- reset()
        would draw a FRESH initial condition from generator_init, which is exactly what a
        resume must NOT do when valid saved buffers are available. Caller (env_loop, via the
        `resuming` path) is responsible for never calling reset() after this."""
        self.obs_buffer = state_dict["obs_buffer"].to(self.device)
        self.act_buffer = state_dict["act_buffer"].to(self.device)
        self.hx_rew_end = state_dict["hx_rew_end"].to(self.device)
        self.cx_rew_end = state_dict["cx_rew_end"].to(self.device)
        self.ep_len = state_dict["ep_len"].to(self.device)

    def preload_state_dict(self) -> Dict[str, Any]:
        return {"preload_cycle": self._preload_cycle, "preload_cursor": self._preload_cursor}

    def load_preload_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Must be called BEFORE generator_init's coroutine is ever sent a real value in this
        process (i.e. before the first reset()/reset_dead() call) -- see make_generator_init's
        docstring for what happens if that invariant is violated."""
        self._pending_resume_preload = (state_dict["preload_cycle"], state_dict["preload_cursor"])

    @torch.no_grad()
    def reset(self, **kwargs) -> ResetOutput:
        obs, act, (hx, cx) = self.generator_init.send(self.num_envs)
        self.obs_buffer = obs
        self.act_buffer = act
        self.hx_rew_end = hx
        self.cx_rew_end = cx
        self.ep_len = torch.zeros(self.num_envs, dtype=torch.long, device=obs.device)
        return self.obs_buffer[:, -1], {}

    @torch.no_grad()
    def reset_dead(self, dead: torch.BoolTensor) -> None:
        obs, act, (hx, cx) = self.generator_init.send(dead.sum().item())
        self.obs_buffer[dead] = obs
        self.act_buffer[dead] = act
        self.hx_rew_end[:, dead] = hx
        self.cx_rew_end[:, dead] = cx
        self.ep_len[dead] = 0

    @torch.no_grad()
    def step(self, act: torch.Tensor) -> StepOutput:
        self.act_buffer[:, -1] = act

        if self.return_imagined_candidate:
            candidate_x_obs = self.obs_buffer.clone()
            candidate_x_act = self.act_buffer.clone()

        next_obs, denoising_trajectory = self.predict_next_obs()
        rew, end = self.predict_rew_end(next_obs.unsqueeze(1))

        self.ep_len += 1
        trunc = (self.ep_len >= self.horizon).long()

        self.obs_buffer = self.obs_buffer.roll(-1, dims=1)
        self.act_buffer = self.act_buffer.roll(-1, dims=1)
        self.obs_buffer[:, -1] = next_obs

        dead = torch.logical_or(end, trunc)

        info = {}
        if self.return_denoising_trajectory:
            info["denoising_trajectory"] = torch.stack(denoising_trajectory, dim=1)

        if self.return_imagined_candidate:
            info["imagined_candidate"] = ImaginedCandidate(candidate_x_obs, candidate_x_act, next_obs.clone())

        if dead.any():
            self.reset_dead(dead)
            info["final_observation"] = next_obs[dead]
            info["burnin_obs"] = self.obs_buffer[dead, :-1]

        return self.obs_buffer[:, -1], rew, end, trunc, info

    @torch.no_grad()
    def predict_next_obs(self) -> Tuple[Tensor, List[Tensor]]:
        return self.sampler.sample(self.obs_buffer, self.act_buffer)

    @torch.no_grad()
    def predict_rew_end(self, next_obs: Tensor) -> Tuple[Tensor, Tensor]:
        rew_out, logits_end, (self.hx_rew_end, self.cx_rew_end) = self.rew_end_model.predict_rew_end(
            self.obs_buffer[:, -1:],
            self.act_buffer[:, -1:],
            next_obs,
            (self.hx_rew_end, self.cx_rew_end),
        )
        if self.rew_end_model.continuous_reward:
            rew = rew_out.squeeze(1)  # (b, 1) -> (b,), predicted scalar reward used directly
        else:
            rew = Categorical(logits=rew_out).sample().squeeze(1) - 1.0  # in {-1, 0, 1}
        end = Categorical(logits=logits_end).sample().squeeze(1)
        return rew, end

    @coroutine
    def make_generator_init(
        self,
        data_loader: DataLoader,
        num_batches_to_preload: int,
    ) -> Generator[InitialCondition, None, None]:
        """Preloads num_batches_to_preload batches' worth of (obs, act, rew/end-model burn-in
        hx/cx) at a time, then yields slices of that block on demand. self._preload_cycle
        (the exact SegmentIds of every batch in the CURRENT block) and self._preload_cursor
        (how far into the flattened block we've yielded) are kept up to date at every
        suspension point so Trainer can checkpoint them; if self._pending_resume_preload was
        set (via load_preload_state_dict) before this coroutine's first real .send() in this
        process, the FIRST block replays that saved state exactly -- same SegmentIds, same
        burn-in recomputed from the just-restored rew_end_model weights (numerically identical
        to what was saved, since burn-in is a pure deterministic forward pass) -- consuming
        ZERO draws from the BatchSampler feeding data_iterator. Every later block (once the
        replayed one is exhausted) draws fresh as normal, correctly continuing from wherever
        the BatchSampler's OWN restored RNG state (see BatchSampler.load_state_dict) left off.
        Violating the "call load_preload_state_dict before the first send()" ordering silently
        discards the pending resume state's usefulness -- it would still apply on the *next*
        block boundary reached, but the block already drawn by then used fresh random values.
        """
        num_dead = yield
        data_iterator = iter(data_loader)

        while True:
            if self._pending_resume_preload is not None:
                segment_ids_cycle, resume_cursor = self._pending_resume_preload
                self._pending_resume_preload = None
                obs_, act_, hx_, cx_ = [], [], [], []
                for segment_ids in segment_ids_cycle:
                    batch = collate_segments_to_batch([self.dataset[sid] for sid in segment_ids]).to(self.device)
                    obs, act = batch.obs, batch.act
                    with torch.no_grad():
                        *_, (hx, cx) = self.rew_end_model.predict_rew_end(obs[:, :-1], act[:, :-1], obs[:, 1:])
                    assert hx.size(0) == cx.size(0) == 1
                    obs_.extend(list(obs))
                    act_.extend(list(act))
                    hx_.extend(list(hx[0]))
                    cx_.extend(list(cx[0]))
                self._preload_cycle = segment_ids_cycle
                c = resume_cursor
            else:
                # Preload on device and burnin rew/end model
                obs_, act_, hx_, cx_ = [], [], [], []
                segment_ids_cycle = []
                for _ in range(num_batches_to_preload):
                    batch = next(data_iterator)
                    segment_ids_cycle.append(batch.segment_ids)
                    obs = batch.obs.to(self.device)
                    act = batch.act.to(self.device)
                    with torch.no_grad():
                        *_, (hx, cx) = self.rew_end_model.predict_rew_end(obs[:, :-1], act[:, :-1], obs[:, 1:])  # Burn-in of rew/end model
                    assert hx.size(0) == cx.size(0) == 1
                    obs_.extend(list(obs))
                    act_.extend(list(act))
                    hx_.extend(list(hx[0]))
                    cx_.extend(list(cx[0]))
                self._preload_cycle = segment_ids_cycle
                c = 0

            # Yield new initial conditions for dead envs
            while c + num_dead <= len(obs_):
                obs = torch.stack(obs_[c : c + num_dead])
                act = torch.stack(act_[c : c + num_dead])
                hx = torch.stack(hx_[c : c + num_dead]).unsqueeze(0)
                cx = torch.stack(cx_[c : c + num_dead]).unsqueeze(0)
                c += num_dead
                self._preload_cursor = c
                num_dead = yield obs, act, (hx, cx)
