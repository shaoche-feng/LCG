from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import torch
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader

from coroutines import coroutine
from models.diffusion import Denoiser, DiffusionSampler, DiffusionSamplerConfig
from models.rew_end_model import RewEndModel
from utils import RegionTimer, timed

ResetOutput = Tuple[torch.FloatTensor, Dict[str, Any]]
StepOutput = Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]
# obs, act, valid, rew/end-model burn-in state, preceding frame/action (feature encoder only)
InitialCondition = Tuple[Tensor, Tensor, Tensor, Optional[Tuple[Tensor, Tensor]], Optional[Tuple[Tensor, Tensor]]]
FeatureEncoder = Callable[[Tensor, Tensor, Tensor], Tensor]
# member_predictor(sampler, obs_buffer, act_buffer, noise) -> (members, b, c, h, w) completed
# next-frame predictions from shared initial noise (e.g. sampled-parameter members).
MemberPredictor = Callable[[DiffusionSampler, Tensor, Tensor, Tensor], Tensor]


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


def ensemble_disagreement(predictions: Tensor) -> Tensor:
    """predictions: (members, batch, channels, height, width) completed next frames as the
    diffusion sampler returns them: float tensors in [-1, 1] whose values lie on the uint8 grid
    (the final Euler step returns the last denoiser output, which Denoiser.wrap_model_output
    quantizes). Population variance across members, averaged over members, channels and
    pixels: mean_pixels(mean_members((p - mean_members(p))^2)) -> (batch,)."""
    centered = predictions - predictions.mean(dim=0, keepdim=True)
    return centered.square().mean(dim=0).flatten(1).mean(dim=1)


@dataclass
class WorldModelEnvConfig:
    horizon: int
    num_batches_to_preload: int
    diffusion_sampler: DiffusionSamplerConfig


class WorldModelEnv:
    """Imagination with the denoiser. Without a reward/end model (rew_end_model=None) it
    returns a zero reward placeholder and no true endings: episodes are only truncated at
    the horizon, and PMPO must supply its own (intrinsic) reward."""

    def __init__(
        self,
        denoiser: Denoiser,
        rew_end_model: Optional[RewEndModel],
        data_loader: DataLoader,
        cfg: WorldModelEnvConfig,
        return_denoising_trajectory: bool = False,
        return_imagined_candidate: bool = False,
        sample_ends: bool = True,
        ensemble: Optional[List[Denoiser]] = None,
        ensemble_noise_samples: int = 1,
        feature_encoder: Optional[FeatureEncoder] = None,
        member_predictor: Optional[MemberPredictor] = None,
    ) -> None:
        self.sampler = DiffusionSampler(denoiser, cfg.diffusion_sampler)
        # Ensemble imagination: all members predict from the same history/action with the same
        # initial noise; disagreement is reported per transition and one uniformly chosen
        # member's prediction (independently per trajectory) becomes the next observation.
        self.ensemble_samplers = None
        if ensemble is not None:
            if len(ensemble) < 2 or ensemble[0] is not denoiser:
                raise ValueError("An imagination ensemble needs >= 2 members, member 0 being the env's denoiser")
            if cfg.diffusion_sampler.s_churn != 0:
                raise ValueError("Ensemble disagreement shares the initial noise only; it needs s_churn=0")
            self.ensemble_samplers = [DiffusionSampler(d, cfg.diffusion_sampler) for d in ensemble]
        # Sampled-parameter ensembles (Laplace) give their members' predictions through
        # member_predictor; disagreement and member selection are the same as above.
        self.member_predictor = member_predictor
        if member_predictor is not None:
            if ensemble is not None:
                raise ValueError("Give either an ensemble of denoisers or a member predictor, not both")
            if cfg.diffusion_sampler.s_churn != 0:
                raise ValueError("Ensemble disagreement shares the initial noise only; it needs s_churn=0")
        self.ensemble_noise_samples = ensemble_noise_samples
        # Plan2Explore: feature_encoder(o_t, o_{t-n..t-1}, a_{t-n..t-1}) of each current frame is
        # reported per step (info["plan2explore_feature"], with the taken action). The encoder's
        # context reaches one frame further back than the sampler's window, so the frame and
        # action preceding the window are tracked (from replay at prompts: the step before the
        # prompt, or zero padding at an episode start, as in the denoiser's training batches).
        self.feature_encoder = feature_encoder
        self.last_disagreement, self.last_member = None, None
        # Monitoring only: the intrinsic-scoring work done here (Plan2Explore feature extraction;
        # ensemble/Laplace member predictions and disagreement, which also give the next frame).
        self.scoring_timer: Optional[RegionTimer] = None
        self.rew_end_model = rew_end_model
        # False: use only the reward head; imagined episodes end by horizon truncation.
        self.sample_ends = sample_ends
        self.horizon = cfg.horizon
        self.return_denoising_trajectory = return_denoising_trajectory
        self.return_imagined_candidate = return_imagined_candidate
        self.num_envs = data_loader.batch_sampler.batch_size
        self.data_loader = data_loader
        self.num_batches_to_preload = cfg.num_batches_to_preload
        self.generator_init = self.make_generator_init(data_loader, cfg.num_batches_to_preload)

    def restart_initial_conditions(self) -> None:
        """Opt-in fresh prompt block used by the feedforward PMPO controller."""
        self.generator_init.close()
        # Explicit DataLoader generator prevents iterator creation from consuming
        # the global CPU stream. Seed is drawn inside PMPO's checkpointed stream.
        self.data_loader.generator = torch.Generator().manual_seed(
            torch.randint(0, 2**31 - 1, ()).item()
        )
        # Do not preload hundreds of unused batches for a block that is discarded
        # after one rollout. Subsequent resets draw more batches on demand.
        self.generator_init = self.make_generator_init(self.data_loader, 1)

    @property
    def device(self) -> torch.device:
        return self.sampler.denoiser.device

    @torch.no_grad()
    def reset(self, **kwargs) -> ResetOutput:
        obs, act, valid, hx_cx, preceding = self.generator_init.send(self.num_envs)
        self.obs_buffer = obs
        self.act_buffer = act
        if self.feature_encoder is not None:
            self.prev_obs, self.prev_act = preceding
        if self.rew_end_model is not None:
            self.hx_rew_end, self.cx_rew_end = hx_cx
        self.ep_len = torch.zeros(self.num_envs, dtype=torch.long, device=obs.device)
        # The replay prompt (frames and padding mask) seeds the policy's frame history.
        return self.obs_buffer[:, -1], {"prompt": (obs.clone(), valid)}

    @torch.no_grad()
    def reset_dead(self, dead: torch.BoolTensor) -> Tuple[Tensor, Tensor]:
        """Returns the new replay prompts of the dead rows."""
        obs, act, valid, hx_cx, preceding = self.generator_init.send(dead.sum().item())
        self.obs_buffer[dead] = obs
        self.act_buffer[dead] = act
        if self.feature_encoder is not None:
            self.prev_obs[dead], self.prev_act[dead] = preceding
        if self.rew_end_model is not None:
            self.hx_rew_end[:, dead], self.cx_rew_end[:, dead] = hx_cx
        self.ep_len[dead] = 0
        return obs, valid

    @torch.no_grad()
    def step(self, act: torch.Tensor) -> StepOutput:
        self.act_buffer[:, -1] = act

        if self.return_imagined_candidate:
            candidate_x_obs = self.obs_buffer.clone()
            candidate_x_act = self.act_buffer.clone()

        feature = None
        if self.feature_encoder is not None:
            # Current frame o_t with o_{t-n..t-1} and a_{t-n..t-1}; the action a_t just taken
            # is not part of it (it goes to the predictors separately). No random numbers.
            context = torch.cat((self.prev_obs.unsqueeze(1), self.obs_buffer[:, :-1]), dim=1)
            context_act = torch.cat((self.prev_act.unsqueeze(1), self.act_buffer[:, :-1]), dim=1)
            with timed(self.scoring_timer):
                feature = self.feature_encoder(self.obs_buffer[:, -1], context, context_act)

        next_obs, denoising_trajectory = self.predict_next_obs()
        if self.rew_end_model is None:
            rew = torch.zeros(self.num_envs, device=next_obs.device)
            end = torch.zeros(self.num_envs, dtype=torch.long, device=next_obs.device)
        else:
            rew, end = self.predict_rew_end(next_obs.unsqueeze(1))

        self.ep_len += 1
        trunc = (self.ep_len >= self.horizon).long()

        if self.feature_encoder is not None:
            self.prev_obs, self.prev_act = self.obs_buffer[:, 0].clone(), self.act_buffer[:, 0].clone()
        self.obs_buffer = self.obs_buffer.roll(-1, dims=1)
        self.act_buffer = self.act_buffer.roll(-1, dims=1)
        self.obs_buffer[:, -1] = next_obs

        dead = torch.logical_or(end, trunc)

        info = {}
        if self.return_denoising_trajectory:
            info["denoising_trajectory"] = torch.stack(denoising_trajectory, dim=1)

        if self.return_imagined_candidate:
            info["imagined_candidate"] = ImaginedCandidate(candidate_x_obs, candidate_x_act, next_obs.clone())

        if self.ensemble_samplers is not None or self.member_predictor is not None:
            info["ensemble_disagreement"] = self.last_disagreement
            info["ensemble_member"] = self.last_member

        if feature is not None:
            info["plan2explore_feature"] = feature
            info["plan2explore_action"] = act.clone()

        if dead.any():
            # final_observation ends the old trajectory; the returned observation,
            # burnin_obs and prompt belong to the new replay prompt.
            info["prompt"] = self.reset_dead(dead)
            info["final_observation"] = next_obs[dead]
            info["burnin_obs"] = self.obs_buffer[dead, :-1]

        return self.obs_buffer[:, -1], rew, end, trunc, info

    @torch.no_grad()
    def imagine_candidates(self, obs: Tensor, act: Tensor, actions: Tensor) -> List[ImaginedCandidate]:
        """Open-loop imagination from given prompts, outside the env's own rollout: obs
        (b, t, C, H, W) and act (b, t, A) prompt windows, actions (horizon, b, A), one per step.
        Each step's candidate is what step() reports with return_imagined_candidate=True: the
        window the sampler was conditioned on (with the step's action) and its prediction.
        Single-denoiser sampler and the global RNG; the env's buffers and prompt stream are
        untouched (used for LCG's reference rollouts, not by PMPO)."""
        obs_buffer, act_buffer = obs.clone(), act.clone()
        candidates = []
        for step_act in actions:
            act_buffer[:, -1] = step_act
            next_obs = self.sampler.sample(obs_buffer, act_buffer)[0]
            candidates.append(ImaginedCandidate(obs_buffer.clone(), act_buffer.clone(), next_obs.clone()))
            obs_buffer = obs_buffer.roll(-1, dims=1)
            act_buffer = act_buffer.roll(-1, dims=1)
            obs_buffer[:, -1] = next_obs
        return candidates

    @torch.no_grad()
    def predict_next_obs(self) -> Tuple[Tensor, List[Tensor]]:
        if self.ensemble_samplers is None and self.member_predictor is None:
            return self.sampler.sample(self.obs_buffer, self.act_buffer)
        with timed(self.scoring_timer):
            return self.predict_next_obs_ensemble(), []

    @torch.no_grad()
    def predict_next_obs_ensemble(self) -> Tensor:
        """Disagreement for each shared-noise sample, averaged over samples; the next
        observation is taken from the first noise sample (fixed before any score is seen), from
        a member drawn uniformly and independently for every trajectory at every transition."""
        b, _, c, h, w = self.obs_buffer.shape
        scores, first = [], None
        for s in range(self.ensemble_noise_samples):
            noise = torch.randn(b, c, h, w, device=self.obs_buffer.device)
            if self.member_predictor is not None:
                predictions = self.member_predictor(self.sampler, self.obs_buffer, self.act_buffer, noise)
            else:
                predictions = torch.stack([smp.sample(self.obs_buffer, self.act_buffer, noise)[0]
                                           for smp in self.ensemble_samplers])  # (members, b, c, h, w)
            scores.append(ensemble_disagreement(predictions))
            if s == 0:
                first = predictions
        member = torch.randint(first.shape[0], (b,), device=self.obs_buffer.device)
        self.last_disagreement = torch.stack(scores).mean(dim=0)
        self.last_member = member
        return first[member, torch.arange(b, device=member.device)]

    @torch.no_grad()
    def predict_rew_end(self, next_obs: Tensor) -> Tuple[Tensor, Tensor]:
        rew_out, logits_end, (self.hx_rew_end, self.cx_rew_end) = self.rew_end_model.predict_rew_end(
            self.obs_buffer[:, -1:],
            self.act_buffer[:, -1:],
            next_obs,
            (self.hx_rew_end, self.cx_rew_end),
        )
        rew = rew_out.squeeze(1)  # Predicted scalar reward used directly.
        if not self.sample_ends:
            return rew, torch.zeros(rew.shape, dtype=torch.long, device=rew.device)
        end = Categorical(logits=logits_end).sample().squeeze(1)
        return rew, end

    @coroutine
    def make_generator_init(
        self,
        data_loader: DataLoader,
        num_batches_to_preload: int,
    ) -> Generator[InitialCondition, None, None]:
        num_dead = yield
        data_iterator = iter(data_loader)

        while True:
            # Preload on device and burnin rew/end model
            obs_, act_, valid_, hx_, cx_, prev_obs_, prev_act_ = [], [], [], [], [], [], []
            for _ in range(num_batches_to_preload):
                batch = next(data_iterator)
                obs = batch.obs.to(self.device)
                act = batch.act.to(self.device)
                valid = batch.mask_padding.to(self.device)
                obs_.extend(list(obs))
                act_.extend(list(act))
                valid_.extend(list(valid))
                if self.feature_encoder is not None:
                    prev_obs, prev_act = self.preceding_step(batch)
                    prev_obs_.extend(list(prev_obs))
                    prev_act_.extend(list(prev_act))
                if self.rew_end_model is not None:
                    with torch.no_grad():
                        *_, (hx, cx) = self.rew_end_model.predict_rew_end(obs[:, :-1], act[:, :-1], obs[:, 1:])  # Burn-in of rew/end model
                    assert hx.size(0) == cx.size(0) == 1
                    hx_.extend(list(hx[0]))
                    cx_.extend(list(cx[0]))

            # Yield new initial conditions for dead envs
            c = 0
            while c + num_dead <= len(obs_):
                obs = torch.stack(obs_[c : c + num_dead])
                act = torch.stack(act_[c : c + num_dead])
                valid = torch.stack(valid_[c : c + num_dead])
                hx_cx = None
                if self.rew_end_model is not None:
                    hx = torch.stack(hx_[c : c + num_dead]).unsqueeze(0)
                    cx = torch.stack(cx_[c : c + num_dead]).unsqueeze(0)
                    hx_cx = (hx, cx)
                preceding = None
                if self.feature_encoder is not None:
                    preceding = (torch.stack(prev_obs_[c : c + num_dead]), torch.stack(prev_act_[c : c + num_dead]))
                c += num_dead
                num_dead = yield obs, act, valid, hx_cx, preceding

    def preceding_step(self, batch) -> Tuple[Tensor, Tensor]:
        """Frame and action of the replay step just before each prompt window (zeros when the
        window is left-padded or starts its episode). Reads only obs/act of the replay episode."""
        dataset = self.data_loader.dataset
        b, _, c, h, w = batch.obs.shape
        prev_obs = torch.zeros(b, c, h, w, dtype=batch.obs.dtype)
        prev_act = torch.zeros(b, batch.act.shape[-1], dtype=batch.act.dtype)
        for i, segment_id in enumerate(batch.segment_ids):
            if bool(batch.mask_padding[i, 0]) and segment_id.start > 0:
                episode = dataset.load_episode(segment_id.episode_id)
                prev_obs[i] = episode.obs[segment_id.start - 1]
                prev_act[i] = episode.act[segment_id.start - 1]
        return prev_obs.to(self.device), prev_act.to(self.device)
