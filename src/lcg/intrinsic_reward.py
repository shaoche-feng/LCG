from typing import Callable, Dict, List, Tuple

import torch
from torch import Tensor
import torch.nn as nn

from models.diffusion.denoiser import Denoiser

from .batched_scoring import imagined_candidates_from_batch
from .batched_vjp import score_candidates_batched
from .crn import CRNBank
from .forward_jvp import JVPBank, score_one_jvp_bank


def make_lcg_intrinsic_reward_fn(
    denoiser: Denoiser,
    params: List[nn.Parameter],
    h_D: Tensor,
    banks: Tuple[CRNBank, ...],
    chunk_size: int = 4,
) -> Callable[[List[Dict], Tensor], Tensor]:
    """Builds an ActorCritic.intrinsic_reward_fn(infos, env_rew) -> Tensor hook (see
    ActorCritic.set_intrinsic_reward_fn) that scores every rollout step's
    info["imagined_candidate"] (populated by WorldModelEnv when constructed with
    return_imagined_candidate=True) against the frozen (denoiser, h_D, banks), using the
    validated Stage-4.5 batched CRN scorer -- no different VJP/estimator, CRN semantics,
    h_D, or K/num_strata are introduced here.

    All H rollout steps' candidates (H x num_envs total, all sharing the same frozen
    denoiser/h_D/banks) are flattened into a single list and scored with ONE
    score_candidates_batched call (chunk_size candidates per chunk), rather than one call
    per step -- purely a batching/throughput change (Stage 5B Part A); the math and the
    per-candidate probes are identical to the Stage 5A per-step version. The flat result is
    reshaped back to env_rew's (num_envs, num_steps) layout, preserving temporal/env order:
    flat index t*num_envs + b holds step t / env b (matching how `infos` is ordered), so
    `.view(num_steps, num_envs).transpose(0, 1)` recovers env_rew[b, t].

    denoiser/h_D/banks are captured by reference and only ever read, never modified. Every
    (env, step) imagined transition is scored exactly once -- no resampling or redundant
    recomputation.
    """

    def intrinsic_reward_fn(infos: List[Dict], env_rew: Tensor) -> Tensor:
        num_envs, num_steps = env_rew.shape
        all_candidates = []
        for t in range(num_steps):
            assert "imagined_candidate" in infos[t], (
                "make_lcg_intrinsic_reward_fn requires WorldModelEnv(..., return_imagined_candidate=True)"
            )
            batch = infos[t]["imagined_candidate"]
            all_candidates.extend(imagined_candidates_from_batch(batch.x_obs, batch.x_act, batch.y_star))

        scores_flat = score_candidates_batched(denoiser, params, h_D, banks, all_candidates, chunk_size)
        scores_flat = scores_flat.to(device=env_rew.device, dtype=env_rew.dtype)
        return scores_flat.view(num_steps, num_envs).transpose(0, 1).contiguous()

    return intrinsic_reward_fn


def make_lcg_forward_jvp_intrinsic_reward_fn(
    denoiser: Denoiser,
    theta_s_named: Dict[str, Tensor],
    frozen_named: Dict[str, Tensor],
    h_D: Tensor,
    bank: JVPBank,
    chunk_size: int = 4,
) -> Callable[[List[Dict], Tensor], Tensor]:
    """Production candidate scorer (Stage 6): builds an ActorCritic.intrinsic_reward_fn
    hook using the validated forward-mode JVP estimator (genuine torch.func.jvp, simple
    IID Monte Carlo over the full training sigma distribution, Full CRN, M=len(bank))
    instead of the backward/VJP 3-stratum scorer above. Same rollout-flattening/reshaping
    convention as make_lcg_intrinsic_reward_fn (H x num_envs candidates scored in one
    score_one_jvp_bank call, reshaped back to env_rew's (num_envs, num_steps) layout) --
    only the per-candidate scoring backend differs.

    theta_s_named/frozen_named/h_D/bank are captured by reference and only ever read,
    never modified or resampled here -- refresh cadence (when a new bank/h_D is built) is
    owned entirely by LCGLifecycle.refresh(), unchanged from the backward/VJP path.
    """

    def intrinsic_reward_fn(infos: List[Dict], env_rew: Tensor) -> Tensor:
        num_envs, num_steps = env_rew.shape
        all_candidates = []
        for t in range(num_steps):
            assert "imagined_candidate" in infos[t], (
                "make_lcg_forward_jvp_intrinsic_reward_fn requires "
                "WorldModelEnv(..., return_imagined_candidate=True)"
            )
            batch = infos[t]["imagined_candidate"]
            all_candidates.extend(imagined_candidates_from_batch(batch.x_obs, batch.x_act, batch.y_star))

        h_D_inv_sqrt = h_D.rsqrt()
        scores_flat = score_one_jvp_bank(
            denoiser, theta_s_named, frozen_named, theta_s_named, h_D, h_D_inv_sqrt, bank, all_candidates, chunk_size
        )
        scores_flat = scores_flat.to(device=env_rew.device, dtype=env_rew.dtype)
        return scores_flat.view(num_steps, num_envs).transpose(0, 1).contiguous()

    return intrinsic_reward_fn
