from typing import Callable, Dict, List, Tuple

from torch import Tensor

from models.diffusion.denoiser import Denoiser

from .forward_jvp import JVPBank, score_one_jvp_bank

Candidate = Tuple[Tensor, Tensor, Tensor]  # (x_obs_flat, x_act, y_star), each batch size 1


def imagined_candidates_from_batch(x_obs: Tensor, x_act: Tensor, y_star: Tensor) -> List[Candidate]:
    """Bridges envs.world_model_env.ImaginedCandidate's batched fields to a list of
    per-candidate tuples for scoring. x_obs: (num_envs, t, C, H, W); x_act: (num_envs, t,
    ...); y_star: (num_envs, C, H, W) -- exactly WorldModelEnv's own buffer/output shapes,
    unflattened. Slices into num_envs single-candidate tuples shaped for
    score_one_jvp_bank: x_obs_flat (1, t*C, H, W), x_act (1, t, ...), y_star (1, C, H, W).
    """
    num_envs = x_obs.size(0)
    candidates = []
    for i in range(num_envs):
        obs_i = x_obs[i : i + 1]
        act_i = x_act[i : i + 1]
        y_i = y_star[i : i + 1]
        obs_flat = obs_i.reshape(1, -1, obs_i.shape[-2], obs_i.shape[-1])
        candidates.append((obs_flat, act_i, y_i))
    return candidates


def make_lcg_intrinsic_reward_fn(
    denoiser: Denoiser,
    theta_s_named: Dict[str, Tensor],
    frozen_named: Dict[str, Tensor],
    h_D: Tensor,
    bank: JVPBank,
    chunk_size: int = 16,
) -> Callable[[List[Dict], Tensor], Tensor]:
    """Builds an ActorCritic.intrinsic_reward_fn(infos, env_rew) -> Tensor hook (see
    ActorCritic.set_intrinsic_reward_fn) using the production forward-mode JVP estimator
    (genuine torch.func.jvp, simple IID Monte Carlo over the full training sigma
    distribution, Full CRN, M=bank.num_samples): scores every rollout step's
    info["imagined_candidate"] (populated by WorldModelEnv when constructed with
    return_imagined_candidate=True) against the frozen (denoiser, h_D, bank).
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

        h_D_inv_sqrt = h_D.rsqrt()
        scores_flat = score_one_jvp_bank(
            denoiser, theta_s_named, frozen_named, theta_s_named, h_D, h_D_inv_sqrt, bank, all_candidates, chunk_size
        )
        scores_flat = scores_flat.to(device=env_rew.device, dtype=env_rew.dtype)
        return scores_flat.view(num_steps, num_envs).transpose(0, 1).contiguous()

    return intrinsic_reward_fn
