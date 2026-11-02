"""Replay-value learning support for the scalar-EMA Gaussian-PMPO controller (stage3-gaussian-h10-ema-replayvalue).

The functions and classes below are copied VERBATIM from the reviewed DreamerV3 controller
(LCG-stage3-dreamerv3-controller/src/models/dreamerv3_controller.py, sha256 3b0415974770b24ba83ea36ffba458762892f995dfae64211e13ec5e4998640d),
whose references are dreamerv3/agent.py (lambda_return, repl_loss) of github.com/danijar/dreamerv3 at
e01491fad6434b2245a3b8ca201dd7faedcc458c. tests/models/test_replay_value.py checks them against the reference module.

Sampling (only when agent.actor_critic.replay_value_coef > 0): each controller update draws `segments` replay segments of
window + starts - 1 frames with BatchSampler's own placement rule and sample weights (episode chosen with the configured
weights, time step uniform within the episode, padding only before an episode's start; numpy RNG inside the controller's
private checkpointed imagination stream), the segment end raised to at least `starts` frames so every imagination start is
a real frame. Each segment yields `starts` consecutive imagination prompts (segments * starts = 32 prompts).
"""
import torch

from data import Batch, BatchSampler, collate_segments_to_batch


def replay_lambda_return(rew, term, last, boot, gamma, lam):
    """agent.py lambda_return as used by repl_loss (val argument unused there), re-indexed to this
    codebase's transition convention. Replay states t = 0..K-1; rew[t], term[t], last[t] describe
    the transition leaving state t (reference rew/term/last at index t+1); boot[t] is the
    imagination lambda-return started at replay state t. Returns R_t for t = 0..K-2:
      R_{K-1} = boot_{K-1};  R_t = rew_t + live_t * ((1 - cont_t) * boot_{t+1} + cont_t * R_{t+1}),
      live_t = gamma * (1 - term_t),  cont_t = lam * (1 - last_t)."""
    K = boot.shape[1]
    live = gamma * (1 - term[:, :K - 1].float())
    cont = lam * (1 - last[:, :K - 1].float())
    rets = [boot[:, -1]]
    for t in reversed(range(K - 1)):
        rets.append(rew[:, t] + live[:, t] * ((1 - cont[:, t]) * boot[:, t + 1] + cont[:, t] * rets[-1]))
    return torch.stack(list(reversed(rets))[:-1], 1)


# ----------------------------------------------------------------------------------------------
# Replay windows: consecutive imagination starts per replay segment (needed by repl_loss)
# ----------------------------------------------------------------------------------------------

class ReplayStartSampler(BatchSampler):
    """BatchSampler's own segment placement (padding only before an episode's start, same RNG
    draws), for segments of window + starts - 1 frames; the stop is raised to at least `starts`
    so that every imagination start frame is a real frame (never padding)."""

    def __init__(self, dataset, segments, window, starts, sample_weights=None):
        super().__init__(dataset, 0, 1, segments, window + starts - 1, sample_weights, can_sample_beyond_end=False)
        self.window, self.starts = window, starts

    def sample(self):
        ids = super().sample()
        out = []
        for i in ids:
            stop = max(i.stop, self.starts)
            if stop > self.dataset.lengths[i.episode_id]:
                raise ValueError("Replay episode shorter than the number of imagination starts")
            out.append(type(i)(i.episode_id, stop - self.seq_length, stop))
        return out


def unfold_windows(batch: Batch, window: int, starts: int) -> Batch:
    """(B, window+starts-1) segments -> (B*starts, window) prompts, segment-major: prompt b*starts+k
    ends at segment frame window-1+k (the k-th consecutive replay start of segment b)."""
    def cut(x):
        return torch.stack([x[:, k:k + window] for k in range(starts)], 1).flatten(0, 1)
    return Batch(cut(batch.obs), cut(batch.act), cut(batch.rew), cut(batch.end), cut(batch.trunc),
                 cut(batch.mask_padding), [i for i in batch.info for _ in range(starts)],
                 [s for s in batch.segment_ids for _ in range(starts)])


class ReplayWindowLoader:
    """Stands in for the actor-critic DataLoader inside WorldModelEnv. Each new iterator (one per
    PMPO-style prompt block, i.e. per controller update) yields unfolded prompt batches; the FIRST
    segment batch of each iterator -- the one whose 32 prompts WorldModelEnv.reset() consumes in
    order -- is kept as `replay` for the replay-value loss. Single-process, like the original."""

    num_workers = 0

    def __init__(self, dataset, segments, window, starts, sample_weights=None):
        self.dataset = dataset
        # WorldModelEnv read num_envs (= segments * starts prompts) from the original loader at
        # construction; this sampler draws `segments` segments per batch.
        self.batch_sampler = ReplayStartSampler(dataset, segments, window, starts, sample_weights)
        self.window, self.starts, self.segments = window, starts, segments
        self.generator = None  # set by WorldModelEnv.restart_initial_conditions; sampling uses numpy
        self.replay, self.blocks = None, 0

    def __iter__(self):
        first = True
        while True:
            ids = self.batch_sampler.sample()
            batch = collate_segments_to_batch([self.dataset[i] for i in ids])
            if first:
                self.replay, first = batch, False
                self.blocks += 1
            yield unfold_windows(batch, self.window, self.starts)
