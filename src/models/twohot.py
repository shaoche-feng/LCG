"""DreamerV3-style two-hot value distribution (stage3-gaussian-h10-twohot-emacritic).

symlog / symexp / symexp_bins / twohot_encode / twohot_loss / twohot_mean are copied verbatim from the reviewed
DreamerV3 controller (LCG-stage3-dreamerv3-controller, src/models/dreamerv3_controller.py), whose references are
embodied/jax/heads.py (symexp_twohot bins) and embodied/jax/outs.py (TwoHot loss and symmetric pred) of
github.com/danijar/dreamerv3 at e01491fad6434b2245a3b8ca201dd7faedcc458c. tests/models/test_twohot_critic.py checks
them against explicit calculations and, when that worktree is present, against the Dreamer module itself.
"""
import torch
from torch import nn


def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
    return torch.sign(x) * torch.expm1(x.abs())


def symexp_bins(num_bins: int = 255) -> torch.Tensor:
    """heads.py symexp_twohot: symexp(linspace(-20, 0)) mirrored; raw-space bin locations."""
    if num_bins % 2 == 1:
        half = symexp(torch.linspace(-20, 0, (num_bins - 1) // 2 + 1, dtype=torch.float32))
        return torch.cat([half, -half[:-1].flip(0)])
    half = symexp(torch.linspace(-20, 0, num_bins // 2, dtype=torch.float32))
    return torch.cat([half, -half.flip(0)])


def twohot_encode(target: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """outs.TwoHot.loss target construction (no squash: targets live in raw space)."""
    target = target.detach().float()
    n = len(bins)
    below = (bins <= target[..., None]).long().sum(-1) - 1
    above = n - (bins > target[..., None]).long().sum(-1)
    below, above = below.clamp(0, n - 1), above.clamp(0, n - 1)
    equal = below == above
    one = torch.ones_like(target)
    dist_below = torch.where(equal, one, (bins[below] - target).abs())
    dist_above = torch.where(equal, one, (bins[above] - target).abs())
    total = dist_below + dist_above
    weight_below, weight_above = dist_above / total, dist_below / total
    return (nn.functional.one_hot(below, n) * weight_below[..., None]
            + nn.functional.one_hot(above, n) * weight_above[..., None])


def twohot_loss(logits: torch.Tensor, target: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against the two-hot target; per element (no reduction)."""
    log_pred = logits.float() - torch.logsumexp(logits.float(), -1, keepdim=True)
    return -(twohot_encode(target, bins) * log_pred).sum(-1)


def twohot_mean(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """outs.TwoHot.pred: expectation with the reference's symmetric summation (exactly 0 when
    the probabilities are uniform and the bins symmetric)."""
    probs = torch.softmax(logits.float(), -1)
    n = probs.shape[-1]
    if n % 2 == 1:
        m = (n - 1) // 2
        p1, p2, p3 = probs[..., :m], probs[..., m:m + 1], probs[..., m + 1:]
        b1, b2, b3 = bins[:m], bins[m:m + 1], bins[m + 1:]
        return (p2 * b2).sum(-1) + ((p1 * b1).flip(-1) + p3 * b3).sum(-1)
    p1, p2, b1, b2 = probs[..., :n // 2], probs[..., n // 2:], bins[:n // 2], bins[n // 2:]
    return ((p1 * b1).flip(-1) + p2 * b2).sum(-1)
