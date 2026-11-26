"""Explicit MAP training of the predictor (design: docs/implement/LCG_PREDICTOR_DESIGN.md).

Full objective over the N valid replay transitions (standardized z_{t+1} targets y_i, inputs u_i):

    L_D(psi) = sum_i ||y_i - f_psi(u_i)||^2 / (2 tau^2) + lambda ||psi||^2 / 2

Minibatch objective (L_D / N, unbiased for a uniform with-replacement minibatch of B transitions):

    L_batch = sum_{i in batch} ||y_i - f_psi(u_i)||^2 / (2 tau^2 B) + lambda ||psi||^2 / (2 N)

Reductions: ||.||^2 SUMS over the D output feature dimensions and the data term AVERAGES over the B
transitions (it is D times PyTorch's element-mean MSE). The prior is an explicit penalty on every
predictor parameter (weights and biases); the optimizer applies no weight decay. tau^2 is the
feature-likelihood variance in standardized-feature units, unrelated to the diffusion noise sigma.
"""
from typing import Callable, Dict, Tuple

import numpy as np
import torch
from torch import nn, Tensor

OPTIMIZERS = ("adam", "sgd")


def squared_parameter_norm(head: nn.Module) -> Tensor:
    return sum(p.square().sum() for p in head.parameters())


def map_terms(head: nn.Module, z: Tensor, a: Tensor, y: Tensor, tau_squared: float, prior_precision: float,
              num_transitions: int) -> Tuple[Tensor, Tensor]:
    """(data term, prior term) of L_batch; their sum is the minibatch estimate of L_D / N."""
    squared = (head(z, a) - y).square().sum(-1)  # (B,) summed over output dimensions
    data = squared.mean() / (2 * tau_squared)
    prior = prior_precision * squared_parameter_norm(head) / (2 * num_transitions)
    return data, prior


def make_optimizer(head: nn.Module, cfg) -> torch.optim.Optimizer:
    """No weight decay: the prior is the explicit penalty in map_terms."""
    if cfg.optimizer == "adam":
        return torch.optim.Adam(head.parameters(), lr=cfg.lr, betas=tuple(cfg.adam_betas), eps=cfg.adam_eps,
                                weight_decay=0.0)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(head.parameters(), lr=cfg.lr, momentum=0.0, weight_decay=0.0)
    raise ValueError(f"Unknown optimizer {cfg.optimizer!r}")


def fit_map(head: nn.Module, batch: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor]], num_transitions: int,
            cfg, sample_seed: int, on_update: Callable[[], None] = None) -> Dict[str, float]:
    """cfg.fit_updates optimizer steps on L_batch; minibatches of cfg.batch_size transition indices drawn
    uniformly with replacement from a private generator (sample_seed). batch(idx) -> (z, a, y) on the
    head's device. Returns loss statistics (floats)."""
    opt = make_optimizer(head, cfg)
    rng = np.random.default_rng(sample_seed)
    head.train()
    totals, datas, priors, norms = [], [], [], []
    for _ in range(cfg.fit_updates):
        idx = torch.from_numpy(rng.integers(0, num_transitions, size=cfg.batch_size))
        z, a, y = batch(idx)
        data, prior = map_terms(head, z, a, y, cfg.tau_squared, cfg.prior_precision, num_transitions)
        loss = data + prior
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip_norm is not None:
            norms.append(torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.grad_clip_norm).detach())
        opt.step()
        totals.append(loss.detach())
        datas.append(data.detach())
        priors.append(prior.detach())
        if on_update is not None:
            on_update()
    totals, datas, priors = (torch.stack(x).double().cpu() for x in (totals, datas, priors))
    if not torch.isfinite(totals).all():
        raise FloatingPointError("Non-finite LCG-predictor MAP loss")
    logs = dict(loss_first=float(totals[0]), loss_last=float(totals[-1]), loss_last10_mean=float(totals[-10:].mean()),
                data_term_last10_mean=float(datas[-10:].mean()), prior_term_last=float(priors[-1]))
    if norms:
        logs["grad_norm_before_clip_last10_mean"] = float(torch.stack(norms[-10:]).double().mean())
    return logs


@torch.no_grad()
def full_objective(head: nn.Module, batch: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor]], num_transitions: int,
                   tau_squared: float, prior_precision: float, chunk: int = 4096) -> Dict[str, float]:
    """L_D / N over all N transitions (data part, prior part) and the mean squared error per feature
    coordinate (comparable with Plan2Explore's element-mean MSE)."""
    squared, out_dim = 0.0, None
    for start in range(0, num_transitions, chunk):
        idx = torch.arange(start, min(start + chunk, num_transitions))
        z, a, y = batch(idx)
        d = (head(z, a) - y).square()
        out_dim = d.shape[-1]
        squared += float(d.double().sum())
    data = squared / (2 * tau_squared * num_transitions)
    prior = prior_precision * float(squared_parameter_norm(head).double()) / (2 * num_transitions)
    return dict(objective_per_transition=data + prior, data_per_transition=data, prior_per_transition=prior,
                mse_per_feature=squared / (num_transitions * out_dim))
