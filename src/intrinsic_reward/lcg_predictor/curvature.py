"""Exact output-Jacobian quantities of the predictor MLP f_psi(u), u = (z_t, a_t), for all of its
parameters psi (design: docs/implement/LCG_PREDICTOR_DESIGN.md).

The network is PredictorHead.net: Linear, activation, ..., Linear (linear output). For a sample u with
layer inputs h_0 = u, h_l = act(a_l), a_l = W_l h_{l-1} + b_l, and the output-Jacobian column norms
c_l[p] = ||d f(u) / d a_l[p]||^2 of each hidden pre-activation, the diagonal of J^T J (J = d f / d psi,
the full output dimension) is, per parameter:

    hidden W_l[p, q]: c_l[p] * h_{l-1}[q]^2        hidden b_l[p]: c_l[p]
    output W_L[k, j]: h_{L-1}[j]^2                   output b_L[k]: 1

c_l comes from the backward recursion P = W_L^T W_L, Q_l = diag(act'(a_l)) P diag(act'(a_l)),
c_l = diag(Q_l), P <- W_l^T Q_l W_l. This is the exact generalized Gauss-Newton diagonal of a Gaussian
likelihood with unit variance (the caller divides by tau^2); no probes, no squared loss gradients.
The same pieces give the exact diagonal-posterior trace sum_j ||J[:, j]||^2 / h[j] without probes.

ggn_diagonal_autograd is the generic reference (per-sample VJPs against every output basis vector):
exact for any architecture, slow; used as a backend option and to cross-check the closed form.
"""
from typing import Dict, List, Sequence, Tuple

import torch
from torch import func, nn, Tensor
import torch.nn.functional as F

from intrinsic_reward.plan2explore import ACTIVATIONS


def activation_derivative(name: str, pre: Tensor) -> Tensor:
    """d act / d a at pre-activations pre (the PredictorHead activations with default settings)."""
    if name == "silu":
        s = torch.sigmoid(pre)
        return s * (1 + pre * (1 - s))
    if name == "elu":  # alpha = 1
        return torch.where(pre > 0, torch.ones_like(pre), torch.exp(pre))
    if name == "relu":
        return (pre > 0).to(pre.dtype)
    raise ValueError(f"No derivative for activation {name!r}")


def mlp_structure(head: nn.Module, activation: str) -> List[nn.Linear]:
    """The Linear layers of a PredictorHead, checked to be Linear, act, ..., Linear with the given
    activation and default settings; parameters named net.<i>.weight / net.<i>.bias in that order."""
    modules = list(head.net)
    act_type = ACTIVATIONS[activation]
    for i, m in enumerate(modules):
        expected = nn.Linear if i % 2 == 0 else act_type
        if type(m) is not expected:
            raise ValueError(f"Predictor layer {i} is {type(m).__name__}, expected {expected.__name__}")
    if len(modules) % 2 != 1:
        raise ValueError("The predictor must end with a Linear layer")
    if activation == "elu" and any(m.alpha != 1.0 for m in modules[1::2]):
        raise ValueError("Closed-form curvature assumes ELU alpha = 1")
    linears = modules[0::2]
    if any(lin.bias is None for lin in linears):
        raise ValueError("Closed-form curvature assumes every Linear layer has a bias")
    names = [n for n, _ in head.named_parameters()]
    expected_names = [f"net.{2 * i}.{kind}" for i in range(len(linears)) for kind in ("weight", "bias")]
    if names != expected_names:
        raise ValueError(f"Unexpected predictor parameter order {names}")
    return linears


def forward_parts(linears: Sequence[nn.Linear], activation: str, u: Tensor,
                  params: Dict[str, Tensor] = None) -> Tuple[List[Tensor], List[Tensor]]:
    """(layer inputs [h_0 = u, h_1, ..., h_{L-1}], hidden pre-activations [a_1, ..., a_{L-1}]).
    params (optional): name -> tensor overriding the modules' parameters (e.g. a float64 copy)."""
    act = ACTIVATIONS[activation]()
    inputs, pres, h = [u], [], u
    for i, lin in enumerate(linears[:-1]):
        w, b = _wb(lin, i, params)
        a = F.linear(h, w, b)
        h = act(a)
        pres.append(a)
        inputs.append(h)
    return inputs, pres


def _wb(lin: nn.Linear, i: int, params) -> Tuple[Tensor, Tensor]:
    if params is None:
        return lin.weight, lin.bias
    return params[f"net.{2 * i}.weight"], params[f"net.{2 * i}.bias"]


def hidden_column_norms(linears: Sequence[nn.Linear], activation: str, pres: Sequence[Tensor],
                        params: Dict[str, Tensor] = None) -> List[Tensor]:
    """c_l (B, width_l): squared output-Jacobian column norms of each hidden pre-activation."""
    n = len(linears)
    w_out, _ = _wb(linears[-1], n - 1, params)
    P = w_out.T @ w_out  # (w, w), shared by the batch
    cs: List[Tensor] = [None] * len(pres)
    for layer in reversed(range(len(pres))):
        d = activation_derivative(activation, pres[layer])  # (B, w_l)
        cs[layer] = d.square() * P.diagonal(dim1=-2, dim2=-1)
        if layer > 0:
            Q = d.unsqueeze(-1) * P * d.unsqueeze(-2)  # (B, w_l, w_l)
            w, _ = _wb(linears[layer], layer, params)  # (w_l, w_{l-1})
            P = w.T @ Q @ w  # (B, w_{l-1}, w_{l-1})
    return cs


def ggn_diagonal_closed_form(head: nn.Module, activation: str, u: Tensor, chunk_size: int = 64,
                             accumulate_dtype: torch.dtype = torch.float64) -> Tensor:
    """sum_i diag(J_i^T J_i) over the rows of u (N, in), flattened in head.named_parameters() order."""
    linears = mlp_structure(head, activation)
    acc = None
    with torch.no_grad():
        for start in range(0, u.shape[0], chunk_size):
            x = u[start:start + chunk_size]
            inputs, pres = forward_parts(linears, activation, x)
            cs = hidden_column_norms(linears, activation, pres)
            parts = []
            for layer in range(len(linears) - 1):
                parts += [(cs[layer].T @ inputs[layer].square()).reshape(-1), cs[layer].sum(0)]
            out_dim = linears[-1].out_features
            parts += [inputs[-1].square().sum(0).unsqueeze(0).expand(out_dim, -1).reshape(-1),
                      torch.full((out_dim,), float(x.shape[0]), device=x.device, dtype=x.dtype)]
            chunk = torch.cat(parts).to(accumulate_dtype)
            acc = chunk if acc is None else acc + chunk
    return acc


def ggn_diagonal_autograd(head: nn.Module, u: Tensor, output_chunk: int = 128, sample_chunk: int = 4,
                          accumulate_dtype: torch.dtype = torch.float64) -> Tensor:
    """Generic reference: for every sample, VJPs of f(u_i) against all output basis vectors, squared and
    summed. Same layout as ggn_diagonal_closed_form; works for any head whose forward takes (u,) via
    head.net (the concatenated input)."""
    params = {k: v.detach() for k, v in head.named_parameters()}
    names = list(params)
    out_dim = head.net[-1].out_features
    eye = torch.eye(out_dim, device=u.device, dtype=u.dtype)

    def per_sample(x):
        _, vjp_fn = func.vjp(lambda p: func.functional_call(head.net, _strip(p), (x[None],))[0], params)
        res = {k: torch.zeros_like(v) for k, v in params.items()}
        for k0 in range(0, out_dim, output_chunk):
            (grads,) = func.vmap(vjp_fn)(eye[k0:k0 + output_chunk])
            for k in names:
                res[k] = res[k] + grads[k].square().sum(0)
        return res

    acc = {k: torch.zeros(v.shape, device=u.device, dtype=accumulate_dtype) for k, v in params.items()}
    for start in range(0, u.shape[0], sample_chunk):
        r = func.vmap(per_sample)(u[start:start + sample_chunk])
        for k in names:
            acc[k] += r[k].sum(0).to(accumulate_dtype)
    return torch.cat([acc[k].reshape(-1) for k in names])


def _strip(params: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """net.<i>.<kind> -> <i>.<kind> (functional_call on head.net)."""
    return {k[len("net."):]: v for k, v in params.items()}


def exact_trace(head: nn.Module, activation: str, u: Tensor, inv_h: Dict[str, Tensor],
                params: Dict[str, Tensor] = None) -> Tensor:
    """(B,) sum_j ||J(u_b)[:, j]||^2 * inv_h[j] over all parameters j; inv_h: name -> 1/h per parameter
    (shapes of the parameters). params optionally overrides the weights (e.g. a float64 copy)."""
    linears = mlp_structure(head, activation)
    inputs, pres = forward_parts(linears, activation, u, params)
    cs = hidden_column_norms(linears, activation, pres, params)
    r = torch.zeros(u.shape[0], device=u.device, dtype=u.dtype)
    for layer in range(len(linears) - 1):
        iw, ib = inv_h[f"net.{2 * layer}.weight"], inv_h[f"net.{2 * layer}.bias"]
        r = r + ((cs[layer] @ iw) * inputs[layer].square()).sum(1) + cs[layer] @ ib
    last = 2 * (len(linears) - 1)
    r = r + (inputs[-1].square() * inv_h[f"net.{last}.weight"].sum(0)).sum(1) + inv_h[f"net.{last}.bias"].sum()
    return r
