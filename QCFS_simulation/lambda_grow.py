"""Simple IF-threshold growth regularizer.

Used by the threshold-scaling baseline ``l2wo_lambda_grow``:

    R_λ = - mean_l log(λ_l)

This only pushes IF thresholds up. It does not couple to weights, BN-γ,
or the MNE numerator. Combine it with weights-only L2 in the optimizer.
"""
from __future__ import annotations

import torch

from Models.layer import IF


def compute_lambda_grow_regularization(
    model,
    T=None,
    quant_level=None,
    eps: float = 1e-3,
):
    terms = []
    for module in model.modules():
        thresh = getattr(module, "thresh", None)
        if not isinstance(module, IF) or thresh is None:
            continue
        if not thresh.requires_grad:
            continue
        lam = thresh.reshape(-1)[0].clamp(min=eps)
        terms.append(-torch.log(lam))
    if not terms:
        param = next(model.parameters(), None)
        if param is None:
            return torch.tensor(0.0)
        return torch.zeros((), device=param.device, dtype=param.dtype)
    return torch.stack([term.reshape(()) for term in terms]).mean()
