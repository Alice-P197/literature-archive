"""Loss functions, Eq. (1)-(3) of the paper.

    L_total = L_A + alpha * L_theta
    L_A     = mean |E - E_hat|_1                      (amplitude/field MAE)
    L_theta = mean (1 - cos(theta - theta_hat)) / 2   (periodic phase loss)

alpha = 4*sqrt(pi) ~ 7.09 is the paper's recommendation for balancing the two
terms with equal expectation. We expose it as `alpha`; the default 1.0 keeps the
two reported curves (amplitude loss ~0.09, angle loss ~0.11) on the same scale as
the paper's Fig. 3b, since our L_theta is already normalised to [0, 1].
"""

from __future__ import annotations

import math

import torch


def field_amplitude_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def phase_loss(pred_c: torch.Tensor, target_c: torch.Tensor) -> torch.Tensor:
    dtheta = torch.angle(pred_c) - torch.angle(target_c)
    return (1.0 - torch.cos(dtheta)).mean() / 2.0


def to_complex(x: torch.Tensor) -> torch.Tensor:
    """(...,2,H,W) -> complex (...,H,W)."""
    return torch.complex(x[..., 0, :, :], x[..., 1, :, :])


def metaformer_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    amp_weighted_phase: bool = False,
    return_parts: bool = False,
):
    la = field_amplitude_loss(pred, target)
    pc, tc = to_complex(pred), to_complex(target)
    if amp_weighted_phase:
        w = tc.abs()
        w = w / w.mean().clamp_min(1e-6)
        dtheta = torch.angle(pc) - torch.angle(tc)
        lt = ((1.0 - torch.cos(dtheta)) / 2.0 * w).mean()
    else:
        lt = phase_loss(pc, tc)
    total = la + alpha * lt
    if return_parts:
        return total, la.detach(), lt.detach()
    return total
