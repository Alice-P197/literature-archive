"""Evaluation metrics (paper: Method - Evaluation metrics).

normalized MAE of a complex field:
    nm = 0.5 * ( mean|Re E - Re E_hat| / max|E| + mean|Im E - Im E_hat| / max|E| )
"""

from __future__ import annotations

import torch


@torch.no_grad()
def normalized_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """pred/target: (B,2,H,W) -> (B,)"""
    scale = target.abs().amax(dim=(-3, -2, -1)).clamp_min(1e-8)
    d = (pred - target).abs().mean(dim=(-2, -1))  # B,2 -> real / imag
    return (d.mean(dim=1)) / scale


@torch.no_grad()
def amplitude_angle_mae(pred: torch.Tensor, target: torch.Tensor):
    """returns (B,) normalised amplitude MAE and (B,) normalised angle MAE (/pi)."""
    pc = torch.complex(pred[:, 0], pred[:, 1])
    tc = torch.complex(target[:, 0], target[:, 1])
    scale = tc.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    amp = ((pc.abs() - tc.abs()).abs().mean(dim=(-2, -1), keepdim=True) / scale).squeeze()
    dth = torch.angle(pc) - torch.angle(tc)
    dth = torch.atan2(torch.sin(dth), torch.cos(dth)).abs()
    ang = (dth.mean(dim=(-2, -1)) / torch.pi)
    return amp, ang


def summarize(pred, target, labels=None):
    """returns dict of mean metrics, optionally grouped by label."""
    nm = normalized_mae(pred, target)
    amp, ang = amplitude_angle_mae(pred, target)
    out = {"norm_mae": nm.mean().item(), "amp_mae": amp.mean().item(), "angle_mae": ang.mean().item()}
    if labels is not None:
        groups = {}
        for i, l in enumerate(labels):
            groups.setdefault(l, []).append(i)
        out["groups"] = {
            k: {
                "n": len(v),
                "amp_mae": amp[v].mean().item(),
                "angle_mae": ang[v].mean().item(),
            }
            for k, v in groups.items()
        }
    return out
