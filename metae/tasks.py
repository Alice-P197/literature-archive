"""Target wavefronts for inverse design (paper Eq. 7 and Fig. 5/6).

All tasks are expressed as a target phase on the output plane (z = lambda above
the pillars). Metalens/structured-light tasks follow the paper; `custom` accepts an
arbitrary callable or a .npy phase map, which is what makes arbitrary task input
possible: `design.py --task custom --phase my_phase.npy`.
"""

from __future__ import annotations

import math

import numpy as np
import torch

WL = 1064e-9


def _coords(n: int, dx: float, device="cpu"):
    c = (torch.arange(n, device=device, dtype=torch.float32) - (n - 1) / 2) * dx
    Y, X = torch.meshgrid(c, c, indexing="ij")
    return X, Y


def lens_phase(X, Y, focal, n_bg=1.0, wl=WL):
    """Eq. (7): phi(r) = -k0 n ( sqrt(r^2 + F^2) - F )."""
    k = n_bg * 2 * math.pi / wl
    return -k * (torch.sqrt(X**2 + Y**2 + focal**2) - focal)


def vortex_phase(X, Y, focal, l=6, n_bg=1.0, wl=WL):
    return lens_phase(X, Y, focal, n_bg, wl) + l * torch.atan2(Y, X)


def airy_phase(X, Y, aperture=None, strength=30.0):
    a = aperture if aperture is not None else float(X.abs().max())
    a = max(a, 1e-9)
    return strength * math.pi * (X**3 + Y**3) / a**3


def build_target_phase(kind="lens", n_pix=336, dx=50e-9, **kw):
    """Returns (n_pix, n_pix) target phase in radians."""
    X, Y = _coords(n_pix, dx)
    if kind == "lens":
        ph = lens_phase(X, Y, kw.get("focal", 20e-6), kw.get("n_bg", 1.0), kw.get("wl", WL))
    elif kind == "vortex":
        ph = vortex_phase(
            X, Y, kw.get("focal", 20e-6), kw.get("l", 6), kw.get("n_bg", 1.0), kw.get("wl", WL)
        )
    elif kind == "airy":
        ph = airy_phase(X, Y, kw.get("aperture_m"), kw.get("strength", 30.0))
    elif kind == "custom":
        src = kw["phase"]
        if callable(src):
            ph = src(X, Y)
        else:
            arr = np.load(src) if isinstance(src, str) else np.asarray(src)
            ph = torch.from_numpy(arr.astype(np.float32))
            if ph.shape != (n_pix, n_pix):
                raise ValueError(f"custom phase map must be {(n_pix, n_pix)}, got {tuple(ph.shape)}")
    else:
        raise ValueError(f"unknown task: {kind}")
    return ph


TASKS = ("lens", "vortex", "airy", "custom")
