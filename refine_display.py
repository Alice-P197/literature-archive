# -*- coding: utf-8 -*-
"""把已保存的二值/连续结构在细网格 (50nm) 上重做一次前向传播，只用于出图。

不重跑任何优化：结构 (ri / ri_cont) 直接取自 refined.npz，仅提高场采样分辨率，
使 10µm 显示窗内的艾里旁瓣能分辨为分立圆点（150nm 采样时只有 66px，旁瓣糊成团）。

输出: <dir>/refined_hi.npz  (ri, I_focal_hi, I_ideal_hi, I_focal_cont_hi, dx_hi)
"""
import sys
import numpy as np
import torch

sys.path.insert(0, ".")
from metae.solver import SplitStepBPM, angular_spectrum_propagate

LAM = 1064e-9
PITCH = 200e-9
N_LO, N_HI = 1.0, 3.526
DX_HI = 50e-9
pad = 512


def forward(ri, strength, focal, square_ap=True, log=print):
    """ri: (n,n) float RI 图 -> 焦面强度 (细网格)."""
    n_g = ri.shape[0]
    aperture = n_g * PITCH
    grid = int(round(aperture / DX_HI))
    sol = SplitStepBPM(grid=grid, dx=DX_HI)
    with torch.no_grad():
        E = sol(torch.from_numpy(ri.astype(np.float32))[None],
                return_complex=True)[0]
        E = E * (grid / (E.abs().pow(2).sum().sqrt() + 1e-12))  # L2 定标
    big = torch.zeros(grid + 2 * pad, grid + 2 * pad, dtype=torch.complex64)
    big[pad:pad + grid, pad:pad + grid] = E
    with torch.no_grad():
        Ef = angular_spectrum_propagate(big, focal - LAM, DX_HI)
    return (Ef.abs() ** 2).numpy().astype(np.float64), grid


def ideal(strength, aperture, focal, grid, square_ap=True, kind="airy", l=6):
    """理想相位元件（平面波×目标相位）经同一 AS 传播 -> 焦面强度."""
    c = (np.arange(grid) - grid / 2) * DX_HI
    X, Y = np.meshgrid(c, c, indexing="ij")
    a = aperture / 2
    if kind == "airy":
        phi = strength * np.pi * (X ** 3 + Y ** 3) / a ** 3
    else:  # vortex: 透镜 + 螺旋相位
        k = 2 * np.pi / LAM
        phi = (-k * (np.sqrt(X ** 2 + Y ** 2 + focal ** 2) - focal)
               + l * np.arctan2(Y, X))
    if square_ap:
        ap = (np.abs(X) <= a + 1e-12) & (np.abs(Y) <= a + 1e-12)
    else:
        ap = np.hypot(X, Y) <= a + 1e-12
    elem = (np.exp(1j * phi) * ap).astype(np.complex64)
    big = np.zeros((grid + 2 * pad, grid + 2 * pad), dtype=np.complex64)
    big[pad:pad + grid, pad:pad + grid] = elem
    with torch.no_grad():
        Ef = angular_spectrum_propagate(torch.from_numpy(big), focal - LAM,
                                       DX_HI)
    I = (Ef.abs() ** 2).numpy().astype(np.float64)
    I /= I.sum() / grid ** 2
    return I


def main(d, strength, focal, task="airy"):
    arr = np.load(f"{d}/refined.npz", allow_pickle=True)
    meta = arr["meta"].item()
    ri, ri_cont = arr["ri"], arr["ri_cont"]
    grid = int(round(ri.shape[0] * PITCH / DX_HI))
    print(f"{d}: ri {ri.shape} -> fine grid {grid} @ {DX_HI*1e9:.0f}nm")

    out = {"ri": ri, "ri_cont": ri_cont, "dx_hi": DX_HI, "meta": meta}
    for tag, r in (("", ri), ("_cont", ri_cont)):
        I, _ = forward(r, strength, focal)
        out[f"I_focal{tag}_hi"] = I
        print(f"  forward{tag or '_bin'} done, max {I.max():.3e}")
    if task == "airy":
        out["I_ideal_hi"] = ideal(strength, ri.shape[0] * PITCH, focal, grid)
        print("  ideal template done")
    else:
        out["I_ideal_hi"] = ideal(None, ri.shape[0] * PITCH, focal, grid,
                                  square_ap=False, kind="vortex", l=6)
        print("  ideal vortex template done")
    np.savez_compressed(f"{d}/refined_hi.npz", **out)
    print(f"  saved {d}/refined_hi.npz")


if __name__ == "__main__":
    main("outputs/fig6_airy100", strength=25.0, focal=40e-6, task="airy")
    main("outputs/fig6_vortex", strength=None, focal=15e-6, task="vortex")
