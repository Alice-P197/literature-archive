# -*- coding: utf-8 -*-
"""论文 Fig.6 复现渲染（MetaE-former 设计版）。

Design 行 = design_fig6.py 的 MetaE-former 反演结果 (finetune2.pt 代理逐块设计)。
Target 行 = 解析式: 有限能量艾里 Ai² / LG_0^6 环 (半径取设计实测值)。
Real 行   = 已保存结构在 50nm 细网格的一次性前向传播 (BPM 验证)。

窗口构图按论文原生图标定: 面板 ±7.1µm, 艾里主瓣在面板中心 (0.48,0.51),
涡旋环居中; 比例尺 Design 行 5µm(Airy)/10µm(Vortex), 光场行 3µm。
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import airy

sys.path.insert(0, ".")
from metae.solver import SplitStepBPM, angular_spectrum_propagate

LAM, PITCH = 1064e-9, 200e-9
DX, PAD, SIG = 50e-9, 384, 1.5
Z_AIRY, Z_VORTEX = 25e-6, 10e-6       # 目标面 (与 design_fig6.py 一致)
HALF = 7.1                            # 光场显示窗半径 (µm), 论文面板=14.2µm
CACHE = "outputs/fig6_fields_mt.npz"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

FIRE = LinearSegmentedColormap.from_list(
    "fire", [(0.00, (0, 0, 0)), (0.25, (0.35, 0, 0)), (0.50, (0.90, 0.10, 0)),
             (0.75, (1.0, 0.50, 0)), (0.92, (1.0, 0.85, 0.30)), (1.00, (1, 1, 0.85))])


# ---------------------------------------------------------------- physics
def hi_forward(ri, z):
    n = ri.shape[0]
    g = int(round(n * PITCH / DX))
    sol = SplitStepBPM(grid=g, dx=DX, edge_sigma_px=SIG)
    with torch.no_grad():
        E = sol(torch.from_numpy(ri.astype(np.float32))[None], return_complex=True)[0]
        E = E * (g / (E.abs().pow(2).sum().sqrt() + 1e-12))
    big = torch.zeros(g + 2 * PAD, g + 2 * PAD, dtype=torch.complex64)
    big[PAD:PAD + g, PAD:PAD + g] = E
    with torch.no_grad():
        Ef = angular_spectrum_propagate(big, z - LAM, DX)
    return (Ef.abs() ** 2).numpy().astype(np.float32)


def get_fields():
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return {k: d[k] for k in d.files}
    out = {}
    for name, path, z in (("airy", "outputs/fig6_airy_mt", Z_AIRY),
                          ("vortex", "outputs/fig6_vortex_mt", Z_VORTEX)):
        a = np.load(path + "/design.npz", allow_pickle=True)
        out[name + "_bin"] = hi_forward(a["ri"], z)
        out[name + "_cont"] = hi_forward(a["ri_cont"], z)
        out[name + "_ri"] = a["ri"]
        out[name + "_ri_cont"] = a["ri_cont"]
        print(name, "forward done", flush=True)
    np.savez_compressed(CACHE, **out)
    return out


# ---------------------------------------------------------------- targets
def airy_target(u, v, x0):
    """解析艾里: I=Ai²(-t_u)Ai²(-t_v), t=u/x0+1.0188 (主瓣在 u=0)。"""
    tu, tv = u / x0 + 1.0188, v / x0 + 1.0188
    I = airy(-tu)[0] ** 2 * airy(-tv)[0] ** 2
    return I / max(I.max(), 1e-30)


def vortex_target(r, r_peak, lp=6):
    w = r_peak / np.sqrt(lp / 2.0)
    s = r ** 2 / w ** 2
    I = s ** lp * np.exp(-2 * s)
    return I / max(I.max(), 1e-30)


# ---------------------------------------------------------------- helpers
def main_lobe(I, dx):
    n = I.shape[0]
    c = n // 2
    r = int(30e-6 / dx)
    sub = I[c - r:c + r, c - r:c + r]
    ix, iy = np.unravel_index(sub.argmax(), sub.shape)
    return (ix - r) * dx * 1e6, (iy - r) * dx * 1e6


def ring_center(I, dx):
    n = I.shape[0]
    c = n // 2
    r = int(5e-6 / dx)
    sub = I[c - r:c + r, c - r:c + r]
    m = sub > 0.35 * sub.max()
    ys, xs = np.nonzero(m)
    return (xs.mean() - r) * dx * 1e6, (ys.mean() - r) * dx * 1e6


def ring_radius(I, dx, cx, cy):
    n = I.shape[0]
    c = n // 2
    ix, iy = c + int(round(cx * 1e-6 / dx)), c + int(round(cy * 1e-6 / dx))
    w = int(4e-6 / dx)
    y, x = np.ogrid[-w:w + 1, -w:w + 1]
    rr = np.hypot(x, y)
    m = I[iy - w:iy + w + 1, ix - w:ix + w + 1]
    prof = np.bincount(np.clip(rr.astype(int), 0, 79).ravel(),
                       weights=m.ravel(), minlength=80)
    cnt = np.bincount(np.clip(rr.astype(int), 0, 79).ravel(), minlength=80)
    prof = prof / np.maximum(cnt, 1)
    return 0.05 * (prof.argmax() + 0.5)


def win(I, dx, cx, cy, half=HALF):
    n = I.shape[0]
    c = n // 2
    sx = int(round(cx * 1e-6 / dx))
    sy = int(round(cy * 1e-6 / dx))
    h = int(round(half * 1e-6 / dx))
    return I[c + sx - h:c + sx + h, c + sy - h:c + sy + h]


def lobe_spacing(I, dx, mlx, mly):
    from scipy.signal import find_peaks
    n = I.shape[0]
    c = n // 2
    row = I[c + int(round(mly * 1e-6 / dx)), :]
    cc = c + int(round(mlx * 1e-6 / dx))
    seg = row[cc:cc + int(14e-6 / dx)]
    pk, _ = find_peaks(seg, height=seg.max() * 0.04, distance=int(0.4e-6 / dx))
    d = np.diff(pk) * dx * 1e6
    return float(np.median(d[1:])) if len(d) > 1 else float(np.median(d))


# ---------------------------------------------------------------- panels
def panel_design(ax, ri, label, bar_um):
    n = ri.shape[0]
    ap = n * PITCH * 1e6
    ax.imshow(ri.T, cmap="gray", vmin=1.0, vmax=3.526, origin="lower",
              extent=[-ap / 2, ap / 2, -ap / 2, ap / 2], interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.4")
        s.set_linewidth(0.8)
    ax.text(0.035, 0.045, label, transform=ax.transAxes, color="white",
            fontsize=13, va="bottom")
    frac = bar_um / ap
    x1, y0 = 0.965, 0.055
    ax.add_patch(Rectangle((x1 - frac, y0), frac, 0.02, transform=ax.transAxes,
                           facecolor="white", edgecolor="black", lw=0.5))
    ax.text(x1 - frac - 0.02, y0 + 0.01, f"{bar_um}µm", transform=ax.transAxes,
            color="white", fontsize=11, ha="right", va="center")
    # 局部放大 inset (白框)
    zw = int(n * 0.12)
    off = int(n * 0.18)
    c = n // 2
    patch = ri[c + off:c + off + zw, c - off - zw:c - off]
    axins = ax.inset_axes([0.64, 0.64, 0.33, 0.33])
    axins.imshow(patch.T, cmap="gray", vmin=1.0, vmax=3.526, origin="lower",
                 interpolation="nearest")
    axins.set_xticks([])
    axins.set_yticks([])
    for s in axins.spines.values():
        s.set_color("white")
        s.set_linewidth(1.6)


def panel_field(ax, W, arrows=True):
    h = HALF
    W = W / max(W.max(), 1e-30)
    ax.imshow(W.T, cmap=FIRE, vmin=0, vmax=1.0, origin="lower",
              extent=[-h, h, -h, h], interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    x1 = h * 0.92
    ax.plot([x1 - 3, x1], [-h * 0.86] * 2, color="w", lw=3,
            solid_capstyle="butt")
    ax.text(x1 - 3.35, -h * 0.86, "3µm", color="w", fontsize=11,
            ha="right", va="center")
    if arrows:
        x0, y0 = -h * 0.62, -h * 0.62
        ax.annotate("", xy=(x0 + h * 0.34, y0), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="w", lw=1.8))
        ax.annotate("", xy=(x0, y0 + h * 0.34), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="w", lw=1.8))
        ax.text(x0 + h * 0.40, y0, "X", color="w", fontsize=11,
                ha="left", va="center")
        ax.text(x0, y0 + h * 0.40, "Y", color="w", fontsize=11,
                ha="center", va="bottom")


def colorbar_row(fig, y):
    """底部: Refractive index (air->a-Si 灰条) + Intensity (0->1 火焰条)."""
    fig.text(0.065, y + 0.012, "Refractive index:", fontsize=13, ha="left",
             va="center")
    ax1 = fig.add_axes([0.235, y, 0.17, 0.022])
    grad = np.linspace(0, 1, 256)[None, :]
    ax1.imshow(grad, cmap="gray", aspect="auto")
    ax1.set_xticks([])
    ax1.set_yticks([])
    for s in ax1.spines.values():
        s.set_color("black")
        s.set_linewidth(0.8)
    fig.text(0.235, y + 0.030, "air", fontsize=12, ha="left", va="bottom")
    fig.text(0.405, y + 0.030, "a-Si", fontsize=12, ha="right", va="bottom")

    fig.text(0.50, y + 0.012, "Intensity:", fontsize=13, ha="left", va="center")
    ax2 = fig.add_axes([0.60, y, 0.22, 0.022])
    ax2.imshow(grad, cmap=FIRE, aspect="auto")
    ax2.set_xticks([])
    ax2.set_yticks([])
    for s in ax2.spines.values():
        s.set_color("black")
        s.set_linewidth(0.8)
    fig.text(0.60, y + 0.030, "0", fontsize=12, ha="left", va="bottom")
    fig.text(0.82, y + 0.030, "1", fontsize=12, ha="right", va="bottom")


# ---------------------------------------------------------------- main
def main():
    F = get_fields()
    DXu = DX * 1e6
    data = {}
    # --- Airy
    mlb = main_lobe(F["airy_bin"], DX)
    mlc = main_lobe(F["airy_cont"], DX)
    sp = lobe_spacing(F["airy_bin"], DX, *mlb)
    x0 = sp / 1.35
    print(f"Airy ML bin {mlb} cont {mlc}  spacing {sp:.2f}um -> x0={x0:.2f}um")
    for tag in ("bin", "cont"):
        ml = mlb if tag == "bin" else mlc
        cx, cy = ml[0] + 0.3, ml[1] + 0.15     # 论文: 主瓣在面板中心 (0.48,0.51)
        Wr = win(F["airy_" + tag], DX, cx, cy)
        xs = cx + (np.arange(Wr.shape[0]) - Wr.shape[0] / 2) * DXu
        ys = cy + (np.arange(Wr.shape[1]) - Wr.shape[1] / 2) * DXu
        Wt = airy_target(xs[:, None] - ml[0], ys[None, :] - ml[1], x0)
        data[("airy", tag)] = (Wt, Wr)
    # --- Vortex
    cb = ring_center(F["vortex_bin"], DX)
    cc_ = ring_center(F["vortex_cont"], DX)
    rb = ring_radius(F["vortex_bin"], DX, *cb)
    print(f"Vortex ring center bin {cb} cont {cc_}  r_peak={rb:.2f}um")
    for tag in ("bin", "cont"):
        cc2 = cb if tag == "bin" else cc_
        Wr = win(F["vortex_" + tag], DX, cc2[0], cc2[1])
        xs = cc2[0] + (np.arange(Wr.shape[0]) - Wr.shape[0] / 2) * DXu
        ys = cc2[1] + (np.arange(Wr.shape[1]) - Wr.shape[1] / 2) * DXu
        R = np.hypot(xs[:, None] - cc2[0], ys[None, :] - cc2[1])
        Wt = vortex_target(R, r_peak=rb)
        data[("vortex", tag)] = (Wt, Wr)

    # ---------------------------------------------------------------- layout
    fig = plt.figure(figsize=(11.6, 10.2), dpi=300)
    gs = GridSpec(3, 4, figure=fig, left=0.085, right=0.995, top=0.895,
                  bottom=0.115, wspace=0.035, hspace=0.035)
    axes = gs.subplots()

    # 论文列序: AB-cont | OVB-cont | AB-bin | OVB-bin
    for j, (task, tag, label, bar) in enumerate(
            [("airy", "cont", "Airy", 5), ("vortex", "cont", "Vortex", 10),
             ("airy", "bin", "Airy", 5), ("vortex", "bin", "Vortex", 10)]):
        ri = F[task + "_ri_cont"] if tag == "cont" else F[task + "_ri"]
        panel_design(axes[0, j], ri, label, bar)
        panel_field(axes[1, j], data[(task, tag)][0])
        panel_field(axes[2, j], data[(task, tag)][1])

    fig.canvas.draw()
    for text, j0, j1 in [("Continuous refractive index", 0, 1),
                         ("Binarized refractive index", 2, 3)]:
        p0, p1 = axes[0, j0].get_position(), axes[0, j1].get_position()
        x0b, x1b = p0.x0 - 0.005, p1.x1 + 0.005
        yb, hh = p0.y1 + 0.008, 0.055
        fig.add_artist(Rectangle((x0b, yb), x1b - x0b, hh, facecolor="#f2f2f2",
                                 edgecolor="black", lw=1.2,
                                 transform=fig.transFigure))
        fig.text((x0b + x1b) / 2, yb + hh / 2, text, ha="center", va="center",
                 fontsize=16, fontweight="bold")
    pa, pb = axes[0, 0].get_position(), axes[0, 3].get_position()
    fig.add_artist(plt.Line2D([pa.x0 - 0.005, pb.x1 + 0.005],
                              [pa.y1 + 0.008 + 0.055 + 0.006] * 2,
                              color="black", lw=1.6, transform=fig.transFigure))
    for i, text in enumerate(["Design", "Target light field",
                              "Real light field"]):
        p = axes[i, 0].get_position()
        x0b, yb = p.x0 - 0.058, p.y0 - 0.004
        w, h = 0.045, p.height + 0.008
        fig.add_artist(Rectangle((x0b, yb), w, h, facecolor="#e9e9e9",
                                 edgecolor="none", transform=fig.transFigure))
        fig.text(x0b + w / 2, yb + h / 2, text, rotation=90, ha="center",
                 va="center", fontsize=13)
    colorbar_row(fig, 0.055)

    fig.savefig("outputs/fig6_replication.png", bbox_inches="tight",
                facecolor="white")
    print("saved outputs/fig6_replication.png")


if __name__ == "__main__":
    main()
