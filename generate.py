# -*- coding: utf-8 -*-
"""一键生成: 任务 -> 相位图 -> 超表面逆向设计 -> 验证结果

用法:
    python generate.py --task vortex --l 3 --focal 15e-6
    python generate.py --task metalens --focal 20e-6
    python generate.py --task airy --strength 10
    python generate.py --task custom --phase my_phase.npy     # 任意相位图文件

产出 (outputs/auto_<task>/):
    phase_target.png / phase_target.csv   目标相位图
    design_result.png                     设计总览 (RI/相位/焦面)
    design.npz, summary.json              数据与指标
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metae.pipeline import design_from_phase, builtin_phase, CROP, PITCH_PX
from metae.utils import load_model, load_or_default
from metae.solver import angular_spectrum_propagate, N_HI
from metae.tasks import WL

DX = 50e-9


def main():
    ap = argparse.ArgumentParser(description="一键任务->相位图->超表面设计")
    ap.add_argument("--task", default="metalens",
                    choices=["metalens", "lens", "vortex", "airy", "custom"])
    ap.add_argument("--focal", type=float, default=20e-6, help="焦距 (m)")
    ap.add_argument("--l", type=int, default=6, help="涡旋拓扑荷")
    ap.add_argument("--strength", type=float, default=10.0, help="airy 立方强度")
    ap.add_argument("--phase", default=None, help="custom: 相位图文件 .npy/.csv")
    ap.add_argument("--patches", type=int, default=4, help="每边 patch 数 (口径=patches*4.2um)")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--ckpt", default=None, help="checkpoint; 缺省自动用 finetune.pt->pretrain.pt")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    task = "lens" if args.task == "metalens" else args.task
    out = args.out or os.path.join("outputs", f"auto_{args.task}"
                                    + (f"_l{args.l}" if task == "vortex" else ""))
    os.makedirs(out, exist_ok=True)

    # 1) 任务 -> 相位图 (秒级, 解析公式或文件)
    ext = (args.patches - 1) * CROP + PITCH_PX
    if task == "custom":
        if not args.phase:
            sys.exit("--task custom 需要 --phase 相位图文件 (.npy/.csv)")
        arr = (np.loadtxt(args.phase, delimiter=",") if args.phase.endswith(".csv")
               else np.load(args.phase)).astype(np.float32)
        if arr.shape != (ext, ext):  # 就近缩放
            xi = np.linspace(0, arr.shape[0] - 1, ext)
            yi = np.linspace(0, arr.shape[1] - 1, ext)
            arr = arr[np.ix_(yi.astype(int), xi.astype(int))]
        tgt_ext = arr
    else:
        kw = dict(focal=args.focal, n_bg=1.0)
        if task == "vortex":
            kw["l"] = args.l
        if task == "airy":
            kw["aperture_m"] = args.patches * 4.2e-6 / 2
            kw["strength"] = args.strength
        tgt_ext = builtin_phase(task, ext, **kw)

    phw = np.angle(np.exp(1j * tgt_ext))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    im0 = ax[0].imshow(tgt_ext, cmap="twilight")
    ax[0].set_title("target phase (unwrapped)")
    plt.colorbar(im0, ax=ax[0], fraction=0.046)
    im1 = ax[1].imshow(phw, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax[1].set_title("target phase (wrapped)")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "phase_target.png"), dpi=130)
    plt.close(fig)
    np.savetxt(os.path.join(out, "phase_target.csv"), tgt_ext, delimiter=",", fmt="%.4f")
    print(f"[1/3] 相位图已生成: {out}/phase_target.png / .csv  ({ext}x{ext}px)")

    # 2) 相位图 -> 超表面 (自动逆向设计, 代理模型加速)
    ck = load_or_default(args.ckpt, os.path.join("checkpoints", "finetune.pt"))
    model, _ = load_model(ck)
    ckname = os.path.basename(ck)
    print(f"[2/3] 逆向设计中 (checkpoint: {ckname}, {args.iters} iters) ...")
    res = design_from_phase(
        model, tgt_ext, patches=args.patches, iters=args.iters,
        focal=(None if task == "custom" else args.focal),
        log=lambda m: print("   ", m), threads=args.threads,
        pad_eval=4 if task == "airy" else 1,
    )
    print(f"      相位 MAE {res['phase_mae']:.3f} rad, 用时 {res['opt_time']:.0f}s "
          f"({res['ms_per_patch_iter']:.0f} ms/patch-iter)")

    # 3) 验证图
    valid = args.patches * CROP
    c = (np.arange(valid) - valid / 2) * DX * 1e6
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    im = ax[0].imshow(res["ri"], cmap="gray", vmin=1.0, vmax=N_HI)
    ax[0].set_title("designed RI (binary)"); plt.colorbar(im, ax=ax[0], fraction=0.046)
    im = ax[1].imshow(res["phase_pred"], cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax[1].set_title(f"realized phase (MAE {res['phase_mae']:.2f} rad)")
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    if res.get("focal_int") is not None:
        fi = res["focal_int"]
        fxv = res["focal_x"]
        im = ax[2].imshow(fi**0.25, cmap="inferno", extent=[fxv[0], fxv[-1], fxv[-1], fxv[0]])
        ax[2].set_title(f"z={args.focal*1e6:.0f}um intensity")
        plt.colorbar(im, ax=ax[2], fraction=0.046)
        ax[3].plot(fxv, res["prof_x"], lw=1)
        ax[3].set_xlabel("x (um)"); ax[3].set_title("profile")
    else:
        ax[2].text(0.5, 0.5, "custom task: no PSF", ha="center")
        E = angular_spectrum_propagate(torch.from_numpy(res["field"].astype(np.complex64)),
                                       args.focal, DX)
        fi = (E.abs()**2).numpy()
        im = ax[2].imshow(fi**0.25, cmap="inferno")
        ax[2].set_title(f"z={args.focal*1e6:.0f}um intensity")
        ax[3].plot(c, fi[fi.sum(axis=1).argmax()] / fi.max(), lw=1)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "design_result.png"), dpi=130)

    meta = dict(task=args.task, focal_m=args.focal, l=args.l, strength=args.strength,
                phase_mae_rad=res["phase_mae"], phase_rmse_rad=res["phase_rmse"],
                patches=args.patches, aperture_m=res["aperture"],
                iters=args.iters, opt_time_s=res["opt_time"],
                ckpt=ckname,
                fwhm_x_um=float(res["fwhm_x"]) * 1e6 if np.isfinite(res.get("fwhm_x", np.nan)) else None,
                efficiency=res.get("efficiency"))
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(meta, f, indent=1)
    np.savez_compressed(os.path.join(out, "design.npz"),
                        ri=res["ri"], field=res["field"], phase_target=tgt_ext,
                        phase_pred=res["phase_pred"])
    print(f"[3/3] 完成 -> {out}/  (phase_target.png/csv, design_result.png, "
          f"design.npz, summary.json)")


if __name__ == "__main__":
    main()
