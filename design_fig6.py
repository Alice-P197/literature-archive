# -*- coding: utf-8 -*-
"""Fig.6 复现 —— MetaE-former 正式流程设计脚本。

按论文方法: 微调后的 MetaE-former (NAT 代理) + 逐 5µm meta-atom 相位匹配反演
(design_from_phase), 再拼接成 50µm (OVB) / 100µm (AB) 超表面。

拼接几何: RI 块 25px@200nm=5µm 对应目标块 84px@50nm=4.2µm → 拼装结构物理坐标
= 目标 ext 网格坐标 × 25/21。故对目标相位做 25/21 预畸变, 使拼装后的结构恰好是
理想的透镜+螺旋 (OVB) / 立方相位 (AB), 且口径恰为论文标称尺寸。

输出: outputs/fig6_airy_mt/design.npz, outputs/fig6_vortex_mt/design.npz
  (ri=二值, ri_cont=连续, 各自独立一次设计)
"""
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metae.pipeline import CROP, DX, PITCH_PX, design_from_phase
from metae.tasks import WL
from metae.utils import load_model

STRETCH = 25.0 / 21.0          # 拼装几何拉伸 (RI 5µm 块 <-> 目标 4.2µm 块)
PITCH_RI = 200e-9              # RI 像素间距 (注意: 不是 50nm 场采样!)
CKPT = "checkpoints/finetune2.pt"


def predistorted_phase(kind, n_p, focal=10e-6, l=6, strength=80.0):
    """在 ext 网格上生成预畸变目标相位, 拼装后即得理想相位 (口径=n_p*5µm)。"""
    ext = (n_p - 1) * CROP + PITCH_PX
    u = (np.arange(ext) - ext / 2) * DX
    X = np.repeat(u[:, None], ext, axis=1) * STRETCH      # RI 物理坐标 (m)
    Y = np.repeat(u[None, :], ext, axis=0) * STRETCH
    if kind == "vortex":
        k0 = 2 * math.pi / WL
        r = np.hypot(X, Y)
        ph = -k0 * (np.sqrt(r ** 2 + focal ** 2) - focal) + l * np.arctan2(Y, X)
    elif kind == "airy":
        a = n_p * 25 * PITCH_RI / 2                       # 拼装口径半宽 = 50µm
        ph = strength * math.pi * (X ** 3 + Y ** 3) / a ** 3
    else:
        raise ValueError(kind)
    return ph.astype(np.float32), ext


def run(task, outdir, n_p, iters, threads=4, iband=None, tag="", **kw):
    os.makedirs(outdir, exist_ok=True)
    model, ck = load_model(CKPT)
    log = lambda *a: print(*a, flush=True)
    print(f"[{task}{tag}] model {CKPT} preset={ck.get('preset')} loaded", flush=True)
    torch.set_num_threads(threads)

    phase, ext = predistorted_phase(task, n_p, **kw)
    print(f"[{task}{tag}] ext={ext}  patches={n_p}x{n_p}"
          f"  band={iband}", flush=True)

    t0 = time.time()
    res = design_from_phase(model, phase, patches=n_p, iters=iters, lr=0.1,
                            binarize_from=0.5, threads=threads, log=log,
                            iband=iband, init="phase")
    print(f"[{task}{tag}] phase MAE {res['phase_mae']:.3f} rad  "
          f"loss {res['final_phase_loss']:.4f}  {time.time()-t0:.0f}s", flush=True)
    if iband is None:
        out = dict(ri=res["ri"], ri_cont=res["ri_cont"], field=res["field"],
                   phase_pred=res["phase_pred"],
                   phase_target=res["phase_target"],
                   meta=dict(task=task, n_p=n_p, iters=iters, stretch=STRETCH,
                             kw=kw, ckpt=CKPT, phase_mae=res["phase_mae"]))
        np.savez_compressed(f"{outdir}/design.npz", **out)
        print(f"[{task}] saved {outdir}/design.npz", flush=True)
    else:
        np.savez_compressed(f"{outdir}/band_{iband[0]}.npz",
                            ri=res["ri"], ri_cont=res["ri_cont"],
                            phase_mae=res["phase_mae"], iband=res["iband"])
        print(f"[{task}{tag}] saved band_{iband[0]}.npz", flush=True)


def stitch_airy(outdir="outputs/fig6_airy_mt", n_p=20):
    """合并 4 个行带的 RI -> design.npz"""
    import glob
    files = sorted(glob.glob(f"{outdir}/band_*.npz"),
                   key=lambda p: int(p.split("_")[-1].split(".")[0]))
    ri = np.full((n_p * 25, n_p * 25), 1.0)
    ri_cont = np.full_like(ri, 1.0)
    maes = {}
    for f in files:
        d = np.load(f)
        i0, i1 = d["iband"]
        ri[i0 * 25:i1 * 25] = d["ri"]
        ri_cont[i0 * 25:i1 * 25] = d["ri_cont"]
        maes[f"band{i0}"] = float(d["phase_mae"])
    np.savez_compressed(f"{outdir}/design.npz", ri=ri, ri_cont=ri_cont,
                        meta=dict(task="airy", n_p=n_p, stretch=STRETCH,
                                  phase_mae=maes))
    print("stitched", files, "MAE", maes, flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "vortex"
    if which == "vortex":
        run("vortex", "outputs/fig6_vortex_mt", n_p=10, iters=160,
            focal=10e-6, l=6)
    elif which == "airy":
        run("airy", "outputs/fig6_airy_mt", n_p=20, iters=160, strength=80.0)
    elif which == "airy_band":
        i0, i1 = int(sys.argv[2]), int(sys.argv[3])
        run("airy", "outputs/fig6_airy_mt", n_p=20, iters=160, strength=80.0,
            iband=(i0, i1), tag=f"-band{i0}")
    elif which == "airy_stitch":
        stitch_airy()
