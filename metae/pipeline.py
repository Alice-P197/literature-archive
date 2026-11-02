"""Reusable pipeline functions shared by the CLI scripts and the GUI app.

train_model        : pre-training loop (same recipe as pretrain.py)
design_from_phase  : inverse design for an arbitrary target phase map
evaluate_model     : Table-1 style evaluation on the three pattern families
load_phase_map     : load a phase map from .npy/.csv/.png/.jpg and resize
CSV exporters      : results -> csv files
"""

from __future__ import annotations

import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import FreeformAtomSampler, N_HI, N_LO
from .losses import metaformer_loss
from .metrics import amplitude_angle_mae, normalized_mae
from .model import build_model
from .solver import SplitStepBPM, angular_spectrum_propagate
from .tasks import WL
from .utils import normalize_input

PITCH_PX, CROP = 100, 84
DX = 50e-9


def fwhm_1d(profile: np.ndarray, dx: float) -> float:
    p = profile / max(profile.max(), 1e-30)
    if p.max() < 1.5 * p.mean() or not np.isfinite(p).all():
        return float("nan")
    i = int(np.argmax(p))
    half = 0.5

    def cross(j0, step):
        j = j0
        while 0 < j < len(p) - 1 and p[j] > half:
            j += step
        slope = p[j + step] - p[j] if 0 <= j + step < len(p) else 0.0
        frac = (p[j] - half) / slope if abs(slope) > 0.02 else 0.5
        return j + frac

    return float(abs(cross(i, +1) - cross(i, -1)) * dx)


# --------------------------------------------------------------------- train
def train_model(
    preset="fast",
    steps=1600,
    batch=24,
    lr=3e-4,
    alpha=1.0,
    warmup=100,
    ckpt_out="checkpoints/gui_trained.pt",
    log=print,
    should_stop=None,
    progress=None,
    threads=8,
    seed=0,
):
    """Pre-train MetaE-former. progress(step, loss, amp, ang, val_amp, val_ang)."""
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    os.makedirs(os.path.dirname(os.path.abspath(ckpt_out)), exist_ok=True)

    model = build_model(preset)
    log(f"模型 {preset}: {sum(p.numel() for p in model.parameters())/1e6:.2f}M 参数")
    solver = SplitStepBPM()
    sampler = FreeformAtomSampler(generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # small fixed val set for the curves
    from .data import N_LO as _lo, N_HI as _hi  # noqa

    g = torch.Generator().manual_seed(1234)
    vs = FreeformAtomSampler(generator=g)
    with torch.no_grad():
        xv = torch.cat([vs.random_pattern(48), vs.island_pattern(48), vs.stripe_pattern(48)])
        yv = solver(xv)
    xv_in = normalize_input(xv)

    hist = {"step": [], "loss": [], "val_amp": [], "val_ang": []}
    best = float("inf")
    t0 = time.time()
    val_every = max(50, steps // 20)
    for step in range(1, steps + 1):
        if should_stop is not None and (should_stop.is_set() if hasattr(should_stop, "is_set") else should_stop()):
            log("训练被用户停止")
            break
        lr_s = lr * step / warmup if step <= warmup else (
            1e-6 + (lr - 1e-6) * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
        )
        for g_ in opt.param_groups:
            g_["lr"] = lr_s
        model.train()
        nm = sampler.sample(batch)
        with torch.no_grad():
            fld = solver(nm)
        pred = model(normalize_input(nm))
        loss, la, lt = metaformer_loss(pred, fld, alpha=alpha, return_parts=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        va = vt = float("nan")
        if step % val_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                pv = model(xv_in)
                _, va, vt = metaformer_loss(pv, yv, alpha=alpha, return_parts=True)
            score = va.item() + alpha * vt.item()
            if score < best:
                best = score
                torch.save({"model": model.state_dict(), "preset": preset, "step": step,
                            "alpha": alpha}, ckpt_out)
            hist["step"].append(step)
            hist["loss"].append(loss.item())
            hist["val_amp"].append(va.item())
            hist["val_ang"].append(vt.item())
            log(f"step {step}/{steps} | loss {loss.item():.4f} | val amp {va.item():.4f} "
                f"| val ang {vt.item():.4f} | {time.time()-t0:.0f}s")
        if progress is not None and (step % max(1, steps // 100) == 0 or step == 1):
            progress(step, steps, loss.item(), la.item(), lt.item(), va, vt, hist)
    log(f"训练完成 -> {ckpt_out}")
    return hist, ckpt_out


# -------------------------------------------------------------------- design
def design_from_phase(
    model,
    target_ext: np.ndarray,
    patches=4,
    iters=220,
    lr=0.1,
    n1=1.0,
    n2=N_HI,
    binarize_from=0.75,
    focal=None,
    log=print,
    should_stop=None,
    threads=8,
    pad_eval=1,
    iband=None,
    init="random",
):
    """Inverse design for an arbitrary target phase map.

    target_ext: (ext, ext) radians, ext = (patches-1)*CROP + PITCH_PX.
    focal: optional focal length (m); if given, focal-plane PSF metrics are computed.
    """
    torch.set_num_threads(threads)
    model.train()
    n_p = patches
    ext = (n_p - 1) * CROP + PITCH_PX
    valid = n_p * CROP
    r0 = (PITCH_PX - CROP) // 2
    if target_ext.shape != (ext, ext):
        raise ValueError(f"目标相位图需为 {(ext, ext)} 像素 (patches={patches})，收到 {target_ext.shape}")
    blocks = np.stack(
        [
            np.stack(
                [
                    target_ext[i * CROP + r0 : i * CROP + r0 + CROP,
                               j * CROP + r0 : j * CROP + r0 + CROP]
                    for j in range(n_p)
                ]
            )
            for i in range(n_p)
        ]
    )  # (n_p, n_p, CROP, CROP)
    i0d, i1d = (0, n_p) if iband is None else iband
    sel = slice(i0d * n_p, i1d * n_p)
    tgt = torch.from_numpy(
        blocks.reshape(n_p * n_p, CROP, CROP)[sel].astype(np.float32))
    n_rows = i1d - i0d

    aperture = valid * DX
    na = float("nan")
    if focal:
        na = n1 * math.sin(math.atan(aperture / 2 / focal))

    t0 = time.time()
    v = torch.randn(n_rows * n_p, 25, 25) * 0.1
    if init == "phase":
        # 全息图初始化: n = 1 + wrap(phi_target)/(k0*h), 给优化器正确起点
        k0 = 2 * math.pi / WL
        idx = ((torch.arange(25, dtype=torch.float32) + 0.5) * (CROP / 25.0)).long()
        idx = idx.clamp(0, CROP - 1)
        t25 = tgt[:, idx][:, :, idx]      # 目标块 84px -> RI 像素 25px
        n0 = 1.0 + torch.remainder(t25, 2 * math.pi) / (k0 * 800e-9)
        n0 = n0.clamp(float(n1), float(n2))
        tv = ((n0 - n1) / (n2 - n1) * 2 - 1).clamp(-0.98, 0.98)
        v = torch.atanh(tv)
        log("  init: target-phase hologram init")
    v.requires_grad_(True)
    opt = torch.optim.Adam([v], lr=lr)
    beta_final, loss_hist = 5.0, []
    n_start = int(binarize_from * iters)
    cont_snap = None
    for it in range(1, iters + 1):
        if should_stop is not None and (should_stop.is_set() if hasattr(should_stop, "is_set") else should_stop()):
            log("设计被用户停止")
            break
        beta = beta_final * (it - n_start) / max(1, iters - n_start) if it > n_start else 0.0
        n_norm = torch.tanh(v)
        n_abs = n1 + (n2 - n1) * (n_norm + 1) / 2
        if it == n_start + 1 and cont_snap is None:
            cont_snap = n_abs.detach().clone()  # 二值化起点 = 连续 RI 快照
        pred = model(normalize_input(n_abs))
        pc = torch.complex(pred[:, 0], pred[:, 1])[:, r0 : r0 + CROP, r0 : r0 + CROP]
        l_phase = ((1 - torch.cos(torch.angle(pc) - tgt)) / 2).mean()
        penalty = (1 - n_norm**2).mean()  # Eq.(4): 0 at binary ends
        loss = l_phase + beta * penalty
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_hist.append(loss.item())
        if it % max(1, iters // 10) == 0 or it == 1:
            log(f"  iter {it}/{iters} | 相位损失 {l_phase.item():.4f} | beta {beta:.2f}")

    t_opt = time.time() - t0
    n_patch_iters = iters * n_rows * n_p
    with torch.no_grad():
        n_abs = n1 + (n2 - n1) * (torch.tanh(v) + 1) / 2
        pred = model(normalize_input(n_abs))
    pc = torch.complex(pred[:, 0], pred[:, 1])
    t_infer = time.time() - t0 - t_opt

    # assemble field + RI (仅行带 i0..i1, 其余保持 n1)
    field = torch.zeros(ext, ext, dtype=torch.complex64)
    ri = torch.full((n_p * 25, n_p * 25), n1)
    k = 0
    for i in range(i0d, i1d):
        for j in range(n_p):
            field[i * CROP : i * CROP + PITCH_PX, j * CROP : j * CROP + PITCH_PX] = pc[k]
            ri[i * 25 : (i + 1) * 25, j * 25 : (j + 1) * 25] = n_abs[k]
            k += 1
    field_c = field[r0 : r0 + valid, r0 : r0 + valid].numpy()
    phase_pred = np.angle(field_c)
    phase_target = target_ext[r0 : r0 + valid, r0 : r0 + valid]

    ri_cont = None
    if cont_snap is not None:
        ri_cont = torch.full((n_p * 25, n_p * 25), float(n1))
        k = 0
        for i in range(i0d, i1d):
            for j in range(n_p):
                ri_cont[i * 25 : (i + 1) * 25, j * 25 : (j + 1) * 25] = cont_snap[k]
                k += 1
        ri_cont = ri_cont.numpy()

    dth = np.angle(np.exp(1j * (phase_pred - phase_target)))
    rows = slice(i0d * 25, i1d * 25)
    phase_mae = float(np.abs(dth[rows]).mean())
    phase_rmse = float(np.sqrt((dth[rows] ** 2).mean()))

    res = dict(
        ri=ri.numpy()[rows], field=field_c[rows], amp=np.abs(field_c)[rows],
        phase_pred=phase_pred[rows],
        phase_target=phase_target[rows], phase_mae=phase_mae, phase_rmse=phase_rmse,
        aperture=aperture, na=na, patches=n_p, iters=iters,
        opt_time=t_opt, infer_time=t_infer,
        ms_per_patch_iter=t_opt / n_patch_iters * 1e3,
        final_phase_loss=float(loss_hist[-1]) if loss_hist else float("nan"),
        loss_hist=loss_hist, n1=n1, n2=n2,
        ri_cont=None if ri_cont is None else ri_cont[rows],
        iband=(i0d, i1d),
    )

    if focal:
        # pad_eval>1 (e.g. Airy): zero-pad so one-sided sidelobes outside the
        # aperture window remain visible in the evaluation.
        n0 = field_c.shape[0]
        if pad_eval > 1:
            big = np.zeros((n0 * pad_eval, n0 * pad_eval), np.complex64)
            o = (n0 * (pad_eval - 1)) // 2
            big[o : o + n0, o : o + n0] = field_c
            f_in = torch.from_numpy(big)
        else:
            f_in = torch.from_numpy(field_c)
        v_eval = valid * pad_eval
        prop = angular_spectrum_propagate(f_in, focal - WL, DX, n_bg=n1)
        inten = (prop.abs() ** 2).numpy()
        iy, ix = np.unravel_index(inten.argmax(), inten.shape)
        prof_x, prof_y = inten[iy, :], inten[:, ix]
        fx = fy = float("nan")
        if np.isfinite(inten).all():
            fx = fwhm_1d(prof_x, DX)
            fy = fwhm_1d(prof_y, DX)
        yy, xx = np.meshgrid(np.arange(v_eval), np.arange(v_eval), indexing="ij")
        rr = np.hypot(yy - iy, xx - ix) * DX
        r3 = 3 * max(fx, fy)
        eff = float((inten * (rr < r3)).sum() / max(inten.sum(), 1e-12)) if np.isfinite(r3) else float("nan")
        res.update(
            focal_int=inten, focal_x=np.arange(v_eval) * DX * 1e6 - v_eval * DX * 1e6 / 2,
            prof_x=prof_x / max(prof_x.max(), 1e-30), fwhm_x=fx, fwhm_y=fy,
            efficiency=eff, focal=focal,
            diff_limit=0.51 * WL / n1 / na if na and np.isfinite(na) else float("nan"),
        )
    return res


# ------------------------------------------------------------------ evaluate
def evaluate_model(model, n_per=48, log=print, should_stop=None, threads=8, seed=2026):
    torch.set_num_threads(threads)
    solver = SplitStepBPM()
    sampler = FreeformAtomSampler(generator=torch.Generator().manual_seed(seed))
    fams = ["random", "island", "stripe"]
    fns = [sampler.random_pattern, sampler.island_pattern, sampler.stripe_pattern]
    rows = []
    model.eval()
    with torch.no_grad():
        for name, fn in zip(fams, fns):
            if should_stop is not None and (should_stop.is_set() if hasattr(should_stop, "is_set") else should_stop()):
                break
            nm = fn(n_per)
            gt = solver(nm)
            pred = model(normalize_input(nm))
            amp, ang = amplitude_angle_mae(pred, gt)
            nma = normalized_mae(pred, gt)
            rows.append(dict(family=name, n=n_per, amp_mae=amp.mean().item(),
                             angle_mae=ang.mean().item(), norm_mae=nma.mean().item()))
            log(f"  {name}: amp {rows[-1]['amp_mae']:.3f} | angle {rows[-1]['angle_mae']:.3f} "
                f"| normMAE {rows[-1]['norm_mae']:.3f}")
    if rows:
        n_all = sum(r["n"] for r in rows)
        overall = dict(
            family="overall", n=n_all,
            amp_mae=float(np.average([r["amp_mae"] for r in rows], weights=[r["n"] for r in rows])),
            angle_mae=float(np.average([r["angle_mae"] for r in rows], weights=[r["n"] for r in rows])),
            norm_mae=float(np.average([r["norm_mae"] for r in rows], weights=[r["n"] for r in rows])),
        )
        rows.append(overall)
        log(f"  总体: amp {overall['amp_mae']:.3f} | angle {overall['angle_mae']:.3f} "
            f"| normMAE {overall['norm_mae']:.3f}")
    return rows


# -------------------------------------------------------------- phase loading
def load_phase_map(path: str, size: int) -> np.ndarray:
    """Load a target phase map and resize to (size, size) radians."""
    ext_l = path.lower()
    if ext_l.endswith((".png", ".jpg", ".jpeg", ".bmp")):
        import matplotlib.image as mpimg

        img = mpimg.imread(path)
        img = img[..., :3].mean(axis=2) if img.ndim == 3 else img
        arr = img.astype(np.float64)
        if arr.max() > 1.5:  # 8-bit style
            arr = arr / 255.0
        phase = arr * 2 * math.pi
    else:  # .npy / .csv / .txt : values in radians
        if ext_l.endswith(".npy"):
            phase = np.load(path).astype(np.float64)
        else:
            phase = np.loadtxt(path, delimiter="," if ext_l.endswith(".csv") else None)
    im = torch.from_numpy(phase.astype(np.float32))[None, None]
    im = F.interpolate(im, size=(size, size), mode="bilinear", align_corners=False)
    return im[0, 0].numpy()


def builtin_phase(task: str, ext: int, focal=20e-6, l=6, strength=10.0, n_bg=1.0) -> np.ndarray:
    from .tasks import build_target_phase

    kw = dict(focal=focal, n_bg=n_bg)
    if task == "vortex":
        kw["l"] = l
    if task == "airy":
        kw["aperture_m"] = (ext * DX) / 2
        kw["strength"] = strength
    return build_target_phase(task, n_pix=ext, **kw).numpy()


# ----------------------------------------------------------------------- CSV
def save_matrix_csv(arr: np.ndarray, path: str, pixel_note=""):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if pixel_note:
            w.writerow([pixel_note])
        for row in arr:
            w.writerow([f"{v:.6g}" for v in row])


def save_rows_csv(rows: list[dict], path: str):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def save_design_csvs(res: dict, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    save_matrix_csv(res["ri"], os.path.join(outdir, "ri_map.csv"),
                    f"# RI matrix {res['ri'].shape[0]}x{res['ri'].shape[1]} (200nm/pixel, n1={res['n1']}, n2={res['n2']})")
    save_matrix_csv(res["phase_target"], os.path.join(outdir, "phase_target.csv"),
                    f"# target phase (rad), {res['phase_target'].shape[0]}x{res['phase_target'].shape[1]} (50nm/pixel)")
    save_matrix_csv(res["phase_pred"], os.path.join(outdir, "phase_pred.csv"),
                    "# predicted output phase (rad)")
    metrics = [
        {"metric": "phase_MAE_rad", "value": res["phase_mae"]},
        {"metric": "phase_RMSE_rad", "value": res["phase_rmse"]},
        {"metric": "final_phase_loss", "value": res["final_phase_loss"]},
        {"metric": "aperture_um", "value": res["aperture"] * 1e6},
        {"metric": "NA", "value": res["na"]},
        {"metric": "FWHM_x_um", "value": res.get("fwhm_x", float("nan"))},
        {"metric": "FWHM_y_um", "value": res.get("fwhm_y", float("nan"))},
        {"metric": "diffraction_limit_um", "value": res.get("diff_limit", float("nan"))},
        {"metric": "efficiency_3xFWHM", "value": res.get("efficiency", float("nan"))},
        {"metric": "optimize_time_s", "value": res["opt_time"]},
        {"metric": "infer_time_s", "value": res["infer_time"]},
        {"metric": "ms_per_patch_iter", "value": res["ms_per_patch_iter"]},
    ]
    save_rows_csv(metrics, os.path.join(outdir, "design_metrics.csv"))
    if "focal_int" in res:
        save_matrix_csv(res["focal_int"], os.path.join(outdir, "focal_intensity.csv"),
                        "# focal plane intensity (W/a.u., 50nm/pixel)")
    save_rows_csv(
        [{"iter": i + 1, "loss": v} for i, v in enumerate(res["loss_hist"])],
        os.path.join(outdir, "design_loss_curve.csv"),
    )
