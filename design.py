"""MetaE-former inverse design (paper: Method - Inverse design, Eq. 4-7, Fig. 5/6).

After training, ANY task can be designed in seconds-to-minutes on CPU:

    python design.py --task metalens --focal 20e-6 --n-bg 1.0  --out outputs/metalens
    python design.py --task vortex  --l 6 --focal 20e-6        --out outputs/vortex6
    python design.py --task airy                                --out outputs/airy
    python design.py --task custom --phase my_phase.npy         --out outputs/custom

Pipeline
--------
1. divide the target aperture into 5 um patches with 84 % cutting ratio
   (paper: Supplementary S14);
2. optimise the RI of every patch with Adam on the differentiable surrogate,
   loss = periodic phase loss + beta * binarisation penalty (Eq. 4-6);
3. splice all patches into the full metasurface, predict the full exiting field
   with the network (fast inference), propagate to the focal plane by angular
   spectrum and report PSF / FWHM / efficiency.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from metae.solver import SplitStepBPM, angular_spectrum_propagate, N_LO, N_HI
from metae.tasks import build_target_phase, WL
from metae.utils import load_model, load_or_default, normalize_input

DX = 50e-9          # field sampling
PITCH_PX = 100      # 5 um patch on the field grid
CROP = 84           # 84 % cutting ratio (paper S14)


def fwhm_1d(profile: np.ndarray, dx: float) -> float:
    p = profile / max(profile.max(), 1e-30)
    if p.max() < 1.5 * p.mean() or not np.isfinite(p).all():  # no discernible peak
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

    xl, xr = cross(i, -1), cross(i, +1)
    return float(abs(xr - xl) * dx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="metalens", choices=["metalens", "vortex", "airy", "custom"])
    ap.add_argument("--focal", type=float, default=20e-6)
    ap.add_argument("--l", type=int, default=6, help="vortex topological charge")
    ap.add_argument("--n-bg", type=float, default=1.0, help="ambient medium index (n1)")
    ap.add_argument("--n-si", type=float, default=N_HI, help="pillar material index (n2)")
    ap.add_argument("--patches", type=int, default=4, help="patches per side (aperture = patches*4.2um)")
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--binarize-from", type=float, default=0.75, help="fraction of iters where penalty starts")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--ckpt", default=os.path.join("checkpoints", "finetune.pt"))
    ap.add_argument("--fallback-ckpt", default=os.path.join("checkpoints", "pretrain.pt"))
    ap.add_argument("--phase", default=None, help="custom .npy phase map (ext-canvas px)")
    ap.add_argument("--strength", type=float, default=10.0,
                    help="airy cubic strength; edge fringe period = aperture/(3*S*pi) "
                         "must stay > 2x RI pixel (168nm) to be realisable")
    ap.add_argument("--out", default="outputs/design")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.task == "metalens":
        args.task = "lens"

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    model, ck = load_model(load_or_default(args.ckpt, args.fallback_ckpt))
    print(f"[ckpt] {args.ckpt} (trained step {ck.get('step','?')})")

    # ------------------------------------------------------------ geometry
    n_p = args.patches
    ext = (n_p - 1) * CROP + PITCH_PX          # extended canvas (patches incl. margins)
    valid = n_p * CROP                          # valid aperture pixels
    aperture = valid * DX
    n_bg = args.n_bg
    focal = args.focal
    na = n_bg * np.sin(np.arctan(aperture / 2 / focal))
    print(f"[geom] aperture {aperture*1e6:.1f} um  ({n_p}x{n_p} patches, 5um pitch, 84% crop)")
    print(f"[geom] focal {focal*1e6:.1f} um in n={n_bg} -> N.A. = {na:.3f}")

    # ------------------------------------------------------- target phase
    kw = dict(focal=focal, n_bg=n_bg)
    if args.task == "vortex":
        kw["l"] = args.l
    if args.task == "airy":
        kw["aperture_m"] = aperture / 2
        kw["strength"] = args.strength
    if args.task == "custom":
        if not args.phase:
            ap.error("--task custom requires --phase my_phase.npy (ext-canvas phase map)")
        kw["phase"] = args.phase
    # global coordinates: canvas centre is the optical axis
    if args.task == "custom" and args.phase:
        ph_ext = np.load(args.phase).astype(np.float32)
    else:
        ph_ext = build_target_phase(args.task, n_pix=ext, **kw).numpy()
    target_t = torch.from_numpy(ph_ext.astype(np.float32))

    # central 84x84 loss window (paper cuts hard at 84 %)
    r0 = (PITCH_PX - CROP) // 2
    win = torch.ones(CROP, CROP)

    # ------------------------------------------------------- optimisation
    n1, n2 = args.n_bg, args.n_si
    n_start = int(args.binarize_from * args.iters)
    solver_gt = SplitStepBPM()  # only used for reference timing
    model.train()

    patches = []
    t0 = time.time()
    patch_iters = 0
    beta_final = 5.0
    loss_hist = []
    # process all patches in a single batch (parallel optimisation, paper Fig. 5)
    v = torch.randn(n_p * n_p, 25, 25) * 0.1
    v.requires_grad_(True)
    opt = torch.optim.Adam([v], lr=args.lr)

    # local target phases, cropped to the 84x84 loss window
    tgt = torch.stack(
        [
            target_t[i * CROP : i * CROP + PITCH_PX, j * CROP : j * CROP + PITCH_PX][
                r0 : r0 + CROP, r0 : r0 + CROP
            ]
            for i in range(n_p)
            for j in range(n_p)
        ]
    )

    for it in range(1, args.iters + 1):
        beta = 0.0
        if it > n_start:
            beta = beta_final * (it - n_start) / max(1, args.iters - n_start)
        n_norm = torch.tanh(v)                       # [-1, 1]
        n_abs = n1 + (n2 - n1) * (n_norm + 1) / 2    # physical index
        pred = model(normalize_input(n_abs))
        pc = torch.complex(pred[:, 0], pred[:, 1])[:, r0 : r0 + CROP, r0 : r0 + CROP]
        dth = torch.angle(pc) - tgt
        l_phase = ((1 - torch.cos(dth)) / 2 * win).mean()
        penalty = (1 - n_norm**2).mean()             # Eq.(4): >=0, 0 at binary ends
        loss = l_phase + beta * penalty
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_hist.append(loss.item())
        patch_iters += n_p * n_p
        if it % 25 == 0 or it == 1:
            print(
                f"  iter {it:4d} | phase {l_phase.item():.4f} | beta {beta:.2f} "
                f"| |{time.time()-t0:.0f}s",
                flush=True,
            )

    t_opt = time.time() - t0
    with torch.no_grad():
        n_abs = n1 + (n2 - n1) * (torch.tanh(v) + 1) / 2
    print(f"[opt] {args.iters} iters x {n_p*n_p} patches in {t_opt:.1f}s "
          f"({t_opt/patch_iters*1e3:.1f} ms per patch-iter)")

    # ------------------------------------------------- assemble + verify
    t1 = time.time()
    with torch.no_grad():
        pred = model(normalize_input(n_abs))
    pc = torch.complex(pred[:, 0], pred[:, 1])
    print(f"[infer] full-field prediction of {n_p*n_p} patches in {time.time()-t1:.2f}s "
          f"({(time.time()-t1)/(n_p*n_p)*1e3:.0f} ms/patch)")

    field = torch.zeros(ext, ext, dtype=torch.complex64)
    ri = torch.full((n_p * 25, n_p * 25), n1)
    for idx, (i, j) in enumerate([(i, j) for i in range(n_p) for j in range(n_p)]):
        field[i * CROP : i * CROP + PITCH_PX, j * CROP : j * CROP + PITCH_PX] = pc[idx]
        # physical metasurface = tiling of the 5 um patches (25x25 pillars each)
        ri[i * 25 : (i + 1) * 25, j * 25 : (j + 1) * 25] = n_abs[idx]
    field_c = field[8 : 8 + valid, 8 : 8 + valid]

    # focal-plane propagation
    # Airy: zero-pad x4 — the one-sided Airy sidelobes lie OUTSIDE the aperture
    # window (peak z*lambda*|u|/(2*pi*s)), so a padded window is required.
    PAD = 4 if args.task == "airy" else 1
    if PAD > 1:
        big = torch.zeros(valid * PAD, valid * PAD, dtype=torch.complex64)
        o = (valid * (PAD - 1)) // 2
        big[o : o + valid, o : o + valid] = field_c
        f_eval = big
    else:
        f_eval = field_c
    v_eval = valid * PAD

    prop = angular_spectrum_propagate(f_eval, focal - WL, DX, n_bg=n_bg)
    inten = prop.abs()**2
    iy, ix = np.unravel_index(inten.numpy().argmax(), inten.shape)
    prof_x = inten[iy, :].numpy()
    prof_y = inten[:, ix].numpy()
    fwhm_x = fwhm_1d(prof_x, DX)
    fwhm_y = fwhm_1d(prof_y, DX)
    # focusing efficiency: power within 3x FWHM radius / total power
    yy, xx = np.meshgrid(np.arange(v_eval), np.arange(v_eval), indexing="ij")
    rr = np.hypot(yy - iy, xx - ix) * DX
    r3 = 3 * max(fwhm_x, fwhm_y)
    eff = float((inten.numpy() * (rr < r3)).sum() / max(inten.numpy().sum(), 1e-12))

    print(f"[psf] FWHM x {fwhm_x*1e6:.2f} um, y {fwhm_y*1e6:.2f} um "
          f"(diffraction limit ~{0.51*WL/n_bg/na*1e6:.2f} um), efficiency(3xFWHM) {eff*100:.0f}%")

    # axial scan for the x-z image
    zs = np.linspace(0.2, 1.8, 40) * focal
    xz = np.zeros((len(zs), v_eval), dtype=np.float32)
    for k, z in enumerate(zs):
        p = angular_spectrum_propagate(f_eval, z - WL, DX, n_bg=n_bg)
        xz[k] = (p.abs()**2).mean(dim=0).numpy()

    # ------------------------------------------------------------- save
    np.savez_compressed(
        os.path.join(args.out, "design.npz"),
        ri=ri.numpy(), phase_target=ph_ext[8 : 8 + valid, 8 : 8 + valid],
        field=field_c.numpy(), focal_int=inten.numpy(), xz=xz,
        meta=dict(task=args.task, focal=focal, n_bg=n_bg, na=float(na),
                  fwhm_x=fwhm_x, fwhm_y=fwhm_y, efficiency=float(eff),
                  aperture=aperture, iters=args.iters, opt_time=t_opt),
    )
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(dict(task=args.task, focal_m=focal, n_bg=n_bg, NA=float(na),
                       aperture_m=aperture, fwhm_x_um=fwhm_x*1e6, fwhm_y_um=fwhm_y*1e6,
                       efficiency=eff, iters=args.iters, opt_time_s=t_opt,
                       patch_iters=patch_iters, ms_per_patch_iter=t_opt/patch_iters*1e3), f, indent=1)

    # ------------------------------------------------------------- plots
    ext_m = valid * DX * 1e6
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    im0 = ax[0, 0].imshow(ri, cmap="gray", vmin=n1, vmax=n2,
                          extent=[-ext_m/2, ext_m/2, ext_m/2, -ext_m/2])
    ax[0, 0].set_title(f"binarised RI (n1={n1}, n2={n2})")
    plt.colorbar(im0, ax=ax[0, 0], fraction=0.046)
    ph = np.angle(field_c.numpy())
    im1 = ax[0, 1].imshow(ph, cmap="twilight", vmin=-np.pi, vmax=np.pi,
                          extent=[-ext_m/2, ext_m/2, ext_m/2, -ext_m/2])
    ax[0, 1].set_title("predicted output phase (z=λ)")
    plt.colorbar(im1, ax=ax[0, 1], fraction=0.046)
    im2 = ax[0, 2].imshow(np.abs(field_c.numpy()), cmap="inferno",
                          extent=[-ext_m/2, ext_m/2, ext_m/2, -ext_m/2])
    ax[0, 2].set_title("predicted |E| (z=λ)")
    plt.colorbar(im2, ax=ax[0, 2], fraction=0.046)

    fx = v_eval * DX * 1e6
    im3 = ax[1, 0].imshow(inten.numpy()**0.5, cmap="inferno",
                          extent=[-fx/2, fx/2, fx/2, -fx/2])
    ax[1, 0].set_title(f"focal plane z={focal*1e6:.0f}um (sqrt I)")
    plt.colorbar(im3, ax=ax[1, 0], fraction=0.046)
    ax[1, 1].plot(np.arange(v_eval) * DX * 1e6 - fx / 2, prof_x / prof_x.max())
    ax[1, 1].axhline(0.5, ls="--", c="gray")
    ax[1, 1].set_xlabel("x (um)"); ax[1, 1].set_ylabel("I/I0")
    ax[1, 1].set_title(f"FWHM x = {fwhm_x*1e6:.2f} um")
    im5 = ax[1, 2].imshow(xz**0.5, aspect="auto", cmap="inferno",
                          extent=[-fx/2, fx/2, zs[-1]*1e6, zs[0]*1e6])
    ax[1, 2].set_title("x-z propagation")
    plt.colorbar(im5, ax=ax[1, 2], fraction=0.046)
    for a in ax.flat:
        a.set_xticks([]) if a in (ax[0, 0], ax[0, 1], ax[0, 2]) else None
    fig.suptitle(
        f"{args.task} | aperture {aperture*1e6:.1f}um | N.A.={na:.2f} | "
        f"{args.iters} iters in {t_opt:.0f}s", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out, "design_result.png"), dpi=130)
    print(f"[saved] {args.out}/design_result.png, design.npz, summary.json")


if __name__ == "__main__":
    main()
