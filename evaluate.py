"""Evaluate the pre-trained MetaE-former (paper Fig. 3b/3c/3d, Table 1).

    python evaluate.py --ckpt checkpoints/pretrain.pt --n 128
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from metae.data import FreeformAtomSampler
from metae.losses import metaformer_loss
from metae.metrics import amplitude_angle_mae, normalized_mae, summarize
from metae.solver import SplitStepBPM
from metae.utils import load_model, normalize_input

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "pretrain.pt"))
    ap.add_argument("--n", type=int, default=128, help="samples per pattern family")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    model, ck = load_model(args.ckpt)
    print(f"[ckpt] preset={ck['preset']} step={ck['step']} val={ck.get('val')}")

    solver = SplitStepBPM()
    sampler = FreeformAtomSampler(generator=torch.Generator().manual_seed(2026))
    fams = ["random", "island", "stripe"]
    N, E, labs = [], [], []
    with torch.no_grad():
        for k, fn in enumerate([sampler.random_pattern, sampler.island_pattern, sampler.stripe_pattern]):
            nm = fn(args.n)
            N.append(nm)
            E.append(solver(nm))
            labs += [fams[k]] * args.n
    nm = torch.cat(N)
    gt = torch.cat(E)
    with torch.no_grad():
        pred = model(normalize_input(nm))

    s = summarize(pred, gt, labs)
    print("\n== Table 1 (per pattern family) ==")
    print(f"{'family':>8} | {'n':>4} | {'amp MAE':>8} | {'angle MAE':>9}")
    for k, v in s["groups"].items():
        print(f"{k:>8} | {v['n']:>4} | {v['amp_mae']:>8.3f} | {v['angle_mae']:>9.3f}")
    print(f"{'overall':>8} | {len(gt):>4} | {s['amp_mae']:>8.3f} | {s['angle_mae']:>9.3f}")
    print("(paper Table 1, FDTD ground truth: 0.037-0.114 amp / 0.067-0.123 angle)")

    # ------------------------------------------------------------ Fig 3b
    hp = os.path.join(os.path.dirname(args.ckpt), "pretrain_history.json")
    if os.path.exists(hp):
        h = json.load(open(hp))
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
        axs[0].plot(h["step"], h["train_amp"], label="train")
        axs[0].plot(h["step"], h["val_amp"], label="val")
        axs[0].set_title("amplitude loss $L_A$")
        axs[1].plot(h["step"], h["train_ang"], label="train")
        axs[1].plot(h["step"], h["val_ang"], label="val")
        axs[1].set_title("angle loss $L_\\theta$")
        for a in axs:
            a.set_xlabel("epoch (x200 steps)"); a.legend(); a.grid(alpha=0.3)
        fig.suptitle("Pre-training history (Fig. 3b)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "fig3b_training.png"), dpi=130)

    # ------------------------------------------------------------ Fig 3c
    amp, ang = amplitude_angle_mae(pred, gt)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    colors = {"random": "#4c72b0", "island": "#dd8452", "stripe": "#55a868"}
    for f in fams:
        m = np.array(labs) == f
        ax.scatter(amp[m], ang[m], s=14, alpha=0.65, label=f, c=colors[f], edgecolors="none")
    ax.set_xlabel("normalised amplitude MAE")
    ax.set_ylabel("normalised angle MAE")
    ax.set_title("25k-evaluation equivalent (Fig. 3c)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig3c_scatter.png"), dpi=130)

    # ------------------------------------------------------------ Fig 3d
    pick = [0, args.n, 2 * args.n, 2 * args.n + 1]
    rows = [
        ("Re(E) GT", lambda tc, pc: tc.real, "RdBu"),
        ("Re(E) pred", lambda tc, pc: pc.real, "RdBu"),
        ("Im(E) GT", lambda tc, pc: tc.imag, "RdBu"),
        ("Im(E) pred", lambda tc, pc: pc.imag, "RdBu"),
        ("phase GT", lambda tc, pc: np.angle(tc), "twilight"),
        ("phase pred", lambda tc, pc: np.angle(pc), "twilight"),
    ]
    fig, axs = plt.subplots(len(rows), 4, figsize=(11, 15))
    for c, i in enumerate(pick):
        pc = torch.complex(pred[i, 0], pred[i, 1]).numpy()
        tc = torch.complex(gt[i, 0], gt[i, 1]).numpy()
        for r, (ttl, f, cm) in enumerate(rows):
            axs[r, c].imshow(f(tc, pc), cmap=cm)
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
            if r == 0:
                axs[r, c].set_title(labs[i])
            if c == 0:
                axs[r, c].set_ylabel(ttl)
    fig.suptitle("GT vs prediction (Fig. 3d style)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig3d_field_compare.png"), dpi=120)

    print(f"[saved] {args.out}/fig3b_training.png, fig3c_scatter.png, fig3d_field_compare.png")


if __name__ == "__main__":
    main()
