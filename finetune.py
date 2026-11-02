"""Fine-tune the pre-trained MetaE-former on task-specific data (paper Fig. 2d / Fig. 4).

Following the paper's self-supervised data engine:
1. use the pre-trained model to inverse-design metasurface patches for the target
   task (vortex generators, l = 2,4,6,8, focal 20 um);
2. archive every intermediate RI profile of the optimisation (paper S5);
3. label them with the ground-truth solver (RCWA in the paper, split-step BPM here)
   -> task-specific fine-tuning set;
4. fine-tune with differential learning rates (Method - Pre-training and fine-tuning):
   upsampling 0.5a | NhA blocks 5a/8a/10a | downsampling 15a.

    python finetune.py --ckpt checkpoints/pretrain.pt --collect 60 --steps 300
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from metae.data import N_HI, N_LO
from metae.losses import metaformer_loss
from metae.metrics import summarize
from metae.model import build_model
from metae.solver import SplitStepBPM
from metae.tasks import build_target_phase
from metae.utils import normalize_input

PITCH_PX, CROP = 100, 84


def collect_task_data(model, n_collect, iters, focal, l_list, n_bg, seed, threads=8):
    """Inverse-design patches with the pre-trained model, archive intermediate RIs."""
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    device = "cpu"
    n1, n2 = n_bg, N_HI
    samples = []
    per_round = 16  # patch batch
    rounds = max(1, n_collect // (per_round * iters) + (1 if n_collect % (per_round * iters) else 0))
    for rd in range(rounds):
        l = l_list[rd % len(l_list)]
        # random patch positions inside a 4x4-patch aperture
        pos = torch.randint(0, 4, (per_round, 2))
        ext = 3 * CROP + PITCH_PX
        c = (torch.arange(ext, dtype=torch.float32) - (ext - 1) / 2) * 50e-9
        Yg, Xg = torch.meshgrid(c, c, indexing="ij")
        ph_full = build_target_phase("vortex", n_pix=ext, focal=focal, n_bg=n_bg, l=int(l))
        v = torch.randn(per_round, 25, 25) * 0.1
        v.requires_grad_(True)
        opt = torch.optim.Adam([v], lr=0.1)
        tgt = torch.stack(
            [
                ph_full[i * CROP : i * CROP + PITCH_PX, j * CROP : j * CROP + PITCH_PX][8 : 92, 8 : 92]
                for i, j in pos.tolist()
            ]
        )
        for it in range(1, iters + 1):
            n_norm = torch.tanh(v)
            n_abs = n1 + (n2 - n1) * (n_norm + 1) / 2
            pred = model(normalize_input(n_abs))
            pc = torch.complex(pred[:, 0], pred[:, 1])[:, 8 : 8 + CROP, 8 : 8 + CROP]
            l_phase = ((1 - torch.cos(torch.angle(pc) - tgt)) / 2).mean()
            loss = l_phase + (5.0 if it > 0.75 * iters else 0.0) * (1 - n_norm**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                samples.append(n_abs.clone())
        print(f"  [collect] round {rd+1}/{rounds} l={l} done ({len(samples)} snapshots)", flush=True)
    X = torch.cat(samples)[:n_collect]
    return X


def make_gt(solver, X, chunk=32):
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            outs.append(solver(X[i : i + chunk]))
    return torch.cat(outs)


def build_param_groups(model, a):
    """Differential LRs (paper Method): up 0.5a | enc blocks 5/8/10a | down 15a."""
    groups = [
        (0.5, [model.up1, model.up2, model.up3, model.up4, model.fuse1, model.fuse2,
               model.cpe4, model.cpe5, model.head]),
        (5.0, [model.stem, model.cpe1, model.s1]),
        (8.0, [model.cpe2, model.s2]),
        (10.0, [model.cpe3, model.s3]),
        (15.0, [model.down1, model.down2]),
    ]
    return [
        {"params": [p for m in mods for p in m.parameters()], "lr": a * f}
        for f, mods in groups
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join("checkpoints", "pretrain.pt"))
    ap.add_argument("--out", default=os.path.join("checkpoints", "finetune.pt"))
    ap.add_argument("--task", default="vortex")
    ap.add_argument("--focal", type=float, default=20e-6)
    ap.add_argument("--l-list", default="2,4,6,8")
    ap.add_argument("--n-bg", type=float, default=1.0)
    ap.add_argument("--collect", type=int, default=800, help="RI snapshots to collect")
    ap.add_argument("--iters", type=int, default=50, help="design iters per collection round")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr-base", type=float, default=5e-5, help="a in the differential-LR scheme")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--cache", default=os.path.join("checkpoints", "finetune_data_vortex.npz"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    os.makedirs("checkpoints", exist_ok=True)

    model, ck = load_model_ckpt(args.ckpt)
    print(f"[ckpt] {args.ckpt} preset={ck.get('preset')} step={ck.get('step')}")

    # ------------------------------------------------ collect / load data
    if os.path.exists(args.cache):
        d = np.load(args.cache)
        X, Y = torch.from_numpy(d["X"]), torch.from_numpy(d["Y"])
        print(f"[data] loaded {len(X)} cached task samples")
    else:
        t0 = time.time()
        with torch.no_grad():
            pass
        X = collect_task_data(
            model, args.collect, args.iters, args.focal,
            [int(s) for s in args.l_list.split(",")], args.n_bg, args.seed, args.threads,
        )
        solver = SplitStepBPM()
        Y = make_gt(solver, X)
        np.savez_compressed(args.cache, X=X.numpy(), Y=Y.numpy())
        print(f"[data] collected {len(X)} task samples in {time.time()-t0:.0f}s -> {args.cache}")

    n_val = int(len(X) * args.val_frac)
    perm = torch.randperm(len(X))
    Xv, Yv = normalize_input(X[perm[:n_val]]), Y[perm[:n_val]]
    Xt, Yt = X[perm[n_val:]], Y[perm[n_val:]]

    with torch.no_grad():
        pv = model(Xv)
        _, va, vt = metaformer_loss(pv, Yv, return_parts=True)
        print(f"[test] PRE-trained model on task test set: amp {va.item():.4f} ang {vt.item():.4f} "
              f"normMAE {summarize(pv, Yv)['norm_mae']:.4f}")

    # ------------------------------------------------ fine-tune
    model.train()
    opt = torch.optim.AdamW(build_param_groups(model, args.lr_base), weight_decay=1e-2)
    hist = {"step": [], "test_amp": [], "test_ang": [], "test_nmae": []}
    best = float("inf")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(Xt), (args.batch,))
        nm = Xt[idx]
        with torch.no_grad():
            fld = Yt[idx]
        pred = model(normalize_input(nm))
        loss, la, lt = metaformer_loss(pred, fld, return_parts=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == 1:
            print(f"step {step:4d} | loss {loss.item():.4f} | {time.time()-t0:.0f}s", flush=True)
        if step % 50 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                pv = model(Xv)
                _, va, vt = metaformer_loss(pv, Yv, return_parts=True)
                s = summarize(pv, Yv)
            model.train()
            hist["step"].append(step)
            hist["test_amp"].append(va.item())
            hist["test_ang"].append(vt.item())
            hist["test_nmae"].append(s["norm_mae"])
            print(f"  [test] amp {va.item():.4f} ang {vt.item():.4f} normMAE {s['norm_mae']:.4f}")
            score = va.item() + vt.item()
            if score < best:
                best = score
                torch.save({"model": model.state_dict(), "preset": ck.get("preset"),
                            "step": ck.get("step"), "finetuned": args.task, "val": s}, args.out)
    with open(os.path.join("outputs", "finetune_history.json"), "w") as f:
        json.dump(hist, f, indent=1)
    print(f"[done] fine-tuned -> {args.out}")


def load_model_ckpt(path):
    from metae.utils import load_model

    return load_model(path)


if __name__ == "__main__":
    main()
