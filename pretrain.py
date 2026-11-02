"""Pre-train MetaE-former on streamed freeform meta-atoms (paper: 260k samples, 20 epochs).

Data are generated on the fly (random 72 % / island 12 % / stripe 16 %) instead of
storing 260k field maps, so no download and no disk blow-up is needed.

    python pretrain.py --preset fast --steps 3000 --batch 32
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from metae.data import FreeformAtomSampler, N_HI, N_LO
from metae.losses import metaformer_loss
from metae.metrics import summarize
from metae.model import build_model
from metae.solver import SplitStepBPM

ROOT = os.path.dirname(os.path.abspath(__file__))


def normalize_input(n):
    return (n - N_LO) / (N_HI - N_LO) * 2 - 1


def make_val_set(n=512, seed=1234):
    g = torch.Generator().manual_seed(seed)
    sampler = FreeformAtomSampler(generator=g)
    solver = SplitStepBPM()
    with torch.no_grad():
        batches = []
        labels = []
        per = n // 3
        for k, fn in enumerate([sampler.random_pattern, sampler.island_pattern, sampler.stripe_pattern]):
            nm = fn(per)
            batches.append(nm)
            labels += [k] * per
        nm = torch.cat(batches)
        fld = solver(nm)
    return nm, fld, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="fast", choices=["fast", "tiny", "paper"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--out", default=os.path.join(ROOT, "checkpoints", "pretrain.pt"))
    ap.add_argument("--sched", default="cosine", choices=["cosine", "plateau"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--amp-weight", action="store_true", help="weight phase loss by |E|")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model = build_model(args.preset)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] preset={args.preset} params={n_par/1e6:.2f}M threads={torch.get_num_threads()}")

    solver = SplitStepBPM()
    sampler = FreeformAtomSampler(generator=torch.Generator().manual_seed(args.seed))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.318, min_lr=1e-6)
    base_lrs = [g["lr"] for g in opt.param_groups]

    xv, yv, lab = make_val_set(512)
    xv_in = normalize_input(xv)

    hist = {"step": [], "train_amp": [], "train_ang": [], "val_amp": [], "val_ang": [],
            "val_norm_mae": []}
    best = float("inf")
    t0 = time.time()

    for step in range(1, args.steps + 1):
        # warm-up then cosine / plateau schedule
        if step <= args.warmup:
            lr = args.lr * step / args.warmup
        elif args.sched == "cosine":
            prog = (step - args.warmup) / max(1, args.steps - args.warmup)
            lr = 1e-6 + (args.lr - 1e-6) * 0.5 * (1 + math.cos(math.pi * prog))
        else:
            lr = opt.param_groups[0]["lr"]
        for g in opt.param_groups:
            g["lr"] = lr

        model.train()
        nm = sampler.sample(args.batch)
        with torch.no_grad():
            fld = solver(nm)
        pred = model(normalize_input(nm))
        loss, la, lt = metaformer_loss(
            pred, fld, alpha=args.alpha, amp_weighted_phase=args.amp_weight, return_parts=True
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0 or step == 1:
            print(
                f"step {step:5d} | loss {loss.item():.4f} | amp {la.item():.4f} "
                f"| ang {lt.item():.4f} | lr {lr:.2e} | {time.time()-t0:.0f}s",
                flush=True,
            )

        if step % args.val_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                pv = model(xv_in)
                _, va, vt = metaformer_loss(pv, yv, alpha=args.alpha, return_parts=True)
                s = summarize(pv, yv, lab)
            if args.sched == "plateau":
                sched.step(va.item() + args.alpha * vt.item())
            hist["step"].append(step)
            hist["train_amp"].append(la.item())
            hist["train_ang"].append(lt.item())
            hist["val_amp"].append(va.item())
            hist["val_ang"].append(vt.item())
            hist["val_norm_mae"].append(s["norm_mae"])
            print(
                f"  [val] amp {va.item():.4f} ang {vt.item():.4f} normMAE {s['norm_mae']:.4f}",
                flush=True,
            )
            score = va.item() + args.alpha * vt.item()
            if score < best:
                best = score
                torch.save(
                    {"model": model.state_dict(), "preset": args.preset, "step": step,
                     "alpha": args.alpha, "val": s},
                    args.out,
                )
            hist_path = os.path.join(os.path.dirname(args.out), "pretrain_history.json")
            with open(hist_path, "w") as f:
                json.dump(hist, f, indent=1)

    print(f"[done] best val loss {best:.4f} in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
