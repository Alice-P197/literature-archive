"""Shared helpers."""

from __future__ import annotations

import os

import torch

from .data import N_HI, N_LO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def normalize_input(n: torch.Tensor) -> torch.Tensor:
    """absolute index -> [-1, 1]"""
    return (n - N_LO) / (N_HI - N_LO) * 2 - 1


def denormalize_input(x: torch.Tensor) -> torch.Tensor:
    return (x + 1) / 2 * (N_HI - N_LO) + N_LO


def load_model(ckpt_path: str, device: str = "cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    from .model import build_model

    model = build_model(ck.get("preset", "fast"))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def load_or_default(path: str, fallback: str):
    """Prefer the fine-tuned checkpoint; fall back to the pre-trained one."""
    if path and os.path.exists(path):
        return path
    if fallback and os.path.exists(fallback):
        print(f"[info] {path} not found, falling back to {fallback}")
        return fallback
    raise FileNotFoundError(f"no checkpoint at {path} or {fallback}")
