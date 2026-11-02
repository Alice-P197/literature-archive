"""Freeform meta-atom sampler, following Fig. 2 of the paper.

Four families (25 x 25 pillars, 200 nm pitch):
  * random      : i.i.d. random index matrix                        (72 %)
  * island-like : N random rectangles with random index, summed    (12 %)
  * certain     : 2-D cosine stripes  cos[2 pi f (cos t X + sin t Y) + phi]  (12 %)
  * task        : purpose-designed patterns from inverse-design iterations (fine-tune)

60 % continuous index, 40 % binarised (paper numbers).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

N_LO, N_HI = 1.0, 3.526
SIZE = 25


def _gauss_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """x: (B,1,H,W)."""
    if sigma <= 0:
        return x
    r = max(1, int(math.ceil(3 * sigma)))
    xs = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    g = torch.exp(-0.5 * (xs / sigma) ** 2)
    g = g / g.sum()
    k = torch.outer(g, g)[None, None]
    x = F.pad(x, (r,) * 4, mode="replicate")
    return F.conv2d(x, k)


def _to_index(m01: torch.Tensor, binary: bool, n_lo: float, n_hi: float) -> torch.Tensor:
    """map [0,1] field to refractive index, optionally binarised."""
    if binary:
        m01 = (m01 > 0.5).to(m01.dtype)
    return n_lo + (n_hi - n_lo) * m01


class FreeformAtomSampler:
    def __init__(
        self,
        size: int = SIZE,
        n_lo: float = N_LO,
        n_hi: float = N_HI,
        continuous_ratio: float = 0.6,
        noise: float = 0.06,
        device: str = "cpu",
        generator: torch.Generator | None = None,
    ):
        self.size, self.n_lo, self.n_hi = size, n_lo, n_hi
        self.continuous_ratio = continuous_ratio
        self.noise = noise
        self.device = device
        self.g = generator

    def _rand(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, generator=self.g, device=self.device)

    # ------------------------------------------------------------ families
    def random_pattern(self, B: int, ratio: float | None = None) -> torch.Tensor:
        p = self.continuous_ratio if ratio is None else ratio
        cont = _gauss_blur(self._rand(B, 1, self.size, self.size), 1.0)
        cont = (cont - cont.amin(dim=(-2, -1), keepdim=True)) / (
            cont.amax(dim=(-2, -1), keepdim=True) - cont.amin(dim=(-2, -1), keepdim=True) + 1e-6
        )
        bina = _gauss_blur((self._rand(B, 1, self.size, self.size) > 0.5).float(), 0.6)
        pick = (self._rand(B, 1, 1, 1) < p).float()
        m = pick * cont + (1 - pick) * bina
        binary = self._rand(B) > p
        out = torch.empty(B, self.size, self.size, device=self.device)
        for i in range(B):
            out[i] = _to_index(m[i, 0], bool(binary[i]), self.n_lo, self.n_hi)
        return out

    def island_pattern(self, B: int, ratio: float | None = None) -> torch.Tensor:
        p = self.continuous_ratio if ratio is None else ratio
        out = torch.empty(B, self.size, self.size, device=self.device)
        lo = self.n_lo + 0.33 * (self.n_hi - self.n_lo)  # paper: n in [1.33, 3.526]
        for i in range(B):
            m = torch.zeros(self.size, self.size, device=self.device)
            n_blk = int(torch.randint(3, 9, (1,), generator=self.g))
            for _ in range(n_blk):
                h = int(torch.randint(2, 13, (1,), generator=self.g))
                w = int(torch.randint(2, 13, (1,), generator=self.g))
                y = int(torch.randint(0, self.size - h + 1, (1,), generator=self.g))
                x = int(torch.randint(0, self.size - w + 1, (1,), generator=self.g))
                amp = float(torch.rand(1, generator=self.g)) * (self.n_hi - lo) + lo
                m[y : y + h, x : x + w] += amp
            m += self.noise * torch.randn(self.size, self.size, generator=self.g, device=self.device)
            binary = bool(torch.rand(1, generator=self.g) > p)
            if binary:
                out[i] = torch.where(
                    m > 0.5 * (lo + self.n_hi), torch.tensor(self.n_hi, device=self.device),
                    torch.tensor(self.n_lo, device=self.device),
                )
            else:
                out[i] = m.clamp(self.n_lo, self.n_hi)
        return out

    def stripe_pattern(self, B: int, ratio: float | None = None) -> torch.Tensor:
        p = self.continuous_ratio if ratio is None else ratio
        c = torch.linspace(0.0, 1.0, self.size, device=self.device)
        Y, X = torch.meshgrid(c, c, indexing="ij")
        f = (self._rand(B, 1, 1) * 24 + 1) * 2 * math.pi
        tau = (self._rand(B, 1, 1) * 2 - 1) * math.pi
        phi = (self._rand(B, 1, 1) * 2 - 1) * math.pi
        m = torch.cos(f * (torch.cos(tau) * X + torch.sin(tau) * Y) + phi)
        m = _gauss_blur(m.unsqueeze(1), 0.7).squeeze(1)
        binary = self._rand(B) > p
        out = torch.empty(B, self.size, self.size, device=self.device)
        for i in range(B):
            if binary[i]:
                out[i] = _to_index((m[i] > 0).float(), True, self.n_lo, self.n_hi)
            else:
                mm = (m[i] - m[i].min()) / (m[i].max() - m[i].min() + 1e-6)
                out[i] = _to_index(mm, False, self.n_lo, self.n_hi)
        return out

    # --------------------------------------------------------------- public
    def sample(self, B: int, mixture: str = "pretrain", probs=None) -> torch.Tensor:
        if probs is None:
            probs = [0.72, 0.12, 0.16] if mixture == "pretrain" else [0.0, 0.0, 1.0]
        cls = torch.multinomial(
            torch.tensor(probs, device=self.device), B, replacement=True, generator=self.g
        )
        out = torch.empty(B, self.size, self.size, device=self.device)
        for c, fn in enumerate([self.random_pattern, self.island_pattern, self.stripe_pattern]):
            idx = (cls == c).nonzero(as_tuple=True)[0]
            if len(idx):
                out[idx] = fn(len(idx))
        return out
