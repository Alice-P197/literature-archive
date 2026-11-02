"""MetaE-former: down-sampling / up-sampling Neighborhood-Attention Transformer.

Paper spec (Method - Network architecture):
    16 NhA blocks, NA kernel 5, embed 256, 16 heads, conv/deconv down/up-sampling,
    input 25x25 index matrix, output 100x100 complex field (real + imag).

Topology used here (mirror-symmetric, exact size bookkeeping):
    stem 25x25 -> [n1 x NAB]  -> down 13 -> [n2 x NAB] -> down 7 -> [n3 x NAB]
    -> up 13 (+skip) -> up 25 (+skip) -> up 50 -> up 100 -> conv head -> 2 ch
`preset="paper"` gives dim=256, heads=16, (8,4,4) blocks = 16 blocks.
`preset="fast"`  is a CPU-sized variant with the same structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeighborhoodAttention(nn.Module):
    """Neighborhood attention (Hassani et al., CVPR 2023), dense k x k windows."""

    def __init__(self, dim: int, num_heads: int, kernel_size: int = 5, bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.dim, self.h, self.k = dim, num_heads, kernel_size
        self.dh = dim // num_heads
        self.scale = self.dh**-0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        # learned per-offset relative position bias
        self.rpb = nn.Parameter(torch.zeros(num_heads, kernel_size * kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,C,H,W)
        B, C, H, W = x.shape
        r, kk = self.k // 2, self.k * self.k
        N = H * W

        qkv = self.qkv(x.permute(0, 2, 3, 1))
        q, k, v = qkv.split(self.dim, dim=-1)
        q = q.reshape(B, N, self.h, self.dh).permute(0, 2, 3, 1)  # B,heads,dh,N

        def neighbors(t):
            t = t.reshape(B, H, W, self.h, self.dh).permute(0, 3, 4, 1, 2)  # B,h,dh,H,W
            t = t.reshape(B * self.h, self.dh, H, W)
            t = F.pad(t, (r, r, r, r))
            t = F.unfold(t, self.k)  # (B*h, dh*kk, N)
            return t.view(B, self.h, self.dh, kk, N)

        kk_ = neighbors(k)
        vv_ = neighbors(v)

        attn = torch.einsum("bhdn,bhdkn->bhkn", q, kk_) * self.scale
        attn = attn + self.rpb.view(1, self.h, kk, 1)
        attn = attn.softmax(dim=2)
        out = torch.einsum("bhkn,bhdkn->bhdn", attn, vv_)
        out = out.permute(0, 3, 1, 2).reshape(B, N, C)
        return self.proj(out).reshape(B, H, W, C).permute(0, 3, 1, 2)


class NABlock(nn.Module):
    """LN -> NhA -> skip ; LN -> MLP -> skip  (paper Fig. 3a)."""

    def __init__(self, dim: int, num_heads: int, kernel_size: int = 5, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = NeighborhoodAttention(dim, num_heads, kernel_size)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim)
        )

    def _ln(self, x, ln):
        return ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, x):
        x = x + self.attn(self._ln(x, self.ln1))
        x = x + self.mlp(self._ln(x, self.ln2).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class CPE(nn.Module):
    """Conditional positional encoding: depth-wise conv residual."""

    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x):
        return x + self.dw(x)


def _stage(dim, heads, n, kernel):
    return nn.Sequential(*[NABlock(dim, heads, kernel) for _ in range(n)])


class MetaEFormer(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 2,
        dim: int = 64,
        heads: int = 4,
        blocks: tuple[int, int, int] = (4, 2, 2),
        kernel: int = 5,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, dim, 3, 1, 1)
        self.cpe1 = CPE(dim)
        self.s1 = _stage(dim, heads, blocks[0], kernel)

        self.down1 = nn.Conv2d(dim, dim * 2, 3, 2, 1)          # 25 -> 13
        self.cpe2 = CPE(dim * 2)
        self.s2 = _stage(dim * 2, heads * 2, blocks[1], kernel)

        self.down2 = nn.Conv2d(dim * 2, dim * 2, 3, 2, 1)      # 13 -> 7
        self.cpe3 = CPE(dim * 2)
        self.s3 = _stage(dim * 2, heads * 2, blocks[2], kernel)

        self.up1 = nn.ConvTranspose2d(dim * 2, dim * 2, 3, 2, 1)   # 7 -> 13
        self.fuse1 = nn.Conv2d(dim * 4, dim * 2, 1)
        self.cpe4 = CPE(dim * 2)
        self.up2 = nn.ConvTranspose2d(dim * 2, dim, 3, 2, 1)       # 13 -> 25
        self.fuse2 = nn.Conv2d(dim * 2, dim, 1)
        self.cpe5 = CPE(dim)
        self.up3 = nn.ConvTranspose2d(dim, dim // 2, 3, 2, 1, output_padding=1)      # 25 -> 50
        self.up4 = nn.ConvTranspose2d(dim // 2, dim // 4, 3, 2, 1, output_padding=1)  # 50 -> 100

        self.head = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 4, 3, 1, 1), nn.GELU(),
            nn.Conv2d(dim // 4, out_ch, 1),
        )

    def forward(self, n_map: torch.Tensor) -> torch.Tensor:
        """n_map: (B,25,25) index, or (B,1,25,25). returns (B,2,100,100)."""
        if n_map.dim() == 3:
            n_map = n_map.unsqueeze(1)
        x = self.cpe1(self.stem(n_map))
        s1 = self.s1(x)

        d1 = self.s2(self.cpe2(self.down1(s1)))      # 13
        d2 = self.s3(self.cpe3(self.down2(d1)))      # 7

        u1 = self.up1(d2)                            # 13
        u1 = self.fuse1(torch.cat([u1, d1], 1))
        u2 = self.up2(self.cpe4(u1))                 # 25
        u2 = self.cpe5(self.fuse2(torch.cat([u2, s1], 1)))
        u3 = self.up3(u2)                            # 50
        u4 = self.up4(u3)                            # 100
        return self.head(u4)


PRESETS = {
    "paper": dict(dim=256, heads=16, blocks=(8, 4, 4), kernel=5, mlp_ratio=4.0),
    "fast": dict(dim=48, heads=4, blocks=(3, 2, 2), kernel=5, mlp_ratio=2.0),
    "tiny": dict(dim=32, heads=4, blocks=(2, 1, 1), kernel=5, mlp_ratio=2.0),
}


def build_model(preset: str = "fast", **kw) -> MetaEFormer:
    cfg = dict(PRESETS[preset])
    cfg.update(kw)
    return MetaEFormer(**cfg)
