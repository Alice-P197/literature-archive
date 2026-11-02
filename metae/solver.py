"""Approximate, fully differentiable Maxwell solver for a 5 um x 5 um pillar patch.

The paper obtains ground-truth fields with RCWA/FDTD. Those solvers are not
available on this machine (no Lumerical / no GPU cluster), so we substitute a
physically motivated split-step beam-propagation model (BPM):

    pillars: 25 x 25 square pillars, pitch 200 nm, height 800 nm, n in [1, 3.526]
    field  : x-polarised plane wave, lambda = 1064 nm
    output : complex Ex at z = lambda above the pillar top, 100 x 100 pixels
             (50 nm sampling, exactly the paper's output grid)

The model keeps the two ingredients that make the inverse-design problem
non-trivial: (i) strong phase accumulation inside the pillars (up to ~12 rad),
(ii) non-local, multi-slice diffraction coupling between pillars. It is NOT a
full-wave solver: it neglects reflection / multiple scattering in z, so absolute
numbers differ from FDTD. Everything downstream (dataset, network, loss, inverse
design) follows the paper exactly and is solver-agnostic - swap this class for an
RCWA/FDTD wrapper to run on real data.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

N_LO, N_HI = 1.0, 3.526


class SplitStepBPM(torch.nn.Module):
    def __init__(
        self,
        grid: int = 100,
        dx: float = 50e-9,
        wavelength: float = 1064e-9,
        height: float = 800e-9,
        n_slices: int = 8,
        edge_sigma_px: float = 0.6,
        n_lo: float = N_LO,
        n_hi: float = N_HI,
        device: str = "cpu",
    ):
        super().__init__()
        self.grid, self.dx, self.wl, self.height = grid, dx, wavelength, height
        self.n_slices = n_slices
        self.n_lo, self.n_hi = n_lo, n_hi
        self.k0 = 2.0 * math.pi / wavelength
        self.dz = height / n_slices

        kx = 2.0 * math.pi * torch.fft.fftfreq(grid, d=dx)
        KX, KY = torch.meshgrid(kx, kx, indexing="ij")
        # kz = sqrt(k0^2 - kx^2 - ky^2); principal branch -> propagating real,
        # evanescent pure imaginary (decaying).
        kz = torch.sqrt((self.k0**2 - KX**2 - KY**2).to(torch.complex64))
        self.register_buffer("kz", kz.to(device))

        # anti-aliasing kernel for the hard pillar edges (sub-pixel smoothing)
        if edge_sigma_px > 0:
            r = max(1, int(math.ceil(3 * edge_sigma_px)))
            xs = torch.arange(-r, r + 1, dtype=torch.float32)
            g = torch.exp(-0.5 * (xs / edge_sigma_px) ** 2)
            g = g / g.sum()
            k2d = torch.outer(g, g)[None, None]
            self.register_buffer("aa", k2d.to(device))
            self.aa_r = r
        else:
            self.aa, self.aa_r = None, 0

    # ------------------------------------------------------------------ utils
    def _propagate(self, E: torch.Tensor, dz: float) -> torch.Tensor:
        """Angular-spectrum propagation over dz in vacuum."""
        Fk = torch.fft.fft2(E)
        Fk = Fk * torch.exp(1j * self.kz * dz)
        return torch.fft.ifft2(Fk)

    def _upsample(self, n_map: torch.Tensor) -> torch.Tensor:
        """(B,25,25) index matrix -> (B,100,100) with sub-pixel smoothing."""
        x = n_map.unsqueeze(1)
        x = F.interpolate(x, size=(self.grid, self.grid), mode="nearest")
        if self.aa is not None:
            x = F.pad(x, (self.aa_r,) * 4, mode="replicate")
            x = F.conv2d(x, self.aa.expand(1, 1, *self.aa.shape[-2:]))
        return x.squeeze(1)

    # ------------------------------------------------------------------- core
    def forward(self, n_map: torch.Tensor, return_complex: bool = False):
        """n_map: (B,25,25) refractive index (absolute, 1..3.526).

        returns (B,2,100,100) = [real, imag] of Ex normalised by max|Ex|.
        """
        dtype = n_map.dtype
        n = self._upsample(n_map)
        phase_per_slice = 1j * self.k0 * (n - 1.0) * self.dz
        E = torch.ones_like(n, dtype=torch.complex64)

        for _ in range(self.n_slices):
            E = E * torch.exp(phase_per_slice.to(torch.complex64))
            E = self._propagate(E, self.dz)

        E = self._propagate(E, self.wl)  # pillar top -> z = lambda

        scale = E.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        E = E / scale
        if return_complex:
            return E
        return torch.stack([E.real, E.imag], dim=1).to(dtype)

    # helper for verification of the assembled metasurface
    @torch.no_grad()
    def propagate_field(self, E: torch.Tensor, z: float) -> torch.Tensor:
        """Propagate an already-computed complex field by z (same grid)."""
        return self._propagate(E, z)


def angular_spectrum_propagate(
    E: torch.Tensor, z: float, dx: float, wavelength: float = 1064e-9, n_bg: float = 1.0
) -> torch.Tensor:
    """Propagate a complex field (B,H,W) or (H,W) by z in a medium n_bg."""
    squeeze = E.dim() == 2
    if squeeze:
        E = E.unsqueeze(0)
    _, H, W = E.shape
    k0 = n_bg * 2.0 * math.pi / wavelength
    kx = 2.0 * math.pi * torch.fft.fftfreq(W, d=dx, device=E.device)
    ky = 2.0 * math.pi * torch.fft.fftfreq(H, d=dx, device=E.device)
    KX, KY = torch.meshgrid(kx, ky, indexing="xy")
    kz = torch.sqrt((k0**2 - KX**2 - KY**2).to(torch.complex64))
    out = torch.fft.ifft2(torch.fft.fft2(E) * torch.exp(1j * kz * z))
    return out.squeeze(0) if squeeze else out
