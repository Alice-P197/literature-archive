"""MetaE-former reproduction package.

Modules
-------
solver  : approximate differentiable Maxwell solver (split-step BPM), plays the
          role of the RCWA/FDTD ground-truth engine in the paper.
data    : freeform meta-atom samplers (random / island-like / stripe / task).
model   : Neighborhood-Attention Transformer (NAT) U-Net, i.e. MetaE-former.
losses  : Eq.(1)-(3) amplitude + periodic-phase loss.
metrics : normalized MAE (Method section).
tasks   : target phase/field builders for metalens / vortex / Airy / custom.
"""

from .model import MetaEFormer, build_model
from .losses import metaformer_loss
from .solver import SplitStepBPM
from .data import FreeformAtomSampler

__all__ = [
    "MetaEFormer",
    "build_model",
    "metaformer_loss",
    "SplitStepBPM",
    "FreeformAtomSampler",
]
