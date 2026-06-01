"""Variant diffusion sampler (docs/ARCHITECT.md sec. 2.5).

Isolated module (separate from deform/rewards/rl/hitl, CLAUDE.md modularity): a
low-dim DDPM that learns the *distribution* of per-instance variant vectors and
samples K of them to scatter K distinct WPO instances from one mesh file.

* :mod:`params`  -- ``VariantParamSpec``: the raw<->model-space schema bridging the
  sampler and the Per-Instance Custom Data bake.
* :mod:`ddpm`    -- the denoiser + DDPM train/sample loops.
"""

from .params import ParamField, VariantParamSpec, building_spec, organic_spec
from .ddpm import (
    DiffusionMLP,
    GaussianDiffusion,
    SinusoidalTimeEmbedding,
    train_diffusion,
)

__all__ = [
    "ParamField",
    "VariantParamSpec",
    "building_spec",
    "organic_spec",
    "DiffusionMLP",
    "GaussianDiffusion",
    "SinusoidalTimeEmbedding",
    "train_diffusion",
]
