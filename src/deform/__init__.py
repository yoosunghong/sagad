"""Deformation sandbox.

Isolated, single-responsibility modules for Phase 2's deformation simulator
(CLAUDE.md modularity rule). Kept separate from the data pipeline
(`src/data`), the embedding model (`src/models`), and the downstream reward /
RL stages.

* :mod:`operators` -- pure, vectorized per-channel displacement operators
  (bend / noise / scale) + the seeded 3D value-noise field.
* :mod:`sandbox`   -- applies an [N, 4] RGBA mask to a graph and returns the
  deformed mesh plus vertex-degradation telemetry.

See ``docs/ARCHITECT.md`` sections 2.3 (Deformation Sandbox) and 3 (UE5
channel map) for the operator definitions and the shared mobility convention.
"""

from .operators import bend_offset, noise_offset, scale_offset, value_noise_3d
from .sandbox import DeformConfig, DeformResult, apply_masks

__all__ = [
    "bend_offset",
    "noise_offset",
    "scale_offset",
    "value_noise_3d",
    "DeformConfig",
    "DeformResult",
    "apply_masks",
]
