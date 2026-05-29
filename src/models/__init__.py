"""Geometric embedding models.

Isolated, single-responsibility modules for the structural-embedding stage
(CLAUDE.md modularity rule). Kept separate from the data pipeline (`src/data`)
and the downstream RL / reward / export stages.

* :mod:`mesh_mae` -- GATv2-backed Mesh Masked Autoencoder (MeshMAE) providing
  self-supervised per-node structural embeddings Z in R^{N x D}.

See ``docs/ARCHITECT.md`` section 2.1 (Geometry Encoding Module) for the
tensor contract this model satisfies.
"""

from .mesh_mae import MeshMAE, MeshMAEConfig, MeshMAEOutput

__all__ = ["MeshMAE", "MeshMAEConfig", "MeshMAEOutput"]
