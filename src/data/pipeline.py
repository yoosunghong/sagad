"""End-to-end orchestration: file path -> normalized PyG graph.

Thin composition layer over :mod:`mesh_loader` and :mod:`graph_builder`.
Keeping orchestration separate from the loader/builder preserves the
single-responsibility split required by CLAUDE.md.
"""

from __future__ import annotations

from pathlib import Path

from torch_geometric.data import Data

from .graph_builder import mesh_to_graph
from .logging_utils import get_logger
from .mesh_loader import load_mesh, normalize_mesh

log = get_logger(__name__)


def build_graph_from_path(path: str | Path) -> Data:
    """Load, normalize, and graph-convert the mesh at ``path``.

    Source-frame inverse transform (centroid/scale) is attached to the
    returned graph for later coordinate round-tripping.
    """
    mesh = load_mesh(path)
    loaded = normalize_mesh(mesh, source_path=path)
    data = mesh_to_graph(loaded.mesh)

    # Persist the inverse-normalization telemetry on the graph itself.
    data.source_path = str(loaded.source_path)
    data.source_centroid = loaded.centroid.tolist()
    data.source_scale = float(loaded.scale)

    log.info("pipeline complete for %s", Path(path).name)
    return data
