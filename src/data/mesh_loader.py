"""Mesh loading and normalization (Trimesh backend).

Responsibility boundary: this module turns a file on disk into a clean,
coordinate-invariant :class:`trimesh.Trimesh`. It performs NO graph
construction (see :mod:`graph_builder`).

Normalization rationale (docs/DONE/phase_1: coordinate invariance):
incoming meshes are recentred to their centroid and isotropically scaled to
the unit bounding sphere so that the geometry encoder sees scale- and
translation-invariant coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from .logging_utils import get_logger

log = get_logger(__name__)

# Formats trimesh parses natively. FBX is intentionally excluded -- the
# baseline pipeline consumes OBJ/GLB/PLY/STL (see docs/STACK.md sec. 3).
SUPPORTED_SUFFIXES = {".obj", ".ply", ".stl", ".glb", ".gltf", ".off"}


@dataclass
class LoadedMesh:
    """A normalized mesh plus the affine telemetry used to normalize it."""

    mesh: trimesh.Trimesh
    centroid: np.ndarray  # (3,) original centroid removed during normalization
    scale: float          # isotropic divisor applied to reach the unit sphere
    source_path: Path


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a single concatenated mesh from ``path``.

    ``force='mesh'`` collapses multi-geometry scenes into one Trimesh so the
    downstream graph has a single vertex/face table.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"mesh not found: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"unsupported format '{path.suffix}'. "
            f"Supported: {sorted(SUPPORTED_SUFFIXES)}. "
            "FBX must be re-exported to OBJ/GLB before ingest."
        )

    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.shape[0] == 0:
        raise ValueError(f"loaded object from {path} is not a face-based mesh")

    log.info(
        "loaded %s | vertices=%d faces=%d watertight=%s",
        path.name, len(mesh.vertices), len(mesh.faces), mesh.is_watertight,
    )
    return mesh


def normalize_mesh(mesh: trimesh.Trimesh, source_path: str | Path = "") -> LoadedMesh:
    """Recentre to centroid and scale to the unit bounding sphere (in place).

    Returns a :class:`LoadedMesh` carrying the inverse-transform telemetry so
    masks/deformations can be mapped back to the original coordinate frame.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)

    centroid = vertices.mean(axis=0)
    recentred = vertices - centroid
    # Unit bounding-sphere radius == max vertex norm after recentring.
    radius = float(np.linalg.norm(recentred, axis=1).max())
    if radius <= 0.0 or not np.isfinite(radius):
        raise ValueError(f"degenerate mesh: non-finite/zero scale radius ({radius})")

    mesh.vertices = (recentred / radius).astype(np.float32)

    if not np.isfinite(mesh.vertices).all():
        raise ValueError("NaN/Inf detected in vertices after normalization")

    log.info(
        "normalized | centroid=%s scale_radius=%.6f new_extents=%s",
        np.round(centroid, 4).tolist(), radius, np.round(mesh.extents, 4).tolist(),
    )
    return LoadedMesh(
        mesh=mesh, centroid=centroid, scale=radius, source_path=Path(source_path),
    )
