"""Mesh data pipeline package.

Isolated, single-responsibility modules for the geometric embedding baseline:

* :mod:`mesh_loader`  -- load + normalize raw meshes (Trimesh).
* :mod:`graph_builder` -- convert a normalized mesh into a PyG ``Data`` graph.
* :mod:`pipeline`     -- orchestrate load -> normalize -> graph conversion.

See ``docs/ARCHITECT.md`` section 2.1 (Geometry Encoding Module) for the
tensor contract these modules satisfy.
"""

from .mesh_loader import LoadedMesh, load_mesh, normalize_mesh
from .graph_builder import mesh_to_graph
from .pipeline import build_graph_from_path

__all__ = [
    "LoadedMesh",
    "load_mesh",
    "normalize_mesh",
    "mesh_to_graph",
    "build_graph_from_path",
]
