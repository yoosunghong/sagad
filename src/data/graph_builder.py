"""Trimesh -> PyTorch Geometric graph conversion.

Implements the input contract from docs/ARCHITECT.md sec. 2.1:
    Input: vertex matrix V in R^{N x 3} and mesh adjacency.
    Output: per-node features feeding the GATv2 / MeshMAE encoder.

Node feature layout (x): [pos_xyz (3), normal_xyz (3)] -> R^{N x 6}.
Edges are the undirected, bidirectional mesh edges (COO ``edge_index``).

Optimization note (CLAUDE.md): adjacency is built with vectorized numpy/torch
ops; we never materialize a dense N x N adjacency matrix.
"""

from __future__ import annotations

import numpy as np
import torch
import trimesh
from torch_geometric.data import Data

from .logging_utils import get_logger

log = get_logger(__name__)


def mesh_to_graph(mesh: trimesh.Trimesh) -> Data:
    """Convert a normalized mesh into a PyG :class:`Data` object."""
    # np.array (copy) -> writable buffers, avoiding torch's non-writable warning.
    pos = torch.as_tensor(np.array(mesh.vertices), dtype=torch.float32)
    normals = torch.as_tensor(np.array(mesh.vertex_normals), dtype=torch.float32)
    faces = torch.as_tensor(np.array(mesh.faces), dtype=torch.long)

    # Undirected COO edge_index from unique mesh edges (both directions).
    edges = torch.as_tensor(np.array(mesh.edges_unique), dtype=torch.long)
    edge_index = torch.cat([edges, edges.flip(1)], dim=0).t().contiguous()

    x = torch.cat([pos, normals], dim=1)  # (N, 6)

    if not torch.isfinite(x).all():
        raise ValueError("NaN/Inf detected in node feature tensor x")

    data = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        face=faces.t().contiguous(),  # (3, F) PyG convention
        normal=normals,
    )

    # Telemetry: degree distribution flags isolated / degenerate vertices.
    deg = torch.bincount(edge_index[0], minlength=pos.shape[0])
    log.info(
        "graph built | nodes=%d edges=%d feat_dim=%d degree(min/mean/max)=%d/%.2f/%d isolated=%d",
        data.num_nodes, edge_index.shape[1], x.shape[1],
        int(deg.min()), float(deg.float().mean()), int(deg.max()),
        int((deg == 0).sum()),
    )
    return data
