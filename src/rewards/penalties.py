"""Deterministic geometry & physics penalty functions.

Each penalty is a non-negative scalar measuring how much a deformation
*degraded* the asset relative to its undeformed state; the PPO composite
reward (Phase 4) subtracts a weighted sum of these. Implements the
ARCHITECT.md sec. 2.4 Geometric Regularizer and the Phase 2 checklist:
Laplacian smoothness, normal consistency, ground contact.

Design choices:
* All penalties are computed by *comparison to the original geometry* (same
  topology, deformed vertices). Organic assets (rocks/trees) are already rough
  and non-watertight, so absolute smoothness is meaningless -- what matters is
  the *increase* in roughness/inconsistency the deformation introduced. This
  also makes every term rigid-motion invariant where it should be.
* Per-vertex math stays in vectorized torch over the sparse graph; only the
  static face-adjacency topology is borrowed from trimesh (computed once and
  cacheable across a PPO rollout).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import trimesh
from torch_geometric.data import Data

from data.logging_utils import get_logger

log = get_logger(__name__)

_EPS = 1e-8


@dataclass
class PenaltyWeights:
    """Sub-weights for aggregating the raw penalties into distortion/physics.

    The outer composite-reward coefficients (gamma, delta) are applied later in
    the Phase 4 PPO loop; these only balance terms *within* each group.
    """

    laplacian: float = 1.0
    normal: float = 1.0
    stretch: float = 1.0
    ground_gap: float = 1.0
    ground_penetration: float = 1.0


# -- geometric distortion ---------------------------------------------------

def laplacian_coordinates(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Uniform (umbrella) Laplacian coordinate ``delta_i = p_i - mean(p_neighbors)``.

    Captures local surface detail; rigid motions leave it unchanged.
    """
    src, dst = edge_index[0], edge_index[1]
    n = pos.shape[0]
    deg = torch.bincount(src, minlength=n).clamp(min=1).to(pos.dtype).unsqueeze(1)
    neighbor_sum = torch.zeros_like(pos).index_add_(0, src, pos[dst])
    return pos - neighbor_sum / deg


def laplacian_distortion(orig_pos: torch.Tensor, new_pos: torch.Tensor,
                         edge_index: torch.Tensor) -> torch.Tensor:
    """Mean shift of Laplacian coordinates -- local surface degradation.

    Zero for any rigid motion (translation/rotation) of the whole mesh; grows
    as the deformation roughens or distorts local neighborhoods.
    """
    d0 = laplacian_coordinates(orig_pos, edge_index)
    d1 = laplacian_coordinates(new_pos, edge_index)
    return (d1 - d0).norm(dim=1).mean()


def _face_normals(pos: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Unit face normals from vertex positions and an ``(F, 3)`` face table."""
    v0, v1, v2 = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    n = torch.cross(v1 - v0, v2 - v0, dim=1)
    return n / (n.norm(dim=1, keepdim=True) + _EPS)


def normal_consistency(orig_pos: torch.Tensor, new_pos: torch.Tensor,
                       faces: torch.Tensor, face_adjacency: torch.Tensor) -> torch.Tensor:
    """Mean *increase* in the dihedral angle between adjacent face normals.

    ``faces``: ``(F, 3)``; ``face_adjacency``: ``(P, 2)`` pairs of adjacent
    face indices. Penalizes the deformation creasing/spiking the surface beyond
    its original faceting; rigid motions give zero.
    """
    a, b = face_adjacency[:, 0], face_adjacency[:, 1]
    n0 = _face_normals(orig_pos, faces)
    n1 = _face_normals(new_pos, faces)

    def _angle(na: torch.Tensor, nb: torch.Tensor) -> torch.Tensor:
        cos = (na * nb).sum(dim=1).clamp(-1.0, 1.0)
        return torch.arccos(cos)

    ang_o = _angle(n0[a], n0[b])
    ang_n = _angle(n1[a], n1[b])
    return torch.relu(ang_n - ang_o).mean()


def edge_stretch(orig_pos: torch.Tensor, new_pos: torch.Tensor,
                 edge_index: torch.Tensor) -> torch.Tensor:
    """Mean relative edge-length change -- a UV/texel-stretch proxy.

    Vertex displacement (the CPU sandbox *and* the UE5 WPO shader) leaves UVs
    pinned per vertex, so any change in an edge's length stretches/compresses
    the texture mapped across it. Penalizing it bounds texel-density smearing
    and the UV-seam tear risk under WPO -- distortions the Laplacian/normal
    terms do not see (a uniform radial swell is locally smooth yet stretches
    every texel). Zero for any rigid motion (edge lengths preserved).
    """
    src, dst = edge_index[0], edge_index[1]
    l0 = (orig_pos[src] - orig_pos[dst]).norm(dim=1)
    l1 = (new_pos[src] - new_pos[dst]).norm(dim=1)
    return ((l1 - l0).abs() / l0.clamp(min=_EPS)).mean()


# -- physics ----------------------------------------------------------------

def ground_contact(orig_pos: torch.Tensor, new_pos: torch.Tensor,
                   up: tuple[float, float, float] = (0.0, 0.0, 1.0)
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Ground-plane contact penalties: ``(gap, penetration)``.

    The ground plane is the original lowest extent along ``up``. ``gap`` is how
    far the deformed base floats *above* it (loss of contact); ``penetration``
    is the mean depth of vertices pushed *below* it (sinking into the terrain).
    """
    up_t = torch.tensor(up, dtype=orig_pos.dtype, device=orig_pos.device)
    ground = (orig_pos @ up_t).min()
    h_new = new_pos @ up_t
    gap = torch.relu(h_new.min() - ground)
    penetration = torch.relu(ground - h_new).mean()
    return gap, penetration


# -- aggregate --------------------------------------------------------------

def _face_adjacency(faces: torch.Tensor) -> torch.Tensor:
    """Topological adjacent-face pairs ``(P, 2)`` via trimesh (topology only)."""
    f = faces.detach().cpu().numpy()
    # Dummy vertices: face_adjacency is purely topological.
    mesh = trimesh.Trimesh(
        vertices=np.zeros((int(f.max()) + 1, 3)), faces=f, process=False,
    )
    # np.array (copy) -> writable buffer, avoiding torch's non-writable warning.
    return torch.as_tensor(np.array(mesh.face_adjacency), dtype=torch.long,
                           device=faces.device)


def evaluate_penalties(orig_data: Data, new_pos: torch.Tensor,
                       weights: PenaltyWeights | None = None,
                       up: tuple[float, float, float] = (0.0, 0.0, 1.0),
                       face_adjacency: torch.Tensor | None = None) -> dict:
    """Compute all penalties for a deformation and aggregate distortion/physics.

    ``orig_data`` carries the undeformed ``pos``/``face``/``edge_index``;
    ``new_pos`` is the deformed vertex tensor (e.g. ``DeformResult.pos``).
    Pass a precomputed ``face_adjacency`` to avoid recomputing it per PPO step.
    """
    w = weights or PenaltyWeights()
    orig_pos = orig_data.pos
    faces = orig_data.face.t().contiguous()  # (3, F) -> (F, 3)
    if face_adjacency is None:
        face_adjacency = _face_adjacency(faces)

    lap = laplacian_distortion(orig_pos, new_pos, orig_data.edge_index)
    nrm = normal_consistency(orig_pos, new_pos, faces, face_adjacency)
    stretch = edge_stretch(orig_pos, new_pos, orig_data.edge_index)
    gap, pen = ground_contact(orig_pos, new_pos, up=up)

    terms = {
        "laplacian": float(lap),
        "normal": float(nrm),
        "stretch": float(stretch),
        "ground_gap": float(gap),
        "ground_penetration": float(pen),
    }
    for k, v in terms.items():
        if not np.isfinite(v):
            raise ValueError(f"NaN/Inf detected in penalty term '{k}'")

    distortion = (w.laplacian * terms["laplacian"] + w.normal * terms["normal"]
                  + w.stretch * terms["stretch"])
    physics = (w.ground_gap * terms["ground_gap"]
               + w.ground_penetration * terms["ground_penetration"])
    terms["distortion_total"] = distortion
    terms["physics_total"] = physics

    log.info(
        "penalties | laplacian=%.5f normal=%.5f stretch=%.5f ground(gap/pen)=%.5f/%.5f "
        "=> distortion=%.5f physics=%.5f",
        terms["laplacian"], terms["normal"], terms["stretch"], terms["ground_gap"],
        terms["ground_penetration"], distortion, physics,
    )
    return terms
