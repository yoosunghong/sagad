"""Deformation sandbox: apply an [N, 4] RGBA mask to a mesh graph.

Composes the per-channel operators (see :mod:`operators`) into the composite
World-Position-Offset accumulator, gates it by the mobility channel
(``mob = 1 - fixed``), and returns the deformed mesh plus vertex-degradation
telemetry. Implements docs/ARCHITECT.md sec. 2.3.

Responsibility boundary: this module performs NO rendering (that is the
multi-view pipeline, a separate Phase 2 task) and NO reward computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import trimesh
from torch_geometric.data import Data

from data.logging_utils import get_logger
from .operators import bend_offset, noise_offset, scale_offset

log = get_logger(__name__)


@dataclass
class DeformConfig:
    """Magnitudes / parameters for the deformation operators.

    Defaults are tuned for unit-bounding-sphere normalized meshes (the data
    pipeline output): displacements are a modest fraction of the unit radius.
    """

    bend_strength: float = 0.6      # max bend angle (radians) at full height/mask
    noise_strength: float = 0.05    # max along-normal noise displacement
    noise_frequency: float = 5.0    # spatial frequency of the value-noise field
    scale_strength: float = 0.15    # max radial swell fraction
    noise_seed: int = 0
    bend_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)  # horizontal hinge
    up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)    # height direction


@dataclass
class DeformResult:
    """Deformed geometry plus the telemetry gathered while producing it."""

    pos: torch.Tensor                    # (N, 3) deformed vertex positions
    offset: torch.Tensor                 # (N, 3) applied displacement
    mesh: trimesh.Trimesh                # deformed mesh (orig faces + new verts)
    telemetry: dict = field(default_factory=dict)


def _edge_length_change(pos: torch.Tensor, new_pos: torch.Tensor,
                        edge_index: torch.Tensor) -> dict:
    """Relative per-edge length change -- a vertex-degradation scale metric."""
    src, dst = edge_index[0], edge_index[1]
    l0 = (pos[src] - pos[dst]).norm(dim=1)
    l1 = (new_pos[src] - new_pos[dst]).norm(dim=1)
    rel = (l1 - l0).abs() / l0.clamp(min=1e-8)
    return {
        "edge_len_rel_change_mean": float(rel.mean()),
        "edge_len_rel_change_max": float(rel.max()),
        "edge_len_rel_change_p99": float(rel.quantile(0.99)),
    }


def apply_masks(data: Data, masks: torch.Tensor,
                config: DeformConfig | None = None) -> DeformResult:
    """Deform ``data`` by the per-vertex RGBA ``masks`` ([N, 4], in [0, 1]).

    Channels: ``[bend (R), noise (G), scale (B), fixed (A)]``. The fixed
    channel is inverted into a mobility gate so ``fixed=1`` pins the vertex
    (docs/ARCHITECT.md sec. 2.3 / 3).
    """
    cfg = config or DeformConfig()
    pos, normal = data.pos, data.normal
    device = pos.device

    if masks.shape != (pos.shape[0], 4):
        raise ValueError(f"masks must be (N, 4); got {tuple(masks.shape)} for N={pos.shape[0]}")
    masks = masks.to(device=device, dtype=pos.dtype)
    if not torch.isfinite(masks).all():
        raise ValueError("NaN/Inf detected in input masks")

    m_bend, m_noise, m_scale, m_fixed = masks[:, 0], masks[:, 1], masks[:, 2], masks[:, 3]
    axis = torch.tensor(cfg.bend_axis, device=device, dtype=pos.dtype)
    up = torch.tensor(cfg.up_axis, device=device, dtype=pos.dtype)

    off_bend = bend_offset(pos, m_bend, axis=axis, up=up, strength=cfg.bend_strength)
    off_noise = noise_offset(pos, normal, m_noise,
                             strength=cfg.noise_strength,
                             frequency=cfg.noise_frequency, seed=cfg.noise_seed)
    off_scale = scale_offset(pos, m_scale, strength=cfg.scale_strength)

    for name, off in (("bend", off_bend), ("noise", off_noise), ("scale", off_scale)):
        if not torch.isfinite(off).all():
            raise ValueError(f"NaN/Inf detected in {name} offset")

    composite = off_bend + off_noise + off_scale
    mobility = (1.0 - m_fixed).unsqueeze(1)        # fixed=1 -> locked
    offset = mobility * composite
    new_pos = pos + offset

    if not torch.isfinite(new_pos).all():
        raise ValueError("NaN/Inf detected in deformed vertex positions")

    # -- telemetry (CLAUDE.md: vertex-degradation scales, NaN monitoring) ----
    disp = offset.norm(dim=1)
    telemetry = {
        "num_vertices": int(pos.shape[0]),
        "locked_fraction": float((m_fixed >= 0.999).float().mean()),
        "off_bend_mean": float(off_bend.norm(dim=1).mean()),
        "off_noise_mean": float(off_noise.norm(dim=1).mean()),
        "off_scale_mean": float(off_scale.norm(dim=1).mean()),
        "disp_mean": float(disp.mean()),
        "disp_max": float(disp.max()),
    }
    if hasattr(data, "edge_index") and data.edge_index is not None:
        telemetry.update(_edge_length_change(pos, new_pos, data.edge_index))

    faces = data.face.t().cpu().numpy() if data.face is not None else None
    mesh = trimesh.Trimesh(
        vertices=new_pos.detach().cpu().numpy(),
        faces=faces,
        process=False,
    )

    log.info(
        "deform applied | N=%d locked=%.1f%% disp(mean/max)=%.4f/%.4f "
        "edge_rel(mean/max)=%.4f/%.4f off(bend/noise/scale)=%.4f/%.4f/%.4f",
        telemetry["num_vertices"], 100.0 * telemetry["locked_fraction"],
        telemetry["disp_mean"], telemetry["disp_max"],
        telemetry.get("edge_len_rel_change_mean", float("nan")),
        telemetry.get("edge_len_rel_change_max", float("nan")),
        telemetry["off_bend_mean"], telemetry["off_noise_mean"], telemetry["off_scale_mean"],
    )
    return DeformResult(pos=new_pos, offset=offset, mesh=mesh, telemetry=telemetry)
