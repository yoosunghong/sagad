"""Headless multi-view renderer (Open3D legacy Visualizer, offscreen).

Renders a mesh from ``n_views`` orbit cameras evenly spaced in azimuth at a
fixed elevation, returning an ``(V, H, W, 3) uint8`` snapshot array. Implements
the multi-angle export of docs/ARCHITECT.md sec. 2.3.

Backend choice: the Filament ``OffscreenRenderer`` fails to initialize on this
headless Windows host, but the legacy ``Visualizer`` with ``visible=False``
produces valid GPU renders (see scripts/_probe_render.py). One window is reused
across all views of a mesh; only the camera extrinsic is rebuilt per view so
the intrinsic stays consistent with the framebuffer aspect.

Telemetry (CLAUDE.md): per-view foreground (non-background) pixel fraction is
logged so empty/black renders -- a silent failure mode -- are caught.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from data.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class RenderConfig:
    """Camera / framebuffer parameters for the multi-view orbit."""

    n_views: int = 4                 # azimuths evenly spaced around the up axis
    image_size: int = 256            # square framebuffer (H == W)
    elevation_deg: float = 20.0      # camera tilt above the horizon
    distance_scale: float = 2.5      # eye distance = scale * bounding radius
    up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    color: tuple[float, float, float] = (0.72, 0.72, 0.72)  # neutral lit gray
    background: tuple[float, float, float] = (1.0, 1.0, 1.0)


def _to_o3d(mesh: trimesh.Trimesh, color: tuple[float, float, float],
            vertex_colors: np.ndarray | None = None
            ) -> o3d.geometry.TriangleMesh:
    om = o3d.geometry.TriangleMesh()
    om.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    om.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
    om.compute_vertex_normals()
    if vertex_colors is not None:
        vc = np.asarray(vertex_colors, dtype=np.float64)[:, :3]
        om.vertex_colors = o3d.utility.Vector3dVector(np.clip(vc, 0.0, 1.0))
    else:
        om.paint_uniform_color(list(color))
    return om


def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """World->camera extrinsic (4x4) for a camera at ``eye`` facing ``center``.

    Open3D pinhole cameras look along +Z in camera space.
    """
    z = center - eye
    z /= np.linalg.norm(z) + 1e-12
    x = np.cross(up, z)
    nx = np.linalg.norm(x)
    if nx < 1e-8:  # up parallel to view dir -> pick an arbitrary right vector
        x = np.cross(np.array([1.0, 0.0, 0.0]), z)
        nx = np.linalg.norm(x)
    x /= nx
    y = np.cross(z, x)
    extr = np.eye(4)
    extr[:3, :3] = np.stack([x, y, z], axis=0)  # rows = camera axes
    extr[:3, 3] = -extr[:3, :3] @ eye
    return extr


def render_multiview(mesh: trimesh.Trimesh, config: RenderConfig | None = None,
                     vertex_colors: np.ndarray | None = None
                     ) -> np.ndarray:
    """Render ``mesh`` from ``n_views`` orbit cameras -> ``(V, H, W, 3) uint8``.

    ``vertex_colors`` (optional ``(N, 3)`` floats in ``[0, 1]``) shades the mesh
    per vertex (e.g. a displacement heatmap) instead of the flat ``cfg.color``.
    """
    cfg = config or RenderConfig()
    up = np.asarray(cfg.up_axis, dtype=np.float64)
    up /= np.linalg.norm(up) + 1e-12

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    center = verts.mean(axis=0)
    radius = float(np.linalg.norm(verts - center, axis=1).max())
    distance = cfg.distance_scale * max(radius, 1e-6)

    om = _to_o3d(mesh, cfg.color, vertex_colors=vertex_colors)

    vis = o3d.visualization.Visualizer()
    if not vis.create_window(width=cfg.image_size, height=cfg.image_size, visible=False):
        raise RuntimeError("Open3D failed to create an offscreen window")
    try:
        vis.add_geometry(om)
        opt = vis.get_render_option()
        opt.background_color = np.asarray(cfg.background)
        opt.light_on = True
        ctr = vis.get_view_control()

        # In-plane basis perpendicular to ``up`` for placing the eye by azimuth.
        ref = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e0 = np.cross(up, ref); e0 /= np.linalg.norm(e0)
        e1 = np.cross(up, e0)
        elev = np.radians(cfg.elevation_deg)

        views = []
        for i in range(cfg.n_views):
            az = 2.0 * np.pi * i / cfg.n_views
            horiz = np.cos(az) * e0 + np.sin(az) * e1
            direction = np.cos(elev) * horiz + np.sin(elev) * up
            eye = center + distance * direction
            extr = _look_at(eye, center, up)

            cam = ctr.convert_to_pinhole_camera_parameters()
            cam.extrinsic = extr
            ctr.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
            vis.poll_events()
            vis.update_renderer()

            buf = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            img = (np.clip(buf, 0.0, 1.0) * 255.0).astype(np.uint8)
            views.append(img)

            bg = np.asarray(cfg.background)
            fg_frac = float(np.mean(np.any(np.abs(buf - bg) > 1e-3, axis=-1)))
            if fg_frac < 1e-4:
                log.warning("view %d az=%.0fdeg appears empty (fg_frac=%.5f)",
                            i, np.degrees(az), fg_frac)
            log.info("rendered view %d/%d | az=%.0fdeg fg_frac=%.3f",
                     i + 1, cfg.n_views, np.degrees(az), fg_frac)
    finally:
        vis.destroy_window()

    return np.stack(views, axis=0)


def save_views(views: np.ndarray, out_dir: str | Path, stem: str,
               contact_sheet: bool = True) -> list[Path]:
    """Write per-view PNGs (and an optional horizontal contact sheet)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, img in enumerate(views):
        p = out_dir / f"{stem}_view{i}.png"
        o3d.io.write_image(str(p), o3d.geometry.Image(np.ascontiguousarray(img)))
        paths.append(p)
    if contact_sheet and len(views) > 1:
        sheet = np.concatenate(list(views), axis=1)
        sp = out_dir / f"{stem}_contact.png"
        o3d.io.write_image(str(sp), o3d.geometry.Image(np.ascontiguousarray(sheet)))
        paths.append(sp)
    log.info("saved %d image(s) -> %s", len(paths), out_dir)
    return paths
