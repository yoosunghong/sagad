"""Per-channel deformation operators (vectorized torch).

Each operator returns a per-vertex displacement ``offset in R^{N x 3}`` already
scaled by its mask channel; the sandbox sums them into the composite WPO
accumulator (see docs/ARCHITECT.md sec. 2.3). All ops are pure functions of
tensors -- no per-vertex Python loops, no dense ``N x N`` adjacency
(CLAUDE.md optimization rule).

Channel map (docs/ARCHITECT.md sec. 3):
    R = bend, G = noise, B = scale, A = mobility (= 1 - fixed).
"""

from __future__ import annotations

import torch


def _unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize a 1-D vector to unit length."""
    return v / (v.norm() + eps)


def bend_offset(
    pos: torch.Tensor,
    mask: torch.Tensor,
    *,
    axis: torch.Tensor,
    up: torch.Tensor,
    strength: float,
    pivot: torch.Tensor | None = None,
) -> torch.Tensor:
    """Height-weighted Rodrigues bend about ``axis`` (organic lean profile).

    The rotation angle of each vertex scales with its normalized height above
    the base ``pivot`` and with ``mask`` (R channel), so the lean increases
    toward the top while the base stays put. Returns ``rotated - pos``.
    """
    axis = _unit(axis.to(pos.dtype))
    up = _unit(up.to(pos.dtype))
    if pivot is None:
        # Base of the mesh along the up axis -> bend hinges at the ground.
        heights_all = pos @ up
        pivot = pos[torch.argmin(heights_all)]

    v = pos - pivot                       # (N, 3) vectors from the hinge
    h = v @ up                            # (N,) signed height above pivot
    h_max = float(h.max().clamp(min=1e-8))
    h_norm = (h / h_max).clamp(0.0, 1.0)  # only bend the portion above pivot

    theta = strength * mask * h_norm      # (N,) per-vertex bend angle
    cos_t = torch.cos(theta).unsqueeze(1)
    sin_t = torch.sin(theta).unsqueeze(1)

    # Rodrigues rotation of v about the unit axis k by per-vertex theta.
    k = axis.view(1, 3)
    cross = torch.cross(k.expand_as(v), v, dim=1)
    dot = (v * k).sum(dim=1, keepdim=True)
    v_rot = v * cos_t + cross * sin_t + k * dot * (1.0 - cos_t)
    return v_rot - v


def value_noise_3d(coords: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Deterministic 3D value noise in [-1, 1] (trilinear lattice-hash interp).

    A lightweight, seeded stand-in for the UE5 Perlin node: continuous,
    repeatable, and fully vectorized. ``coords`` is ``(N, 3)`` (typically the
    vertex positions pre-multiplied by a spatial frequency).
    """
    p0 = torch.floor(coords).to(torch.int64)        # lattice cell corner
    f = coords - p0.to(coords.dtype)                 # fractional part in [0,1)
    fade = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)  # quintic smoothstep

    # Large odd primes for an integer spatial hash; int64 wraps on overflow.
    cx, cy, cz = 73856093, 19349663, 83492791
    s = int(seed) * 2654435761

    def corner(ox: int, oy: int, oz: int) -> torch.Tensor:
        ix = p0[:, 0] + ox
        iy = p0[:, 1] + oy
        iz = p0[:, 2] + oz
        h = (ix * cx) ^ (iy * cy) ^ (iz * cz) ^ s
        h = h & 0x7FFFFFFF
        return h.to(coords.dtype) / float(0x7FFFFFFF) * 2.0 - 1.0  # [-1, 1]

    c000, c100 = corner(0, 0, 0), corner(1, 0, 0)
    c010, c110 = corner(0, 1, 0), corner(1, 1, 0)
    c001, c101 = corner(0, 0, 1), corner(1, 0, 1)
    c011, c111 = corner(0, 1, 1), corner(1, 1, 1)

    fx, fy, fz = fade[:, 0], fade[:, 1], fade[:, 2]
    x00 = torch.lerp(c000, c100, fx)
    x10 = torch.lerp(c010, c110, fx)
    x01 = torch.lerp(c001, c101, fx)
    x11 = torch.lerp(c011, c111, fx)
    y0 = torch.lerp(x00, x10, fy)
    y1 = torch.lerp(x01, x11, fy)
    return torch.lerp(y0, y1, fz)


def noise_offset(
    pos: torch.Tensor,
    normal: torch.Tensor,
    mask: torch.Tensor,
    *,
    strength: float,
    frequency: float,
    seed: int = 0,
) -> torch.Tensor:
    """Displace each vertex along its normal by seeded 3D value noise (G)."""
    eta = value_noise_3d(pos * frequency, seed=seed)        # (N,) in [-1, 1]
    return (strength * mask * eta).unsqueeze(1) * normal


def scale_offset(
    pos: torch.Tensor,
    mask: torch.Tensor,
    *,
    strength: float,
    center: torch.Tensor | None = None,
) -> torch.Tensor:
    """Radial swell/shrink from the mesh centroid (B)."""
    if center is None:
        center = pos.mean(dim=0)
    return (strength * mask).unsqueeze(1) * (pos - center)
