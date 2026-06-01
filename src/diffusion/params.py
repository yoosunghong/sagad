"""Per-instance variant parameter schema (the diffusion sampler's latent).

A *variant* is a low-dim vector of per-instance knobs (bend/noise/scale gains,
base-lock band, floor count, ...) that the UE5 WPO shader reads as **Per-Instance
Custom Data**. The diffusion model (`ddpm.py`) learns this vector's *distribution*;
this module only defines the schema and the bijection between raw parameter space
and the standardized [-1, 1] space the denoiser operates in.

It performs NO ML and NO deformation -- it is the typed contract between the
sampler and the §3 bake (docs/ARCHITECT.md sec. 2.5). The channel map stays the
locked ``[R=bend, G=noise, B=scale, A=fixed]`` convention; no virtual channels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ParamField:
    """One named per-instance knob with inclusive ``[lo, hi]`` bounds.

    ``integer=True`` fields (e.g. ``floor_count``) are rounded on decode so the
    sampled value is a valid discrete choice the WPO mask logic can switch on.
    """

    name: str
    lo: float
    hi: float
    integer: bool = False

    def __post_init__(self) -> None:
        if not self.hi > self.lo:
            raise ValueError(f"field '{self.name}': require hi > lo, got [{self.lo}, {self.hi}]")


class VariantParamSpec:
    """Ordered set of :class:`ParamField`s + the raw<->model-space bijection.

    * raw space   -- the meaningful units written to Per-Instance Custom Data and
      decoded into §2.3 mask gains; each field lives in its own ``[lo, hi]``.
    * model space -- standardized ``[-1, 1]`` per dim, what the denoiser sees
      (DDPM assumes roughly unit-scale, zero-centred data).
    """

    def __init__(self, fields: list[ParamField]):
        if not fields:
            raise ValueError("VariantParamSpec needs at least one field")
        names = [f.name for f in fields]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate field names: {names}")
        self.fields = list(fields)
        self._lo = torch.tensor([f.lo for f in fields], dtype=torch.float32)
        self._hi = torch.tensor([f.hi for f in fields], dtype=torch.float32)
        self._int = torch.tensor([f.integer for f in fields], dtype=torch.bool)

    @property
    def dim(self) -> int:
        return len(self.fields)

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.fields]

    def sample_uniform(self, n: int,
                       generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw ``n`` raw param vectors uniformly within the box (integers rounded).

        The proposal distribution for the rejection bootstrap: uniform draws are
        deformed + scored by the §2.4 penalties, and the feasible subset becomes
        the diffusion training set.
        """
        u = torch.rand(n, self.dim, generator=generator)
        raw = self._lo + u * (self._hi - self._lo)
        if self._int.any():
            raw = torch.where(self._int, raw.round(), raw)
        return raw

    def to_model(self, raw: torch.Tensor) -> torch.Tensor:
        """Map raw params ``(..., D)`` in ``[lo, hi]`` to model space ``[-1, 1]``."""
        lo, hi = self._lo.to(raw), self._hi.to(raw)
        return 2.0 * (raw - lo) / (hi - lo) - 1.0

    def from_model(self, z: torch.Tensor) -> torch.Tensor:
        """Map model-space ``(..., D)`` back to raw params; clamp + round integers.

        Clamps to the valid box first (the denoiser can sample slightly outside
        ``[-1, 1]``), then rounds the ``integer`` dims to the nearest valid step.
        """
        lo, hi = self._lo.to(z), self._hi.to(z)
        t = ((z + 1.0) * 0.5).clamp(0.0, 1.0)
        raw = lo + t * (hi - lo)
        if self._int.any():
            mask = self._int.to(z.device)
            raw = torch.where(mask, raw.round(), raw)
        return raw


def organic_spec() -> VariantParamSpec:
    """Default schema for organic assets (trees/rocks): continuous deformation."""
    return VariantParamSpec([
        ParamField("bend_gain", 0.0, 1.0),
        ParamField("noise_gain", 0.0, 1.0),
        ParamField("scale_gain", 0.0, 1.0),
        ParamField("base_lock", 0.05, 0.4),   # fraction of height pinned (A/fixed)
    ])


def building_spec(min_floors: int = 2, max_floors: int = 10) -> VariantParamSpec:
    """Default schema for WPO floor-masked buildings: discrete height + micro-variation."""
    return VariantParamSpec([
        ParamField("floor_count", float(min_floors), float(max_floors), integer=True),
        ParamField("bend_gain", 0.0, 0.3),    # gentle sway only on a building
        ParamField("noise_gain", 0.0, 0.2),
        ParamField("base_lock", 0.02, 0.15),
    ])
