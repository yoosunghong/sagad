"""CLI: exercise the deformation sandbox with a procedural RGBA mask.

The PPO actor (which will emit the [N, 4] masks) is not built yet, so this
script synthesizes a deterministic, geometry-aware mask to validate the
Phase 2 deformation simulator end-to-end:

    * Fixed (A): 1 near the base (bottom height band) -> pins the asset to the
      ground; tapers to 0 higher up.
    * Bend  (R): grows with height -> organic lean.
    * Noise (G): constant mid-level along-normal jitter.
    * Scale (B): small constant radial swell.

It writes the deformed mesh and asserts the locked-base invariant (vertices
with Fixed==1 do not move), demonstrating the §3 mobility convention.

Usage:
    python scripts/deform_demo.py [<graph.pt or mesh>] [--out deformed.obj]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import build_graph_from_path  # noqa: E402
from data.logging_utils import get_logger  # noqa: E402
from deform import DeformConfig, apply_masks  # noqa: E402

log = get_logger("deform_demo")

DEFAULT_GRAPH = ROOT / "data" / "processed" / "gray-big-rock.pt"


def _load(path: Path):
    if path.suffix.lower() == ".pt":
        return torch.load(path, weights_only=False)
    return build_graph_from_path(path)


def _procedural_masks(data, up=(0.0, 0.0, 1.0), base_band: float = 0.25) -> torch.Tensor:
    """Geometry-aware demo mask: base-locked, height-weighted bend."""
    pos = data.pos
    up_t = torch.tensor(up, dtype=pos.dtype, device=pos.device)
    h = pos @ up_t
    h_norm = (h - h.min()) / (h.max() - h.min()).clamp(min=1e-8)  # [0, 1]

    # Fixed=1 over the solid bottom band (h_norm <= floor), then ramp to 0 by
    # the top of the band -> a genuinely pinned base, not just one vertex.
    floor = base_band * 0.4
    fixed = ((base_band - h_norm) / (base_band - floor)).clamp(0.0, 1.0)
    bend = h_norm                          # lean grows with height
    noise = torch.full_like(h_norm, 0.6)   # uniform jitter
    scale = torch.full_like(h_norm, 0.3)   # mild swell
    return torch.stack([bend, noise, scale, fixed], dim=1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deformation sandbox demo.")
    parser.add_argument("input", nargs="?", default=str(DEFAULT_GRAPH))
    parser.add_argument("--out", default=None, help="output deformed mesh path (.obj/.ply)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _load(Path(args.input)).to(device)
    log.info("loaded %s | N=%d", Path(args.input).name, data.num_nodes)

    masks = _procedural_masks(data).to(device)
    cfg = DeformConfig(noise_seed=args.seed)
    result = apply_masks(data, masks, cfg)

    # -- locked-base invariant: fully-fixed (mobility==0) verts must not move -
    # Ramp vertices (0 < fixed < 1) legitimately move proportionally to their
    # mobility; only fixed==1 implies an exact-zero displacement.
    locked = masks[:, 3] >= 1.0
    if locked.any():
        max_locked_disp = float(result.offset[locked].norm(dim=1).max())
        verdict = "PASS" if max_locked_disp < 1e-6 else "FAIL"
        log.info("locked-base invariant | %d locked verts | max disp=%.3e -> %s",
                 int(locked.sum()), max_locked_disp, verdict)
    else:
        log.warning("no fully-locked vertices in demo mask; skipping invariant check")

    log.info("telemetry: %s", result.telemetry)

    out = Path(args.out) if args.out else (
        ROOT / "data" / "deformed" / f"{Path(args.input).stem}_deformed.obj")
    out.parent.mkdir(parents=True, exist_ok=True)
    result.mesh.export(out)
    log.info("exported deformed mesh -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
