"""CLI: render multi-view snapshots of an asset (original vs. deformed).

Validates the Phase 2 rendering pipeline end-to-end: loads a cached graph,
renders the undeformed mesh from N orbit views, applies the procedural demo
deformation, renders that too, and writes per-view PNGs + contact sheets to
``data/renders/``.

Usage:
    python scripts/render_demo.py [<graph.pt>] [--views N] [--size PX]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data.logging_utils import get_logger  # noqa: E402
from deform import DeformConfig, apply_masks  # noqa: E402
from render import RenderConfig, render_multiview, save_views  # noqa: E402
from deform_demo import _procedural_masks  # noqa: E402

log = get_logger("render_demo")

DEFAULT_GRAPH = ROOT / "data" / "processed" / "gray-big-rock.pt"


def _graph_to_mesh(data) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=data.pos.detach().cpu().numpy(),
        faces=data.face.t().detach().cpu().numpy(),
        process=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Render multi-view asset snapshots.")
    parser.add_argument("input", nargs="?", default=str(DEFAULT_GRAPH))
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(Path(args.input), weights_only=False).to(device)
    stem = Path(args.input).stem
    log.info("loaded %s | N=%d", stem, data.num_nodes)

    cfg = RenderConfig(n_views=args.views, image_size=args.size)
    out_dir = ROOT / "data" / "renders"

    # Original.
    log.info("--- rendering original ---")
    views_o = render_multiview(_graph_to_mesh(data), cfg)
    save_views(views_o, out_dir, f"{stem}_orig")

    # Deformed (procedural demo mask).
    log.info("--- rendering deformed ---")
    result = apply_masks(data, _procedural_masks(data).to(device), DeformConfig())
    views_d = render_multiview(result.mesh, cfg)
    save_views(views_d, out_dir, f"{stem}_deformed")

    # Sanity: deformed views should differ from originals.
    diff = float(np.mean(np.abs(views_o.astype(np.int16) - views_d.astype(np.int16))))
    log.info("mean |orig - deformed| pixel diff = %.2f -> %s",
             diff, "PASS" if diff > 1.0 else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
