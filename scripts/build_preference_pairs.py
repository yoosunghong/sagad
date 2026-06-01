"""CLI: build the HITL annotation job (candidate pool + render + pairing schedule).

For each asset it samples a pool of geometry-structured deformations, renders a
multi-view contact sheet per candidate, and writes a manifest of candidate specs
plus a within-asset pairing schedule. The manifest is then consumed by the
Gradio annotation UI (``scripts/annotate_ui.py``) and, later, the Bradley-Terry
reward-model trainer.

Usage:
    python scripts/build_preference_pairs.py [--assets ...] [--candidates N]
        [--pairs-per-asset M] [--seed S] [--views V]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.logging_utils import get_logger  # noqa: E402
from hitl import Manifest, build_candidate_pool, build_pairs, save_manifest  # noqa: E402
from hitl.candidates import export_original_mesh  # noqa: E402
from render import RenderConfig  # noqa: E402

log = get_logger("build_preference_pairs")

DEFAULT_ASSETS = ["gray-big-rock", "rock_17", "tree", "tree_1", "tree_2"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the HITL annotation manifest.")
    parser.add_argument("--assets", nargs="*", default=DEFAULT_ASSETS)
    parser.add_argument("--candidates", type=int, default=6,
                        help="candidate deformations sampled per asset")
    parser.add_argument("--pairs-per-asset", type=int, default=None,
                        help="pairs sampled per asset (default: full round-robin)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--render-dir", default=str(ROOT / "data" / "preferences" / "renders"))
    parser.add_argument("--mesh-dir", default=str(ROOT / "data" / "preferences" / "meshes"))
    parser.add_argument("--manifest", default=str(ROOT / "data" / "preferences" / "manifest.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    render_cfg = RenderConfig(n_views=args.views)
    graphs_dir = ROOT / "data" / "processed"
    render_dir = Path(args.render_dir)
    mesh_dir = Path(args.mesh_dir)
    log.info("device=%s | assets=%s | %d candidates/asset | seed=%d",
             device, args.assets, args.candidates, args.seed)

    specs = []
    originals = {}
    for asset in args.assets:
        specs.extend(build_candidate_pool(
            asset, graphs_dir, render_dir, args.candidates, rng, device,
            mesh_dir=mesh_dir, render_cfg=render_cfg, root=ROOT,
        ))
        originals[asset] = export_original_mesh(asset, graphs_dir, mesh_dir, ROOT, device)

    pairs = build_pairs(specs, rng, n_pairs_per_asset=args.pairs_per_asset)
    manifest = Manifest(render_dir=str(render_dir.relative_to(ROOT)),
                        seed=args.seed, candidates=specs, pairs=pairs, originals=originals)
    save_manifest(manifest, Path(args.manifest))

    log.info("=== ANNOTATION JOB READY ===")
    log.info("  %d candidates across %d assets | %d comparison pairs",
             len(specs), len(args.assets), len(pairs))
    log.info("  launch the UI with: python scripts/annotate_ui.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
