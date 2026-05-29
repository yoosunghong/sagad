"""CLI: build (and optionally cache) a PyG graph from a mesh file.

Usage:
    python scripts/build_graph.py <mesh_path> [--out <cache.pt>]

Defaults to the baseline gray-big-rock OBJ if no path is given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make ``src`` importable when run as a plain script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import build_graph_from_path  # noqa: E402

DEFAULT_MESH = ROOT / "meshes" / "baseline" / "rock" / "source" / "untitled" / "untitled.obj"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a PyG graph from a mesh.")
    parser.add_argument("mesh", nargs="?", default=str(DEFAULT_MESH))
    parser.add_argument("--out", default=None, help="optional .pt cache path")
    args = parser.parse_args()

    data = build_graph_from_path(args.mesh)
    print(data)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, out)
        print(f"cached graph -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
