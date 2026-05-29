"""CLI: validate the deterministic geometry & physics penalties.

Exercises `src/rewards/penalties.py` against three controlled deformations of a
cached asset graph and checks the expected invariants:

  1. **Identity** (no deformation) -> every penalty ~ 0.
  2. **Rigid lift** (translate whole mesh up) -> distortion (Laplacian, normal)
     ~ 0 (rigid-motion invariant), but ground gap > 0 (base lost contact).
  3. **Procedural deform** (the Phase 2 demo mask) -> all penalties > 0, and
     the ground base stays pinned (gap ~ 0) because the Fixed channel locks it.

Usage:
    python scripts/eval_penalties.py [<graph.pt>]
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.logging_utils import get_logger  # noqa: E402
from deform import DeformConfig, apply_masks  # noqa: E402
from rewards import evaluate_penalties  # noqa: E402

# Reuse the exact demo mask used to validate the sandbox.
sys.path.insert(0, str(ROOT / "scripts"))
from deform_demo import _procedural_masks  # noqa: E402

log = get_logger("eval_penalties")

DEFAULT_GRAPH = ROOT / "data" / "processed" / "gray-big-rock.pt"


def _check(name: str, cond: bool) -> None:
    log.info("  invariant [%s] -> %s", name, "PASS" if cond else "FAIL")


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GRAPH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(path, weights_only=False).to(device)
    log.info("loaded %s | N=%d", path.name, data.num_nodes)

    # Precompute face adjacency once; reuse across cases (PPO-loop pattern).
    from rewards.penalties import _face_adjacency
    adj = _face_adjacency(data.face.t().contiguous())

    # 1. Identity -----------------------------------------------------------
    log.info("--- case 1: identity (no deformation) ---")
    t = evaluate_penalties(data, data.pos.clone(), face_adjacency=adj)
    _check("identity all ~0",
           max(t["distortion_total"], t["physics_total"]) < 1e-5)

    # 2. Rigid lift ---------------------------------------------------------
    log.info("--- case 2: rigid lift (+0.2 along up) ---")
    lifted = data.pos.clone()
    lifted[:, 2] += 0.2
    t = evaluate_penalties(data, lifted, face_adjacency=adj)
    _check("rigid lift: distortion ~0", t["distortion_total"] < 1e-4)
    _check("rigid lift: ground gap > 0", t["ground_gap"] > 1e-3)

    # 3. Procedural deform --------------------------------------------------
    log.info("--- case 3: procedural demo deformation ---")
    masks = _procedural_masks(data).to(device)
    result = apply_masks(data, masks, DeformConfig())
    t = evaluate_penalties(data, result.pos, face_adjacency=adj)
    _check("deform: distortion > 0", t["distortion_total"] > 1e-3)
    _check("deform: base stays pinned (gap ~0)", t["ground_gap"] < 1e-4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
