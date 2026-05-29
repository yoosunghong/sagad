"""CLI: validate MeshMAE latent-space representations of organic assets.

Closes the Phase 1 checklist item "Validate latent space representations of
initial organic assets (Trees, Rocks)". Loads pre-trained MeshMAE weights,
encodes each cached asset graph into per-node latents ``Z in R^{N x D}``,
reduces each asset to a graph-level descriptor (mean + std pooling of node
latents), and checks that the embedding separates the two organic classes:
intra-class cosine distance should be smaller than inter-class distance.

Asset class is inferred from the file name (``rock`` vs ``tree``); override
with ``--labels`` (one per input, comma-separated) when names are ambiguous.

Usage:
    python scripts/validate_latents.py <graph.pt> [<graph.pt> ...] \
        --weights data/processed/meshmae_baseline.pt
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import torch
import torch.nn.functional as F

# Make ``src`` importable when run as a plain script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.logging_utils import get_logger  # noqa: E402
from models import MeshMAE, MeshMAEConfig  # noqa: E402

log = get_logger("validate_latents")


def _infer_label(path: Path) -> str:
    name = path.stem.lower()
    if "tree" in name:
        return "tree"
    if "rock" in name:
        return "rock"
    return "unknown"


def _graph_descriptor(z: torch.Tensor) -> torch.Tensor:
    """Permutation-invariant graph-level descriptor from node latents.

    Mean + std pooling captures both the average structural identity and the
    spread of per-vertex embeddings, independent of vertex count/order.
    """
    return torch.cat([z.mean(dim=0), z.std(dim=0)], dim=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate MeshMAE latent space.")
    parser.add_argument("inputs", nargs="+", help="cached PyG graph .pt files")
    parser.add_argument("--weights", required=True, help="trained MeshMAE .pt")
    parser.add_argument("--labels", default=None,
                        help="comma-separated class labels (override name inference)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    ckpt = torch.load(args.weights, weights_only=False)
    cfg = MeshMAEConfig(**ckpt["config"])
    model = MeshMAE(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    log.info("loaded MeshMAE weights %s (latent_dim=%d)", Path(args.weights).name, cfg.latent_dim)

    labels = (args.labels.split(",") if args.labels
              else [_infer_label(Path(p)) for p in args.inputs])
    if len(labels) != len(args.inputs):
        raise ValueError("number of --labels must match number of inputs")

    # Encode each asset -> graph-level descriptor.
    names, descriptors = [], []
    with torch.no_grad():
        for path, label in zip(args.inputs, labels):
            g = torch.load(path, weights_only=False).to(device)
            z = model.encode(g)
            d = _graph_descriptor(z)
            if not torch.isfinite(d).all():
                raise ValueError(f"non-finite descriptor for {path}")
            names.append((Path(path).stem, label))
            descriptors.append(d)
            log.info("encoded %-16s class=%-5s | Z=%s desc_norm=%.4f",
                     Path(path).stem, label, tuple(z.shape), float(d.norm()))

    desc = F.normalize(torch.stack(descriptors), dim=1)  # (A, 2D) unit vectors

    # Pairwise cosine distances; split into intra- vs inter-class.
    intra, inter = [], []
    log.info("--- pairwise cosine distances ---")
    for (i, j) in combinations(range(len(names)), 2):
        dist = 1.0 - float(torch.dot(desc[i], desc[j]))
        same = names[i][1] == names[j][1]
        (intra if same else inter).append(dist)
        log.info("  %-16s <-> %-16s | %s | dist=%.4f",
                 names[i][0], names[j][0], "intra" if same else "inter", dist)

    if intra and inter:
        mi, me = sum(intra) / len(intra), sum(inter) / len(inter)
        margin = me - mi
        verdict = "PASS" if margin > 0 else "FAIL"
        log.info("--- separability ---")
        log.info("mean intra-class dist=%.4f | mean inter-class dist=%.4f | margin=%+.4f -> %s",
                 mi, me, margin, verdict)
        if verdict == "FAIL":
            log.warning("latent space does not separate classes (margin <= 0); "
                        "consider more epochs / assets per class.")
    else:
        log.warning("need >=2 classes with >=2 members each to score separability "
                    "(intra=%d inter=%d pairs).", len(intra), len(inter))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
