"""CLI: self-supervised MeshMAE pre-training + latent-space validation.

Runs the masked-reconstruction objective on one (or more) cached/built mesh
graphs, then reports the learned per-node embedding ``Z in R^{N x D}`` and a
few sanity diagnostics (latent statistics, masked-reconstruction error drop).

Usage:
    python scripts/train_meshmae.py [<mesh_or_graph_path> ...] \
        [--epochs N] [--lr LR] [--latent-dim D] [--out <weights.pt>]

With no path, defaults to the baseline gray-big-rock OBJ. ``.pt`` inputs are
loaded as pre-built PyG graphs; mesh files are run through the data pipeline.
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
from data.logging_utils import get_logger  # noqa: E402
from models import MeshMAE, MeshMAEConfig  # noqa: E402

log = get_logger("train_meshmae")

DEFAULT_MESH = ROOT / "meshes" / "baseline" / "rock" / "source" / "untitled" / "untitled.obj"
GRAPH_SUFFIXES = {".pt"}


def _load_graph(path: Path):
    """Load a cached PyG graph (.pt) or build one from a mesh file."""
    if path.suffix.lower() in GRAPH_SUFFIXES:
        log.info("loading cached graph %s", path.name)
        return torch.load(path, weights_only=False)
    return build_graph_from_path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-train MeshMAE on mesh graphs.")
    parser.add_argument("inputs", nargs="*", default=[str(DEFAULT_MESH)])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--out", default=None, help="optional .pt weights path")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    graphs = [_load_graph(Path(p)).to(device) for p in args.inputs]
    log.info("loaded %d graph(s) | total nodes=%d",
             len(graphs), sum(g.num_nodes for g in graphs))

    cfg = MeshMAEConfig(latent_dim=args.latent_dim, mask_ratio=args.mask_ratio)
    model = MeshMAE(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    log.info("MeshMAE params=%d latent_dim=%d mask_ratio=%.2f",
             sum(p.numel() for p in model.parameters()), cfg.latent_dim, cfg.mask_ratio)

    # -- self-supervised training loop --------------------------------------
    model.train()
    first_loss = None
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for g in graphs:
            opt.zero_grad()
            out = model(g)
            out.loss.backward()
            opt.step()
            epoch_loss += float(out.loss)
        epoch_loss /= len(graphs)
        if first_loss is None:
            first_loss = epoch_loss
        if epoch % args.log_every == 0 or epoch == args.epochs:
            log.info("epoch %4d/%d | mean sce_loss=%.6f", epoch, args.epochs, epoch_loss)

    log.info("training done | sce_loss %.6f -> %.6f (%.1f%% reduction)",
             first_loss, epoch_loss, 100.0 * (1.0 - epoch_loss / max(first_loss, 1e-12)))

    # -- latent-space validation -------------------------------------------
    model.eval()
    with torch.no_grad():
        for g, p in zip(graphs, args.inputs):
            z = model.encode(g)
            log.info(
                "latent %s | Z shape=%s mean=%.4f std=%.4f min=%.4f max=%.4f finite=%s",
                Path(p).name, tuple(z.shape), float(z.mean()), float(z.std()),
                float(z.min()), float(z.max()), bool(torch.isfinite(z).all()),
            )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__}, out)
        log.info("saved MeshMAE weights -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
