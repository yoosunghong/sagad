"""Grid of distinct sampled variants from ONE mesh (ARCHITECT sec. 2.5).

Shows the *diversity* the variant-diffusion sampler produces: draw K variants,
deform the single base asset by each, and render a grid -- top row shaded by
per-vertex displacement (so you see how each one differs), bottom-labeled with
its sampled params + extent stats. This is the "many instances from one file"
payoff made visible.

    python scripts/visualize_variant_grid.py --asset gray-big-rock --grid 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from matplotlib import cm
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion import DiffusionMLP, GaussianDiffusion, ParamField, VariantParamSpec  # noqa: E402
from deform import DeformConfig, apply_masks  # noqa: E402
from rewards import evaluate_penalties  # noqa: E402
from rewards.penalties import _face_adjacency  # noqa: E402
from hitl.candidates import mask_from_gains  # noqa: E402
from render import RenderConfig, render_multiview  # noqa: E402
from data.logging_utils import get_logger  # noqa: E402

log = get_logger("visualize_variant_grid")
_CMAP = matplotlib.colormaps["turbo"]


def load_sampler(ckpt_path: Path) -> tuple[GaussianDiffusion, VariantParamSpec]:
    ck = torch.load(ckpt_path, weights_only=False)
    spec = VariantParamSpec([ParamField(n, lo, hi, bool(it)) for n, lo, hi, it in ck["spec_fields"]])
    diff = GaussianDiffusion(DiffusionMLP(dim=ck["dim"], hidden=128),
                             dim=ck["dim"], timesteps=ck["timesteps"])
    diff.load_state_dict(ck["model_state"])
    diff.eval()
    return diff, spec


def render_variant(data, gains, cfg, fa, rc, dmax_ref=None):
    """Deform by `gains`, render one displacement-heatmap view, return img + stats."""
    masks = mask_from_gains(data, gains)
    result = apply_masks(data, masks, cfg)
    pen = evaluate_penalties(data, result.pos, face_adjacency=fa)
    disp = result.offset.norm(dim=1).cpu().numpy()
    dmax = dmax_ref if dmax_ref is not None else (float(disp.max()) or 1e-6)
    colors = _CMAP(np.clip(disp / dmax, 0, 1))[:, :3]
    views = render_multiview(result.mesh, rc, vertex_colors=colors)
    img = views[1]  # side (lean-visible) orbit angle
    ei = data.edge_index
    l0 = (data.pos[ei[0]] - data.pos[ei[1]]).norm(dim=1)
    l1 = (result.pos[ei[0]] - result.pos[ei[1]]).norm(dim=1)
    p99 = float(((l1 - l0).abs() / l0.clamp(min=1e-8)).quantile(0.99)) * 100
    return img, {"max": float(disp.max()), "p99": p99, "dist": pen["distortion_total"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset", default="gray-big-rock")
    p.add_argument("--grid", type=int, default=6, help="number of variants to show")
    p.add_argument("--seed", type=int, default=2)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    data = torch.load(ROOT / "data" / "processed" / f"{args.asset}.pt", weights_only=False)
    cfg = DeformConfig()
    fa = _face_adjacency(data.face.t().contiguous())
    diff, spec = load_sampler(ROOT / "data" / "diffusion" / f"variant_diffusion_{args.asset}.pt")

    raw = spec.from_model(diff.sample(args.grid))
    variants = [dict(zip(spec.names, raw[i].tolist())) for i in range(args.grid)]

    # shared colour scale across the grid so panels are comparable
    rc = RenderConfig(n_views=4, image_size=300)  # n_views=4 -> view[1] is the side (lean-visible) angle
    rc_side = RenderConfig(n_views=4, image_size=300, elevation_deg=15.0)

    # first pass: global dmax for a shared colorbar
    results = []
    dmax_global = 1e-6
    for g in variants:
        masks = mask_from_gains(data, g)
        off = apply_masks(data, masks, cfg).offset.norm(dim=1).max().item()
        dmax_global = max(dmax_global, off)

    imgs, stats = [], []
    for g in variants:
        img, st = render_variant(data, g, cfg, fa, rc_side, dmax_ref=dmax_global)
        imgs.append(img)
        stats.append(st)

    # also render the undeformed reference
    orig_mesh = trimesh.Trimesh(vertices=data.pos.cpu().numpy(),
                                faces=data.face.t().cpu().numpy(), process=False)
    orig_img = render_multiview(orig_mesh, rc_side)[1]

    # -- compose grid ------------------------------------------------------------
    cols = min(args.grid + 1, 4)
    rows = int(np.ceil((args.grid + 1) / cols))
    fig = plt.figure(figsize=(3.4 * cols + 0.6, 3.6 * rows))
    gs = fig.add_gridspec(rows, cols + 1, width_ratios=[1] * cols + [0.06], wspace=0.06, hspace=0.18)

    ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(orig_img)
    ax0.set_xticks([]); ax0.set_yticks([]); ax0.set_title("ORIGINAL\n(one mesh file)", fontsize=10, fontweight="bold")

    for k in range(args.grid):
        idx = k + 1
        ax = fig.add_subplot(gs[idx // cols, idx % cols])
        ax.imshow(imgs[k]); ax.set_xticks([]); ax.set_yticks([])
        g, st = variants[k], stats[k]
        ax.set_title(f"variant {k+1}", fontsize=10, fontweight="bold")
        ax.set_xlabel(
            f"bend {g['bend']:.2f} · noise {g['noise']:.2f} · scale {g['scale']:.2f}\n"
            f"max disp {st['max']:.2f} · p99 stretch {st['p99']:.0f}% · dist {st['dist']:.2f}",
            fontsize=8)

    cax = fig.add_subplot(gs[:, cols])
    fig.colorbar(cm.ScalarMappable(norm=Normalize(0, dmax_global), cmap=_CMAP), cax=cax,
                 label="|displacement| (shared scale)")
    fig.suptitle(f"{args.asset}: {args.grid} diffusion-sampled variants from one mesh "
                 f"(shaded by displacement)", fontsize=13, fontweight="bold")

    out_dir = ROOT / "data" / "diffusion" / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"variant_grid_{args.asset}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    log.info("saved %s", out.name)
    print(f"\nGRID: {out}")


if __name__ == "__main__":
    main()
