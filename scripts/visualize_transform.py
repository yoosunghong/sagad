"""Visualize HOW and HOW MUCH a sampled variant deforms the mesh (ARCHITECT 2.5).

Loads the trained variant-diffusion sampler, draws a representative variant,
deforms the asset, and renders an orbit montage:

    row 1 -- original mesh (4 views)
    row 2 -- deformed mesh, shaded by per-vertex displacement magnitude (heatmap)

plus a stats panel (displacement + per-edge stretch distributions) so the
*extent* of the transform is quantified, and an interactive heatmap GLB.

    python scripts/visualize_transform.py --asset gray-big-rock
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

log = get_logger("visualize_transform")
_CMAP = matplotlib.colormaps["turbo"]


def load_sampler(ckpt_path: Path) -> tuple[GaussianDiffusion, VariantParamSpec]:
    ck = torch.load(ckpt_path, weights_only=False)
    spec = VariantParamSpec([ParamField(n, lo, hi, bool(it)) for n, lo, hi, it in ck["spec_fields"]])
    diff = GaussianDiffusion(DiffusionMLP(dim=ck["dim"], hidden=128),
                             dim=ck["dim"], timesteps=ck["timesteps"])
    diff.load_state_dict(ck["model_state"])
    diff.eval()
    return diff, spec


def pick_representative(diff, spec, data, cfg, fa, k: int = 9) -> torch.Tensor:
    """Sample k variants; return the one with median distortion (a typical one)."""
    raw = spec.from_model(diff.sample(k))
    dist = []
    for i in range(k):
        masks = mask_from_gains(data, dict(zip(spec.names, raw[i].tolist())))
        pen = evaluate_penalties(data, apply_masks(data, masks, cfg).pos, face_adjacency=fa)
        dist.append(pen["distortion_total"])
    return raw[int(np.argsort(dist)[k // 2])]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset", default="gray-big-rock")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--views", type=int, default=4)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    data = torch.load(ROOT / "data" / "processed" / f"{args.asset}.pt", weights_only=False)
    cfg = DeformConfig()
    fa = _face_adjacency(data.face.t().contiguous())

    diff, spec = load_sampler(ROOT / "data" / "diffusion" / f"variant_diffusion_{args.asset}.pt")
    variant = pick_representative(diff, spec, data, cfg, fa)
    gains = dict(zip(spec.names, variant.tolist()))
    log.info("representative variant: %s", {k: round(v, 3) for k, v in gains.items()})

    # -- deform + measure --------------------------------------------------------
    masks = mask_from_gains(data, gains)
    result = apply_masks(data, masks, cfg)
    pen = evaluate_penalties(data, result.pos, face_adjacency=fa)

    pos0 = data.pos
    disp = result.offset.norm(dim=1).cpu().numpy()        # per-vertex |displacement|
    dmax = float(disp.max()) or 1e-6
    ei = data.edge_index
    l0 = (pos0[ei[0]] - pos0[ei[1]]).norm(dim=1)
    l1 = (result.pos[ei[0]] - result.pos[ei[1]]).norm(dim=1)
    stretch = ((l1 - l0).abs() / l0.clamp(min=1e-8)).cpu().numpy()

    orig_mesh = trimesh.Trimesh(vertices=pos0.cpu().numpy(),
                                faces=data.face.t().cpu().numpy(), process=False)
    heat_colors = _CMAP(disp / dmax)[:, :3]

    rc = RenderConfig(n_views=args.views, image_size=320)
    orig_views = render_multiview(orig_mesh, rc)
    heat_views = render_multiview(result.mesh, rc, vertex_colors=heat_colors)

    out_dir = ROOT / "data" / "diffusion" / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- montage figure ----------------------------------------------------------
    nv = args.views
    fig = plt.figure(figsize=(3.2 * nv + 1.0, 6.6))
    gs = fig.add_gridspec(2, nv + 1, width_ratios=[1] * nv + [0.07], hspace=0.05, wspace=0.05)
    for j in range(nv):
        ax = fig.add_subplot(gs[0, j]); ax.imshow(orig_views[j]); ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("Original", fontsize=12, fontweight="bold")
        a2 = fig.add_subplot(gs[1, j]); a2.imshow(heat_views[j]); a2.set_xticks([]); a2.set_yticks([])
        if j == 0:
            a2.set_ylabel("Deformed\n(displacement)", fontsize=12, fontweight="bold")
    cax = fig.add_subplot(gs[1, nv])
    fig.colorbar(cm.ScalarMappable(norm=Normalize(0, dmax), cmap=_CMAP), cax=cax,
                 label="|displacement| (unit-sphere)")
    fig.suptitle(
        f"{args.asset}  |  variant {{{', '.join(f'{k}={v:.2f}' for k, v in gains.items())}}}\n"
        f"max disp={disp.max():.3f}  mean disp={disp.mean():.3f}  "
        f"p99 edge-stretch={np.quantile(stretch, 0.99)*100:.1f}%  "
        f"distortion={pen['distortion_total']:.3f}  locked-base={result.telemetry['locked_fraction']*100:.0f}%",
        fontsize=11)
    montage = out_dir / f"transform_{args.asset}.png"
    fig.savefig(montage, dpi=110, bbox_inches="tight")
    plt.close(fig)

    # -- stats figure ------------------------------------------------------------
    fig2, (axd, axs) = plt.subplots(1, 2, figsize=(11, 3.6))
    axd.hist(disp, bins=50, color="#3b7dd8"); axd.axvline(disp.mean(), color="k", ls="--", label=f"mean {disp.mean():.3f}")
    axd.set_title("Per-vertex displacement"); axd.set_xlabel("|displacement|"); axd.set_ylabel("vertices"); axd.legend()
    axs.hist(stretch * 100, bins=50, color="#d8743b"); axs.axvline(np.quantile(stretch, 0.99) * 100, color="k", ls="--", label=f"p99 {np.quantile(stretch,0.99)*100:.1f}%")
    axs.set_title("Per-edge stretch (texture-smear proxy)"); axs.set_xlabel("|Δlength|/length  (%)"); axs.set_ylabel("edges"); axs.legend()
    fig2.tight_layout()
    stats_png = out_dir / f"transform_{args.asset}_stats.png"
    fig2.savefig(stats_png, dpi=110); plt.close(fig2)

    # -- interactive heatmap GLB -------------------------------------------------
    glb_mesh = result.mesh.copy()
    glb_mesh.visual = trimesh.visual.ColorVisuals(
        mesh=glb_mesh, vertex_colors=(heat_colors * 255).astype(np.uint8))
    glb_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    glb = out_dir / f"transform_{args.asset}_heatmap.glb"
    glb_mesh.export(glb)

    log.info("saved montage=%s stats=%s glb=%s", montage.name, stats_png.name, glb.name)
    print(f"\nMONTAGE: {montage}\nSTATS:   {stats_png}\nGLB:     {glb}")


if __name__ == "__main__":
    main()
