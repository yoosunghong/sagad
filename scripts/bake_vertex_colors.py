"""Bake the shared deformation mask into Static Mesh Vertex Colors (Phase 5, task 1).

Produces the **single, shared** RGBA vertex-color buffer that every WPO instance of
an asset reads (per-instance variation travels separately, through Per-Instance
Custom Data -- see ``scripts/sample_variants.py``). The mask is written onto the
*undeformed* base mesh; the UE5 material applies the deformation at runtime via World
Position Offset (docs/ARCHITECT.md sec. 3).

Two correctness invariants from the architecture doc are enforced here:

* **Bake convention (sec. 3):** Alpha stores **mobility** = ``1 - fixed`` so that
  ``fixed=1 -> Alpha=0 -> locked vertex``. Channels: ``R=bend, G=noise, B=scale,
  A=mobility``.
* **Seam coherence (sec. 2.4):** UV-seam vertices are *duplicated* in a UE Static
  Mesh; coincident duplicates MUST carry identical RGBA or WPO moves the two halves
  apart and tears the seam. We **merge-then-average**: vertices sharing a position
  (within a tolerance) get the mean RGBA of their group written back to all members.

Mask source:
* ``--source policy`` -- the trained PPO actor's deterministic mask (MeshMAE latent
  -> ``MaskActorCritic.act(deterministic=True)``), the reward-optimal shared mask.
* ``--source gains``  -- a deterministic geometry-structured mask from explicit
  per-channel gains (``mask_from_gains``); no trained policy needed.

Output (``data/bake/<asset>/``):
* ``<asset>_baked.glb``     -- base mesh + COLOR_0 vertex colors (import into UE5).
* ``<asset>_preview.glb``   -- the deformed mesh (sanity-check the bake; NOT imported).
* ``<asset>_bake.json``     -- channel map, source transform, mask/seam telemetry.

Run with the conda `sagad` python (CLAUDE.md / MEMORY):
    C:\\Users\\PC\\anaconda3\\envs\\sagad\\python.exe scripts/bake_vertex_colors.py \\
        --asset gray-big-rock --source gains
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.logging_utils import get_logger  # noqa: E402
from deform import DeformConfig, apply_masks  # noqa: E402
from hitl.candidates import mask_from_gains  # noqa: E402

log = get_logger("bake_vertex_colors")

# Channel map -- locked [R, G, B, A] convention (CLAUDE.md Domain Constraints).
# NOTE: A is mobility (1 - fixed), per the bake convention in ARCHITECT sec. 3.
CHANNEL_MAP = {"R": "bend", "G": "noise", "B": "scale", "A": "mobility (1 - fixed)"}

DEFAULT_GAINS = {"bend": 0.6, "noise": 0.4, "scale": 0.3, "base_band": 0.25}


def mask_from_policy(data, asset: str, weights: Path, policy_ckpt: Path,
                     device: torch.device) -> torch.Tensor:
    """Deterministic reward-optimal mask from the trained PPO actor.

    Reproduces the train_ppo latent path exactly: frozen MeshMAE encode -> the
    weight-shared actor's distribution mean (``deterministic=True``).
    """
    from models import MeshMAE, MeshMAEConfig
    from rl import MaskActorCritic

    enc_ckpt = torch.load(weights, weights_only=False)
    encoder = MeshMAE(MeshMAEConfig(**enc_ckpt["config"])).to(device)
    encoder.load_state_dict(enc_ckpt["state_dict"])
    encoder.eval()
    with torch.no_grad():
        Z = encoder.encode(data).detach()

    pol_ckpt = torch.load(policy_ckpt, weights_only=False)
    if pol_ckpt.get("latent_dim", Z.shape[1]) != Z.shape[1]:
        raise ValueError(
            f"policy latent_dim={pol_ckpt['latent_dim']} != encoder D={Z.shape[1]}")
    policy = MaskActorCritic(Z.shape[1]).to(device)
    policy.load_state_dict(pol_ckpt["state_dict"])
    policy.eval()
    mask, *_ = policy.act(Z, deterministic=True)
    log.info("mask source=policy | %s -> N=%d D=%d", policy_ckpt.name, Z.shape[0], Z.shape[1])
    return mask.detach()


def seam_coherent_average(pos: np.ndarray, rgba: np.ndarray,
                          decimals: int = 5) -> tuple[np.ndarray, dict]:
    """Average RGBA across vertices that share a position (merge-then-average).

    UV-seam duplicates are split in the mesh but coincide in space; quantizing the
    position to ``decimals`` groups the duplicates, and each group is assigned its
    mean RGBA so every coincident copy is identical (no WPO seam tear).
    """
    keys = np.round(pos, decimals=decimals)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)  # np.unique can return a (N,1) index array on some versions

    summed = np.zeros((counts.shape[0], rgba.shape[1]), dtype=np.float64)
    np.add.at(summed, inv, rgba.astype(np.float64))
    averaged = (summed / counts[:, None])[inv]

    n_groups = int(counts.shape[0])
    n_seam = int((counts > 1).sum())
    max_dev = float(np.abs(averaged - rgba).max()) if rgba.size else 0.0
    telem = {
        "vertices": int(pos.shape[0]),
        "unique_positions": n_groups,
        "seam_groups": n_seam,
        "seam_vertices": int(pos.shape[0] - n_groups),
        "max_rgba_shift_from_merge": max_dev,
        "merge_decimals": decimals,
    }
    log.info("seam merge | N=%d unique=%d seam_groups=%d shifted_max=%.4f",
             telem["vertices"], n_groups, n_seam, max_dev)
    return averaged.astype(np.float32), telem


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--asset", default="gray-big-rock")
    p.add_argument("--source", choices=["policy", "gains"], default="gains")
    p.add_argument("--gains", type=str, default="",
                   help='JSON gains for --source gains, e.g. '
                        '\'{"bend":0.6,"noise":0.4,"scale":0.3,"base_band":0.25}\'')
    p.add_argument("--weights", default=str(ROOT / "data" / "processed" / "meshmae_baseline.pt"))
    p.add_argument("--policy", default="",
                   help="PPO policy ckpt (default data/rl/ppo_policy_<asset>.pt)")
    p.add_argument("--merge-decimals", type=int, default=5,
                   help="position-quantization decimals for seam grouping")
    p.add_argument("--out", default="")
    args = p.parse_args()

    device = torch.device("cpu")
    data = torch.load(ROOT / "data" / "processed" / f"{args.asset}.pt",
                      weights_only=False).to(device)
    N = int(data.pos.shape[0])

    # -- 1. shared mask M[N,4] = [bend, noise, scale, fixed] --------------------
    if args.source == "policy":
        policy_ckpt = Path(args.policy) if args.policy else \
            ROOT / "data" / "rl" / f"ppo_policy_{args.asset}.pt"
        mask = mask_from_policy(data, args.asset, Path(args.weights), policy_ckpt, device)
    else:
        gains = json.loads(args.gains) if args.gains else dict(DEFAULT_GAINS)
        mask = mask_from_gains(data, gains)
        log.info("mask source=gains | %s", gains)

    if mask.shape != (N, 4) or not torch.isfinite(mask).all():
        raise ValueError(f"bad mask: shape={tuple(mask.shape)} finite={torch.isfinite(mask).all()}")

    m_np = mask.detach().cpu().numpy()  # [N,4] = bend, noise, scale, fixed in [0,1]

    # -- 2. apply bake convention: A = mobility = 1 - fixed --------------------
    rgba01 = np.empty((N, 4), dtype=np.float32)
    rgba01[:, 0] = m_np[:, 0]            # R = bend
    rgba01[:, 1] = m_np[:, 1]            # G = noise
    rgba01[:, 2] = m_np[:, 2]            # B = scale
    rgba01[:, 3] = 1.0 - m_np[:, 3]      # A = mobility (1 - fixed)

    # -- 3. seam-coherent merge-then-average (ARCHITECT sec. 2.4) --------------
    pos_np = data.pos.detach().cpu().numpy()
    rgba01, seam_telem = seam_coherent_average(pos_np, rgba01, decimals=args.merge_decimals)
    rgba01 = np.clip(rgba01, 0.0, 1.0)
    rgba8 = np.rint(rgba01 * 255.0).astype(np.uint8)

    # -- 4. export base mesh + vertex colors (the WPO input) -------------------
    faces = data.face.t().cpu().numpy()
    base = trimesh.Trimesh(vertices=pos_np, faces=faces, process=False)
    base.visual.vertex_colors = rgba8  # -> glTF COLOR_0 on export
    out_dir = Path(args.out) if args.out else ROOT / "data" / "bake" / args.asset
    out_dir.mkdir(parents=True, exist_ok=True)
    baked = out_dir / f"{args.asset}_baked.glb"
    base.export(baked)

    # -- 5. deformed preview (sanity only) using the seam-averaged mask --------
    #     reconstruct fixed = 1 - mobility so the sandbox matches the baked buffer.
    avg_mask = torch.from_numpy(np.stack(
        [rgba01[:, 0], rgba01[:, 1], rgba01[:, 2], 1.0 - rgba01[:, 3]], axis=1)).float()
    result = apply_masks(data, avg_mask, DeformConfig())
    preview = out_dir / f"{args.asset}_preview.glb"
    result.mesh.export(preview)

    # -- 6. sidecar manifest for the UE5 import / material side ----------------
    transform = {
        "source_centroid": np.asarray(getattr(data, "source_centroid", [0, 0, 0])).reshape(-1).tolist(),
        "source_scale": float(getattr(data, "source_scale", 1.0)),
        "note": "mesh exported in unit-bounding-sphere normalized space; "
                "rescale on UE import or author WPO constants accordingly.",
    }
    manifest = {
        "asset": args.asset,
        "mask_source": args.source,
        "channel_map": CHANNEL_MAP,
        "num_vertices": N,
        "seam": seam_telem,
        "transform": transform,
        "mask_means": {"bend": float(rgba01[:, 0].mean()),
                       "noise": float(rgba01[:, 1].mean()),
                       "scale": float(rgba01[:, 2].mean()),
                       "mobility": float(rgba01[:, 3].mean())},
        "deform_telemetry": result.telemetry,
        "outputs": {"baked_glb": baked.name, "preview_glb": preview.name},
    }
    (out_dir / f"{args.asset}_bake.json").write_text(json.dumps(manifest, indent=2))
    log.info("baked %s | %d verts | seam_groups=%d -> %s",
             args.asset, N, seam_telem["seam_groups"], baked)
    log.info("UE5 import: glTF Interchange, Vertex Color Import Option = Replace")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
