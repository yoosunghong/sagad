"""Texture-safe bootstrap for the variant diffusion sampler (ARCHITECT sec. 2.5).

End-to-end, no human labels: draw random variant vectors, deform the asset through
the §2.3 sandbox, score them with the §2.4 deterministic penalties (including the
new ``stretch`` / texture-integrity term), keep the *feasible* subset, and train the
§2.5 DDPM on it. The learned distribution then only covers deformations that do not
smear/tear textures under WPO -- "texture-safe by construction".

    sandbox + penalties  ->  feasible variant set  ->  DDPM  ->  sample K variants

Run (use the conda `sagad` python, not the Store stub):
    python scripts/build_variant_dataset.py --asset gray-big-rock
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion import (  # noqa: E402
    DiffusionMLP,
    GaussianDiffusion,
    ParamField,
    VariantParamSpec,
    train_diffusion,
)
from deform import DeformConfig, apply_masks  # noqa: E402
from rewards import PenaltyWeights, evaluate_penalties  # noqa: E402
from rewards.penalties import _face_adjacency  # noqa: E402
from hitl.candidates import mask_from_gains  # noqa: E402
from data.logging_utils import get_logger  # noqa: E402

log = get_logger("build_variant_dataset")


# Variant schema whose field names ARE the mask-gain keys, so a sampled vector
# decodes straight into `mask_from_gains` (ranges mirror geometry_structured_mask).
def organic_gain_spec() -> VariantParamSpec:
    return VariantParamSpec([
        ParamField("bend", 0.0, 1.2),
        ParamField("noise", 0.0, 1.0),
        ParamField("scale", 0.0, 0.8),
        ParamField("base_band", 0.15, 0.35),
    ])


def score_variant(data, raw: torch.Tensor, spec: VariantParamSpec,
                  cfg: DeformConfig, face_adjacency: torch.Tensor) -> dict:
    """Deform by one raw variant vector and return its penalty breakdown + disp."""
    gains = dict(zip(spec.names, raw.tolist()))
    masks = mask_from_gains(data, gains)
    result = apply_masks(data, masks, cfg)
    pen = evaluate_penalties(data, result.pos, face_adjacency=face_adjacency)
    pen["disp_mean"] = result.telemetry["disp_mean"]
    pen["stretch_val"] = pen["stretch"]
    return pen


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset", default="gray-big-rock")
    p.add_argument("--draws", type=int, default=400, help="random proposals to score")
    p.add_argument("--keep-frac", type=float, default=0.5, help="feasible fraction kept")
    p.add_argument("--min-disp", type=float, default=0.005, help="reject trivial no-ops")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--sample-k", type=int, default=512, help="variants to sample post-train")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    device = torch.device("cpu")

    data = torch.load(ROOT / "data" / "processed" / f"{args.asset}.pt",
                      weights_only=False).to(device)
    spec = organic_gain_spec()
    cfg = DeformConfig()
    faces = data.face.t().contiguous()
    fa = _face_adjacency(faces)
    log.info("asset=%s N=%d | drawing %d proposals", args.asset, data.pos.shape[0], args.draws)

    # -- 1. propose + score ------------------------------------------------------
    proposals = spec.sample_uniform(args.draws, generator=gen)
    rows = []
    for i in range(args.draws):
        pen = score_variant(data, proposals[i], spec, cfg, fa)
        rows.append(pen)
    distortion = torch.tensor([r["distortion_total"] for r in rows])
    disp = torch.tensor([r["disp_mean"] for r in rows])

    # -- 2. feasibility filter: non-trivial AND low distortion -------------------
    nontrivial = disp >= args.min_disp
    thresh = torch.quantile(distortion[nontrivial], args.keep_frac)
    keep = nontrivial & (distortion <= thresh)
    kept_raw = proposals[keep]
    log.info("feasible set | %d/%d kept (distortion<=%.4f, disp>=%.3f)",
             int(keep.sum()), args.draws, float(thresh), args.min_disp)
    if kept_raw.shape[0] < 16:
        raise SystemExit("too few feasible samples; widen --keep-frac or --draws")

    # -- 3. train the DDPM on the feasible distribution --------------------------
    train_x = spec.to_model(kept_raw)
    diff = GaussianDiffusion(DiffusionMLP(dim=spec.dim, hidden=128),
                             dim=spec.dim, timesteps=100)
    train_diffusion(diff, train_x, epochs=args.epochs, lr=1e-3,
                    batch_size=min(128, train_x.shape[0]), log_every=max(1, args.epochs // 4))

    # -- 4. sample + re-score: how texture-safe is the learned distribution? -----
    diff.eval()
    sampled_raw = spec.from_model(diff.sample(args.sample_k))
    samp_distortion = torch.tensor([
        score_variant(data, sampled_raw[i], spec, cfg, fa)["distortion_total"]
        for i in range(args.sample_k)
    ])
    safe_frac = float((samp_distortion <= thresh).float().mean())
    log.info("sampled %d | %.1f%% within feasible threshold | distortion mean train=%.4f sampled=%.4f",
             args.sample_k, 100.0 * safe_frac, float(distortion[keep].mean()),
             float(samp_distortion.mean()))

    # -- 5. persist --------------------------------------------------------------
    out_dir = ROOT / "data" / "diffusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"variant_diffusion_{args.asset}.pt"
    torch.save({
        "model_state": diff.state_dict(),
        "dim": spec.dim,
        "timesteps": diff.timesteps,
        "spec_fields": [(f.name, f.lo, f.hi, f.integer) for f in spec.fields],
        "asset": args.asset,
    }, ckpt)
    report = {
        "asset": args.asset,
        "n_drawn": args.draws,
        "n_kept": int(keep.sum()),
        "distortion_threshold": float(thresh),
        "train_distortion_mean": float(distortion[keep].mean()),
        "sampled_distortion_mean": float(samp_distortion.mean()),
        "sampled_texture_safe_frac": safe_frac,
        "example_variants": [dict(zip(spec.names, sampled_raw[i].tolist())) for i in range(3)],
    }
    (out_dir / f"variant_report_{args.asset}.json").write_text(json.dumps(report, indent=2))
    log.info("saved %s + report | example variant: %s", ckpt.name, report["example_variants"][0])


if __name__ == "__main__":
    main()
