"""Subprocess bridge: sample K per-instance variant vectors as JSON (Phase 5).

This is the Python endpoint the UE5 plugin calls via ``FPlatformProcess`` (a plain
stdin-less subprocess). It loads the trained variant-diffusion checkpoint for an
asset, samples K vectors from the learned distribution, decodes them to raw
Per-Instance Custom Data units, and prints **one JSON object to stdout**.

Contract (so the C++ side can parse deterministically):
* **stdout carries ONLY the JSON payload.** All logging goes to stderr (the repo's
  ``get_logger`` uses ``logging.basicConfig``, which defaults to stderr), so stdout
  is never polluted by INFO lines or NaN-guard telemetry.
* Field order in ``"fields"`` is the order to write Per-Instance Custom Data floats
  on each HISM/Nanite instance; ``"variants"`` rows align with that order.
* Exit code 0 + valid JSON on success; non-zero with a JSON ``{"error": ...}`` on
  stdout on failure, so the plugin always gets parseable output.

Run (conda `sagad` python; see MEMORY):
    C:\\Users\\PC\\anaconda3\\envs\\sagad\\python.exe scripts/sample_variants.py \\
        --asset gray-big-rock --k 64 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _emit(obj: dict, code: int = 0) -> int:
    """Write the single JSON payload to stdout and return the process exit code."""
    sys.stdout.write(json.dumps(obj))
    sys.stdout.flush()
    return code


def main() -> int:
    p = argparse.ArgumentParser(description="Sample K variant vectors -> JSON (UE5 bridge).")
    p.add_argument("--asset", default="gray-big-rock")
    p.add_argument("--k", type=int, default=64, help="number of instances to sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", default="", help="override diffusion ckpt path")
    p.add_argument("--out", default="", help="also write JSON to this file")
    args = p.parse_args()

    try:
        import torch  # imported lazily so --help is instant and import errors are caught

        from diffusion import DiffusionMLP, GaussianDiffusion, ParamField, VariantParamSpec

        if args.k <= 0:
            raise ValueError(f"--k must be positive, got {args.k}")

        ckpt_path = Path(args.ckpt) if args.ckpt else \
            ROOT / "data" / "diffusion" / f"variant_diffusion_{args.asset}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"no diffusion checkpoint for asset '{args.asset}': {ckpt_path}. "
                "Train it first with scripts/build_variant_dataset.py.")

        torch.manual_seed(args.seed)
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")

        spec = VariantParamSpec([ParamField(*f) for f in ckpt["spec_fields"]])
        diff = GaussianDiffusion(
            DiffusionMLP(dim=ckpt["dim"], hidden=128),
            dim=ckpt["dim"], timesteps=ckpt["timesteps"])
        diff.load_state_dict(ckpt["model_state"])
        diff.eval()

        raw = spec.from_model(diff.sample(args.k))  # (K, D) in raw PICD units
        if not torch.isfinite(raw).all():           # NaN guard (CLAUDE.md)
            raise ValueError("NaN/Inf in sampled variant vectors")

        rows = raw.tolist()
        payload = {
            "asset": args.asset,
            "seed": args.seed,
            "count": int(args.k),
            "fields": spec.names,                      # PICD float order
            "variants": rows,                          # K rows aligned to fields
            "named": [dict(zip(spec.names, r)) for r in rows],
        }
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(payload, indent=2))
        return _emit(payload, 0)

    except Exception as exc:  # always hand the plugin parseable output
        return _emit({"error": f"{type(exc).__name__}: {exc}"}, 1)


if __name__ == "__main__":
    raise SystemExit(main())
