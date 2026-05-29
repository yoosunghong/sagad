"""CLI: evaluate the heuristic-driven baseline (no human feedback).

Ties together the Phase 2 stack -- deformation sandbox + deterministic penalties
+ heuristic reward + multi-view renderer -- into a reference baseline. For each
asset it samples a batch of geometry-structured candidate deformations, scores
each by the human-free composite reward

    R = beta*variety - gamma*distortion - delta*physics,

selects the best variant, renders it, and records batch diversity. Results are
written to ``data/baseline/heuristic_baseline.json`` as the reference the future
RLHF/PPO policy (Phases 3-4) must beat.

Usage:
    python scripts/eval_baseline.py [--graphs <dir>] [--candidates K] [--seed S]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.logging_utils import get_logger  # noqa: E402
from deform import DeformConfig, apply_masks  # noqa: E402
from rewards import HeuristicWeights, batch_diversity, evaluate_penalties, heuristic_reward  # noqa: E402
from rewards.penalties import _face_adjacency  # noqa: E402
from render import RenderConfig, render_multiview, save_views  # noqa: E402

log = get_logger("eval_baseline")

DEFAULT_GRAPHS = ["gray-big-rock", "rock_17", "tree", "tree_1", "tree_2"]


def _candidate_masks(data, rng: np.random.Generator) -> tuple[torch.Tensor, int]:
    """Geometry-structured, base-locked mask with randomized channel gains."""
    pos = data.pos
    up = torch.tensor([0.0, 0.0, 1.0], dtype=pos.dtype, device=pos.device)
    h = pos @ up
    h_norm = (h - h.min()) / (h.max() - h.min()).clamp(min=1e-8)

    base_band, = (0.25,)
    floor = base_band * 0.4
    fixed = ((base_band - h_norm) / (base_band - floor)).clamp(0.0, 1.0)

    g_bend = float(rng.uniform(0.3, 1.0))
    g_noise = float(rng.uniform(0.1, 0.9))
    g_scale = float(rng.uniform(0.1, 0.7))
    bend = g_bend * h_norm
    noise = torch.full_like(h_norm, g_noise)
    scale = torch.full_like(h_norm, g_scale)
    seed = int(rng.integers(0, 1 << 30))
    return torch.stack([bend, noise, scale, fixed], dim=1), seed


def _evaluate_asset(stem: str, device, K: int, rng: np.random.Generator,
                    weights: HeuristicWeights, render_cfg: RenderConfig) -> dict:
    data = torch.load(ROOT / "data" / "processed" / f"{stem}.pt", weights_only=False).to(device)
    adj = _face_adjacency(data.face.t().contiguous())
    log.info("=== %s | N=%d | %d candidates ===", stem, data.num_nodes, K)

    variants, offsets = [], []
    for k in range(K):
        masks, noise_seed = _candidate_masks(data, rng)
        result = apply_masks(data, masks, DeformConfig(noise_seed=noise_seed))
        pen = evaluate_penalties(data, result.pos, face_adjacency=adj)
        r = heuristic_reward(result.telemetry["disp_mean"],
                             pen["distortion_total"], pen["physics_total"], weights)
        r["candidate"] = k
        variants.append((r, result))
        offsets.append(result.offset)
        log.info("  cand %d | reward=%.4f variety=%.4f distortion=%.4f physics=%.4f",
                 k, r["reward"], r["variety"], r["distortion"], r["physics"])

    diversity = batch_diversity(torch.stack(offsets, dim=0))
    best_r, best_result = max(variants, key=lambda rv: rv[0]["reward"])
    rewards = np.array([rv[0]["reward"] for rv in variants])
    log.info("  -> best cand=%d reward=%.4f | batch diversity=%.4f | mean reward=%.4f",
             best_r["candidate"], best_r["reward"], diversity, float(rewards.mean()))

    # Render the best variant for inspection.
    views = render_multiview(best_result.mesh, render_cfg)
    save_views(views, ROOT / "data" / "baseline" / "renders", f"{stem}_best")

    return {
        "asset": stem,
        "num_vertices": int(data.num_nodes),
        "best": best_r,
        "mean_reward": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "batch_diversity": diversity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the heuristic baseline.")
    parser.add_argument("--graphs", nargs="*", default=DEFAULT_GRAPHS)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--views", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    weights = HeuristicWeights()
    render_cfg = RenderConfig(n_views=args.views)
    log.info("device=%s | weights(beta/gamma/delta)=%.1f/%.1f/%.1f | seed=%d",
             device, weights.variety, weights.distortion, weights.physics, args.seed)

    results = [_evaluate_asset(s, device, args.candidates, rng, weights, render_cfg)
               for s in args.graphs]

    # -- aggregate baseline report -----------------------------------------
    best_rewards = np.array([r["best"]["reward"] for r in results])
    report = {
        "weights": weights.__dict__,
        "candidates_per_asset": args.candidates,
        "seed": args.seed,
        "per_asset": results,
        "aggregate": {
            "mean_best_reward": float(best_rewards.mean()),
            "mean_batch_diversity": float(np.mean([r["batch_diversity"] for r in results])),
            "num_assets": len(results),
        },
    }
    out = ROOT / "data" / "baseline" / "heuristic_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    log.info("=== BASELINE SUMMARY ===")
    for r in results:
        log.info("  %-16s | best reward=%+.4f (cand %d) | diversity=%.4f",
                 r["asset"], r["best"]["reward"], r["best"]["candidate"], r["batch_diversity"])
    log.info("  AGGREGATE mean best reward=%+.4f | mean diversity=%.4f",
             report["aggregate"]["mean_best_reward"], report["aggregate"]["mean_batch_diversity"])
    log.info("report -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
