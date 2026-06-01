"""CLI: synthetic-oracle preference labels (stand-in for human collection).

A real artist is not in the loop here, so to bring up and validate the
Bradley-Terry training pipeline we generate preference labels from a
deterministic *oracle*: each candidate is re-scored with the Phase 2 heuristic
reward (variety - distortion - physics), reconstructed exactly from the gains
stored in the manifest. For every scheduled pair we sample a Bradley-Terry-
consistent label

    P(a beats b) = sigmoid( (r_oracle(a) - r_oracle(b)) / temperature ),

with ``temperature`` injecting realistic label noise. Labels stream into a
PreferenceStore (annotator ``synthetic-oracle``) and the per-candidate oracle
scores are saved for the correlation check in ``train_reward_model.py``.

This is explicitly NOT real artist feedback -- it bootstraps + validates the
machinery so the same trainer runs unchanged on real labels from the Gradio UI.

Usage:
    python scripts/simulate_preferences.py [--temperature T] [--seed S]
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
from hitl import PreferenceStore, load_manifest, mask_from_gains  # noqa: E402
from rewards import HeuristicWeights, evaluate_penalties, heuristic_reward  # noqa: E402
from rewards.penalties import _face_adjacency  # noqa: E402

log = get_logger("simulate_preferences")


def _oracle_scores(manifest, device, weights: HeuristicWeights) -> dict[str, float]:
    """Heuristic reward per candidate, reconstructed from stored gains+seed."""
    graphs = ROOT / "data" / "processed"
    cache: dict[str, tuple] = {}                      # asset -> (data, face_adjacency)
    scores: dict[str, float] = {}
    for c in manifest.candidates:
        if c.asset not in cache:
            data = torch.load(graphs / f"{c.asset}.pt", weights_only=False).to(device)
            adj = _face_adjacency(data.face.t().contiguous())
            cache[c.asset] = (data, adj)
        data, adj = cache[c.asset]

        masks = mask_from_gains(data, c.gains)
        result = apply_masks(data, masks, DeformConfig(noise_seed=c.noise_seed))
        pen = evaluate_penalties(data, result.pos, face_adjacency=adj)
        r = heuristic_reward(result.telemetry["disp_mean"],
                             pen["distortion_total"], pen["physics_total"], weights)
        scores[c.candidate_id] = r["reward"]
        log.info("  oracle %s | reward=%+.4f (variety=%.4f distortion=%.4f physics=%.4f)",
                 c.candidate_id, r["reward"], r["variety"], r["distortion"], r["physics"])
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic-oracle preference labels.")
    parser.add_argument("--manifest", default=str(ROOT / "data" / "preferences" / "manifest.json"))
    parser.add_argument("--store", default=str(ROOT / "data" / "preferences" / "preferences_synthetic.jsonl"))
    parser.add_argument("--oracle-out", default=str(ROOT / "data" / "preferences" / "oracle_rewards.json"))
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="logistic label-noise scale; smaller => cleaner labels")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    manifest = load_manifest(Path(args.manifest))
    log.info("manifest=%s | %d candidates / %d pairs | temp=%.3f",
             args.manifest, len(manifest.candidates), len(manifest.pairs), args.temperature)

    scores = _oracle_scores(manifest, device, HeuristicWeights())

    store_path = Path(args.store)
    if store_path.exists():                            # fresh synthetic run
        store_path.unlink()
    store = PreferenceStore(store_path)

    agree = 0
    for pair in manifest.pairs:
        ra, rb = scores[pair.a], scores[pair.b]
        p_a = 1.0 / (1.0 + np.exp(-(ra - rb) / max(args.temperature, 1e-6)))
        choice = "a" if rng.random() < p_a else "b"
        winner = pair.a if choice == "a" else pair.b
        if (winner == pair.a) == (ra >= rb):
            agree += 1
        store.record(pair.pair_id, pair.asset, pair.a, pair.b, choice, "synthetic-oracle")

    Path(args.oracle_out).write_text(json.dumps(scores, indent=2))
    log.info("=== SYNTHETIC PREFERENCES READY ===")
    log.info("  %d labels -> %s", len(manifest.pairs), store_path)
    log.info("  label/oracle agreement = %.1f%% (noise from temperature=%.3f)",
             100.0 * agree / max(len(manifest.pairs), 1), args.temperature)
    log.info("  oracle scores -> %s", args.oracle_out)
    log.info("  train with: python scripts/train_reward_model.py --store %s", store_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
