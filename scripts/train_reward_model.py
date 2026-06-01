"""CLI: train + validate the Bradley-Terry preference reward model (Phase 3).

Consumes the pairwise judgements collected in a PreferenceStore (real artist
labels from the Gradio UI, or the synthetic-oracle stand-in), featurizes each
candidate with CLIP multi-view embeddings, and fits the Bradley-Terry reward
head. Validation reports:

* **held-out pairwise accuracy** -- does the learned reward order unseen pairs
  the way the annotator did?
* **rank correlation with the oracle / qualitative scores** -- Spearman & Pearson
  between the learned per-candidate reward and ``oracle_rewards.json`` (the
  ARCHITECT.md sec. 2.4 "correlation with human qualitative scores" check; with
  the synthetic oracle this is correlation against the heuristic ground truth).

Usage:
    python scripts/train_reward_model.py [--store PATH] [--hidden H]
        [--epochs N] [--val-frac F] [--seed S]
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
from hitl import PreferenceStore, load_manifest  # noqa: E402
from hitl.features import candidate_features  # noqa: E402
from rewards import TrainConfig, pairwise_accuracy, train_reward_model  # noqa: E402

log = get_logger("train_reward_model")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x, y).statistic)
    except Exception:                                  # rank-corr fallback
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the Bradley-Terry reward model.")
    parser.add_argument("--manifest", default=str(ROOT / "data" / "preferences" / "manifest.json"))
    parser.add_argument("--store", default=str(ROOT / "data" / "preferences" / "preferences_synthetic.jsonl"))
    parser.add_argument("--oracle", default=str(ROOT / "data" / "preferences" / "oracle_rewards.json"))
    parser.add_argument("--feature-cache", default=str(ROOT / "data" / "preferences" / "clip_features.pt"))
    parser.add_argument("--out", default=str(ROOT / "data" / "preferences" / "reward_model.pt"))
    parser.add_argument("--report", default=str(ROOT / "data" / "preferences" / "reward_report.json"))
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    manifest = load_manifest(Path(args.manifest))
    store = PreferenceStore(args.store)
    comps = store.to_comparisons()
    if not comps:
        log.error("no (winner, loser) comparisons in %s -- collect/simulate first", args.store)
        return 1
    log.info("device=%s | %d comparisons from %s", device, len(comps), args.store)

    # -- featurize candidates (CLIP, cached) -------------------------------
    feats = candidate_features(manifest, ROOT, device, cache_path=Path(args.feature_cache))
    dim = next(iter(feats.values())).shape[0]

    # -- assemble aligned (winner, loser) feature pairs --------------------
    win = torch.stack([feats[c["winner"]] for c in comps])     # (P, dim)
    lose = torch.stack([feats[c["loser"]] for c in comps])

    # -- train / val split over comparison pairs ---------------------------
    P = win.shape[0]
    perm = rng.permutation(P)
    n_val = int(round(args.val_frac * P))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr_win, tr_lose = win[tr_idx], lose[tr_idx]
    va_win, va_lose = win[val_idx], lose[val_idx]

    cfg = TrainConfig(epochs=args.epochs, hidden=args.hidden, seed=args.seed)
    model, history = train_reward_model(tr_win, tr_lose, va_win, va_lose, cfg)

    model.eval()
    with torch.no_grad():
        train_acc = pairwise_accuracy(model(tr_win), model(tr_lose))
        val_acc = pairwise_accuracy(model(va_win), model(va_lose)) if n_val else float("nan")

    # -- per-asset breakdown (we lean on trees -- PROPOSAL.md sec. 5.1) -----
    # Accuracy over *all* comparisons grouped by asset; trees are the
    # discriminating organic case, rocks the easy one. Diagnostic (mixes
    # train+val), so read alongside the held-out val_acc above.
    with torch.no_grad():
        correct = (model(win) > model(lose)).cpu().numpy()
    assets = np.array([c["asset"] for c in comps])
    is_tree = np.array([a.startswith("tree") for a in assets])
    per_asset_acc = {str(a): float(correct[assets == a].mean()) for a in sorted(set(assets))}
    group_acc = {
        "tree": float(correct[is_tree].mean()) if is_tree.any() else float("nan"),
        "rock": float(correct[~is_tree].mean()) if (~is_tree).any() else float("nan"),
    }
    log.info("per-asset fit acc: %s", {k: round(v, 3) for k, v in per_asset_acc.items()})
    log.info("group fit acc | tree=%.3f rock=%.3f (n_tree=%d n_rock=%d)",
             group_acc["tree"], group_acc["rock"], int(is_tree.sum()), int((~is_tree).sum()))

    # -- correlation with oracle / qualitative scores ----------------------
    corr = {}
    oracle_path = Path(args.oracle)
    if oracle_path.exists():
        oracle = json.loads(oracle_path.read_text())
        ids = [cid for cid in oracle if cid in feats]
        feat_mat = torch.stack([feats[cid] for cid in ids])
        with torch.no_grad():
            learned = model(feat_mat).cpu().numpy()
        truth = np.array([oracle[cid] for cid in ids])
        corr = {
            "n_candidates": len(ids),
            "spearman": _spearman(learned, truth),
            "pearson": float(np.corrcoef(learned, truth)[0, 1]),
        }
        log.info("reward vs oracle | n=%d spearman=%.3f pearson=%.3f",
                 corr["n_candidates"], corr["spearman"], corr["pearson"])

    torch.save({"state_dict": model.state_dict(), "in_dim": dim, "hidden": args.hidden}, args.out)
    report = {
        "store": str(args.store),
        "n_comparisons": len(comps),
        "n_train_pairs": int(tr_win.shape[0]),
        "n_val_pairs": int(va_win.shape[0]),
        "feature": {"source": "clip-vit-base-patch32", "dim": int(dim)},
        "train_acc": train_acc,
        "val_acc": val_acc,
        "per_asset_acc": per_asset_acc,
        "group_acc": group_acc,
        "correlation": corr,
        "history": history,
    }
    Path(args.report).write_text(json.dumps(report, indent=2))

    log.info("=== REWARD MODEL TRAINED ===")
    log.info("  train pairwise acc=%.3f | val pairwise acc=%.3f", train_acc, val_acc)
    if corr:
        log.info("  correlation w/ oracle: spearman=%.3f pearson=%.3f",
                 corr["spearman"], corr["pearson"])
    log.info("  model -> %s | report -> %s", args.out, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
