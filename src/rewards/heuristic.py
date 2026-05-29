"""Heuristic composite reward (no human feedback).

The Phase 4 composite reward is
    R = alpha*R_human + beta*R_variety - gamma*R_distortion - delta*R_physics.
This module implements the *human-free* subset used for the Phase 2 baseline:
    R_heuristic = beta*R_variety - gamma*R_distortion - delta*R_physics.

``R_human`` (Bradley-Terry preference model) arrives in Phase 3; the CLIP-based
diversity evaluator of ARCHITECT.md sec. 2.4 will later replace/augment the
geometric variety proxy used here. Keeping this isolated in `src/rewards`
honors the CLAUDE.md "reward constraints in dedicated modules" rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HeuristicWeights:
    """Coefficients for the human-free composite reward."""

    variety: float = 1.0     # beta  -- rewards meaningful deformation
    distortion: float = 1.0  # gamma -- penalizes surface degradation
    physics: float = 1.0     # delta -- penalizes ground-contact violations


def heuristic_reward(variety: float, distortion: float, physics: float,
                     weights: HeuristicWeights | None = None) -> dict:
    """Combine the heuristic terms into a single reward + its breakdown.

    ``variety`` is a geometric activity proxy (e.g. mean per-vertex
    displacement); ``distortion`` / ``physics`` are the aggregated penalties
    from :func:`rewards.penalties.evaluate_penalties`.
    """
    w = weights or HeuristicWeights()
    reward = (w.variety * variety
              - w.distortion * distortion
              - w.physics * physics)
    return {
        "reward": float(reward),
        "variety": float(variety),
        "distortion": float(distortion),
        "physics": float(physics),
    }


def batch_diversity(displacements: torch.Tensor) -> float:
    """Mean pairwise per-vertex distance across a batch of displacement fields.

    ``displacements`` is ``(K, N, 3)`` for K variants of the *same* asset; the
    return is the average L2 distance between distinct variant pairs, normalized
    per vertex -- a geometric stand-in for the §2.4 CLIP diversity metric until
    the renders-based version lands. Returns 0.0 for K < 2.
    """
    k = displacements.shape[0]
    if k < 2:
        return 0.0
    flat = displacements.reshape(k, -1)
    # Pairwise distances over the upper triangle (no self-pairs, no double count).
    dists = torch.cdist(flat, flat)
    iu = torch.triu_indices(k, k, offset=1, device=dists.device)
    per_vertex = dists[iu[0], iu[1]] / displacements.shape[1] ** 0.5
    return float(per_vertex.mean())
