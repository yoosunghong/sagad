"""Unified composite reward for the PPO loop (docs/PLAN.md Phase 4).

    R = alpha*R_human + beta*R_variety - gamma*R_distortion - delta*R_physics

Generalizes the Phase 2 human-free `heuristic_reward` by adding the learned
`alpha*R_human` term (the Bradley-Terry reward model of `rewards.preference`).
The four sub-terms are supplied by the caller (the env): `R_variety` from the
sandbox displacement activity, `R_distortion`/`R_physics` from
`rewards.evaluate_penalties`, and `R_human` from the reward model on the
deformed render (0 when human feedback is disabled).

Scale note: the geometric terms live on the unit-bounding-sphere scale while
`R_human` is a CLIP-feature logit; the weights below are placeholders to be
tuned against the heuristic baseline once `R_human` is enabled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CompositeWeights:
    """Composite-reward coefficients (alpha/beta/gamma/delta)."""

    human: float = 1.0       # alpha -- learned artist preference (R_human)
    variety: float = 1.0     # beta  -- rewards meaningful deformation
    distortion: float = 1.0  # gamma -- penalizes surface degradation
    physics: float = 1.0     # delta -- penalizes ground-contact violations


def composite_reward(variety: float, distortion: float, physics: float,
                     human: float = 0.0,
                     weights: CompositeWeights | None = None) -> dict:
    """Combine the four reward terms into a scalar + its breakdown.

    ``human`` is the (already-scaled) reward-model output; pass 0.0 to recover
    the human-free heuristic reward. NaN guard on every term (CLAUDE.md).
    """
    w = weights or CompositeWeights()
    reward = (w.human * human
              + w.variety * variety
              - w.distortion * distortion
              - w.physics * physics)
    out = {
        "reward": float(reward),
        "r_human": float(human),
        "variety": float(variety),
        "distortion": float(distortion),
        "physics": float(physics),
    }
    if not all(math.isfinite(v) for v in out.values()):
        raise ValueError(f"NaN/Inf detected in composite reward terms: {out}")
    return out
