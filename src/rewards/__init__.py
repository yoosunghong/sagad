"""Reward constraints (deterministic geometry & physics penalties).

Isolated module for the heuristic penalty terms that regularize deformations
(CLAUDE.md: "Keep ... reward constraints ... strictly isolated within
dedicated modules"). Kept separate from the deformation sandbox
(`src/deform`) and the learned/human reward terms that arrive in Phases 3-4.

* :mod:`penalties` -- Laplacian smoothness, normal consistency, and ground
  contact penalties + an aggregate evaluator.

These map to the ``-gamma R_distortion - delta R_physics`` terms of the
composite reward (docs/ARCHITECT.md / docs/PLAN.md Phase 4) and to the
Geometric Regularizer of ARCHITECT.md sec. 2.4.
"""

from .penalties import (
    PenaltyWeights,
    evaluate_penalties,
    ground_contact,
    laplacian_coordinates,
    laplacian_distortion,
    normal_consistency,
)
from .heuristic import HeuristicWeights, batch_diversity, heuristic_reward
from .preference import (
    BradleyTerryReward,
    TrainConfig,
    bt_loss,
    pairwise_accuracy,
    train_reward_model,
)

__all__ = [
    "PenaltyWeights",
    "evaluate_penalties",
    "ground_contact",
    "laplacian_coordinates",
    "laplacian_distortion",
    "normal_consistency",
    "HeuristicWeights",
    "batch_diversity",
    "heuristic_reward",
    "BradleyTerryReward",
    "TrainConfig",
    "bt_loss",
    "pairwise_accuracy",
    "train_reward_model",
]
