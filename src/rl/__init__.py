"""Reinforcement-learning stage (Phase 4): PPO over the deformation sandbox.

Isolated module (CLAUDE.md modularity rule) orchestrating the lower-level
pieces -- the deformation sandbox (`src/deform`), the deterministic penalties +
learned `R_human` reward (`src/rewards`), and the MeshMAE encoder
(`src/models`) -- into a Gymnasium environment and a node-shared PPO
actor-critic that predicts continuous per-vertex RGBA masks.

* :mod:`composite` -- the unified reward
  `R = alpha*R_human + beta*R_variety - gamma*R_distortion - delta*R_physics`.
* :mod:`env`       -- single-asset, single-step Gymnasium environment that
  applies a predicted mask and returns the composite reward.
* :mod:`policy`    -- node-shared actor-critic over the MeshMAE latent Z[N,D].

See docs/ARCHITECT.md sec. 2.2 (actor policy) and docs/PLAN.md Phase 4.
"""

from .composite import CompositeWeights, composite_reward
from .env import DeformEnv
from .policy import MaskActorCritic

__all__ = [
    "CompositeWeights",
    "composite_reward",
    "DeformEnv",
    "MaskActorCritic",
]
