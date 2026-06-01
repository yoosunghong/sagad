"""Single-asset Gymnasium environment over the deformation sandbox (Phase 4).

Each episode is a single step (a contextual-bandit formulation of the
"generate a mask -> score it" task): ``reset`` returns the asset's fixed
MeshMAE latent ``Z`` as the observation, ``step(action)`` applies the predicted
``[N, 4]`` RGBA mask through the Phase 2 sandbox, scores the result with the
composite reward, and terminates. This keeps a standard Gymnasium interface
(docs/PLAN.md Phase 4) while matching the one-shot deformation task.

The environment owns *all* reward computation (deterministic penalties +
optional learned ``R_human``); the policy stays a pure mask predictor. The
``R_human`` term is injected as a ``human_reward_fn(mesh) -> float`` callback so
this module stays decoupled from CLIP / the reward model (those live in
`hitl`/`rewards`).
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from data.logging_utils import get_logger
from deform import DeformConfig, apply_masks
from rewards import evaluate_penalties
from rewards.penalties import _face_adjacency
from .composite import CompositeWeights, composite_reward

log = get_logger("rl.env")


class DeformEnv(gym.Env):
    """Gymnasium env: predict an RGBA mask for one asset, get composite reward."""

    metadata = {"render_modes": []}

    def __init__(self, data, latent: torch.Tensor,
                 deform_cfg: DeformConfig | None = None,
                 weights: CompositeWeights | None = None,
                 human_reward_fn: Callable | None = None):
        super().__init__()
        self.data = data
        self.device = latent.device
        self.latent = latent                      # (N, D) frozen MeshMAE encoding
        self.deform_cfg = deform_cfg or DeformConfig()
        self.weights = weights or CompositeWeights()
        self.human_reward_fn = human_reward_fn

        self.N, self.D = latent.shape
        self._face_adjacency = _face_adjacency(data.face.t().contiguous())
        self._obs = latent.detach().cpu().numpy().astype(np.float32).reshape(-1)

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.N * self.D,),
                                             dtype=np.float32)
        self.action_space = spaces.Box(0.0, 1.0, shape=(self.N * 4,), dtype=np.float32)
        log.info("DeformEnv | N=%d D=%d | human_reward=%s",
                 self.N, self.D, human_reward_fn is not None)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs, {}

    def step(self, action):
        masks = torch.as_tensor(np.asarray(action), dtype=self.data.pos.dtype,
                                device=self.device).reshape(self.N, 4)
        result = apply_masks(self.data, masks, self.deform_cfg)
        pen = evaluate_penalties(self.data, result.pos, face_adjacency=self._face_adjacency)

        human = float(self.human_reward_fn(result.mesh)) if self.human_reward_fn else 0.0
        rew = composite_reward(result.telemetry["disp_mean"],
                               pen["distortion_total"], pen["physics_total"],
                               human=human, weights=self.weights)

        info = {**rew, "telemetry": result.telemetry}
        return self._obs, rew["reward"], True, False, info
