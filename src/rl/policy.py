"""Node-shared actor-critic for continuous per-vertex mask prediction.

Implements the Actor Policy Module of docs/ARCHITECT.md sec. 2.2: an MLP
decoding block (capped with Sigmoid) that maps the MeshMAE latent
``Z in R^{N x D}`` to a continuous RGBA mask ``M in [0, 1]^{N x 4}``. The same
MLP is applied to every node (weight-shared), so the policy is independent of
the vertex count ``N`` and matches the "localized" embedding design.

Continuous-action parameterization: the actor outputs a per-node, per-channel
*logit* mean; actions are sampled in logit space from a diagonal Gaussian
(global per-channel ``log_std``) and squashed through a sigmoid to land in
``[0, 1]``. Working in logit space keeps the log-prob well-defined (no density
spikes at the 0/1 boundaries that a Beta/clipped-Gaussian-on-[0,1] would hit)
and is numerically friendly for PPO. The critic mean-pools the per-node value
heads into a single scalar graph value V(Z).

Log-prob convention: the action has ``N x 4`` dimensions (``N`` in the
thousands), so the joint log-prob is *averaged* over the action dims rather than
summed. Summing would make the PPO importance ratio ``exp(sum of N*4 tiny
per-dim diffs)`` explode (huge KL, clipping saturates, no learning); the
per-dimension mean keeps the ratio ``O(1)`` for this weight-shared policy while
preserving the gradient direction.

NaN guards (CLAUDE.md): the continuous PPO output path is asserted finite at the
logit-mean and value heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class MaskActorCritic(nn.Module):
    """Weight-shared per-node Gaussian-in-logit-space policy + scalar critic."""

    def __init__(self, latent_dim: int, hidden: int = 128, n_channels: int = 4,
                 init_log_std: float = -0.5):
        super().__init__()
        self.n_channels = n_channels
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_channels),
        )
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.full((n_channels,), float(init_log_std)))

    # -- distribution / heads ---------------------------------------------
    def _dist(self, z: torch.Tensor) -> Normal:
        mean = self.actor(z)                       # (N, 4) logit means
        if not torch.isfinite(mean).all():
            raise ValueError("NaN/Inf detected in actor logit mean")
        std = self.log_std.exp().clamp(1e-3, 10.0).expand_as(mean)
        return Normal(mean, std)

    def value(self, z: torch.Tensor) -> torch.Tensor:
        """Scalar graph value V(Z) = mean over per-node value heads."""
        v = self.critic(z).mean()
        if not torch.isfinite(v).all():
            raise ValueError("NaN/Inf detected in critic value")
        return v

    # -- rollout / update --------------------------------------------------
    @torch.no_grad()
    def act(self, z: torch.Tensor, deterministic: bool = False):
        """Sample a mask for graph ``z``; return (mask, logits, logp, value).

        ``logits`` are the pre-sigmoid samples (stored for the PPO update);
        ``logp`` is the log-prob averaged over the ``N x 4`` action dims.
        """
        dist = self._dist(z)
        logits = dist.mean if deterministic else dist.sample()
        mask = torch.sigmoid(logits)
        logp = dist.log_prob(logits).mean()
        return mask, logits, logp, self.value(z)

    def evaluate_actions(self, z: torch.Tensor, logits: torch.Tensor):
        """Recompute (log-prob, entropy) for stored ``logits`` under the policy.

        ``logits`` may carry a leading batch dim ``(B, N, 4)``; the Gaussian
        mean ``(N, 4)`` broadcasts, and the log-prob is averaged over the
        ``N x 4`` action dims -> ``(B,)``. Entropy is the per-distribution
        constant averaged over action dims (scalar).
        """
        dist = self._dist(z)
        logp = dist.log_prob(logits).mean(dim=(-2, -1))
        entropy = dist.entropy().mean()
        return logp, entropy
