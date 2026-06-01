"""Low-dimensional DDPM over the per-instance variant vector (ARCHITECT sec. 2.5).

The goal is *many plausible variants* -- a distribution to sample, not the single
reward-optimal mask a policy converges to -- so the variation engine is a denoising
diffusion model rather than the §2.2 PPO actor. Operating space is the standardized
``[-1, 1]`` model space of :class:`~diffusion.params.VariantParamSpec`; the sampled
vectors decode into Per-Instance Custom Data + §2.3 mask gains.

Standard epsilon-prediction DDPM (Ho et al. 2020) with a linear beta schedule, over
a tiny MLP denoiser -- the parameter vector is O(4-16) dims, so this is cheap to
train and sample. Optional conditioning on the pooled MeshMAE asset latent makes the
sampler asset-aware. NaN guards on the loss and every sampled batch (CLAUDE.md).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.logging_utils import get_logger

log = get_logger(__name__)


def _linear_beta_schedule(timesteps: int, beta_start: float = 1e-4,
                          beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


class SinusoidalTimeEmbedding(nn.Module):
    """Transformer-style sinusoidal embedding of the integer diffusion step."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class DiffusionMLP(nn.Module):
    """Epsilon predictor ``eps_theta(x_t, t, [cond])`` over the variant vector."""

    def __init__(self, dim: int, cond_dim: int = 0, hidden: int = 128,
                 time_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.time = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.Linear(dim + time_dim + cond_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor | None = None) -> torch.Tensor:
        h = [x, self.time(t)]
        if self.cond_dim:
            if cond is None:
                raise ValueError("model was built with cond_dim>0 but cond is None")
            h.append(cond.expand(x.shape[0], -1) if cond.shape[0] == 1 else cond)
        return self.net(torch.cat(h, dim=1))


class GaussianDiffusion(nn.Module):
    """DDPM wrapper holding the noise schedule + train/sample loops.

    An ``nn.Module`` so the schedule rides along as buffers under ``.to(device)``;
    ``forward`` returns the training loss.
    """

    def __init__(self, model: DiffusionMLP, dim: int, timesteps: int = 100):
        super().__init__()
        self.model = model
        self.dim = dim
        self.timesteps = timesteps
        betas = _linear_beta_schedule(timesteps)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", acp)
        self.register_buffer("sqrt_acp", torch.sqrt(acp))
        self.register_buffer("sqrt_one_minus_acp", torch.sqrt(1.0 - acp))

    @property
    def device(self) -> torch.device:
        return self.betas.device

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: sample ``x_t ~ q(x_t | x0)``."""
        return (self.sqrt_acp[t].unsqueeze(1) * x0
                + self.sqrt_one_minus_acp[t].unsqueeze(1) * noise)

    def forward(self, x0: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Epsilon-prediction MSE loss over a random timestep per sample."""
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps = self.model(x_t, t, cond)
        loss = F.mse_loss(eps, noise)
        if not torch.isfinite(loss):
            raise ValueError("NaN/Inf in diffusion training loss")
        return loss

    @torch.no_grad()
    def sample(self, n: int, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Ancestral DDPM sampling: returns ``(n, dim)`` vectors in model space."""
        x = torch.randn(n, self.dim, device=self.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            eps = self.model(x, t, cond)
            beta = self.betas[i]
            alpha = self.alphas[i]
            sqrt_1m_acp = self.sqrt_one_minus_acp[i]
            mean = (x - beta / sqrt_1m_acp * eps) / torch.sqrt(alpha)
            if i > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x)
            else:
                x = mean
        if not torch.isfinite(x).all():
            raise ValueError("NaN/Inf in diffusion sample")
        return x


def train_diffusion(diffusion: GaussianDiffusion, data: torch.Tensor,
                    cond: torch.Tensor | None = None, epochs: int = 2000,
                    lr: float = 1e-3, batch_size: int = 128,
                    log_every: int = 500) -> list[float]:
    """Adam-train the denoiser on model-space samples ``data`` ``(M, dim)``.

    ``cond`` (optional, ``(M, cond_dim)``) is the per-sample conditioning. Returns
    the per-logged-epoch loss trace.
    """
    diffusion.train()
    opt = torch.optim.Adam(diffusion.parameters(), lr=lr)
    m = data.shape[0]
    losses: list[float] = []
    for epoch in range(epochs):
        idx = torch.randint(0, m, (min(batch_size, m),), device=data.device)
        loss = diffusion(data[idx], None if cond is None else cond[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
        opt.step()
        if epoch % log_every == 0 or epoch == epochs - 1:
            losses.append(float(loss))
            log.info("diffusion | epoch=%d loss=%.5f", epoch, float(loss))
    return losses
