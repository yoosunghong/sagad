"""MeshMAE -- a GATv2-backed Mesh Masked Autoencoder.

Implements the self-supervised structural-embedding stage from
``docs/ARCHITECT.md`` section 2.1: "Graph Attention Networks (GATv2) stacked
on top of a self-supervised MeshMAE baseline", emitting localized node
embeddings ``Z in R^{N x D}``.

Self-supervision strategy (GraphMAE-style masked feature reconstruction
adapted to mesh graphs):

1. A random subset of nodes (``mask_ratio``) has its input feature
   ``x = [pos_xyz, normal_xyz] in R^{N x 6}`` overwritten with a learnable
   ``[MASK]`` token.
2. The full graph (visible + masked tokens) is encoded by a GATv2 stack into
   latent embeddings ``Z in R^{N x D}``.
3. Embeddings at the masked nodes are re-masked with a second learnable token
   ("re-mask" trick) and a lighter GATv2 decoder reconstructs the original
   per-node features only at the masked positions.
4. The objective is the scaled cosine error (SCE) between the reconstructed
   and original features at masked nodes -- robust to the differing scales of
   the position vs. normal sub-vectors.

After (or without) pre-training, :meth:`MeshMAE.encode` returns the
unmasked node embeddings consumed downstream by the PPO actor policy.

Telemetry / NaN guards follow CLAUDE.md (Logging Protocols): every forward
pass logs the realized mask ratio and reconstruction loss, and asserts that
latents/loss stay finite (guarding the continuous PPO output path upstream).

Optimization note (CLAUDE.md): operates on the sparse COO ``edge_index``; the
dense ``N x N`` adjacency is never materialized.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

from data.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class MeshMAEConfig:
    """Hyper-parameters for :class:`MeshMAE`.

    Defaults target the baseline organic assets (single-mesh graphs of a few
    thousand vertices) on a single RTX 4060 Ti.
    """

    in_dim: int = 6          # node feature width: [pos_xyz, normal_xyz]
    hidden_dim: int = 128    # per-layer width (total across heads)
    latent_dim: int = 64     # D -- structural embedding width Z in R^{N x D}
    encoder_layers: int = 3
    decoder_layers: int = 2
    heads: int = 4           # GATv2 attention heads (hidden split across heads)
    mask_ratio: float = 0.5  # fraction of nodes masked during pre-training
    sce_gamma: float = 2.0   # scaled-cosine-error sharpening exponent
    dropout: float = 0.0
    feat_dims: tuple[int, ...] = field(default=(3, 3))  # pos | normal split


@dataclass
class MeshMAEOutput:
    """Container for a masked-reconstruction forward pass."""

    loss: torch.Tensor          # scalar SCE loss over masked nodes
    latent: torch.Tensor        # Z in R^{N x D} (masked-input encoding)
    mask: torch.Tensor          # bool (N,) -- True where the node was masked
    recon: torch.Tensor         # reconstructed features at masked nodes


def _gat_stack(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int,
    heads: int,
    dropout: float,
) -> nn.ModuleList:
    """Build a GATv2 stack: concat-heads in hidden layers, mean-heads on output.

    Hidden layers keep ``hidden_dim`` total channels by giving each of ``heads``
    heads ``hidden_dim // heads`` channels and concatenating. The final layer
    averages heads to land exactly on ``out_dim``.
    """
    if hidden_dim % heads != 0:
        raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")
    per_head = hidden_dim // heads

    layers = nn.ModuleList()
    if num_layers == 1:
        layers.append(GATv2Conv(in_dim, out_dim, heads=heads, concat=False, dropout=dropout))
        return layers

    layers.append(GATv2Conv(in_dim, per_head, heads=heads, concat=True, dropout=dropout))
    for _ in range(num_layers - 2):
        layers.append(GATv2Conv(hidden_dim, per_head, heads=heads, concat=True, dropout=dropout))
    layers.append(GATv2Conv(hidden_dim, out_dim, heads=heads, concat=False, dropout=dropout))
    return layers


class MeshMAE(nn.Module):
    """GATv2 Mesh Masked Autoencoder for self-supervised vertex embeddings."""

    def __init__(self, config: MeshMAEConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or MeshMAEConfig()
        cfg = self.cfg

        if sum(cfg.feat_dims) != cfg.in_dim:
            raise ValueError(
                f"feat_dims {cfg.feat_dims} must sum to in_dim ({cfg.in_dim})"
            )
        if not 0.0 < cfg.mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {cfg.mask_ratio}")

        self.encoder = _gat_stack(
            cfg.in_dim, cfg.hidden_dim, cfg.latent_dim,
            cfg.encoder_layers, cfg.heads, cfg.dropout,
        )
        # Project latent back up before the decoder GAT stack.
        self.enc_to_dec = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.decoder = _gat_stack(
            cfg.hidden_dim, cfg.hidden_dim, cfg.in_dim,
            cfg.decoder_layers, cfg.heads, cfg.dropout,
        )

        # Learnable tokens: input-space [MASK] and latent-space re-mask.
        self.mask_token = nn.Parameter(torch.zeros(cfg.in_dim))
        self.dec_mask_token = nn.Parameter(torch.zeros(cfg.hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.dec_mask_token, std=0.02)

    # -- core graph passes -------------------------------------------------

    def _run_stack(self, stack: nn.ModuleList, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(stack):
            x = conv(x, edge_index)
            if i < len(stack) - 1:
                x = F.elu(x)
        return x

    def encode(self, data: Data) -> torch.Tensor:
        """Return unmasked node embeddings ``Z in R^{N x D}`` (inference path)."""
        z = self._run_stack(self.encoder, data.x, data.edge_index)
        if not torch.isfinite(z).all():
            raise ValueError("NaN/Inf detected in MeshMAE latent embedding Z")
        return z

    # -- self-supervised masked reconstruction -----------------------------

    def _sample_mask(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        """Random boolean mask selecting ``mask_ratio`` of nodes (>=1 node)."""
        k = max(1, int(round(self.cfg.mask_ratio * num_nodes)))
        perm = torch.randperm(num_nodes, device=device)
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        mask[perm[:k]] = True
        return mask

    def _sce_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Scaled cosine error: mean (1 - cos)^gamma over the two sub-vectors.

        ``pos`` and ``normal`` live on different scales, so cosine similarity is
        computed per sub-vector (see :attr:`MeshMAEConfig.feat_dims`) and
        averaged -- this avoids the larger-magnitude block dominating the loss.
        """
        offset = 0
        terms = []
        for dim in self.cfg.feat_dims:
            r = recon[:, offset:offset + dim]
            t = target[:, offset:offset + dim]
            cos = F.cosine_similarity(r, t, dim=-1, eps=1e-8)
            terms.append((1.0 - cos).pow(self.cfg.sce_gamma))
            offset += dim
        return torch.stack(terms, dim=0).mean()

    def forward(self, data: Data, mask: torch.Tensor | None = None) -> MeshMAEOutput:
        """Masked-feature reconstruction forward pass (training path)."""
        x, edge_index = data.x, data.edge_index
        num_nodes = x.size(0)
        if mask is None:
            mask = self._sample_mask(num_nodes, x.device)

        # 1. Overwrite masked node features with the [MASK] token.
        x_masked = x.clone()
        x_masked[mask] = self.mask_token
        if not torch.isfinite(x_masked).all():
            raise ValueError("NaN/Inf detected in masked input features")

        # 2. Encode the corrupted graph.
        latent = self._run_stack(self.encoder, x_masked, edge_index)

        # 3. Re-mask the latents at masked positions, then decode.
        h = self.enc_to_dec(latent)
        h[mask] = self.dec_mask_token
        recon_full = self._run_stack(self.decoder, h, edge_index)

        # 4. SCE loss on masked nodes only.
        recon = recon_full[mask]
        target = x[mask]
        loss = self._sce_loss(recon, target)

        if not torch.isfinite(loss):
            raise ValueError("NaN/Inf detected in MeshMAE reconstruction loss")

        # Per-forward telemetry at DEBUG so the training loop's per-epoch INFO
        # stays readable; the NaN guards above are the hard safety net.
        log.debug(
            "meshmae forward | nodes=%d masked=%d (%.1f%%) latent_dim=%d sce_loss=%.6f",
            num_nodes, int(mask.sum()), 100.0 * float(mask.float().mean()),
            latent.size(1), float(loss),
        )
        return MeshMAEOutput(loss=loss, latent=latent, mask=mask, recon=recon)
