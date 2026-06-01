"""CLIP image featurizer for candidate multi-view renders (ARCHITECT.md sec. 2.4).

The Bradley-Terry reward model scores a deformation from its *appearance*, not
its raw geometry, so each candidate is represented by the CLIP ViT-B/32 image
embedding of its multi-view renders, mean-pooled across views and L2-normalized.
Embeddings are cached to disk (one ``torch.save`` dict keyed by candidate_id) so
the one-time CLIP forward pass is not repeated across training runs / the Phase 4
PPO loop.

Notes:
* ``use_safetensors=True`` is required -- transformers refuses to ``torch.load``
  ``.bin`` weights on torch < 2.6 (CVE-2025-32434); the CLIP repo ships a
  safetensors checkpoint.
* NaN guard on the emitted features (CLAUDE.md Logging Protocols).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.logging_utils import get_logger
from .candidates import Manifest

log = get_logger("hitl.features")

CLIP_NAME = "openai/clip-vit-base-patch32"


class CLIPFeaturizer:
    """Wraps CLIP ViT-B/32 image-feature extraction."""

    def __init__(self, device: torch.device):
        from transformers import CLIPModel, CLIPProcessor  # lazy: heavy import

        self.device = device
        self.model = CLIPModel.from_pretrained(CLIP_NAME, use_safetensors=True).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(CLIP_NAME)
        self.dim = int(self.model.config.projection_dim)
        log.info("CLIP %s loaded on %s | proj_dim=%d", CLIP_NAME, device, self.dim)

    @torch.no_grad()
    def embed(self, images: list[np.ndarray]) -> torch.Tensor:
        """Embed a list of HxWx3 uint8 views -> ``(V, dim)`` L2-normalized (CPU)."""
        pil = [Image.fromarray(np.ascontiguousarray(im)).convert("RGB") for im in images]
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        out = self.model.get_image_features(**inputs)
        # transformers >=5 may wrap the result; unwrap to the projected embedding.
        if isinstance(out, torch.Tensor):
            feats = out
        elif getattr(out, "image_embeds", None) is not None:
            feats = out.image_embeds
        else:
            feats = out.pooler_output
        feats = torch.nn.functional.normalize(feats, dim=-1)
        if not torch.isfinite(feats).all():
            raise ValueError("NaN/Inf detected in CLIP image features")
        return feats.cpu()


def _view_paths(contact: Path) -> list[Path]:
    """Per-view PNGs accompanying a ``*_contact.png`` sheet (fallback: the sheet)."""
    suffix = "_contact.png"
    stem = contact.name[:-len(suffix)] if contact.name.endswith(suffix) else contact.stem
    views = sorted(contact.parent.glob(f"{stem}_view*.png"))
    return views or [contact]


def candidate_features(
    manifest: Manifest,
    root: Path,
    device: torch.device,
    cache_path: Path | None = None,
) -> dict[str, torch.Tensor]:
    """Map ``candidate_id -> (dim,)`` mean-pooled CLIP feature; cached to disk."""
    if cache_path is not None and Path(cache_path).exists():
        cached = torch.load(cache_path, weights_only=False)
        log.info("loaded cached CLIP features for %d candidates <- %s",
                 len(cached), cache_path)
        return cached

    fz = CLIPFeaturizer(device)
    feats: dict[str, torch.Tensor] = {}
    for c in manifest.candidates:
        contact = root / c.render
        imgs = [np.asarray(Image.open(p).convert("RGB")) for p in _view_paths(contact)]
        emb = fz.embed(imgs)                       # (V, dim)
        feats[c.candidate_id] = emb.mean(dim=0)    # (dim,)
        log.info("  embedded %s | %d view(s) -> feat[%d]",
                 c.candidate_id, len(imgs), feats[c.candidate_id].shape[0])

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(feats, cache_path)
        log.info("cached CLIP features for %d candidates -> %s", len(feats), cache_path)
    return feats
