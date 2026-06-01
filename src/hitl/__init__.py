"""Human-in-the-Loop (HITL) pairwise-comparison annotation.

Isolated module for Phase 3's artist-preference collection (CLAUDE.md
modularity rule): the data graphs (`src/data`), deformation sandbox
(`src/deform`), and deterministic reward constraints (`src/rewards`) stay
strictly separate from this human-feedback layer.

Pipeline (docs/PLAN.md Phase 3, docs/ARCHITECT.md sec. 2.4 Human Preference
Module):

* :mod:`candidates` -- generate a pool of geometry-structured deformations per
  asset, render each to a contact sheet, and emit a pairing schedule manifest.
* :mod:`store`      -- append-only JSONL preference store recording artist
  pairwise judgements (winner/loser), resumable and ready for the Bradley-Terry
  reward-model trainer (next Phase 3 task).
* :mod:`app`        -- a lightweight Gradio interface presenting two candidates
  side by side for naturalness/utility comparison.
"""

from .candidates import (
    CandidateSpec,
    Manifest,
    PairSpec,
    build_candidate_pool,
    build_pairs,
    load_manifest,
    mask_from_gains,
    save_manifest,
)
from .store import PreferenceRecord, PreferenceStore

__all__ = [
    "CandidateSpec",
    "PairSpec",
    "Manifest",
    "build_candidate_pool",
    "build_pairs",
    "load_manifest",
    "mask_from_gains",
    "save_manifest",
    "PreferenceRecord",
    "PreferenceStore",
]
