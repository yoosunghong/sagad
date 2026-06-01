"""Candidate-pool generation and pairing schedule for HITL annotation.

For each asset we sample a pool of geometry-structured, base-locked candidate
deformations (varying per-channel gains + noise seed), deform the mesh with the
Phase 2 sandbox, render a multi-view contact sheet, and record a manifest entry.
A within-asset pairing schedule is then sampled -- cross-asset pairs are
meaningless for a *naturalness* comparison, so every pair shares one base mesh.

The manifest (candidate specs + pairs + render paths) is the single artefact the
:mod:`app` annotation UI and the downstream Bradley-Terry trainer consume.

CLAUDE.md compliance:
* Vectorized torch sandbox ops only (no per-vertex Python loops, no dense N x N).
* Telemetry: per-candidate vertex-degradation scales (disp / edge-length change)
  are logged and persisted; `isfinite` is enforced inside the sandbox.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import trimesh

from data.logging_utils import get_logger
from deform import DeformConfig, apply_masks
from render import RenderConfig, render_multiview, save_views

log = get_logger("hitl.candidates")


@dataclass
class CandidateSpec:
    """One generated deformation: its parameters, render, and telemetry."""

    candidate_id: str            # e.g. "tree_2#03"
    asset: str
    gains: dict                  # {bend, noise, scale, base_band}
    noise_seed: int
    render: str                  # path to the contact-sheet PNG (relative to root)
    mesh: str                    # path to the deformed GLB for the 3D viewer
    telemetry: dict = field(default_factory=dict)


@dataclass
class PairSpec:
    """A scheduled A-vs-B comparison (always within one asset)."""

    pair_id: str                 # stable id, used to resume / de-dupe annotation
    asset: str
    a: str                       # candidate_id of the left option
    b: str                       # candidate_id of the right option


@dataclass
class Manifest:
    """Full annotation job: candidate pool + pairing schedule."""

    render_dir: str
    seed: int
    candidates: list[CandidateSpec]
    pairs: list[PairSpec]
    originals: dict = field(default_factory=dict)   # asset -> original GLB path

    def candidate_map(self) -> dict[str, CandidateSpec]:
        return {c.candidate_id: c for c in self.candidates}


def geometry_structured_mask(
    data, rng: np.random.Generator
) -> tuple[torch.Tensor, dict, int]:
    """A base-locked, height-weighted RGBA mask with randomized channel gains.

    Mirrors the heuristic baseline's candidate generator (height ramp + pinned
    base band) but with wider gain ranges and a jittered base band so the pool
    spans a range an artist can actually discriminate on naturalness. Returns
    ``(masks[N, 4], gains, noise_seed)``.
    """
    pos = data.pos
    up = torch.tensor([0.0, 0.0, 1.0], dtype=pos.dtype, device=pos.device)
    h = pos @ up
    h_norm = (h - h.min()) / (h.max() - h.min()).clamp(min=1e-8)

    base_band = float(rng.uniform(0.15, 0.35))   # fraction of height pinned at base
    floor = base_band * 0.4
    fixed = ((base_band - h_norm) / max(base_band - floor, 1e-8)).clamp(0.0, 1.0)

    g_bend = float(rng.uniform(0.0, 1.2))
    g_noise = float(rng.uniform(0.0, 1.0))
    g_scale = float(rng.uniform(0.0, 0.8))
    bend = (g_bend * h_norm).clamp(0.0, 1.0)
    noise = torch.full_like(h_norm, g_noise)
    scale = torch.full_like(h_norm, g_scale)

    seed = int(rng.integers(0, 1 << 30))
    masks = torch.stack([bend, noise, scale, fixed], dim=1)
    gains = {
        "bend": round(g_bend, 4),
        "noise": round(g_noise, 4),
        "scale": round(g_scale, 4),
        "base_band": round(base_band, 4),
    }
    return masks, gains, seed


def _rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


# Matte stone-tan material so the viewer shades the surface form instead of the
# washed-out light-gray default (an untextured GLB + bright default material).
_VIEWER_COLOR = np.array([165, 150, 130, 255], dtype=np.uint8)


def _export_viewer_mesh(mesh: trimesh.Trimesh, path: Path) -> None:
    """Export a Y-up, matte-shaded GLB copy of the mesh for the Gradio 3D viewer.

    The pipeline is Z-up (height along +Z) but three.js / `gr.Model3D` is Y-up,
    so rotate -90 deg about X to make assets stand upright by default (the
    annotator still orbits freely). A low-metallic / high-roughness PBR material
    with a uniform base colour is attached so the surface reads as matte stone
    and the lighting reveals the deformation (not a flat light-gray blob). GLB is
    self-contained and renders cleanly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    m = mesh.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2.0, [1.0, 0.0, 0.0]))
    m.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=_VIEWER_COLOR, metallicFactor=0.0, roughnessFactor=0.9,
        )
    )
    m.vertex_normals  # force normal computation so the GLB ships smooth normals
    m.export(path)


def export_original_mesh(asset: str, graphs_dir: Path, mesh_dir: Path,
                         root: Path, device: torch.device) -> str:
    """Export the undeformed asset as a viewer GLB (the comparison reference)."""
    data = torch.load(graphs_dir / f"{asset}.pt", weights_only=False).to(device)
    mesh = trimesh.Trimesh(vertices=data.pos.detach().cpu().numpy(),
                           faces=data.face.t().cpu().numpy(), process=False)
    glb = mesh_dir / f"{asset}_original.glb"
    _export_viewer_mesh(mesh, glb)
    log.info("original reference mesh | %s -> %s", asset, glb.name)
    return _rel(glb, root)


def mask_from_gains(data, gains: dict) -> torch.Tensor:
    """Rebuild a candidate's ``[N, 4]`` mask deterministically from stored gains.

    The inverse of :func:`geometry_structured_mask` (sans RNG): given the gains
    persisted in the manifest it reproduces the *exact* mask that was deformed,
    so the synthetic-oracle / re-scoring path matches the rendered candidate.
    """
    pos = data.pos
    up = torch.tensor([0.0, 0.0, 1.0], dtype=pos.dtype, device=pos.device)
    h = pos @ up
    h_norm = (h - h.min()) / (h.max() - h.min()).clamp(min=1e-8)

    base_band = float(gains["base_band"])
    floor = base_band * 0.4
    fixed = ((base_band - h_norm) / max(base_band - floor, 1e-8)).clamp(0.0, 1.0)
    bend = (float(gains["bend"]) * h_norm).clamp(0.0, 1.0)
    noise = torch.full_like(h_norm, float(gains["noise"]))
    scale = torch.full_like(h_norm, float(gains["scale"]))
    return torch.stack([bend, noise, scale, fixed], dim=1)


def build_candidate_pool(
    asset: str,
    graphs_dir: Path,
    render_dir: Path,
    n_candidates: int,
    rng: np.random.Generator,
    device: torch.device,
    mesh_dir: Path | None = None,
    render_cfg: RenderConfig | None = None,
    root: Path | None = None,
) -> list[CandidateSpec]:
    """Generate, render, and export-as-GLB ``n_candidates`` deformations."""
    render_cfg = render_cfg or RenderConfig()
    root = root or Path.cwd()
    mesh_dir = mesh_dir or (render_dir.parent / "meshes")
    data = torch.load(graphs_dir / f"{asset}.pt", weights_only=False).to(device)
    log.info("=== %s | N=%d | generating %d candidates ===",
             asset, data.num_nodes, n_candidates)

    specs: list[CandidateSpec] = []
    for k in range(n_candidates):
        masks, gains, seed = geometry_structured_mask(data, rng)
        result = apply_masks(data, masks, DeformConfig(noise_seed=seed))

        stem = f"{asset}_cand{k:02d}"
        views = render_multiview(result.mesh, render_cfg)
        paths = save_views(views, render_dir, stem)
        sheet = next((p for p in paths if p.name.endswith("_contact.png")), paths[0])

        glb = mesh_dir / f"{stem}.glb"
        _export_viewer_mesh(result.mesh, glb)

        spec = CandidateSpec(
            candidate_id=f"{asset}#{k:02d}",
            asset=asset,
            gains=gains,
            noise_seed=seed,
            render=_rel(sheet, root),
            mesh=_rel(glb, root),
            telemetry={
                "disp_mean": result.telemetry["disp_mean"],
                "disp_max": result.telemetry["disp_max"],
                "locked_fraction": result.telemetry["locked_fraction"],
                "edge_len_rel_change_max": result.telemetry.get("edge_len_rel_change_max"),
            },
        )
        specs.append(spec)
        log.info("  %s | gains=%s seed=%d disp_mean=%.4f edge_max=%.4f",
                 spec.candidate_id, gains, seed, spec.telemetry["disp_mean"],
                 spec.telemetry["edge_len_rel_change_max"] or float("nan"))
    return specs


def build_pairs(
    specs: list[CandidateSpec],
    rng: np.random.Generator,
    n_pairs_per_asset: int | None = None,
) -> list[PairSpec]:
    """Sample within-asset unordered comparison pairs.

    With ``n_pairs_per_asset=None`` the full round-robin (every distinct pair)
    is used; otherwise that many distinct pairs are sampled per asset. The A/B
    side is randomized to avoid a positional bias in the collected preferences.
    """
    pairs: list[PairSpec] = []
    by_asset: dict[str, list[CandidateSpec]] = {}
    for s in specs:
        by_asset.setdefault(s.asset, []).append(s)

    for asset, group in by_asset.items():
        all_pairs = list(itertools.combinations(range(len(group)), 2))
        if not all_pairs:
            log.warning("asset %s has <2 candidates; no pairs generated", asset)
            continue
        if n_pairs_per_asset is not None and n_pairs_per_asset < len(all_pairs):
            idx = rng.choice(len(all_pairs), size=n_pairs_per_asset, replace=False)
            chosen = [all_pairs[i] for i in idx]
        else:
            chosen = all_pairs

        for n, (i, j) in enumerate(chosen):
            if rng.random() < 0.5:       # randomize left/right placement
                i, j = j, i
            pairs.append(PairSpec(
                pair_id=f"{asset}|{n:03d}",
                asset=asset,
                a=group[i].candidate_id,
                b=group[j].candidate_id,
            ))
    log.info("built %d comparison pairs across %d asset(s)", len(pairs), len(by_asset))
    return pairs


def save_manifest(manifest: Manifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "render_dir": manifest.render_dir,
        "seed": manifest.seed,
        "originals": manifest.originals,
        "candidates": [asdict(c) for c in manifest.candidates],
        "pairs": [asdict(p) for p in manifest.pairs],
    }
    path.write_text(json.dumps(payload, indent=2))
    log.info("manifest -> %s (%d candidates, %d pairs)",
             path, len(manifest.candidates), len(manifest.pairs))


def load_manifest(path: Path) -> Manifest:
    payload = json.loads(Path(path).read_text())
    return Manifest(
        render_dir=payload["render_dir"],
        seed=payload["seed"],
        originals=payload.get("originals", {}),
        candidates=[CandidateSpec(**c) for c in payload["candidates"]],
        pairs=[PairSpec(**p) for p in payload["pairs"]],
    )
