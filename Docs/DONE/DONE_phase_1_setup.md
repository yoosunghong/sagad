# Phase 1 Progress Log: Environment Setup & Geometric Embedding

## Status: ✅ Complete

### Executed Tasks
- [x] Initialized project repository with the core structure (`PROPOSAL.md`, `ARCHITECT.md`, `STACK.md`, `CLAUDE.md`, `PLAN.md`).
- [x] Conda environment `sagad` created (Python 3.10, CUDA 12.1 / RTX 4060 Ti).
  - PyTorch 2.5.1+cu121, torch_geometric 2.7.0, pyg_lib+torch_scatter+torch_sparse+torch_cluster+torch_spline_conv (pt25cu121)
  - trimesh 4.12.2, open3d 0.19.0
  - stable-baselines3 2.8.0, gymnasium 1.2.3
  - wandb 0.27.0, transformers 5.9.0, gradio 6.15.2, streamlit 1.58.0
  - Full pinned dependency list exported to `requirements.txt`
- [x] Prepared initial asset dataset: **2 rocks (OBJ) + 3 trees (GLB)**.
  - Rocks: `meshes/baseline/gray-big-rock/` (FBX source) re-exported to OBJ at `.../untitled/untitled.obj` (7,106 verts); `.../rock-17-free-rock-pack-vol3/.../rock_17.obj` (63,817 verts).
  - Trees: `meshes/baseline/tree.glb` (20,473), `tree (1).glb` (164,135), `tree (2).glb` (2,552) — wide resolution spread across the class.
  - **Format constraint discovered:** Trimesh/Open3D cannot parse FBX natively (no assimp/blender/pyassimp in env). Baseline ingest standardized on OBJ/GLB/PLY/STL; FBX must be re-exported before ingest.
  - **GLB trees ingest without code change** — `.glb`/`.gltf` were already in `mesh_loader.SUPPORTED_SUFFIXES`; `load_mesh(force='mesh')` collapses the multi-geometry GLB scene into a single vertex/face table. All 5 graphs cached to `data/processed/*.pt` (isolated=0 for every asset).
- [x] Implement 3D mesh data pipeline (loading → normalization → PyG graph conversion).
- [x] Integrate a Mesh Masked Autoencoder (MeshMAE) for self-supervised structural embedding (`src/models/`).

### Technical Notes
* Objective is to ensure that incoming meshes are normalized regarding scale and orientation before passing into the geometric feature encoder to maintain coordinate invariance.
* **Pipeline modules (`src/data/`)** — kept single-responsibility per CLAUDE.md modularity rule:
  * `mesh_loader.py` — Trimesh load + unit-bounding-sphere normalization (recentre to centroid, isotropic scale). Returns `LoadedMesh` carrying the inverse transform (`centroid`, `scale`) for round-tripping.
  * `graph_builder.py` — Trimesh → PyG `Data`. Node features `x = [pos(3), normal(3)] ∈ R^{N×6}`, undirected COO `edge_index`, `face` table retained for later Laplacian penalties. Sparse adjacency only (no dense N×N).
  * `pipeline.py` — orchestrator; `logging_utils.py` — shared telemetry.
* **Telemetry / NaN guards (CLAUDE.md Logging Protocols):** logs vertex/face counts, centroid + scale radius, post-normalize extents, and degree distribution (min/mean/max, isolated-vertex count). Explicit `isfinite` checks abort on NaN/Inf in vertices and node features.
* **Validation:** ran `scripts/build_graph.py` on the baseline rock → graph `nodes=7106 edges=41662 feat_dim=6 degree(min/mean/max)=2/5.86/10 isolated=0`; cached to `data/processed/gray-big-rock.pt`. ARCHITECT.md §2.1 updated with the realized I/O contract.

### MeshMAE Integration (`src/models/`)
* **New isolated module** (`src/models/`, kept separate from `src/data/` per the CLAUDE.md modularity rule):
  * `mesh_mae.py` — `MeshMAE` (+ `MeshMAEConfig`, `MeshMAEOutput`). GATv2 encoder (3 layers, `hidden_dim=128`, `heads=4`) → latent `Z ∈ R^{N×D}`, `D=64`; 2-layer GATv2 decoder for reconstruction.
  * Reuses the shared `data.logging_utils` telemetry helper rather than introducing a parallel logger.
* **Self-supervision (GraphMAE-style masked feature reconstruction):** `mask_ratio` of nodes → learnable `[MASK]` token, encode corrupted graph, re-mask masked latents with a second token, decode, and reconstruct the original `[pos, normal]` features at masked nodes only.
* **Loss:** scaled cosine error (SCE), per `pos`/`normal` sub-vector then averaged — avoids the larger-magnitude block dominating. `isfinite` guards on masked inputs, latent, and loss (CLAUDE.md NaN protocol); per-forward telemetry at DEBUG, per-epoch at INFO.
* **Validation:** `scripts/train_meshmae.py` pre-trained on the baseline gray-big-rock graph (7,106 nodes), 100 epochs CUDA (RTX 4060 Ti): **SCE 1.5414 → 0.0053 (99.7% reduction)**. Latent sanity: `Z (7106, 64)`, mean≈0, std≈0.29, range [-0.93, 0.97], all finite. Weights cached to `data/processed/meshmae_baseline.pt`. ARCHITECT.md §2.1 updated with the realized encoder/objective contract.
* **Operation note:** sparse COO `edge_index` only — no dense `N×N` adjacency materialized (CLAUDE.md optimization rule).

### Latent-Space Validation (Trees vs. Rocks)
* **Joint MeshMAE pre-training** across all 5 baseline graphs (`scripts/train_meshmae.py`), 150 epochs CUDA: **SCE 1.4376 → 0.0379 (97.4% reduction)**, every per-asset latent finite. Weights re-cached to `data/processed/meshmae_baseline.pt`.
* **Separability check** (`scripts/validate_latents.py`, new isolated analysis script): each asset → per-node latents `Z ∈ R^{N×64}` → permutation-invariant graph-level descriptor (mean ‖ std pooling, vertex-count/order independent) → pairwise cosine distances split into intra- vs inter-class.
  * **Result: PASS** — mean intra-class dist `0.0381` < mean inter-class dist `0.0435` (margin `+0.0055`). Class label inferred from filename (`rock`/`tree`), overridable via `--labels`.
  * **Caveat:** thin margin, expected with only 2+3 assets and one weakly-separated cross pair (`gray-big-rock ↔ tree_2`). The validation *harness* is the Phase 1 deliverable; margin should widen as the dataset grows. Re-run the script after staging more assets.

### Next Steps (Phase 2)
* Begin the deformation sandbox: apply RGBA masks to the 4 channels — Bend (R), Noise (G), Scale (B), Fixed (A) — per ARCHITECT.md §2.3.
* The MeshMAE `encode()` latents `Z ∈ R^{N×64}` are now the structural input for the §2.2 PPO actor policy.
