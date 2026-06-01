# sagad — Self-supervised Automatic Generation of Asset Deformation Masks

**sagad** automates the creation of deformation masks for 3D game assets. Instead of storing hundreds of unique mesh variants, a single base mesh paired with learned RGBA vertex-color masks drives a World Position Offset (WPO) shader to produce unlimited natural-looking instances at runtime — with zero additional mesh memory.

The system combines self-supervised graph neural network embeddings with reinforcement learning from human feedback (RLHF) to discover *where*, *what type*, and *how much* deformation preserves the geometric and aesthetic naturalness that artists expect.

---

## How It Works

```
Raw Mesh (OBJ/FBX/GLB)
        │
        ▼
  MeshMAE Encoder          ← GATv2 self-supervised pre-training
  (Node embeddings Z)
        │
        ▼
  PPO Policy               ← Trained with composite reward
  (RGBA mask M ∈ [0,1]^N)
        │
        ▼
  Deformation Sandbox      ← Applies bend / noise / scale offsets
        │
        ├── Heuristic rewards   (Laplacian smoothness, normal consistency,
        │                         ground contact, edge stretch)
        └── Preference rewards  ← Bradley-Terry model trained on artist A/B pairs
                                    (CLIP ViT-B/32 embeddings of multi-view renders)
        │
        ▼
  DDPM Variant Sampler     ← Samples per-instance gain vectors from low-dim diffusion
        │
        ▼
  Vertex Color Baking      ← Seam-coherent RGBA → GLB export
        │
        ▼
  Unreal Engine 5          ← WPO material + HISM scatter tool
  (K instances, 1 mesh,       → K distinct appearances, GPU-driven
   Per-Instance Custom Data)
```

### Vertex Color Channel Convention

| Channel | Mask Type | Deformation Applied |
|---------|-----------|---------------------|
| R | Bend | Height-weighted Rodrigues rotation |
| G | Noise | Normal-displaced Perlin perturbation |
| B | Scale | Radial outward / inward scaling |
| A | Mobility | Ground lock (0 = fixed, 1 = free) |

---

## Project Structure

```
sagad/
├── src/
│   ├── models/mesh_mae.py        # GATv2 MeshMAE encoder
│   ├── rl/
│   │   ├── env.py                # Gymnasium DeformEnv (single-step bandit)
│   │   ├── policy.py             # MaskActorCritic (per-node weight-shared MLP)
│   │   └── composite.py          # R = αR_human + βR_variety − γR_distortion − δR_physics
│   ├── deform/
│   │   ├── operators.py          # Bend, Noise, Scale offset kernels
│   │   └── sandbox.py            # Deformation simulator + multi-view renderer
│   ├── diffusion/
│   │   ├── params.py             # VariantParamSpec schema
│   │   └── ddpm.py               # Asset-conditioned low-dim DDPM sampler
│   ├── hitl/
│   │   ├── app.py                # Gradio annotation UI
│   │   ├── candidates.py         # Candidate generation + contact sheets
│   │   ├── features.py           # CLIP ViT-B/32 featurizer
│   │   └── store.py              # Append-only JSONL preference store
│   ├── rewards/
│   │   ├── heuristic.py          # Deterministic geometry-driven baseline
│   │   ├── penalties.py          # Laplacian, normal, contact, edge-stretch
│   │   └── preference.py         # Bradley-Terry reward model
│   └── render/multiview.py       # Orthogonal multi-view rendering
├── scripts/
│   ├── train_meshmae.py          # Pre-train MeshMAE encoder
│   ├── train_ppo.py              # Main RL training loop
│   ├── train_reward_model.py     # Fit Bradley-Terry model on annotations
│   ├── build_preference_pairs.py # Generate candidate meshes for labelling
│   ├── annotate_ui.py            # Serve Gradio annotation interface
│   ├── build_variant_dataset.py  # Bootstrap DDPM training data
│   ├── sample_variants.py        # Sample K variant vectors from DDPM
│   ├── bake_vertex_colors.py     # Bake masks → seam-coherent RGBA GLB
│   └── simulate_preferences.py   # Synthetic oracle labels for offline testing
├── ue/
│   ├── sagad_ue.uproject
│   └── Plugins/sagad/
│       ├── Source/sagad/
│       │   ├── Public/
│       │   │   ├── SagadScatterActor.h   # HISM placement + Per-Instance Custom Data
│       │   │   └── SagadBridge.h         # Python IPC bridge for runtime sampling
│       │   └── Private/
│       │       ├── SagadScatterActor.cpp
│       │       └── SagadBridge.cpp
│       └── Content/Python/build_wpo_material.py
├── Docs/
│   ├── PLAN.md                   # Phase roadmap
│   ├── ARCHITECT.md              # Tensor flows, reward math, channel maps
│   ├── PROPOSAL.md               # Research proposal
│   └── DONE/                     # Phase completion logs
├── meshes/baseline/              # Reference organic assets (rocks, trees)
├── tests/
│   └── test_diffusion.py
└── requirements.txt
```

---

## Development Phases

| Phase | Status | Deliverable |
|-------|--------|-------------|
| 1 | ✅ | Environment setup, MeshMAE baseline, PyG graph pipeline |
| 2 | ✅ | Deformation sandbox, heuristic rewards, multi-view rendering |
| 3 | ✅ | HITL annotation UI, Bradley-Terry reward model (0.73 held-out accuracy) |
| 4 | ✅ | PPO convergence — composite reward: −0.847 → −0.018 |
| 4.5 | ✅ | DDPM variant sampler, texture-safe rejection filtering |
| 5 | 🔶 | Vertex color baking ✅, UE5 scatter tool ✅, HISM stress test ⏳ |

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (tested on CUDA 12.1)
- Unreal Engine 5.3+ (for the runtime plugin)

### Installation

```bash
git clone https://github.com/Yoosung-H/sagad.git
cd sagad
pip install -r requirements.txt
```

### Training Pipeline

```bash
# 1. Pre-train the mesh encoder
python scripts/train_meshmae.py --mesh_dir meshes/baseline

# 2. (Optional) Collect artist preferences
python scripts/build_preference_pairs.py --mesh path/to/mesh.obj
python scripts/annotate_ui.py          # Opens Gradio UI at localhost:7860

# 3. Train the preference reward model
python scripts/train_reward_model.py

# 4. Train the PPO policy
python scripts/train_ppo.py --mesh path/to/mesh.obj

# 5. Bootstrap and train the DDPM variant sampler
python scripts/build_variant_dataset.py
python scripts/sample_variants.py --k 32

# 6. Bake vertex colors and export
python scripts/bake_vertex_colors.py --mesh path/to/mesh.obj --out out/mesh_baked.glb
```

### UE5 Plugin

1. Copy `ue/Plugins/sagad/` into your project's `Plugins/` folder.
2. Regenerate project files and build.
3. Place `BP_SagadScatterActor` in a level and point it at the baked GLB asset.
4. The scatter actor calls the Python IPC bridge at runtime to sample variant vectors and writes them as Per-Instance Custom Data.
5. Assign `M_SagadWPO` to the static mesh — the material reads vertex colors and PICD to drive WPO.

---

## Key Design Decisions

**Why vertex colors, not blend shapes or morph targets?**
Blend shapes require per-variant geometry storage (O(K·N) vertices). Vertex colors store a single mask (O(N)) and defer variation to a GPU WPO shader. At K=1000 instances, the memory savings are ~3 orders of magnitude.

**Why GATv2 instead of PointNet?**
Graph attention respects mesh connectivity, allowing the encoder to propagate structural context across neighbors — important for detecting silhouette edges, concavities, and load-bearing regions that should not deform.

**Why single-step PPO instead of multi-step MDP?**
Deformation quality is a property of the final mask, not a sequence of edits. The contextual bandit formulation (reset = frozen embedding, action = full mask) avoids credit-assignment complexity while still benefiting from PPO's clipped surrogate and KL stability.

---

## Reward Function

```
R_total = α · R_human
        + β · R_variety
        − γ · R_distortion
        − δ · R_physics

where:
  R_human     = Bradley-Terry preference score (CLIP features of renders)
  R_variety   = Mean pairwise CLIP cosine distance across policy samples
  R_distortion = Laplacian smoothness + edge-stretch penalty
  R_physics   = Normal consistency + ground-contact enforcement
```

Default weights: α=0.4, β=0.2, γ=0.2, δ=0.2 (see `Docs/ARCHITECT.md`).

---

## Dependencies

| Category | Library |
|----------|---------|
| Deep learning | PyTorch 2.5, torch-geometric 2.7, transformers 5.9 |
| Geometry | trimesh 4.12, open3d 0.19 |
| Reinforcement learning | gymnasium 1.2, stable-baselines3 2.8 |
| Human-in-the-loop UI | gradio 6.15 |
| Experiment tracking | wandb 0.27 |
| Numerical | numpy 2.2, scipy 1.15 |

---

## Documentation

- [`Docs/ARCHITECT.md`](Docs/ARCHITECT.md) — full tensor flow specs, reward math, channel map definitions
- [`Docs/PLAN.md`](Docs/PLAN.md) — phase roadmap and task checklist
- [`Docs/DONE/`](Docs/DONE/) — per-phase completion logs

---

## License

Research prototype. Contact the author before using in production or commercial contexts.
