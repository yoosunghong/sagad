# Project Master Plan & Roadmap

This document serves as the single source of truth for the project lifecycle. Refer to this checklist before executing any tasks, and update it immediately upon phase completion. Detailed execution logs must be documented in the `DONE/` directory.

## Phase 1: Environment Setup & Geometric Embedding Baseline ✅
- [x] Initialize repository structure and configure development environments (PyTorch, PyG, UE5).
- [x] Implement 3D mesh data pipelines (loading, normalization, and graph conversion using Trimesh/PyTorch Geometric).
- [x] Implement or integrate a Mesh Masked Autoencoder (MeshMAE) for self-supervised structural embedding.
- [x] Validate latent space representations of initial organic assets (Trees, Rocks).

## Phase 2: Deformation Sandbox & Heuristic Reward Engineering ✅
- [x] Develop the 3D deformation simulator applying masks to 4 channels: Bend (R), Noise (G), Scale (B), Fixed (A).
- [x] Formulate deterministic geometry & physics penalty functions (Laplacian smoothness, normal consistency, ground contact).
- [x] Build a multi-angle rendering pipeline to export synthetic views of deformed assets.
- [x] Evaluate a heuristic-driven baseline model without human feedback.

## Phase 3: RLHF Framework & Reward Model Training ▢
- [x] Design a lightweight Human-in-the-Loop (HITL) annotation interface (Gradio/Streamlit) for pairwise comparison.
- [x] Collect initial artist preference data (Asset A vs. Asset B) for naturalness and artistic utility. *(designer annotated all 75 pairs; quality is weak — see note — so the UI was upgraded to interactive 3D meshes for a cleaner re-annotation round.)*
- [x] Train a Bradley-Terry preference reward model based on the collected feedback. *(trained on the 73 real comparisons: held-out pairwise acc 0.73.)*
- [x] Validate reward model correlation with human qualitative scores. *(held-out acc 0.73, but ≈0 correlation with the geometry heuristic — consistent with the designer's report that the old 2D grayscale UI made pairs hard to judge; motivates the 3D re-annotation.)*

## Phase 4: PPO Optimization Loop Integration ▢
- [x] Assemble the full Reinforcement Learning environment using the OpenAI Gym/Gymnasium interface (`src/rl/env.py`).
- [x] Configure the PPO (Proximal Policy Optimization) actor-critic networks to predict continuous vertex-level masks (`src/rl/policy.py`).
- [x] Integrate the composite reward function: $R = \alpha R_{human} + \beta R_{variety} - \gamma R_{distortion} - \delta R_{physics}$ (`src/rl/composite.py`; `R_human` render+CLIP path verified).
- [x] Run optimization loops and monitor convergence using Weights & Biases (WandB) (`scripts/train_ppo.py`).

*Machinery complete and convergent (reward −0.847 → −0.018, stable KL, NaN-free). Follow-up tuning remains: the free-form α=0 policy does not yet beat the hand-structured heuristic baseline (+0.216), and `R_human` needs scale calibration before it enters the composite at α>0.*

## Phase 5: UE5 Production Pipeline & Empirical Validation ▢
- [ ] Build an automation script to bake learned deformation masks into Static Mesh Vertex Colors (RGBA).
- [ ] Author custom Unreal Engine 5 Materials utilizing World Position Offset (WPO) driven by Vertex Color channels.
- [ ] Set up an Hierarchical Instanced Static Mesh (HISM) stress test scene in UE5.
- [ ] Profile performance metrics (FPS, Draw Calls, VRAM, Shader Instructions) and document asset efficiency.
