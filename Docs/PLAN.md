# Project Master Plan & Roadmap

This document serves as the single source of truth for the project lifecycle. Refer to this checklist before executing any tasks, and update it immediately upon phase completion. Detailed execution logs must be documented in the `DONE/` directory.

## Phase 1: Environment Setup & Geometric Embedding Baseline ⏳
- [ ] Initialize repository structure and configure development environments (PyTorch, PyG, UE5).
- [ ] Implement 3D mesh data pipelines (loading, normalization, and graph conversion using Trimesh/PyTorch Geometric).
- [ ] Implement or integrate a Mesh Masked Autoencoder (MeshMAE) for self-supervised structural embedding.
- [ ] Validate latent space representations of initial organic assets (Trees, Rocks).

## Phase 2: Deformation Sandbox & Heuristic Reward Engineering ▢
- [ ] Develop the 3D deformation simulator applying masks to 4 channels: Bend (R), Noise (G), Scale (B), Fixed (A).
- [ ] Formulate deterministic geometry & physics penalty functions (Laplacian smoothness, normal consistency, ground contact).
- [ ] Build a multi-angle rendering pipeline to export synthetic views of deformed assets.
- [ ] Evaluate a heuristic-driven baseline model without human feedback.

## Phase 3: RLHF Framework & Reward Model Training ▢
- [ ] Design a lightweight Human-in-the-Loop (HITL) annotation interface (Gradio/Streamlit) for pairwise comparison.
- [ ] Collect initial artist preference data (Asset A vs. Asset B) for naturalness and artistic utility.
- [ ] Train a Bradley-Terry preference reward model based on the collected feedback.
- [ ] Validate reward model correlation with human qualitative scores.

## Phase 4: PPO Optimization Loop Integration ▢
- [ ] Assemble the full Reinforcement Learning environment using the OpenAI Gym/Gymnasium interface.
- [ ] Configure the PPO (Proximal Policy Optimization) actor-critic networks to predict continuous vertex-level masks.
- [ ] Integrate the composite reward function: $R = \alpha R_{human} + \beta R_{variety} - \gamma R_{distortion} - \delta R_{physics}$.
- [ ] Run optimization loops and monitor convergence using Weights & Biases (WandB).

## Phase 5: UE5 Production Pipeline & Empirical Validation ▢
- [ ] Build an automation script to bake learned deformation masks into Static Mesh Vertex Colors (RGBA).
- [ ] Author custom Unreal Engine 5 Materials utilizing World Position Offset (WPO) driven by Vertex Color channels.
- [ ] Set up an Hierarchical Instanced Static Mesh (HISM) stress test scene in UE5.
- [ ] Profile performance metrics (FPS, Draw Calls, VRAM, Shader Instructions) and document asset efficiency.
