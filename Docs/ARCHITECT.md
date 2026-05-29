# System Architecture

This document describes the structural design and tensor flow of the deformation-aware mask generation system.

## 1. Component Overview

The system is split into two primary runtime blocks: the **Training Loop (Python Backend)** and the **Production Deployment (Unreal Engine 5 Core)**.

```
+---------------------------------------------------------------------------------+
|                           TRAINING LOOP (PYTHON PIPELINE)                       |
|                                                                                 |
|  +---------------+      +-------------------+      +-------------------------+  |
|  | Input 3D Mesh | ---> |  MeshMAE Encoder  | ---> |  Mask Predictor Network |  |
|  |    (Graph)    |      | (Structural Latent|      |     (PPO Actor Policy)  |  |
|  +---------------+      +-------------------+      +-------------------------+  |
|                                                                 |               |
|                                                                 v               |
|  +---------------+      +-------------------+      +-------------------------+  |
|  | Optimized     | <--- |  PPO Optimization | <--- | Predicted RGBA Masks    |  |
|  | Model Weights |      |       Loop        |      | (Bend, Noise, Scale, Fix)  |  |
|  +---------------+      +-------------------+      +-------------------------+  |
|                                  ^                              |               |
|                                  |                              v               |
|                         +------------------+       +-------------------------+  |
|                         | Unified Reward   | <---| |   Deformation Sandbox   |  |
|                         | Framework        |       |  & Rendering Engine     |  |
|                         +------------------+       +-------------------------+  |
+------------------------------------------|--------------------------------------+
                                           | (Export Target Weights)
                                           v
+---------------------------------------------------------------------------------+
|                        PRODUCTION PIPELINE (UNREAL ENGINE 5)                    |
|                                                                                 |
|  +-----------------------+      +--------------------+      +----------------+  |
|  | Base Static Mesh      | ---> | Custom Shader Graph| ---> | Infinite HISM  |  |
|  | (With RGBA V-Colors)  |      |  (WPO Deformation) |      | Varied Batches |  |
|  +-----------------------+      +--------------------+      +----------------+  |
+---------------------------------------------------------------------------------+
```

---

## 2. Module Specifications

### 2.1 Geometry Encoding Module
* **Input:** Raw vertex matrices $V \in \mathbb{R}^{N 	imes 3}$ and adjacency matrices $A \in \mathbb{R}^{N 	imes N}$.
* **Architecture:** Graph Attention Networks (GATv2) stacked on top of a self-supervised MeshMAE baseline.
* **Output:** Localized node embeddings $Z \in \mathbb{R}^{N 	imes D}$, representing the structural and geometric identity of every vertex.

### 2.2 Actor Policy Module (Mask Generator)
* **Architecture:** Multi-Layer Perceptron (MLP) decoding block capped with a Sigmoid activation function.
* **Output:** Continuous mask spaces $M \in [0, 1]^{N 	imes 4}$. Channels correspond to $[M_{bend}, M_{noise}, M_{scale}, M_{fixed}]$.

### 2.3 Deformation Sandbox & Environment Simulation
* This module applies the continuous mask values to deformation operators (Trimesh-based bending, noise vector shifting along normals, and part scaling).
* Outputs a temporary modified mesh file and triggers headless rendering instances via PyOpenGL or Open3D to capture orthogonal multi-view snapshot arrays.

### 2.4 Evaluator & Unified Reward Network
* **Human Preference Module:** Processes multi-view snapshot combinations into a Bradley-Terry preference loss, refining the linear reward estimator.
* **Diversity Evaluator:** Uses HuggingFace CLIP (`ViT-B/32`) to compute pairwise cosine distance matrices across concurrent mutation loops.
* **Geometric Regularizer:** Approximates mesh distortion by computing vertex-wise Laplacian coordinates to trace surface structure degradation.

---

## 3. Production Deployment Strategy (UE5)
1. **Baking Sequence:** The finalized model evaluates target production assets. The generated RGBA arrays are written directly into the mesh's `Vertex Color` buffer.
2. **Shader Translation:** A master material parameter collection handles runtime mutation seeds. The material graph samples the Vertex Color channels:
   * **Red Value:** Multiplies a directional or localized rotational vector (World Position Offset - WPO) to simulate organic bend profiles.
   * **Green Value:** Scales a high-frequency 3D Perlin Noise expression displacing the vertex position along its absolute normal vector.
   * **Alpha Value:** Acts as a direct lerp mask against the composite offset accumulator (0 forces original vertex position, keeping the base locked to the landscape).
