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
* **Input:** Raw vertex matrices $V \in \mathbb{R}^{N 	imes 3}$ and mesh adjacency. The data pipeline (`src/data/`) emits a PyG `Data` graph where:
  * `pos` $\in \mathbb{R}^{N 	imes 3}$ — unit-bounding-sphere normalized vertex coordinates (recentred to centroid, isotropically scaled). Inverse transform (`source_centroid`, `source_scale`) is attached for coordinate round-tripping.
  * `x` $\in \mathbb{R}^{N 	imes 6}$ — per-node features `[pos_xyz, normal_xyz]` fed to the encoder.
  * `edge_index` $\in \mathbb{Z}^{2 	imes 2E}$ — undirected COO mesh adjacency (sparse; the dense $A \in \mathbb{R}^{N 	imes N}$ is never materialized).
  * `face` $\in \mathbb{Z}^{3 	imes F}$ — triangle table retained for downstream Laplacian / normal-consistency penalties.
* **Architecture:** Graph Attention Networks (GATv2) stacked on top of a self-supervised MeshMAE baseline. Realized in `src/models/mesh_mae.py` (`MeshMAE`):
  * **Encoder** — a 3-layer GATv2 stack (`hidden_dim=128`, `heads=4`, concat heads in hidden layers, mean heads on the output) mapping `x` to the latent $Z$ with $D=64$ (`latent_dim`).
  * **Self-supervised objective (GraphMAE-style masked reconstruction):** a random `mask_ratio` (default 0.5) of nodes has its input feature replaced by a learnable `[MASK]` token; the corrupted graph is encoded, masked latents are re-masked with a second learnable token, and a 2-layer GATv2 decoder reconstructs the original `[pos, normal]` features only at masked nodes.
  * **Reconstruction loss:** scaled cosine error (SCE), the mean of $(1 - \cos(\hat{x}_c, x_c))^{\gamma}$ ($\gamma=2$) computed per `pos`/`normal` sub-vector so their differing magnitudes are balanced.
  * **NaN guards (CLAUDE.md):** `isfinite` asserts on masked inputs, latent $Z$, and the loss; per-forward telemetry (mask ratio, SCE) logs at DEBUG.
* **Inference:** `MeshMAE.encode(data)` runs the unmasked encoder to emit $Z$; consumed downstream by the §2.2 PPO actor policy.
* **Output:** Localized node embeddings $Z \in \mathbb{R}^{N 	imes D}$, representing the structural and geometric identity of every vertex.

### 2.2 Actor Policy Module (Mask Generator)
* **Architecture:** Multi-Layer Perceptron (MLP) decoding block capped with a Sigmoid activation function.
* **Output:** Continuous mask spaces $M \in [0, 1]^{N 	imes 4}$. Channels correspond to $[M_{bend}, M_{noise}, M_{scale}, M_{fixed}]$.

### 2.3 Deformation Sandbox & Environment Simulation
* This module applies the continuous mask values to deformation operators (Trimesh-based bending, noise vector shifting along normals, and part scaling). Realized in `src/deform/` (`operators.py` + `sandbox.py`); all operators are vectorized torch tensor ops over the sparse graph (no per-vertex Python loops, no dense `N x N`).
* **Per-channel offset operators** (each a per-vertex displacement, gated by its mask channel $M_{\bullet} \in [0,1]^N$):
  * **Bend ($M_{bend}$, R):** height-weighted Rodrigues rotation of $(p - \text{pivot})$ about a horizontal `bend_axis`; rotation angle $\theta_i = \text{strength} \cdot M_{bend,i} \cdot \hat{h}_i$ scales with normalized height $\hat{h}$ above the base pivot, so the lean grows toward the top (organic bend profile).
  * **Noise ($M_{noise}$, G):** displacement along the vertex normal, $\Delta_i = \text{strength} \cdot M_{noise,i} \cdot \eta(f \cdot p_i)\, \hat{n}_i$, where $\eta$ is a deterministic seeded 3D value-noise field (trilinear-interpolated lattice hash) and $f$ is the spatial frequency.
  * **Scale ($M_{scale}$, B):** radial displacement from the mesh centroid, $\Delta_i = \text{strength} \cdot M_{scale,i} \cdot (p_i - c)$.
* **Composite + mobility gate:** $\Delta^{\text{composite}}_i = \Delta^{bend}_i + \Delta^{noise}_i + \Delta^{scale}_i$; the final position is $p'_i = p_i + \text{mob}_i \cdot \Delta^{\text{composite}}_i$ where the **mobility** $\text{mob}_i = (1 - M_{fixed,i})$. Thus $M_{fixed}=1 \Rightarrow$ vertex locked in place (used to pin the asset base to the landscape); $M_{fixed}=0 \Rightarrow$ full offset applied. This single convention is shared by the CPU sandbox and the UE5 bake (§3) so the simulation matches the shader.
* **Telemetry / NaN guards (CLAUDE.md Logging Protocols):** per-channel mean offset magnitude, total max/mean vertex displacement, relative edge-length change distribution (vertex-degradation scale), and locked-vertex fraction; `isfinite` asserts on every offset block and the final positions.
* Outputs a temporary modified mesh file and triggers headless rendering instances via PyOpenGL or Open3D to capture orthogonal multi-view snapshot arrays.

### 2.4 Evaluator & Unified Reward Network
* **Human Preference Module:** Processes multi-view snapshot combinations into a Bradley-Terry preference loss, refining the linear reward estimator.
* **Diversity Evaluator:** Uses HuggingFace CLIP (`ViT-B/32`) to compute pairwise cosine distance matrices across concurrent mutation loops.
* **Geometric Regularizer:** Approximates mesh distortion by computing vertex-wise Laplacian coordinates to trace surface structure degradation.

---

## 3. Production Deployment Strategy (UE5)
1. **Baking Sequence:** The finalized model evaluates target production assets. The generated RGBA arrays are written directly into the mesh's `Vertex Color` buffer.
2. **Shader Translation:** A master material parameter collection handles runtime mutation seeds. The material graph samples the Vertex Color channels:
   * **Red Value ($M_{bend}$):** Multiplies a directional or localized rotational vector (World Position Offset - WPO) to simulate organic bend profiles.
   * **Green Value ($M_{noise}$):** Scales a high-frequency 3D Perlin Noise expression displacing the vertex position along its absolute normal vector.
   * **Blue Value ($M_{scale}$):** Scales a radial offset from the mesh pivot/centroid (part swell/shrink), added into the composite WPO accumulator alongside Red and Green.
   * **Alpha Value (mobility $= 1 - M_{fixed}$):** Acts as a direct lerp mask against the composite offset accumulator (0 forces original vertex position, keeping the base locked to the landscape). **Bake convention:** the actor's $M_{fixed}$ is inverted at bake time so Alpha stores mobility; $M_{fixed}=1 \Rightarrow$ Alpha $=0 \Rightarrow$ locked vertex. This matches the §2.3 sandbox gate exactly.
