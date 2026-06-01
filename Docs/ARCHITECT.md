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
* **Architecture:** Multi-Layer Perceptron (MLP) decoding block capped with a Sigmoid activation function. Realized in `src/rl/policy.py` (`MaskActorCritic`): the MLP is **weight-shared per node** (applied to each row of the latent $Z$ independently), so the policy is invariant to the vertex count $N$.
* **Continuous-action parameterization:** the actor emits a per-node, per-channel *logit mean*; actions are sampled in **logit space** from a diagonal Gaussian (global per-channel `log_std`) and squashed through a sigmoid into $[0,1]$. Logit space keeps the log-prob well-defined (no density spikes at the 0/1 boundary). **Log-prob is averaged over the $N \times 4$ action dims, not summed** — summing makes the PPO importance ratio $\exp(\sum \text{tiny per-dim diffs})$ explode for $N$ in the thousands (huge KL, clipping saturates, no learning); the per-dimension mean keeps the ratio $O(1)$.
* **Critic:** the value head is applied per node and **mean-pooled** into a single scalar graph value $V(Z)$.
* **Output:** Continuous mask spaces $M \in [0, 1]^{N 	imes 4}$. Channels correspond to $[M_{bend}, M_{noise}, M_{scale}, M_{fixed}]$.

### 2.2b RL Environment & PPO Loop (Phase 4)
* **Environment (`src/rl/env.py`, `DeformEnv`):** a single-asset Gymnasium env. Each episode is **one step** (a contextual-bandit formulation of "predict a mask → score it"): `reset` returns the asset's frozen MeshMAE latent $Z$ as the observation; `step` applies the predicted $[N,4]$ mask through the §2.3 sandbox, scores it with the composite reward, and terminates. The env owns all reward computation; $R_{human}$ is injected as a `human_reward_fn(mesh) -> float` callback so the env stays decoupled from CLIP / the reward model.
* **Composite reward (`src/rl/composite.py`):** $R = \alpha R_{human} + \beta R_{variety} - \gamma R_{distortion} - \delta R_{physics}$ — the §2.4 learned term plus the Phase 2 variety proxy and deterministic penalties. NaN-guarded.
* **PPO (`scripts/train_ppo.py`):** collects a batch of sampled masks for the same frozen $Z$, computes single-step advantages $A = R - V(Z)$ (batch-normalized), and applies the clipped surrogate + value baseline + entropy bonus. Convergence is logged to Weights & Biases (**offline by default** — no login/network). $R_{human}$ (render+CLIP per rollout) is the expensive path, enabled with `--alpha > 0 --reward-model`; the geometric composite validates convergence cheaply, and the final reward is compared against the Phase 2 heuristic baseline.

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
  * **HITL annotation pipeline (`src/hitl/`, Phase 3):** an isolated module separate from `src/deform` / `src/rewards`.
    * `candidates.py` samples a pool of geometry-structured, base-locked deformations per asset (randomized per-channel gains + jittered base band + noise seed), deforms via the §2.3 sandbox, renders each to a §2.3 multi-view **contact sheet**, and emits a **manifest** (`data/preferences/manifest.json`): candidate specs (`candidate_id`, `gains`, `noise_seed`, `render` path, degradation telemetry) + a **within-asset** pairing schedule (cross-asset pairs are meaningless for a naturalness comparison; A/B side is randomized to remove positional bias).
    * Each candidate is exported both as a multi-view PNG contact sheet (for the §2.4 CLIP featurizer) **and as a GLB mesh** (`data/preferences/meshes/`, rotated Z-up→Y-up so assets stand upright in the viewer).
    * `app.py` builds a Gradio side-by-side comparison UI showing the two candidates as **interactive 3D meshes** (`gr.Model3D`, orbit/zoom) — static grayscale renders made similar trees hard to discriminate, so annotators inspect the geometry directly — with `◀ A better` / `Tie` / `Skip` / `B better ▶`; `scripts/build_preference_pairs.py` builds the job and `scripts/annotate_ui.py` serves it.
    * `store.py` is an **append-only JSONL** preference store (`data/preferences/preferences.jsonl`), one record per judgement: `{pair_id, asset, a, b, choice ∈ {a,b,tie,skip}, annotator, timestamp}`. Each line is flushed immediately (crash-safe) and the UI skips already-annotated `pair_id`s (resumable). `to_comparisons()` emits clean `(winner, loser)` pairs (ties/skips dropped) — the direct training signal for the Bradley-Terry reward model.
  * **Candidate featurizer (`src/hitl/features.py`):** a deformation is represented by its *appearance*, not raw geometry. `CLIPFeaturizer` embeds each candidate's multi-view renders with **CLIP ViT-B/32** (`get_image_features`, L2-normalized), mean-pooled across views → a `512`-d feature; `candidate_features` caches the per-candidate embeddings to disk (`clip_features.pt`). `use_safetensors=True` is mandatory (transformers refuses `.bin` `torch.load` on torch < 2.6). This is the input to the reward model below and reused by the §2.4 CLIP diversity evaluator.
  * **Bradley-Terry reward model (`src/rewards/preference.py`, the learned `R_human`):** a scalar reward head over the CLIP feature — a plain linear estimator (`hidden=0`) or a small GELU MLP (default) — trained with the logistic loss `L = -mean log σ(r(win) − r(lose))` (`bt_loss`). `train_reward_model` does full-batch Adam over the aligned `(winner, loser)` feature pairs from `to_comparisons()`. `isfinite` guards on the reward output and loss. Produces the `α·R_human` term of the Phase 4 composite reward; kept in `src/rewards` with the other reward components but distinct from the *deterministic* penalties (this term is *learned*).
  * **Bootstrap / validation harness:** `scripts/build_preference_pairs.py` → `scripts/annotate_ui.py` collects real labels; pending a human annotator, `scripts/simulate_preferences.py` emits a **synthetic-oracle** label set (Bradley-Terry-sampled from the deterministic Phase 2 heuristic reward, reconstructed from the manifest gains) so the trainer is exercised on the *same* `to_comparisons()` interface. `scripts/train_reward_model.py` trains and reports **held-out pairwise accuracy** and **rank correlation** (Spearman/Pearson) of the learned reward against the oracle / qualitative scores — the "correlation with human qualitative scores" gate of this section.
* **Diversity Evaluator:** Uses HuggingFace CLIP (`ViT-B/32`) to compute pairwise cosine distance matrices across concurrent mutation loops.
* **Geometric Regularizer:** Approximates mesh distortion by computing vertex-wise Laplacian coordinates to trace surface structure degradation.
* **Stretch / texture-integrity penalty (`edge_stretch`):** mean relative per-edge length change between the undeformed and deformed mesh. Vertex displacement (the CPU sandbox *and* the UE5 WPO shader) leaves UVs pinned per vertex, so any edge-length change stretches/compresses the texture mapped across that edge — the source of texel-density smearing and UV-seam tearing under WPO. This term enters the **distortion** group (alongside Laplacian + normal-consistency) so the policy/sampler is penalized for texture-breaking deformations, not only geometry-roughening ones. Rigid motions give zero. (The sandbox already emits `edge_len_rel_change_*` as telemetry; this promotes it to a reward term.)
  * **Bake-time companion (Phase 5):** UV-seam vertices are *duplicated* in a UE5 Static Mesh; the baked RGBA must be **identical across coincident duplicates** or WPO moves the two halves apart and tears the seam. The bake step enforces seam-coherent vertex colors (merge-then-average on coincident positions).

### 2.5 Variant Diffusion Sampler (per-instance "DNA" generator)

The variation engine for the single-file + WPO-masking deployment. A single Static
Mesh asset carries **one** baked vertex-color buffer (shared by every instance), so
per-instance variation cannot live in vertex color — it travels through UE5
**Per-Instance Custom Data**: a short float vector written when instances are
scattered into a level. This module *generates the distribution* of those vectors.

* **Module (`src/diffusion/`, isolated from deform/rewards/rl/hitl).** Performs no
  deformation and no rendering — it learns and samples the variant-parameter
  distribution only.
* **Variant parameter schema (`params.py`, `VariantParamSpec`):** the typed contract
  between the sampler and the bake. An ordered list of named continuous fields with
  `[lo, hi]` bounds (e.g. `bend_gain`, `noise_gain`, `scale_gain`, `base_lock`, and —
  for buildings — `floor_count` flagged `integer`). `to_model`/`from_model` is the
  bijection to the standardized $[-1,1]$ space the denoiser operates in (integer
  fields are rounded on decode). The decoded raw vector is exactly the Per-Instance
  Custom Data payload; the same fields decode into the §2.3 mask gains for the CPU
  sandbox so the simulation matches the shader.
* **Denoiser + DDPM (`ddpm.py`):** a small `DiffusionMLP` (sinusoidal time embedding,
  GELU MLP, optional conditioning on the pooled MeshMAE asset latent $Z$ so the
  sampler is asset-aware) trained with standard $\epsilon$-prediction DDPM (linear
  $\beta$ schedule, `q_sample`, MSE loss, ancestral `sample`). Why diffusion rather
  than the §2.2 PPO policy: the goal is **many plausible variants** — a *distribution*
  to sample, which is what a denoiser is — not the single reward-optimal mask a policy
  converges to. NaN-guarded on the loss and on every sampled batch (CLAUDE.md).
* **Texture-safe by construction:** training samples are filtered through the §2.4
  penalties (the §2.4 stretch term especially), so the learned distribution only
  covers deformations that do not smear/tear textures under WPO. With no artist data,
  the bootstrap distribution is rejection-sampled from random gains scored by those
  deterministic penalties — using existing models/constraints, no new labels.
* **Output:** $K$ sampled variant vectors → written as Per-Instance Custom Data on the
  $K$ placed HISM/Nanite instances; the §3 WPO shader combines them with the shared
  baked masks to render $K$ distinct instances from one mesh file.

---

## 3. Production Deployment Strategy (UE5)
1. **Baking Sequence:** The finalized model evaluates target production assets. The generated RGBA arrays are written directly into the mesh's `Vertex Color` buffer.
2. **Shader Translation:** A master material parameter collection handles runtime mutation seeds. The material graph samples the Vertex Color channels:
   * **Red Value ($M_{bend}$):** Multiplies a directional or localized rotational vector (World Position Offset - WPO) to simulate organic bend profiles.
   * **Green Value ($M_{noise}$):** Scales a high-frequency 3D Perlin Noise expression displacing the vertex position along its absolute normal vector.
   * **Blue Value ($M_{scale}$):** Scales a radial offset from the mesh pivot/centroid (part swell/shrink), added into the composite WPO accumulator alongside Red and Green.
   * **Alpha Value (mobility $= 1 - M_{fixed}$):** Acts as a direct lerp mask against the composite offset accumulator (0 forces original vertex position, keeping the base locked to the landscape). **Bake convention:** the actor's $M_{fixed}$ is inverted at bake time so Alpha stores mobility; $M_{fixed}=1 \Rightarrow$ Alpha $=0 \Rightarrow$ locked vertex. This matches the §2.3 sandbox gate exactly.
