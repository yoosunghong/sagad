## Final Research Proposal

### Anticipated Paper Title

**Learning Deformation-Aware Variant Regions for Single-Asset Game Object Diversification via Human Feedback**

---

### 1. Research Objectives and Background

* **Limitations of Existing Methods:** To secure visual diversity in game production, applying simple seed-based random deformations often results in unnatural outputs, such as destroying the semantic structure of the mesh or causing the grounding plane to float in mid-air. To prevent this, Technical Artists manually design procedural rules (PCG). However, this approach suffers from poor scalability as rules must be redesigned from scratch for every unique asset type.
* **Proposed Solution:** This study proposes an interpretable **Deformation Mask** model that automatically predicts *where, what type, and how much* deformation should be applied to remain natural, based on geometric characteristics of the 3D mesh such as shape, curvature, and height.
* **Key Differentiation:** Unlike conventional 3D generative AIs that output heavy raw vertex coordinates, this research establishes an engine-friendly mask pipeline that seamlessly integrates with shaders (World Position Offset, WPO) of commercial engines like Unreal Engine 5 (UE5). This simultaneously ensures practical utility and academic originality.

---

### 2. Core Hypothesis and Research Questions

* **Research Question (RQ):** By simultaneously incorporating the morphological characteristics of a 3D mesh and the aesthetic preferences of artists, can a model learn in a self-supervised manner to identify which regions a seed-based deformation should be applied to achieve natural and diverse results?
* **Core Hypothesis:** Analyzing the geometric features of a mesh (e.g., ground proximity, surface curvature, local thickness) allows for defining optimal deformable regions that minimize structural damage.
* Regions close to the ground must activate a **fixed mask** to maintain structural stability.
* Thin, protruding, or highly curved regions naturally tend to activate **noise and color deformation masks**.



---

### 3. Input and Output Data Definitions

#### 3.1 Input Features

Geometric feature vectors extracted from the 3D mesh are used as inputs for models such as Graph Neural Networks (GNNs):

* Vertex Position / Normal / Curvature
* Height from ground / Distance from mesh center
* Local thickness / Ambient occlusion
* Connected components and structural topology information

#### 3.2 Output Target

A set of channel-wise deformation masks valued between 0 and 1, mapped at the vertex or region level. For engine compatibility, it is designed to be compactly stored within the **Vertex Color (RGBA)** channels:

* **R Channel:** Bend Mask (controls vertical bending)
* **G Channel:** Noise Displacement Mask (alters surface detail deformation)
* **B Channel:** Scale Deformation Mask (controls localized scale variances)
* **A Channel:** Fixed Region Mask (preserves ground contact and structural integrity; 0 = fully fixed)

---

### 4. Methodology

To overcome the absence of vertex-level ground-truth mask datasets, this study proposes a pipeline that fuses geometric pre-training with Reinforcement Learning from Human Feedback (RLHF) reward model optimization.

```
[3D Input Mesh] ──> [MeshMAE Semantic Embedding] ──> [Mask Prediction Network] ──> [Apply Mesh Deformation]
                                                                                           │
[Optimal Mask Output] <── [RL Optimization (PPO)] <── [Composite Reward Function] ─────────┘

```

#### Step 1: Semantic Feature Embedding for Geometric Understanding

To flexibly adapt to variations in mesh scale and topology, we introduce the **Mesh Masked Autoencoder (MeshMAE)** technique. Through self-supervised learning—where random patches of the mesh are masked and reconstructed—the model develops an embedding capability that deeply understands localized and structural dependencies, such as small tree branches or rock contours.

#### Step 2: Designing the RLHF Reward Model

Instead of having artists manually paint deformation masks, they evaluate preferences based on AI-generated outputs.

* **A/B Testing-Based Evaluation:** Two distinct deformation variations (A and B) generated from the same mesh are presented to an artist, who selects "Which looks more natural?", thereby gathering human preference data.
* **Reward Model Training:** Based on the collected artist preference data, a lightweight neural network (**Human RM**) is trained to predict the quality score of the deformed 3D mesh.

#### Step 3: Reward-Driven Reinforcement Learning Loop (RL Optimization)

The mask prediction network is optimized using the Proximal Policy Optimization (PPO) algorithm to maximize the following composite reward function ($Standardized\ R$):

$$R = \alpha R_{\text{human}} + \beta R_{\text{variety}} - \gamma R_{\text{distortion}} - \delta R_{\text{physics}}$$

* $R_{\text{human}}$ **(Human Aesthetic Reward):** The artist preference score predicted by the reward model trained in Step 2.
* $R_{\text{variety}}$ **(Visual Diversity Reward):** After rendering the deformed meshes from multiple angles, vector distances are calculated within the CLIP image embedding space. Higher rewards are given to assets with distinctly different silhouettes.
* $R_{\text{distortion}}$ **(Geometric Distortion Penalty):** Penalties are applied if Laplacian smoothness collapses or if face flipping (Normal Inconsistency) occurs.
* $R_{\text{physics}}$ **(Physical Grounding Penalty):** Penalties are given if the fixed mask near the ground ($Height \approx 0$) fails, causing the mesh to float in the air or clip into the terrain.

---

### 5. Initial Research Scope and Experimental Design

#### 5.1 Target Asset Scope

In the initial phase, we focus on validating generalization within the ecosystem of **Organic Assets**, which possess clear structural rules (man-made objects and architectural structures are categorized for future extended research):

* **Organic Vertical Structures (Trees/Bushes):** Assets where bending relative to the central axis and self-intersection avoidance serve as core constraints.
* **Organic Volumetric Structures (Rocks/Cliffs):** Assets devoid of specific functional structures, where surface noise and multi-dimensional scaling variations are paramount.

#### 5.2 Baseline Methodologies for Comparison

* **Baseline 1:** Random uniform deformation
* **Baseline 2:** Geometry-based deterministic rule deformation (Height/Curvature-based rules)
* **Baseline 3:** Artist-authored preset masks (manually painted)
* **Proposed:** Geometry-Aware Learned Mask (Our proposed geometry and human-feedback-aware approach)

---

### 6. Evaluation Metrics

#### 6.1 Visual Diversity

For $n$ variants generated from the same original mesh, multi-angle renderings are performed. The breadth of deformation is quantitatively evaluated using the **Cosine Distance** between CLIP image embeddings and the silhouette intersection areas.

#### 6.2 Naturalness

* **Quantitative Evaluation:** Measuring the frequency of self-intersections and the rate of Ground Contact Violations in the post-deformed mesh.
* **Qualitative Evaluation:** Comparing Mean Opinion Scores (MOS) evaluating naturalness gathered via blind tests from groups of professional artists and gamers.

#### 6.3 Runtime Performance and Asset Efficiency

* **Production Metrics:** Comparing total disk space utilization when deploying 1,000 polymorphic objects (Multi-asset approach vs. Single-asset + Mask approach).
* **Engine Performance Metrics:** Measuring Frame Rate (FPS), Draw Calls, and VRAM utilization when applying Instanced Static Meshes (ISM/HISM) within an Unreal Engine 5 environment.

---

### 7. Expected Contributions

* **Organic Synergy of Artist Feedback and 3D Geometry:** Even in the absence of ground-truth mask data, this study presents a framework that successfully transplants an artist’s aesthetic intuition into a 3D graphics optimization loop via RLHF mechanisms.
* **AI Positioning Centered on Practical Production:** Instead of relying on heavy generative AIs that forge completely new meshes, our model predicts an interpretable, lightweight control mask that is 100% compatible with legacy asset pipelines, positioning itself as an automated assistant for technical art.
* **Cost Innovation in Game Engine Pipelines:** We demonstrate a practical methodology that achieves infinite visual diversity through real-time GPU computation while minimizing data storage footprints.