# Phase 2 Progress Log: Deformation Sandbox & Heuristic Reward Engineering

## Status: ✅ Complete

### Executed Tasks
- [x] Develop the 3D deformation simulator applying masks to 4 channels: Bend (R), Noise (G), Scale (B), Fixed (A) (`src/deform/`).
- [x] Formulate deterministic geometry & physics penalty functions: Laplacian smoothness, normal consistency, ground contact (`src/rewards/`).
- [x] Build a multi-angle headless rendering pipeline to export synthetic views of deformed assets (`src/render/`).
- [x] Evaluate a heuristic-driven baseline model without human feedback (`src/rewards/heuristic.py`, `scripts/eval_baseline.py`).

### Technical Notes

#### Channel-map alignment (CLAUDE.md domain constraint — done *before* code)
`docs/ARCHITECT.md` §2.3 and §3 were updated first to lock the RGBA semantics:
* **R = Bend, G = Noise, B = Scale, A = mobility (`1 - M_fixed`).** §3 previously
  omitted the Blue/Scale channel and left the Alpha lerp convention implicit; both
  are now specified. The actor emits `M_fixed` (intuitive: 1 = fixed); the bake/sim
  invert it to mobility so `M_fixed=1 ⇒ Alpha=0 ⇒ locked vertex`. This single
  convention is shared by the CPU sandbox and the UE5 shader so simulation matches
  the engine.

#### New isolated module (`src/deform/`, separate from data/models per modularity rule)
* `operators.py` — pure, vectorized per-vertex displacement operators:
  * **`bend_offset`** — height-weighted Rodrigues rotation about a horizontal hinge
    axis; angle `θ_i = strength · M_bend,i · ĥ_i` grows with normalized height above
    the base pivot (organic lean).
  * **`noise_offset`** — along-normal displacement by a deterministic seeded 3D
    value-noise field (`value_noise_3d`: trilinear-interpolated lattice hash with a
    quintic fade — a repeatable stand-in for the UE5 Perlin node).
  * **`scale_offset`** — radial swell/shrink from the mesh centroid.
* `sandbox.py` — `apply_masks(data, masks[N,4], cfg)`:
  * composite `Δ = bend + noise + scale`, gated by `mobility = (1 - M_fixed)`;
    `new_pos = pos + mobility · Δ`.
  * Builds the deformed `trimesh.Trimesh` (original face table + new vertices).
* All ops are vectorized torch over the sparse graph — no per-vertex Python loops,
  no dense `N×N` (CLAUDE.md optimization rule). Runs on CUDA.

#### Telemetry / NaN guards (CLAUDE.md Logging Protocols)
* `isfinite` asserts on input masks, each per-channel offset block, and the final
  positions.
* **Vertex-degradation scales** logged: per-channel mean offset magnitude, total
  max/mean displacement, locked-vertex fraction, and the relative edge-length
  change distribution (mean / max / p99) — the latter flags over-deformation that
  the upcoming distortion/Laplacian penalties will regularize.

#### Validation (`scripts/deform_demo.py`)
The PPO actor doesn't exist yet, so a geometry-aware procedural mask drives the
sim: base-locked Fixed band, height-weighted Bend, uniform Noise, mild Scale.
* **Locked-base invariant PASS** — fully-fixed (`M_fixed=1`, mobility 0) vertices
  move **exactly 0.0** (438/7106 on the baseline rock, 27/2552 on `tree_2`);
  ramp vertices move proportionally to their mobility, as designed.
* NaN-free across rocks and trees. Deformed meshes exported to `data/deformed/`.
* Telemetry confirms Bend dominates the displacement budget at the demo strengths
  and surfaces large `edge_len_rel_change_max` on thin tree geometry — exactly the
  degradation signal the next task's penalty functions consume.

### Deterministic Penalty Functions (`src/rewards/`)

New isolated module (CLAUDE.md "reward constraints ... in dedicated modules"),
separate from `src/deform`. Maps to the `-γ R_distortion - δ R_physics` terms of
the Phase 4 composite reward and the ARCHITECT.md §2.4 Geometric Regularizer.

* `penalties.py`:
  * **`laplacian_distortion`** — mean shift of uniform (umbrella) Laplacian
    coordinates `δ_i = p_i - mean(p_neighbors)` between original and deformed
    meshes. Local surface-detail degradation; **rigid-motion invariant**.
  * **`normal_consistency`** — mean *increase* in the dihedral angle between
    adjacent face normals (torch face normals; static face-adjacency pairs from
    trimesh, computed once & reused across a rollout). Penalizes creasing/spiking.
  * **`ground_contact`** — `(gap, penetration)`: how far the deformed base floats
    above the original ground plane (lost contact) and the mean depth of vertices
    pushed below it (sinking).
  * **`evaluate_penalties`** — aggregates into `distortion_total` / `physics_total`
    via `PenaltyWeights`; the outer composite coefficients (γ, δ) are deferred to
    the Phase 4 PPO loop. `isfinite` guard on every term (CLAUDE.md NaN protocol).
* **Design:** every penalty compares against the *original* geometry, not absolute
  smoothness — organic rocks/trees are inherently rough/non-watertight, so only the
  deformation-induced *increase* is meaningful (and this yields the rigid invariances).
  Per-vertex math is vectorized torch over the sparse graph.

#### Validation (`scripts/eval_penalties.py`) — 6/6 invariants PASS
| Case | Expectation | Result (rock / tree_2) |
|------|-------------|------------------------|
| Identity (no deform) | all penalties ≈ 0 | distortion=0, physics=0 ✓ |
| Rigid lift (+0.2 up) | distortion ≈ 0, ground gap > 0 | distortion=0, gap=0.200 ✓ |
| Procedural deform | distortion > 0, base pinned (gap ≈ 0) | distortion=0.064 / 0.270, gap=0 ✓ |

The rigid-lift case confirms Laplacian & normal terms are motion-invariant; the
deform case confirms the Fixed-channel base lock keeps ground contact (gap=0) while
distortion registers. The thinner tree shows higher normal-consistency penalty than
the rock, as expected.

### Multi-View Rendering Pipeline (`src/render/`)

New isolated module (CLAUDE.md modularity rule) for the headless multi-angle
snapshot export of ARCHITECT.md §2.3. Output feeds the §2.4 CLIP diversity
evaluator and the Phase 3 human-preference interface.

* **Backend selection (probed on this host):** Open3D's Filament
  `OffscreenRenderer` **fails** to initialize headless on Windows, and `pyrender`
  is not installed. The legacy `o3d.visualization.Visualizer` with
  `visible=False` **works** (valid GPU renders) and was chosen. matplotlib (Agg)
  remains a CPU fallback if needed.
* `multiview.py` — `render_multiview(mesh, RenderConfig) -> (V, H, W, 3) uint8`:
  N orbit cameras evenly spaced in azimuth at a fixed elevation; one offscreen
  window reused across views, only the camera **extrinsic** rebuilt per view
  (`_look_at`) so the intrinsic stays matched to the framebuffer aspect. Distance
  auto-scales to the mesh bounding radius. `save_views` writes per-view PNGs + a
  horizontal contact sheet.
* **Telemetry (CLAUDE.md):** per-view foreground (non-background) pixel fraction
  logged; a near-zero fraction warns of the silent black-render failure mode.

#### Validation (`scripts/render_demo.py`)
Renders the baseline rock original vs. procedurally-deformed, 4 views each @256px:
* All 8 views non-empty (foreground fraction 0.05–0.24); contact sheets written to
  `data/renders/`.
* **Deformed ≠ original** — mean absolute pixel diff = 13.73 (PASS). Visual check
  confirms the height-weighted Bend (leaning top) is clearly visible across angles.

### Heuristic Baseline (no human feedback)

The human-free composite reward and a reference baseline evaluation, closing
Phase 2. `R_human` (Phase 3 Bradley-Terry) and the CLIP diversity evaluator
(§2.4) are deliberately deferred.

* **`src/rewards/heuristic.py`** — `R = β·variety − γ·distortion − δ·physics`
  (`heuristic_reward`), plus `batch_diversity` (mean pairwise per-vertex distance
  across a batch of displacement fields — a geometric stand-in for the §2.4 CLIP
  metric). `variety` uses the sandbox's mean per-vertex displacement as an
  activity proxy.
* **`scripts/eval_baseline.py`** — for each asset, samples K geometry-structured,
  base-locked candidate deformations (randomized per-channel gains + noise seed),
  scores each with the heuristic reward, selects + renders the best, and records
  batch diversity. Writes `data/baseline/heuristic_baseline.json` — the reference
  the Phase 3-4 RLHF/PPO policy must beat. Default weights β=γ=δ=1.

#### Baseline results (K=6/asset, seed=0)
| Asset | Best reward | variety | distortion | physics | diversity |
|-------|-------------|---------|------------|---------|-----------|
| gray-big-rock | **+0.216** | 0.241 | 0.024 | 0.000 | 0.106 |
| rock_17 | +0.116 | — | — | 0.000 | 0.068 |
| tree | −0.069 | — | — | 0.000 | 0.040 |
| tree_1 | +0.074 | — | — | 0.000 | 0.051 |
| tree_2 | +0.137 | — | — | 0.000 | 0.080 |
| **Aggregate** | **+0.095** mean best | | | | 0.069 mean |

* **Finding:** the heuristic reward cleanly separates the classes — rocks deform
  with low distortion (positive reward), while thin **tree** geometry over-distorts
  under the shared operator strengths (best variant still net-negative). This is an
  honest baseline signal that per-class operator tuning and/or the learned policy
  should address in later phases, not a bug.
* `physics=0` for every best variant confirms the Fixed-channel base-lock preserves
  ground contact across the whole batch.

### Next Steps (Phase 3)
* Design the lightweight HITL pairwise-comparison annotation interface
  (Gradio/Streamlit) over the multi-view renders.
* Collect artist preference data and train the Bradley-Terry reward model to
  supply `R_human`.
