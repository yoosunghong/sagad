# Phase 4.5 Progress Log: Variant Diffusion Sampler & Texture Safety

## Status: ✅ Offline variant-generation chain complete & validated (UE5 bake is next)

Bridges Phase 4 (single optimal mask) → Phase 5 (UE5 production) by adding the
**variation engine**: given one mesh file, sample a *distribution* of texture-safe
per-instance variants to scatter as WPO/HISM instances. Motivated by the Phase 4
finding that a single PPO policy converges to one mask, whereas "many instances
from one file" is inherently a *distribution to sample* — the natural job of a
diffusion model, not a policy.

### Executed Tasks
- [x] **Texture-integrity penalty** (`src/rewards/penalties.py`, `edge_stretch`) —
  promotes the sandbox's `edge_len_rel_change` telemetry to a *reward term* in the
  distortion group, guarding texel-smear / UV-seam tearing under WPO.
- [x] **Variant parameter schema** (`src/diffusion/params.py`, `VariantParamSpec`) —
  typed raw↔model-space bijection bridging the sampler and the Per-Instance Custom
  Data bake; `organic_spec` / `building_spec` factories; integer fields (floor count).
- [x] **DDPM sampler** (`src/diffusion/ddpm.py`) — ε-prediction diffusion over the
  low-dim variant vector (sinusoidal time embed, GELU MLP, optional MeshMAE-latent
  conditioning), NaN-guarded train + ancestral sample.
- [x] **Texture-safe bootstrap** (`scripts/build_variant_dataset.py`) — propose →
  sandbox-deform → penalty-score → keep feasible → train DDPM, no human labels.
- [x] **Visualization** — `scripts/visualize_transform.py` (displacement heatmap +
  stretch distributions) and `scripts/visualize_variant_grid.py` (diversity grid).
- [x] **Unit tests** (`tests/test_diffusion.py`) — param round-trip / integer clamp
  and bimodal-recovery (no mode collapse).

### Technical Notes

#### New isolated module (`src/diffusion/`, separate from deform/rewards/rl/hitl)
CLAUDE.md modularity rule. Performs no deformation/rendering — learns and samples
the variant-parameter distribution only. Channel map stays the locked
`[R=bend, G=noise, B=scale, A=fixed]`; no virtual channels.

#### The single-mesh constraint (drives the whole design)
One Static Mesh asset carries **one** baked vertex-color buffer shared by every
instance. Per-instance variation therefore cannot live in vertex color — it travels
through UE5 **Per-Instance Custom Data** (a short float vector). So:
* **Vertex color (shared, baked once):** static per-vertex structure (region masks,
  floor index).
* **Per-Instance Custom Data (varies):** the decoded variant vector — what the DDPM
  samples.
* **WPO shader:** combines the two → each instance's unique deformation.

#### Texture-safe by construction
The bootstrap trains only on deformations whose §2.4 penalties (incl. the new
`stretch` term) are feasible, so the learned distribution avoids texture-breaking
masks. With no artist data, the proposal distribution is uniform over the param box,
rejection-filtered by the deterministic penalties — uses existing constraints, zero
new labels.

### Validation

#### Diffusion sanity (`tests/test_diffusion.py`)
Bimodal target recovered with **no mode collapse** (47/53 split, mode-mean error
0.016 / 0.053); param round-trip exact; integer floor-count clamps correctly. PASS.

#### Texture-safe bootstrap (`gray-big-rock`, N=7106, 300 proposals)
| metric | value |
|--|--:|
| feasible kept | 150 / 300 (distortion ≤ 0.207) |
| train distortion mean | 0.149 |
| **sampled distortion mean** | **0.152** (sampler reproduces feasible quality) |
| **sampled texture-safe fraction** | **90.2 %** |

The DDPM learned the *shape* of the feasible region — 9 of 10 fresh samples land
inside the texture-safe envelope (not memorization; distinct distortion-matched
variants). Artifacts: `data/diffusion/variant_diffusion_gray-big-rock.pt`,
`variant_report_gray-big-rock.json`.

#### Visualization (`data/diffusion/viz/`)
* `transform_gray-big-rock.png` — original vs. displacement-heatmap orbit; the
  representative variant `{bend .60, noise .27, scale .51}` shows max disp 0.571,
  mean 0.178, **p99 edge-stretch 51.8 %** (the busy tail the `stretch` penalty
  exists to bound), locked-base 8 %.
* `transform_*_stats.png` — per-vertex displacement + per-edge stretch histograms.
* `variant_grid_gray-big-rock.png` — 6 variants from one mesh, shared color scale:
  visibly diverse displacement patterns, all sharing the blue locked base.
* `*_heatmap.glb` — interactive (orbit/zoom) displacement-shaded mesh.

### Environment note
Same `sagad` conda env (`C:\Users\PC\anaconda3\envs\sagad`). Renderer is the Open3D
legacy `Visualizer` (offscreen) from Phase 2, extended with an additive
`vertex_colors` argument (backward-compatible) for the heatmap shading.

### Next Steps (→ Phase 5, UE5 production)
1. **Seam-coherent vertex-color bake.** Write the shared `[R,G,B,A]` masks into the
   Static Mesh vertex-color buffer; **coincident UV-seam duplicates must receive
   identical RGBA** (merge-then-average) or WPO tears the seam (ARCHITECT §2.4 note).
2. **WPO master material.** Sample the four vertex-color channels (bend/noise/scale +
   mobility=1−fixed) into the World-Position-Offset accumulator; read the variant
   vector from **Per-Instance Custom Data** (and, for buildings, a `floor_index`
   vertex-color channel collapsed above the per-instance floor count).
3. **HISM/Nanite scatter tool.** At placement, `diffusion.sample(K)` →
   `spec.from_model` → write K variant vectors as Per-Instance Custom Data.
4. **Per-asset bootstrap sweep.** Run `build_variant_dataset.py` on the tree assets;
   add the optional MeshMAE-latent conditioning so one sampler serves multiple assets.
5. **Profile** FPS / draw calls / VRAM / shader instructions on the HISM stress scene
   (PLAN Phase 5).
