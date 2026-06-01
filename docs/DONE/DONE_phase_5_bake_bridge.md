# Phase 5 — Vertex-Color Bake + UE5 Subprocess Bridge (Steps 1 & 2)

Date: 2026-06-01

The first two Phase 5 deliverables: the Python **bake** that produces the shared
vertex-color buffer, and the **subprocess bridge** that lets the UE5 plugin pull
per-instance variant vectors from the diffusion sampler. Interface choice = subprocess
bridge (Python is the source of truth; UE5 calls it and parses JSON).

## Step 1 — Vertex-color bake (`scripts/bake_vertex_colors.py`)

Produces the single shared RGBA buffer every WPO instance reads. Written onto the
**undeformed** base mesh; the material deforms at runtime.

- **Mask source:** `--source policy` (trained PPO actor, deterministic `act`, via the
  same frozen MeshMAE encode as `train_ppo`) or `--source gains` (deterministic
  `mask_from_gains`).
- **Bake convention (ARCHITECT §3):** RGBA = `[R=bend, G=noise, B=scale, A=mobility]`
  with `A = 1 - fixed`, so `fixed=1 → A=0 → locked vertex`. Matches the sandbox gate
  and the shader lerp exactly.
- **Seam coherence (ARCHITECT §2.4):** `seam_coherent_average` quantizes positions
  (`--merge-decimals`, default 5), groups coincident UV-seam duplicates, and writes
  the per-group mean RGBA back to all members. **This is load-bearing:** the raw PPO
  policy assigned coincident duplicates RGBA differing by up to **0.35** on
  gray-big-rock (135 seam groups) — that delta would tear the seam under WPO; after
  merge it is 0. The geometry-structured gains mask is naturally seam-coherent
  (shift 0.0).
- **Outputs (`data/bake/<asset>/`):** `<asset>_baked.glb` (base mesh + COLOR_0;
  verified to round-trip a `(N,4)` vertex-color buffer), `<asset>_preview.glb`
  (deformed sanity check), `<asset>_bake.json` (channel map, source transform, seam
  + deform telemetry).
- **UE5 import:** glTF Interchange, *Vertex Color Import Option = Replace*.

## Step 2 — Subprocess bridge

### Python endpoint (`scripts/sample_variants.py`)
- Loads `data/diffusion/variant_diffusion_<asset>.pt`, rebuilds `VariantParamSpec`
  from `spec_fields`, samples K vectors, decodes to raw PICD units.
- **Contract:** exactly one JSON object on **stdout**; all logging on **stderr**
  (repo `get_logger` → `basicConfig` → stderr). Failures emit `{"error": ...}` + exit 1
  so the caller always gets parseable output. `fields` order = PICD float write order.
- Verified: clean stdout JSON at k=3/k=64, and the missing-asset error path.

### UE5 plugin side (`Plugins/sagad`)
- `USagadBridge` (`Source/sagad/Public/SagadBridge.h` + `Private/SagadBridge.cpp`):
  Blueprint-callable `SampleVariants(PythonExe, RepoRoot, Asset, K, Seed, …)` runs the
  endpoint via `FPlatformProcess::ExecProcess`, parses stdout with the `Json` module,
  and returns `OutFields` + `OutVariants` (`FSagadVariant.Values` aligned to fields).
  Defaults point at the conda `sagad` python.exe and the repo root (MEMORY: bare
  `python` is a broken Store stub).
- `sagad.Build.cs`: added `Json` to private deps.

## Step 3 — WPO material + scatter tool (plugin compiles; in-editor run pending)

### Scatter tool (`ASagadScatterActor`, `Source/sagad/{Public,Private}/SagadScatterActor.*`)
- A `UHierarchicalInstancedStaticMeshComponent` actor. `Scatter()` (CallInEditor):
  `USagadBridge::SampleVariants(asset, K, seed)` → `HISM.NumCustomDataFloats =
  Fields.Num()` → per instance `AddInstance(xf)` + `SetCustomDataValue(i, c, v)`.
- Placement: jittered grid in `AreaHalfSize`, random yaw, `InstanceScale`; the **same
  seed** drives the Python sampler and the grid RNG, so a scatter is reproducible.
- PICD layout matches the diffusion spec / material: `[0]=bend_gain, [1]=noise_gain,
  [2]=scale_gain, [3]=base_band`.
- Exposes `BakedMesh` (import `<asset>_baked.glb`) and `WpoMaterial` (M_SagadWPO).

### WPO material (`Content/Python/build_wpo_material.py` → `/sagad/Materials/M_SagadWPO`)
- In-editor Python builder (Python Editor Script Plugin). WPO accumulator, an
  in-shader approximation of the §2.3 sandbox:
  - `bend  = BendAxis    * (BendStrength  · R · bend_gain  · heightZ)`
  - `noise = VertexNormal * (NoiseStrength · G · noise_gain · Noise(worldPos))`
  - `scale = (worldPos − objPos) · (ScaleStrength · B · scale_gain)`
  - `WPO   = (bend + noise + scale) · A`  (A = mobility = 1 − fixed, from the bake)
- Strengths are ScalarParameters (tune to the asset's real UE size — the baked mesh is
  unit-normalized; see the bake sidecar `transform`). Noise frequency is a node
  property (not a pin), so it's a constant, not a parameter. `used_with_instanced_
  static_meshes` flag set.

### Usage in-editor
1. Import `data/bake/<asset>/<asset>_baked.glb` (glTF Interchange, Vertex Color =
   Replace).
2. `py ".../Plugins/sagad/Content/Python/build_wpo_material.py"` to build M_SagadWPO.
3. Drop `ASagadScatterActor`, set BakedMesh + WpoMaterial + Asset, click **Scatter**.

## Not yet done (remaining Phase 5)
- In-editor validation: run the material builder, scatter, and confirm K instances
  deform distinctly with no seam tears.
- HISM/Nanite stress scene + perf profiling (FPS, draw calls, VRAM, shader
  instructions).

## Repro
```
PY="C:/Users/PC/anaconda3/envs/sagad/python.exe"
$PY scripts/bake_vertex_colors.py --asset gray-big-rock --source gains
$PY scripts/sample_variants.py   --asset gray-big-rock --k 64 --seed 0
```
