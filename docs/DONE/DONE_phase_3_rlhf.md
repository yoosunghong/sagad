# Phase 3 Progress Log: RLHF Framework & Reward Model Training

## Status: ▢ In progress

### Executed Tasks
- [x] Design a lightweight Human-in-the-Loop (HITL) annotation interface
  (Gradio) for pairwise comparison (`src/hitl/`, `scripts/build_preference_pairs.py`,
  `scripts/annotate_ui.py`).
- [~] Collect initial artist preference data (Asset A vs. Asset B) — **UI ready**;
  bootstrapped with a **synthetic-oracle** label set pending a real artist session
  (`scripts/simulate_preferences.py`).
- [x] Train a Bradley-Terry preference reward model (`src/rewards/preference.py`,
  `src/hitl/features.py`, `scripts/train_reward_model.py`).
- [x] Validate reward model correlation with qualitative scores (validated against
  the synthetic oracle; re-run on real labels when collected).

### Technical Notes

#### New isolated module (`src/hitl/`, separate from deform/rewards per modularity rule)
The human-feedback layer is kept strictly separate from the data graphs
(`src/data`), deformation sandbox (`src/deform`), and the deterministic reward
constraints (`src/rewards`) — CLAUDE.md §2 modularity. Three single-responsibility
files:

* **`candidates.py`** — candidate-pool generation + pairing schedule.
  * `geometry_structured_mask` reuses the Phase 2 candidate recipe (height-weighted
    Bend ramp + base-pinned Fixed band) with **wider gain ranges** and a **jittered
    base band**, so the pool spans deformations an artist can actually discriminate
    on naturalness (vs. the narrow heuristic-baseline sampler).
  * `build_candidate_pool` deforms each candidate through the §2.3 sandbox and
    renders a §2.3 multi-view **contact sheet**; `build_pairs` samples **within-asset**
    comparison pairs (cross-asset naturalness comparison is meaningless) with
    **randomized A/B placement** to remove positional bias. Full round-robin by
    default, or `--pairs-per-asset M` to subsample.
  * `save_manifest`/`load_manifest` persist the job to `data/preferences/manifest.json`.
* **`store.py`** — `PreferenceStore`, an **append-only JSONL** store. One record per
  judgement `{pair_id, asset, a, b, choice ∈ {a,b,tie,skip}, annotator, timestamp}`,
  flushed immediately (crash-safe). `annotated_pair_ids()` powers **resume**;
  `to_comparisons()` emits clean `(winner, loser)` pairs (ties/skips dropped) — the
  direct training signal for the next task's Bradley-Terry model. Invalid `choice`
  values are rejected.
* **`app.py`** — `build_app` assembles a Gradio `Blocks` side-by-side UI
  (`◀ A better` / `Tie` / `Skip` / `B better ▶`) over the **un-annotated** pair
  queue; each click streams a record to the store and advances. Build is decoupled
  from launch so the UI layer stays import-safe / headless-testable.

#### Channel-map / I/O alignment (CLAUDE.md domain constraint — docs before code)
`docs/ARCHITECT.md` §2.4 was extended first to specify the HITL pipeline I/O: the
manifest schema, the within-asset randomized pairing, and the JSONL preference-record
schema feeding the Bradley-Terry loss. No new mask channels were introduced — the
candidate masks remain the locked `[R=Bend, G=Noise, B=Scale, A=Fixed]` convention.

#### Telemetry (CLAUDE.md Logging Protocols)
Each candidate persists its vertex-degradation scales into the manifest
(`disp_mean`, `disp_max`, `locked_fraction`, `edge_len_rel_change_max`) and surfaces
them in the UI caption, so an annotator sees *why* a candidate looks over-deformed
and the downstream reward model can be cross-checked against geometry. `isfinite`
guards are inherited from the sandbox; the renderer's empty-frame warning still
applies.

### Validation
* **Unit (store / pairing / app build):** JSONL round-trip + resume set, `to_comparisons`
  drops ties/skips and orients winner/loser correctly, invalid `choice` rejected,
  `build_pairs` yields `C(n,2)` within-asset pairs with correct asset tagging, and
  `build_app` wires the Gradio `Blocks` without launching. **All PASS.**
* **End-to-end:** `build_preference_pairs.py` (default 5 assets × 6 candidates,
  seed 0) generated **30 candidates / 75 pairs** (`C(6,2)=15` per asset), rendering
  all contact sheets (non-empty foreground) to `data/preferences/renders/` and
  writing `data/preferences/manifest.json`. Manifest reload confirms every render
  path resolves to an on-disk file and every pair references same-asset candidates.
* **Resume:** recording one judgement shrinks the served queue from 75 → 74 on the
  next `build_app`, confirming crash-safe resumability.

### Environment note
The repo's Python is the conda env **`sagad`** (`C:\Users\PC\anaconda3\envs\sagad`,
py 3.10.20, torch 2.5.1+cu121, gradio 6.15.2, CUDA available). The bare `python` on
PATH is the Windows Store stub (exit 49) — use the env interpreter directly.

### Bradley-Terry Reward Model (`R_human`)

The learned human-reward term that closes Phase 3 and feeds the `α·R_human` slot
of the Phase 4 composite. Kept in `src/rewards` with the other reward components
but distinct from the *deterministic* penalties — this term is **learned**.

#### Candidate featurizer (`src/hitl/features.py`)
A deformation is scored from its *appearance*, not raw geometry (ARCHITECT §2.4).
`CLIPFeaturizer` embeds each candidate's multi-view renders with **CLIP ViT-B/32**
(`get_image_features`, L2-normalized), mean-pooled across views → a 512-d feature;
`candidate_features` caches them (`data/preferences/clip_features.pt`) so the
one-time CLIP pass is not repeated.
* **Gotcha (recorded):** `use_safetensors=True` is **mandatory** — transformers 5.9
  refuses to `torch.load` the `.bin` checkpoint on torch 2.5.1 (CVE-2025-32434).
  Also `get_image_features` returns a wrapped output in transformers ≥5, so the
  embedding is unwrapped via `image_embeds` / `pooler_output`.

#### Reward model + loss (`src/rewards/preference.py`)
* `BradleyTerryReward` — scalar reward head over the CLIP feature; plain **linear**
  estimator (`hidden=0`, the spec's "linear reward estimator") or a small GELU MLP
  (default `hidden=128`).
* `bt_loss` — `L = -mean log σ(r(win) − r(lose))`; `isfinite` guards on the reward
  output and the loss (CLAUDE.md NaN protocol).
* `train_reward_model` — full-batch Adam over the aligned `(winner, loser)` feature
  pairs from `store.to_comparisons()`; emits per-epoch loss + train/val pairwise
  accuracy telemetry.

#### Synthetic-oracle bootstrap (`scripts/simulate_preferences.py`)
No real artist is in the loop, so to bring up + validate the trainer we label the
manifest pairs from a deterministic oracle: each candidate is re-scored with the
Phase 2 heuristic reward (rebuilt exactly from the manifest gains via
`hitl.mask_from_gains`), and a label is **Bradley-Terry-sampled**,
`P(a≻b)=σ((rₐ−r_b)/τ)`, with `τ=0.05` injecting realistic noise. Streams into a
`PreferenceStore` (`annotator=synthetic-oracle`) on the *same* `to_comparisons()`
interface the real UI uses, and saves per-candidate oracle scores for the
correlation check. **Explicitly not real artist feedback** — it exercises the
machinery so the trainer runs unchanged on real labels later.

#### Training + validation (`scripts/train_reward_model.py`)
75 synthetic labels over the 30-candidate / 75-pair manifest (label↔oracle
agreement 86.7% at τ=0.05 → ~0.87 is the achievable val-accuracy ceiling), 80/20
pair split, CLIP features cached.

| Variant | train pair-acc | **val pair-acc** | Spearman(reward, oracle) | Pearson |
|---------|---------------:|-----------------:|-------------------------:|--------:|
| MLP (hidden=128, default) | 0.933 | **0.800** | **0.585** | 0.615 |
| Linear (hidden=0)         | 0.850 | 0.800 | 0.549 | 0.586 |

* **Held-out pairwise accuracy 0.80** sits right at the 0.867 label-noise ceiling
  — the model recovers the annotator's ordering on unseen pairs about as well as
  the noisy labels allow. The MLP's train/val gap (0.93/0.80) is mild overfit on
  only 60 train pairs × 512-d features; the linear head closes the gap with equal
  val accuracy and is the more spec-faithful "linear reward estimator".
* **Positive rank correlation** with the geometric oracle (Spearman 0.585) — the
  CLIP-appearance reward and the heuristic geometry reward agree in direction
  without being identical, which is the expected/honest result (the human term is
  meant to capture what geometry penalties cannot).
* Artifacts: `reward_model.pt` (default MLP), `reward_report.json`,
  `clip_features.pt`, `oracle_rewards.json`, `preferences_synthetic.jsonl`.

### Addendum — interactive 3D annotation UI + real designer labels

**Designer feedback:** under the static grayscale multi-view contact sheets the
two options were hard to tell apart (especially trees), so judging was unreliable.

**Fix — 3D viewer.** The annotation UI now shows each candidate as an
**interactive 3D mesh** (`gr.Model3D`, orbit/zoom) instead of a 2D image:
* `candidates.py` also exports each deformed candidate as a **GLB**
  (`data/preferences/meshes/`), rotated Z-up→Y-up via `_export_viewer_mesh` so
  assets stand upright by default. The PNG contact sheets are *still* produced —
  the §2.4 CLIP featurizer/reward model consumes them.
* `app.py` swaps the two `gr.Image` panes for `gr.Model3D`. Manifest regeneration
  is deterministic (same seed ⇒ identical candidate/pair ids), so existing
  judgements stay valid; verified all 30 GLBs render and the app builds.

**Real-label training (designer kept the 75 existing labels).** Retrained the
Bradley-Terry model on `data/preferences/preferences.jsonl` (73 comparisons after
dropping ties/skips):

| | train pair-acc | val pair-acc | Spearman(reward, geom-oracle) |
|--|--:|--:|--:|
| real designer labels | 0.862 | **0.733** | **−0.094** |

* Held-out accuracy 0.73 is above chance but modest, and the learned reward is
  **uncorrelated with the geometry heuristic** (Spearman ≈ 0) — i.e. the labels
  carry only weak, geometry-independent signal. This is consistent with the
  designer's report that the 2D UI made discrimination hard, so the labels are
  likely noisy. The 3D viewer is the remedy; a cleaner re-annotation round is
  expected to raise both held-out accuracy and signal strength.
* `reward_model.pt` / `reward_report.json` now reflect the **real-label** model
  (the synthetic-oracle results above are retained as the pipeline bring-up record).

### Next Steps
* Have the designer run a **3D re-annotation round** (archive the current labels
  first if a clean slate is wanted) and re-train; compare held-out accuracy.
* Phase 4: wire `R_human` into the composite reward
  `R = α·R_human + β·R_variety − γ·R_distortion − δ·R_physics` and assemble the
  Gymnasium PPO environment.
