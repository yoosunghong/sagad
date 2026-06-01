# Phase 4 Progress Log: PPO Optimization Loop Integration

## Status: ✅ Machinery complete & convergent (tuning is follow-up)

### Executed Tasks
- [x] Assemble the full RL environment using the Gymnasium interface (`src/rl/env.py`).
- [x] Configure the PPO actor-critic to predict continuous vertex-level masks (`src/rl/policy.py`).
- [x] Integrate the composite reward
  $R = \alpha R_{human} + \beta R_{variety} - \gamma R_{distortion} - \delta R_{physics}$
  (`src/rl/composite.py`; `R_human` render+CLIP path verified end-to-end).
- [x] Run optimization loops + monitor convergence with WandB (`scripts/train_ppo.py`).

### Technical Notes

#### New isolated module (`src/rl/`, separate from deform/rewards/models)
Orchestrates the lower-level stages into the RL loop (CLAUDE.md modularity rule):

* **`composite.py`** — `composite_reward(variety, distortion, physics, human, weights)`
  → the four-term composite. Generalizes the Phase 2 `heuristic_reward` by adding
  the learned `alpha*R_human`. NaN-guarded on every term.
* **`policy.py`** — `MaskActorCritic`, a **weight-shared per-node** MLP over the
  MeshMAE latent $Z[N,D]$ → per-node logit means; the value head is applied per
  node and **mean-pooled** to a scalar $V(Z)$. The policy is invariant to vertex
  count $N$ (ARCHITECT §2.2).
* **`env.py`** — `DeformEnv`, a single-asset Gymnasium env. **Single-step
  (contextual-bandit)** episodes: obs = frozen $Z$, `step(mask)` → sandbox deform
  → penalties → composite reward → terminate. `R_human` enters via a
  `human_reward_fn(mesh)` callback so the env stays decoupled from CLIP.

#### Continuous-action design (the key correctness fix)
Actions are sampled in **logit space** (diagonal Gaussian, global per-channel
`log_std`) then sigmoid-squashed to $[0,1]$ — avoids the density spikes a
Beta/clipped-Gaussian-on-$[0,1]$ hits at the boundary.

* **Averaged (not summed) log-prob.** The action has $N\times4 \approx 28\text{k}$
  dims. The first run summed the joint log-prob → the PPO ratio
  $\exp(\sum \text{tiny per-dim diffs})$ exploded (`approx_kl≈2.5`, clipping
  saturated, **reward flat at −0.84**). Averaging the log-prob/entropy over the
  action dims keeps the ratio $O(1)$; after the fix `approx_kl≈0.02` and the
  **reward climbed −0.847 → −0.018**. This is recorded in `policy.py` so the
  choice is not silently "optimized away" later.

#### PPO loop (`scripts/train_ppo.py`)
Frozen MeshMAE encoder (`meshmae_baseline.pt`) → $Z$ once; batch of single-step
rollouts on the same $Z$; advantages $A = R - V(Z)$ (batch-normalized); clipped
surrogate + 0.5·value loss + small entropy bonus; grad-norm clip 1.0. Inner-module
INFO logging is silenced during rollouts. **WandB offline by default** (no
login/network; `wandb sync` later). NaN guard on the PPO loss.

### Validation

#### Geometric composite convergence (α=0, `gray-big-rock`, 80 iters × batch 16)
| | reward (mean) | distortion | variety | approx_kl |
|--|--:|--:|--:|--:|
| start | −0.847 | ~0.95 | ~0.10 | — |
| end   | **−0.018** (best −0.007) | 0.187 | 0.169 | 0.016 |

Monotonic improvement, stable KL, NaN-free, policy saved to
`data/rl/ppo_policy_gray-big-rock.pt` + `ppo_report_*.json`. The free-form policy
learns to cut distortion ~5× and raise variety from a random start.

* **Honest gap:** it does **not** yet beat the hand-structured heuristic baseline
  (**+0.216**) — the baseline bakes in a base-locked, height-weighted Bend prior
  the free policy must discover from scratch, and α=0 means no artist signal.
  Closing this is a **tuning** task (reward shaping / base-lock inductive bias /
  longer training / per-class weights), not a machinery defect.

#### `R_human` integration (α=0.5, tiny run)
Verified the expensive path end-to-end: CLIP ViT-B/32 loads, each rollout is
rendered (4 views) + embedded + scored by the trained Bradley-Terry model, and
`r_human` flows into the composite (`alpha * r_human`). **Caveat:** the reward
model's raw output is only ordinally meaningful (BT training fixes ranking, not
absolute scale), so before α>0 is used in earnest `R_human` needs centering /
scaling against the geometric terms.

### Environment note
Same `sagad` conda env as prior phases (`C:\Users\PC\anaconda3\envs\sagad`); the
bare `python` on PATH is the broken Windows Store stub. CLIP requires
`use_safetensors=True` (see Phase 3 log).

### Next Steps
* Tune the composite to **beat the +0.216 baseline** (base-lock prior, reward
  shaping, per-class operator strengths, longer runs); sweep with WandB.
* Calibrate `R_human` scale, fold in **real** designer labels (the Gradio UI is
  live, writing `data/preferences/preferences.jsonl`), and retrain the reward
  model before enabling α>0 in production runs.
* Phase 5: bake the learned policy's masks into UE5 Static Mesh Vertex Colors.
