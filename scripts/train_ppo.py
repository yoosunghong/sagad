"""CLI: PPO optimization of the per-vertex mask policy (docs/PLAN.md Phase 4).

Assembles the full RL stack -- MeshMAE latent encoder (frozen) -> node-shared
actor-critic -> Gymnasium DeformEnv -> composite reward -- and optimizes the
policy with PPO. Because each episode is a single step (predict a mask, score
it), the loop collects a batch of sampled masks for the *same* asset latent,
computes single-step advantages ``A = R - V(Z)``, and applies the clipped PPO
surrogate with a value baseline and entropy bonus.

``R_human`` (the learned Bradley-Terry term) is enabled with ``--alpha > 0`` and
``--reward-model``: each rollout is rendered and CLIP-scored. This is the
expensive path, so it defaults off; the geometric composite
(``beta*variety - gamma*distortion - delta*physics``) validates convergence
cheaply, and the final mean reward is compared against the Phase 2 heuristic
baseline.

Convergence is monitored with Weights & Biases (offline by default -- no login /
network needed; sync later with ``wandb sync``).

Usage:
    python scripts/train_ppo.py [--asset NAME] [--iters N] [--batch B]
        [--alpha A --reward-model PATH] [--wandb-online] [--seed S]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.logging_utils import get_logger  # noqa: E402
from models import MeshMAE, MeshMAEConfig  # noqa: E402
from rl import CompositeWeights, DeformEnv, MaskActorCritic  # noqa: E402

log = get_logger("train_ppo")


def _quiet(*names: str) -> None:
    """Silence per-rollout INFO spam from inner modules during training."""
    for n in names:
        logging.getLogger(n).setLevel(logging.WARNING)


def _build_human_reward_fn(reward_model_path: Path, device: torch.device):
    """Build ``mesh -> R_human`` from the trained reward model + CLIP featurizer."""
    from hitl.features import CLIPFeaturizer
    from render import RenderConfig, render_multiview
    from rewards import BradleyTerryReward

    ckpt = torch.load(reward_model_path, weights_only=False)
    model = BradleyTerryReward(ckpt["in_dim"], ckpt["hidden"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    fz = CLIPFeaturizer(device)
    rcfg = RenderConfig(n_views=4, image_size=224)

    @torch.no_grad()
    def human_reward(mesh) -> float:
        views = render_multiview(mesh, rcfg)
        feat = fz.embed(list(views)).mean(dim=0).to(device)
        return float(model(feat.unsqueeze(0)).item())

    return human_reward


def main() -> int:
    parser = argparse.ArgumentParser(description="PPO mask-policy optimization (Phase 4).")
    parser.add_argument("--asset", default="gray-big-rock")
    parser.add_argument("--weights", default=str(ROOT / "data" / "processed" / "meshmae_baseline.pt"))
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16, help="rollouts per PPO iteration")
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.0, help="R_human weight (enables render+CLIP)")
    parser.add_argument("--reward-model", default=str(ROOT / "data" / "preferences" / "reward_model.pt"))
    parser.add_argument("--wandb-online", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "data" / "rl"))
    args = parser.parse_args()

    _quiet("deform.sandbox", "rewards.penalties", "rl.env", "render.multiview", "hitl.features")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- frozen MeshMAE latent for the asset -------------------------------
    data = torch.load(ROOT / "data" / "processed" / f"{args.asset}.pt", weights_only=False).to(device)
    ckpt = torch.load(args.weights, weights_only=False)
    encoder = MeshMAE(MeshMAEConfig(**ckpt["config"])).to(device)
    encoder.load_state_dict(ckpt["state_dict"])
    encoder.eval()
    with torch.no_grad():
        Z = encoder.encode(data).detach()                 # (N, D), frozen
    log.info("asset=%s | N=%d D=%d | device=%s", args.asset, Z.shape[0], Z.shape[1], device)

    # -- env + composite reward -------------------------------------------
    weights = CompositeWeights(human=args.alpha)
    human_fn = None
    if args.alpha > 0.0:
        log.info("R_human ENABLED (alpha=%.3f) -- render+CLIP per rollout", args.alpha)
        human_fn = _build_human_reward_fn(Path(args.reward_model), device)
    env = DeformEnv(data, Z, weights=weights, human_reward_fn=human_fn)

    policy = MaskActorCritic(Z.shape[1]).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    # -- wandb (offline by default) ---------------------------------------
    run = None
    try:
        import wandb
        run = wandb.init(project="sagad-ppo", mode="online" if args.wandb_online else "offline",
                         config=vars(args), reinit=True)
    except Exception as e:                                 # never let logging break training
        log.warning("wandb unavailable (%s); continuing without it", e)

    baseline = None
    bpath = ROOT / "data" / "baseline" / "heuristic_baseline.json"
    if bpath.exists():
        rep = json.loads(bpath.read_text())
        for a in rep.get("per_asset", []):
            if a["asset"] == args.asset:
                baseline = a["best"]["reward"]

    history = []
    best_reward = -float("inf")
    for it in range(args.iters):
        # -- collect a batch of single-step rollouts ----------------------
        logits_b, logp_b, rew_b, breakdown = [], [], [], []
        for _ in range(args.batch):
            mask, logits, logp, _ = policy.act(Z)
            _, reward, _, _, info = env.step(mask.reshape(-1).cpu().numpy())
            logits_b.append(logits); logp_b.append(logp)
            rew_b.append(reward); breakdown.append(info)
        logits_b = torch.stack(logits_b)                   # (B, N, 4)
        old_logp = torch.stack(logp_b).detach()            # (B,)
        returns = torch.tensor(rew_b, dtype=torch.float32, device=device)
        adv = returns - policy.value(Z).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # -- PPO clipped update -------------------------------------------
        for _ in range(args.update_epochs):
            new_logp, entropy = policy.evaluate_actions(Z, logits_b)
            ratio = (new_logp - old_logp).exp()
            surr = torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv)
            policy_loss = -surr.mean()
            value_loss = (policy.value(Z) - returns).pow(2).mean()
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy
            if not torch.isfinite(loss):
                raise ValueError("NaN/Inf detected in PPO loss")
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

        with torch.no_grad():
            approx_kl = float((old_logp - new_logp).mean())
        mean_reward = float(returns.mean())
        best_reward = max(best_reward, mean_reward)
        metrics = {
            "iter": it,
            "reward_mean": mean_reward,
            "reward_max": float(returns.max()),
            "variety": float(np.mean([b["variety"] for b in breakdown])),
            "distortion": float(np.mean([b["distortion"] for b in breakdown])),
            "physics": float(np.mean([b["physics"] for b in breakdown])),
            "r_human": float(np.mean([b["r_human"] for b in breakdown])),
            "policy_loss": float(policy_loss), "value_loss": float(value_loss),
            "entropy": float(entropy), "approx_kl": approx_kl,
            "grad_norm": float(grad_norm),
        }
        history.append(metrics)
        if run is not None:
            run.log(metrics)
        if it % 5 == 0 or it == args.iters - 1:
            log.info("it %3d | reward(mean/max)=%.4f/%.4f var=%.3f dist=%.3f phys=%.3f "
                     "| ploss=%.3f vloss=%.3f ent=%.1f kl=%.4f",
                     it, mean_reward, metrics["reward_max"], metrics["variety"],
                     metrics["distortion"], metrics["physics"],
                     metrics["policy_loss"], metrics["value_loss"],
                     metrics["entropy"], approx_kl)

    # -- save policy + report ---------------------------------------------
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "latent_dim": Z.shape[1], "asset": args.asset},
               out / f"ppo_policy_{args.asset}.pt")
    report = {
        "asset": args.asset, "iters": args.iters, "batch": args.batch,
        "alpha": args.alpha, "final_reward_mean": history[-1]["reward_mean"],
        "best_reward_mean": best_reward, "heuristic_baseline_best": baseline,
        "history": history,
    }
    (out / f"ppo_report_{args.asset}.json").write_text(json.dumps(report, indent=2))
    if run is not None:
        run.finish()

    first = history[0]["reward_mean"]
    log.info("=== PPO DONE === asset=%s | reward %.4f -> %.4f (best %.4f)",
             args.asset, first, history[-1]["reward_mean"], best_reward)
    if baseline is not None:
        verdict = "BEATS" if best_reward > baseline else "below"
        log.info("  heuristic baseline best=%.4f | policy %s baseline", baseline, verdict)
    log.info("  policy -> %s", out / f"ppo_policy_{args.asset}.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
