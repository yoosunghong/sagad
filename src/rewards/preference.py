"""Bradley-Terry preference reward model (the learned ``R_human`` term).

Implements the Human Preference Module of docs/ARCHITECT.md sec. 2.4: a reward
estimator over CLIP image features (see :mod:`hitl.features`) trained from artist
pairwise judgements with the Bradley-Terry logistic loss

    P(a > b) = sigmoid( r(a) - r(b) ),   L = -mean log sigmoid( r(win) - r(lose) ).

This supplies the ``alpha * R_human`` term of the Phase 4 composite reward. Kept
in ``src/rewards`` with the other reward components (CLAUDE.md: reward constraints
in dedicated modules), but distinct from the *deterministic* penalties in
:mod:`rewards.penalties` -- this term is *learned* from human (or oracle) feedback.

NaN guards (CLAUDE.md Logging Protocols) sit on the loss and on emitted rewards.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.logging_utils import get_logger

log = get_logger("rewards.preference")


class BradleyTerryReward(nn.Module):
    """Scalar reward head over a fixed feature vector.

    ``hidden=0`` gives the plain *linear* reward estimator of the spec; a small
    GELU MLP (default) adds mild capacity for the low-data preference regime.
    """

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
            )
        else:
            self.net = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.net(x).squeeze(-1)
        if not torch.isfinite(r).all():
            raise ValueError("NaN/Inf detected in reward output")
        return r


def bt_loss(r_win: torch.Tensor, r_lose: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry logistic loss for (winner, loser) reward pairs."""
    loss = -F.logsigmoid(r_win - r_lose).mean()
    if not torch.isfinite(loss):
        raise ValueError("NaN/Inf detected in Bradley-Terry loss")
    return loss


def pairwise_accuracy(r_win: torch.Tensor, r_lose: torch.Tensor) -> float:
    """Fraction of pairs the model orders correctly (r_win > r_lose)."""
    if r_win.numel() == 0:
        return float("nan")
    return float((r_win > r_lose).float().mean())


@dataclass
class TrainConfig:
    epochs: int = 400
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden: int = 128
    seed: int = 0
    log_every: int = 50


def train_reward_model(
    train_win: torch.Tensor, train_lose: torch.Tensor,
    val_win: torch.Tensor, val_lose: torch.Tensor,
    cfg: TrainConfig | None = None,
) -> tuple[BradleyTerryReward, list[dict]]:
    """Fit a Bradley-Terry reward model on (winner, loser) feature pairs.

    Full-batch Adam -- the preference set is small (tens-hundreds of pairs). The
    feature tensors are ``(P, dim)`` aligned so row ``i`` is the i-th comparison.
    Returns the trained model and an epoch-sampled history (loss + train/val
    pairwise accuracy) for telemetry.
    """
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    in_dim = train_win.shape[1]
    model = BradleyTerryReward(in_dim, cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    log.info("training BT reward | in_dim=%d hidden=%d | %d train / %d val pairs",
             in_dim, cfg.hidden, train_win.shape[0], val_win.shape[0])

    history: list[dict] = []
    for ep in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        loss = bt_loss(model(train_win), model(train_lose))
        loss.backward()
        opt.step()

        if ep % cfg.log_every == 0 or ep == cfg.epochs - 1:
            model.eval()
            with torch.no_grad():
                tr_acc = pairwise_accuracy(model(train_win), model(train_lose))
                va_acc = (pairwise_accuracy(model(val_win), model(val_lose))
                          if val_win.numel() else float("nan"))
            rec = {"epoch": ep, "loss": float(loss), "train_acc": tr_acc, "val_acc": va_acc}
            history.append(rec)
            log.info("  ep %3d | loss=%.4f train_acc=%.3f val_acc=%.3f",
                     ep, rec["loss"], tr_acc, va_acc)
    return model, history
