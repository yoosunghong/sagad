"""Append-only preference store for artist pairwise judgements.

One JSON object per line (JSONL): durable, crash-safe (each judgement is flushed
immediately), and trivially resumable -- the annotation UI skips ``pair_id``s
already present. Records feed the Bradley-Terry reward model (next Phase 3 task);
:meth:`PreferenceStore.to_comparisons` emits clean ``(winner, loser)`` pairs with
ties/skips dropped.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from data.logging_utils import get_logger

log = get_logger("hitl.store")

# Judgement vocabulary. "a"/"b" pick the corresponding candidate as more
# natural/useful; "tie" = indistinguishable; "skip" = annotator abstained.
VALID_CHOICES = ("a", "b", "tie", "skip")


@dataclass
class PreferenceRecord:
    """A single recorded pairwise judgement."""

    pair_id: str
    asset: str
    a: str               # candidate_id shown on the left
    b: str               # candidate_id shown on the right
    choice: str          # one of VALID_CHOICES
    annotator: str
    timestamp: float


class PreferenceStore:
    """Resumable JSONL store of pairwise preference judgements."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- write -------------------------------------------------------------
    def record(self, pair_id: str, asset: str, a: str, b: str, choice: str,
               annotator: str) -> PreferenceRecord:
        if choice not in VALID_CHOICES:
            raise ValueError(f"choice must be one of {VALID_CHOICES}; got {choice!r}")
        rec = PreferenceRecord(
            pair_id=pair_id, asset=asset, a=a, b=b, choice=choice,
            annotator=annotator, timestamp=time.time(),
        )
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(rec)) + "\n")
            fh.flush()
        log.info("recorded %s | asset=%s choice=%s by=%s", pair_id, asset, choice, annotator)
        return rec

    # -- read --------------------------------------------------------------
    def load(self) -> list[PreferenceRecord]:
        if not self.path.exists():
            return []
        records: list[PreferenceRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(PreferenceRecord(**json.loads(line)))
        return records

    def annotated_pair_ids(self) -> set[str]:
        """``pair_id``s with at least one judgement -- used to resume a session."""
        return {r.pair_id for r in self.load()}

    def count(self) -> int:
        return len(self.load())

    def to_comparisons(self) -> list[dict]:
        """Clean ``(winner, loser)`` pairs for Bradley-Terry; ties/skips dropped."""
        out: list[dict] = []
        for r in self.load():
            if r.choice == "a":
                out.append({"asset": r.asset, "winner": r.a, "loser": r.b})
            elif r.choice == "b":
                out.append({"asset": r.asset, "winner": r.b, "loser": r.a})
        return out
