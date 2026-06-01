"""CLI: launch the Gradio pairwise-comparison annotation UI (Phase 3 HITL).

Loads an annotation manifest (built by ``scripts/build_preference_pairs.py``) and
serves the side-by-side comparison interface. Judgements stream into a resumable
JSONL preference store; relaunching skips pairs already annotated.

Usage:
    python scripts/annotate_ui.py [--annotator NAME] [--manifest PATH]
        [--store PATH] [--port 7860] [--share]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.logging_utils import get_logger  # noqa: E402
from hitl import PreferenceStore, load_manifest  # noqa: E402
from hitl.app import build_app  # noqa: E402

log = get_logger("annotate_ui")


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the HITL annotation UI.")
    parser.add_argument("--annotator", default="artist",
                        help="label stored with each judgement")
    parser.add_argument("--manifest", default=str(ROOT / "data" / "preferences" / "manifest.json"))
    parser.add_argument("--store", default=str(ROOT / "data" / "preferences" / "preferences.jsonl"))
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="expose a public Gradio link")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        log.error("manifest not found: %s -- run scripts/build_preference_pairs.py first",
                  manifest_path)
        return 1

    manifest = load_manifest(manifest_path)
    store = PreferenceStore(args.store)
    log.info("manifest=%s | %d pairs | store=%s (%d existing judgements)",
             manifest_path, len(manifest.pairs), args.store, store.count())

    app = build_app(manifest, store, annotator=args.annotator, root=ROOT)
    app.launch(server_port=args.port, share=args.share, inbrowser=not args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
