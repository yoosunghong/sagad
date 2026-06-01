"""Gradio pairwise-comparison annotation UI (Phase 3 HITL interface).

Presents two candidate deformations of the *same* base asset side by side as
**interactive 3D meshes** (orbit / zoom) and asks the artist which is more
natural / artistically useful -- static grayscale renders made similar trees
hard to tell apart, so the annotator inspects the geometry directly. Each
judgement is written immediately to a :class:`PreferenceStore` (resumable), and
the queue auto-skips pairs already annotated so a session can be stopped and
resumed safely.

This module only *builds* the interface (``build_app``); launching is left to
``scripts/annotate_ui.py`` so the UI layer stays import-safe and headless-test
friendly (CLAUDE.md modularity / export-script isolation).
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from data.logging_utils import get_logger
from .candidates import Manifest
from .store import PreferenceStore

log = get_logger("hitl.app")

_INSTRUCTIONS = (
    "### Pairwise deformation preference\n"
    "Both options are the **same base asset** deformed differently — shown as "
    "**interactive 3D meshes** (drag to rotate, scroll to zoom). Pick the one "
    "that looks **more natural and artistically useful** (organic lean, believable "
    "surface variation, no broken/spiky geometry). Use **Tie** if they are "
    "indistinguishable, **Skip** to abstain."
)


def _resolve(path: str, root: Path) -> str:
    """Resolve a manifest path (root-relative or absolute) to an absolute path."""
    p = Path(path)
    return str(p if p.is_absolute() else (root / p))


def _caption(spec) -> str:
    g = spec.gains
    t = spec.telemetry
    return (f"`{spec.candidate_id}` — bend={g['bend']} noise={g['noise']} "
            f"scale={g['scale']} | disp_mean={t['disp_mean']:.3f} "
            f"edge_max={t.get('edge_len_rel_change_max') or float('nan'):.3f}")


def build_app(manifest: Manifest, store: PreferenceStore, annotator: str,
              root: Path) -> gr.Blocks:
    """Assemble the annotation Blocks app over an un-annotated pair queue."""
    cand = manifest.candidate_map()
    done = store.annotated_pair_ids()
    queue = [p for p in manifest.pairs if p.pair_id not in done]
    total = len(manifest.pairs)
    log.info("annotator=%s | %d/%d pairs remaining (%d already done)",
             annotator, len(queue), total, len(done))

    def _view(idx: int):
        """Return UI updates for queue position ``idx``."""
        if idx >= len(queue):
            msg = f"✅ Finished. {store.count()} judgements recorded for this store."
            return (gr.update(value=None), gr.update(value=None), gr.update(value=None),
                    gr.update(value="—"), gr.update(value="—"),
                    msg, gr.update(interactive=False), gr.update(interactive=False),
                    gr.update(interactive=False), gr.update(interactive=False), idx)
        pair = queue[idx]
        a, b = cand[pair.a], cand[pair.b]
        orig = manifest.originals.get(pair.asset)
        progress = (f"**Pair {idx + 1} / {len(queue)}**  &nbsp;·&nbsp; "
                    f"asset: `{pair.asset}`  &nbsp;·&nbsp; "
                    f"{store.count()} recorded so far")
        return (_resolve(orig, root) if orig else None,
                _resolve(a.mesh, root), _resolve(b.mesh, root),
                _caption(a), _caption(b), progress,
                gr.update(interactive=True), gr.update(interactive=True),
                gr.update(interactive=True), gr.update(interactive=True), idx)

    def _submit(choice: str, idx: int):
        if idx < len(queue):
            pair = queue[idx]
            store.record(pair.pair_id, pair.asset, pair.a, pair.b, choice, annotator)
        return _view(idx + 1)

    with gr.Blocks(title="HITL Deformation Preference") as app:
        gr.Markdown(_INSTRUCTIONS)
        progress_md = gr.Markdown()
        with gr.Row():
            img_orig = gr.Model3D(label="Original (undeformed reference)", interactive=False,
                                  height=340, clear_color=(1.0, 1.0, 1.0, 1.0))
        with gr.Row():
            with gr.Column():
                img_a = gr.Model3D(label="Option A (left)", interactive=False,
                                   height=420, clear_color=(1.0, 1.0, 1.0, 1.0))
                cap_a = gr.Markdown()
            with gr.Column():
                img_b = gr.Model3D(label="Option B (right)", interactive=False,
                                   height=420, clear_color=(1.0, 1.0, 1.0, 1.0))
                cap_b = gr.Markdown()
        with gr.Row():
            btn_a = gr.Button("◀ A is better", variant="primary")
            btn_tie = gr.Button("Tie")
            btn_skip = gr.Button("Skip")
            btn_b = gr.Button("B is better ▶", variant="primary")

        idx_state = gr.State(0)
        outputs = [img_orig, img_a, img_b, cap_a, cap_b, progress_md,
                   btn_a, btn_b, btn_tie, btn_skip, idx_state]

        btn_a.click(lambda i: _submit("a", i), idx_state, outputs)
        btn_b.click(lambda i: _submit("b", i), idx_state, outputs)
        btn_tie.click(lambda i: _submit("tie", i), idx_state, outputs)
        btn_skip.click(lambda i: _submit("skip", i), idx_state, outputs)
        app.load(lambda: _view(0), None, outputs)

    return app
