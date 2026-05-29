"""Multi-view rendering pipeline.

Isolated module (CLAUDE.md modularity rule) for the headless multi-angle
snapshot export of deformed assets (docs/ARCHITECT.md sec. 2.3). The exported
orthogonal multi-view arrays feed the CLIP diversity evaluator (sec. 2.4) and
the human-preference interface (Phase 3).

* :mod:`multiview` -- orbit-camera offscreen renderer (Open3D legacy
  Visualizer backend, ``visible=False``).
"""

from .multiview import RenderConfig, render_multiview, save_views

__all__ = ["RenderConfig", "render_multiview", "save_views"]
