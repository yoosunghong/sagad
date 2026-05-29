# Claude Agent Development Workflow Guidelines

This document establishes development rules, system constraints, and behavioral expectations for the AI Agent (Claude Code / Cursor) handling this codebase.

## 1. Operational Philosophy & Workflow Execution
* **Plan Verification First:** Before writing or changing any code, you must read `docs/PLAN.md` to identify the current operational phase. Do not implement out-of-scope modules ahead of timeline synchronization.
* **Progress Recording:** Upon completing a task or milestone within a phase, you must record execution logs inside the `docs/DONE/` directory (e.g., updating `docs/DONE/phase_1_setup.md`). Check off the respective item within the `docs/PLAN.md` checklist.
* **Architectural Alignment:** Any alterations to input/output structures, network parameters, or material channel maps must be fully updated inside `docs/ARCHITECT.md` before execution.

## 2. Code Quality & Formatting Rules
* **Modularity:** Maintain clean separation of concerns. Keep data graphs, reinforcement learning states, reward constraints, and export scripts strictly isolated within dedicated modules.
* **Math Representation:** If writing inline math formulas or formal criteria within documentation, follow standard Markdown/LaTeX formatting.
* **Logging Protocols:** Implement explicit telemetry tracking across the data pipeline (e.g., tracking vertex degradation scales, monitoring NaN values during continuous PPO output processing).

## 3. Domain Constraints
* **Vertex Channel Maps:** Keep mask variables mapping directly to the `[R, G, B, A]` conventions matching the `docs/ARCHITECT.md` engine pipeline definitions. Do not add arbitrary virtual channels that cannot be mapped into a native UE5 Static Mesh Vertex Color allocation.
* **Optimization Awareness:** Always keep processing memory overhead under consideration. Avoid loading highly heavy 3D structures into CPU memory arrays repeatedly; leverage vectorized PyG tensor computations where possible.
