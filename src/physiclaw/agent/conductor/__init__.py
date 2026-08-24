"""Conductor package — deterministic step arbiter at the provider seam.

Design doc: CONDUCTOR_PLAN.md at the repo root.

Layout:
  - `conductor.py` — the ``Conductor`` turn arbiter the loop asks for
    each assistant turn (``advance``); it owns the provider call.
"""

from physiclaw.agent.conductor.conductor import Conductor

__all__ = ["Conductor"]
