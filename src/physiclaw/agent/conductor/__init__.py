"""Conductor package — deterministic step arbiter at the provider seam.

Layout:
  - `conductor.py` — the ``Conductor`` turn arbiter the loop asks for
    each assistant turn (``advance``); it owns the provider call.
  - `pages.py` — page declarations (pages.yml) + learned geometry store.
  - `match.py` — the open-set page matcher (match / occluded / unknown).
  - `capture.py` — mine geometry + calibrate thresholds from observations.
  - `corpus.py` — recorded-session listings for offline matching work.
"""

from physiclaw.agent.conductor.conductor import Conductor

__all__ = ["Conductor"]
