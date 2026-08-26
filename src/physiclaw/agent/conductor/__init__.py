"""Conductor package — deterministic step arbiter at the provider seam.

Layout:
  - `conductor.py` — the ``Conductor`` turn arbiter the loop asks for
    each assistant turn (``advance``); it owns the provider call.
  - `pages.py` — page declarations (pages.yml) + learned geometry store.
  - `match.py` — the open-set page matcher (match / occluded / unknown).
  - `capture.py` — mine geometry + calibrate thresholds from observations.
  - `corpus.py` — recorded-session listings for offline matching work.
  - `calls.py` — the code-owned DECIDE call-type declarations.
  - `micro.py` — the scoped decision calls: prompt shapes, validation,
    the confidence gate; engine-wired with the session's sinks.
  - `_spec.py` — shared spec-file substrate: the YAML loader and the
    macro layer's scalar rules, bound to each spec's error class.
  - `playbook.py` — PLAYBOOK.yml model + parser + graph/money/ledger lints.
  - `program.py` — a playbook mid-walk: synthesized turns (legs,
    decisions' primitives, channel asks), the HUMAN_GATE ask-and-hold,
    the ledger loop + cart reconciliation, parking/resume, activation,
    the money predicates; arm/parked files.
  - `reply.py` — the gate's deterministic reply tiers: word lists and
    new-incoming-bubble detection.
  - `scaffold.py` — app-pack init templates + playbooks/README.md.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physiclaw.agent.conductor.conductor import Conductor

__all__ = ["Conductor"]


def __getattr__(name: str):
    # Lazy: `conductor.py` pulls the provider stack (~120ms), which the
    # parse-only CLI surfaces (`playbooks check`, `conductor match`) never
    # need — same pattern as the top-level package's lazy `PhysiClaw`.
    if name == "Conductor":
        from physiclaw.agent.conductor.conductor import Conductor

        return Conductor
    raise AttributeError(name)
