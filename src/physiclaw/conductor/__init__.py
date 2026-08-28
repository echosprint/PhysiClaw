"""Conductor package — deterministic step arbiter behind the plugin seam.

A peer of the agent, not a part of it: the engine loads this package
only through `contract.plugin` and the `[agent] plugins` dotted path
(`plugin.py` below), and neither package imports the other —
`tests/test_architecture.py` enforces the graph. The conductor stands
on `contract` (message shapes, the plugin protocol), `macros`,
`provider`, and `common`.

Layout:
  - `plugin.py` — the package's one composition point: the conductor as
    a `TurnPlugin` (wake setup, micro wiring, per-turn arbitration).
  - `conductor.py` — the ``Conductor`` turn arbiter asked for each
    assistant turn (``advance``); None means the LLM speaks.
  - `pages.py` — page declarations (pages.yml) + learned geometry store.
  - `match.py` — the open-set page matcher (match / occluded / unknown).
  - `capture.py` — mine geometry + calibrate thresholds from observations.
  - `corpus.py` — recorded-session listings for offline matching work.
  - `calls.py` — the code-owned DECIDE call-type declarations.
  - `micro.py` — the scoped decision calls: prompt shapes, validation,
    the confidence gate; wired by `plugin.py` with the session's sinks.
  - `_spec.py` — shared spec-file substrate: the YAML loader and the
    macro layer's scalar rules, bound to each spec's error class.
  - `playbook.py` — PLAYBOOK.yml model + parser + graph/money/ledger lints.
  - `turns.py` — minting a synthesized turn: the `[note, one-other]`
    shape, the call-id convention, and the one action in flight.
  - `overture.py` — what plays before the program: the deterministic
    boot to the user's thread (unlock, open, verify) and the one
    parse_task ask that either hands a Program back or goes quiet.
  - `program.py` — a playbook mid-walk: synthesized turns (legs,
    decisions' primitives, channel asks), the HUMAN_GATE ask-and-hold,
    the ledger loop + cart reconciliation, suspending, money predicates.
  - `setup.py` — how a Program comes to exist: `build_program` and the
    spec/input/readiness rules behind it, the suspended loader,
    Activation (parse_task), and `session_setup`.
  - `suspension.py` — the suspended.json file: the ONE piece of
    cross-wake state a walk leaves behind.
  - `channel.py` — the user-channel pack (thread prints + send/open) and
    the qualified `app/name` dispatch-key spelling.
  - `ledger.py` — the buying list: value contract, item state, and the
    reconciler's cart readers (row match, stepper quantities).
  - `memory.py` — fail-closed `memory.<slug>` context slices (NOT
    engine.memory — the conductor reads the file via paths only).
  - `views.py` — reading tool results out of the transcript.
  - `reply.py` — the gate's deterministic reply tiers: word lists and
    new-incoming-bubble detection.
  - `scaffold.py` — app-pack init templates + playbooks/README.md.
"""

# No re-exports: every consumer imports its module directly (the engine
# reaches `plugin.py` only via the config-listed dotted path), so the
# package stays import-free and the parse-only CLI surfaces never pay
# for the provider stack.
