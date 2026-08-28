"""The conductor — deterministic turns for the routine parts of a task.

A peer of the agent, not part of it. The engine loads this package only
through the `TurnPlugin` seam (`contract.plugin`, named by the
`[agent] plugins` config path); neither package imports the other, and
`tests/test_architecture.py` enforces that. Each turn the conductor
either synthesizes the assistant message itself (no provider call) or
passes, and the LLM speaks.

The seam:
    plugin.py      the composition point: wake setup, micro wiring
    conductor.py   the per-turn arbiter; None means "the LLM speaks"

Driving:
    overture.py    boot to the user's thread, read the intent there
    program.py     one playbook mid-walk: legs, gates, ledger, money rules
    setup.py       how a Program comes to exist (activation, resume)
    turns.py       minting a synthesized turn
    suspension.py  suspended.json, the one cross-wake file
    micro.py       scoped cheap-tier decision calls
    calls.py       the DECIDE call-type table

Seeing the screen:
    pages.py       declared and learned page fingerprints
    match.py       the open-set page matcher
    views.py       reading tool results out of the transcript
    capture.py     mining fingerprints from recorded observations
    corpus.py      recorded-session listings for offline matching

Specs and the user:
    playbook.py    PLAYBOOK.yml model, parser, lints
    _spec.py       shared YAML substrate
    channel.py     the user-channel pack (thread, send, open)
    ledger.py      the buying list and its cart readers
    reply.py       deterministic confirm/deny reply reading
    memory.py      fail-closed memory slices for decisions
    scaffold.py    pack templates and playbooks/README.md
"""

# No re-exports: consumers import their module directly, so parse-only
# CLI paths never pay for the provider stack.
