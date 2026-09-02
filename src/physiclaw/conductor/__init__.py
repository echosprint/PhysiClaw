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
    program.py     one playbook mid-walk: cursor, verdicts, recovery
    step.py        the step executor contract the walk routes to
    step_do.py     the `do`/`start` step: enter, macro, verify
    step_agent.py  the `agent` step: the pure-text call, the episode
    step_ask.py    the `ask` and `tell` steps: message, hold, consent
    money.py       the payment predicates, pure
    recover.py     declared recovery: the page's `recover:` hand, only
    setup.py       how a Program comes to exist (activation, resume)
    turns.py       minting a synthesized turn
    brief.py       the handover report a driver's last turn carries
    suspension.py  suspended.json, the one cross-wake file
    micro.py       the scoped model calls (agent, activation)
    calls.py       the episode vocabulary the parser and walk share
    context.py     what an agent step's `context:` loads beside it

Seeing the screen:
    pages.py       declared and learned page fingerprints
    match.py       the open-set page matcher
    views.py       reading tool results out of the transcript
    capture.py     mining fingerprints from recorded observations
    corpus.py      recorded-session listings for offline matching

Specs and the user:
    playbook.py    PLAYBOOK.yml model, the pack, the ref grammar
    route.py       the route compiler and its lints
    _spec.py       shared YAML substrate
    channel.py     the user-channel pack (thread, send, open)
    reply.py       deterministic confirm/deny reply reading
    scaffold.py    pack templates and playbooks/README.md

Telemetry and tuning:
    walklog.py     runs.jsonl — per-walk outcomes, the escalation KPI
    evalset.py     labeled replay cases for `conductor eval`
"""

# No re-exports: consumers import their module directly, so parse-only
# CLI paths never pay for the provider stack.
