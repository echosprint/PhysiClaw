"""The conductor — deterministic turns for the routine parts of a task.

A peer of the agent, not part of it. The engine loads this package only
through the `TurnPlugin` seam (`contract.plugin`, named by the
`[agent] plugins` config path); neither package imports the other, and
`tests/test_architecture.py` enforces that. Each turn the conductor
either synthesizes the assistant message itself (no provider call) or
passes, and the LLM speaks.

Vocabulary, in the order a reader meets it:

    pack       one app's directory: PLAYBOOK.yml + macros/
    playbook   one task in a pack, written as a route
    entry      one YAML item of a route (its leading key is the kind)
    page       a waypoint entry: where the walk must BE, checked every
               time; declared once (anchors), referenced bare elsewhere
    move       any non-page entry: start, do, agent, ask, tell
    node       a compiled move (pages compile away into the adjacent
               moves' enter/verify checks)
    walk       one playbook executing (`Program`), cursor over nodes
    step       the executor running the node at the cursor
    turn       one synthesized [note, action] assistant message; the
               action's result comes back as ordinary history
    verdict    the matcher's reading of a screen: match / occluded /
               unknown, with the page id
    hand       a declared recovery action: one gesture or one macro
    landmark   a named fixed spot the author knows ({label, bbox})
    grant      what an episode may name blind: a landmark or a macro
    gate       the ask-and-hold state (reply words, consent)
    brief      the walk's last note: why it stopped, where it stands
    handover   the walk goes quiet; the model takes the session

The seam:
    plugin.py      the composition point: wake setup, micro wiring
    conductor.py   the per-turn arbiter; None means "the LLM speaks"

Driving:
    overture.py    boot to the user's thread, read the intent there
    program.py     one playbook mid-walk: cursor, verdicts, recovery
    step.py        the step executor contract + the Walk surface
    step_do.py     the `do`/`start` step: enter, macro, verify
    step_agent.py  the `agent` step: the pure-text call, the episode
    step_ask.py    the `ask` and `tell` steps: message, hold, consent
    gate.py        the ask-and-hold state, one suspension projection
    money.py       the declared total and the payment predicates, pure
    recover.py     declared recovery: the page's hands and their bounds
    setup.py       how a Program comes to exist (activation, resume)
    rehearsal.py   the engine's loop without the session: run, step, replay
    replay.py      the real walk over recorded screens, writing nothing
    turns.py       minting a synthesized turn
    brief.py       the handover report a driver's last turn carries
    suspension.py  suspended.json, the one cross-wake file
    micro.py       the scoped model calls (agent, activation)
    calls.py       the episode vocabulary the parser and walk share
    context.py     what an agent step's `context:` loads beside it
    hooks.py       the typed callable seams a driver takes
    limits.py      every bound a walk runs under, in one place

Seeing the screen:
    pages.py       declared and learned page fingerprints
    match.py       the open-set page matcher
    views.py       reading tool results out of the transcript
    capture.py     mining fingerprints from recorded observations
    corpus.py      recorded-session listings for offline matching

Specs and the user:
    playbook.py    PLAYBOOK.yml model, the pack, the ref grammar
    route.py       the route compiler and its lints
    specfile.py    shared YAML substrate
    channel.py     the user-channel pack (thread, send, open)
    reply.py       deterministic confirm/deny reply reading
    scaffold.py    pack templates and playbooks/README.md

Telemetry:
    walklog.py     runs.jsonl — per-walk outcomes, the escalation KPI
"""

# No re-exports: consumers import their module directly, so parse-only
# CLI paths never pay for the provider stack.
