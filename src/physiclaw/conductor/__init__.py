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

One wake, end to end:

    plugin.session_setup  → setup: a suspension resumes, else the boot
    the boot              → Program over channel/boot.yml: peek, the thread
                            page's declared hands, then `activate` —
                            parse_task over the enabled playbooks
    the baton             → activation builds the matching playbook's
                            Program; the conductor drives it next
    the walk              → each node's step: a move's enter check, its
                            macro, its verify; an agent's fenced calls;
                            an ask's send, hold, and read; a tell's send
    a deviation           → the page's `recover:` hand, or handover
    the end               → completion, handover (one brief turn), or an
                            ask out of patience (suspended.json); the
                            record writes the runs row and the daily log

The seam:
    plugin.py           the composition point the engine names: wake setup,
                        micro wiring, one Conductor per session

spec/ — what a pack declares (imports nothing from walk or drive):
    conventions.py      the fixed names: channel, ios, thread, boot, lock
    model.py            the playbook grammar as dataclasses, the Pack
    refs.py             the `{inputs.x}` / `{move.field}` ref grammar
    pack.py             the pack door: load, scan, qualified names, live rule
    route.py            the route compiler
    lints.py            the whole-route checks and the check-time advisories
    pages.py            declared and learned page fingerprints
    match.py            the open-set page matcher (the lock screen by shape)
    channel.py          the user-channel pack: thread, send, open, boot
    reply.py            deterministic confirm/deny reply reading
    context.py          what an agent step's `context:` loads beside it
    calls.py            the episode vocabulary the parser and walk share
    limits.py           every bound a walk runs under, in one place
    specfile.py         shared YAML substrate
    scaffold.py         pack templates and playbooks/README.md

walk/ — one playbook executing (imports spec, never drive):
    program.py          the walk: cursor and phase, verdicts, recovery, ends
    step.py             the step executor contract + the Walk surface
    step_do.py          `do`/`start`: enter check, the macro, verify
    step_agent.py       `agent`: the pure-text call, or the episode
    step_ask.py         `ask`: send, hold, judge the reply, bind consent
    step_tell.py        `tell`: send, then move on
    step_activate.py    the boot's `activate`: parse_task, the baton
    speak.py            the walk's voice: send, land, read replies
    gate.py             the ask-and-hold state, one suspension projection
    money.py            the declared total and the payment predicates
    recover.py          declared recovery: the page's hands and bounds
    turns.py            minting a synthesized turn, the one in flight
    views.py            reading tool results out of the transcript
    brief.py            the report the walk's last turn carries
    suspension.py       suspended.json, the one cross-wake file
    micro.py            the scoped model calls (agent, activation)
    prompts.py          what those calls say — the prompt texts
    record.py           the walk's writes: the runs row, the daily log
    walklog.py          runs.jsonl — per-walk outcomes, the escalation KPI

drive/ — the doors and tools that drive a walk (imports walk and spec):
    conductor.py        the per-turn arbiter; None means "the LLM speaks"
    setup.py            the wake's doors: a resumed suspension, or the boot
    build.py            the one Program constructor call, load_spec, inputs
    activation.py       the boot's menu, parse_task, the program it builds
    rehearsal.py        the engine's loop without the session: run, step
    replay.py           the real walk over recorded screens, writing nothing
    capture.py          mining fingerprints from recorded observations
    corpus.py           recorded-session listings for offline matching
    hooks.py            the typed callable seams a driver takes
"""

# No re-exports: consumers import their module directly, so parse-only
# CLI paths never pay for the provider stack.
