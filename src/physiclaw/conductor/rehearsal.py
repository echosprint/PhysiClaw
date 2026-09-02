"""Drive one playbook to its end on the live phone — the rehearsal core.

This is the engine's turn loop with everything session-shaped removed:
no policy gates, no compaction, no trace, no sentinel. What remains is
exactly the conductor's contract — ask the Program for a turn, dispatch
its one action, feed the result back — so a rehearsal exercises the
real walk rather than a simulation of it.

Two drivers share it: `physiclaw playbooks run` (typer around it) and
the studio's rehearse button (its persistent MCP session, lines into
the browser log). The split is deliberate: `arm` loads and validates
everything BEFORE any connection exists (bad inputs must fail without
touching the phone), `walk` runs the loop over an already-open client.
`emit` receives each progress line (`arm`'s advisory lines go to its
`emit_warn`) — the core never prints.
"""

import asyncio

# A rehearsal is a person watching, so the bound is "long enough for a
# real walk" rather than the engine's session budget. A gate polling for
# a reply is the long pole.
REHEARSE_MAX_TURNS = 60


def arm(app: str, name: str, values: dict[str, str], emit_warn) -> tuple:
    """Load, validate, and build the walk — no connection, no gesture.
    Raises PlaybookError/MacroError on a bad spec or bad inputs.
    Returns (program, registry) ready for `walk`."""
    from physiclaw.conductor import channel as channel_mod
    from physiclaw.conductor import setup as conductor_setup

    spec, pack = conductor_setup.load_spec(app, name, require_live=False)
    values = conductor_setup.resolve_inputs(spec, values)
    if not spec.enabled:
        emit_warn(f"{app}/{name} is disabled — rehearsing it anyway")
    for line in conductor_setup.readiness_warnings(spec):
        emit_warn(line)
    channel = channel_mod.load_channel()
    program = conductor_setup.build_program(spec, pack, values, channel)
    # The dispatch registry a real wake would arm — one spelling, shared
    # with `session_setup` (a gate's ask dispatches `channel/send`,
    # which is not this pack's macro).
    registry = conductor_setup.walk_registry(program, channel)
    return program, registry


async def walk(
    program,
    registry: dict,
    mcp,
    emit,
    caller: str = "cli",
) -> str:
    """One armed walk over an already-open MCP client, one turn at a
    time, until it finishes, hands over, suspends, or hits the turn cap.
    Raises RuntimeError when a model call fires with no model configured."""
    from physiclaw.conductor.micro import DecisionRequest
    from physiclaw.contract.dto import SystemMessage, ToolResultMessage, UserMessage

    history: list = [
        SystemMessage(content="rehearsal"),
        UserMessage(content="rehearse the armed walk"),
    ]
    micro = None
    try:
        # A rehearsal drives the phone NOW — wake it first if it locked
        # between runs (the runtime's overture does this at every real
        # wake; a rehearsal owes the walk the same floor).
        await unlock_if_covered(mcp, emit)
        for _ in range(REHEARSE_MAX_TURNS):
            step = program.advance(history)
            while isinstance(step, DecisionRequest):
                # Built on FIRST use, and after the connection — so
                # "start the server first" is what a user without one
                # hears, and a walk that never calls a model never pays
                # a model-config error either.
                if micro is None:
                    micro = micro_caller()
                outcome = (await micro.run(step)).outcome
                step = program.resolve(outcome)
            if step is None:
                return "walk finished or handed over — see the notes above"
            note, act = step.tool_calls
            emit(f"  {note.arguments['summary']}")
            if act.name == "end_session":
                # The walk wants to suspend for a later wake. A
                # rehearsal has no later wake, and `_suspend` already
                # wrote the file — drop it so it cannot ambush the
                # next real session.
                from physiclaw.conductor.suspension import clear_suspended

                clear_suspended()
                return "walk suspended waiting on you — suspension dropped"
            text, is_error = await dispatch(mcp, act, registry, caller=caller)
            history.append(step)
            history.append(
                ToolResultMessage(tool_call_id=act.id, content=text, is_error=is_error)
            )
        return f"stopped after {REHEARSE_MAX_TURNS} turns"
    finally:
        # A rehearsal cut short (Ctrl-C, the turn cap) records its
        # abandoned row like a real wake's teardown would — latched, so
        # a walk that closed properly is a no-op.
        program.abandon()
        if micro is not None:
            await micro.aclose()


async def unlock_if_covered(mcp, emit) -> None:
    """One peek; a lock-screen reading (the cover's hero clock, or the
    unlock hint text) gets one `unlock_phone`. Fail-open — a camera blip
    just lets the walk meet the world as it is."""
    from physiclaw.common import gesture_vocab, verdict
    from physiclaw.common.listing import Screen
    from physiclaw.conductor.match import reads_as_locked

    try:
        screen = Screen.read(
            verdict.screen_text(await mcp.call_tool(gesture_vocab.PEEK, {}))
        )
        if reads_as_locked(screen):
            emit("  phone is locked — unlocking first")
            await mcp.call_tool(gesture_vocab.UNLOCK_PHONE, {})
    except Exception as e:
        emit(f"unlock preamble skipped ({e})")


async def dispatch(mcp, call, registry: dict, caller: str = "cli") -> tuple[str, bool]:
    """One synthesized action → the text its result carries.

    Routes the way the engine does: `run_macro` is the LOCAL tool (it
    runs a macro through the macro runner, which drives the same MCP
    connection step by step), everything else is a plain MCP call.
    `registry` holds qualified `app/name` macros — the pack's plus the
    channel's, like the engine's hidden registry. The reply text is what
    the Program reads a screen out of, so both arms must hand back the
    same listing shape."""
    from physiclaw.common import gesture_vocab, verdict
    from physiclaw.macros import runner as macro_runner

    try:
        if call.name == "wait":
            # An engine-LOCAL tool, not an MCP one — the gate's reply
            # polling rides it, so the rehearsal sleeps in place exactly
            # like the engine's handler (which also requires `seconds`).
            seconds = float(call.arguments["seconds"])
            await asyncio.sleep(seconds)
            return f"waited {seconds:g}s", False
        if call.name == gesture_vocab.RUN_MACRO:
            qualified = call.arguments.get("name", "")
            spec = registry.get(qualified)
            if spec is None:
                return f"unknown pack macro {qualified!r}", True
            result = await macro_runner.run_and_record(
                spec, call.arguments.get("inputs") or {}, mcp, caller=caller
            )
            return verdict.screen_text(result.blocks), not result.ok
        return verdict.screen_text(
            await mcp.call_tool(call.name, call.arguments)
        ), False
    except Exception as e:  # a rehearsal reports, it does not crash
        return f"{call.name} failed: {e}", True


def micro_caller():
    """The decision channel a rehearsal needs — same resolution as the
    engine's (`[conductor] micro_model`, else the session model), so a
    rehearsal spends the model a real wake would. Raises RuntimeError
    when no model is configured."""
    from physiclaw.common.config import CONFIG, model_ref, parse_model_ref
    from physiclaw.conductor.micro import MicroCaller
    from physiclaw.provider import make_provider

    ref = CONFIG.conductor.micro_model or model_ref()
    pid, mid = parse_model_ref(ref)
    return MicroCaller(
        make_provider(pid, mid), confidence_floor=CONFIG.conductor.micro_confidence
    )
