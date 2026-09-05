"""Drive one playbook on the live phone — the engine's loop without the
session.

No policy gates, no compaction, no trace, no sentinel: just the
conductor's contract — ask the Program for a turn, dispatch its one
action, feed the result back — so a rehearsal exercises the real walk.
`arm` loads and validates everything BEFORE any connection exists;
`walk` runs the loop over an open client and returns `WALK_ENDED`,
`WALK_SUSPENDED`, or `WALK_PAUSED`. `emit` receives every progress
line; the core never prints.

`playbooks run` is typer around it, `playbooks replay` shares `arm`,
and `playbooks step` (`debug/stepping.py`, the studio's driver too)
runs the same loop one node at a time with `Program.step_one` set, a
`transform` hook for the virtual channel, and `macro_opts` to narrow
the first macro run to a step range.
"""

import asyncio
import json
from typing import TYPE_CHECKING

from physiclaw.common import gesture_vocab
from physiclaw.common.listing import is_header
from physiclaw.conductor.drive.hooks import (
    Emit,
    McpCaller,
    Observe,
    OnExchange,
    Transform,
)
from physiclaw.conductor.spec.limits import REHEARSE_MAX_TURNS
from physiclaw.contract.wire import leaf_blocks

if TYPE_CHECKING:
    from physiclaw.conductor.spec.match import Verdict
    from physiclaw.conductor.walk.micro import DecisionRequest, MicroCaller, MicroResult
    from physiclaw.conductor.walk.program import Program
    from physiclaw.contract.dto import ToolCall
    from physiclaw.contract.plugin import WireSink
    from physiclaw.macros.model import Macro


def arm(
    app: str, name: str, values: dict[str, str], emit_warn: Emit, *, dry: bool = False
) -> "tuple[Program, dict[str, Macro]]":
    """Load, validate, and build the walk — no connection, no gesture.
    Raises PlaybookError/MacroError on a bad spec or bad inputs.
    Returns (program, registry) ready for `walk`; `dry` builds the
    walk the offline replay drives (it writes nothing)."""
    from physiclaw.conductor.drive import activation, build
    from physiclaw.conductor.drive import setup as conductor_setup
    from physiclaw.conductor.spec import channel as channel_mod
    from physiclaw.conductor.spec import lints

    spec, pack = build.load_spec(app, name, require_live=False)
    values = build.resolve_inputs(spec, values)
    if not spec.enabled:
        emit_warn(f"{app}/{name} is disabled — rehearsing it anyway")
    for line in lints.readiness_warnings(spec, pack):
        emit_warn(line)
    channel = channel_mod.load_channel()
    program = build.build_program(
        spec,
        pack,
        values,
        channel,
        dry=dry,
        activation=activation.activation_for(channel) if spec.activates else None,
    )
    # The dispatch registry a real wake would arm — one spelling, shared
    # with `session_setup` (a gate's ask dispatches `channel/send`,
    # which is not this pack's macro).
    registry = conductor_setup.walk_registry(program, channel)
    return program, registry


class ModelLog:
    """A `contract.plugin.WireSink` for a rehearsal: every model
    round-trip the micro-caller makes (each attempt's serialized
    request and the provider's raw reply), kept in memory so the
    debugger can show exactly what went to the provider and what came
    back. `walk` drains it after each decision."""

    def __init__(self) -> None:
        self.exchanges: list[dict] = []

    def write_micro(self, call: str, request: list[dict], raw: dict) -> None:
        self.exchanges.append({"call": call, "request": request, "reply": raw})

    def drain(self) -> list[dict]:
        out, self.exchanges = self.exchanges, []
        return out


# What `walk` answers when the walk ends — pinned so a caller reads
# the outcome instead of parsing prose. (The stepping driver has its
# own outcome vocabulary one level up; these are the loop's.)
WALK_ENDED = "walk finished or handed over — see the notes above"
WALK_SUSPENDED = "walk suspended waiting on you — suspension dropped"
WALK_PAUSED = "walk paused — the node settled"


async def walk(
    program: "Program",
    registry: "dict[str, Macro]",
    mcp: McpCaller,
    emit: Emit,
    caller: str = "cli",
    *,
    transform: Transform | None = None,
    macro_opts: dict | None = None,
    verbose: bool = False,
    observe: Observe | None = None,
    raw: bool = False,
    on_exchange: OnExchange | None = None,
    unlock: bool = True,
) -> str:
    """One armed walk over an already-open MCP client, one turn at a
    time, until it finishes, hands over, suspends, pauses (a stepping
    program's cursor moved), or hits the turn cap. Emits each turn's
    note, the verdict the next turn acted on, and (`verbose`) the
    result text — a macro's step log and the listing.
    `transform(call, blocks) -> blocks | None` rewrites a result before
    the program reads it (the debug fake-channel); `macro_opts` are
    `run_and_record` keywords for the first macro run only;
    `observe(call, blocks)` sees every real result before any rewrite
    (the studio renders the phone off it). Every model round-trip is
    captured (`ModelLog`): `raw` emits each one — the messages as sent
    and the reply as received — and `on_exchange(record)` receives it
    (the studio's expandable log entry). `unlock` is the one lock-screen
    preamble; a caller re-entering per node pays it once. Raises
    RuntimeError when a model call fires with no model configured."""
    from physiclaw.conductor.walk.micro import DecisionRequest
    from physiclaw.conductor.walk.step import Paused
    from physiclaw.contract.dto import SystemMessage, ToolResultMessage, UserMessage

    history: list = [
        SystemMessage(content="rehearsal"),
        UserMessage(content="rehearse the armed walk"),
    ]
    micro = None
    wire = ModelLog()
    opts = dict(macro_opts or {})
    shown = None  # the last verdict printed — one line per reading
    try:
        # A rehearsal drives the phone NOW — wake it first if it locked
        # between runs (the runtime's boot does this at every real wake
        # through its `locked:` hand; a rehearsal owes the walk the
        # same floor) — unless the route's own start page declares that
        # hand, in which case the walk wakes the phone itself.
        if unlock and not _declares_locked_hand(program):
            await unlock_if_covered(mcp, emit)
        for _ in range(REHEARSE_MAX_TURNS):
            step = program.advance(history)
            if program.verdict is not None and program.verdict is not shown:
                shown = program.verdict
                emit(f"  {_describe_verdict(shown)}")
            while isinstance(step, DecisionRequest):
                # Built on FIRST use, and after the connection — so
                # "start the server first" is what a user without one
                # hears, and a walk that never calls a model never pays
                # a model-config error either.
                if micro is None:
                    micro = micro_caller(rlog=wire)
                result = await micro.run(step)
                decision = _describe(result)
                emit(f"  model {step.call} ({step.node_id}): {decision}")
                for record in exchanges(wire.drain(), step, decision):
                    if raw:
                        for line in record["lines"]:
                            emit(f"      {line}")
                    if on_exchange is not None:
                        on_exchange(record)
                step = program.resolve(result.outcome)
            if isinstance(step, Paused):
                return WALK_PAUSED
            if step is None:
                if program.baton is not None:
                    # The boot decided: the rehearsal ends where the
                    # wake would go on — say which walk, and how to
                    # rehearse it on its own.
                    baton = program.baton
                    emit(
                        f"  boot hands over to {baton.app}/{baton.spec.name} "
                        f"{baton.values} — rehearse it: physiclaw playbooks run "
                        f"{baton.app}/{baton.spec.name}"
                    )
                return WALK_ENDED
            note, act = step.tool_calls
            emit(f"  {note.arguments['summary']}")
            if act.name == "end_session":
                # The walk wants to suspend for a later wake. A
                # rehearsal has no later wake, and `_suspend` already
                # wrote the file — drop it so it cannot ambush the
                # next real session.
                from physiclaw.conductor.walk.suspension import clear_suspended

                clear_suspended()
                return WALK_SUSPENDED
            emit(f"    → {act.name}({_args(act.arguments)})")
            run_opts: dict = {}
            if opts and act.name == gesture_vocab.RUN_MACRO:
                run_opts, opts = opts, {}  # the first macro run only
            text, is_error = await dispatch(
                mcp,
                act,
                registry,
                caller=caller,
                transform=transform,
                observe=observe,
                **run_opts,
            )
            if verbose or is_error:
                for line in result_lines(text, verbose):
                    emit(f"      {line}")
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


def _declares_locked_hand(program: "Program") -> bool:
    recovery = program.spec.recovers.get(program.spec.start)
    return recovery is not None and recovery.locked is not None


async def unlock_if_covered(mcp: McpCaller, emit: Emit) -> None:
    """One peek; a lock-screen reading (the cover's hero clock, or the
    unlock hint text) gets one `unlock_phone`. Fail-open — a camera blip
    just lets the walk meet the world as it is."""
    from physiclaw.common import verdict
    from physiclaw.common.listing import Screen
    from physiclaw.conductor.spec.match import reads_as_locked

    try:
        screen = Screen.read(
            verdict.screen_text(await mcp.call_tool(gesture_vocab.PEEK, {}))
        )
        if reads_as_locked(screen):
            emit("  phone is locked — unlocking first")
            await mcp.call_tool(gesture_vocab.UNLOCK_PHONE, {})
    except Exception as e:
        emit(f"unlock preamble skipped ({e})")


def exchanges(drained: list[dict], req: "DecisionRequest", decision: str) -> list[dict]:
    """The drained round-trips of one decision as debugger records:
    which call and node, the attempt (a repair retry is a second
    round-trip), how many replayed episode turns the request omits
    (they are byte-identical to earlier calls' records), the decision
    the caller made of the last reply, and `lines` — the round-trip
    rendered once for every eye (the CLI prints them, the studio shows
    them), so no skin parses the provider wire itself."""
    out = []
    for i, x in enumerate(drained, start=1):
        record = {
            **x,
            "node": req.node_id,
            "attempt": i,
            "attempts": len(drained),
            "history": len(req.history),
            "outcome": decision if i == len(drained) else "",
        }
        record["lines"] = exchange_lines(record)
        out.append(record)
    return out


def exchange_lines(record: dict) -> list[str]:
    """One round-trip for the eye: each request message with its role,
    then the reply's text (or the whole raw reply when its shape is
    unknown), then the decision."""
    head = f"── model {record['call']} ({record['node']})"
    if record["attempts"] > 1:
        head += f" attempt {record['attempt']}/{record['attempts']}"
    if record["history"]:
        head += f" · {record['history']} replayed turn(s) not repeated"
    lines = [head]
    for m in record["request"]:
        lines.append(f"[{m.get('role', '?')}]")
        lines.extend(_message_text(m).splitlines())
    lines.append("── reply")
    lines.extend(reply_text(record["reply"]).splitlines())
    if record["outcome"]:
        lines.append(f"── decision: {record['outcome']}")
    return lines


def _message_text(message: dict) -> str:
    """A serialized message's text, whichever wire shape — the wire
    codec's own flattening."""
    return "\n".join(
        str(b.get("text", "")) for b in leaf_blocks(message.get("content"))
    )


def reply_text(raw: dict) -> str:
    """The model's text out of a raw provider reply — the two wire
    shapes in the tree, else the reply verbatim."""
    try:
        choices = raw.get("choices")
        if choices:  # the OpenAI shape
            return str(choices[0]["message"]["content"])
        content = raw.get("content")
        if isinstance(content, list):  # the Anthropic shape
            return "\n".join(
                str(b.get("text", "")) for b in content if isinstance(b, dict)
            )
    except (KeyError, IndexError, TypeError, AttributeError):
        pass
    return json.dumps(raw, ensure_ascii=False)


def _describe_verdict(v: "Verdict") -> str:
    """What the matcher made of the screen the next turn acts on."""
    return f"screen reads {v.describe()}"


def _describe(result: "MicroResult") -> str:
    o = result.outcome
    if o is None:
        return f"no outcome — {result.detail} ({result.elapsed_ms} ms)"
    picked = f" → {o.picked.key!r}" if o.picked is not None else ""
    fields = f" {o.payload}" if o.payload else ""
    return (
        f"{o.out}{picked}{fields} [{o.confidence:.2f}] {o.reason} "
        f"({result.elapsed_ms} ms)"
    )


def _args(arguments: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in (arguments or {}).items())


def result_lines(text: str, verbose: bool) -> list[str]:
    """A result's text for the eye: a macro's header and step log always,
    the listing rows only when `verbose`. Shared with the macro stepping
    driver, which reads a macro run the same way."""
    lines = text.splitlines()
    head: list[str] = []
    for i, line in enumerate(lines):
        if is_header(line):
            return head + (lines[i:] if verbose else [f"({len(lines) - i - 1} rows)"])
        head.append(line)
    return head


async def dispatch(
    mcp: McpCaller,
    call: "ToolCall",
    registry: "dict[str, Macro]",
    caller: str = "cli",
    *,
    transform: Transform | None = None,
    observe: Observe | None = None,
    start_at: str = "",
    stop_after: str = "",
) -> tuple[str, bool]:
    """One synthesized action → the text its result carries.

    Routes the way the engine does: `run_macro` is the LOCAL tool (it
    runs a macro through the macro runner, which drives the same MCP
    connection step by step), everything else is a plain MCP call.
    `registry` holds qualified `app/name` macros — the pack's plus the
    channel's, like the engine's hidden registry. The reply text is
    EVERY text block of the result (`verdict.all_text`), exactly what
    the engine's tool result carries — a gesture's action line, a
    macro's header and step log — ahead of the listing the Program
    reads its screen out of, so an aborted macro's cause reaches the
    handover reason and the eye. `transform(call, blocks)` may replace a
    successful result's blocks (None keeps them) — the engine's
    debug-intercept seam, reproduced; `observe(call, blocks)` is told
    the real blocks first, rewritten or not; `start_at`/`stop_after`
    narrow a macro run to a step range (`run_and_record`'s own knobs)."""
    from physiclaw.common import verdict
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
                spec,
                call.arguments.get("inputs") or {},
                mcp,
                caller=caller,
                start_at=start_at,
                stop_after=stop_after,
            )
            blocks, is_error = result.blocks, not result.ok
        else:
            blocks, is_error = await mcp.call_tool(call.name, call.arguments), False
        if observe is not None:
            observe(call, blocks)
        if transform is not None and not is_error:
            faked = transform(call, blocks)
            if faked is not None:
                blocks = faked
        return verdict.all_text(blocks), is_error
    except Exception as e:  # a rehearsal reports, it does not crash
        return f"{call.name} failed: {e}", True


def micro_caller(rlog: "WireSink | None" = None) -> "MicroCaller":
    """The decision channel a rehearsal needs — same resolution as the
    engine's (`[conductor] micro_model`, else the session model), so a
    rehearsal spends the model a real wake would. `rlog` is the wire
    sink every round-trip goes to (`ModelLog`). Raises RuntimeError
    when no model is configured."""
    from physiclaw.common.config import CONFIG, model_ref, parse_model_ref
    from physiclaw.conductor.walk.micro import MicroCaller
    from physiclaw.provider import make_provider

    ref = CONFIG.conductor.micro_model or model_ref()
    pid, mid = parse_model_ref(ref)
    return MicroCaller(
        make_provider(pid, mid),
        confidence_floor=CONFIG.conductor.micro_confidence,
        rlog=rlog,
    )
