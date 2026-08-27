"""`physiclaw playbooks` — scaffold, list, validate, and rehearse app packs.

Mirrors the macros CLI's shapes throughout: `init` prints the next
steps, `check` is the all-or-nothing gate with valid-is-not-live
warnings, and `run` rehearses one playbook against the live server the
way `macros run` rehearses one macro — the test you do BEFORE enabling.
The scaffolding itself lives in `agent/conductor/scaffold.py` (the
`store.init_macro` split: the CLI only prints).

Nothing here writes cross-wake state. A rehearsal drives the phone now
and stops; the conductor's own doors at wake are the suspension file and
the overture."""

import asyncio
from typing import TYPE_CHECKING, Annotated

import typer

from physiclaw.cli._format import (
    exit_error,
    ok,
    parse_inputs,
    state_tag,
    step_fail,
    warn,
)

if TYPE_CHECKING:
    from physiclaw.agent.conductor.playbook import Pack, PlaybookEntry

playbooks_app = typer.Typer(no_args_is_help=True)


@playbooks_app.command()
def init(
    app: Annotated[str, typer.Argument(help="App name (lowercase/digits/hyphens).")],
) -> None:
    """Scaffold a new app pack: pages.yml, an example playbook, and an
    example pack macro — all parse-clean, all disabled."""
    from physiclaw.agent.conductor import scaffold
    from physiclaw.agent.conductor.pages import CHANNEL_APP, IOS_APP, THREAD_PAGE
    from physiclaw.agent.conductor.playbook import PlaybookError

    try:
        root = scaffold.init_pack(app)
    except PlaybookError as e:
        exit_error(str(e))
    typer.echo(ok(str(root)))
    typer.echo("Next:")
    if app == CHANNEL_APP:
        typer.echo(
            f"  1. anchor the `{THREAD_PAGE}` page on YOUR chat header in pages.yml"
        )
        typer.echo("  2. record the send/open gesture paths in macros/*/MACRO.yml")
        typer.echo("  3. rehearse both, then enable (physiclaw macros run is")
        typer.echo("     per-user-macro; drive a pack's via physiclaw playbooks run)")
        typer.echo(
            f"  4. capture geometry: physiclaw conductor calibrate {CHANNEL_APP}"
        )
    elif app == IOS_APP:
        typer.echo("  1. check the lock-screen reading matches your phone's language")
        typer.echo("     (lock it, then: physiclaw conductor propose --live)")
        typer.echo(
            f"  2. capture geometry: physiclaw conductor calibrate {IOS_APP} --guided"
        )
    else:
        typer.echo(
            "  1. declare pages in pages.yml (physiclaw conductor propose --live)"
        )
        typer.echo(
            "  2. write pack macros + the playbook, then: physiclaw playbooks check"
        )
        typer.echo("  3. capture geometry: physiclaw conductor calibrate " + app)


@playbooks_app.command("list")
def list_cmd() -> None:
    """List every pack and its playbooks: enabled, disabled, or invalid."""
    from physiclaw.agent.conductor import playbook as pb
    from physiclaw.agent.conductor import scaffold

    scaffold.ensure_format_readme()
    apps = pb.list_apps()
    if not apps:
        typer.echo(
            "No app packs found. Scaffold one: physiclaw playbooks init <app> "
            "(see ~/.physiclaw/playbooks/README.md)"
        )
        return
    for app in apps:
        typer.echo(f"{app}/")
        for e in pb.scan_playbooks(app):
            tag = state_tag(
                valid=e.spec is not None, enabled=bool(e.spec and e.spec.enabled)
            )
            detail = (
                (e.error or "")
                if e.spec is None
                else f"{e.spec.description} ({len(e.spec.nodes)} nodes)"
            )
            typer.echo(f"  {tag} {e.name}  {detail}")


@playbooks_app.command()
def run(
    ref: Annotated[str, typer.Argument(help="<app>/<playbook> to rehearse.")],
    inputs: Annotated[
        list[str] | None,
        typer.Option(
            "--input",
            "-i",
            help="NAME=VALUE for a declared playbook input (repeatable).",
        ),
    ] = None,
) -> None:
    """Rehearse one playbook against the running server — the mirror of
    `physiclaw macros run`, and the way to test a walk without waiting for
    a wake or hoping the boot picks it.

    Drives the real walk on the real phone: legs run their macros, DECIDE
    nodes spend real micro-calls, and a HUMAN_GATE really messages you and
    waits for your reply. Works while the playbook is disabled — rehearse
    first, enable after. Nothing is persisted: a walk that suspends here
    stops instead of leaving a file for the next wake."""
    from physiclaw.agent.conductor.playbook import PlaybookError

    app, _, name = ref.partition("/")
    if not app or not name:
        exit_error(f"expected <app>/<playbook> (got {ref!r})")
    try:
        outcome = asyncio.run(_rehearse(app, name, parse_inputs(inputs or [])))
    except PlaybookError as e:
        exit_error(str(e))
    except ConnectionError as e:
        # `mcp`, not `server`: both serve the same endpoint, but `server`
        # also spawns the agent runtime, which would wake on its own hooks
        # and drive the phone mid-rehearsal. A rehearsal wants the rig to
        # itself. (Same branch as `macros run`.)
        exit_error(f"{e}. Start it first: physiclaw mcp")
    typer.echo(outcome)


@playbooks_app.command()
def check() -> None:
    """Validate every pack: pages, pack macros, playbooks. Exit 1 if any
    is invalid."""
    from physiclaw.agent.conductor import playbook as pb
    from physiclaw.agent.conductor import scaffold

    scaffold.ensure_format_readme()
    apps = pb.list_apps()
    if not apps:
        typer.echo("No app packs found.")
        return
    if any([_check_app(app) for app in apps]):
        raise typer.Exit(1)


def _check_app(app: str) -> bool:
    """Report one pack; True when anything in it is invalid."""
    from physiclaw.agent.conductor import playbook as pb
    from physiclaw.agent.conductor import setup as conductor_setup

    try:
        pack = pb.load_pack(app)
    except pb.PlaybookError as e:
        typer.echo(step_fail(str(e)))
        return True
    bad = False
    for macro_name, err in sorted(pack.macro_errors.items()):
        typer.echo(step_fail(f"{app}/macros/{macro_name}: {err}"))
        bad = True
    entries = pb.scan_playbooks(app, pack)
    disabled: list[str] = []
    for entry in entries:
        if entry.spec is None:
            typer.echo(step_fail(f"{app}/{entry.name}: {entry.error or ''}"))
            bad = True
            continue
        typer.echo(
            ok(f"{app}/{entry.name}" + ("" if entry.spec.enabled else "  (disabled)"))
        )
        if not entry.spec.enabled:
            disabled.append(entry.name)
    _report_not_live(app, pack, entries, disabled)
    for entry in entries:
        if entry.spec is None:
            continue
        # Advisories that used to be reachable only by arming, so almost
        # nobody saw them: a memory slice with no section on THIS device,
        # a gate ask quoting no word-tier reply word, a gate with no way
        # back into the app.
        for line in conductor_setup.readiness_warnings(entry.spec):
            typer.echo(warn(f"{app}/{entry.name}: {line}"))
    return bad


def _report_not_live(
    app: str,
    pack: "Pack",
    entries: "list[PlaybookEntry]",
    disabled: list[str],
) -> None:
    """Valid is not live: a green check invites the wrong assumption. Say
    which playbooks the boot will not offer, and why — disabled files,
    and referenced pack macros that are themselves disabled. Rehearse
    them (`playbooks run`), then enable."""
    from physiclaw.agent.conductor.playbook import disabled_leg_macros

    if disabled:
        typer.echo(
            warn(
                f"{app}: disabled, so the boot will not offer: "
                f"{', '.join(disabled)}. Set `enabled: true` once rehearsed."
            )
        )
    disabled_macros = sorted(
        {
            m
            for e in entries
            if e.spec is not None
            for m in disabled_leg_macros(e.spec, pack)
        }
    )
    if disabled_macros:
        typer.echo(
            warn(
                f"{app}: referenced pack macro(s) still disabled: "
                f"{', '.join(disabled_macros)} — rehearse, then enable."
            )
        )


# ---------- rehearsal (`playbooks run`) ----------

# A rehearsal is a person watching, so the bound is "long enough for a
# real walk" rather than the engine's session budget. A gate polling for
# a reply is the long pole.
REHEARSE_MAX_TURNS = 60


async def _rehearse(app: str, name: str, values: dict[str, str]) -> str:
    """Drive one playbook to its end on the live phone, printing each
    synthesized turn as it goes.

    This is the engine's turn loop with everything session-shaped removed:
    no policy gates, no compaction, no trace, no sentinel. What remains is
    exactly the conductor's contract — ask the Program for a turn,
    dispatch its one action, feed the result back — so a rehearsal
    exercises the real walk rather than a simulation of it."""
    from physiclaw.agent.conductor import channel as channel_mod
    from physiclaw.agent.conductor import setup as conductor_setup
    from physiclaw.agent.conductor.ledger import check_ledger_value
    from physiclaw.agent.conductor.micro import DecisionRequest
    from physiclaw.agent.engine.dto import SystemMessage, ToolResultMessage, UserMessage
    from physiclaw.agent.engine.mcp_tool import McpClient

    spec, pack = conductor_setup.load_spec(app, name, require_live=False)
    values = conductor_setup.resolve_inputs(spec, values)
    # A `kind: list` value is a JSON ledger — hold it to the same contract
    # the overture's activation does, so a typo'd list fails here rather
    # than mid-walk at the shopping loop.
    check_ledger_value(spec, values)
    if not spec.enabled:
        typer.echo(warn(f"{app}/{name} is disabled — rehearsing it anyway"))
    for line in conductor_setup.readiness_warnings(spec):
        typer.echo(warn(line))
    program = conductor_setup.build_program(
        app, spec, pack, values, channel_mod.load_channel()
    )
    history: list = [
        SystemMessage(content="rehearsal"),
        UserMessage(content=f"rehearse {app}/{name}"),
    ]
    micro = None
    try:
        async with McpClient() as mcp:
            # Built AFTER the connection, so "start the server first" is
            # what a user without one hears — not a model-config error for
            # a provider the rehearsal never got to use.
            micro = _micro_caller() if program.needs_micro else None
            for _ in range(REHEARSE_MAX_TURNS):
                step = program.advance(history)
                while isinstance(step, DecisionRequest):
                    outcome = (await micro.run(step)).outcome if micro else None
                    step = program.resolve(outcome)
                if step is None:
                    return "walk finished or handed over — see the notes above"
                note, act = step.tool_calls
                typer.echo(f"  {note.arguments['summary']}")
                if act.name == "end_session":
                    # The walk wants to suspend for a later wake. A
                    # rehearsal has no later wake, and `_suspend` already
                    # wrote the file — drop it so it cannot ambush the
                    # next real session.
                    from physiclaw.agent.conductor.suspension import clear_suspended

                    clear_suspended()
                    return "walk suspended waiting on you — suspension dropped"
                text, is_error = await _dispatch(mcp, act, pack, app)
                history.append(step)
                history.append(
                    ToolResultMessage(
                        tool_call_id=act.id, content=text, is_error=is_error
                    )
                )
        return f"stopped after {REHEARSE_MAX_TURNS} turns"
    finally:
        if micro is not None:
            await micro.aclose()


async def _dispatch(mcp, call, pack: "Pack", app: str) -> tuple[str, bool]:
    """One synthesized action → the text its result carries.

    Routes the way the engine does: `run_macro` is the LOCAL tool (it
    runs a macro through the macro runner, which drives the same MCP
    connection step by step), everything else is a plain MCP call. The
    reply text is what the Program reads a screen out of, so both arms
    must hand back the same listing shape."""
    from physiclaw.agent.macros import runner as macro_runner
    from physiclaw.common import gesture_vocab, verdict

    try:
        if call.name == gesture_vocab.RUN_MACRO:
            qualified = call.arguments.get("name", "")
            spec = pack.macros.get(qualified.partition("/")[2])
            if spec is None:
                return f"unknown pack macro {qualified!r}", True
            result = await macro_runner.run_and_record(
                spec, call.arguments.get("inputs") or {}, mcp, caller="cli"
            )
            return verdict.screen_text(result.blocks), not result.ok
        return verdict.screen_text(
            await mcp.call_tool(call.name, call.arguments)
        ), False
    except Exception as e:  # a rehearsal reports, it does not crash
        return f"{call.name} failed: {e}", True


def _micro_caller():
    """The decision channel a rehearsal needs — same resolution as the
    engine's (`[conductor] micro_model`, else the session model), so a
    rehearsal spends the model a real wake would."""
    from physiclaw.agent.conductor.micro import MicroCaller
    from physiclaw.agent.provider import make_provider
    from physiclaw.common.config import CONFIG, model_ref, parse_model_ref

    try:
        ref = CONFIG.conductor.micro_model or model_ref()
    except RuntimeError as e:
        exit_error(str(e))
    pid, mid = parse_model_ref(ref)
    return MicroCaller(
        make_provider(pid, mid), confidence_floor=CONFIG.conductor.micro_confidence
    )
