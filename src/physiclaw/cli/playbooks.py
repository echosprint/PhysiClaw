"""`physiclaw playbooks` — scaffold, list, validate, and rehearse app packs.

Mirrors the macros CLI's shapes throughout: `init` prints the next
steps, `check` is the all-or-nothing gate with valid-is-not-live
warnings, and `run` rehearses one playbook against the live server the
way `macros run` rehearses one macro — the test you do BEFORE enabling.
The scaffolding itself lives in `conductor/scaffold.py` (the
`store.init_macro` split: the CLI only prints).

Nothing here writes cross-wake state. A rehearsal drives the phone now
and stops; the conductor's own doors at wake are the suspension file and
the overture."""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Optional

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
    from physiclaw.conductor.playbook import Pack, PlaybookEntry

playbooks_app = typer.Typer(no_args_is_help=True)


@playbooks_app.command()
def init(
    app: Annotated[str, typer.Argument(help="App name (lowercase/digits/hyphens).")],
) -> None:
    """Scaffold a new app pack: one PLAYBOOK.yml (meta + pages + an
    example walk) and an example pack macro — parse-clean, disabled."""
    from physiclaw.common.paths import PACK_FILENAME
    from physiclaw.conductor import scaffold
    from physiclaw.conductor.pages import CHANNEL_APP, IOS_APP, THREAD_PAGE
    from physiclaw.conductor.playbook import PlaybookError

    try:
        root = scaffold.init_pack(app)
    except PlaybookError as e:
        exit_error(str(e))
    typer.echo(ok(str(root)))
    typer.echo("Next:")
    if app == CHANNEL_APP:
        typer.echo(
            f"  1. anchor the `{THREAD_PAGE}` page on YOUR chat header in {PACK_FILENAME}"
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
            f"  1. declare `pages:` in {PACK_FILENAME} (physiclaw conductor propose --live)"
        )
        typer.echo(
            "  2. write pack macros + the playbook, then: physiclaw playbooks check"
        )
        typer.echo("  3. capture geometry: physiclaw conductor calibrate " + app)


@playbooks_app.command()
def install(
    src: Annotated[
        Path,
        typer.Argument(
            help="A template pack directory (e.g. the repo's playbooks/taobao)."
        ),
    ],
    set_values: Annotated[
        Optional[list[str]],
        typer.Option(
            "--set",
            help="Fill a placeholder non-interactively: KEY=VALUE (repeatable).",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Replace an already-installed pack wholesale."),
    ] = False,
) -> None:
    """Install a shared template pack into ~/.physiclaw/playbooks/<app>
    verbatim — `<<PLACEHOLDER>>` tokens stay in the files (diffable
    against the template); their values — per-installation constants
    like the IM contact name — go to playbooks/placeholders.yml, where
    every parser fills them at load. Everything lands disabled:
    rehearse, then enable."""
    import shutil

    from physiclaw.common import paths
    from physiclaw.common.placeholders import (
        PLACEHOLDER_VALUES_FILENAME,
        find_placeholders,
        placeholder_values,
        write_placeholder_values,
    )
    from physiclaw.common.text import read_text, write_text
    from physiclaw.conductor import scaffold
    from physiclaw.conductor.playbook import PlaybookError

    if not src.is_dir():
        exit_error(f"{src} is not a directory")
    app = src.name
    dest = paths.playbooks_dir() / app

    files = sorted(src.rglob("*.yml"))
    if not files:
        exit_error(f"{src} contains no pack files")
    texts = {f: read_text(f) for f in files}
    # PLAYBOOK.yml is the pack manifest — the `action.yml` analog:
    # `name`/`description` plus the `placeholders:` map that drives the
    # prompts below. It installs WITH the pack, tokens intact.
    try:
        meta = scaffold.read_template_manifest(src)
    except PlaybookError as e:
        exit_error(str(e))
    if meta.get("description"):
        typer.echo(meta["description"])
    tokens = list(
        dict.fromkeys(t for text in texts.values() for t in find_placeholders(text))
    )
    try:
        existing = placeholder_values()
    except ValueError as e:
        exit_error(str(e))
    new_values = _gather_values(
        tokens, meta.get("placeholders") or {}, set_values, existing
    )

    if dest.exists():
        if not force:
            exit_error(
                f"{dest} already exists — remove it or pass --force to replace it"
            )
        shutil.rmtree(dest)
    if new_values:
        write_placeholder_values({**existing, **new_values})
        typer.echo(
            f"values recorded in playbooks/{PLACEHOLDER_VALUES_FILENAME}: "
            + ", ".join(sorted(new_values))
        )
    for f in files:
        # Tokens stay in the copied files: the parsers fill them from
        # placeholders.yml at load (and `_check_app` below proves it).
        out = dest / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_text(out, texts[f])
    typer.echo(f"installed {app} → {dest} ({len(files)} file(s))")

    # The check's own output surfaces real errors; its disabled-state
    # warnings are a fresh install's EXPECTED shape, so the next steps
    # print unconditionally.
    _check_app(app)
    typer.echo(
        "next: rehearse (`physiclaw playbooks run`), capture pages "
        f"(`physiclaw conductor calibrate {app}`), then set `enabled: true`."
    )


def _gather_values(
    tokens: list[str],
    manifest: dict,
    set_values: list[str] | None,
    existing: dict[str, str],
) -> dict[str, str]:
    """The values to ADD to placeholders.yml: `--set` pairs first (they
    may override an existing value), a prompt (with the manifest's
    prose) for tokens with no value yet — every value plain and
    non-empty."""
    from physiclaw.common.placeholders import find_placeholders

    values = parse_inputs(set_values or [], flag="--set")
    unknown = sorted(set(values) - set(tokens))
    if unknown:
        exit_error(
            f"--set names no placeholder in this pack: {', '.join(unknown)} "
            f"(pack has: {', '.join(tokens) or '(none)'})"
        )
    for tok in tokens:
        if tok in values or tok in existing:
            continue
        meta = manifest.get(tok) or {}
        prompt = f"{tok} — {meta.get('description', 'value')}"
        if meta.get("example"):
            prompt += f" (e.g. {meta['example']})"
        values[tok] = typer.prompt(prompt).strip()
    for tok, value in values.items():
        if not value or find_placeholders(value):
            exit_error(f"placeholder {tok} needs a plain non-empty value")
    return values


@playbooks_app.command("list")
def list_cmd() -> None:
    """List every pack and its playbooks: enabled, disabled, or invalid."""
    from physiclaw.conductor import playbook as pb
    from physiclaw.conductor import scaffold

    scaffold.ensure_format_readme()
    apps = pb.list_apps()
    if not apps:
        typer.echo(
            "No app packs found. Scaffold one (physiclaw playbooks init "
            "<app>) or install a shared template (physiclaw playbooks "
            "install <dir>) — see ~/.physiclaw/playbooks/README.md"
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
    moves spend real micro-calls, and an `ask` really messages you and
    waits for your reply. Works while the playbook is disabled — rehearse
    first, enable after. Nothing is persisted: a walk that suspends here
    stops instead of leaving a file for the next wake."""
    from physiclaw.conductor.playbook import PlaybookError

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
def stats(
    last: Annotated[
        int,
        typer.Option("--last", help="How many recent runs to list (0 = none)."),
    ] = 10,
) -> None:
    """Per-playbook walk outcomes from playbooks/runs.jsonl — the
    escalation-rate KPI. A playbook that keeps handing over at one node
    is a rehearsal bug this table points at."""
    from physiclaw.conductor import walklog

    rows = walklog.load()
    if not rows:
        typer.echo(
            "No walks recorded yet — runs land in playbooks/runs.jsonl "
            "as playbooks execute (wakes and rehearsals alike)."
        )
        return
    for key, st in sorted(walklog.summarize(rows).items()):
        typer.echo(
            f"{key}: runs={st.runs} completed={st.completed} "
            f"suspended={st.suspended} handover={st.handover} "
            f"crashed={st.crashed} abandoned={st.abandoned} "
            f"escalation={st.escalation_rate:.0%} "
            f"micros={st.micros} rescues={st.rescues}"
        )
        for node, count, reason in st.hot_nodes():
            typer.echo(f"  ✗ {node} ×{count}  {reason}")
    if last > 0:
        typer.echo("\nrecent:")
        for rec in rows[-last:]:
            where = rec.get("node") or "(end)"
            reason = str(rec.get("reason") or "")
            typer.echo(
                f"  [{rec.get('ts', '?')}] {rec.get('app', '?')}/"
                f"{rec.get('playbook', '?')} {rec.get('outcome', '?')} "
                f"at {where}" + (f" — {reason}" if reason else "")
            )


@playbooks_app.command()
def propose(
    top: Annotated[
        int,
        typer.Option("--top", help="How many escalation sites to triage."),
    ] = 3,
) -> None:
    """Turn the hottest escalation sites into concrete authoring work:
    where walks keep handing over, why, and the exact mining commands
    that make a pack patch out of the recorded evidence. Nothing
    self-applies — every escalation the author fixes here prevents the
    next one."""
    from physiclaw.conductor import walklog

    sites = walklog.escalation_sites(walklog.load(), top=top)
    if not sites:
        typer.echo("No escalations recorded — nothing to propose.")
        return
    for s in sites:
        typer.echo(f"{s.app}/{s.playbook} escalates at {s.node} ×{s.count}")
        typer.echo(f"  last reason: {s.reason}")
        sid = s.sessions[-1] if s.sessions else "<session-id>"
        if s.sessions:
            typer.echo(f"  sessions to mine: {', '.join(s.sessions)}")
        typer.echo("  work it:")
        typer.echo(f"    physiclaw conductor match {s.app} --session {sid}")
        typer.echo(f"    physiclaw conductor extract {sid} --out {s.app}-corpus.jsonl")
        typer.echo(
            f"    physiclaw conductor calibrate {s.app} {s.app}-corpus.jsonl"
            "  (after labeling)"
        )


@playbooks_app.command()
def check() -> None:
    """Validate every pack: pages, pack macros, playbooks. Exit 1 if any
    is invalid."""
    from physiclaw.conductor import playbook as pb
    from physiclaw.conductor import scaffold

    scaffold.ensure_format_readme()
    apps = pb.list_apps()
    if not apps:
        typer.echo("No app packs found.")
        return
    if any([_check_app(app) for app in apps]):
        raise typer.Exit(1)


def _check_app(app: str) -> bool:
    """Report one pack; True when anything in it is invalid."""
    from physiclaw.conductor import playbook as pb
    from physiclaw.conductor import setup as conductor_setup

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
    from physiclaw.conductor.playbook import disabled_leg_macros

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
    from physiclaw.agent.engine.mcp_tool import McpClient
    from physiclaw.conductor import channel as channel_mod
    from physiclaw.conductor import setup as conductor_setup
    from physiclaw.conductor.ledger import check_ledger_value
    from physiclaw.conductor.micro import DecisionRequest
    from physiclaw.contract.dto import SystemMessage, ToolResultMessage, UserMessage

    spec, pack = conductor_setup.load_spec(app, name, require_live=False)
    values = conductor_setup.resolve_inputs(spec, values)
    # A `type: list` value is a JSON ledger — hold it to the same contract
    # the overture's activation does, so a typo'd list fails here rather
    # than mid-walk at the shopping loop.
    check_ledger_value(spec, values)
    if not spec.enabled:
        typer.echo(warn(f"{app}/{name} is disabled — rehearsing it anyway"))
    for line in conductor_setup.readiness_warnings(spec):
        typer.echo(warn(line))
    channel = channel_mod.load_channel()
    program = conductor_setup.build_program(app, spec, pack, values, channel)
    # The dispatch registry a real wake would arm — one spelling, shared
    # with `session_setup` (a gate's ask dispatches `channel/send`,
    # which is not this pack's macro).
    registry = conductor_setup.walk_registry(program, channel)
    history: list = [
        SystemMessage(content="rehearsal"),
        UserMessage(content=f"rehearse {app}/{name}"),
    ]
    micro = None
    try:
        async with McpClient() as mcp:
            # A rehearsal drives the phone NOW — wake it first if it
            # locked between runs (the runtime's overture does this at
            # every real wake; a rehearsal owes the walk the same floor).
            await _unlock_if_covered(mcp)
            for _ in range(REHEARSE_MAX_TURNS):
                step = program.advance(history)
                while isinstance(step, DecisionRequest):
                    # Built on FIRST use (a rescue clear_overlay can fire
                    # even on a pure-LEG walk), and after the connection —
                    # so "start the server first" is what a user without
                    # one hears, and a walk that never decides never pays
                    # a model-config error either.
                    if micro is None:
                        micro = _micro_caller()
                    outcome = (await micro.run(step)).outcome
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
                    from physiclaw.conductor.suspension import clear_suspended

                    clear_suspended()
                    return "walk suspended waiting on you — suspension dropped"
                text, is_error = await _dispatch(mcp, act, registry)
                history.append(step)
                history.append(
                    ToolResultMessage(
                        tool_call_id=act.id, content=text, is_error=is_error
                    )
                )
        return f"stopped after {REHEARSE_MAX_TURNS} turns"
    finally:
        # A rehearsal cut short (Ctrl-C, the turn cap) records its
        # abandoned row like a real wake's teardown would — latched, so
        # a walk that closed properly is a no-op.
        program.abandon()
        if micro is not None:
            await micro.aclose()


async def _unlock_if_covered(mcp) -> None:
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
            typer.echo("  phone is locked — unlocking first")
            await mcp.call_tool(gesture_vocab.UNLOCK_PHONE, {})
    except Exception as e:
        typer.echo(warn(f"unlock preamble skipped ({e})"))


async def _dispatch(mcp, call, registry: dict) -> tuple[str, bool]:
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
    from physiclaw.common.config import CONFIG, model_ref, parse_model_ref
    from physiclaw.conductor.micro import MicroCaller
    from physiclaw.provider import make_provider

    try:
        ref = CONFIG.conductor.micro_model or model_ref()
    except RuntimeError as e:
        exit_error(str(e))
    pid, mid = parse_model_ref(ref)
    return MicroCaller(
        make_provider(pid, mid), confidence_floor=CONFIG.conductor.micro_confidence
    )
