"""`physiclaw playbooks` — the agent's door to app packs: scaffold,
validate, rehearse, and step them; `pages` for their fingerprints.

The CLI is a skin. Every command is typer around one driver — the
rehearsal core (`conductor/rehearsal.py`), the stepping driver
(`debug/stepping.py`), the replay (`conductor/replay.py`), the page
tools (`cli/pages.py` over `conductor/capture.py`) — and the studio
puts a browser UI over the same drivers for a person. Mirrors the
macros CLI's shapes throughout: `init` prints the next steps, `check`
is the all-or-nothing gate with valid-is-not-live warnings, and `run`
rehearses one playbook against the live server the way `macros run`
rehearses one macro — the test you do BEFORE enabling. The scaffolding
itself lives in `conductor/scaffold.py` (the `store.init_macro` split:
the CLI only prints).

Nothing here writes cross-wake state. A rehearsal drives the phone now
and stops; the conductor's own doors at wake are the suspension file and
the channel pack's boot."""

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
from physiclaw.cli.pages import pages_app
from physiclaw.common.paths import PACK_FILENAME
from physiclaw.common.ready import START_HINT
from physiclaw.conductor.drive import rehearsal
from physiclaw.conductor.spec.model import PlaybookError

if TYPE_CHECKING:
    from physiclaw.conductor.spec.model import Pack, Playbook, PlaybookEntry

playbooks_app = typer.Typer(no_args_is_help=True)

# The two knobs `run` and `step` share, spelled once.
VerboseOpt = Annotated[
    bool, typer.Option("--verbose", "-v", help="Print every result's listing rows.")
]
RawOpt = Annotated[
    bool,
    typer.Option(
        "--raw",
        help="Print every model round-trip: the messages as sent to the "
        "provider and the reply as received (an agent step's context, "
        "prompt, screen block, and answer).",
    ),
]
playbooks_app.add_typer(
    pages_app,
    name="pages",
    help="A pack's page fingerprints: propose anchors, match screens, "
    "extract a corpus, calibrate geometry.",
)


@playbooks_app.command()
def init(
    app: Annotated[str, typer.Argument(help="App name (lowercase/digits/hyphens).")],
) -> None:
    """Scaffold a new app pack: one PLAYBOOK.yml (meta + pages + an
    example walk) and an example pack macro — parse-clean, disabled."""
    from physiclaw.common.paths import PACK_FILENAME
    from physiclaw.conductor.spec import scaffold
    from physiclaw.conductor.spec.conventions import (
        BOOT_PLAYBOOK,
        CHANNEL_APP,
        IOS_APP,
        THREAD_PAGE,
    )
    from physiclaw.conductor.spec.specfile import SpecError

    try:
        root = scaffold.init_pack(app)
    except SpecError as e:
        exit_error(str(e))
    typer.echo(ok(str(root)))
    typer.echo("Next:")
    if app == CHANNEL_APP:
        typer.echo(
            f"  1. anchor the `{THREAD_PAGE}` page on YOUR chat header in {PACK_FILENAME}"
        )
        typer.echo("  2. record the send/open gesture paths in macros/*/MACRO.yml")
        typer.echo("  3. rehearse both, then enable (physiclaw macros run is")
        typer.echo("     per-user-macro; drive a pack's via physiclaw playbooks run);")
        typer.echo(f"     the boot ({BOOT_PLAYBOOK}.yml) goes live with `open`")
        typer.echo(
            f"  4. capture geometry: physiclaw playbooks pages calibrate {CHANNEL_APP}"
        )
    elif app == IOS_APP:
        typer.echo("  1. check the lock-screen reading matches your phone's language")
        typer.echo("     (lock it, then: physiclaw playbooks pages propose --live)")
        typer.echo(
            f"  2. capture geometry: physiclaw playbooks pages calibrate {IOS_APP} --guided"
        )
    else:
        typer.echo(
            f"  1. declare `pages:` in {PACK_FILENAME} (physiclaw playbooks pages propose --live)"
        )
        typer.echo(
            f"  2. write the route in {scaffold.EXAMPLE_PLAYBOOK}.yml (one playbook "
            "per file beside the manifest) and its macros, then: physiclaw "
            "playbooks check"
        )
        typer.echo("  3. capture geometry: physiclaw playbooks pages calibrate " + app)


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
    from physiclaw.conductor.spec import scaffold
    from physiclaw.conductor.spec.specfile import SpecError

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
    except SpecError as e:
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
        f"(`physiclaw playbooks pages calibrate {app}`), then set `enabled: true`."
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


def _warn_strays() -> None:
    from physiclaw.conductor.spec import pack as pb

    for stray in pb.stray_dirs():
        typer.echo(
            warn(
                f"{stray} holds playbook files but no {PACK_FILENAME} — not a pack "
                "until the manifest exists (it may be empty)"
            )
        )


@playbooks_app.command("list")
def list_cmd() -> None:
    """List every pack and its playbooks: enabled, disabled, or invalid."""
    from physiclaw.conductor.spec import pack as pb
    from physiclaw.conductor.spec import scaffold

    scaffold.ensure_format_readme()
    _warn_strays()
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
        try:
            entries = pb.scan_playbooks(app)
        except pb.PlaybookError as e:
            typer.echo(f"  {step_fail(str(e))}")
            continue
        for entry in entries:
            tag = state_tag(
                valid=entry.spec is not None,
                enabled=bool(entry.spec and entry.spec.enabled),
            )
            detail = (
                (entry.error or "")
                if entry.spec is None
                else f"{entry.spec.description} ({len(entry.spec.nodes)} nodes)"
            )
            typer.echo(f"  {tag} {entry.name}  {detail}")


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
    verbose: VerboseOpt = False,
    raw: RawOpt = False,
) -> None:
    """Rehearse one playbook against the running server — the mirror of
    `physiclaw macros run`, and the way to test a walk without waiting for
    a wake or hoping the boot picks it.

    Drives the real walk on the real phone: moves run their macros, agent
    steps spend real model calls, and an `ask` really messages you and
    waits for your reply. Works while the playbook is disabled — rehearse
    first, enable after. Nothing is persisted: a walk that suspends here
    stops instead of leaving a file for the next wake."""
    from physiclaw.conductor.spec.specfile import SpecError

    app, name = _split_ref(ref)
    try:
        outcome = asyncio.run(
            _rehearse(app, name, parse_inputs(inputs or []), verbose=verbose, raw=raw)
        )
    except SpecError as e:
        exit_error(str(e))
    except ConnectionError as e:
        # Same branch as `macros run` (`START_HINT` says why `mcp`).
        exit_error(f"{e}. {START_HINT}")
    except RuntimeError as e:
        # `rehearsal.micro_caller` — an agent step fired with no model configured.
        exit_error(str(e))
    typer.echo(outcome)


@playbooks_app.command()
def step(
    ref: Annotated[str, typer.Argument(help="<app>/<playbook> to step through.")],
    inputs: Annotated[
        list[str] | None,
        typer.Option(
            "--input",
            "-i",
            help="NAME=VALUE for a declared playbook input (a fresh walk only).",
        ),
    ] = None,
    at: Annotated[
        Optional[str],
        typer.Option(
            "--at",
            help="Put the cursor on this node (its name) first — jump there, "
            "or re-run the node after editing it.",
        ),
    ] = None,
    to: Annotated[
        Optional[str],
        typer.Option(
            "--to",
            help="Keep stepping until the cursor has passed this node "
            "(`end`: run the route out).",
        ),
    ] = None,
    outputs: Annotated[
        list[str] | None,
        typer.Option(
            "--output",
            "-o",
            help="NODE.FIELD=VALUE — seed a pure-text agent's answer; the walk "
            "starts past it without a call.",
        ),
    ] = None,
    reply: Annotated[
        Optional[str],
        typer.Option(
            "--reply",
            help="Stage the user's reply for the next ask (the user channel "
            "is virtual while stepping).",
        ),
    ] = None,
    start_at: Annotated[
        str,
        typer.Option(
            "--start-at",
            help="Begin this node's macro at this step (its handle).",
        ),
    ] = "",
    stop_after: Annotated[
        str,
        typer.Option(
            "--stop-after",
            help="Stop this node's macro after this step (same handle).",
        ),
    ] = "",
    verbose: VerboseOpt = False,
    raw: RawOpt = False,
    as_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Print the outcome and position as JSON (for an agent's loop).",
        ),
    ] = False,
    reset: Annotated[
        bool, typer.Option("--reset", help="Forget the stored position and stop.")
    ] = False,
    status: Annotated[
        bool, typer.Option("--status", help="Show the stored position and stop.")
    ] = False,
) -> None:
    """Step one playbook through the live phone, one node per invocation
    — the playbook debugger (the studio's playbook panel drives the
    same core with a mouse).

    Each invocation re-reads PLAYBOOK.yml (an edit applies at once),
    rebuilds the walk at its stored position (cursor, agent outputs, the
    gate's consent), runs the node there — its enter check, its macro or
    episode or ask, its verify check, its declared recovery — and pauses
    the moment the cursor moves, printing each turn's verdict, model
    answer, and macro step log. `--at` re-runs a node you just edited;
    `--start-at`/`--stop-after` narrow its macro to one gesture. The user
    channel is virtual (`--reply` stages the answer; the send still runs
    on the phone), no model ever takes over, and nothing is persisted
    beyond the position file (`debug/walk.json`). `--json` prints the
    outcome (paused / suspended / stopped / completed) and the position
    for a loop to read."""
    import json

    from physiclaw.conductor.spec.specfile import SpecError
    from physiclaw.debug import stepping

    app, name = _split_ref(ref)
    if reset:
        typer.echo(stepping.reset(app, name))
        return
    if status:
        try:
            pos = stepping.status(app, name)
        except SpecError as e:
            exit_error(str(e))
        if as_json:
            typer.echo(json.dumps(pos.to_dict() if pos else None, ensure_ascii=False))
        else:
            typer.echo(
                pos.describe()
                if pos
                else f"{app}/{name}: no stored position — the next step starts the route"
            )
        return
    try:
        result = asyncio.run(
            _step(
                app,
                name,
                parse_inputs(inputs or []),
                at=at,
                to=to,
                outputs=parse_inputs(outputs or [], "--output"),
                reply=reply,
                start_at=start_at,
                stop_after=stop_after,
                verbose=verbose,
                raw=raw,
                # Under --json stdout is the JSON alone; the progress
                # lines go to stderr, where a loop can still read them.
                emit=lambda line: typer.echo(line, err=as_json),
            )
        )
    except SpecError as e:
        exit_error(str(e))
    except ConnectionError as e:
        exit_error(f"{e}. {START_HINT}")
    except RuntimeError as e:
        exit_error(str(e))
    typer.echo(
        json.dumps(result.to_dict(), ensure_ascii=False) if as_json else result.message
    )


@playbooks_app.command()
def replay(
    ref: Annotated[str, typer.Argument(help="<app>/<playbook> to replay.")],
    session: Annotated[
        Optional[str],
        typer.Option("--session", help="A recorded session id (suffix ok if unique)."),
    ] = None,
    listings: Annotated[
        Optional[list[Path]],
        typer.Option("--listing", help="A listing text file, one screen (repeatable)."),
    ] = None,
    inputs: Annotated[
        list[str] | None,
        typer.Option("--input", "-i", help="NAME=VALUE for a declared input."),
    ] = None,
    outputs: Annotated[
        list[str] | None,
        typer.Option(
            "--output",
            "-o",
            help="NODE.FIELD=VALUE — what a pure-text agent step answers.",
        ),
    ] = None,
) -> None:
    """Replay the real walk over recorded screens — no phone, no model.
    Each screen is the result of the walk's next action; the report
    shows the node, the verdict it acted on, the tool, and where the
    walk stopped. A session's screens come from its wire log; a
    listing file is one screen. Writes nothing."""
    from physiclaw.cli._sessions import resolve_sid
    from physiclaw.common.text import read_text
    from physiclaw.conductor.drive import corpus
    from physiclaw.conductor.drive import replay as replay_mod

    app, name = _split_ref(ref)
    if (session is None) == (not listings):
        exit_error("need exactly one of --session or --listing", code=2)
    try:
        if listings:
            screens = [read_text(p) for p in listings]
        else:
            assert session is not None  # the exactly-one check above
            screens = corpus.session_listings(resolve_sid(session))
        program, _ = rehearsal.arm(
            app,
            name,
            parse_inputs(inputs or []),
            emit_warn=lambda s: typer.echo(warn(s)),
            dry=True,
        )
    except (OSError, PlaybookError) as e:
        exit_error(str(e))
    result = replay_mod.replay(
        program, screens, parse_inputs(outputs or [], flag="--output")
    )
    for i, t in enumerate(result.turns, 1):
        typer.echo(
            f"[{i:3d}] {t.node or '(end)':16s} {t.verdict:28s} {t.tool:12s} {t.note}"
        )
    typer.echo(f"{result.outcome}: {result.detail}")


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
    from physiclaw.conductor.walk import walklog

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
    from physiclaw.conductor.walk import walklog

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
        typer.echo(f"    physiclaw playbooks pages match {s.app} --session {sid}")
        typer.echo(
            f"    physiclaw playbooks pages extract {sid} --out {s.app}-corpus.jsonl"
        )
        typer.echo(
            f"    physiclaw playbooks pages calibrate {s.app} {s.app}-corpus.jsonl"
            "  (after labeling)"
        )


@playbooks_app.command()
def check() -> None:
    """Validate every pack: pages, pack macros, playbooks. Exit 1 if any
    is invalid."""
    from physiclaw.conductor.spec import lints, scaffold
    from physiclaw.conductor.spec import pack as pb

    scaffold.ensure_format_readme()
    _warn_strays()
    apps = pb.list_apps()
    if not apps:
        typer.echo("No app packs found.")
        return
    specs: dict[str, "Playbook"] = {}
    bad = False
    for app in apps:
        app_bad, app_specs = _check_app(app)
        bad = bad or app_bad
        specs.update(app_specs)
    for line in lints.menu_warnings(specs):
        typer.echo(warn(line))
    if bad:
        raise typer.Exit(1)


def _check_app(app: str) -> "tuple[bool, dict[str, Playbook]]":
    """Report one pack: (anything invalid, the valid playbooks by ref —
    what the cross-pack advisory reads, so nothing loads twice)."""
    from physiclaw.conductor.spec import lints
    from physiclaw.conductor.spec import pack as pb

    try:
        pack = pb.load_pack(app)
    except pb.PlaybookError as e:
        typer.echo(step_fail(str(e)))
        return True, {}
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
        # Advisories: an ask quoting none of its reply words, a route
        # page too thinly anchored to survive one OCR miss.
        for line in lints.readiness_warnings(entry.spec, pack):
            typer.echo(warn(f"{app}/{entry.name}: {line}"))
    return bad, {f"{app}/{e.name}": e.spec for e in entries if e.spec is not None}


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
    from physiclaw.conductor.spec.pack import disabled_macros

    if disabled:
        typer.echo(
            warn(
                f"{app}: disabled, so the boot will not offer: "
                f"{', '.join(disabled)}. Set `enabled: true` once rehearsed."
            )
        )
    # Only enabled playbooks: a disabled playbook's macros are not "not
    # live" beyond the playbook itself, already reported above.
    not_live = sorted(
        {
            m
            for e in entries
            if e.spec is not None and e.spec.enabled
            for m in disabled_macros(e.spec, pack)
        }
    )
    if not_live:
        typer.echo(
            warn(
                f"{app}: referenced pack macro(s) still disabled: "
                f"{', '.join(not_live)} — rehearse, then enable."
            )
        )


def _split_ref(ref: str) -> tuple[str, str]:
    """`<app>/<playbook>` — the pack's own parse, exiting on a bad one."""
    from physiclaw.conductor.spec.pack import split_ref
    from physiclaw.conductor.spec.specfile import SpecError

    try:
        return split_ref(ref)
    except SpecError as e:
        exit_error(str(e))


# ---------- rehearsal (`playbooks run`) ----------
# The core lives in `conductor.rehearsal` (`replay` shares its `arm`);
# this layer adds the connection and typer skin.


# ---------- stepping (`playbooks step`) ----------
# One call into the driver (`debug/stepping.py`); the CLI adds the MCP
# connection and typer.


async def _step(app: str, name: str, values: dict[str, str], *, emit, **opts):
    # Local import: the mcp SDK only loads when actually stepping.
    from physiclaw.agent.engine.mcp_tool import McpClient
    from physiclaw.debug import stepping

    async with McpClient() as mcp:
        return await stepping.step(
            app,
            name,
            mcp,
            values=values,
            emit=emit,
            emit_warn=lambda line: emit(warn(line)),
            **opts,
        )


async def _rehearse(
    app: str,
    name: str,
    values: dict[str, str],
    *,
    verbose: bool = False,
    raw: bool = False,
) -> str:
    """Arm first (bad inputs fail before any connection), then drive the
    walk over one client, printing each synthesized turn as it goes."""
    # Local import: the mcp SDK only loads when actually rehearsing.
    from physiclaw.agent.engine.mcp_tool import McpClient

    program, registry = rehearsal.arm(
        app, name, values, emit_warn=lambda s: typer.echo(warn(s))
    )
    async with McpClient() as mcp:
        return await rehearsal.walk(
            program,
            registry,
            mcp,
            emit=typer.echo,
            verbose=verbose,
            raw=raw,
        )
