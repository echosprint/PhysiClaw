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
from physiclaw.conductor import rehearsal

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

    Drives the real walk on the real phone: moves run their macros, agent
    steps spend real model calls, and an `ask` really messages you and
    waits for your reply. Works while the playbook is disabled — rehearse
    first, enable after. Nothing is persisted: a walk that suspends here
    stops instead of leaving a file for the next wake."""
    from physiclaw.conductor.playbook import PlaybookError

    app, name = _split_ref(ref)
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
    except RuntimeError as e:
        # `rehearsal.micro_caller` — an agent step fired with no model configured.
        exit_error(str(e))
    typer.echo(outcome)


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
    from physiclaw.cli.conductor import resolve_sid
    from physiclaw.common.text import read_text
    from physiclaw.conductor import corpus
    from physiclaw.conductor import replay as replay_mod
    from physiclaw.conductor.playbook import PlaybookError

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
        # Advisories: an ask quoting none of its reply words, a route
        # page too thinly anchored to survive one OCR miss.
        for line in conductor_setup.readiness_warnings(entry.spec, pack):
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
    from physiclaw.conductor.playbook import disabled_macros

    if disabled:
        typer.echo(
            warn(
                f"{app}: disabled, so the boot will not offer: "
                f"{', '.join(disabled)}. Set `enabled: true` once rehearsed."
            )
        )
    not_live = sorted(
        {
            m
            for e in entries
            if e.spec is not None
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
    """`<app>/<playbook>` — the one spelling of a playbook ref on the CLI."""
    app, _, name = ref.partition("/")
    if not app or not name:
        exit_error(f"expected <app>/<playbook> (got {ref!r})")
    return app, name


# ---------- rehearsal (`playbooks run`) ----------
# The core lives in `conductor.rehearsal` — the studio's rehearse button
# drives the same loop; this layer adds the connection and typer skin.


async def _rehearse(app: str, name: str, values: dict[str, str]) -> str:
    """Arm first (bad inputs fail before any connection), then drive the
    walk over one client, printing each synthesized turn as it goes."""
    # Local import: the mcp SDK only loads when actually rehearsing.
    from physiclaw.agent.engine.mcp_tool import McpClient

    program, registry = rehearsal.arm(
        app, name, values, emit_warn=lambda s: typer.echo(warn(s))
    )
    async with McpClient() as mcp:
        return await rehearsal.walk(program, registry, mcp, emit=typer.echo)
