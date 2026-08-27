"""`physiclaw playbooks` — scaffold, list, validate, and arm app packs.

Nothing here executes a playbook: `arm` only writes the arm file the
engine reads at the next wake (`agent/conductor/arming.py`). Mirrors
the macros CLI's shapes — init prints the next steps, check is the
all-or-nothing gate with valid-is-not-live warnings — while the
scaffolding itself lives in `agent/conductor/scaffold.py` (the
`store.init_macro` split: the CLI only prints)."""

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
    from physiclaw.agent.conductor.pages import CHANNEL_APP, THREAD_PAGE
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
        typer.echo("  3. rehearse both (physiclaw macros rehearse is per-user-macro;")
        typer.echo("     drive them via an armed test playbook), then enable")
        typer.echo(
            f"  4. capture geometry: physiclaw conductor calibrate {CHANNEL_APP}"
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
    from physiclaw.agent.conductor import arming, scaffold
    from physiclaw.agent.conductor import playbook as pb

    scaffold.ensure_format_readme()
    apps = pb.list_apps()
    if not apps:
        typer.echo(
            "No app packs found. Scaffold one: physiclaw playbooks init <app> "
            "(see ~/.physiclaw/playbooks/README.md)"
        )
        return
    armed = arming.armed_ref()
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
            mark = "  [armed]" if (app, e.name) == armed else ""
            typer.echo(f"  {tag} {e.name}{mark}  {detail}")


@playbooks_app.command()
def arm(
    ref: Annotated[str, typer.Argument(help="<app>/<playbook> to arm.")],
    inputs: Annotated[
        list[str] | None,
        typer.Option(
            "--input",
            "-i",
            help="NAME=VALUE for a declared playbook input (repeatable).",
        ),
    ] = None,
) -> None:
    """Arm one playbook: the next engine sessions follow it (legs run as
    synthesized turns; anything it can't handle hands over to the model).
    A manual testing surface — one playbook at a time, sticky until
    `disarm`."""
    from physiclaw.agent.conductor import arming
    from physiclaw.agent.conductor.playbook import PlaybookError

    app, _, name = ref.partition("/")
    if not app or not name:
        exit_error(f"expected <app>/<playbook> (got {ref!r})")
    values = parse_inputs(inputs or [])
    try:
        spec, warnings = arming.arm(app, name, values)
    except PlaybookError as e:
        exit_error(str(e))
    typer.echo(ok(f"armed {ref} ({len(spec.nodes)} nodes)"))
    for w in warnings:
        typer.echo(warn(w))
    typer.echo("Sticky until `physiclaw playbooks disarm`.")


@playbooks_app.command()
def disarm() -> None:
    """Disarm — sessions go back to plain model-driven turns. Also drops
    a suspended walk: disarming means stop, and a surviving suspension
    would resume on the next wake."""
    from physiclaw.agent.conductor import arming

    was_armed, dropped = arming.stand_down()
    if not was_armed and dropped is None:
        typer.echo("nothing was armed")
        return
    if was_armed:
        typer.echo(ok("disarmed"))
    if dropped is not None:
        typer.echo(ok(f"dropped the suspended walk for {dropped[0]}/{dropped[1]}"))


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
    return bad


def _report_not_live(
    app: str,
    pack: "Pack",
    entries: "list[PlaybookEntry]",
    disabled: list[str],
) -> None:
    """Valid is not live: a green check invites the wrong assumption. Say
    which playbooks the conductor cannot arm, and why — disabled files,
    and referenced pack macros that are themselves disabled (rehearse,
    then enable; the rehearsal gate itself lands with execution)."""
    from physiclaw.agent.conductor.playbook import disabled_leg_macros

    if disabled:
        typer.echo(
            warn(
                f"{app}: disabled, so the conductor cannot arm: "
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
