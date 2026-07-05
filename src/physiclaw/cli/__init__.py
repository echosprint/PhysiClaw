"""PhysiClaw command-line interface."""

from typing import Annotated

import typer

from physiclaw import __version__ as _pkg_version
from physiclaw.cli.auto import auto
from physiclaw.cli.clear import clear
from physiclaw.cli.config import config_app
from physiclaw.cli.doctor import doctor
from physiclaw.cli.flash import flash
from physiclaw.cli.models import models_app
from physiclaw.cli.prompt import prompt_app
from physiclaw.cli.reset import reset
from physiclaw.cli.server import server
from physiclaw.cli.setup import setup_app
from physiclaw.cli.skills import skills_app
from physiclaw.cli.status import status
from physiclaw.cli.uninstall import uninstall
from physiclaw.cli.update import update

app = typer.Typer(
    help=f"PhysiClaw {_pkg_version} — let AI agents physically operate a phone.",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=False,
    add_completion=False,
)

app.command()(doctor)
app.command()(server)
app.command()(auto)
app.command()(status)
app.command()(update)
app.command()(flash)
app.command()(clear)
app.command()(reset)
app.command()(uninstall)

app.add_typer(
    setup_app,
    name="setup",
    help="One-time configuration: hardware, local models, phone.",
)
app.add_typer(
    config_app,
    name="config",
    help="Show or edit the user config file.",
)
app.add_typer(
    models_app,
    name="models",
    help="List, inspect, and switch the active model.",
)
app.add_typer(
    skills_app,
    name="skills",
    help="Install, list, and remove skills from a git-repo source.",
)
app.add_typer(
    prompt_app,
    name="prompt",
    help="Dump the SYSTEM prompt / full LLM request for inspection.",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"physiclaw, version {_pkg_version}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    # Bare `physiclaw` (no subcommand) defaults to `physiclaw server`.
    if ctx.invoked_subcommand is None:
        server()


__all__ = ["app"]
