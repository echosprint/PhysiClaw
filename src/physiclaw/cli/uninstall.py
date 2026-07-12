"""``physiclaw uninstall`` — remove PhysiClaw user data.

The CLI binary itself is removed via the package manager that
installed it (``uv tool uninstall physiclaw``). A process can't
reliably remove its own running binary — especially on Windows —
so this command prints that step as a final reminder rather than
trying to do it.

Default-keep: ``physiclaw uninstall`` (no flags) prompts before
deleting anything. Pass ``--data`` / ``--config`` / ``--all`` to
target specific pieces; ``--yes`` skips confirmations; ``--dry-run``
prints what would happen without touching the filesystem.
"""

import shutil
from pathlib import Path
from typing import Annotated

import typer

from physiclaw.common import config as _config
from physiclaw.common import paths


def _delete(target: Path) -> None:
    """Remove a file or directory tree."""
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()


def uninstall(
    data: Annotated[
        bool,
        typer.Option("--data", help="Remove the entire PhysiClaw data directory."),
    ] = False,
    config: Annotated[
        bool,
        typer.Option(
            "--config",
            help="Remove only the user config file. Calibration, memory, "
            "and downloaded models are kept.",
        ),
    ] = False,
    all_: Annotated[
        bool,
        typer.Option("--all", help="Alias for --data."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompts."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print what would be removed; don't delete."),
    ] = False,
) -> None:
    """Remove PhysiClaw user data; the CLI binary is removed manually via uv."""
    home = paths.HOME
    cfg_path = _config.config_path()

    if all_:
        data = True

    # Pick a target. --data wins over --config when both are passed (superset).
    if data:
        target: Path | None = home
        label = f"all PhysiClaw data ({home})"
    elif config:
        target = cfg_path
        label = f"config file ({cfg_path})"
    else:
        # Interactive: discover by asking.
        if not home.exists():
            typer.echo(f"No PhysiClaw data found at {home}.")
            target = None
        else:
            typer.echo(f"PhysiClaw data location: {home}")
            typer.echo(
                "Contents: calibration, memory, downloaded models, config, logs."
            )
            if typer.confirm("Remove everything?", default=False):
                target, label = home, f"all PhysiClaw data ({home})"
            else:
                target = None

    if target is not None:
        if not target.exists():
            typer.echo(f"{target} does not exist; nothing to remove.")
        elif dry_run:
            typer.echo(f"[dry-run] would remove: {target}")
        elif yes or typer.confirm(f"Remove {label}?", default=False):
            _delete(target)
            typer.echo(f"Removed: {target}")
        else:
            typer.echo("Cancelled.")
            raise typer.Exit(code=1)

    # Always print the manual final step.
    typer.echo()
    typer.echo("To remove the physiclaw CLI itself, run:")
    typer.echo("  uv tool uninstall physiclaw")
