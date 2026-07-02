"""``physiclaw reset`` — clear the learned screen layout.

Deletes ``~/.physiclaw/screen-layout/`` (the input-box / keyboard / Paste
bboxes learned on first run). On the next run PhysiClaw re-learns them — the
first-run hook wakes the agent to run the ``screen-layout`` skill. Calibration,
memory, models, and config are left untouched (use ``physiclaw uninstall`` for
those).
"""

import shutil
from typing import Annotated

import typer

from physiclaw import paths


def reset(
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip the confirmation prompt."),
    ] = False,
) -> None:
    """Clear the learned screen layout so it's re-learned on the next run."""
    d = paths.screen_layout_dir()
    files = [p for p in d.rglob("*") if p.is_file()] if d.exists() else []
    if not files:
        typer.echo("No learned screen layout to reset.")
        return

    typer.echo(f"Will clear the learned screen layout: {d}  ({len(files)} file(s))")
    if not yes and not typer.confirm("Reset it?", default=False):
        typer.echo("Cancelled.")
        raise typer.Exit(1)

    shutil.rmtree(d, ignore_errors=True)
    typer.echo("Screen layout cleared — PhysiClaw will re-learn it on the next run.")
