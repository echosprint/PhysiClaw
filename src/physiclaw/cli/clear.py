"""``physiclaw clear`` — delete saved debug images.

Removes everything the ``--save-*`` flags on ``physiclaw server`` write under
``~/.physiclaw``: camera snapshots, phone screenshots, peek/screenshot
tool-call dumps, and raw camera frames. The dirs are recreated on the next
save. Calibration, memory, models, and config are left untouched (use
``physiclaw uninstall`` for those).
"""

import shutil
from typing import Annotated

import typer

from physiclaw.common import paths


def clear(
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip the confirmation prompt."),
    ] = False,
) -> None:
    """Delete all saved debug images (snapshots, screenshots, tool calls, raw camera)."""
    dirs = [
        paths.snapshots_dir(),
        paths.screenshots_dir(),
        paths.tool_calls_dir(),
        paths.raw_camera_dir(),
    ]

    present: list[tuple] = []
    total = 0
    size = 0
    for d in dirs:
        if not d.exists():
            continue
        files = [p for p in d.rglob("*") if p.is_file()]
        if not files:
            continue
        present.append((d, len(files)))
        total += len(files)
        size += sum(p.stat().st_size for p in files)

    if total == 0:
        typer.echo("No saved images to clear.")
        return

    typer.echo(f"Found {total} file(s) ({size / 1e6:.1f} MB):")
    for d, n in present:
        typer.echo(f"  {d}  ({n})")

    if not yes and not typer.confirm("Delete them?", default=False):
        typer.echo("Cancelled.")
        raise typer.Exit(1)

    for d, _ in present:
        shutil.rmtree(d, ignore_errors=True)
    typer.echo(f"Cleared {total} file(s).")
