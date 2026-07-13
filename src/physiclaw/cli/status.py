"""``physiclaw status`` — quick snapshot (compare to `doctor` which probes deeper)."""

import sys

import typer

from physiclaw.cli._format import next_hint, section
from physiclaw.cli._update_check import maybe_print_update_banner
from physiclaw.common import paths


def status() -> None:
    """Show calibration + model state at a glance. Doesn't probe hardware."""
    typer.echo(section("PhysiClaw status"))

    model = paths.omniparser_onnx()
    if model.exists():
        size_mb = model.stat().st_size / 1024 / 1024
        typer.echo(
            typer.style("  vision model  ", fg=typer.colors.GREEN)
            + f"ok  ({size_mb:.0f} MB)"
        )
    else:
        typer.echo(typer.style("  vision model  ", fg=typer.colors.YELLOW) + "missing")

    data = paths.load_calibration_bundle()
    if data is None:
        typer.echo(typer.style("  calibration   ", fg=typer.colors.YELLOW) + "missing")
    else:
        complete = bool(data.get("complete"))
        tag = "complete" if complete else "partial"
        color = typer.colors.GREEN if complete else typer.colors.YELLOW
        typer.echo(typer.style("  calibration   ", fg=color) + tag)

    jobs = paths.jobs_file()
    if jobs.exists():
        typer.echo(typer.style("  jobs file     ", fg=typer.colors.GREEN) + str(jobs))
    else:
        typer.echo(
            typer.style("  jobs file     ", fg=typer.colors.YELLOW)
            + f"none yet ({jobs})"
        )

    # Suppress on pipe — `status` is a quick snapshot meant for grep/jq.
    if sys.stdout.isatty():
        typer.echo()
        typer.echo(
            next_hint(
                "physiclaw doctor  (for deeper checks — server, hardware, provider)"
            )
        )

    maybe_print_update_banner()
