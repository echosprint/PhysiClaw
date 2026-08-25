"""Shared output formatters for ``physiclaw`` CLI commands.

Every command uses the same ``✓ ok`` / ``! warn`` line shape and the same
"Next: <cmd>" footer; centralizing here keeps tone uniform and avoids
ad-hoc ``typer.style(...fg=...)`` repetition.

Two voices live here: the flat ``ok``/``warn``/``info`` lines most
commands echo, and the indented ``step_*`` lines of the setup wizard
(``physiclaw setup hardware``), which mirror the browser wizard's step
markers. ``exit_error`` is the one failure shape for expected errors.
"""

from typing import NoReturn

import typer


def ok(msg: str) -> str:
    """Green ``✓`` prefix + msg."""
    return typer.style("✓ ", fg=typer.colors.GREEN) + msg


def warn(msg: str) -> str:
    """Yellow ``!`` prefix + msg."""
    return typer.style("! ", fg=typer.colors.YELLOW) + msg


def next_hint(line: str) -> str:
    """Bold ``Next:`` prefix + the rest of the line."""
    return typer.style("Next:", bold=True) + " " + line


def info(msg: str) -> str:
    """Two-space indent + msg. Use for neutral state lines that shouldn't
    read as a problem (unlike ``warn``) or a confirmation (unlike ``ok``)."""
    return f"  {msg}"


def section(title: str) -> str:
    """Bold bright-cyan section header. Distinct from ok (green) and warn
    (yellow) so the reader can scan section boundaries at a glance in
    commands that emit multi-section reports (``doctor``, ``status``)."""
    return typer.style(title, fg=typer.colors.BRIGHT_CYAN, bold=True)


def step_ok(msg: str = "OK") -> str:
    """Setup-wizard step line: two-space indent + green ``✓`` + msg."""
    return f"  {typer.style('✓', fg=typer.colors.GREEN)} {msg}"


def step_fail(msg: str) -> str:
    """Setup-wizard step failure: two-space indent + red ``✗ msg``."""
    return "  " + typer.style(f"✗ {msg}", fg=typer.colors.RED)


def step_warn(msg: str) -> str:
    """Setup-wizard step warning: two-space indent + yellow ``⚠ msg``."""
    return "  " + typer.style(f"⚠ {msg}", fg=typer.colors.YELLOW)


def exit_error(msg: str, code: int = 1) -> NoReturn:
    """Echo ``error: <msg>`` to stderr and abort with ``code`` — the one
    failure shape expected CLI errors use (no traceback for bad input,
    network down, corrupt downloads, missing files)."""
    typer.echo(f"error: {msg}", err=True)
    raise typer.Exit(code=code)


def parse_inputs(pairs: list[str]) -> dict[str, str]:
    """Repeatable ``--input NAME=VALUE`` pairs → dict — the one parser for
    every command that feeds declared inputs (`macros rehearse`,
    `playbooks arm`). Exits code 2 on a malformed pair."""
    values: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            exit_error(f"bad --input {pair!r} (want key=value)", code=2)
        values[key] = value
    return values


def state_tag(*, valid: bool, enabled: bool = False) -> str:
    """The colored invalid/enabled/disabled tag both artifact listings
    (`macros list`, `playbooks list`) render — one palette, one wording."""
    import typer

    if not valid:
        return typer.style("invalid ", fg=typer.colors.RED)
    if enabled:
        return typer.style("enabled ", fg=typer.colors.GREEN)
    return typer.style("disabled", fg=typer.colors.YELLOW)
