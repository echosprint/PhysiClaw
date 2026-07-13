"""``physiclaw config`` — inspect and edit ``~/.physiclaw/config.toml``.

``set`` / ``unset`` use tomlkit so user comments + ordering survive the
round-trip. For secrets like API keys, prefer ``physiclaw config edit``
to keep them out of shell history.
"""

import dataclasses
import os
import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from physiclaw.cli._format import exit_error, ok
from physiclaw.common import config as _config

config_app = typer.Typer(
    help="Show or edit the user config file.",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    add_completion=False,
)


def _load_or_exit(path: Path | None = None) -> _config.Config:
    try:
        return _config.load(path)
    except _config.ConfigError as e:
        exit_error(str(e))


@config_app.command("path")
def _path() -> None:
    """Print the config file path (whether or not it exists)."""
    typer.echo(_config.config_path())


def _redact_secrets(cfg: _config.Config) -> _config.Config:
    """Return a copy of cfg with any ``*_api_key`` field masked.

    Self-maintaining via the field-name suffix — new credential fields
    added later are masked automatically.
    """
    p = cfg.provider
    masked = {
        f.name: ("<redacted>" if getattr(p, f.name) else "")
        for f in dataclasses.fields(p)
        if f.name.endswith("_api_key")
    }
    return dataclasses.replace(cfg, provider=dataclasses.replace(p, **masked))


@config_app.command("show")
def _show() -> None:
    """Dump the effective (merged) config as TOML. API keys are masked."""
    cfg = _load_or_exit()
    typer.echo(
        f"# API key values are masked. Read the file directly to see them: "
        f"{_config.config_path()}"
    )
    typer.echo()
    typer.echo(_config.to_toml(_redact_secrets(cfg)), nl=False)


@config_app.command("get")
def _get(
    dotted: Annotated[
        str,
        typer.Argument(help="Section.key, e.g. engine.max_turns."),
    ],
) -> None:
    """Read one dotted key from the effective config."""
    cfg = _load_or_exit()
    try:
        val = _config.get(cfg, dotted)
    except _config.ConfigError as e:
        exit_error(str(e))
    if isinstance(val, bool):
        typer.echo("true" if val else "false")
    else:
        typer.echo(val)


@config_app.command("set")
def _set(
    dotted: Annotated[
        str,
        typer.Argument(help="Section.field, e.g. engine.max_turns."),
    ],
    value: Annotated[
        str,
        typer.Argument(help="New value (parsed by type — int/float/bool/str)."),
    ],
) -> None:
    """Set one dotted key. Preserves comments. Note: secrets passed here
    land in shell history — use ``physiclaw config edit`` for API keys."""
    try:
        _config.set_dotted(dotted, value)
    except _config.ConfigError as e:
        exit_error(str(e))
    typer.echo(ok(f"{dotted} updated"))
    typer.echo("Restart `physiclaw server` to apply.")


@config_app.command("unset")
def _unset(
    dotted: Annotated[
        str,
        typer.Argument(help="Section.field to revert to its built-in default."),
    ],
) -> None:
    """Remove one dotted key from the file so the built-in default applies."""
    try:
        removed = _config.unset_dotted(dotted)
    except _config.ConfigError as e:
        exit_error(str(e))
    if removed:
        typer.echo(ok(f"{dotted} reverted to default"))
        typer.echo("Restart `physiclaw server` to apply.")
    else:
        typer.echo(f"  {dotted} was already at default (not present in file)")


@config_app.command("edit")
def _edit() -> None:
    """Open the config file in ``$EDITOR`` (creates it from the template if
    missing)."""
    path = _config.write_default()
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if not editor:
        for candidate in ("nano", "vim", "vi"):
            if shutil.which(candidate):
                editor = candidate
                break
    if not editor:
        typer.echo(
            f"no $EDITOR set and no nano/vim/vi on PATH. Edit by hand: {path}",
            err=True,
        )
        raise typer.Exit(code=1)
    subprocess.run([editor, str(path)], check=False)
    _load_or_exit(path)
    typer.echo(ok("config OK"))
    typer.echo("Restart `physiclaw server` to apply.")


__all__ = ["config_app"]
