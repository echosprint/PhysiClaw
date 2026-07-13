"""``physiclaw logs`` — list and inspect engine session logs.

The analysis entry point for `~/.physiclaw/log/engine/sessions/`:
`physiclaw logs` tables the recent sessions from their summary.json;
`physiclaw logs <sid>` prints one session's summary plus the tail of its
narrative (re-rendered from events.jsonl — the daily log interleaves
sessions, so the per-session stream is the clean source). `--json` emits
machine-readable output for scripting; `--save [DEST]` zips a session
(with its format README) for backups or bug reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from physiclaw.cli._format import info, section, warn


def logs(
    sid: Annotated[
        str | None,
        typer.Argument(
            help="Session id, or just its 6-hex-digit suffix; omit to list recent sessions.",
        ),
    ] = None,
    n: Annotated[
        int,
        typer.Option("-n", help="Sessions to list / narrative lines to show."),
    ] = 20,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Machine-readable output."),
    ] = False,
    save: Annotated[
        bool,
        typer.Option(
            "--save",
            help="Save the session as a single zip (for backups or bug reports).",
        ),
    ] = False,
    dest: Annotated[
        Path | None,
        typer.Argument(
            help="With --save: destination directory or .zip path (default: current dir).",
        ),
    ] = None,
) -> None:
    """List recent agent sessions, or inspect one session's log artifacts."""
    from physiclaw.common import paths

    sessions_dir = paths.engine_sessions_dir()
    if sid is None:
        if save:
            typer.echo(warn("--save needs a session: physiclaw logs <sid> --save"))
            raise typer.Exit(1)
        _list_sessions(sessions_dir, n=n, as_json=as_json)
    elif save:
        _save_session(_resolve(sessions_dir, sid), dest)
    elif dest is not None:
        # A destination without --save is a forgotten flag, not a request
        # for the detail view — refuse rather than silently ignore it.
        typer.echo(
            warn(f"a destination needs --save: physiclaw logs {sid} --save {dest}")
        )
        raise typer.Exit(1)
    else:
        _show_session(_resolve(sessions_dir, sid), n=n, as_json=as_json)


def _resolve(sessions_dir: Path, query: str) -> Path:
    """Resolve a session by full id, or by any unique trailing fragment —
    the sid's 6-hex-digit random suffix is the intended short handle. Exits
    with the candidates when the fragment is ambiguous."""
    exact = sessions_dir / query
    if exact.is_dir():
        return exact
    try:
        matches = sorted(
            d for d in sessions_dir.iterdir() if d.is_dir() and d.name.endswith(query)
        )
    except OSError:
        matches = []
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        typer.echo(warn(f"'{query}' is ambiguous — matches:"))
        for m in matches:
            typer.echo(info(m.name))
        raise typer.Exit(1)
    return exact  # no match — _show_session reports it


# ---------- list mode ----------


def _load_summary(d: Path) -> dict[str, Any] | None:
    try:
        return json.loads((d / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _stub_summary(d: Path) -> dict[str, Any]:
    """Row for a session dir without a summary.json (hard-killed session
    or one still running) — visible rather than silently missing."""
    return {
        "sid": d.name,
        "outcome": {"sentinel": "?", "recap": "(no summary — killed or still running)"},
    }


def _collect(sessions_dir: Path, n: int) -> list[dict[str, Any]]:
    try:
        dirs = sorted(
            (d for d in sessions_dir.iterdir() if d.is_dir()),
            key=lambda d: d.name,
            reverse=True,
        )
    except OSError:
        dirs = []
    return [(_load_summary(d) or _stub_summary(d)) for d in dirs[:n]]


def _list_sessions(sessions_dir: Path, *, n: int, as_json: bool) -> None:
    summaries = _collect(sessions_dir, n)
    if as_json:
        typer.echo(json.dumps(summaries, ensure_ascii=False, indent=2))
        return
    if not summaries:
        typer.echo(
            info(
                "no sessions yet — logs appear under "
                f"{sessions_dir} after the first agent wake"
            )
        )
        return
    typer.echo(section(f"Sessions ({len(summaries)} most recent)"))
    typer.echo(
        f"  {'SID':<22} {'OUTCOME':<7} {'TURNS':>5} {'DUR':>6} "
        f"{'TOKENS':>12} {'CACHE':>5}  RECAP"
    )
    for s in summaries:
        typer.echo("  " + _row(s))


def _row(s: dict[str, Any]) -> str:
    from physiclaw.agent.engine.trace import brief, fmt_tokens

    outcome = s.get("outcome") or {}
    sentinel = outcome.get("sentinel") or "?"
    if outcome.get("crashed"):
        sentinel = "CRASH"
    u = s.get("usage") or {}
    tokens = (
        f"{fmt_tokens(u['input_tokens'])}/{fmt_tokens(u['output_tokens'])}"
        if u
        else "-"
    )
    cache = f"{u['cache_hit_pct']:.0f}%" if u else "-"
    dur = f"{s['duration_s']:.0f}s" if "duration_s" in s else "-"
    turns = s.get("turns", "-")
    recap = brief(outcome.get("recap") or "", 48)
    return (
        f"{s.get('sid', '?'):<22} {sentinel:<7} {turns:>5} {dur:>6} "
        f"{tokens:>12} {cache:>5}  {recap}"
    )


# ---------- save mode ----------


def _save_session(d: Path, dest: Path | None) -> None:
    """Zip the session dir to `dest` (dir or .zip path; default cwd).
    The zip stays local — whether it leaves the machine is the user's
    call."""
    import zipfile

    if not d.is_dir():
        typer.echo(warn(f"no such session: {d.name} (looked in {d.parent})"))
        raise typer.Exit(1)

    default_name = f"physiclaw-session-{d.name}.zip"
    out = (dest or Path.cwd()).expanduser()
    if out.is_dir() or not out.suffix:
        out = out / default_name
    out.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in d.rglob("*") if p.is_file())
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            # Forward slashes explicitly — zip arcnames must not carry
            # Windows backslashes or extractors mis-nest the tree.
            z.write(p, arcname=f"{d.name}/{p.relative_to(d).as_posix()}")
        # Ship the format doc with the data so any analyst — human or
        # AI agent — can bootstrap from the zip alone.
        from physiclaw.agent.engine.trace import SESSIONS_README

        z.writestr(f"{d.name}/README.md", SESSIONS_README)
    images = sum(1 for p in files if p.parent.name == "images")
    size_kb = out.stat().st_size / 1024

    from physiclaw.cli._format import ok

    typer.echo(
        ok(
            f"saved {len(files)} file(s) ({images} screenshots, "
            f"{size_kb:.0f} KB) → {out}"
        )
    )
    typer.echo(
        warn("PRIVATE: phone screenshots + full prompts inside — review before sharing")
    )


# ---------- detail mode ----------


def _show_session(d: Path, *, n: int, as_json: bool) -> None:
    if not d.is_dir():
        typer.echo(warn(f"no such session: {d.name} (looked in {d.parent})"))
        raise typer.Exit(1)
    summary = _load_summary(d)
    if as_json:
        typer.echo(
            json.dumps(summary or _stub_summary(d), ensure_ascii=False, indent=2)
        )
        return
    if summary is not None:
        typer.echo(section(f"Session {d.name}"))
        typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        typer.echo(warn("no summary.json — session was killed or is still running"))
    _echo_narrative(d / "events.jsonl", n)
    typer.echo("")
    typer.echo(info(f"wire log:  {d / 'wire.jsonl'}"))
    typer.echo(info(f"images:    {d / 'images'}"))
    typer.echo(info(f"save a copy: physiclaw logs {d.name.rsplit('_', 1)[-1]} --save"))


def _echo_narrative(events_path: Path, n: int) -> None:
    """Re-render the last `n` events with the daily log's formatter — the
    per-session equivalent of the day file's narrative, without the other
    sessions interleaved."""
    from physiclaw.agent.engine.trace import summarize_event

    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        typer.echo(warn("no events.jsonl"))
        return
    typer.echo("")
    typer.echo(section(f"Last {min(n, len(lines))} of {len(lines)} events"))
    for line in lines[-n:]:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        stamp = str(event.pop("t", ""))[11:19]  # popped: the line carries it,
        msg = summarize_event(event)  # and fallback repr would echo it
        if msg is not None:
            typer.echo(f"  [{stamp}] {msg}")
