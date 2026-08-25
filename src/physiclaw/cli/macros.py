"""``physiclaw macros`` — author, rehearse, and track gesture macros.

The user's side of `agent.macros`: `init` scaffolds one, `list` shows
what's on disk, `check` lints every ``MACRO.yml`` (printing exactly why a
file is excluded), `runs` replays a past run's per-step log,
`run` rehearses one macro against the live server — the mandatory test
step before setting ``enabled: true``, and it takes ``--start-at`` so the
resumed path the agent can take is rehearsable too — and `stats` shows the run
counters the runner records. Rehearsals count in stats on purpose
(`runner.run_and_record` owns that fold, shared with the engine): a
successful rehearsal is evidence the path works and resets
``consecutive_aborts``.
"""

import asyncio
import json
from typing import Annotated

import typer

from physiclaw.agent.macros import runlog as macro_runlog
from physiclaw.agent.macros import runner as macro_runner
from physiclaw.agent.macros import stats as macro_stats
from physiclaw.agent.macros import store as macro_store
from physiclaw.agent.macros.model import Macro, MacroError
from physiclaw.cli._format import (
    exit_error,
    ok,
    parse_inputs,
    state_tag,
    step_fail,
    warn,
)
from physiclaw.common import paths, verdict
from physiclaw.common.config import CONFIG
from physiclaw.common.text import read_text

macros_app = typer.Typer(
    no_args_is_help=True,
    help="User-authored gesture macros (~/.physiclaw/macros/<name>/MACRO.yml).",
)

# Indent of a step's detail lines, under the step line's own two spaces.
_STEP_INDENT = " " * 6
# `args` is generous because a bbox row must never be the thing that gets
# clipped — a `send_to_clipboard` text is the only argument that runs long,
# and the one place a lost tail matters least. `screen` is the OCR haystack,
# already clipped to 500 on write by `runlog`; this is the reading budget.
_ARGS_CHARS = 240
_SCREEN_CHARS = 200


@macros_app.command()
def init(
    name: Annotated[str, typer.Argument(help="Macro name (lowercase/digits/hyphens).")],
) -> None:
    """Scaffold a new macro from a commented template."""
    try:
        path = macro_store.init_macro(name)
    except MacroError as e:
        exit_error(str(e))
    typer.echo(ok(str(path)))
    typer.echo("Next:")
    typer.echo("  1. edit it, then: physiclaw macros check")
    typer.echo(f"  2. rehearse:      physiclaw macros run {name} -i message=hi")
    typer.echo(
        "  3. once it succeeds, flip `enabled: false` to true (or delete the line)"
    )


@macros_app.command("list")
def list_cmd() -> None:
    """List every macro directory: enabled, disabled, or invalid."""
    macro_store.ensure_format_readme()
    entries = macro_store.scan()
    if not entries:
        typer.echo(
            "No macros found. Scaffold one: physiclaw macros init <name> "
            "(see ~/.physiclaw/macros/README.md)"
        )
        return
    for e in entries:
        tag = state_tag(
            valid=e.spec is not None, enabled=bool(e.spec and e.spec.enabled)
        )
        detail = (
            (e.error or "")
            if e.spec is None
            else f"{e.spec.description} ({len(e.spec.steps)} steps)"
        )
        typer.echo(f"  {tag} {e.dir_name}  {detail}")


@macros_app.command()
def check() -> None:
    """Validate every MACRO.yml; exit 1 if any is invalid."""
    macro_store.ensure_format_readme()
    entries = macro_store.scan()
    bad = [e for e in entries if e.spec is None]
    for e in bad:
        typer.echo(step_fail(f"{e.dir_name}: {e.error or ''}"))
    for e in entries:
        if e.spec is not None:
            typer.echo(
                ok(e.dir_name if e.spec.enabled else f"{e.dir_name}  (disabled)")
            )
    if not entries:
        typer.echo("No macros found.")
    _report_unreachable(entries)
    if bad:
        raise typer.Exit(1)


def _report_unreachable(entries: list[macro_store.ScanEntry]) -> None:
    """Valid is not the same as live. `check` answers "does it parse", and a
    disabled macro passes it while the agent never sees it — the one thing a
    green checkmark invites you to assume. Say which macros the agent cannot
    call, and why.

    The fleet-wide gate is reported instead of, not alongside, the per-file
    advice: it overrides every file's own flag, so telling someone to set
    `enabled: true` while it is off sends them to edit a file that changes
    nothing."""
    valid = [e.spec for e in entries if e.spec is not None]
    if not valid:
        return
    if not CONFIG.macros.enabled:
        typer.echo(
            warn(
                "[macros] enabled = false in config.toml, so the agent sees "
                "no macros at all, whatever each file says. Rehearsing with "
                "`physiclaw macros run` still works."
            )
        )
        return
    off = [s.name for s in valid if not s.enabled]
    if off:
        typer.echo(
            warn(
                f"disabled, so the agent cannot call: {', '.join(off)}. "
                "Set `enabled: true` in MACRO.yml once rehearsed."
            )
        )


@macros_app.command()
def run(
    name: Annotated[str, typer.Argument(help="Macro name (its directory name).")],
    inputs: Annotated[
        list[str],
        typer.Option(
            "--input",
            "-i",
            help="Input value as key=value; repeat per input.",
        ),
    ] = [],
    start_at: Annotated[
        str,
        typer.Option(
            "--start-at",
            help="Begin at this step NAME, as the agent's `start_at` does; "
            "the earlier steps are NOT executed.",
        ),
    ] = "",
) -> None:
    """Rehearse a macro against the running server (works while disabled —
    rehearse first, then set `enabled: true`)."""
    spec = _resolve_spec(macro_store.scan(), name)
    values = parse_inputs(inputs)
    try:
        result = asyncio.run(_run_live(spec, values, start_at))
    except MacroError as e:
        # Already recorded as bad_input by run_and_record.
        exit_error(str(e))
    except ConnectionError as e:
        # Connection-level failure (no server, refused) — nothing executed,
        # so no stats: this rehearses the environment, not the macro.
        # Narrow on purpose: `McpClient.__aenter__` raises exactly this and
        # names the URL. A blanket `except Exception` here told the user to
        # start a server that was already running, for any bug in the runner.
        #
        # `mcp`, not `server`: both serve the same MCP endpoint, but `server`
        # also spawns the agent runtime, which would wake on its own hooks
        # and drive the phone between the steps being rehearsed. A rehearsal
        # wants the rig to itself.
        exit_error(f"{e}. Start it first: physiclaw mcp")

    # The composed header + step log is always block 0 (runner contract).
    typer.echo(verdict.action_text(result.blocks))
    images = sum(1 for b in result.blocks if b.get("type") == "image")
    if images:
        typer.echo(f"({images} image(s) in the result — check the phone screen)")
    if result.run_id:
        typer.echo(
            f"run: {result.run_id} — inspect with: "
            f"physiclaw macros runs {result.run_id[-6:]}"
        )
    if not result.ok:
        raise typer.Exit(1)


@macros_app.command("runs")
def runs_cmd(
    run_id: Annotated[
        str,
        typer.Argument(
            help="A run id (`macro-run-<hex6>`, bare hex6, or a prefix); "
            "omit to list recent runs."
        ),
    ] = "",
    n: Annotated[int, typer.Option("-n", help="Runs to list.")] = 10,
) -> None:
    """List recent macro runs, or replay one run's per-step log."""
    if not run_id:
        runs = _collect_runs(limit=n)
        if not runs:
            typer.echo("No macro runs logged yet.")
            return
        for events in runs:
            typer.echo(_run_row(events))
        return
    # The id IS the directory name, so match on that and read only the hits
    # rather than parsing the whole retention window to find one run.
    matches = _collect_runs(tail=run_id.removeprefix(macro_runlog.RUN_ID_PREFIX))
    if not matches:
        exit_error(f"no run matching {run_id!r}")
    if len(matches) > 1:
        typer.echo(f"{run_id!r} is ambiguous — {len(matches)} runs match:")
    for events in matches[:3]:
        typer.echo(_run_row(events))
        typer.echo(f"  dir: {macro_runlog.run_dir(events[0]['run'])}")
        for e in events:
            if e["event"] == "step":
                mark = {"ok": "✓", "skipped": "↷"}.get(e["outcome"], "✗")
                line = (
                    f"  {mark} {e['i']}. {e['tool']}"
                    f"{' ' + repr(e['name']) if e.get('name') else ''}"
                    f" [{e['outcome']}] {e.get('ms', 0)}ms"
                )
                if e.get("verdict"):
                    line += f" — {e['verdict']}"
                if e.get("detail"):
                    line += f" — {e['detail']}"
                if e.get("image"):
                    line += f" — images/{e['image']}"
                typer.echo(line)
                # The two things you read after a failure: what the step
                # fired with (the bbox you edit when a tap lands wrong) and
                # what the camera saw. Both were already on disk; `args` was
                # merely never rendered. Own lines, not suffixes to skim.
                if e.get("args"):
                    typer.echo(_field("args", _fmt_args(e["args"]), _ARGS_CHARS))
                if e.get("screen_text"):
                    typer.echo(_field("screen", e["screen_text"], _SCREEN_CHARS))
            elif e["event"] == "end" and e.get("detail"):
                typer.echo(f"  end: {e.get('reason')} — {e['detail']}")


def _field(label: str, value: str, limit: int) -> str:
    """One labeled detail line under a step. `args` and `screen` are the
    same kind of thing — what this step did, and what it saw — so they are
    rendered by one function rather than two f-strings that drift.

    Continuation rows hang under the value: an element listing carries
    embedded newlines, and left as-is those rows restart at column 0, which
    stops the block reading as one field of the step above it."""
    pad = _STEP_INDENT + " " * (len(label) + 2)
    body = _clip(value, limit).replace("\n", "\n" + pad)
    return f"{_STEP_INDENT}{label}: {body}"


def _clip(text: str, limit: int) -> str:
    """Bound one rendered field, marking the cut. Truncating silently would
    read as a short value rather than a long one — the same reason
    `runlog` marks its own clips."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fmt_args(args: dict) -> str:
    """A step's recorded arguments, rendered to paste straight back into
    the `with:` table that produced them — `bbox: [0.12, 0.04, 0.34, 0.1]`,
    not a Python dict repr. JSON per VALUE (not over the whole mapping)
    keeps YAML flow syntax for lists and quotes strings, so whitespace in a
    clipboard value is visible instead of silently trailing. Values are
    post-substitution, so `{name}` placeholders show what actually fired."""
    return ", ".join(
        f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in args.items()
    )


def _collect_runs(*, limit: int | None = None, tail: str = "") -> list[list[dict]]:
    """Recent runs, newest first — each a list of that run's event lines,
    read from the per-run dirs (`macro-run-<hex6>/events.jsonl`). `tail`
    keeps only ids starting with that hex fragment and `limit` caps how many
    are read: both filter on the DIRECTORY NAME, so we never open, let alone
    JSON-parse, a run the caller is not going to show."""
    root = paths.macros_log_dir()
    if not root.exists():
        return []
    dirs = sorted(
        (
            d
            for d in root.iterdir()
            if d.name.startswith(macro_runlog.RUN_ID_PREFIX)
            and d.name[len(macro_runlog.RUN_ID_PREFIX) :].startswith(tail)
        ),
        key=_mtime,
        reverse=True,
    )
    out: list[list[dict]] = []
    for d in dirs if limit is None else dirs[:limit]:
        try:
            lines = read_text(d / "events.jsonl").splitlines()
        except OSError:
            continue
        events = []
        for raw in lines:
            try:
                events.append(json.loads(raw))
            except ValueError:
                continue
        if events:
            out.append(events)
    return out


def _mtime(d) -> float:
    """Sort key that survives a dir vanishing mid-listing (retention purge)."""
    try:
        return d.stat().st_mtime
    except OSError:
        return 0.0


def _run_row(events: list[dict]) -> str:
    first, last = events[0], events[-1]
    end = last if last.get("event") == "end" else {}
    state = ok("ok") if end.get("ok") else step_fail(end.get("reason", "incomplete"))
    steps = sum(1 for e in events if e.get("event") == "step")
    # A start_at run replayed only part of the macro — without this the row
    # reads like a full run that happens to have fewer steps.
    resumed = f"  from {first['start_at']!r}" if first.get("start_at") else ""
    return (
        f"{first['run']}  {first.get('ts', '')[:16]}  {first['macro']}  "
        f"{first.get('caller', '?')}  {state}  {steps} step(s){resumed}"
    )


@macros_app.command("stats")
def stats_cmd() -> None:
    """Show run counters per macro (recorded by agent runs and rehearsals)."""
    data = macro_stats.load()
    if not data:
        typer.echo("No macro runs recorded yet.")
        return
    for name in sorted(data):
        s = data[name]
        streak = s.get("consecutive_aborts", 0)
        line = (
            f"  {name}: {s.get('total_successes', 0)}/{s.get('total_runs', 0)} ok, "
            f"{s.get('total_aborts', 0)} abort(s), streak {streak}"
        )
        color = typer.colors.RED if streak else None
        typer.echo(typer.style(line, fg=color) if color else line)
        last = s.get("last_abort")
        if last:
            run_ref = f" — {last['run']}" if last.get("run") else ""
            typer.echo(
                f"      last abort: step {last.get('step')} "
                f"[{last.get('reason')}] {last.get('detail')} "
                f"({last.get('ts')}){run_ref}"
            )


async def _run_live(
    spec: Macro, values: dict[str, str], start_at: str = ""
) -> macro_runner.MacroRunResult:
    # Local import: the mcp SDK only loads when actually rehearsing.
    from physiclaw.agent.engine.mcp_tool import McpClient

    async with McpClient() as mcp:
        return await macro_runner.run_and_record(
            spec, values, mcp, caller="cli", start_at=start_at
        )


def _resolve_spec(entries: list[macro_store.ScanEntry], name: str) -> Macro:
    for e in entries:
        if e.dir_name == name:
            if e.spec is None:
                exit_error(f"{name}: {e.error}")
            return e.spec
    exit_error(f"no macro named {name!r}")
