"""`physiclaw debug` — the one-command e2e debug runner.

    physiclaw debug --task "buy two boxes of milk" \
        --reply "ok, but make it three boxes" --reply "ok"

One line does the whole run: seed the virtual thread with the user's
message, stage the replies, arm the debug wake, and — when no server is
live — start one right here in debug mode (hot start, this terminal).
Seeding happens BEFORE the server exists, so the task is always on the
thread whichever hook fires first. With a server already running, the
same command just seeds and wakes it (that server must itself have been
started by `physiclaw debug`, or the files are ignored whole — debug
mode is env-only and per-run, never persisted).

`--reply` alone appends to a running script; `--status` shows the
harness state; `--clear` resets it. `--macro-failure` defaults ON in
debug mode — the first macro abort halts the server for inspection;
pass `--no-macro-failure` to drive on past aborts instead.

Each `--reply` is released at the right moment — after the agent's next
ask lands, on the peek that polls for it (`debug.thread` owns the
rule). Every action runs for real — the gate's ask is genuinely sent
into the IM thread; only the conductor's channel observations are
rewritten to the script — so park the phone OFF the real IM app before
a run (a leftover real thread on screen would be read for real at the
boot peek), and point the channel pack at a thread you don't mind
receiving the asks.
"""

import os
from typing import Optional

import typer

from physiclaw.agent.hooks.debug import wake_path
from physiclaw.cli._format import exit_error
from physiclaw.common.config import DEBUG_ENV_VAR, MACRO_FAILURE_ENV_VAR
from physiclaw.common.logger import write_json_atomic


def debug(
    task: Optional[str] = typer.Option(
        None,
        "--task",
        help="Start a run: the user's message, e.g. 'buy two boxes of milk'.",
    ),
    reply: Optional[list[str]] = typer.Option(
        None,
        "--reply",
        help="Stage a reply (repeatable) — released after the agent's next ask.",
    ),
    wake: bool = typer.Option(
        True,
        "--wake/--no-wake",
        help="Arm a debug wake (default with --task; a bare --reply "
        "wakes only a suspended walk).",
    ),
    macro_failure: bool = typer.Option(
        True,
        "--macro-failure/--no-macro-failure",
        help="Halt the server at the first macro abort for inspection — "
        "the debug default. --no-macro-failure drives on past aborts. "
        "Applies when this command starts the server.",
    ),
    status: bool = typer.Option(False, "--status", help="Show the harness state."),
    clear: bool = typer.Option(False, "--clear", help="Reset thread and wake."),
) -> None:
    """Run the e2e harness: seed the task, stage the replies, wake the
    agent — starting the server in debug mode if none is running. No
    human on the other phone."""
    from physiclaw.debug import thread as vthread

    if clear:
        _clear()
        return
    if status:
        _status()
        return
    if task is None and not reply:
        exit_error("nothing to do — pass --task (and --reply), or --status/--clear")
    replies = list(reply or [])
    if task is not None:
        vthread.seed(task, replies)
        typer.echo(f"thread reset: user: {task!r}, {len(replies)} staged reply(ies)")
        if wake:
            _arm_wake(f"debug: user sent a message: {task!r}")
        _run_server_if_none(macro_failure)
    else:
        from physiclaw.conductor.walk.suspension import suspended_ref

        vthread.stage(replies)
        typer.echo(f"staged {len(replies)} reply(ies) onto the running script")
        # A reply for a suspended walk needs the wake that resumes it;
        # in-session staging (gate still polling) needs none.
        if wake and suspended_ref() is not None:
            _arm_wake("debug: user replied")


def _run_server_if_none(macro_failure: bool) -> None:
    """The one-command experience: no live server → become one, in debug
    mode (hot start, foreground). The env flags are set in THIS process
    — the server — so the runtime subprocess inherits them; they die
    with it (debug mode is per-run by design). A live server instead
    gets the seeded files and the hint."""
    from physiclaw.common import runtime_state

    if runtime_state.read_live() is not None:
        typer.echo(
            "note: a server is already running — the script takes effect "
            "only if it was started by `physiclaw debug`."
        )
        return
    os.environ[DEBUG_ENV_VAR] = "1"
    if macro_failure:
        os.environ[MACRO_FAILURE_ENV_VAR] = "1"
    typer.echo(
        "no live server — starting one in debug mode (hot start"
        + (", halt on macro failure" if macro_failure else "")
        + ")…"
    )
    from physiclaw.cli.server import server

    server(hot_start=True)


def _arm_wake(description: str) -> None:
    wake_path().parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(wake_path(), {"description": description})
    typer.echo("wake armed — the runtime picks it up once the server is ready.")


def _status() -> None:
    from physiclaw.conductor.walk.suspension import suspended_ref
    from physiclaw.debug import thread as vthread

    thread = vthread.load()
    for b in thread.bubbles:
        typer.echo(f"  {b.sender:>5}: {b.text}")
    typer.echo(f"staged: {thread.staged}")
    typer.echo(f"wake armed: {wake_path().exists()}")
    ref = suspended_ref()
    typer.echo(f"suspended walk: {f'{ref[0]}/{ref[1]}' if ref else False}")


def _clear() -> None:
    from physiclaw.debug import thread as vthread

    removed = []
    for p in (vthread.thread_path(), wake_path()):
        if p.exists():
            p.unlink()
            removed.append(p.name)
    typer.echo(f"cleared: {', '.join(removed) or '(nothing to clear)'}")
