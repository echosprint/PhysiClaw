"""`physiclaw studio` — the browser skin over the same drivers the CLI
wraps: drive the phone by hand, and step a playbook node by node.

A frontend only. It drives a running `physiclaw mcp` (the live
server's own record wins, else the configured address) and never
starts one — the same "start it first" rule every rehearsal command
follows, so one server is shared by whoever is at the desk and
whatever agent is at the terminal. `--session` puts a recorded
session's frames behind the same page instead of hardware.
"""

import threading
import webbrowser
from pathlib import Path
from typing import Annotated, Optional

import typer

from physiclaw.cli._format import exit_error
from physiclaw.common.ready import START_HINT
from physiclaw.studio.session import Session


def studio(
    mcp_url: Annotated[
        Optional[str],
        typer.Option(
            "--mcp-url",
            help="MCP server to drive. Default: the live server, else the "
            "configured address.",
        ),
    ] = None,
    port: Annotated[
        int, typer.Option("--port", help="Studio port (loopback only).")
    ] = 8058,
    open_browser: Annotated[
        bool, typer.Option("--open/--no-open", help="Open the browser.")
    ] = True,
    session: Annotated[
        Optional[str],
        typer.Option(
            "--session",
            help="Drive a recorded session's frames instead of hardware "
            "(session id, a unique suffix, or a session directory). Views "
            "show the current frame; gestures step to the next.",
        ),
    ] = None,
) -> None:
    """Drive the phone by hand from the browser, and step playbooks.

    Every gesture is one standard MCP tool call: draw a box on the
    camera frame or pick an element on the virtual screen, then tap,
    long-press, or swipe it. The playbook panel steps a pack one node
    at a time — the same core as `physiclaw playbooks step`, the same
    position file, so a terminal and the page can take turns. A
    frontend only: it needs `physiclaw mcp` already running.
    """
    import uvicorn

    from physiclaw.studio.server import build_app

    driven: Session
    if session is not None:
        recorded = _recorded_session(session)
        typer.echo(f"studio: recorded {recorded.label} ({len(recorded.frames)} frames)")
        driven = recorded
    else:
        from physiclaw.studio.session import StudioSession

        driven = StudioSession(mcp_url or _server_base())
        typer.echo(f"studio: hardware via {driven.mcp_url}")
    url = f"http://127.0.0.1:{port}/"
    typer.echo(f"studio: {url}")
    if open_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    uvicorn.Server(
        uvicorn.Config(
            build_app(driven), host="127.0.0.1", port=port, log_level="warning"
        )
    ).run()


def _recorded_session(ref: str):
    """A `MockSession` over a session directory — a path, or an id
    resolved the `logs <suffix>` way."""
    from physiclaw.cli._sessions import resolve_sid
    from physiclaw.common import paths
    from physiclaw.studio.mock import MockSession, session_frames

    d = Path(ref).expanduser()
    if not d.is_dir():
        d = paths.engine_sessions_dir() / resolve_sid(ref)
    try:
        return MockSession(session_frames(d), d.name)
    except (FileNotFoundError, ValueError) as e:
        raise typer.BadParameter(str(e), param_hint="--session") from e


def _server_base() -> str:
    """The server to drive. The live server's own record wins
    (`runtime_state`, the pid-checked answer every CLI shares — a
    server on a non-default port is found, never doubled); otherwise
    the configured address, which must already answer."""
    from physiclaw.common import runtime_state
    from physiclaw.common.config import server_url, url_host

    live = runtime_state.read_live()
    if live:
        base = f"http://{url_host(live['host'])}:{live['port']}"
        typer.echo(f"studio: MCP server running at {base}")
        return base
    base = server_url().rstrip("/")
    if not _listening(base):
        exit_error(f"no MCP server at {base}. {START_HINT}")
    return base


def _listening(base: str) -> bool:
    """Whether anything answers at `base` — a refused or timed-out
    connect is the only "nothing there"; any HTTP answer, ready or
    not, is a server (the first action's banner reports the rest)."""
    import httpx

    from physiclaw.common.ready import check_ready_once

    try:
        check_ready_once(base, timeout=1.0)
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return False
    except Exception:
        pass
    return True
