"""`physiclaw studio` — drive the phone by hand from the browser."""

import atexit
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Annotated, Optional
from urllib.parse import urlparse

import typer

from physiclaw.cli._format import exit_error
from physiclaw.cli.server import terminate_child
from physiclaw.studio.session import Session


def studio(
    mcp_url: Annotated[
        Optional[str],
        typer.Option(
            "--mcp-url",
            help="MCP server to drive. Default: the live server, else the "
            "configured address, starting `physiclaw mcp -H` there when "
            "nothing is listening.",
        ),
    ] = None,
    port: Annotated[
        int, typer.Option("--port", help="Studio port (loopback only).")
    ] = 8058,
    open_browser: Annotated[
        bool, typer.Option("--open/--no-open", help="Open the browser.")
    ] = True,
    mock: Annotated[
        Optional[str],
        typer.Option(
            "--mock",
            help="Replay a recorded session instead of driving hardware "
            "(session id, a unique suffix, or a session directory). Views "
            "show the current frame; gestures step to the next.",
        ),
    ] = None,
) -> None:
    """Drive the phone by hand from the browser.

    Every gesture is one standard MCP tool call: draw a box on the
    camera frame or pick an element on the virtual screen, then tap,
    long-press, or swipe it. With no server running, the studio starts
    `physiclaw mcp -H` itself and stops it on exit; a server already
    running is driven as is.
    """
    import uvicorn

    from physiclaw.studio.server import build_app

    session: Session
    if mock is not None:
        mocked = _mock_session(mock)
        typer.echo(f"studio: mock of {mocked.label} ({len(mocked.frames)} frames)")
        session = mocked
    else:
        from physiclaw.studio.session import StudioSession

        session = StudioSession(mcp_url or _server_base())
        typer.echo(f"studio: hardware via {session.mcp_url}")
    url = f"http://127.0.0.1:{port}/"
    typer.echo(f"studio: {url}")
    if open_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    uvicorn.Server(
        uvicorn.Config(
            build_app(session), host="127.0.0.1", port=port, log_level="warning"
        )
    ).run()


def _mock_session(ref: str):
    """A `MockSession` over a session directory — a path, or an id
    resolved the `logs <suffix>` way."""
    from physiclaw.cli.conductor import resolve_sid
    from physiclaw.common import paths
    from physiclaw.studio.mock import MockSession, session_frames

    d = Path(ref).expanduser()
    if not d.is_dir():
        d = paths.engine_sessions_dir() / resolve_sid(ref)
    try:
        return MockSession(session_frames(d), d.name)
    except (FileNotFoundError, ValueError) as e:
        raise typer.BadParameter(str(e), param_hint="--mock") from e


# ---------- the server beside the studio ----------


def _server_base() -> str:
    """The server to drive, ready. The live server's own record wins
    (`runtime_state`, the pid-checked answer every CLI shares — a
    server on a non-default port is found, never doubled); otherwise
    the configured address, with `physiclaw mcp -H` spawned there when
    nothing is listening."""
    from physiclaw.common import runtime_state
    from physiclaw.common.config import server_url, url_host

    live = runtime_state.read_live()
    if live:
        base = f"http://{url_host(live['host'])}:{live['port']}"
        typer.echo(f"studio: MCP server already running at {base}")
        _wait_ready(base, None)
        return base
    base = server_url().rstrip("/")
    proc = None
    if not _listening(base):
        typer.echo("studio: no MCP server — starting physiclaw mcp -H")
        proc = _spawn_server(base)
        atexit.register(terminate_child, proc, 10)
    _wait_ready(base, proc)
    return base


def _listening(base: str) -> bool:
    """Whether anything answers at `base` — a refused or timed-out
    connect is the only "nothing there"; any HTTP answer, ready or
    not, is a server."""
    import httpx

    try:
        _ready(base)
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return False
    except Exception:
        pass
    return True


def _ready(base: str) -> bool:
    from physiclaw.common.ready import check_ready_once

    return check_ready_once(base, timeout=1.0)


def _spawn_server(base: str) -> subprocess.Popen:
    """`physiclaw mcp -H` at `base`, as a child on this interpreter —
    its logs share the terminal so hardware bring-up stays visible."""
    u = urlparse(base)
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "physiclaw.cli",
            "mcp",
            "-H",
            "--host",
            u.hostname or "127.0.0.1",
            "--port",
            str(u.port or 80),
        ]
    )


def _wait_ready(
    base: str,
    proc: Optional[subprocess.Popen],
    *,
    timeout: float = 120.0,
    poll: float = 0.5,
) -> bool:
    """Block until the server reports ready. A spawned child that exits
    first is fatal (its own log said why); a server that is still not
    ready at the deadline is reported, not fatal — the first action's
    banner will say what the rig is missing. The default deadline
    covers a warm start's sanity tap and a first-run wizard."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            exit_error(f"physiclaw mcp -H exited (code {proc.returncode})")
        try:
            if _ready(base):
                typer.echo("studio: MCP server ready")
                return True
        except Exception:
            pass
        time.sleep(poll)
    typer.echo(
        f"studio: MCP server not ready after {timeout:.0f}s — continuing; "
        "the first action will report what is still missing"
    )
    return False
