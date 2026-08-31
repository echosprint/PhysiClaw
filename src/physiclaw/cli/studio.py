"""`physiclaw studio` — the browser authoring tool."""

import threading
import webbrowser
from typing import Annotated, Optional

import typer


def studio(
    mcp_url: Annotated[
        Optional[str],
        typer.Option(
            "--mcp-url",
            help="MCP server to drive (default: the configured server URL). "
            "Author against `physiclaw mcp -H` so the agent runtime never "
            "contends for the arm.",
        ),
    ] = None,
    port: Annotated[
        int, typer.Option("--port", help="Studio port (loopback only).")
    ] = 8058,
    app: Annotated[
        Optional[str],
        typer.Option("--app", help="App pack to author (e.g. taobao)."),
    ] = None,
    open_browser: Annotated[
        bool, typer.Option("--open/--no-open", help="Open the browser.")
    ] = True,
) -> None:
    """Drive the phone from the browser and author app packs.

    An independent process: hardware only through the standard MCP
    surface of a running server, pack files through the local
    filesystem. Start the server first: physiclaw mcp -H
    """
    import uvicorn

    from physiclaw.studio.server import build_app
    from physiclaw.studio.session import StudioSession

    session = StudioSession(mcp_url=mcp_url, app=app)
    url = f"http://127.0.0.1:{port}/"
    typer.echo(f"studio: {url} (hardware via {session.mcp_url})")
    if open_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    uvicorn.Server(
        uvicorn.Config(
            build_app(session), host="127.0.0.1", port=port, log_level="warning"
        )
    ).run()
