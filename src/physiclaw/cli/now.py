"""``physiclaw now`` — hot-start the server, trusting the saved calibration.

The subcommand spelling of ``physiclaw server --hot-start``: resume from
the saved calibration bundle without touching the phone — no setup
wizard, no bridge wait, no sanity tap. Use only when nothing moved since
the last clean shutdown.
"""

from typing import Annotated, Optional

import typer

from physiclaw.cli.server import server
from physiclaw.common.config import CONFIG


def now(
    port: Annotated[
        int, typer.Option("--port", help="MCP server port.")
    ] = CONFIG.server.port,
    host: Annotated[
        str, typer.Option("--host", help="Bind address.")
    ] = CONFIG.server.host,
    verbose: Annotated[
        bool,
        typer.Option("-v", "--verbose", help="Show detailed debug output."),
    ] = False,
    cam_index: Annotated[
        Optional[int],
        typer.Option(
            "--cam-index",
            help="Camera index override (default: value stored in the bundle).",
        ),
    ] = None,
) -> None:
    """Start the server immediately, skipping setup and verification.

    Trusts the saved calibration and the parked arm — same as
    ``physiclaw server --hot-start``.
    """
    server(
        port=port,
        host=host,
        verbose=verbose,
        hot_start=True,
        cam_index=cam_index,
    )
