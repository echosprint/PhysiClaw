"""``physiclaw mcp`` — run the MCP server for an external agent.

The subcommand spelling of ``physiclaw server --no-runtime``: the full
two-plane server (MCP + setup + phone bridge) without the built-in agent
runtime subprocess, so an external MCP client — Claude Code, Claude
Desktop, any MCP-speaking agent — drives the phone instead. Hardware
bring-up is unchanged: first run opens the wizard; on a calibrated rig
``-H`` trusts the parked arm and is ready instantly.

The module is named ``mcp_cmd`` (not ``mcp``) so it can't shadow the
``mcp`` SDK package in imports; the command name is set at registration.
"""

from typing import Annotated, Optional

import typer

from physiclaw.cli.server import server
from physiclaw.common.config import CONFIG


def mcp(
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
    warm_start: Annotated[
        bool,
        typer.Option(
            "--warm-start",
            help="Auto-connect hardware from the saved calibration bundle "
            "and verify with a sanity tap, skipping `setup hardware`.",
        ),
    ] = False,
    hot_start: Annotated[
        bool,
        typer.Option(
            "-H",
            "--hot-start",
            help="Like --warm-start, but trust that the arm still sits where "
            "it parked — ready as soon as hardware reconnects. Use only "
            "when nothing moved since the last run.",
        ),
    ] = False,
    cam_index: Annotated[
        Optional[int],
        typer.Option(
            "--cam-index",
            help="Camera index override for --warm-start / --hot-start "
            "(default: value stored in the bundle).",
        ),
    ] = None,
    no_setup_hardware: Annotated[
        bool,
        typer.Option(
            "--no-setup-hardware",
            help="Don't auto-open the browser hardware-setup wizard on start.",
        ),
    ] = False,
) -> None:
    """Run the MCP server only — your own agent drives, no built-in runtime.

    Connect an external MCP client (Claude Code, Claude Desktop, …) to the
    logged /mcp URL. Same as ``physiclaw server --no-runtime``; like every
    server mode it owns the arm, so only one can run per machine.
    """
    server(
        port=port,
        host=host,
        verbose=verbose,
        no_runtime=True,
        warm_start=warm_start,
        hot_start=hot_start,
        cam_index=cam_index,
        no_setup_hardware=no_setup_hardware,
    )
