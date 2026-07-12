"""``physiclaw auto`` — start the server and calibrate hands-free.

Runs the MCP server with the desktop hardware-setup wizard disabled. As
soon as the phone opens ``/bridge``, calibration runs unattended — the same
flow as ``physiclaw setup hardware --auto``, but triggered by the phone
connecting rather than an operator at the terminal.
"""

from typing import Annotated

import typer

from physiclaw.cli.server import server
from physiclaw.common.config import CONFIG


def auto(
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
) -> None:
    """Start the server; calibrate automatically when the phone bridge opens.

    Headless setup with no desktop wizard — seat the rig, open /bridge on the
    phone, and calibration proceeds on its own.
    """
    # Confirm the phone is up BEFORE starting the server, so the reminder
    # isn't buried under the startup logs. The bridge page reconnects on its
    # own, so opening it now and letting it connect once the server is up is
    # fine. Both URLs from the same source as the banner (mDNS + IP fallback).
    from physiclaw.core.bridge import bridge_base_urls

    primary, fallback = bridge_base_urls(port)
    url = f"{primary}/bridge"
    if primary != fallback:
        url += f"  (or {fallback}/bridge)"
    print(f"\n  Open on the phone: {url}")
    try:
        ans = input("  Ready? [y/n] ")
    except EOFError:
        ans = ""  # non-interactive stdin (e.g. a service) — proceed
    if ans.strip().lower() not in ("", "y", "yes"):
        raise typer.Exit()

    server(port=port, host=host, verbose=verbose, auto_calibrate=True)
