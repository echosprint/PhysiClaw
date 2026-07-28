"""Tests for `physiclaw.cli.mcp_cmd` — the MCP-only alias command.

`mcp` is a thin alias: it starts the server with `no_runtime=True` and the
server's own tested branches drive the rest. We assert the delegation and
the flag passthrough here.
"""

from __future__ import annotations

import importlib

mcp_mod = importlib.import_module("physiclaw.cli.mcp_cmd")


def test_mcp_invokes_server_with_no_runtime(mocker) -> None:
    server_spy = mocker.patch.object(mcp_mod, "server")

    mcp_mod.mcp(
        port=9000,
        host="127.0.0.1",
        verbose=True,
        warm_start=False,
        hot_start=True,
        cam_index=2,
        no_setup_hardware=False,
    )

    server_spy.assert_called_once_with(
        port=9000,
        host="127.0.0.1",
        verbose=True,
        no_runtime=True,
        warm_start=False,
        hot_start=True,
        cam_index=2,
        no_setup_hardware=False,
    )


def test_mcp_defaults_pin_no_runtime_only(mocker) -> None:
    # The alias pins the runtime axis; the bring-up axis stays at the
    # server's own defaults (wizard on first run).
    server_spy = mocker.patch.object(mcp_mod, "server")

    mcp_mod.mcp()

    kwargs = server_spy.call_args.kwargs
    assert kwargs["no_runtime"] is True
    assert kwargs["warm_start"] is False
    assert kwargs["hot_start"] is False
    assert kwargs["cam_index"] is None
    assert kwargs["no_setup_hardware"] is False
