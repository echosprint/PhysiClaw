"""Tests for `physiclaw.cli.now` — the hot-start alias command.

`now` is a thin alias: it starts the server with `hot_start=True` and the
server's own tested branches drive the rest. We assert the delegation and
the flag passthrough here.
"""

from __future__ import annotations

import importlib

now_mod = importlib.import_module("physiclaw.cli.now")


def test_now_invokes_server_with_hot_start(mocker) -> None:
    server_spy = mocker.patch.object(now_mod, "server")

    now_mod.now(port=9000, host="127.0.0.1", verbose=True, cam_index=2)

    server_spy.assert_called_once_with(
        port=9000,
        host="127.0.0.1",
        verbose=True,
        hot_start=True,
        cam_index=2,
    )


def test_now_defaults_leave_cam_index_to_the_bundle(mocker) -> None:
    server_spy = mocker.patch.object(now_mod, "server")

    now_mod.now()

    kwargs = server_spy.call_args.kwargs
    assert kwargs["hot_start"] is True
    assert kwargs["cam_index"] is None
