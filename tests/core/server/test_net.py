"""Tests for `physiclaw.core.server.net.wait_for_port` — socket-probe with a
loopback-rewrite for wildcard hosts. All socket/time interaction is mocked;
no real ports are opened.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from physiclaw.core.server import net
from physiclaw.core.server.net import wait_for_port


def _fake_socket(mocker, *, connect_error=None) -> MagicMock:
    fake_sock = MagicMock()
    fake_sock.__enter__ = MagicMock(return_value=fake_sock)
    fake_sock.__exit__ = MagicMock(return_value=None)
    if connect_error is not None:
        fake_sock.connect.side_effect = connect_error
    else:
        fake_sock.connect.return_value = None
    mocker.patch.object(net.socket, "socket", return_value=fake_sock)
    return fake_sock


def test_wait_for_port_returns_true_on_immediate_connect(mocker) -> None:
    _fake_socket(mocker)

    assert wait_for_port("127.0.0.1", 8048, timeout=1.0) is True


def test_wait_for_port_returns_false_on_timeout(mocker) -> None:
    _fake_socket(mocker, connect_error=OSError("connection refused"))
    mocker.patch.object(net.time, "sleep")
    # Force the deadline immediately by faking monotonic.
    times = iter([0.0, 100.0])  # deadline=1.0; first check passes, second exits
    mocker.patch.object(net.time, "monotonic", side_effect=lambda: next(times))

    assert wait_for_port("127.0.0.1", 8048, timeout=1.0) is False


def test_wait_for_port_uses_loopback_when_host_is_wildcard(mocker) -> None:
    fake_sock = _fake_socket(mocker)

    wait_for_port("0.0.0.0", 8048, timeout=1.0)

    fake_sock.connect.assert_called_once_with(("127.0.0.1", 8048))


def test_wait_for_port_uses_loopback_when_host_is_empty(mocker) -> None:
    fake_sock = _fake_socket(mocker)

    wait_for_port("", 8048, timeout=1.0)

    fake_sock.connect.assert_called_once_with(("127.0.0.1", 8048))


def test_wait_for_port_uses_named_host_unchanged(mocker) -> None:
    fake_sock = _fake_socket(mocker)

    wait_for_port("api.host", 8048, timeout=1.0)

    fake_sock.connect.assert_called_once_with(("api.host", 8048))
