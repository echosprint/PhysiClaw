"""Tests for `physiclaw.core.bridge.lan` — LAN address helpers.

Network calls (`socket.socket`, `socket.gethostbyname`) and the platform
hostname helper are mocked so tests don't hit the real network or shell.
"""

from __future__ import annotations

import socket
from unittest.mock import MagicMock

from physiclaw.core.bridge import lan


# ---------- get_lan_ip ----------


def test_get_lan_ip_returns_local_ip_on_success(mocker) -> None:
    fake_sock = MagicMock()
    fake_sock.getsockname.return_value = ("192.168.1.42", 12345)
    mocker.patch.object(lan.socket, "socket", return_value=fake_sock)

    assert lan.get_lan_ip() == "192.168.1.42"


def test_get_lan_ip_falls_back_to_loopback_on_socket_error(mocker) -> None:
    fake_sock = MagicMock()
    fake_sock.connect.side_effect = OSError("network unreachable")
    mocker.patch.object(lan.socket, "socket", return_value=fake_sock)

    assert lan.get_lan_ip() == "127.0.0.1"


def test_get_lan_ip_falls_back_to_loopback_on_timeout(mocker) -> None:
    fake_sock = MagicMock()
    fake_sock.connect.side_effect = socket.timeout()
    mocker.patch.object(lan.socket, "socket", return_value=fake_sock)

    assert lan.get_lan_ip() == "127.0.0.1"


# ---------- get_mdns_host ----------


def test_get_mdns_host_returns_lowercased_local_when_platform_provides_name(
    mocker,
) -> None:
    mocker.patch.object(lan.platform, "local_hostname", return_value="My-Mac")
    mocker.patch.object(socket, "gethostbyname", return_value="192.168.1.5")

    assert lan.get_mdns_host() == "my-mac.local"


def test_get_mdns_host_returns_none_when_platform_returns_none(mocker) -> None:
    mocker.patch.object(lan.platform, "local_hostname", return_value=None)

    assert lan.get_mdns_host() is None


def test_get_mdns_host_returns_none_when_resolution_fails(mocker) -> None:
    mocker.patch.object(lan.platform, "local_hostname", return_value="my-mac")
    mocker.patch.object(socket, "gethostbyname", side_effect=socket.gaierror)

    assert lan.get_mdns_host() is None


def test_get_mdns_host_returns_none_when_resolution_times_out(mocker) -> None:
    mocker.patch.object(lan.platform, "local_hostname", return_value="my-mac")
    mocker.patch.object(socket, "gethostbyname", side_effect=socket.timeout)

    assert lan.get_mdns_host() is None


def test_get_mdns_host_restores_default_socket_timeout(mocker) -> None:
    mocker.patch.object(lan.platform, "local_hostname", return_value="my-mac")
    mocker.patch.object(socket, "gethostbyname", return_value="10.0.0.1")
    prev = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(7.5)
        lan.get_mdns_host()
        assert socket.getdefaulttimeout() == 7.5
    finally:
        socket.setdefaulttimeout(prev)


# ---------- bridge_base_urls ----------
# Input is the CONTROL port; the URLs point at the LAN bridge listener
# one port up (`bridge_port`).


def test_bridge_port_is_control_port_plus_one() -> None:
    assert lan.bridge_port(8048) == 8049


def test_bridge_base_urls_uses_mdns_when_available(mocker) -> None:
    mocker.patch.object(lan, "get_lan_ip", return_value="192.168.1.10")
    mocker.patch.object(lan, "get_mdns_host", return_value="mac.local")

    primary, fallback = lan.bridge_base_urls(8048)

    assert primary == "http://mac.local:8049"
    assert fallback == "http://192.168.1.10:8049"


def test_bridge_base_urls_primary_equals_fallback_when_no_mdns(mocker) -> None:
    mocker.patch.object(lan, "get_lan_ip", return_value="10.0.0.1")
    mocker.patch.object(lan, "get_mdns_host", return_value=None)

    primary, fallback = lan.bridge_base_urls(8048)

    assert primary == fallback == "http://10.0.0.1:8049"


def test_bridge_base_urls_default_port_is_8048() -> None:
    import inspect

    sig = inspect.signature(lan.bridge_base_urls)
    assert sig.parameters["port"].default == 8048


def test_bridge_base_urls_uses_custom_port(mocker) -> None:
    mocker.patch.object(lan, "get_lan_ip", return_value="10.0.0.1")
    mocker.patch.object(lan, "get_mdns_host", return_value=None)

    primary, _ = lan.bridge_base_urls(9999)

    assert primary == "http://10.0.0.1:10000"
