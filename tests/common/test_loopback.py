"""Tests for `physiclaw.common.loopback` — the spellings of "this host".

Exact-value pins, deliberately: two of the three forms are security
allowlists (the control plane's Host gate, the bridge's TOFU
exemption), so a widened set must fail a test, not slip through as a
"shared constant" cleanup.
"""

from physiclaw.common.loopback import (
    LOOPBACK_HOST_HEADERS,
    LOOPBACK_HOSTS,
    LOOPBACK_IPS,
)


def test_hosts_are_the_no_proxy_entries_in_order() -> None:
    assert LOOPBACK_HOSTS == ("127.0.0.1", "localhost", "::1")


def test_ips_are_hosts_minus_the_hostname() -> None:
    assert LOOPBACK_IPS == frozenset({"127.0.0.1", "::1"})


def test_host_headers_add_only_the_bracketed_v6() -> None:
    assert LOOPBACK_HOST_HEADERS == frozenset(
        {"127.0.0.1", "localhost", "::1", "[::1]"}
    )
