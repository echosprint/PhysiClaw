"""Tests for `physiclaw.common.proxy` — proxy env normalization.

The regression under guard: on Linux with a VPN on, `all_proxy=
socks://127.0.0.1:7897` (GNOME / Clash Verge convention) made httpx
raise `ValueError: Unknown scheme for proxy URL` inside every
`trust_env=True` client, so `physiclaw models discover <provider>`
died before sending a single byte.
"""

import os

import httpx
import pytest

from physiclaw.common.proxy import normalize_proxy_env


@pytest.mark.parametrize("var", ["all_proxy", "ALL_PROXY", "http_proxy", "HTTPS_PROXY"])
def test_socks_scheme_rewritten_to_socks5(monkeypatch, var):
    monkeypatch.setenv(var, "socks://127.0.0.1:7897")
    normalize_proxy_env()
    assert os.environ[var] == "socks5://127.0.0.1:7897"


def test_uppercase_scheme_rewritten(monkeypatch):
    monkeypatch.setenv("all_proxy", "SOCKS://proxy.lan:1080")
    normalize_proxy_env()
    assert os.environ["all_proxy"] == "socks5://proxy.lan:1080"


@pytest.mark.parametrize(
    "value",
    [
        "socks5://127.0.0.1:7897",  # already correct
        "socks4://127.0.0.1:7897",  # explicit socks4 — not ours to reinterpret
        "http://127.0.0.1:7897",
        "127.0.0.1:7897",  # schemeless — httpx assumes http:// itself
        "",
    ],
)
def test_other_values_left_untouched(monkeypatch, value):
    monkeypatch.setenv("all_proxy", value)
    normalize_proxy_env()
    assert os.environ["all_proxy"] == value


def test_non_proxy_vars_left_untouched(monkeypatch):
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1")
    monkeypatch.setenv("MY_APP_SOCKS", "socks://x:1")
    normalize_proxy_env()
    assert os.environ["no_proxy"] == "localhost,127.0.0.1"
    assert os.environ["MY_APP_SOCKS"] == "socks://x:1"


def test_httpx_accepts_normalized_env(monkeypatch):
    """End-to-end guard: the exact failing setup builds a client after
    normalization (socksio ships as a dependency, so the socks5
    transport mounts instead of raising ImportError). The shell's own
    proxy vars are already stripped by conftest's `scrub_proxy_env`."""
    monkeypatch.setenv("all_proxy", "socks://127.0.0.1:7897")
    normalize_proxy_env()
    client = httpx.Client(trust_env=True)  # raised ValueError before the fix
    client.close()


# ---------- loopback NO_PROXY exemption ----------
# Guards: a proxied env must never route the runtime's own loopback
# clients (or the claude child's inherited env) through the proxy.
# Conftest's autouse `scrub_proxy_env` clears every proxy var first.


@pytest.mark.parametrize(
    ("existing", "expected"),
    [
        (None, "127.0.0.1,localhost,::1"),  # absent → set outright
        (  # user entries survive; only missing loopback hosts join
            "example.com,localhost",
            "example.com,localhost,127.0.0.1,::1",
        ),
        (  # fully covered (whitespace tolerated) → byte-untouched
            "127.0.0.1, localhost, ::1",
            "127.0.0.1, localhost, ::1",
        ),
    ],
)
def test_loopback_pinned_into_no_proxy(monkeypatch, existing, expected):
    monkeypatch.setenv("HTTP_PROXY", "http://localhost:6984")
    if existing is not None:
        monkeypatch.setenv("NO_PROXY", existing)

    normalize_proxy_env()

    assert os.environ["NO_PROXY"] == expected
    if existing == expected:  # already covered — untouched means untouched
        assert "no_proxy" not in os.environ
    else:  # a write sets both spellings
        assert os.environ["no_proxy"] == expected


def test_no_proxy_not_set_without_any_proxy_var():
    normalize_proxy_env()

    # Without a proxy, NO_PROXY is inert — don't surprise the environment.
    assert "NO_PROXY" not in os.environ
    assert "no_proxy" not in os.environ
