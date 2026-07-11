"""Tests for `physiclaw.proxy` — proxy env normalization.

The regression under guard: on Linux with a VPN on, `all_proxy=
socks://127.0.0.1:7897` (GNOME / Clash Verge convention) made httpx
raise `ValueError: Unknown scheme for proxy URL` inside every
`trust_env=True` client, so `physiclaw models discover <provider>`
died before sending a single byte.
"""

import os

import httpx
import pytest

from physiclaw.proxy import normalize_proxy_env


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
    transport mounts instead of raising ImportError)."""
    monkeypatch.setenv("all_proxy", "socks://127.0.0.1:7897")
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    normalize_proxy_env()
    client = httpx.Client(trust_env=True)  # raised ValueError before the fix
    client.close()
