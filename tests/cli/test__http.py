"""Tests for `physiclaw.cli._http` — the CLI's one HTTP layer.

`api`/`fetch_json` are tested at the `_OPENER` seam; `http_get`/`stream`
directly here (and end-to-end by the download-path tests in
setup/test_vision and test_sync_official_skills).
"""
from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import MagicMock

import pytest

from physiclaw.cli import _http

BASE = "http://localhost:8048"


def _resp(payload: dict | bytes) -> MagicMock:
    """Build a context-manager-compatible urllib response."""
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = lambda s, *a: None
    return resp


# ---------- api ----------


def test_api_get_no_body(mocker) -> None:
    spy = mocker.patch.object(
        _http._OPENER, "open",
        return_value=_resp({"x": 1}),
    )

    out = _http.api(BASE, "GET", "/api/status")

    assert out == {"x": 1}
    req = spy.call_args.args[0]
    assert req.method == "GET"
    assert req.full_url == f"{BASE}/api/status"


def test_api_post_with_body(mocker) -> None:
    spy = mocker.patch.object(
        _http._OPENER, "open",
        return_value=_resp({"status": "ok"}),
    )

    out = _http.api(BASE, "POST", "/api/x", body={"k": "v"})

    assert out["status"] == "ok"
    req = spy.call_args.args[0]
    assert req.headers["Content-type"] == "application/json"


def test_api_post_no_body_sends_empty_bytes(mocker) -> None:
    spy = mocker.patch.object(
        _http._OPENER, "open",
        return_value=_resp({"status": "ok"}),
    )

    _http.api(BASE, "POST", "/api/x")

    # Empty bytes body for POST.
    req = spy.call_args.args[0]
    assert req.data == b""


def test_api_returns_parsed_error_body_on_http_error(mocker) -> None:
    err = urllib.error.HTTPError(
        url="x", code=500, msg="boom", hdrs=None, fp=None,
    )
    err.read = lambda: b'{"status": "error", "message": "x"}'
    mocker.patch.object(_http._OPENER, "open", side_effect=err)

    out = _http.api(BASE, "GET", "/api/x")

    assert out == {"status": "error", "message": "x"}


def test_api_returns_none_on_unparseable_error_body(mocker) -> None:
    err = urllib.error.HTTPError(
        url="x", code=500, msg="boom", hdrs=None, fp=None,
    )
    err.read = lambda: b"not json"
    mocker.patch.object(_http._OPENER, "open", side_effect=err)

    assert _http.api(BASE, "GET", "/api/x") is None


def test_api_returns_none_on_connection_error(mocker) -> None:
    mocker.patch.object(
        _http._OPENER, "open",
        side_effect=ConnectionError("refused"),
    )

    assert _http.api(BASE, "GET", "/api/x") is None


# ---------- fetch_json ----------


def test_fetch_json_returns_parsed_body(mocker) -> None:
    mocker.patch.object(
        _http._OPENER, "open",
        return_value=_resp({"connected": True}),
    )

    assert _http.fetch_json(f"{BASE}/api/bridge/state") == {"connected": True}


def test_fetch_json_raises_on_transport_error(mocker) -> None:
    # Unlike `api`, callers see the failure detail.
    mocker.patch.object(
        _http._OPENER, "open",
        side_effect=urllib.error.URLError("refused"),
    )

    with pytest.raises(urllib.error.URLError):
        _http.fetch_json(f"{BASE}/api/bridge/state")


def test_fetch_json_raises_on_bad_json(mocker) -> None:
    mocker.patch.object(_http._OPENER, "open", return_value=_resp(b"not json"))

    with pytest.raises(ValueError):
        _http.fetch_json(f"{BASE}/api/bridge/state")


# ---------- http_get / stream ----------


class _StreamResp:
    """Minimal urlopen-response stand-in for `stream`: chunked ``read``
    plus a Content-Length header."""

    def __init__(self, data: bytes, content_length: str | None = None) -> None:
        self._buf = io.BytesIO(data)
        self._len = content_length

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def getheader(self, name: str, default=None):
        if name.lower() == "content-length" and self._len is not None:
            return self._len
        return default


def test_http_get_sets_user_agent(mocker) -> None:
    # Cloudflare's WAF 403s the default Python-urllib UA — http_get must
    # pin one on every request.
    spy = mocker.patch.object(_http.urllib.request, "urlopen")

    _http.http_get("https://example.com/x")

    req = spy.call_args.args[0]
    assert req.get_header("User-agent") == _http.USER_AGENT


def test_stream_writes_all_bytes_with_known_length() -> None:
    data = b"x" * 5000
    out = bytearray()

    _http.stream(
        _StreamResp(data, content_length=str(len(data))), out.extend, "test"
    )

    assert bytes(out) == data


def test_stream_writes_all_bytes_with_unknown_length() -> None:
    data = b"y" * 5000
    out = bytearray()

    _http.stream(_StreamResp(data), out.extend, "test")  # no Content-Length

    assert bytes(out) == data
