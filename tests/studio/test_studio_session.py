"""Tests for `physiclaw.studio.session` — block normalization and the
session's connect/act lifecycle (against a fake McpClient)."""

from __future__ import annotations

import pytest

from physiclaw.studio import session as session_mod
from physiclaw.studio.session import StudioSession, view_reply

LISTING = (
    'listing (id [kind] "label" [left,top,right,bottom] conf):\n'
    '1 [text] "加入购物车" [0.100,0.900,0.400,0.950] 0.95\n'
    '2 [icon] "" [0.850,0.050,0.950,0.120] 0.80'
)
IMAGE = {"type": "image", "mime_type": "image/jpeg", "data": "aGk="}


# ---------- view_reply ----------


def test_view_reply_peek_shape_image_then_listing() -> None:
    out = view_reply([IMAGE, {"type": "text", "text": LISTING}])

    assert out["text"] == ""
    assert out["image"] == {"mime_type": "image/jpeg", "data": "aGk="}
    assert [r["label"] for r in out["rows"]] == ["加入购物车", ""]
    assert out["rows"][0]["bbox"] == [0.1, 0.9, 0.4, 0.95]


def test_view_reply_gesture_shape_text_before_image_is_action_text() -> None:
    out = view_reply(
        [
            {"type": "text", "text": "tap ok — screen changed"},
            IMAGE,
            {"type": "text", "text": LISTING},
        ]
    )

    assert out["text"] == "tap ok — screen changed"
    assert len(out["rows"]) == 2


def test_view_reply_text_only_fallback_has_no_view() -> None:
    out = view_reply([{"type": "text", "text": "copied to clipboard"}])

    assert out == {
        "text": "copied to clipboard",
        "image": None,
        "rows": [],
        "listing": "",
    }


# ---------- session lifecycle ----------


class FakeClient:
    """Stands in for McpClient: canned tools/replies, records calls."""

    def __init__(self, base_url=None, reply=None, fail=None):
        self.calls: list[tuple[str, dict]] = []
        self._reply = reply or [IMAGE, {"type": "text", "text": LISTING}]
        self._fail = fail

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def list_tools(self):
        return [{"name": "peek"}, {"name": "tap"}]

    async def call_tool(self, name, args=None):
        if self._fail is not None:
            raise self._fail
        self.calls.append((name, args or {}))
        return self._reply


def _session(monkeypatch, **kwargs) -> StudioSession:
    monkeypatch.setattr(
        session_mod, "McpClient", lambda base: FakeClient(base, **kwargs)
    )
    return StudioSession(mcp_url="http://127.0.0.1:8048/mcp")


@pytest.mark.asyncio
async def test_act_connects_lazily_and_normalizes(monkeypatch) -> None:
    s = _session(monkeypatch)
    assert s.state()["connected"] is False

    out = await s.act("tap", {"bbox": [0.1, 0.2, 0.3, 0.4]})

    assert s.state()["connected"] is True
    assert s.state()["tools"] == ["peek", "tap"]
    assert len(out["rows"]) == 2


@pytest.mark.asyncio
async def test_act_refuses_a_tool_off_the_published_surface(monkeypatch) -> None:
    s = _session(monkeypatch)

    with pytest.raises(ValueError, match="published"):
        await s.act("calibrate", {})


@pytest.mark.asyncio
async def test_tool_error_keeps_the_session(monkeypatch) -> None:
    # The tool ran and said no — a rig-state complaint, not a dead
    # transport. The client must survive for the next action.
    s = _session(monkeypatch, fail=RuntimeError("tool 'tap' failed: not calibrated"))

    with pytest.raises(RuntimeError, match="not calibrated"):
        await s.act("tap", {})

    assert s.state()["connected"] is True


@pytest.mark.asyncio
async def test_transport_error_drops_the_client_for_reconnect(monkeypatch) -> None:
    s = _session(monkeypatch, fail=OSError("connection reset"))

    with pytest.raises(ConnectionError, match="connection reset"):
        await s.act("tap", {})

    assert s.state()["connected"] is False


def test_mcp_url_accepts_both_spellings() -> None:
    with_suffix = StudioSession(mcp_url="http://127.0.0.1:8048/mcp")
    without = StudioSession(mcp_url="http://127.0.0.1:8048")

    assert with_suffix.mcp_url == "http://127.0.0.1:8048/mcp"
    assert without.mcp_url == "http://127.0.0.1:8048/mcp"
