"""Tests for `physiclaw.core.server.tools` — MCP tool registrations.

Each tool is registered via `@mcp.tool()` and delegates to a method on
the PhysiClaw orchestrator via `asyncio.to_thread`. We feed register a
fake mcp that records and exposes each registered tool, then invoke
each one and verify it dispatches with the right args, returns the
right shape, and ends with the right hint constant.

Hint strings are pinned to module-level constants in `tools.py`. Any
silent reword to a hint surfaces here as a failed test — the hints
are part of the prompt contract sent to the agent.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from mcp.server.fastmcp import Image

from physiclaw.core.server import tools as tools_mod

# ---------- Fake MCP ----------


class FakeMcp:
    """Records every @mcp.tool()'d coroutine in registration order."""

    def __init__(self) -> None:
        self.tools: dict = {}
        self.tool_kwargs: dict = {}

    def tool(self, **kwargs):
        def deco(fn):
            self.tools[fn.__name__] = fn
            self.tool_kwargs[fn.__name__] = kwargs
            return fn

        return deco


# ---------- helpers ----------


@pytest.fixture
def registered(mocker):
    """Build a registered mcp + a bare-mock physiclaw.

    Each test stubs only the orchestrator methods it touches. The
    fixture itself sets nothing — fewer cross-test couplings, easier
    to add new tools."""
    mcp = FakeMcp()
    pl = MagicMock()
    mocker.patch.object(tools_mod, "save_tool_call")
    tools_mod.register(mcp, pl)
    return mcp, pl


def _gr(text: str, *, with_view: bool = True):
    """Fake GestureResult for orchestrator mocks."""
    from physiclaw.core.orchestration import GestureResult

    if with_view:
        return GestureResult(text=text, jpeg=b"VIEW", listing="LISTING")
    return GestureResult(text=text)


def _assert_view_reply(out, prefix: str, hint: str) -> None:
    """Pin the fused-view reply shape: [text-with-hint, Image, listing]."""
    assert isinstance(out, list) and len(out) == 3
    assert out[0].startswith(prefix)
    assert out[0].endswith(hint)
    assert isinstance(out[1], Image)
    assert out[1].data == b"VIEW"
    assert out[2] == "LISTING"


# ---------- Registration ----------


def test_register_wires_all_expected_tools(registered) -> None:
    """Every documented tool is registered. Set-based — answers
    'is anything missing or extra' independent of order."""
    mcp, _ = registered

    expected = {
        "peek",
        "screenshot",
        "tap",
        "double_tap",
        "long_press",
        "swipe",
        "home_screen",
        "go_back",
        "force_quit",
        "unlock_phone",
        "send_to_clipboard",
        "sequence",
    }
    assert set(mcp.tools.keys()) == expected


def test_tool_registration_order_is_stable(registered) -> None:
    """Source order is wire order (per module docstring); LLM position
    bias means changing it can change agent behavior. Lock the sequence."""
    mcp, _ = registered

    assert list(mcp.tools.keys()) == [
        "peek",
        "screenshot",
        "tap",
        "double_tap",
        "long_press",
        "swipe",
        "home_screen",
        "go_back",
        "force_quit",
        "unlock_phone",
        "send_to_clipboard",
        "sequence",
    ]


def test_all_tools_disable_structured_output(registered) -> None:
    """An `Image` in a reply is unserializable by the SDK's structured-
    output path ("Unable to serialize unknown type") — every tool must
    register with structured_output=False so results go through the
    content-block conversion that handles Image."""
    mcp, _ = registered

    assert all(
        kw.get("structured_output") is False for kw in mcp.tool_kwargs.values()
    ), mcp.tool_kwargs


# ---------- peek / screenshot ----------


@pytest.mark.asyncio
async def test_peek_returns_image_and_listing(registered) -> None:
    mcp, pl = registered
    pl.peek.return_value = (b"PEEK_JPG", "peek-listing")

    out = await mcp.tools["peek"]()

    assert pl.peek.called
    assert len(out) == 2
    assert isinstance(out[0], Image)
    assert out[0].data == b"PEEK_JPG"
    assert out[0]._mime_type == "image/jpeg"
    assert out[1] == "peek-listing"
    tools_mod.save_tool_call.assert_called_once_with(
        "peek",
        "peek-listing",
        b"PEEK_JPG",
    )


@pytest.mark.asyncio
async def test_screenshot_returns_image_and_listing(registered) -> None:
    mcp, pl = registered
    pl.screenshot.return_value = (b"SS_JPG", "ss-listing")

    out = await mcp.tools["screenshot"]()

    assert pl.screenshot.called
    assert len(out) == 2
    assert isinstance(out[0], Image)
    assert out[0].data == b"SS_JPG"
    assert out[0]._mime_type == "image/jpeg"
    assert out[1] == "ss-listing"
    tools_mod.save_tool_call.assert_called_once_with(
        "screenshot",
        "ss-listing",
        b"SS_JPG",
    )


# ---------- tap / double_tap / long_press ----------


@pytest.mark.asyncio
async def test_tap_dispatches_and_appends_hint(registered) -> None:
    mcp, pl = registered
    pl.tap.return_value = _gr("tap result")

    out = await mcp.tools["tap"]([0.1, 0.1, 0.2, 0.2])

    pl.tap.assert_called_once_with([0.1, 0.1, 0.2, 0.2])
    _assert_view_reply(out, "tap result ", tools_mod.HINT_VIEW_AFTER_TAP)


@pytest.mark.asyncio
async def test_double_tap_dispatches_and_appends_hint(registered) -> None:
    mcp, pl = registered
    pl.double_tap.return_value = _gr("dt result")

    out = await mcp.tools["double_tap"]([0, 0, 1, 1])

    pl.double_tap.assert_called_once_with([0, 0, 1, 1])
    _assert_view_reply(out, "dt result ", tools_mod.HINT_VIEW_AFTER_DOUBLE_TAP)


@pytest.mark.asyncio
async def test_long_press_dispatches_and_appends_hint(registered) -> None:
    mcp, pl = registered
    pl.long_press.return_value = _gr("lp result")

    out = await mcp.tools["long_press"]([0, 0, 1, 1])

    pl.long_press.assert_called_once_with([0, 0, 1, 1])
    _assert_view_reply(out, "lp result ", tools_mod.HINT_VIEW_AFTER_LONG_PRESS)


# ---------- swipe ----------


@pytest.mark.asyncio
async def test_swipe_dispatches_with_all_args(registered) -> None:
    mcp, pl = registered
    pl.swipe.return_value = _gr("swipe result")

    out = await mcp.tools["swipe"]([0, 0, 1, 1], "up", "l", "fast")

    pl.swipe.assert_called_once_with([0, 0, 1, 1], "up", "l", "fast")
    _assert_view_reply(out, "swipe result ", tools_mod.HINT_VIEW_AFTER_SWIPE)


@pytest.mark.asyncio
async def test_swipe_default_size_when_only_speed_given(registered) -> None:
    """size defaults to 'm'."""
    mcp, pl = registered
    pl.swipe.return_value = _gr("ok")

    await mcp.tools["swipe"]([0, 0, 1, 1], "down", speed="fast")

    pl.swipe.assert_called_once_with([0, 0, 1, 1], "down", "m", "fast")


@pytest.mark.asyncio
async def test_swipe_default_speed_when_only_size_given(registered) -> None:
    """speed defaults to 'medium'."""
    mcp, pl = registered
    pl.swipe.return_value = _gr("ok")

    await mcp.tools["swipe"]([0, 0, 1, 1], "down", size="l")

    pl.swipe.assert_called_once_with([0, 0, 1, 1], "down", "l", "medium")


@pytest.mark.asyncio
async def test_swipe_both_defaults(registered) -> None:
    mcp, pl = registered
    pl.swipe.return_value = _gr("ok")

    await mcp.tools["swipe"]([0, 0, 1, 1], "down")

    pl.swipe.assert_called_once_with([0, 0, 1, 1], "down", "m", "medium")


# ---------- Navigation ----------


@pytest.mark.asyncio
async def test_home_screen_dispatches_and_appends_hint(registered) -> None:
    mcp, pl = registered
    pl.home_screen.return_value = _gr("home result")

    out = await mcp.tools["home_screen"]()

    assert pl.home_screen.called
    _assert_view_reply(out, "home result ", tools_mod.HINT_VIEW_AFTER_HOME)


@pytest.mark.asyncio
async def test_go_back_dispatches_and_appends_hint(registered) -> None:
    mcp, pl = registered
    pl.go_back.return_value = _gr("back result")

    out = await mcp.tools["go_back"]()

    assert pl.go_back.called
    _assert_view_reply(out, "back result ", tools_mod.HINT_VIEW_AFTER_BACK)


@pytest.mark.asyncio
async def test_force_quit_dispatches_and_appends_hint(registered) -> None:
    mcp, pl = registered
    pl.force_quit.return_value = _gr("fq result")

    out = await mcp.tools["force_quit"]()

    assert pl.force_quit.called
    _assert_view_reply(out, "fq result ", tools_mod.HINT_VIEW_AFTER_FORCE_QUIT)


@pytest.mark.asyncio
async def test_unlock_phone_dispatches_and_appends_hint(registered) -> None:
    mcp, pl = registered
    pl.unlock_phone.return_value = _gr("unlock result")

    out = await mcp.tools["unlock_phone"]()

    assert pl.unlock_phone.called
    _assert_view_reply(out, "unlock result ", tools_mod.HINT_VIEW_AFTER_UNLOCK)


def test_unlock_docstring_quotes_the_still_locked_retry_trigger(registered) -> None:
    # Cross-file byte contract: the tool docstring tells the agent to
    # retry on the orchestrator's "still locked" phrase, so a reword of
    # one side that doesn't update the other silently breaks retries.
    from physiclaw.core.orchestration.orchestrator import UNLOCK_STILL_LOCKED

    mcp, _ = registered
    doc = " ".join((mcp.tools["unlock_phone"].__doc__ or "").split())
    assert "still shows the lock screen" in UNLOCK_STILL_LOCKED
    assert "still shows the lock screen" in doc


def test_unlock_docstring_quotes_the_current_passcode(registered) -> None:
    # Same contract class as the retry trigger above: the docstring
    # tells the agent (and the STUCK message tells the user) the literal
    # passcode, so a UNLOCK_PASSCODE change that skips the docstring
    # keeps steering everyone to the old code.
    from physiclaw.core.orchestration.gestures import UNLOCK_PASSCODE

    mcp, _ = registered
    doc = mcp.tools["unlock_phone"].__doc__ or ""
    assert UNLOCK_PASSCODE in doc


# ---------- send_to_clipboard ----------


@pytest.mark.asyncio
async def test_send_to_clipboard_dispatches_and_appends_hint(registered) -> None:
    mcp, pl = registered
    pl.send_to_clipboard.return_value = "clip result"

    out = await mcp.tools["send_to_clipboard"]("hello world")

    pl.send_to_clipboard.assert_called_once_with("hello world")
    assert out.startswith("clip result ")
    assert out.endswith(tools_mod.HINT_AFTER_CLIPBOARD)


# ---------- sequence ----------


@pytest.mark.asyncio
async def test_sequence_passes_actions_through(registered) -> None:
    mcp, pl = registered
    pl.sequence.return_value = _gr("seq ok")
    actions = [
        {"tool_name": "tap", "arg": [0, 0, 1, 1]},
        {"tool_name": "tap", "arg": [0, 0, 1, 1]},
    ]

    out = await mcp.tools["sequence"](actions)

    pl.sequence.assert_called_once_with(actions)
    # Hint sits on its own line after a result newline; view attached.
    assert out[0].endswith("\n" + tools_mod.HINT_VIEW_AFTER_SEQUENCE)
    assert isinstance(out[1], Image)


@pytest.mark.asyncio
async def test_sequence_single_action(registered) -> None:
    mcp, pl = registered
    pl.sequence.return_value = _gr("ok")
    actions = [{"tool_name": "send_to_clipboard", "arg": "hi"}]

    await mcp.tools["sequence"](actions)

    pl.sequence.assert_called_once_with(actions)


@pytest.mark.asyncio
async def test_sequence_all_five_actions(registered) -> None:
    mcp, pl = registered
    pl.sequence.return_value = _gr("ok")
    actions = [{"tool_name": "tap", "arg": [0, 0, 1, 1]} for _ in range(5)]

    await mcp.tools["sequence"](actions)

    pl.sequence.assert_called_once_with(actions)


@pytest.mark.asyncio
async def test_gesture_without_view_falls_back_to_text(registered) -> None:
    # Camera/detector hiccup -> text-only reply routing the agent to peek.
    mcp, pl = registered
    pl.tap.return_value = _gr("tap result", with_view=False)

    out = await mcp.tools["tap"]([0.1, 0.1, 0.2, 0.2])

    assert isinstance(out, str)
    assert out.startswith("tap result ")
    assert out.endswith(tools_mod.NO_VIEW_FALLBACK)


# ---------- Error propagation ----------


@pytest.mark.asyncio
async def test_tap_propagates_hardware_error(registered) -> None:
    """Hardware exceptions surface unchanged so the engine sees the
    real error, not a string-wrapped placeholder."""
    mcp, pl = registered
    pl.tap.side_effect = RuntimeError("stylus offline")

    with pytest.raises(RuntimeError, match=r"^stylus offline$"):
        await mcp.tools["tap"]([0, 0, 1, 1])


@pytest.mark.asyncio
async def test_peek_propagates_camera_error(registered) -> None:
    mcp, pl = registered
    pl.peek.side_effect = RuntimeError("camera busy")

    with pytest.raises(RuntimeError, match=r"^camera busy$"):
        await mcp.tools["peek"]()


@pytest.mark.asyncio
async def test_screenshot_propagates_timeout_error(registered) -> None:
    mcp, pl = registered
    pl.screenshot.side_effect = TimeoutError("upload timed out")

    with pytest.raises(TimeoutError, match=r"^upload timed out$"):
        await mcp.tools["screenshot"]()


@pytest.mark.asyncio
async def test_sequence_propagates_mid_flight_failure(registered) -> None:
    """`sequence` docstring promises stop-at-first-failure, no rollback.
    The orchestrator surfaces the exception as-is; the tool wrapper
    must not swallow it into a string."""
    mcp, pl = registered
    pl.sequence.side_effect = ValueError("step 2: arm jammed")

    with pytest.raises(ValueError, match=r"^step 2: arm jammed$"):
        await mcp.tools["sequence"](
            {"tool_name": "tap", "arg": [0, 0, 1, 1]},
        )


@pytest.mark.asyncio
async def test_send_to_clipboard_propagates_bridge_error(registered) -> None:
    mcp, pl = registered
    pl.send_to_clipboard.side_effect = ConnectionError("bridge gone")

    with pytest.raises(ConnectionError, match=r"^bridge gone$"):
        await mcp.tools["send_to_clipboard"]("x")


# ---------- real FastMCP serialization (regression) ----------
#
# The FakeMcp harness above calls tool functions directly, bypassing
# FastMCP's result conversion — which is exactly where a `list | str`
# return annotation once made the SDK build a structured-output model
# and choke on `Image` ("Unable to serialize unknown type"). These tests
# push gesture/view replies through a REAL FastMCP instance.


@pytest.mark.asyncio
async def test_real_fastmcp_serializes_gesture_reply() -> None:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ImageContent, TextContent

    mcp = FastMCP("t")
    pl = MagicMock()
    pl.tap.return_value = _gr("Tapped at bbox [...] | screen: changed")
    tools_mod.register(mcp, pl)

    blocks = await mcp.call_tool("tap", {"bbox": [0.1, 0.1, 0.2, 0.2]})

    assert isinstance(blocks[0], TextContent)
    assert blocks[0].text.startswith("Tapped at bbox")
    assert isinstance(blocks[1], ImageContent)
    assert isinstance(blocks[2], TextContent)
    assert blocks[2].text == "LISTING"


@pytest.mark.asyncio
async def test_real_fastmcp_serializes_text_fallback_and_view_tools() -> None:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ImageContent, TextContent

    mcp = FastMCP("t")
    pl = MagicMock()
    pl.go_back.return_value = _gr("Went back", with_view=False)
    pl.peek.return_value = (b"PEEK_JPG", "peek-listing")
    tools_mod.register(mcp, pl)

    fallback = await mcp.call_tool("go_back", {})
    assert isinstance(fallback[0], TextContent)
    assert tools_mod.NO_VIEW_FALLBACK in fallback[0].text

    view = await mcp.call_tool("peek", {})
    assert isinstance(view[0], ImageContent)
    assert isinstance(view[1], TextContent)
