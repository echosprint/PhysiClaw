"""
All MCP tools — the agent's surface for controlling the phone.

Mental model: **See → Act**. Take a photo, pick a bbox, do something there.

All tools are async — blocking hardware I/O runs in a thread pool
via asyncio.to_thread() so the event loop stays free for HTTP routes.
"""

import asyncio
import logging
import time
from functools import wraps
from typing import Callable, Literal, TypeVar, cast

from mcp.server.fastmcp import FastMCP, Image

from physiclaw.core import PhysiClaw

log = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)

# Max length of the arg repr in a tool-call log line. Protects the log
# from dumping large text/dicts (e.g. 100KB clipboard bodies).
_MAX_ARG_LOG_LEN = 80


def _format_args(fn_name: str, kwargs: dict) -> str:
    """Build a safe arg string for the tool-call log line.

    Redacts clipboard text (user-visible content — IM bodies, search
    queries, anything pasted) and summarizes sequence steps to tool
    names only, since a step may itself be a send_to_clipboard.
    """
    if fn_name == "send_to_clipboard":
        return f"text=<{len(kwargs.get('text', ''))} chars>"
    if fn_name == "sequence":
        steps = [v for v in kwargs.values() if isinstance(v, dict)]
        names = [s.get("tool_name", "?") for s in steps]
        return f"{len(steps)} steps: {', '.join(names)}"
    arg_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    if len(arg_str) > _MAX_ARG_LOG_LEN:
        arg_str = arg_str[: _MAX_ARG_LOG_LEN - 3] + "..."
    return arg_str


def _logged(fn: F) -> F:
    # FastMCP dispatches tool calls with keyword args only (positional
    # args land in `args` but never in practice); the log reads kwargs.
    @wraps(fn)
    async def wrapper(*args, **kwargs):
        info = log.isEnabledFor(logging.INFO)
        arg_str = _format_args(fn.__name__, kwargs) if info else ""
        if info:
            log.info("tool %s(%s) start", fn.__name__, arg_str)
        t0 = time.monotonic()
        try:
            return await fn(*args, **kwargs)
        finally:
            if info:
                log.info("tool %s done — %.1fs", fn.__name__, time.monotonic() - t0)
    return cast(F, wrapper)


def register(mcp: FastMCP, physiclaw: PhysiClaw):
    """Register every MCP tool on the given FastMCP instance."""

    # ─── See ─────────────────────────────────────────────────

    @mcp.tool()
    @_logged
    async def scan() -> str:
        """OCR the overhead camera view. Text only, no image. ~1s.

        Use for: reading text on screen, checking status, verifying results.
        Returns JSON in same format as screenshot() but text-only (no icons).
        """
        return await asyncio.to_thread(physiclaw.scan)

    @mcp.tool()
    @_logged
    async def peek() -> Image:
        """Quick look via the overhead camera. ~3s.

        Use for: verifying an action landed, checking current status.
        For precise bboxes before acting, use screenshot() instead.
        """
        data = await asyncio.to_thread(physiclaw.peek)
        return Image(data=data, format="jpeg")

    @mcp.tool()
    @_logged
    async def screenshot() -> list:
        """Pixel-perfect screenshot with UI elements detected. ~12s.

        Use for: planning an action — returns precise bboxes to feed
        straight into tap/swipe/etc.
        Returns Image (numbered bboxes drawn) + JSON list, each entry:
            id:    int — index of the bbox, same number drawn on the image
            kind:  "icon" | "text"
            label: str — OCR text for "text", "" for "icon"
            bbox:  [left, top, right, bottom] — 0-1 decimals
            conf:  float — detector confidence, 0-1
        """
        jpeg, elements_json = await asyncio.to_thread(physiclaw.screenshot)
        return [Image(data=jpeg, format="jpeg"), elements_json]

    # ─── Act ─────────────────────────────────────────────────

    @mcp.tool()
    @_logged
    async def tap(bbox: list[float]) -> str:
        """Single tap at the bbox center.

        Use for: buttons, links, selecting items, dismissing dialogs.
        bbox: [left, top, right, bottom] as 0-1 decimals.
        """
        return await asyncio.to_thread(physiclaw.tap, bbox)

    @mcp.tool()
    @_logged
    async def double_tap(bbox: list[float]) -> str:
        """Double tap at the bbox center.

        Use for: zooming maps/photos/web pages, selecting a word.
        bbox: [left, top, right, bottom] as 0-1 decimals.
        """
        return await asyncio.to_thread(physiclaw.double_tap, bbox)

    @mcp.tool()
    @_logged
    async def long_press(bbox: list[float]) -> str:
        """Long press at the bbox center. ~1.2s hold.

        Use for: context menus, edit mode, paste, rearranging icons.
        bbox: [left, top, right, bottom] as 0-1 decimals.
        """
        return await asyncio.to_thread(physiclaw.long_press, bbox)

    # ─── Swipe ───────────────────────────────────────────────

    @mcp.tool()
    @_logged
    async def swipe(
        bbox: list[float],
        direction: Literal["up", "down", "left", "right"],
        size: Literal["s", "m", "l", "xl", "xxl"] = "m",
        speed: Literal["slow", "medium", "fast"] = "medium",
    ) -> str:
        """Stylus slides across a region. The direction is the stylus motion.

        Use for: scrolling (swipe up to scroll down), dismissing cards,
        changing pages, revealing list-item actions.
        bbox:      [left, top, right, bottom] as 0-1 decimals — region to swipe in.
        direction: 'up' | 'down' | 'left' | 'right' — stylus motion direction.
        size:      's' | 'm' | 'l' | 'xl' | 'xxl'.
        speed:     'slow' | 'medium' | 'fast'.
        """
        return await asyncio.to_thread(physiclaw.swipe, bbox, direction, size, speed)

    # ─── Navigate ────────────────────────────────────────────

    @mcp.tool()
    @_logged
    async def home_screen() -> str:
        """Go to the home screen. iPhone swipe-up-from-bottom gesture.

        Use for: exiting any app, returning to the launcher.
        """
        return await asyncio.to_thread(physiclaw.home_screen)

    @mcp.tool()
    @_logged
    async def go_back() -> str:
        """Go back one screen. iPhone swipe-from-left-edge gesture.

        Use for: navigating back in apps with a nav stack.
        """
        return await asyncio.to_thread(physiclaw.go_back)

    @mcp.tool()
    @_logged
    async def unlock_phone() -> str:
        """Unlock the phone by entering passcode 111111. ~12s.

        Wakes screen, swipes up, waits for Face ID to fail, OCRs the
        keypad, taps passcode. Hardcoded to 111111 — a throwaway
        tool-phone code so a real password never leaks via git or logs.

        If unlock fails, tell the user: change phone passcode to 111111,
        or turn off auto-lock (Settings → Display & Brightness → Auto-Lock
        → Never), though always-on will wear the display over time.
        """
        return await asyncio.to_thread(physiclaw.unlock_phone)

    # ─── Text ────────────────────────────────────────────────

    @mcp.tool()
    @_logged
    async def send_to_clipboard(text: str) -> str:
        """Copy text into the phone's clipboard.

        Use for: entering text into a field — on-screen typing is slow.
        After this returns, paste with: long_press(field_bbox) → tap "Paste".
        text: the string to put on the clipboard.
        """
        return await asyncio.to_thread(physiclaw.send_to_clipboard, text)

    # ─── Sequence ────────────────────────────────────────────

    @mcp.tool()
    @_logged
    async def sequence(
        step1: dict,
        step2: dict | None = None,
        step3: dict | None = None,
        step4: dict | None = None,
        step5: dict | None = None,
    ) -> str:
        """Run up to 5 actions sequentially in one call.

        Best for high-frequency deterministic flows — opening an app,
        navigating to a chat, pasting and sending an IM message, etc.
        Use when you already know each action and the screen it lands on,
        so observing between steps would add nothing. Stops at the first
        failure; earlier steps are not rolled back, so `scan()` before
        retrying if a step fails.

        Each step has two fields:
            tool_name: the tool to run — MUST be one of:
                       tap, double_tap, long_press, swipe, send_to_clipboard.
                       No other tools are accepted.
            arg:       the argument that tool accepts (see its own docstring).
        """
        steps = [s for s in (step1, step2, step3, step4, step5) if s is not None]
        return await asyncio.to_thread(physiclaw.sequence, steps)
