"""
All MCP tools — the agent's surface for controlling the phone.

Mental model: **See → Act**. Take a photo, pick a bbox, do something there.

All tools are async — blocking hardware I/O runs in a thread pool
via asyncio.to_thread() so the event loop stays free for HTTP routes.
"""

import asyncio
from typing import Literal

from mcp.server.fastmcp import FastMCP, Image

from physiclaw.core import PhysiClaw
from physiclaw.logger import logged


def register(mcp: FastMCP, physiclaw: PhysiClaw):
    """Register every MCP tool on the given FastMCP instance."""

    # ─── See ─────────────────────────────────────────────────

    @mcp.tool()
    @logged
    async def scan() -> str:
        """Icon detection + OCR on the overhead camera view. Returns the
        element listing only (no image)."""
        return await asyncio.to_thread(physiclaw.scan)

    @mcp.tool()
    @logged
    async def peek() -> list:
        """Overhead camera snapshot + icon detection + OCR. Returns
        [Image, listing] — JPEG with numbered bboxes drawn over the
        cropped camera view, plus the element listing."""
        jpeg, listing = await asyncio.to_thread(physiclaw.peek)
        return [Image(data=jpeg, format="jpeg"), listing]

    @mcp.tool()
    @logged
    async def screenshot() -> list:
        """Pixel-perfect phone screenshot + icon detection + OCR. Returns
        [Image, listing] — JPEG with numbered bboxes drawn, plus the
        element listing."""
        jpeg, listing = await asyncio.to_thread(physiclaw.screenshot)
        return [Image(data=jpeg, format="jpeg"), listing]

    # ─── Act ─────────────────────────────────────────────────

    @mcp.tool()
    @logged
    async def tap(bbox: list[float]) -> str:
        """Single tap at the bbox center.

        Use for: buttons, links, selecting items, dismissing dialogs.
        """
        return await asyncio.to_thread(physiclaw.tap, bbox)

    @mcp.tool()
    @logged
    async def double_tap(bbox: list[float]) -> str:
        """Double tap at the bbox center.

        Use for: zooming maps/photos/web pages, selecting a word.
        """
        return await asyncio.to_thread(physiclaw.double_tap, bbox)

    @mcp.tool()
    @logged
    async def long_press(bbox: list[float]) -> str:
        """Long press at the bbox center. ~1.2s hold.

        Use for: context menus, edit mode, paste, rearranging icons.
        """
        return await asyncio.to_thread(physiclaw.long_press, bbox)

    # ─── Swipe ───────────────────────────────────────────────

    @mcp.tool()
    @logged
    async def swipe(
        bbox: list[float],
        direction: Literal["up", "down", "left", "right"],
        size: Literal["s", "m", "l", "xl", "xxl"] = "m",
        speed: Literal["slow", "medium", "fast"] = "medium",
    ) -> str:
        """Stylus slides across ``bbox`` in ``direction`` (the stylus motion).

        Use for: scrolling (swipe up to scroll down), dismissing cards,
        changing pages, revealing list-item actions.
        """
        return await asyncio.to_thread(physiclaw.swipe, bbox, direction, size, speed)

    # ─── Navigate ────────────────────────────────────────────

    @mcp.tool()
    @logged
    async def home_screen() -> str:
        """Go to the home screen. iPhone swipe-up-from-bottom gesture.

        Use for: exiting any app, returning to the launcher.
        """
        return await asyncio.to_thread(physiclaw.home_screen)

    @mcp.tool()
    @logged
    async def go_back() -> str:
        """Go back one screen. iPhone swipe-from-left-edge gesture.

        Use for: navigating back in apps with a nav stack.
        """
        return await asyncio.to_thread(physiclaw.go_back)

    @mcp.tool()
    @logged
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
    @logged
    async def send_to_clipboard(text: str) -> str:
        """Copy text into the phone's clipboard.

        Use for: entering text into a field — on-screen typing is slow.
        After this returns, paste with: long_press(field_bbox) → tap "Paste".
        """
        return await asyncio.to_thread(physiclaw.send_to_clipboard, text)

    # ─── Sequence ────────────────────────────────────────────

    @mcp.tool()
    @logged
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
