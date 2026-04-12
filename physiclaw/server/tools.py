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


def register(mcp: FastMCP, physiclaw: PhysiClaw):
    """Register every MCP tool on the given FastMCP instance."""

    # ─── See ─────────────────────────────────────────────────

    @mcp.tool()
    async def scan() -> str:
        """OCR the overhead camera view. Text only, no image. ~1s.

        Use for: reading text on screen, checking status, verifying results.
        Returns JSON in same format as screenshot() but text-only (no icons).
        """
        return await asyncio.to_thread(physiclaw.scan)

    @mcp.tool()
    async def peek() -> Image:
        """Quick look via the overhead camera. ~3s.

        Use for: verifying an action landed, checking current status.
        For precise bboxes before acting, use screenshot() instead.
        """
        data = await asyncio.to_thread(physiclaw.peek)
        return Image(data=data, format="jpeg")

    @mcp.tool()
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
    async def tap(bbox: list[float]) -> str:
        """Single tap at the bbox center.

        Use for: buttons, links, selecting items, dismissing dialogs.
        bbox: [left, top, right, bottom] as 0-1 decimals.
        """
        return await asyncio.to_thread(physiclaw.tap, bbox)

    @mcp.tool()
    async def double_tap(bbox: list[float]) -> str:
        """Double tap at the bbox center.

        Use for: zooming maps/photos/web pages, selecting a word.
        bbox: [left, top, right, bottom] as 0-1 decimals.
        """
        return await asyncio.to_thread(physiclaw.double_tap, bbox)

    @mcp.tool()
    async def long_press(bbox: list[float]) -> str:
        """Long press at the bbox center. ~1.2s hold.

        Use for: context menus, edit mode, paste, rearranging icons.
        bbox: [left, top, right, bottom] as 0-1 decimals.
        """
        return await asyncio.to_thread(physiclaw.long_press, bbox)

    # ─── Swipe ───────────────────────────────────────────────

    @mcp.tool()
    async def swipe(
        bbox: list[float],
        direction: Literal["up", "down", "left", "right"],
        size: Literal["s", "m", "l", "xl"] = "m",
        speed: Literal["slow", "medium", "fast"] = "medium",
    ) -> str:
        """Stylus slides across a region. The direction is the stylus motion.

        Use for: scrolling (swipe up to scroll down), dismissing cards,
        changing pages, revealing list-item actions.
        bbox:      [left, top, right, bottom] as 0-1 decimals — region to swipe in.
        direction: 'up' | 'down' | 'left' | 'right' — stylus motion direction.
        size:      's' | 'm' | 'l' | 'xl'.
        speed:     'slow' | 'medium' | 'fast'.
        """
        return await asyncio.to_thread(physiclaw.swipe, bbox, direction, size, speed)

    # ─── Navigate ────────────────────────────────────────────

    @mcp.tool()
    async def home_screen() -> str:
        """Go to the home screen. iPhone swipe-up-from-bottom gesture.

        Use for: exiting any app, returning to the launcher.
        """
        return await asyncio.to_thread(physiclaw.home_screen)

    @mcp.tool()
    async def go_back() -> str:
        """Go back one screen. iPhone swipe-from-left-edge gesture.

        Use for: navigating back in apps with a nav stack.
        """
        return await asyncio.to_thread(physiclaw.go_back)

    # ─── Text ────────────────────────────────────────────────

    @mcp.tool()
    async def send_to_clipboard(text: str) -> str:
        """Copy text into the phone's clipboard.

        Use for: entering text into a field — on-screen typing is slow.
        After this returns, paste with: long_press(field_bbox) → tap "Paste".
        text: the string to put on the clipboard.
        """
        return await asyncio.to_thread(physiclaw.send_to_clipboard, text)
