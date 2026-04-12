"""
All MCP tools — the agent's surface for controlling the phone.

Mental model: **See → Act**. Take a photo, pick a bbox, do something there.
"""

from mcp.server.fastmcp import FastMCP, Image

from physiclaw.core import PhysiClaw


def register(mcp: FastMCP, physiclaw: PhysiClaw):
    """Register every MCP tool on the given FastMCP instance."""

    # ─── See ─────────────────────────────────────────────────

    @mcp.tool()
    def scan() -> str:
        """OCR the overhead camera view. Text only, no image. ~1s.

        Use for: reading text on screen, checking status, verifying results.
        Returns JSON in same format as screenshot() but text-only (no icons).
        """
        return physiclaw.scan()

    @mcp.tool()
    def peek() -> Image:
        """Quick look via the overhead camera. ~3s.

        Use for: verifying an action landed, checking current status.
        For precise bboxes before acting, use screenshot() instead.
        """
        return Image(data=physiclaw.peek(), format="jpeg")

    @mcp.tool()
    def screenshot() -> list:
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
        jpeg, elements_json = physiclaw.screenshot()
        return [Image(data=jpeg, format="jpeg"), elements_json]

    # ─── Act ─────────────────────────────────────────────────

    @mcp.tool()
    def tap(bbox: list[float]) -> str:
        """Single tap at the bbox center.

        Use for: buttons, links, selecting items, dismissing dialogs.
        bbox: [left, top, right, bottom] as 0-1 decimals.
        """
        return physiclaw.tap(bbox)

    @mcp.tool()
    def double_tap(bbox: list[float]) -> str:
        """Double tap at the bbox center.

        Use for: zooming maps/photos/web pages, selecting a word.
        bbox: [left, top, right, bottom] as 0-1 decimals.
        """
        return physiclaw.double_tap(bbox)

    @mcp.tool()
    def long_press(bbox: list[float]) -> str:
        """Long press at the bbox center. ~1.2s hold.

        Use for: context menus, edit mode, paste, rearranging icons.
        bbox: [left, top, right, bottom] as 0-1 decimals.
        """
        return physiclaw.long_press(bbox)

    # ─── Swipe ───────────────────────────────────────────────

    @mcp.tool()
    def swipe(bbox: list[float], direction: str, size: str = "medium") -> str:
        """Stylus slides across a region. The direction is the stylus motion.

        Use for: scrolling (swipe up to scroll down), dismissing cards,
        changing pages, revealing list-item actions.
        bbox:      [left, top, right, bottom] as 0-1 decimals — region to swipe in.
        direction: 'up' | 'down' | 'left' | 'right' — stylus motion direction.
        size:      'small' | 'medium' | 'large'.
        """
        return physiclaw.swipe(bbox, direction, size)

    # ─── Navigate ────────────────────────────────────────────

    @mcp.tool()
    def home_screen() -> str:
        """Go to the home screen. iPhone swipe-up-from-bottom gesture.

        Use for: exiting any app, returning to the launcher.
        """
        return physiclaw.home_screen()

    @mcp.tool()
    def go_back() -> str:
        """Go back one screen. iPhone swipe-from-left-edge gesture.

        Use for: navigating back in apps with a nav stack.
        """
        return physiclaw.go_back()

    # ─── Text ────────────────────────────────────────────────

    @mcp.tool()
    def send_to_clipboard(text: str) -> str:
        """Copy text into the phone's clipboard.

        Use for: entering text into a field — on-screen typing is slow.
        After this returns, paste with: long_press(field_bbox) → tap "Paste".
        text: the string to put on the clipboard.
        """
        return physiclaw.send_to_clipboard(text)
