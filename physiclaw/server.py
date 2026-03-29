"""
PhysiClaw MCP Server — gives AI agents a physical finger to operate any phone.

Launch:
    uv run python -m physiclaw.server [--port 8048] [--verbose]

Connect (Claude Desktop / Claude Code / OpenClaw):
    {
      "mcpServers": {
        "physiclaw": {
          "type": "streamable-http",
          "url": "http://localhost:8048/mcp"
        }
      }
    }

Hardware connects and calibrates on startup.
"""

import argparse
import atexit
import logging

from mcp.server.fastmcp import FastMCP, Image

from physiclaw.core import PhysiClaw

# ─── MCP server ─────────────────────────────────────────────

mcp = FastMCP(
    "physiclaw",
    instructions="""PhysiClaw gives you a physical finger (robotic stylus arm) and an eye (camera) to operate any phone.

You control a real phone sitting on a desk — a camera sees the screen from directly above, and a 3-axis arm moves and taps a capacitive stylus.

## How it works

A camera is mounted directly above the phone, looking straight down. You see the screen from above. The stylus taps the screen at coordinates you specify as screen percentages.

## Operation cycle

1. park() + screenshot() — see the full phone screen (stylus parked out of frame)
2. Identify the target's position as screen percentages (0=left/top edge, 100=right/bottom edge)
3. bbox_target(left, right, top, bottom) — draws colored rectangles on a fresh photo
4. For large targets: one green rectangle. Verify it covers the target.
   For small targets (like keyboard keys): multiple colored rectangles appear,
   shifted along the small dimension(s). Pick the one that best covers the target.
   If none fit well, call bbox_target() again with corrected percentages.
5. confirm_bbox(shift) — lock in the chosen rectangle ("center", "top", "bottom", "left", "right")
6. tap() / double_tap() / long_press() / swipe() — executes at the bbox center
7. park() + screenshot() — verify the result

## CRITICAL

bbox_target() is cheap — it just takes a photo and draws rectangles on it. But tap() is expensive — it moves the mechanical arm physically, which is slow. A wrong
tap can send a wrong message, transfer the wrong amount, or trigger an irreversible action.
Always iterate bbox_target() until a rectangle precisely covers the target before confirming.
Never rush to confirm an imprecise bbox.
""",
)

# ─── Tools ──────────────────────────────────────────────────

@mcp.tool()
def screenshot() -> Image:
    """Take a screenshot of the phone screen.

    Use this to read screen content, check stylus position, or verify results.
    The stylus may be visible in the frame — call park() first if you need
    an unobstructed view of the screen.
    """
    physiclaw.acquire()
    try:
        frame = physiclaw.screenshot()
        return Image(data=physiclaw.frame_to_jpeg(frame), format="jpeg")
    finally:
        physiclaw.release()


@mcp.tool()
def park() -> str:
    """Park the stylus out of the camera frame so it doesn't occlude the screen.

    Call this before screenshot() when you need a clear view of the full screen
    (e.g. to read text or identify UI elements). The stylus moves 100mm away
    and will need to be repositioned with move() afterward.
    """
    physiclaw.acquire()
    try:
        physiclaw.park()
        return "Stylus parked out of frame"
    finally:
        physiclaw.release()


@mcp.tool()
def bbox_target(left: int, right: int, top: int, bottom: int) -> Image:
    """Target a screen region by bounding box using screen percentages (0-100).

    Takes a fresh screenshot and draws colored rectangles at the specified position.

    For large targets: one green rectangle labeled "center".
    For small targets (< 15% in either dimension): multiple colored rectangles
    shifted along the small dimension(s), each labeled with its shift direction.
    Pick the rectangle that best covers the target.

    IMPORTANT: Do NOT confirm until a rectangle precisely covers the target.
    bbox_target() is cheap — it just takes a photo and draws rectangles on it. But tap() is expensive —
    a wrong tap is hard to undo and wastes time. So iterate bbox_target() as
    many times as needed until a rectangle matches, then confirm and tap.
    Confirming an imprecise bbox risks tapping the wrong button — sending a wrong
    message, transferring the wrong amount, or triggering an irreversible action.

    Args:
        left: left edge (0=left edge of screen, 100=right edge)
        right: right edge
        top: top edge (0=top of screen, 100=bottom)
        bottom: bottom edge
    """
    import time
    physiclaw.acquire()
    try:
        physiclaw.park()
        time.sleep(1.5)  # let arm settle after parking
        physiclaw.set_pending_bbox(left, right, top, bottom)
        frame = physiclaw.screenshot_with_bboxes()
        return Image(data=physiclaw.frame_to_jpeg(frame), format="jpeg")
    finally:
        physiclaw.release()


@mcp.tool()
def confirm_bbox(shift: str = "center") -> str:
    """Confirm a bounding box from the last bbox_target() call.

    For large targets, just call confirm_bbox() or confirm_bbox("center").
    For small targets, pick the shifted variant that best covers the target:
      "center" — the original bbox (green)
      "top"    — shifted up (red)
      "bottom" — shifted down (blue)
      "left"   — shifted left (yellow)
      "right"  — shifted right (magenta)

    IMPORTANT: Only confirm when a rectangle precisely covers the target.
    If none fit, call bbox_target() again with corrected percentages instead.
    Confirming an imprecise bbox wastes time — the tap will miss.

    After confirmation, the next gesture (tap, double_tap, long_press, swipe)
    will auto-move to the bbox center before executing.

    Args:
        shift: "center", "top", "bottom", "left", or "right"
    """
    physiclaw.acquire()
    try:
        physiclaw.confirm_bbox(shift)
        return f"Bbox '{shift}' confirmed — next gesture will target this location"
    finally:
        physiclaw.release()


def _maybe_move_to_bbox():
    """If a bbox is confirmed, move arm to its center and clear it."""
    bbox = physiclaw.consume_confirmed_bbox()
    if bbox is not None:
        physiclaw.move_to_bbox_center(bbox)


@mcp.tool()
def tap() -> str:
    """Single tap — like a finger tap on the screen.

    Use for: pressing buttons, selecting items, opening apps, following links, dismissing dialogs.
    Call bbox_target() + confirm_bbox() first to set the target location.
    After tapping, use park() + screenshot() to verify the result.
    """
    physiclaw.acquire()
    try:
        _maybe_move_to_bbox()
        physiclaw.arm.tap()
        return "Tapped"
    finally:
        physiclaw.release()


@mcp.tool()
def double_tap() -> str:
    """Double tap — two quick taps in succession.

    Use for: zooming in (maps, photos, web pages), selecting a word in text.
    Call bbox_target() + confirm_bbox() first to set the target location.
    """
    physiclaw.acquire()
    try:
        _maybe_move_to_bbox()
        physiclaw.arm.double_tap()
        return "Double tapped"
    finally:
        physiclaw.release()


@mcp.tool()
def long_press() -> str:
    """Long press — holds contact for ~1.2 seconds.

    Use for: opening context menus, entering edit/selection mode, selecting text,
    rearranging home screen icons, or any action that requires a sustained press.
    Call bbox_target() + confirm_bbox() first to set the target location.
    """
    physiclaw.acquire()
    try:
        _maybe_move_to_bbox()
        physiclaw.arm.long_press()
        return "Long pressed"
    finally:
        physiclaw.release()


@mcp.tool()
def swipe(direction: str, speed: str = "medium") -> str:
    """Swipe in a cardinal direction — the stylus touches down, slides, and lifts.

    Use for: scrolling content, switching pages, pulling down notifications,
    unlocking the phone, navigating between screens.
    Call bbox_target() + confirm_bbox() first to set the starting position.

    Args:
        direction: 'top', 'bottom', 'left', 'right'
        speed: 'slow' (gentle scroll), 'medium' (normal), 'fast' (fling)
    """
    physiclaw.acquire()
    try:
        _maybe_move_to_bbox()
        physiclaw.arm.swipe(direction, speed)
        return f"Swiped {direction} {speed}"
    finally:
        physiclaw.release()


# ─── Start ───────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="PhysiClaw MCP Server")
parser.add_argument("--port", type=int, default=8048)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--verbose", "-v", action="store_true",
                    help="Show detailed debug output")
args = parser.parse_args()

logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO,
    format="%(message)s",
)

physiclaw = PhysiClaw()
atexit.register(physiclaw.shutdown)

mcp.settings.host = args.host
mcp.settings.port = args.port

log = logging.getLogger(__name__)
log.info(f"PhysiClaw MCP server on http://{args.host}:{args.port}/mcp")
mcp.run(transport="streamable-http")
