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

## Camera and stylus layout

- The camera is mounted directly above the phone screen, looking straight down. What you see in screenshots is exactly what's on the screen, from directly above.
- The stylus is L-shaped: a horizontal arm extends from the gantry, and at the end it bends down to a conductive tip that touches the screen.
- From the top-down camera view, the stylus appears as a thin line (the horizontal arm) ending in a small round circle (the tip pointing down at the screen).
- Alignment check: from the top-down view, when the tip is directly above a target (icon, button, text), the tip HIDES/COVERS the target — you can barely see the target because the tip is blocking the camera's view. If you can still clearly see the target, the tip is NOT over it yet. Keep moving until the tip occludes the target.
- When parked, the stylus moves out of frame so you get a clear, unobstructed view of the full screen.

## Operation cycle

1. park() + screenshot() — park the stylus away, then see the full phone screen
2. Decide what to tap/interact with
3. move(direction, distance) — position the stylus over the target
4. screenshot() — check if the stylus tip is on top of the target
5. If not overlapping → move() to adjust, screenshot() again
6. If the tip covers the target → tap() / double_tap() / long_press() / swipe()
7. park() + screenshot() — verify the result, then continue

## Key principles

- You never specify pixel coordinates. Use direction + distance level to navigate.
- Use park() before screenshot() when you need a clear, unobstructed view of the screen.
- Use screenshot() without park() to see where the stylus tip is relative to the target.
- If you can still clearly see the target icon/button, the tip is NOT aligned — keep moving. The tip is aligned only when it hides the target from view.
- Use move() to incrementally approach the target — start with 'large' or 'medium', refine with 'small' and 'nudge'.
- After any tap/swipe, park() + screenshot() to confirm the expected screen change happened.
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
def move(direction: str, distance: str = "medium") -> str:
    """Move the stylus relative to its current position without touching the screen.

    The stylus hovers above the phone — this only changes the XY position, it does NOT tap.
    Use screenshot() after moving to verify position before tapping.

    Strategy: start with 'large' or 'medium' to get close, then refine with 'small' or 'nudge'.
    Diagonal directions are supported for efficient movement.

    Args:
        direction: 'top', 'bottom', 'left', 'right',
                   'top-left', 'top-right', 'bottom-left', 'bottom-right'
        distance: 'large' (~20mm, half the screen),
                  'medium' (~8mm, a few icons),
                  'small' (~3mm, one icon),
                  'nudge' (~1mm, fine-tune)
    """
    physiclaw.acquire()
    try:
        physiclaw.arm.move(direction, distance)
        return f"Moved {direction} {distance}"
    finally:
        physiclaw.release()


@mcp.tool()
def tap() -> str:
    """Single tap at current stylus position — like a finger tap on the screen.

    Use for: pressing buttons, selecting items, opening apps, following links, dismissing dialogs.
    The stylus descends, touches the screen briefly (~80ms), and lifts back up.

    After tapping, use screenshot() to verify the expected screen change.
    """
    physiclaw.acquire()
    try:
        physiclaw.arm.tap()
        return "Tapped"
    finally:
        physiclaw.release()


@mcp.tool()
def double_tap() -> str:
    """Double tap at current stylus position — two quick taps in succession.

    Use for: zooming in (maps, photos, web pages), selecting a word in text, or any
    UI element that responds to double-tap.
    """
    physiclaw.acquire()
    try:
        physiclaw.arm.double_tap()
        return "Double tapped"
    finally:
        physiclaw.release()


@mcp.tool()
def long_press() -> str:
    """Long press at current stylus position — holds contact for ~1.2 seconds.

    Use for: opening context menus, entering edit/selection mode, triggering drag-and-drop,
    selecting text, rearranging home screen icons, or any action that requires a sustained press.
    """
    physiclaw.acquire()
    try:
        physiclaw.arm.long_press()
        return "Long pressed"
    finally:
        physiclaw.release()


@mcp.tool()
def swipe(direction: str, speed: str = "medium") -> str:
    """Swipe from current position in a cardinal direction — the stylus touches down, slides, and lifts.

    Use for: scrolling content, switching pages, pulling down notifications, dismissing items,
    unlocking the phone (swipe top), navigating between screens.

    The swipe travels ~15mm in the given direction. The stylus position will change after swiping.

    Args:
        direction: 'top', 'bottom', 'left', 'right'
        speed: 'slow' (gentle scroll, careful drag),
               'medium' (normal swipe, ~100mm/s),
               'fast' (fling, page switch, quick dismiss)
    """
    physiclaw.acquire()
    try:
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
