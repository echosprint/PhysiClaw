"""
PhysiClaw MCP Server — gives AI agents a physical finger to operate any phone.

Launch:
    uv run python -m physiclaw.server [--port 8048]

Connect (Claude Desktop / Claude Code / OpenClaw):
    {
      "mcpServers": {
        "physiclaw": {
          "type": "streamable-http",
          "url": "http://localhost:8048/mcp"
        }
      }
    }

Hardware is initialized lazily on first tool call.
"""

import cv2
from mcp.server.fastmcp import FastMCP, Image

from physiclaw.camera import Camera
from physiclaw.vision import PhoneDetector
from physiclaw.stylus_arm import StylusArm

mcp = FastMCP(
    "physiclaw",
    instructions="""PhysiClaw gives you a physical finger (robotic stylus arm) and eyes (cameras) to operate any phone.

You control a real phone sitting on a desk — a top camera sees the screen, a side camera checks stylus alignment, and a 3-axis arm moves and taps a capacitive stylus.

## Operation cycle

1. screenshot_top() — see what's on the phone screen (stylus must be parked/out of frame)
2. Decide what to tap/interact with
3. move(direction, distance) — position the stylus over the target
4. screenshot_side() — verify the stylus tip is aligned with your target
5. If misaligned → move() again to adjust, then screenshot_side() again
6. If aligned → tap() / double_tap() / long_press() / swipe()
7. screenshot_top() — verify the result, then continue

## Key principles

- You never specify pixel coordinates. Use direction + distance level to navigate.
- Always verify alignment with screenshot_side() before tapping.
- Always park the stylus out of frame before using screenshot_top() so it doesn't occlude the screen.
- Use move() to incrementally approach the target — start with 'large' or 'medium', refine with 'small' and 'nudge'.
- After any tap/swipe, take screenshot_top() to confirm the expected screen change happened before proceeding.
""",
)


# ─── Lazy hardware singleton ────────────────────────────────

class PhysiClaw:
    """Lazy-initialized hardware. First access triggers full setup."""

    def __init__(self):
        self._arm: StylusArm | None = None
        self._top_cam: Camera | None = None
        self._side_cam: Camera | None = None
        self._ready = False

    def _init(self):
        if self._ready:
            return

        # 1. Connect GRBL arm
        self._arm = StylusArm()
        self._arm.setup()

        # 2. Open cameras and identify top vs side
        detector = PhoneDetector()
        cameras = detector.identify_cameras()

        if 'top' not in cameras:
            raise RuntimeError("Top camera not found — is the phone under the camera?")

        self._top_cam = Camera(cameras['top'])
        if 'side' in cameras:
            self._side_cam = Camera(cameras['side'])

        # 3. Load calibration
        self._arm.load_calibration()

        self._ready = True

    @property
    def arm(self) -> StylusArm:
        self._init()
        assert self._arm is not None
        return self._arm

    @property
    def top_cam(self) -> Camera:
        self._init()
        assert self._top_cam is not None
        return self._top_cam

    @property
    def side_cam(self) -> Camera:
        self._init()
        if self._side_cam is None:
            raise RuntimeError("Side camera not available")
        return self._side_cam

    def shutdown(self):
        if self._arm:
            self._arm._pen_up()
            self._arm._fast_move(0, 0)
            self._arm.close()
        if self._top_cam:
            self._top_cam.close()
        if self._side_cam:
            self._side_cam.close()


physiclaw = PhysiClaw()


# ─── Helper ─────────────────────────────────────────────────

def _camera_to_image(cam: Camera) -> Image:
    """Capture a frame and return as MCP Image."""
    frame = cam.snapshot()
    if frame is None:
        raise RuntimeError("Camera capture failed")
    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Image(data=jpeg.tobytes(), format="jpeg")


# ─── Tools ──────────────────────────────────────────────────

@mcp.tool()
def screenshot_top() -> Image:
    """Take a screenshot from the top camera looking straight down at the phone screen.

    Use this to read screen content — text, icons, buttons, app UI, notifications, etc.
    The stylus should be out of frame (call move() away or ensure it's parked) before
    taking this screenshot, otherwise the stylus will occlude part of the screen.

    Typical usage: call this at the start of each action cycle to understand what's on
    screen, and again after tap/swipe to verify the result.
    """
    return _camera_to_image(physiclaw.top_cam)


@mcp.tool()
def screenshot_side() -> Image:
    """Take a screenshot from the side camera at ~45° angle, showing the stylus tip and the phone screen.

    Use this to check whether the stylus is aligned with the intended target before tapping.
    You can see the stylus tip position relative to screen elements from this perspective.

    If the stylus is not over the right spot, call move() to adjust, then screenshot_side() again.
    Only proceed with tap/swipe once alignment is confirmed.
    """
    return _camera_to_image(physiclaw.side_cam)


@mcp.tool()
def move(direction: str, distance: str = "medium") -> str:
    """Move the stylus relative to its current position without touching the screen.

    The stylus hovers above the phone — this only changes the XY position, it does NOT tap.
    Use screenshot_side() after moving to verify alignment before tapping.

    Strategy: start with 'large' or 'medium' to get close, then refine with 'small' or 'nudge'.
    Diagonal directions are supported for efficient movement.

    Args:
        direction: 'up', 'down', 'left', 'right',
                   'up-left', 'up-right', 'down-left', 'down-right'
        distance: 'large' (~20mm, half the screen),
                  'medium' (~8mm, a few icons),
                  'small' (~3mm, one icon),
                  'nudge' (~1mm, fine-tune)
    """
    physiclaw.arm.move(direction, distance)
    return f"Moved {direction} {distance}"


@mcp.tool()
def tap() -> str:
    """Single tap at current stylus position — like a finger tap on the screen.

    Use for: pressing buttons, selecting items, opening apps, following links, dismissing dialogs.
    The stylus descends, touches the screen briefly (~80ms), and lifts back up.

    Always confirm alignment with screenshot_side() before tapping.
    After tapping, use screenshot_top() to verify the expected screen change.
    """
    physiclaw.arm.tap()
    return "Tapped"


@mcp.tool()
def double_tap() -> str:
    """Double tap at current stylus position — two quick taps in succession.

    Use for: zooming in (maps, photos, web pages), selecting a word in text, or any
    UI element that responds to double-tap.
    """
    physiclaw.arm.double_tap()
    return "Double tapped"


@mcp.tool()
def long_press() -> str:
    """Long press at current stylus position — holds contact for ~1.2 seconds.

    Use for: opening context menus, entering edit/selection mode, triggering drag-and-drop,
    selecting text, rearranging home screen icons, or any action that requires a sustained press.
    """
    physiclaw.arm.long_press()
    return "Long pressed"


@mcp.tool()
def swipe(direction: str, speed: str = "medium") -> str:
    """Swipe from current position in a cardinal direction — the stylus touches down, slides, and lifts.

    Use for: scrolling content, switching pages, pulling down notifications, dismissing items,
    unlocking the phone (swipe up), navigating between screens.

    The swipe travels ~15mm in the given direction. The stylus position will change after swiping.

    Args:
        direction: 'up', 'down', 'left', 'right'
        speed: 'slow' (gentle scroll, careful drag),
               'medium' (normal swipe, ~100mm/s),
               'fast' (fling, page switch, quick dismiss)
    """
    physiclaw.arm.swipe(direction, speed)
    return f"Swiped {direction} {speed}"


# ─── Lifecycle ──────────────────────────────────────────────

import atexit
atexit.register(physiclaw.shutdown)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PhysiClaw MCP Server")
    parser.add_argument("--port", type=int, default=8048)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    print(f"PhysiClaw MCP server starting on http://{args.host}:{args.port}/mcp")
    print("Hardware will initialize on first tool call.")
    mcp.run(transport="streamable-http", host=args.host, port=args.port)
