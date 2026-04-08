"""
PhysiClaw MCP Server — wires the FastMCP instance, hardware singleton,
and all tool/route registrations.

Started by physiclaw.main. The server starts instantly without hardware.
Run /setup to connect and calibrate.
"""

import logging

from mcp.server.fastmcp import FastMCP

from physiclaw.annotation import AnnotationState
from physiclaw.bridge import BridgeState, CalibrationState, PageState
from physiclaw.core import PhysiClaw

log = logging.getLogger(__name__)

# ─── MCP server ─────────────────────────────────────────────

mcp = FastMCP(
    "physiclaw",
    instructions="""PhysiClaw gives you a physical finger (robotic stylus arm) and an eye (camera) to operate any phone.

You control a real phone sitting on a desk — a camera sees the screen from directly above, and a 3-axis arm moves and taps a capacitive stylus.

## Before every tap, classify the target

**Fixed UI element** (button, icon, nav control, text field, keyboard key — same position every visit):
- In preset (.claude/ui-presets/) → use preset coordinates with bbox_target(bbox).
- Not in preset → propose_bboxes() → wait_for_confirmation() → save to preset.
- You CANNOT guess coordinates for fixed UI. Your estimates are unreliable.

**Dynamic content** (list item, menu entry, product card — large, changes every visit):
- Visual targeting OK: grid_overlay() → bbox_target(bbox) → label test → confirm_bbox() → tap.

## Operation cycle (dynamic content / preset path)

1. Check UI presets — if the target has known coordinates, use them directly.
2. If no preset: park() + camera_view(). Optionally detect_elements() to find icons + text with coordinates. Or grid_overlay() to estimate manually.
3. bbox_target(bbox) — bbox = [left, top, right, bottom] as 0-1 decimals.
4. **Label test:** name the element INSIDE each rectangle.
   - Covers the target → confirm_bbox()
   - Misses → call bbox_target() with corrected coordinates. 2-3 attempts is normal.
5. tap() / double_tap() / long_press() / swipe() — executes at the bbox center.
6. park() + camera_view() — verify the result.

## Propose-confirm cycle (fixed UI without preset)

1. park() + camera_view() — reason about visible elements.
2. propose_bboxes([{"bbox": [l,t,r,b], "label": "..."}]) — sends guesses to /annotate.
3. Tell user to review and confirm at /annotate.
4. wait_for_confirmation() — blocks until user confirms.
5. Use confirmed coordinates: bbox_target(bbox) → confirm_bbox() → gesture.
6. Save to preset for future autonomous use.

## Setup

All tools require hardware to be set up first. If you get "Hardware not set up",
tell the user to run /setup. Do not attempt to call setup endpoints yourself.

## CRITICAL

- bbox_target() is cheap (just a photo). tap() is expensive (physical arm, irreversible).
- Never guess coordinates for fixed UI elements — propose and let the user confirm.
- Before confirming, ask: "Am I choosing this because it COVERS the target, or because it's the closest option?" If closest → reject, re-bbox.
""",
)

# ─── Singletons ─────────────────────────────────────────────

physiclaw = PhysiClaw()
_bridge = BridgeState()
_calib = CalibrationState()
_phone = PageState(_bridge, _calib)
_ann = AnnotationState()


def shutdown():
    """Clean up hardware resources."""
    physiclaw.shutdown()


# ─── Register tools and routes ──────────────────────────────
# Late imports: each submodule imports from this __init__ for the `mcp`
# singleton and friends, so they must be imported after the singletons
# above are constructed. The E402 suppression is intentional.

from physiclaw.server.tools import register as _register_tools  # noqa: E402
from physiclaw.server.bridge import register as _register_bridge  # noqa: E402
from physiclaw.server.annotation import register as _register_annotation  # noqa: E402
from physiclaw.server.hardware import register as _register_hardware  # noqa: E402
from physiclaw.server.calibration import register as _register_calibration  # noqa: E402

_register_tools(mcp, physiclaw, _bridge, _calib, _ann)
_register_bridge(mcp, physiclaw, _bridge, _calib, _phone)
_register_annotation(mcp, physiclaw, _ann)
_register_hardware(mcp, physiclaw)
_register_calibration(mcp, physiclaw, _bridge, _calib, _phone)


__all__ = ["mcp", "physiclaw", "shutdown"]
