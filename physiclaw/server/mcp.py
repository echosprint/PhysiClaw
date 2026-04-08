"""FastMCP instance construction.

Isolated from `physiclaw.server.__init__` so the agent-facing instructions
prompt has a single, focused home. The instance is imported and wired up
(tools, routes, singletons) by `physiclaw.server.__init__`.
"""

from mcp.server.fastmcp import FastMCP

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
