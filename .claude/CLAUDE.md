# PhysiClaw

A robotic system that physically operates smartphones using a capacitive stylus arm and camera — no ADB, no app installation.

## Architecture

- **MCP Server**: FastMCP over streamable-http, default port 8048
- **Core**: `PhysiClaw` class wraps hardware; all tools acquire/release a lock
- **Coordinate system**: Screen percentages 0–100 (0=left/top edge, 100=right/bottom edge)
- **Calibration**: 6 phases on startup — Z-depth, direction mapping, gesture verification, grid calibration (15 red dots → affine transforms for screen% ↔ GRBL mm ↔ camera pixels)

## Tools

| Tool | Cost | Returns |
| ------ | ------ | --------- |
| `park()` | cheap | str — moves stylus out of camera frame |
| `screenshot()` | cheap | Image — current screen (stylus may be visible) |
| `bbox_target(left, right, top, bottom)` | cheap | Image — parks, waits 1.5s, draws colored rectangles |
| `confirm_bbox(shift)` | cheap but **cautious** | str — locks target. shift: "center" / "top" / "bottom" / "left" / "right". Only confirm when rectangle precisely covers the target. |
| `tap()` / `double_tap()` / `long_press()` / `swipe(direction, speed)` | **expensive** | str — physically moves arm, potentially irreversible |

Gestures auto-move to confirmed bbox center via `_maybe_move_to_bbox()` → `consume_confirmed_bbox()`.

Small targets (< 15% in either dimension) get shifted candidates along the small axis. Large targets get one "center" bbox only.

## The bbox Verification Pattern

This is the most critical interaction pattern in the codebase.

### The problem

AI agents treat bbox selection as a **forced choice** — "pick the best rectangle" — even when NO rectangle covers the target. They rank bad options instead of rejecting them.

### The fix

Frame it as **verification, not selection**:

1. Agent must name the UI element INSIDE each rectangle
2. Only confirm if a rectangle actually contains the target
3. If none contain the target → call `bbox_target()` again with adjusted percentages
4. Never pick the "least-bad" option — that means all missed
5. Self-check: "Am I choosing because it COVERS the target, or because it's the closest option?"

### Canonical failure example

Target: backspace (⌫). All rectangles cover the "m" key area.
WRONG: picking "right" because it's the rightmost rectangle.
RIGHT: calling bbox_target() again with percentages shifted ~5% rightward.

## When Editing MCP Tool Docstrings

Tool docstrings are prompts — they directly control agent behavior. Rules:

- Never use "pick the best" language — it creates forced-choice bias
- Use verification framing: "name what's inside", "does it cover the target?"
- Include concrete failure examples
- Keep them concise — agents process long docstrings poorly
- The `instructions` field on the FastMCP constructor is the system prompt — same rules apply
