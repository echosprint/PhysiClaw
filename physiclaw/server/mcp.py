"""FastMCP instance construction.

Isolated from `physiclaw.server.__init__` so the agent-facing instructions
prompt has a single, focused home. The instance is imported and wired up
(tools, routes, singletons) by `physiclaw.server.__init__`.

The `instructions` field is delivered to the client once at the MCP
initialization handshake. Keep it focused on cross-tool reasoning:
mental model, tool-choice trade-offs, operating loop, coordinate
conventions, global safety, and setup gating. Per-tool mechanics
live in `@mcp.tool()` docstrings and are auto-delivered as tool
schemas — do not duplicate them here.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "physiclaw",
    instructions="""PhysiClaw gives you a physical finger (robotic stylus arm) and an eye (camera) to operate a real phone sitting on a desk. The camera looks down at the phone from above, and a 3-axis arm moves and taps a capacitive stylus on its screen.

## Mental model: See → Act

Every task reduces to: see the screen, pick a target, do something.

When a tool takes a `bbox` argument, it's `[left, top, right, bottom]` as 0-1 decimals on the phone screen (0 = left/top edge, 1 = right/bottom edge).

## Seeing the screen

Pick the cheapest tool that answers your question:

| Tool           | Time | Output                                     |
|----------------|------|------------------------------------------- |
| `scan()`       | ~1s  | JSON list of text elements of camera image |
| `peek()`       | ~3s  | Raw camera image — rough, may have glare   |
| `screenshot()` | ~12s | Pixel-perfect image + detected UI bboxes   |

Time is wall clock: hardware action + your inference on the returned content.

`scan()` and `peek()` are for orientation and verification. `screenshot()` is for planning actions — use its detected bboxes, never eyeball coordinates from a `peek()`.

## Operating loop

1. **Orient.** `scan()` or `peek()` to see what's on screen.
2. **Plan.** When you need a target to act on, call `screenshot()` and pick a bbox from its JSON list.
3. **Act.** Call a gesture tool.
4. **Verify.** `scan()` or `peek()` again. If the screen did not change, retry the same call a few times — the capacitive stylus occasionally fails to register. If it still does not change, the bbox was likely wrong; re-examine with `screenshot()`.

## Safety

Wrong taps on a real phone are irreversible — a bad coordinate can send a message, transfer money, or trigger an action you can't undo.

## Setup

All tools require hardware to be set up first. If a tool returns "Hardware not set up", tell the user to run `/setup`.
""",
)
