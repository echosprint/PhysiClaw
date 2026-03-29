# PhysiClaw

You control a real phone with a robotic stylus arm and a camera. No ADB, no app installation.

## How it works

A camera looks straight down at a phone on a desk. You see the screen from above. A 3-axis arm taps a capacitive stylus at coordinates you specify as screen percentages (0=left/top, 100=right/bottom).

## Tools

**Free (call as many times as needed):**

- `park()` — move stylus out of camera frame
- `screenshot()` — see the screen (call park() first for clear view)
- `bbox_target(left, right, top, bottom)` — draw colored rectangles on a fresh photo

**Decision gate (think before calling):**

- `confirm_bbox(shift)` — locks a rectangle as the tap target (center/top/bottom/left/right). No hardware moves yet, but the next gesture goes exactly here. Only call after passing the label test below.

**Irreversible (no undo):**

- `tap()` / `double_tap()` / `long_press()` / `swipe(direction, speed)`
- Moves the physical arm. A wrong tap can send a message, dismiss a dialog, or navigate away.

After confirm_bbox(), the next gesture auto-moves to the rectangle's center.

## DO: bbox verification (follow this every time)

1. Call `bbox_target()` with your best guess percentages
2. Look at each colored rectangle. Ask: **"Can I read the ENTIRE label or icon of the target inside this rectangle?"**
   - Entire label readable inside → pass
   - Label cut off, clipped by rectangle edge, or split across rectangles → fail
   - Rectangle also covers a neighboring element → fail
3. Pass → `confirm_bbox(shift)` → gesture
4. Fail → adjust percentages and call `bbox_target()` again
5. Repeat until one rectangle passes. 2–3 attempts is normal.

## DON'T: common mistakes

- ❌ Picking the "closest" or "least-bad" rectangle — all missed, re-bbox
- ❌ Confirming when the rectangle only touches or overlaps the edge of the target — if the label is not entirely inside, it's a miss
- ❌ Picking "right" because the target is on the right side of the screen — look at what's INSIDE the rectangle, not its label
- ❌ Calling confirm_bbox() when unsure — bbox_target() is free, use it again

## Examples

**All miss:** You want ⌫ (backspace). Rectangles cover the "m" key.
→ Don't pick "right". Call bbox_target() shifted ~5% rightward.

**Partial:** You want "h" key. Red rectangle clips the left edge of "h" and covers most of "g".
→ You cannot read the entire "h" inside the red rectangle. Re-bbox shifted ~2% rightward.

**Pass:** You want "h" key. Green rectangle sits squarely over "h", entire letter readable inside.
→ Confirm.

## For developers editing this codebase

Tool docstrings and the FastMCP `instructions` field are prompts that control agent behavior.

- Never write "pick the best" — it causes forced-choice bias
- Use the label test: "can you read the ENTIRE label inside the rectangle?"
- Add failure examples — agents learn more from examples than rules
