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

- `confirm_bbox(shift)` — locks a rectangle as the tap target (center/top/bottom/left/right). No hardware moves yet, but the next gesture goes exactly here. Only call when the rectangle **fully** covers the target.

**Irreversible (no undo):**

- `tap()` / `double_tap()` / `long_press()` / `swipe(direction, speed)`
- Moves the physical arm. A wrong tap can send a message, dismiss a dialog, or navigate away.

After confirm_bbox(), the next gesture auto-moves to the rectangle's center.

## DO: bbox verification (follow this every time)

1. Call `bbox_target()` with your best guess percentages
2. Look at each colored rectangle. Ask: **"What UI element is inside this rectangle?"**
3. If the target is **fully** inside a rectangle → `confirm_bbox(shift)` → gesture
4. If not → adjust percentages and call `bbox_target()` again
5. Repeat until one rectangle fully covers the target. 2–3 attempts is normal.

## DON'T: common mistakes

- ❌ Picking the "closest" or "least-bad" rectangle — that means all missed, re-bbox
- ❌ Confirming partial coverage (rectangle clips the target or spans two elements) — re-bbox
- ❌ Picking "right" because the target is on the right side of the screen — look at what's INSIDE the rectangle, not its label
- ❌ Calling confirm_bbox() when unsure — bbox_target() is free, use it again

## Examples

**All miss:** You want ⌫ (backspace). Rectangles cover the "m" key.
→ Don't pick "right". Call bbox_target() shifted ~5% rightward.

**Partial:** You want "Send". Rectangle covers half of Send and half of the next icon.
→ Don't confirm. Call bbox_target() shifted ~2% to center on Send.

## For developers editing this codebase

Tool docstrings and the FastMCP `instructions` field are prompts that control agent behavior.

- Never write "pick the best" — it causes forced-choice bias
- Write "name what's inside", "fully covers"
- Add failure examples — agents learn more from examples than rules
