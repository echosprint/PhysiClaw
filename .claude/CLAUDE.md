# PhysiClaw

You control a real phone with a robotic stylus arm and a camera. No ADB, no app installation.

## How it works

A camera looks straight down at a phone on a desk. You see the screen from above. A 3-axis arm taps a capacitive stylus at coordinates you specify as 0-1 decimals (0=left/top, 1=right/bottom).

## Tools

**Free (call as many times as needed):**

- `park()` — move stylus out of camera frame
- `screenshot()` — see the screen (call park() first for clear view)
- `detect_elements()` — detect all icons and text on screen. Returns two annotated images (icon boxes + OCR boxes) and a text listing with bounding boxes as 0-1 decimals. Use the coordinates for bbox_target.
- `grid_overlay(density, color)` — show screen with numbered grid lines (0-1 scale). density: "sparse", "normal" (default), or "dense" (0.05 spacing). Fallback when detect_elements() misses the target.
- `bbox_target(left, top, right, bottom)` — draw colored rectangles on a fresh photo

**Decision gate (think before calling):**

- `confirm_bbox(shift)` — locks a rectangle as the tap target (center/top/bottom/left/right). No hardware moves yet, but the next gesture goes exactly here. Only call after passing the label test below.

**Irreversible (no undo):**

- `tap()` / `double_tap()` / `long_press()` / `swipe(direction, speed)`
- Moves the physical arm. A wrong tap can send a message, dismiss a dialog, or navigate away.

After confirm_bbox(), the next gesture auto-moves to the rectangle's center.

## DO: targeting workflow (follow this every time)

1. `detect_elements()` — detect icons and text, get bounding boxes as 0-1 decimals
2. If the target is in the list, use its coordinates for step 3.
   If not, use `grid_overlay()` to estimate coordinates manually.
3. `bbox_target(left, top, right, bottom)` — draw rectangles at those coordinates
4. **Label test:** For each rectangle, ask: **"Can I read the ENTIRE label or icon of the target inside this rectangle?"**
   - Entire label readable inside → pass
   - Label cut off, clipped by rectangle edge, or split across rectangles → fail
   - Rectangle also covers a neighboring element → fail
5. Pass → `confirm_bbox(shift)` → gesture
6. Fail → adjust coordinates and call `bbox_target()` again
7. Repeat steps 3–6 until one rectangle passes. 2–3 attempts is normal.

## DON'T: common mistakes

- ❌ Picking the "closest" or "least-bad" rectangle — all missed, re-bbox
- ❌ Confirming when the rectangle only touches or overlaps the edge of the target — if the label is not entirely inside, it's a miss
- ❌ Picking "right" because the target is on the right side of the screen — look at what's INSIDE the rectangle, not its label
- ❌ Calling confirm_bbox() when unsure — bbox_target() is free, use it again

## Examples

**All miss:** You want ⌫ (backspace). Rectangles cover the "m" key.
→ Don't pick "right". Call bbox_target() shifted ~0.05 rightward.

**Partial:** You want "h" key. Red rectangle clips the left edge of "h" and covers most of "g".
→ You cannot read the entire "h" inside the red rectangle. Re-bbox shifted ~0.02 rightward.

**Pass:** You want "h" key. Green rectangle sits squarely over "h", entire letter readable inside.
→ Confirm.

## UI Presets

Phone UI element annotations are stored in `.claude/ui-presets/` directory as markdown files, one per app.

### Using presets during operation

Before starting the bbox_target() targeting workflow, check if the target element exists in a preset file:

1. Read the relevant app's preset file (e.g., `.claude/ui-presets/wechat.md`)
2. Match the current page by its fingerprint description
3. If the target element is listed, use its 位置 coordinates directly with bbox_target()
4. If not found, fall back to the normal detect_elements() → bbox_target() workflow

Preset coordinates are 0-1 decimals in [left, top, right, bottom] order, same format as bbox_target().

### Annotation workflow

When the user asks to annotate UI elements:

1. Ensure the user has drawn boxes at [annotate](http://localhost:8048/annotate)
2. Call `get_user_annotations()` to get the screenshot with drawn boxes and coordinates
3. Look at the image and identify: what app, what page, and what each boxed element is
4. List your findings and ask the user to confirm
5. On confirmation, write the preset markdown file to `.claude/ui-presets/{app_slug}.md`
6. Annotations are cleared automatically after `get_user_annotations()` returns

## For developers editing this codebase

Tool docstrings and the FastMCP `instructions` field are prompts that control agent behavior.

- Never write "pick the best" — it causes forced-choice bias
- Use the label test: "can you read the ENTIRE label inside the rectangle?"
- Add failure examples — agents learn more from examples than rules
