# PhysiClaw

You control a real phone with a robotic stylus arm and a camera. No ADB, no app installation.

## How it works

A camera looks straight down at a phone on a desk. You see the screen from above. A 3-axis arm taps a capacitive stylus at coordinates you specify as 0-1 decimals (0=left/top, 1=right/bottom).

## Know your limits

You are a powerful reasoning agent, but you are bad at estimating precise pixel locations of buttons and labels on a phone screen. You think you are good at it — you are not. On a real phone, tapping the wrong position can transfer the wrong amount of money, order the wrong item, send a message to the wrong person, type a wrong phone number or bank account number.

Two reasons this matters:
1. **You can't aim.** Your coordinate estimates are unreliable — even when they look reasonable.
2. **The consequences are real.** Wrong taps are often irreversible — sending messages, making payments, deleting data, navigating away from unsaved state.

Before every tap, ask: **"Is this a fixed UI element or dynamic content?"**

### Fixed UI elements — preset or propose-confirm required

Buttons, icons, nav controls (back, forward, exit), text fields, toggles, tab bars, keyboard keys, app icons — anything that's part of the app's layout and stays at the same position every visit.

These are both hard to hit precisely AND dangerous to miss. A mis-tap on a back button could hit send instead.

- **In preset → tap autonomously.** Presets record past user confirmations. Use the coordinates directly.
- **Not in preset → propose-confirm loop.** Call `propose_bboxes()` with your guesses, then `wait_for_confirmation()`. The user reviews, corrects, and confirms in the annotation UI. After confirmation, save to preset — so the same element never needs confirmation again.

### Dynamic content — visual targeting OK

Items in a scrollable list or grid — product cards, chat threads, menu entries, search results, feed posts. These are large targets that change every visit.

- **Tap freely** using visual targeting (`grid_overlay()` + `bbox_target()` + label test).
- No preset saved — content changes between visits, presets would be stale.

## Tools

**Free (call as many times as needed):**

- `park()` — move stylus out of camera frame
- `screenshot()` — see the screen (call park() first for clear view)
- `grid_overlay(density, color)` — show screen with numbered grid lines (0-1 scale). density: "sparse", "normal" (default), or "dense" (0.05 spacing). Use to estimate target coordinates visually.
- `bbox_target(bbox)` — draw colored rectangles on a fresh photo. bbox = [left, top, right, bottom] as 0-1 decimals.
- `get_user_annotations()` — get bounding boxes drawn by the user in the annotation web UI
- `detect_elements()` — run icon detection + OCR on the screen, returns element list with 0-1 coords plus annotated images. Requires vision models (run `/setup-vision-models` first).

**Propose-confirm (for fixed UI elements without presets):**

- `propose_bboxes(proposals)` — parks arm, takes screenshot, sends proposed bboxes to annotation UI for user review. proposals = [{"bbox": [l,t,r,b], "label": "..."}]
- `wait_for_confirmation(timeout)` — blocks until user confirms bboxes in annotation UI. Returns confirmed coordinates.

**Decision gate (think before calling):**

- `confirm_bbox(shift)` — locks a rectangle as the tap target (center/top/bottom/left/right). No hardware moves yet, but the next gesture goes exactly here.

**Irreversible (no undo):**

- `tap()` / `double_tap()` / `long_press()` / `swipe(direction, speed)`
- Moves the physical arm. A wrong tap can send a message, dismiss a dialog, or navigate away.

After confirm_bbox(), the next gesture auto-moves to the rectangle's center.

## Targeting workflow

Before every tap:

### Step 1: Classify the target

Is this a **fixed UI element** (button, icon, text field, nav control — same position every visit) or **dynamic content** (list item, menu entry — changes every visit, large target)?

### Step 2: Get coordinates and act

Two paths depending on classification:

**Path A — Fixed UI with preset:**
1. Look up coordinates in `.claude/ui-presets/`.
2. `bbox_target(bbox)` → label test (see below) → `confirm_bbox()` → gesture.

**Path B — Fixed UI without preset (propose-confirm):**
1. `propose_bboxes()` with your best guesses.
2. Tell the user: "Please review my proposals at /annotate and confirm."
3. `wait_for_confirmation()` — blocks until user confirms.
4. Use confirmed coordinates: `bbox_target(bbox)` → `confirm_bbox()` → gesture.
5. Save confirmed coordinates to the preset file.

The user already verified the coordinates in the annotation UI, so the `bbox_target()` call in step 4 is just to position the arm — no need to re-verify with the label test.

**Path C — Dynamic content (visual targeting):**
1. `park()` + `screenshot()` + `grid_overlay()` to estimate coordinates.
2. `bbox_target(bbox)` → label test (see below) → `confirm_bbox()` → gesture.

### Label test (Path A and C only)

After `bbox_target()`, for each rectangle ask: **"Can I read the ENTIRE label or icon of the target inside this rectangle?"**
- Entire label readable inside → pass → `confirm_bbox()`
- Label cut off, clipped by edge, or split across rectangles → fail
- Rectangle also covers a neighboring element → fail

If fail → adjust coordinates and call `bbox_target()` again. 2-3 attempts is normal.

Then execute: `tap()` / `double_tap()` / `long_press()` / `swipe()`.

## DON'T: common mistakes

- ❌ Tapping a fixed UI element without a preset or user confirmation — propose first
- ❌ Guessing coordinates for buttons, icons, or nav controls — your estimates are unreliable
- ❌ Picking the "closest" or "least-bad" rectangle — all missed, re-bbox
- ❌ Confirming when the rectangle only touches or overlaps the edge of the target — if the label is not entirely inside, it's a miss
- ❌ Picking "right" because the target is on the right side of the screen — look at what's INSIDE the rectangle, not its label
- ❌ Calling confirm_bbox() when unsure — bbox_target() is free, use it again

## UI Presets

Phone UI element annotations are stored in `.claude/ui-presets/` directory as markdown files, one per app.

### Using presets during operation

Before calling bbox_target(), check if the target element exists in a preset file:

1. Read the relevant app's preset file (e.g., `.claude/ui-presets/wechat.md`)
2. Match the current page by its fingerprint description
3. If the target element is listed, use its position coordinates directly with `bbox_target()`
4. If not found:
   - **Fixed UI** → call `propose_bboxes()` with your guesses, wait for user confirmation, then save to preset.
   - **Dynamic content** → fall back to visual estimation with `grid_overlay()`.

Preset coordinates are 0-1 decimals. Three types, classified by aspect ratio:

| Type | Position | Meaning |
|------|----------|---------|
| **box** | `[left, top, right, bottom]` | Normal bounding box |
| **column** | `[left, right]` | X interval (tall thin box in annotation UI, aspect ratio > 10) |
| **row** | `[top, bottom]` | Y interval (wide thin box in annotation UI, aspect ratio > 10) |

In the annotation UI, the user always draws boxes. The backend automatically classifies by aspect ratio and strips the irrelevant axis. Columns/rows render as open-ended lines in the UI.

For scrollable layouts (e.g. cart items), controls are at fixed X positions across all rows. These become **column** presets. At runtime, combine the column's X with the visually-detected row's Y:
- Column preset gives `[left, right]`
- Visual targeting gives row Y as `[top, bottom]`
- Compose: `bbox_target([left, top, right, bottom])`

### Preset file format

One file per app or system context, saved as `.claude/ui-presets/{slug}.md` (lowercase kebab-case, e.g. `wechat.md`, `home-screen.md`).

Each page section has a fingerprint for visual matching, an optional entry path for navigation, and a table of interactive elements.

- **`#` heading** — app name (e.g. `# WeChat`) or system context (e.g. `# Home Screen`, `# Lock Screen`)
- **Fingerprint** — one-line visual description for identifying the current page
- **Entry** — how to reach this page: `{Source Page}` → `{element to tap}`. Omit for home/default screens.
- **Position** — `[left, top, right, bottom]` as 0-1 decimals, up to 3 decimal places
- **Action** column — what tapping does. Use `→ {Page Name}` for navigation (must match a `##` heading in any preset file), or a short verb phrase for non-navigation actions.

```markdown
# {App or Context Name}

## {Page Name}

Fingerprint: {one-line visual description}
Entry: {Source Page} → {element tapped to get here}

| Element      | Position                         | Action               |
|--------------|----------------------------------|----------------------|
| Settings     | [0.750, 0.931, 1.000, 1.000]     | → Settings           |
| Search bar   | [0.045, 0.083, 0.850, 0.133]     | Opens search overlay |

## Cart Items (scrollable)

Fingerprint: list of items with checkboxes, images, quantity controls
Entry: Product Page → Add to Cart

| Element      | Position      | Action            |
|--------------|---------------|-------------------|
| Checkbox     | [0.020, 0.080] | Toggle select     |
| Qty -        | [0.620, 0.700] | Decrease quantity |
| Qty +        | [0.780, 0.860] | Increase quantity |
```

The cart item positions are **columns** `[left, right]` — drawn as tall thin boxes in the annotation UI, automatically classified by the backend. At runtime, the agent combines the column X with the visually-detected row Y.

### Propose-confirm workflow

When a fixed UI element is not in any preset:

1. `park()` + `screenshot()` — see the current screen, reason about what UI elements are visible
2. Call `propose_bboxes([{"bbox": [l,t,r,b], "label": "element name"}, ...])` — parks arm, takes a fresh screenshot, and sends your guesses to the annotation UI
3. Tell the user: "I've proposed N bounding boxes on the annotation page. Please review, adjust, and confirm at /annotate"
4. Call `wait_for_confirmation()` — blocks until the user confirms
5. Receive confirmed boxes with user-corrected coordinates and labels
6. Use the confirmed coordinates: `bbox_target(bbox)` → `confirm_bbox()` → gesture
7. Save confirmed boxes to the preset file for future autonomous use

In the annotation UI at /annotate:
- Agent-proposed boxes appear as **orange rectangles** with an "AI" badge inside
- User-drawn boxes appear as **solid colored rectangles**
- The user can move, resize, delete, or relabel any box
- The user can add new boxes
- When satisfied, the user clicks **Confirm** to send boxes back to the agent

### User-initiated annotation

When the user wants to annotate UI elements manually:

1. Ask the user to open [annotate](http://localhost:8048/annotate) and draw bounding boxes
2. Wait for the user to confirm they are done
3. Call `get_user_annotations()` to get the screenshot with boxes and coordinates
4. Identify what app, what page, and what each boxed element is
5. List findings and ask the user to confirm
6. Write the preset markdown file to `.claude/ui-presets/{app_slug}.md`

### Keyboard typing

Keyboard key positions are stored in `.claude/ui-presets/system-keyboard.md`. To type text, look up each character's Position in the preset and tap it — no need for the visual bbox_target() workflow.

If the preset file doesn't exist or needs recalibration:

1. Ask the user to take two phone screenshots with the keyboard open — one showing the alpha keyboard (default), one showing the numeric keyboard (tap 123 key first)
2. Save 2 screenshots in `data/image/keyboard/`
3. Run `uv run python scripts/calibrate_keyboard.py`
4. The script detects key bounding boxes and writes `system-keyboard.md` with positions filled in
5. Keys marked ??? in the preset need to be identified from the bounding box images in `data/image/keyboard/bbox/`

## For developers editing this codebase

Tool docstrings and the FastMCP `instructions` field are prompts that control agent behavior.

- Never write "pick the best" — it causes forced-choice bias
- Use the label test: "can you read the ENTIRE label inside the rectangle?"
- Add failure examples — agents learn more from examples than rules
