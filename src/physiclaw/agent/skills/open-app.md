---
name: open-app
description: Use when you need to launch an app that is NOT on the current screen and NOT in the dock — any time "open <app>" is the next step and you can't see the icon. Use FIRST before tapping blindly. NOT needed when the target app's icon is already visible.
---

# Open App via Spotlight

**Argument:** app name (e.g. `美团`, `WhatsApp`, `Safari`). Each step is one `[note, one-other]` turn; placeholders → **Fixed elements**.

1. `send_to_clipboard(text="<app name>")` — the exact name the user used; a Chinese app keeps its Chinese name.
2. `home_screen` — **skip if already on the home screen** (icon grid + dock, no in-app chrome).
3. `swipe(bbox=[0.3, 0.4, 0.7, 0.6], direction="down", size="l")` — mid-screen pull opens Spotlight (a top-edge origin would open Notification Center); `size="l"` avoids overshoot. The attached view shows the search field + keyboard.
4. **Stale text in the field?** Clear it — skip when empty; over-tapping an empty field is a no-op, so over-estimate freely (tap, don't long-press — deterministic per tap):

   ```python
   sequence(actions=[
       {"tool_name": "tap", "arg": {{search-field}}},   # Spotlight search field
       {"tool_name": "tap", "arg": {{backspace}}},      # backspace ×4
       {"tool_name": "tap", "arg": {{backspace}}},
       {"tool_name": "tap", "arg": {{backspace}}},
       {"tool_name": "tap", "arg": {{backspace}}},
   ])
   ```

5. Paste the name (both boxes learned):

   ```python
   sequence(actions=[
       {"tool_name": "long_press", "arg": {{search-field}}},   # Spotlight search field → Paste popover
       {"tool_name": "tap",        "arg": {{paste-button}}},   # Paste / 粘贴 — NOT AutoFill
   ])
   ```

   The attached view shows results below the field; no results = the paste missed — re-ground and redo.
6. `tap(<app-icon>)` — launch; the attached view confirms the app opened.

   **No app icon?** You searched an app name **+ an entry inside it** — no installed app matches. Backspace to the bare app-name prefix (step 4) until the icon shows. Never tap an "in Safari" / web result — that lands in the browser.

   **Still nothing (mis-spelled)?** `home_screen`, then page for the icon by hand — `swipe(bbox=[0.01, 0.4, 0.02, 0.6], direction="right", size="xl", speed="medium")` = previous page, `swipe(bbox=[0.98, 0.4, 0.99, 0.6], direction="left", size="xl", speed="medium")` = next — then tap it.

## Fixed elements

The commented boxes above come from SYSTEM § Screen layout, pre-substituted (`{{…}}` still showing = layout not learned — ground live). Also:

- Paste decoy — pick `Paste` / `粘贴` in the pill popover ABOVE the field, NOT `AutoFill`; it dismisses if you tap elsewhere first.
- `<app-icon>` — from the search results in the latest listing (§ Bboxes); label matches **exactly** — skip App Store badges and "in Safari" / web hits.
