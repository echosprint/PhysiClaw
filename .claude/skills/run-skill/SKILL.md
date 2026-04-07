---
name: run-skill
description: Execute an app automation skill from data/app-skills/. Reads the SKILL.md, matches screens, executes steps with parameters. Pass the skill name and parameters as arguments.
allowed-tools: Bash, Read, mcp__physiclaw__park, mcp__physiclaw__screenshot, mcp__physiclaw__detect_elements, mcp__physiclaw__grid_overlay, mcp__physiclaw__bbox_target, mcp__physiclaw__confirm_bbox, mcp__physiclaw__tap, mcp__physiclaw__double_tap, mcp__physiclaw__long_press, mcp__physiclaw__swipe, mcp__physiclaw__propose_bboxes, mcp__physiclaw__wait_for_confirmation, mcp__physiclaw__phone_screenshot, mcp__physiclaw__bridge_status, mcp__physiclaw__bridge_send_text, mcp__physiclaw__bridge_tap, mcp__physiclaw__check_screen, mcp__physiclaw__check_screen_changed, mcp__physiclaw__get_user_annotations
---

# Run App Skill

Execute an app automation skill. The skill was previously built with `/build-skill`.

**Arguments:** `{skill_name} [param1=value1] [param2=value2] ...`

Example: `/run-skill meituan item_name=宫保鸡丁 quantity=1`

## Step 1: Load the skill

```bash
cat data/app-skills/{skill_name}/SKILL.md
```

If the skill doesn't exist, list available skills:
```bash
ls data/app-skills/
```

Parse the SKILL.md to extract:
- **Parameters** and their values (from arguments)
- **Screen flow** (ordered list of screens)
- **Per-screen data:** fingerprint, reference screenshot path, fixed element positions, dynamic element rules, navigation actions

## Step 2: Pre-execution planning (search-first)

Before touching the app, think about what information you need:
- What parameters were provided?
- Are any parameters missing? Ask the user.
- What text needs to be pasted via bridge? (item names, search queries)
- What's the optimal path through the app? (search, not scroll)

**Critical principle: Don't mimic humans.** If the skill needs to find an item in a list, use the app's search box (clipboard paste), not scrolling.

## Step 3: Ensure bridge is ready

Call `bridge_status()`. If the phone isn't connected, tell the user to open the bridge URL on the phone.

The bridge page must be in the foreground for clipboard operations. If the last operation left the phone in another app, guide the user to reopen Safari, or:
1. Swipe up from bottom → home screen
2. Open Safari via Spotlight (if Safari isn't already the frontmost)
3. Safari should reopen to the /message page

## Step 4: Open the app

Use the Open App flow:
1. `bridge_send_text("{app_name}")` → `bridge_tap()` — clipboard ready
2. Swipe up → home screen
3. Swipe down from middle → Spotlight
4. Long press search field → tap Paste
5. Tap first result → app opens

Or if the app is already open (check with `check_screen(reference)`), skip this.

## Step 5: Execute screen flow

For each screen in the skill:

### 5a. Identify current screen

Use `check_screen(reference_path)` with the reference screenshot from the skill.

If the screen doesn't match:
- Check the next/previous screens in the flow — you might have skipped one
- Check for popups: `check_screen_changed()` shows if something unexpected appeared
- If a dark overlay is detected, look for a close/dismiss button

### 5b. Execute fixed element actions

For each fixed element in the skill's screen data:
- Use the saved position directly with `bbox_target(position)`
- Verify with the label test (read what's inside the rectangle)
- `confirm_bbox()` → gesture (tap, swipe, etc.)

### 5c. Handle dynamic elements

For elements that change (list items, search results):
- **Search-first:** `bridge_send_text(search_term)` → return to app → paste into search box → tap first result
- **OCR fallback:** if no search box, use `detect_elements()` to find text, then target it
- **Card y-scan:** for scrollable lists, the skill's layout constants give the x-column position of content images; the card y-scan finds their vertical positions

### 5d. Verify screen transition

After each action that should change screens:
- `park()` + `camera_view()` to see the result
- `check_screen(next_reference)` to confirm we reached the next screen
- If screen didn't change, the gesture may not have registered — retry (just call the gesture again, bbox is retained)

## Step 6: Complete and report

After executing all screens:
- Take a final screenshot to verify the end state
- Report the result to the user

If anything went wrong:
- Describe what happened and which screen/step failed
- Take a screenshot showing the current state
- Ask the user if they want to retry or abort

## Error handling

**Gesture didn't register:** Call the same gesture again. The confirmed bbox is retained.

**Wrong screen:** Take a screenshot, try to identify where you are. If lost, ask the user.

**Popup/overlay:** Detect dark overlay with `check_screen`. Look for close/X button at top-right, or tap outside the popup to dismiss.

**Bridge disconnected:** Clipboard operations will fail. Ask the user to reopen the bridge page on their phone.

**App not responding:** If the screen doesn't change after 3 retries, the app might be loading. Wait 2-3 seconds and retry. If still stuck, ask the user.
