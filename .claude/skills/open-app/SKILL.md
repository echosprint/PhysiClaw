---
name: open-app
description: Open any app on the phone via Spotlight search. Uses the LAN bridge for clipboard paste — much faster than typing. Pass the app name as argument.
allowed-tools: mcp__physiclaw__bridge_status, mcp__physiclaw__bridge_send_text, mcp__physiclaw__bridge_tap, mcp__physiclaw__park, mcp__physiclaw__screenshot, mcp__physiclaw__bbox_target, mcp__physiclaw__confirm_bbox, mcp__physiclaw__tap, mcp__physiclaw__long_press, mcp__physiclaw__swipe, mcp__physiclaw__propose_bboxes, mcp__physiclaw__wait_for_confirmation, mcp__physiclaw__grid_overlay, Read
---

# Open App via Spotlight

Open any app by name using Spotlight search + clipboard paste. This is a fixed sequence — same physical actions every time, only the app name changes.

**Argument:** The app name to open (e.g., "美团外卖", "WeChat", "Safari").

## Prerequisites

1. Hardware calibrated (`/setup` done)
2. Bridge connected (phone has `/message` page open in browser)

Check bridge status first. If not connected, tell the user to open the bridge URL on their phone.

## Sequence

### Step 1: Ensure /message page is in foreground

The bridge /message page must be visible on the phone for clipboard copy to work. If the user was in another app, we need to get back to Safari first.

Check: is the bridge connected? Call `bridge_status()`.

If not connected, tell the user:
> Please open Safari on your phone and navigate to the bridge URL, then come back here.

### Step 2: Send text to clipboard

Call `bridge_send_text("{app_name}")` with the app name argument.

Then call `bridge_tap()` to tap the screen center — this copies the text to clipboard.

Verify the tool returns "Clipboard ready". If it fails, the /message page might not be in the foreground.

### Step 3: Go to home screen

Swipe up from the bottom of the screen to go home. This ensures a clean starting state for Spotlight.

Target the bottom-center of the screen:
- `bbox_target([0.3, 0.95, 0.7, 1.0])` → confirm → `swipe(direction="top", speed="medium")`

After swiping, `park()` + `camera_view()` to verify we're on the home screen.

### Step 4: Open Spotlight

Swipe down from the middle of the home screen to open Spotlight search.

Target the center of the screen:
- `bbox_target([0.3, 0.4, 0.7, 0.6])` → confirm → `swipe(direction="bottom", speed="slow")`

After swiping, `park()` + `camera_view()` to verify Spotlight is open (look for the search bar at the top).

### Step 5: Paste into Spotlight search

The Spotlight search field should already be focused (cursor blinking). Long press to get the paste menu, then tap "Paste".

1. Target the Spotlight search field (near the top of the screen, roughly [0.1, 0.06, 0.9, 0.10])
2. `long_press()` — context menu with "Paste" appears
3. `park()` + `camera_view()` — look for the "Paste" option
4. Target and tap "Paste"

After pasting, `park()` + `camera_view()` to verify the app name appears in the search field and results show the target app.

### Step 6: Tap the first search result

The first search result should be the app. It appears below the search field, typically as a large icon with the app name.

Target the first result (roughly [0.1, 0.12, 0.9, 0.22]) and tap it.

After tapping, `park()` + `camera_view()` to verify the app has opened.

## Notes

- The "Paste" context menu position depends on where you long-press. It appears as a floating bar near the press location. Use visual targeting (grid_overlay + bbox_target) to find and tap "Paste".
- If Spotlight doesn't open, the phone might not be on the home screen. Try swiping up again.
- If the app doesn't appear in results, the name might be spelled differently. Try the Chinese or English name.
- After the app opens, the bridge page is in the background. If you need to clipboard-paste again later, you'll need to return to Safari first.
