---
name: setup-screenshot
description: Set up pixel-perfect phone screenshots via AssistiveTouch + iOS Shortcuts. Single tap = screenshot, double tap = screenshot + upload to PhysiClaw. One-time setup.
allowed-tools: Bash, Read, Write, Edit, mcp__physiclaw__park, mcp__physiclaw__screenshot, mcp__physiclaw__propose_bboxes, mcp__physiclaw__wait_for_confirmation, mcp__physiclaw__bbox_target, mcp__physiclaw__confirm_bbox, mcp__physiclaw__tap, mcp__physiclaw__bridge_status, mcp__physiclaw__phone_screenshot
---

# Screenshot Setup — AssistiveTouch + iOS Shortcuts

Guide the user step-by-step to set up their iPhone so that:

- **Single tap** AssistiveTouch → takes a screenshot (saved on phone)
- **Double tap** AssistiveTouch → takes a screenshot AND uploads it to PhysiClaw

This is a one-time setup. After this, the agent can call `phone_screenshot()` to get pixel-perfect screenshots.

## Step 1: Get the server URL

Get the LAN IP — both devices must be on the same WiFi:

```bash
uv run python -c "from physiclaw.bridge import get_lan_ip; print(f'http://{get_lan_ip()}:8048')"
```

Save this URL for the Shortcut. Tell the user:

> First, make sure your iPhone is on the **same WiFi** as this computer.

## Step 2: Enable AssistiveTouch

Tell the user:

> **On your iPhone:**
>
> 1. Open **Settings**
> 2. Go to **Accessibility** → **Touch** → **AssistiveTouch**
> 3. Turn **AssistiveTouch ON**
>
> You should see a floating semi-transparent circle button appear on screen.

Wait for user confirmation before proceeding.

## Step 3: Create the iOS Shortcut

Tell the user (replace `SERVER_IP` with the actual IP from Step 1):

> **Open the Shortcuts app** on your iPhone and create a new shortcut:
>
> 1. Tap **+** (top right) to create a new shortcut
> 2. Tap **Add Action**
> 3. Search for **"Take Screenshot"** and add it
> 4. Tap **+** again to add a second action
> 5. Search for **"Get Contents of URL"** and add it
> 6. Configure "Get Contents of URL":
>    - **URL**: `http://SERVER_IP:8048/api/bridge/screenshot`
>    - Tap **Show More** (or the arrow)
>    - **Method**: change to **POST**
>    - **Request Body**: change to **File**
>    - **File**: tap and select the **Screenshot** variable from the previous step
> 7. Tap the shortcut name at the top and rename it to **"PhysiClaw Screenshot"**
> 8. Tap **Done** to save

Wait for user confirmation.

## Step 4: Configure AssistiveTouch actions

Tell the user:

> **Go back to Settings → Accessibility → Touch → AssistiveTouch**
>
> Under **Custom Actions**, set:
>
> - **Single-Tap** → **Screenshot** (built-in, just takes a screenshot)
> - **Double-Tap** → **Shortcut** → select **"PhysiClaw Screenshot"**
>
> This way:
> - One tap = quick screenshot saved on your phone
> - Two taps = screenshot + upload to PhysiClaw server

Wait for user confirmation.

## Step 5: Position the AssistiveTouch button

Tell the user:

> **Drag the AssistiveTouch button** to the **right edge of the screen**, roughly **vertically centered**. This keeps it out of the way of most app UIs.

Wait for user confirmation.

## Step 6: Test the double-tap upload

Tell the user:

> **Double-tap the AssistiveTouch button** now. You should see a brief screenshot flash. I'll check if it arrived on my end.

Wait a few seconds, then check:

```bash
ls -lt data/phone/screenshot/ 2>/dev/null | head -5
```

If a file appears, read it to verify it's a valid screenshot image:

```bash
uv run python -c "
from pathlib import Path
files = sorted(Path('data/phone/screenshot').glob('*'), key=lambda f: f.stat().st_mtime, reverse=True)
if files:
    f = files[0]
    print(f'Latest: {f.name} ({f.stat().st_size:,} bytes)')
else:
    print('No screenshots found')
"
```

If no screenshot arrived, troubleshoot:

- Open the Shortcuts app → tap the "PhysiClaw Screenshot" shortcut → tap the **play ▶** button to run it manually. Does it show any errors?
- Check the URL in the shortcut matches `http://SERVER_IP:8048/api/bridge/screenshot`
- Verify both devices are on the same WiFi
- Try opening `http://SERVER_IP:8048/bridge` in Safari on the phone — if it loads, the network works

Repeat until a screenshot arrives successfully.

## Step 7: Find AssistiveTouch button position (for agent use)

The agent needs to know where the AssistiveTouch button is so it can double-tap it with the stylus arm.

Use MCP tools: `park()` then `screenshot()` to see the phone.

The AssistiveTouch button is a semi-transparent circle, typically on the right edge. Propose its position:

```
propose_bboxes([{"bbox": [estimated_l, estimated_t, estimated_r, estimated_b], "label": "AssistiveTouch"}])
```

Tell the user: "I've proposed the AssistiveTouch button position. Please review and correct at /annotate."

Call `wait_for_confirmation()` to get the confirmed position.

## Step 8: Save to preset

Save the confirmed AssistiveTouch position to the system preset file.

Read the existing preset:
```bash
cat .claude/ui-presets/system.md 2>/dev/null || echo "not found"
```

Create or update `.claude/ui-presets/system.md`:

```markdown
# System

## Any Screen

Fingerprint: Any screen with AssistiveTouch button visible (semi-transparent circle on right edge)

| Element | Position | Action |
|---------|----------|--------|
| AssistiveTouch | [left, top, right, bottom] | Single tap: screenshot; Double tap: screenshot + upload |
```

Use the confirmed coordinates from Step 7.

## Step 9: Verify autonomous double-tap

Test that the agent can trigger a screenshot upload by double-tapping AssistiveTouch:

Use the `phone_screenshot()` MCP tool. It will double-tap AssistiveTouch and wait for the upload.

If it returns a screenshot image, setup is complete.

If it fails:
- The button position might be slightly off — redo Step 7
- The Shortcut might not be linked to double-tap — check AssistiveTouch settings
- Timeout — the Shortcut takes a moment; try increasing timeout

## Done

Tell the user:

> Screenshot setup complete!
>
> - **Single tap** the AssistiveTouch button → screenshot saved on phone
> - **Double tap** the AssistiveTouch button → screenshot uploaded to PhysiClaw
>
> The agent can now call `phone_screenshot()` to get pixel-perfect screenshots for UI analysis and skill building.
