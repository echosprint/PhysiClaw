---
name: setup-screenshot
description: Set up pixel-perfect phone screenshots via AssistiveTouch + iOS Shortcuts. One-time setup — enables phone_screenshot() tool for skill building.
allowed-tools: Bash, Read, Write, Edit, mcp__physiclaw__park, mcp__physiclaw__screenshot, mcp__physiclaw__propose_bboxes, mcp__physiclaw__wait_for_confirmation, mcp__physiclaw__bbox_target, mcp__physiclaw__confirm_bbox, mcp__physiclaw__tap, mcp__physiclaw__bridge_status, mcp__physiclaw__phone_screenshot
---

# Screenshot Skill Setup

Set up the phone to take pixel-perfect screenshots and upload them to the server. This is a one-time setup. After this, the agent can call `phone_screenshot()` to get clean screenshots for skill building and UI analysis.

**Prerequisites:** Hardware must be calibrated (`/setup` done), and the bridge must be accessible from the phone (same WiFi).

## Step 1: Get bridge URL

```bash
curl -s http://localhost:8048/api/status | python3 -m json.tool
```

Confirm hardware is calibrated. Then get the bridge URL:

Tell the user the server's LAN IP by running:
```bash
python3 -c "from physiclaw.bridge import get_lan_ip; print(f'http://{get_lan_ip()}:8048')"
```

The user's phone must be on the same WiFi network as this computer.

## Step 2: Guide AssistiveTouch setup

Walk the user through these steps via chat. Be patient and clear — this is a manual process on the phone.

Tell the user:

> We need to set up screenshot capability. I'll walk you through each step.
>
> **Step A:** Open **Settings** on your phone. Go to **Accessibility → Touch → AssistiveTouch**. Turn it **ON**. You'll see a floating circle button appear on screen.
>
> **Step B:** Still in AssistiveTouch settings, tap **Custom Actions**. Set **Single-Tap** to **"Run Shortcut"** (we'll create the shortcut next).

Wait for user confirmation before proceeding to the Shortcut step.

## Step 3: Guide Shortcut creation

Tell the user:

> **Step C:** Open the **Shortcuts** app. Tap **+** to create a new shortcut. Add these two actions in order:
>
> 1. **Take Screenshot** — search for it in the actions list
> 2. **Get Contents of URL** — set it up as:
>    - URL: `http://SERVER_IP:8048/api/bridge/screenshot`
>    - Method: **POST**
>    - Request Body: **File**
>    - File: select the **Screenshot** variable from step 1
>
> Name the shortcut **"PhysiClaw Screenshot"** and save it.

Replace `SERVER_IP` with the actual LAN IP from Step 1.

Wait for user confirmation.

## Step 4: Link AssistiveTouch to the Shortcut

Tell the user:

> **Step D:** Go back to **Settings → Accessibility → Touch → AssistiveTouch → Custom Actions → Single-Tap**. Select your **"PhysiClaw Screenshot"** shortcut.
>
> **Step E:** Drag the AssistiveTouch button to the **right edge of the screen**, roughly **vertically centered**. This keeps it out of the way of most app UIs.

Wait for user confirmation.

## Step 5: Test manual trigger

Tell the user:

> Tap the AssistiveTouch button once. You should see a brief screenshot flash. I'll check if the screenshot arrived on my end.

Wait a few seconds, then check:

```bash
curl -s http://localhost:8048/api/status | python3 -m json.tool
```

Also check if screenshot data arrived by looking at recent files:
```bash
ls -la data/snapshot/ | tail -5
```

If no screenshot arrived, troubleshoot:
- Check the Shortcut: open it and tap the play button manually. Does it show errors?
- Check the URL in the Shortcut matches the server IP
- Check both devices are on the same WiFi
- Try the URL in the phone's browser: `http://SERVER_IP:8048/message` — if this loads, the network connection works

Repeat until a screenshot arrives successfully.

## Step 6: Find AssistiveTouch button position

Now use the propose-confirm workflow to find the exact position of the AssistiveTouch button on screen. The button is visible in camera screenshots.

Take a camera screenshot first to see the current screen state:

Use the MCP tools: `park()` then `screenshot()` to see the phone.

The AssistiveTouch button is a semi-transparent circle, typically on the right edge. Propose its position:

Use `propose_bboxes([{"bbox": [estimated_l, estimated_t, estimated_r, estimated_b], "label": "AssistiveTouch"}])`

Tell the user: "I've proposed the AssistiveTouch button position. Please review and correct at /annotate."

Call `wait_for_confirmation()` to get the confirmed position.

## Step 7: Save to preset

Save the confirmed AssistiveTouch position to the system preset file.

Read the existing preset (if any):
```bash
cat .claude/ui-presets/system.md 2>/dev/null || echo "not found"
```

Create or update `.claude/ui-presets/system.md` with the AssistiveTouch position. Use this format:

```markdown
# System

## Main Screen

Fingerprint: Any screen with AssistiveTouch button visible

| Element | Position | Action |
|---------|----------|--------|
| AssistiveTouch | [left, top, right, bottom] | Takes screenshot (via Shortcut) |
```

Use the confirmed coordinates from Step 6.

## Step 8: Verify autonomous screenshot

Test that the agent can take a screenshot by tapping AssistiveTouch with the stylus:

Use the `phone_screenshot()` MCP tool. If it returns a screenshot image, the setup is complete.

If it fails:
- The button position might be slightly off — redo Step 6
- The Shortcut might not be linked — check AssistiveTouch settings
- There might be a timeout — the Shortcut takes a moment to run

## Done

Tell the user:

> Screenshot setup complete. The agent can now take pixel-perfect screenshots of your phone using `phone_screenshot()`. This enables accurate UI analysis for building automation skills.
