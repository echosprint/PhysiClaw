---
name: setup
description: Connect the robotic arm and camera, then calibrate. Required before using any PhysiClaw MCP tools (screenshot, tap, swipe, etc.).
allowed-tools: Bash, Read
---

# PhysiClaw Hardware Setup

Connect the arm, camera, and run calibration step by step. Each step calls a server endpoint and reports the result.

The server must be running at <http://localhost:8048> before starting.

## Step 1: Live camera preview

Before any hardware setup, let the user see the camera feed to align the phone under the camera.

Tell the user:
> Place the phone face-up under the camera, screen on. I'm opening Photo Booth so you can see the live camera feed and adjust the phone position.

```bash
open -a "Photo Booth"
```

Tell the user:
> Adjust the phone until the full screen is centered in Photo Booth. If Photo Booth shows the wrong camera, click **Camera** in the title bar to switch to the overhead camera. Close Photo Booth when you're done.

Wait for the user to confirm alignment, then proceed.

## Step 2: Check current status

```bash
curl -s http://localhost:8048/api/status 2>/dev/null | python3 -m json.tool 2>/dev/null
```

- If the command fails or returns nothing: the server is not running. Tell the user to start it in another terminal with `uv run physiclaw` and wait for "PhysiClaw MCP server on ..." to appear, then retry this step.
- If `calibrated` is true: tell the user everything is already set up and stop.
- If some phases are done: resume from the next incomplete step.

## Step 3: Connect arm

Tell the user:
> Plug the robotic arm into USB. The control board is powered by USB, but the arm motors have a separate power supply — make sure the motor power switch is ON, otherwise the arm will connect but won't move. Also check that the stylus is attached.

Wait for user confirmation, then:

```bash
curl -s -X POST http://localhost:8048/api/connect-arm | python3 -m json.tool
```

If error "GRBL device not found": ask the user to check the USB cable and try unplugging/replugging.

## Step 4: Connect camera

Tell the user:
> Place the phone face-up under the camera. Turn the screen on. Make sure the full screen is visible from above.

Wait for user confirmation. Then scan cameras one by one to let the user pick the overhead camera:

```bash
rm -f /tmp/physiclaw_cam*.jpg
uv run python -c "
import json, base64, urllib.request
for i in range(4):
    try:
        resp = urllib.request.urlopen(f'http://localhost:8048/api/camera-preview/{i}')
        d = json.loads(resp.read())
        path = f'/tmp/physiclaw_cam{i}.jpg'
        with open(path, 'wb') as f:
            f.write(base64.b64decode(d['image']))
        print(f'Camera {i}: saved to {path}')
    except Exception:
        print(f'Camera {i}: not available')
"
```

Open the saved images so the user can see them:

```bash
open /tmp/physiclaw_cam*.jpg
```

Ask the user: "Which camera shows the overhead view of the phone?"

Once the user picks a camera index, connect it:

```bash
curl -s -X POST http://localhost:8048/api/connect-camera \
  -H 'Content-Type: application/json' \
  -d '{"index": CHOSEN_INDEX}' | python3 -m json.tool
```

Replace `CHOSEN_INDEX` with the user's choice (e.g., `0`, `1`, `2`).

## Step 5: Prepare for calibration

Tell the user:
> Open **<https://www.physiclaw.ai/pen-calib>** on the **phone** (not the computer). Make the page full screen (hide the browser address bar). Position the stylus tip directly above the **center orange circle**. The tip should hover ~3mm above the screen, not touching it.

Wait for user confirmation before proceeding.

## Step 6: Z-depth (~15-25s)

Tell the user:
> The arm will probe downward to find the screen surface. The phone flashes green on each successful tap. Don't touch anything during calibration.

```bash
curl -s -X POST http://localhost:8048/api/calibrate/z-depth --max-time 60 | python3 -m json.tool
```

If error: ask user to reposition the stylus closer to the center circle and retry.

## Step 7: Find phone-right (~5-15s)

Tell the user:
> The arm will probe in 4 directions to find the right circle on the calibration page.

```bash
curl -s -X POST http://localhost:8048/api/calibrate/find-right --max-time 60 | python3 -m json.tool
```

## Step 8: Find phone-down (~3-8s)

Tell the user:
> Probing for the down circle. Almost done with direction mapping.

```bash
curl -s -X POST http://localhost:8048/api/calibrate/find-down --max-time 30 | python3 -m json.tool
```

## Step 9: Long press (~5s)

Tell the user:
> Verifying long press. The stylus will hold contact for about 1 second, 3 times.

```bash
curl -s -X POST http://localhost:8048/api/calibrate/long-press --max-time 30 | python3 -m json.tool
```

## Step 10: Swipe (~5s)

Tell the user:
> Verifying swipe. The stylus will slide in 4 directions from center.

```bash
curl -s -X POST http://localhost:8048/api/calibrate/swipe --max-time 30 | python3 -m json.tool
```

## Step 11: Grid calibration (~40-90s)

Tell the user:
> The calibration page should now show red dots. The arm will visit all 15 dots to build a coordinate map. This takes 1-2 minutes.

```bash
curl -s -X POST http://localhost:8048/api/calibrate/grid --max-time 180 | python3 -m json.tool
```

## Step 12: Edge trace verification (~20s)

Tell the user:
> The arm will now trace the phone screen border clockwise — moving to 8 edge points and pausing at each. Watch the stylus tip and confirm it follows the screen edges accurately.

```bash
curl -s -X POST http://localhost:8048/api/calibrate/verify-edge --max-time 60 | python3 -m json.tool
```

If the arm doesn't follow the screen edges: the calibration may be off. Ask the user if they want to redo from Step 6 (phase 1).

## Step 13: Verify

```bash
curl -s http://localhost:8048/api/status | python3 -m json.tool
```

Confirm `calibrated` is true. Tell the user: "PhysiClaw is ready. All MCP tools are now available. If tools can't connect, run `/mcp` in Claude Code to reconnect."

## Error handling

If any step fails:

1. Show the error message to the user
2. Guide them to fix the physical setup (reposition stylus, check screen, replug USB)
3. Retry that specific step — no need to restart from the beginning
