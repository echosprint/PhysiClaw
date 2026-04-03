---
name: setup
description: Connect the robotic arm and camera, then calibrate using the plan's 7-step touch-based calibration. Required before using any PhysiClaw MCP tools.
allowed-tools: Bash, Read
---

# PhysiClaw Hardware Setup

Connect hardware and run the plan's calibration. Each step uses touch coordinates from the /calibrate page — no green flash.

The server must be running at <http://localhost:8048> before starting.

## Step 1: Live camera preview

Tell the user:
> Place the phone face-up under the camera, screen on. I'm opening Photo Booth so you can see the live camera feed and adjust the phone position.

```bash
open -a "Photo Booth"
```

Tell the user:
> Adjust the phone until the full screen is centered in Photo Booth. Close Photo Booth when done.

Wait for confirmation.

## Step 2: Check status

```bash
curl -s http://localhost:8048/api/status 2>/dev/null | python3 -m json.tool 2>/dev/null
```

- Server not running → tell user to start with `uv run physiclaw`
- Already calibrated → done
- Partially done → resume from next step

## Step 3: Connect arm

Tell the user:
> Plug the robotic arm into USB. Make sure the motor power switch is ON. Check stylus is attached.

```bash
curl -s -X POST http://localhost:8048/api/connect-arm | python3 -m json.tool
```

## Step 4: Connect camera

Scan cameras and let user pick:

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
open /tmp/physiclaw_cam*.jpg
```

Ask user which camera shows the overhead view, then connect:

```bash
curl -s -X POST http://localhost:8048/api/connect-camera \
  -H 'Content-Type: application/json' \
  -d '{"index": CHOSEN_INDEX}' | python3 -m json.tool
```

## Step 5: Open calibration page on phone

Get the calibration URL:

```bash
python3 -c "from physiclaw.bridge import get_lan_ip; print(f'http://{get_lan_ip()}:8048/calibrate')"
```

Tell the user:
> Open this URL on the **phone** in Safari or Chrome. The phone must be on the same WiFi as this computer. Make the page full screen (swipe up to hide the address bar).

Also tell them the QR page:
> Or open http://localhost:8048/qr on this computer and scan the QR code with your phone.

Wait for confirmation that the calibration page is open and showing "Waiting for calibration..."

## Step 6: Position stylus

Tell the user:
> The calibration page now shows an orange circle at the center of the screen. Position the stylus tip directly above the orange circle, about 3mm above the screen surface. Don't touch the screen yet.

Now set the calibration page to "center":

```bash
curl -s -X POST http://localhost:8048/api/calibrate/set-phase \
  -H 'Content-Type: application/json' \
  -d '{"phase": "center"}' | python3 -m json.tool
```

Wait for user confirmation.

## Step 7: Step 0 — Z-depth (~5s)

Tell the user:
> The arm will probe downward in small steps to find the screen surface. A touch event tells us when it makes contact. Don't touch anything.

```bash
curl -s -X POST http://localhost:8048/api/calibrate/step0-z-depth --max-time 30 | python3 -m json.tool
```

## Step 8: Step 1 — Alignment check (~3s)

Tell the user:
> The arm will tap two points to check if the phone is aligned with the arm.

```bash
curl -s -X POST http://localhost:8048/api/calibrate/step1-alignment --max-time 30 | python3 -m json.tool
```

If `aligned` is false, tell the user to adjust the phone rotation slightly and retry this step.

## Step 9: Step 2 — Camera rotation check (~2s)

```bash
curl -s -X POST http://localhost:8048/api/calibrate/step2-camera-rotation --max-time 10 | python3 -m json.tool
```

If `ok` is false, tell the user: "Please rotate the camera 90° and retry."

## Step 10: Step 3 — Software rotation (~2s)

Tell the user:
> The calibration page will now show blue UP and RIGHT markers. The camera detects these to determine the correct image rotation.

```bash
curl -s -X POST http://localhost:8048/api/calibrate/step3-sw-rotation --max-time 15 | python3 -m json.tool
```

## Step 11: Step 4 — GRBL↔Screen mapping (~15s)

Tell the user:
> The arm will tap 11 distributed points across the screen. Each tap reports its touch coordinate for precise mapping. This builds Mapping A (screen → arm position).

```bash
curl -s -X POST http://localhost:8048/api/calibrate/step4-mapping-a --max-time 60 | python3 -m json.tool
```

## Step 12: Step 5 — Camera↔Screen mapping (~5s)

Tell the user:
> The calibration page will show 15 red dots. The camera detects their positions to build Mapping B (camera pixels → screen coordinates). The arm will park out of the way.

```bash
curl -s -X POST http://localhost:8048/api/calibrate/step5-mapping-b --max-time 30 | python3 -m json.tool
```

## Step 13: Step 6 — Full-chain validation (~10s)

Tell the user:
> Final validation: I'll show orange dots at random positions, tap them, and compare the touch coordinates against the expected position. This tests the entire pipeline.

```bash
curl -s -X POST http://localhost:8048/api/calibrate/step6-validate --max-time 60 | python3 -m json.tool
```

Check `calibrated` in the response. If true, calibration succeeded.

## Step 14: Edge trace verification (~20s)

Tell the user:
> The arm will trace the phone screen border clockwise — moving to 8 edge points and pausing at each. Watch and confirm it follows the screen edges accurately.

```bash
curl -s -X POST http://localhost:8048/api/calibrate/verify-edge --max-time 60 | python3 -m json.tool
```

## Step 15: Verify

```bash
curl -s http://localhost:8048/api/status | python3 -m json.tool
```

Confirm `calibrated` is true. Tell the user:
> PhysiClaw is ready. All MCP tools are now available.

## Error handling

If any step fails:
1. Show the error to the user
2. Guide them to fix the physical setup
3. Retry that specific step — no need to restart from the beginning
