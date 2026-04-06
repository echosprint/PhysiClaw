---
name: setup
description: Connect the robotic arm and camera, then calibrate using the plan's 7-step touch-based calibration. Required before using any PhysiClaw MCP tools.
allowed-tools: Bash, Read
---

# PhysiClaw Hardware Setup

Connect hardware and run the plan's calibration. Each step uses touch coordinates from the /calibrate page — no green flash.

The server must be running at <http://localhost:8048> before starting.

## Step 1: Hardware setup

### 1.1 Live camera preview

Tell the user:
> Place the phone face-up under the camera, screen on. I'm opening Photo Booth so you can see the live camera feed and adjust the phone position.

```bash
open -a "Photo Booth"
```

Tell the user:
> Adjust the phone until the full screen is centered in Photo Booth. **Close Photo Booth when done** — it locks the camera and blocks OpenCV from accessing it.

Wait for confirmation that Photo Booth is closed.

### 1.2 Check status

```bash
curl -s http://localhost:8048/api/status 2>/dev/null | python3 -m json.tool 2>/dev/null
```

- Server not running → tell user to start with `uv run physiclaw`
- Already calibrated → done
- Partially done → resume from next step

### 1.3 Connect arm

Tell the user:
> Plug the robotic arm into USB. Make sure the motor power switch is ON. Check stylus is attached.

```bash
curl -s -X POST http://localhost:8048/api/connect-arm | python3 -m json.tool
```

### 1.4 Connect camera

Scan cameras and let user pick:

```bash
rm -f /tmp/physiclaw_cam*.jpg
uv run python -c "
import json, base64, urllib.request
for i in range(4):
    try:
        resp = urllib.request.urlopen(f'http://localhost:8048/api/camera-preview/{i}?watermark=1')
        d = json.loads(resp.read())
        path = f'/tmp/physiclaw_cam{i}.jpg'
        with open(path, 'wb') as f:
            f.write(base64.b64decode(d['image']))
        print(f'Camera {i}: saved to {path}')
    except Exception:
        print(f'Camera {i}: not available')
"
```

```bash
open /tmp/physiclaw_cam*.jpg
```

Ask user which camera shows the overhead view of the phone screen, then connect:

```bash
curl -s -X POST http://localhost:8048/api/connect-camera \
  -H 'Content-Type: application/json' \
  -d '{"index": CHOSEN_INDEX}' | python3 -m json.tool
```

### 1.5 Open phone page

Print the phone URL and open the QR page:

```bash
uv run python -c "from physiclaw.bridge import get_lan_ip; print(f'Phone URL: http://{get_lan_ip()}:8048/bridge')"
```

```bash
open http://localhost:8048/api/bridge/qr
```

Tell the user:
> I've opened the QR code page in your browser. Scan the QR code with your phone camera. Or type the URL directly in phone Safari/Chrome — it's shown under the QR code. The phone must be on the same WiFi as this computer. This single page handles both calibration and bridge — no need to open a second page later.

Wait for confirmation that the page is open on the phone (shows "PhysiClaw" on black background).

### 1.6 Screenshot coordinate calibration

Tell the user:
> The phone will show an orange square at the center of the screen. **Double-tap AssistiveTouch** to upload a screenshot. This lets me figure out the exact mapping between the phone page and the physical screen.

Note: This requires `/phone-setup` to be done first (AssistiveTouch + iOS Shortcut configured).

```bash
curl -s -X POST http://localhost:8048/api/calibrate/step-screenshot-cal --max-time 35 | python3 -m json.tool
```

If it fails with "no screenshot received", remind the user to double-tap AssistiveTouch and retry.

### 1.7 Position stylus

Switch the phone to calibrate center phase:

```bash
curl -s -X POST http://localhost:8048/api/bridge/switch \
  -H 'Content-Type: application/json' \
  -d '{"mode": "calibrate", "phase": "center"}' | python3 -m json.tool
```

Tell the user:
> The calibration page now shows an orange circle at the center of the screen. Position the stylus tip directly above the orange circle, about 3mm above the screen surface. Don't touch the screen yet.

Wait for user confirmation.

## Step 2: Calibration

### 2.0 Z-depth (~5s, or instant if cached)

Tell the user:
> The arm will probe downward in small steps to find the screen surface. A touch event tells us when it makes contact. Don't touch anything.

```bash
curl -s -X POST http://localhost:8048/api/calibrate/step0-z-depth --max-time 30 | python3 -m json.tool
```

If the response has `"cached": true`, the pen depth was loaded from a previous run — no probing needed. Tell the user it was instant. If the cached value seems wrong (taps miss in later steps), delete `data/pen/z-tap` and rerun this step.

### 2.1 Alignment check (~3s)

Tell the user:
> The arm will tap two points to check if the phone is aligned with the arm.

```bash
curl -s -X POST http://localhost:8048/api/calibrate/step1-alignment --max-time 30 | python3 -m json.tool
```

If `aligned` is false, tell the user to adjust the phone rotation slightly and retry this step.

### 2.2 Camera rotation check (~2s)

First, open Photo Booth so the user can see the live camera feed and adjust position:

```bash
open -a "Photo Booth"
```

Tell the user:
> Adjust the camera so the phone screen fills most of the frame, edges parallel to the image edges. **Close Photo Booth when done** — it locks the camera and blocks OpenCV.

Wait for confirmation that Photo Booth is closed. Reconnect the camera before running the check:

```bash
curl -s -X POST http://localhost:8048/api/connect-camera -H 'Content-Type: application/json' -d '{"index": CHOSEN_INDEX}' | python3 -m json.tool
```

Then run the check:

```bash
rm -f /tmp/physiclaw*.jpg && curl -s -X POST http://localhost:8048/api/calibrate/step2-camera-rotation --max-time 10 | python3 -m json.tool && open /tmp/physiclaw_step2.jpg
```

If `ok` is false, tell the user what to fix based on the `issues` list and retry.

### 2.3 Software rotation (~2s)

Tell the user:
> The calibration page will now show blue UP and RIGHT markers. The camera detects these to determine the correct image rotation.

```bash
curl -s -X POST http://localhost:8048/api/calibrate/step3-sw-rotation --max-time 15 | python3 -m json.tool
```

### 2.4 GRBL↔Screen mapping (~15s)

Tell the user:
> The arm will tap up to 18 points across the screen (3 scale probes + 15 grid points). Each tap reports its touch coordinate for precise mapping. This builds Mapping A (screen → arm position).

```bash
curl -s -X POST http://localhost:8048/api/calibrate/step4-mapping-a --max-time 60 | python3 -m json.tool
```

If step 4 fails with "no touch at +X/+Y probe", the probe taps landed outside the screen. Possible fixes:
- Reposition the phone so the stylus is more centered over the screen
- The z_tap may be too shallow — delete `data/pen/z-tap` and rerun step 2.0 to re-probe
- Then retry step 4

### 2.5 Camera↔Screen mapping (~5s)

Tell the user:
> The calibration page will show 15 red dots. The camera detects their positions to build Mapping B (camera pixels → screen coordinates). The arm will park out of the way.

```bash
curl -s -X POST http://localhost:8048/api/calibrate/step5-mapping-b --max-time 30 | python3 -m json.tool
```

### 2.6 Full-chain validation (~10s)

Tell the user:
> Final validation: I'll show orange dots at random positions, tap them, and compare the touch coordinates against the expected position. This tests the entire pipeline.

```bash
curl -s -X POST http://localhost:8048/api/calibrate/step6-validate --max-time 60 | python3 -m json.tool
```

Check `calibrated` in the response. If true, calibration succeeded.

## Step 3: Verification

### 3.1 Edge trace verification (~20s)

Tell the user:
> The arm will trace the phone screen border clockwise — moving to 8 edge points and pausing at each. Watch and confirm it follows the screen edges accurately.

```bash
curl -s -X POST http://localhost:8048/api/calibrate/verify-edge --max-time 60 | python3 -m json.tool
```

### 3.2 Final status check

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
