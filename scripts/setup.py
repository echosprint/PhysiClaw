"""PhysiClaw setup. Usage: uv run python scripts/setup.py [-y]"""

import base64
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

BASE = "http://localhost:8048"
PEN_CACHE = Path("data/calibration/cache/z-tap")
VIEWPORT_CACHE_CANDIDATES = [
    Path("data/calibration/cache/viewport.png"),
    Path("data/calibration/cache/viewport.jpg"),
]


def api(method, path, body=None, timeout=60):
    data = json.dumps(body).encode() if body else (b"" if method == "POST" else None)
    hdrs = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read())
        except Exception:
            return None
    except Exception:
        return None


def ok(r):
    return r is not None and r.get("status") == "ok"


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def wait(msg):
    input(f"  {msg} [Enter] ")


def ask(msg, auto):
    return True if auto else input(f"  {msg} [Enter/q] ").strip().lower() != "q"


def calibrate(step, timeout=60):
    return api("POST", f"/api/calibrate/{step}", timeout=timeout)


def calibrate_retry(step, fail_msg, retry_prompt, auto, predicate=None, timeout=30):
    """Run a calibration step in a retry loop until ``predicate(r)`` passes.

    ``fail_msg`` may be a string or a callable taking the response.
    Exits on user 'q' (manual) or immediately (auto) without retry.
    """
    if predicate is None:
        predicate = ok
    while True:
        r = calibrate(step, timeout)
        if predicate(r):
            return r
        msg = fail_msg(r) if callable(fail_msg) else fail_msg
        fail(msg)
        # In auto mode there's no human to fix the physical setup, so just exit.
        if auto or not ask(retry_prompt, auto=False):
            sys.exit(1)


def done(msg="OK"):
    print(f"  \033[32m✓\033[0m {msg}")


def fail(msg):
    print(f"  \033[31m✗ {msg}\033[0m")


def main():
    auto = "-y" in sys.argv
    t0 = time.time()

    # Pre-check
    status = api("GET", "/api/status")
    if not status:
        sys.exit("Server not running. Start: uv run physiclaw")
    if status.get("ready"):
        print("Already ready.")
        return
    if status.get("calibrated"):
        # Calibration cached but server restarted — just finish the last step.
        print("Already calibrated, finalizing...")
        api("POST", "/api/phone/home")
        time.sleep(3)  # let home-screen animation settle
        api("POST", "/api/ready")
        done("Phone on Home Screen, PhysiClaw ready")
        return

    # 1a. Scan QR
    print("\n── 1a. Scan QR code ──")
    print(f"  Phone URL: http://{lan_ip()}:8048/bridge")
    if not auto:
        webbrowser.open(f"{BASE}/api/bridge/qr")
        wait("Scan QR on phone, confirm page shows 'PhysiClaw'")
    done("Phone page ready")

    # 1b. Position phone
    print("\n── 1b. Position phone ──")
    if not auto:
        subprocess.run(["open", "-a", "Photo Booth"])
        wait("Place phone under camera, adjust in Photo Booth, then close Photo Booth")
        done("Phone positioned")
    else:
        done("Skipped (auto)")

    # 2. Connect arm
    print("\n── 2. Connect arm ──")
    if ask("USB plugged, power ON, stylus on?", auto):
        if not ok(api("POST", "/api/connect-arm")):
            fail("Arm connection failed"); sys.exit(1)
    done("Arm connected")

    # 3. Connect camera
    print("\n── 3. Connect camera ──")
    subprocess.run("rm -f /tmp/physiclaw_cam*.jpg", shell=True)
    for i in range(4):
        r = api("GET", f"/api/camera-preview/{i}?watermark=1", timeout=10)
        if r and r.get("image"):
            with open(f"/tmp/physiclaw_cam{i}.jpg", "wb") as f:
                f.write(base64.b64decode(r["image"]))
    if auto:
        cam = 0
    else:
        subprocess.run("open /tmp/physiclaw_cam*.jpg", shell=True)
        try:
            cam = int(input("  Which camera? [0-3, default=0]: ").strip())
        except ValueError:
            cam = 0
    if not ok(api("POST", "/api/connect-camera", {"index": cam})):
        fail("Camera connection failed"); sys.exit(1)
    done(f"Camera {cam} connected")

    # 4. Viewport shift
    print("\n── 4. Viewport shift ──")
    vp_cache = next((p for p in VIEWPORT_CACHE_CANDIDATES if p.exists()), None)
    if vp_cache is not None:
        print(f"  Using cached screenshot: {vp_cache} (delete to re-measure)")
    else:
        print("  Phone shows an orange square.")
        print("  Tap AssistiveTouch once (screenshot), then double-tap (upload).")
    while True:
        if ok(calibrate("viewport-shift", 35)):
            break
        wait("Failed. Tap AT once, then double-tap. Ready to retry?")
    done("Viewport shift measured")

    # 5. Position stylus
    print("\n── 5. Position stylus ──")
    r = api("POST", "/api/bridge/switch", {"mode": "calibrate", "phase": "center"})
    if not r or not r.get("ok"):
        fail("Failed to show orange circle on phone — is the bridge page open?")
        sys.exit(1)
    time.sleep(0.5)  # let phone poll and render
    print("  Phone should show an orange circle at screen center.")
    print("  If screen is off, wake the phone and reopen the bridge page.")
    if not auto:
        wait("Position stylus tip above the orange circle (~3mm above screen)")
    done("Stylus positioned")

    # 6. Pen depth
    print("\n── 6. Pen depth ──")
    print("  Arm probes downward to find screen surface.")
    if ask("Don't touch anything. Ready?", auto):
        r = calibrate("pen-depth", 30)
        if not ok(r):
            fail("Pen depth probe failed"); sys.exit(1)
        if r.get("cached"):
            done(f"Pen depth loaded from cache: {PEN_CACHE} (delete to re-measure)")
        else:
            done("Pen depth measured")

    # 7. Arm tilt
    print("\n── 7. Arm tilt ──")
    print("  Arm taps two points to check alignment with phone.")
    if ask("Ready?", auto):
        def _tilt_fail(resp):
            tilt = (resp or {}).get("tilt_ratio")
            detail = f"tilt {tilt*100:.1f}%" if tilt is not None else "no response"
            return f"Not aligned — {detail}. Adjust phone rotation"

        calibrate_retry(
            "arm-tilt",
            _tilt_fail,
            "Retry?",
            auto,
            predicate=lambda resp: resp and resp.get("aligned"),
        )
        done("Arm aligned with phone")

    # 8. Camera rotation
    print("\n── 8. Camera rotation ──")
    print("  Check camera angle: phone edges should be parallel to image edges.")
    if not auto:
        subprocess.run(["open", "-a", "Photo Booth"])
        wait("Adjust camera if needed, then close Photo Booth")
    api("POST", "/api/connect-camera", {"index": cam})
    r = calibrate("camera-rotation", 10)
    if r and r.get("ok") is False:
        fail(f"Issues: {r.get('issues', [])}")
    else:
        done("Camera rotation OK")

    # 9. Frame rotation
    print("\n── 9. Frame rotation ──")
    print("  Detecting UP/RIGHT markers for software rotation.")
    if ask("Ready?", auto):
        if not ok(calibrate("frame-rotation", 15)):
            fail("Frame rotation failed"); sys.exit(1)
    done("Frame rotation set")

    # 10. GRBL mapping
    print("\n── 10. GRBL mapping ──")
    print("  Arm taps up to 18 points across the screen for precise mapping.")
    if ask("Don't touch anything. Ready?", auto):
        if not ok(calibrate("grbl-mapping", 60)):
            fail("GRBL mapping failed"); sys.exit(1)
    done("Screen→arm mapping computed")

    # 11. Camera mapping
    print("\n── 11. Camera mapping ──")
    print("  Camera detects 15 red dots on phone screen.")
    if ask("Ready?", auto):
        calibrate_retry(
            "camera-mapping",
            lambda r: f"Camera mapping failed: {(r or {}).get('message', 'no response')}",
            "Adjust lighting/glare. Retry?",
            auto,
        )
    done("Screen→camera mapping computed")

    # 12. Validate
    print("\n── 12. Validate ──")
    print("  Arm taps random dots and compares touch vs expected position.")
    if ask("Ready?", auto):
        r = calibrate("validate", 60)
        if not (r and r.get("calibrated")):
            print(f"  {json.dumps(r, ensure_ascii=False) if r else 'no response'}")
            fail("Validation failed"); sys.exit(1)
    done("Calibration validated")

    # 13. AssistiveTouch
    print("\n── 13. AssistiveTouch ──")
    print("  Verifying screenshot + clipboard pipeline.")
    calibrate("assistive-touch/show")
    if not auto:
        wait("Drag AssistiveTouch button to overlap the orange circle")

    def _at_fail(resp):
        msg = "AT verification failed — check AT position and iOS Shortcuts"
        clip = (resp or {}).get("clipboard") or {}
        if clip.get("fetched"):
            msg += f" (clipboard fetched: {clip.get('text')!r})"
        return msg

    r = calibrate_retry(
        "assistive-touch/verify",
        _at_fail,
        "Adjust AT position. Retry?",
        auto,
        predicate=lambda resp: resp and resp.get("passed"),
        timeout=20,
    )
    if r.get("clipboard", {}).get("fetched"):
        print(f"  Clipboard text: {r['clipboard'].get('text')}")
        if not auto:
            wait("Paste in Notes to verify it matches")
    done("Screenshot + clipboard pipeline verified")

    # 14. Edge trace
    print("\n── 14. Edge trace ──")
    print("  Arm traces phone screen border clockwise, pausing at 8 points.")
    if ask("Watch for accuracy. Ready?", auto):
        calibrate("trace-edge", 60)
    done("Edge trace complete")

    # 15. Go to Home Screen + mark ready
    print("\n── 15. Home Screen ──")
    api("POST", "/api/phone/home")
    time.sleep(3)  # let home-screen animation settle
    api("POST", "/api/ready")
    done("Phone on Home Screen, PhysiClaw ready")

    # 16. Summary
    elapsed = time.time() - t0
    mins, secs = int(elapsed // 60), int(elapsed % 60)
    print(f"\n{'='*40}")
    print(f"  Setup completed in {mins}m {secs}s")
    done("PhysiClaw is ready. All MCP tools available.")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
