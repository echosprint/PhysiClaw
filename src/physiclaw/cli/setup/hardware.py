"""``physiclaw setup hardware`` — interactive arm + camera calibration.

Talks to a running ``physiclaw server`` over HTTP.
"""

import base64
import contextlib
import logging
import os
import socket
import sys
import tempfile
import time
import webbrowser
from pathlib import Path
from typing import Annotated

import typer

from physiclaw import paths
from physiclaw.cli import _http
from physiclaw.cli._format import step_fail, step_ok, step_warn
from physiclaw.core import platform

BASE = os.environ.get("PHYSICLAW_SERVER", "http://localhost:8048")


def _viewport_cache_candidates() -> list:
    root = paths.calibration_cache_dir()
    return [root / "viewport.png", root / "viewport.jpg"]


def api(method, path, body=None, timeout=60):
    """This wizard's server call — `_http.api` bound to the (mutable)
    module-global BASE."""
    return _http.api(BASE, method, path, body=body, timeout=timeout)


def ok(r):
    return r is not None and r.get("status") == "ok"


def _msg(r, fallback="no response"):
    """Server error string from a response, with a fallback when absent."""
    return (r or {}).get("message", fallback)


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


def _camera_aim_adjust(prompt: str) -> None:
    """Release the server's camera, open the OS camera-preview app for
    aim, wait for the user, then quit the aim app so the next
    ``/api/connect-camera`` can reacquire. Platform-specific app
    choices live in ``physiclaw.core.platform``.

    Used by the camera-calibration step to re-aim before reading the
    frame. The leading disconnect releases the camera the server has
    held since the connect-camera step — Windows Media Foundation
    enforces exclusive access, so without it the OS Camera app shows
    "another app is using the camera". It's idempotent if no camera is
    connected (the server returns ``released=False``)."""
    api("POST", "/api/disconnect-camera")
    platform.open_camera_aim_app()
    # The aim-app launch is best-effort on some platforms (e.g. Linux); the
    # backend supplies a fallback instruction when one is warranted.
    aim_hint = platform.camera_aim_hint()
    if aim_hint:
        print(f"  {aim_hint}")
    wait(prompt)
    platform.quit_camera_aim_app()


def ask(msg, auto):
    # Prompt label matches `wait()`'s `[Enter]` for visual consistency.
    # `q` still quits — kept as a silent safety affordance, not advertised.
    return True if auto else input(f"  {msg} [Enter] ").strip().lower() != "q"


def calibrate(step, timeout=60, body=None):
    return api("POST", f"/api/calibrate/{step}", body=body, timeout=timeout)


def calibrate_retry(step, fail_msg, retry_prompt, auto, predicate=None, timeout=30, body=None):
    if predicate is None:
        predicate = ok
    while True:
        r = calibrate(step, timeout, body=body)
        if predicate(r):
            return r
        msg = fail_msg(r) if callable(fail_msg) else fail_msg
        _fail(msg)
        if auto or not ask(retry_prompt, auto=False):
            sys.exit(1)


def _done(msg="OK"):
    print(step_ok(msg))


def _fail(msg):
    print(step_fail(msg))


def _warn(msg):
    print(step_warn(msg))


def run(auto: bool = False, trace: bool = False) -> None:
    # Step names + wording mirror the browser wizard
    # (core/static/setup-hardware.html) so the two surfaces stay consistent.
    t0 = time.time()

    status = api("GET", "/api/status")
    if not status:
        sys.exit("Server not running. Start: physiclaw server")
    if status.get("ready"):
        print("PhysiClaw is already ready.")
        return
    if status.get("calibrated"):
        print("Already calibrated, finalizing...")
        api("POST", "/api/phone/home")
        time.sleep(3)
        api("POST", "/api/ready")
        _done("PhysiClaw is ready")
        return

    # ── 1. Connect phone ──
    print("\n── 1. Connect phone ──")
    if status.get("bridge"):
        _done("Phone connected")
    else:
        print(f"  Phone URL: http://{lan_ip()}:8048/bridge")
        if not auto:
            webbrowser.open(f"{BASE}/api/bridge/qr")
            wait("Scan the QR on your phone — the page should say 'PhysiClaw'")
        _done("Phone connected")

    # ── 2. Position the rig ──
    print("\n── 2. Position the rig ──")
    print("  1. Connect the control board — USB to the computer, plus 12 V power.")
    print("  2. Connect the camera to the computer over USB.")
    print("  3. Seat the phone in the holder, top-left corner against the holder's corner;")
    print("     keep the screen level and facing straight up.")
    print("  4. Keep the phone unlocked, with the bridge page in the foreground.")
    if not auto:
        wait("Everything in place?")
    _done("Rig in place")

    # ── 3. Connect the arm ──
    print("\n── 3. Connect the arm ──")
    print("  Control board connected over USB with its 12 V power on — PhysiClaw")
    print("  scans the computer's serial ports to find it (FluidNC firmware).")
    if ask("Ready?", auto):
        if not ok(api("POST", "/api/connect-arm")):
            _fail("Couldn't connect — check the USB cable and 12 V power")
            sys.exit(1)
    _done("Arm connected")

    # ── 4. Connect the camera ──
    print("\n── 4. Connect the camera ──")
    print("  Camera directly above the phone. PhysiClaw draws colored corner markers")
    print("  on the bridge page, then scans the cameras and picks the one that sees")
    print("  them (keep /bridge open + awake).")
    r = api("POST", "/api/connect-camera", {"index": "auto"}, timeout=60)
    if ok(r):
        cam = r.get("index", 0)
        _done(f"Camera {cam} connected")
    else:
        tmp_dir = Path(tempfile.gettempdir())
        for stale in tmp_dir.glob("physiclaw_cam*.jpg"):
            stale.unlink(missing_ok=True)
        preview_paths: list[str] = []
        for i in range(4):
            rr = api("GET", f"/api/camera-preview/{i}?watermark=1", timeout=10)
            if rr and rr.get("image"):
                p = tmp_dir / f"physiclaw_cam{i}.jpg"
                p.write_bytes(base64.b64decode(rr["image"]))
                preview_paths.append(str(p))
        if auto:
            cam = 0
        else:
            platform.open_image_files(preview_paths)
            try:
                cam = int(
                    input(
                        "  Couldn't auto-detect. Which camera? [0-3, default=0]: "
                    ).strip()
                )
            except ValueError:
                cam = 0
        if not ok(api("POST", "/api/connect-camera", {"index": cam})):
            _fail("Couldn't find the camera — make sure /bridge is open and awake")
            sys.exit(1)
        _done(f"Camera {cam} connected")

    # ── 5. Locate the screen ──
    print("\n── 5. Locate the screen ──")
    print("  Lines up where PhysiClaw draws with the real screen, so taps land right.")
    # Cache policy: interactive setup always re-measures; --auto trusts
    # the cached screenshot at ~/.physiclaw/calibration/cache/viewport.png
    # if it exists.
    vp_cache = next(
        (p for p in _viewport_cache_candidates() if p.exists()), None
    )
    if auto and vp_cache is not None:
        print(f"  Using cached screenshot: {vp_cache} (delete to re-measure)")
    else:
        if vp_cache is not None:
            print(f"  Cached screenshot at {vp_cache} ignored (interactive: fresh measurement).")
        print("  The phone shows an orange square. Tap AssistiveTouch once (screenshot),")
        print("  then double-tap it (upload).")
    while True:
        if ok(calibrate("viewport-shift", 35, body={"fresh": not auto})):
            break
        wait("Couldn't read the screenshot. Tap AT once, then double-tap. Retry?")
    _done("Screen located")

    # ── 6. Calibrate the arm (position stylus, then tap 18 points) ──
    print("\n── 6. Calibrate the arm ──")
    r = api("POST", "/api/bridge/switch", {"mode": "calibrate", "phase": "center"})
    if not r or not r.get("ok"):
        _fail("Couldn't show the center circle — is the bridge page open and awake?")
        sys.exit(1)
    time.sleep(0.5)
    print("  The phone shows an orange circle at screen center.")
    if not auto:
        wait("Move the stylus tip over the orange circle, then continue")
    print("  The arm taps 18 points to learn how its motion lines up with the screen.")
    if ask("Don't touch the rig. Ready?", auto):
        def _arm_fail(resp):
            return (
                "Couldn't calibrate: "
                f"{_msg(resp)} — "
                "make sure the stylus tip is over the center circle"
            )

        # In auto mode the stylus is parked off-screen — tell the server to
        # drive it onto the screen center first (mirrors the wizard's auto).
        r = calibrate_retry(
            "arm", _arm_fail, "Retry?", auto, timeout=120,
            body={"from_park": True} if auto else None,
        )
        tilt = r.get("tilt_ratio", 0)
        if not r.get("aligned"):
            _warn(
                f"Phone looks slightly rotated relative to the arm ({tilt*100:.1f}%) — "
                "straighten it and rerun if validation fails later"
            )
        _done(f"Arm calibrated — mapped {r.get('pairs')} points")

    # ── 7. Calibrate the camera (rotation/coverage check, then 15-dot mapping) ──
    print("\n── 7. Calibrate the camera ──")
    print("  Keep the whole screen in view, evenly lit and free of glare.")
    # Interactive only: aiming with the OS camera app releases the device, so
    # reopen it afterwards. In auto mode the camera stays connected from step 4
    # — mirror the browser wizard, which calibrates the still-connected camera
    # with no reopen (calibrateCamera in core/static/setup-hardware.html).
    if not auto:
        _camera_aim_adjust("Adjust the camera angle/distance if needed")
        r_conn = api("POST", "/api/connect-camera", {"index": cam})
        if not ok(r_conn):
            _fail(
                f"Couldn't reopen the camera: {_msg(r_conn)}. "
                "Another app (Photo Booth / Camera / Zoom / FaceTime) may still be holding it."
            )
            sys.exit(1)
    r = calibrate("camera", 15)
    if not ok(r):
        _fail(f"Couldn't read the camera: {_msg(r)}")
        sys.exit(1)
    for issue in r.get("issues") or []:
        _warn(issue)
    print(f"  rotation {r.get('rotation_name')}, coverage {r.get('coverage'):.0%}")
    m = calibrate_retry(
        "camera-mapping",
        lambda r: (
            "Couldn't map the dots: "
            f"{_msg(r)}"
        ),
        "Reduce glare / fix lighting. Retry?",
        auto,
    )
    _done(f"Camera calibrated — found all {m.get('dots', 15)} dots")

    # ── 8. Validate ──
    print("\n── 8. Validate ──")
    print("  For each dot: find it with the camera, tap it with the arm, and compare")
    print("  the tap to where the dot was drawn. Passing saves the calibration.")
    if ask("Ready?", auto):
        r = calibrate("validate", 60)
        if not (r and r.get("calibrated")):
            _fail(
                "Validation failed — check the lighting, or redo the camera or arm "
                "calibration"
            )
            sys.exit(1)
        _done(
            f"Validated — {r.get('passed')}/{r.get('total')} taps on target. "
            "Calibration saved."
        )

    # ── 9. Verify AssistiveTouch ──
    print("\n── 9. Verify AssistiveTouch ──")
    print("  Confirms the screenshot + clipboard pipeline works.")
    calibrate("assistive-touch/show")
    if not auto:
        wait("Drag the AssistiveTouch button over the orange circle")

    def _at_fail(resp):
        msg = (
            "Couldn't verify — re-position the AssistiveTouch button over the circle "
            "and check the iOS Shortcuts"
        )
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
    _done("Screenshot + clipboard verified")

    if trace:
        print("\n── Edge trace ──")
        print("  Arm traces phone screen border clockwise, pausing at 8 points.")
        if ask("Watch for accuracy. Ready?", auto):
            calibrate("trace-edge", 60)
        _done("Edge trace complete")

    # ── 10. Finish ──
    print("\n── 10. Finish ──")
    api("POST", "/api/phone/home")
    time.sleep(3)
    api("POST", "/api/ready")

    elapsed = time.time() - t0
    mins, secs = int(elapsed // 60), int(elapsed % 60)
    print(f"\n{'='*40}")
    _done(f"PhysiClaw is ready — set up in {mins}m {secs}s.")
    print("  The arm, camera, and screen are calibrated and working together.")
    print("  All MCP tools are now available.")
    print(f"{'='*40}")


def await_bridge_and_calibrate(host: str, port: int) -> None:
    """Wait for the server, then for the phone bridge to open, then run the
    calibration wizard unattended — the engine behind ``physiclaw auto``.

    Runs in a daemon thread off the MCP event loop and drives ``run(auto=True)``
    over HTTP against this server, with the wizard's ``print()`` output
    re-tagged as ``[setup wizard]`` (via :class:`LineLogStream`) so it joins
    the tagged log stream instead of clashing with it. No desktop wizard is
    opened; the operator just opens ``/bridge`` on the phone.
    """
    global BASE
    log = logging.getLogger(__name__)
    # Lazy: importing anything under core.server builds the app singletons,
    # which the plain `physiclaw setup hardware` process must not trigger —
    # this worker only ever runs inside the server process.
    from physiclaw.core.server.net import wait_for_port

    if not wait_for_port(host, port):
        log.error("auto: server never started accepting connections; "
                  "skipping auto-calibration.")
        return

    BASE = f"http://localhost:{port}"
    log.info("auto: waiting for the phone bridge to connect…")
    while True:
        status = api("GET", "/api/status")
        if status and (
            status.get("bridge") or status.get("calibrated") or status.get("ready")
        ):
            break
        time.sleep(1.0)

    log.info("auto: phone bridge connected — starting calibration.")
    from physiclaw.core.logger import LineLogStream, make_tagged_logger

    wizard_log = make_tagged_logger("physiclaw.setup_wizard", "setup wizard")
    try:
        with contextlib.redirect_stdout(LineLogStream(wizard_log)):
            run(auto=True)
    except SystemExit as e:
        log.error("auto-calibration stopped: %s", e)
    except Exception:  # noqa: BLE001 — a failed setup must not crash the server
        log.exception("auto-calibration failed")


def hardware(
    auto: Annotated[
        bool,
        typer.Option("-a", "--auto", help="Auto mode: skip prompts."),
    ] = False,
    trace: Annotated[
        bool,
        typer.Option("--trace", help="Run the optional edge-trace step."),
    ] = False,
    server_url: Annotated[
        str,
        typer.Option(
            "--server-url",
            help="Running MCP server URL. Defaults to $PHYSICLAW_SERVER or "
            "http://localhost:8048.",
        ),
    ] = BASE,
) -> None:
    """Calibrate the robotic arm + camera (server must be running)."""
    global BASE
    BASE = server_url
    run(auto=auto, trace=trace)
