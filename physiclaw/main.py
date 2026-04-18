"""PhysiClaw CLI entry point.

Usage:
    uv run physiclaw [--port 8048] [--host 127.0.0.1] [--verbose]
    uv run physiclaw --warm-start [--cam-index 0]

``--warm-start`` resumes from the last saved calibration: if
``data/calibration/bundle.json`` is complete, connect arm + camera and
mark the server ready without running setup.py. Falls back to normal
boot if the bundle is missing or hardware reconnect fails.
"""

import argparse
import atexit
import logging
import subprocess
import sys


def _warm_start_sanity(physiclaw, calib, phone) -> bool:
    """Run a compact end-to-end tap verification. Returns True iff every
    tap landed within tolerance. No-touches counts as failure — warm-start
    only succeeds when we've proven the calibration still holds.

    On failure, logs the specific diagnosis (no touches = /bridge not open;
    touches but off = arm/phone/camera moved) so the caller can stay terse.
    """
    from physiclaw.calibration.calibrate import validate_calibration

    log = logging.getLogger(__name__)
    cal = physiclaw.calibration
    phone.set_mode("calibrate")
    try:
        results = validate_calibration(
            physiclaw.arm,
            physiclaw.cam,
            calib,
            cal.z_tap,
            cal.effective_rotation(),
            cal.pct_to_grbl,
            cal.pct_to_cam,
            cam_size=cal.cam_size,
            num_tests=2,
        )
    finally:
        phone.set_mode("bridge")

    total = len(results)
    received = sum(1 for r in results if r.get("error", 999) < 999)
    passed = sum(1 for r in results if r["passed"])
    if passed == total:
        log.info(f"--warm-start: sanity passed ({passed}/{total} taps)")
        return True
    if received == 0:
        log.error(
            "--warm-start: sanity — no taps registered. "
            "Is the phone's /bridge page open and foregrounded?"
        )
    else:
        log.error(
            f"--warm-start: sanity — {passed}/{total} taps within tolerance "
            f"({received}/{total} touches received). Calibration looks stale "
            f"(arm, phone, or camera likely moved since last setup)."
        )
    return False


def _try_warm_start(cam_index_override: int | None) -> bool:
    """Resume from the saved Calibration bundle.

    The bundle is already loaded into ``physiclaw.calibration`` at
    ``physiclaw.server.app`` import time. Here we connect hardware,
    ask the user to confirm the bridge page is open, run an end-to-end
    sanity tap, and flip the ready flag if every test passes. Any
    failure returns False; the caller exits non-zero so the user can
    fall back to ``uv run physiclaw`` + ``setup.py``.

    Camera index comes from ``--cam-index`` if provided, else from
    ``bundle.cam_index``, else 0.
    """
    from physiclaw.server.app import physiclaw, _calib, _phone

    log = logging.getLogger(__name__)
    cal = physiclaw.calibration
    if not cal.complete:
        log.error("--warm-start: no complete calibration on disk")
        return False
    cam_index = cam_index_override if cam_index_override is not None else (
        cal.cam_index if cal.cam_index is not None else 0
    )
    try:
        physiclaw.connect_arm()
        physiclaw.connect_camera(cam_index)
    except Exception as e:
        log.error(f"--warm-start: hardware reconnect failed: {e}")
        return False

    # Clean shutdown parks the stylus at (0, 0) = screen center, so the
    # fresh setup() on reconnect re-origins there. Warm-start assumes
    # that invariant held; the sanity tap catches cases where it didn't
    # (killed without shutdown, power yank, arm bumped).
    if sys.stdin.isatty():
        print()
        print("━" * 60)
        print("Warm-start")
        print("  Open the phone's /bridge page (keep it foregrounded).")
        print("  The arm will tap 2 random dots to verify the calibration.")
        print("━" * 60)
        input("Press Enter when ready... ")
    else:
        log.info("--warm-start: non-interactive; running sanity immediately")

    if not _warm_start_sanity(physiclaw, _calib, _phone):
        # _warm_start_sanity already logged the specific diagnosis.
        return False

    physiclaw.mark_ready()
    log.info(
        f"--warm-start: resumed from bundle "
        f"(z_tap={cal.z_tap}mm, cam={cam_index}) — MCP tools ready"
    )
    return True


def _spawn_runtime(port: int, verbose: bool) -> subprocess.Popen:
    """Launch the hook loop as a child process.

    Runs out-of-process so long-running hooks (e.g. shelling out to `claude`)
    don't block the MCP event loop. Terminated via atexit when the server
    exits.
    """
    log = logging.getLogger(__name__)
    cmd = [
        sys.executable,
        "-m",
        "agent.runtime",
        "--server",
        f"http://127.0.0.1:{port}",
    ]
    if verbose:
        cmd.append("--verbose")
    proc = subprocess.Popen(cmd)
    log.info(f"Runtime loop started as subprocess (pid={proc.pid})")
    return proc


def main():
    parser = argparse.ArgumentParser(description="PhysiClaw MCP Server")
    parser.add_argument("--port", type=int, default=8048)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed debug output"
    )
    parser.add_argument(
        "--no-runtime",
        action="store_true",
        help="Don't spawn the runtime loop subprocess",
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Auto-connect hardware from the saved calibration bundle and "
        "mark ready, skipping setup.py. Falls through if the bundle is "
        "incomplete or hardware connect fails.",
    )
    parser.add_argument(
        "--cam-index",
        type=int,
        default=None,
        help="Camera index override for --warm-start (default: value "
        "stored in the bundle, falling back to 0)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    from physiclaw.server import mcp, shutdown

    atexit.register(shutdown)

    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.settings.log_level = "WARNING"

    from physiclaw.bridge import bridge_base_urls

    log = logging.getLogger(__name__)
    primary, fallback = bridge_base_urls(args.port)
    display_host = "localhost" if args.host == "0.0.0.0" else args.host
    log.info(f"PhysiClaw MCP server on http://{display_host}:{args.port}/mcp")
    log.info(f"QR code (scan with phone): http://localhost:{args.port}/api/bridge/qr")
    if primary != fallback:
        log.info(f"Phone page: {primary}/bridge  (recommended — survives IP changes)")
        log.info(f"Fallback:   {fallback}/bridge  (if mDNS blocked)")
    else:
        log.info(f"Phone page: {fallback}/bridge")
        log.info(
            "Tip: set a stable LocalHostName for <name>.local URLs — "
            "see /phone-setup"
        )
    if args.warm_start:
        if not _try_warm_start(args.cam_index):
            log.error(
                "Exiting. Re-run without --warm-start and then setup.py to "
                "recalibrate: `uv run physiclaw` then `uv run python scripts/setup.py`."
            )
            sys.exit(1)
    else:
        log.info(
            "Run /setup in Claude Code (or: uv run python scripts/setup.py) "
            "to connect hardware and calibrate — server is waiting."
        )

    if not args.no_runtime:
        runtime_proc = _spawn_runtime(args.port, args.verbose)

        def _stop_runtime():
            if runtime_proc.poll() is None:
                runtime_proc.terminate()
                try:
                    runtime_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    runtime_proc.kill()

        atexit.register(_stop_runtime)

    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
