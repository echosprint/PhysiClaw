"""
PhysiClaw setup CLI — friendly wrapper around the server's HTTP endpoints.

The PhysiClaw server (started via `uv run physiclaw`) exposes hardware
setup and calibration as HTTP endpoints. This CLI is a thin wrapper that
POSTs/GETs to those endpoints and pretty-prints the JSON response —
much friendlier than the equivalent curl commands the /setup skill would
otherwise need.

The server must be running. The CLI exits non-zero if a step fails or
the server is unreachable.

Usage:
    uv run physiclaw-setup status
    uv run physiclaw-setup connect-arm
    uv run physiclaw-setup connect-camera               # auto-detect
    uv run physiclaw-setup connect-camera --index 1
    uv run physiclaw-setup camera-preview 0             # save & open JPEG
    uv run physiclaw-setup switch bridge
    uv run physiclaw-setup switch calibrate --phase center
    uv run physiclaw-setup calibrate pen-depth          # run a calibration step
    uv run physiclaw-setup calibrate --list             # list calibration steps
"""

import argparse
import base64
import json
import sys
import urllib.error
import urllib.parse
import urllib.request


# Calibration step name → one-line description.
# Order matches the canonical /setup flow so `--list` reads top-down.
CALIBRATION_STEPS: dict[str, str] = {
    "viewport-shift": "measure viewport→screenshot offset and DPR (pre-cal)",
    "pen-depth": "discover Z depth that just touches the screen",
    "arm-tilt": "measure arm tilt vs screen plane",
    "camera-rotation": "detect physical camera rotation from a frame",
    "frame-rotation": "choose cv2 rotation to apply to camera frames",
    "grbl-mapping": "compute screen 0-1 → GRBL mm affine",
    "camera-mapping": "compute screen 0-1 → camera 0-1 affine",
    "validate": "round-trip validate the calibration chain",
    "trace-edge": "arm traces phone screen border for visual check",
    "assistive-touch/show": "display AT positioning circle + color nonce",
    "assistive-touch/verify": "tap AT, verify screenshot upload via color nonce",
}


# ─── HTTP helpers ────────────────────────────────────────────


def _request(method: str, url: str, body: dict | None, timeout: float):
    """Send an HTTP request, return (status_code, body_text).

    Exits with code 2 if the server is unreachable.
    """
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    elif method == "POST":
        data = b""

    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except (urllib.error.URLError, TimeoutError) as e:
        reason = getattr(e, "reason", e)
        print(f"error: cannot reach server at {url}\n  {reason}", file=sys.stderr)
        print(
            "\nIs the server running? Start it with: uv run physiclaw", file=sys.stderr
        )
        sys.exit(2)


def _print_json(body: str) -> dict | None:
    """Pretty-print JSON if possible. Returns the parsed dict or None."""
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        print(body)
        return None
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
    return parsed if isinstance(parsed, dict) else None


def _call(args, method: str, path: str, body: dict | None = None) -> int:
    """POST/GET to a server endpoint, pretty-print, return exit code."""
    url = f"http://{args.host}:{args.port}{path}"
    status, raw = _request(method, url, body, timeout=args.timeout)
    _print_json(raw)
    return 0 if status < 400 else 1


# ─── Subcommand handlers ─────────────────────────────────────


def cmd_status(args) -> int:
    return _call(args, "GET", "/api/status")


def cmd_connect_arm(args) -> int:
    return _call(args, "POST", "/api/connect-arm")


def cmd_connect_camera(args) -> int:
    return _call(args, "POST", "/api/connect-camera", {"index": args.index})


def cmd_camera_preview(args) -> int:
    """Fetch a single frame and save it to /tmp/physiclaw_camN.jpg."""
    query = "?watermark=1" if args.watermark else ""
    url = f"http://{args.host}:{args.port}/api/camera-preview/{args.index}{query}"
    status, raw = _request("GET", url, None, timeout=args.timeout)
    parsed = _print_json(raw)
    if status >= 400 or parsed is None or parsed.get("status") != "ok":
        return 1
    # Decode and save the base64 JPEG so the user can `open` it
    img_b64 = parsed.get("image")
    if img_b64:
        out_path = f"/tmp/physiclaw_cam{args.index}.jpg"
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(img_b64))
        print(f"\nSaved preview to {out_path}")
    return 0


def cmd_switch(args) -> int:
    body: dict = {"mode": args.mode}
    if args.mode == "calibrate":
        if not args.phase:
            print("error: switch calibrate requires --phase NAME", file=sys.stderr)
            return 1
        body["phase"] = args.phase
        for kv in args.extra or []:
            if "=" not in kv:
                print(f"error: --extra expects key=value, got {kv!r}", file=sys.stderr)
                return 1
            k, v = kv.split("=", 1)
            # Best-effort numeric parse
            try:
                body[k] = float(v) if "." in v else int(v)
            except ValueError:
                body[k] = v
    return _call(args, "POST", "/api/bridge/switch", body)


def cmd_calibrate(args) -> int:
    if args.list:
        _print_calibration_steps()
        return 0
    if args.step_timeout is not None:
        args.timeout = args.step_timeout
    if not args.step:
        print("error: calibrate requires a step name (or --list)", file=sys.stderr)
        return 1
    if args.step not in CALIBRATION_STEPS:
        print(f"error: unknown calibration step '{args.step}'", file=sys.stderr)
        print(
            "\nRun `physiclaw-setup calibrate --list` to see all steps.",
            file=sys.stderr,
        )
        return 1
    return _call(args, "POST", f"/api/calibrate/{args.step}")


def _print_calibration_steps():
    width = max(len(s) for s in CALIBRATION_STEPS)
    print("Calibration steps (run in this order during /setup):\n")
    for name, desc in CALIBRATION_STEPS.items():
        print(f"  {name:<{width}}  {desc}")
    print("\nExample: uv run physiclaw-setup calibrate pen-depth")


# ─── argparse wiring ─────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physiclaw-setup",
        description="Friendly CLI for PhysiClaw setup + calibration HTTP endpoints.",
    )
    parser.add_argument(
        "--host", default="localhost", help="Server host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=8048, help="Server port (default: 8048)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds (default: 60)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show hardware + calibration status")

    sub.add_parser("connect-arm", help="Connect the GRBL stylus arm (auto-detect USB)")

    p_cam = sub.add_parser(
        "connect-camera", help="Connect a camera by index (preview each one first)"
    )
    p_cam.add_argument(
        "--index", type=int, required=True, help="Camera index to connect"
    )

    p_prev = sub.add_parser(
        "camera-preview", help="Capture one frame from a camera index, save to /tmp"
    )
    p_prev.add_argument("index", type=int, help="Camera index to preview")
    p_prev.add_argument(
        "--watermark", action="store_true", help="Draw the camera index as a watermark"
    )

    p_sw = sub.add_parser(
        "switch", help="Switch the phone page mode (bridge | calibrate)"
    )
    p_sw.add_argument("mode", choices=["bridge", "calibrate"])
    p_sw.add_argument(
        "--phase",
        default=None,
        help="Required when mode=calibrate (e.g. center, dot, markers)",
    )
    p_sw.add_argument(
        "--extra",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Extra phase kwargs, repeat for multiple "
        "(e.g. --extra dot_x=0.5 --extra dot_y=0.5)",
    )

    p_cal = sub.add_parser(
        "calibrate", help="Run a calibration step against the server"
    )
    p_cal.add_argument(
        "step",
        nargs="?",
        default=None,
        help="Step name (e.g. pen-depth). Omit and pass --list to see all.",
    )
    p_cal.add_argument(
        "--list", action="store_true", help="List all calibration steps and exit"
    )
    p_cal.add_argument(
        "--timeout",
        dest="step_timeout",
        type=float,
        default=None,
        help="Override request timeout (seconds) for this step",
    )

    return parser


_DISPATCH = {
    "status": cmd_status,
    "connect-arm": cmd_connect_arm,
    "connect-camera": cmd_connect_camera,
    "camera-preview": cmd_camera_preview,
    "switch": cmd_switch,
    "calibrate": cmd_calibrate,
}


def main():
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(_DISPATCH[args.command](args))


if __name__ == "__main__":
    main()
