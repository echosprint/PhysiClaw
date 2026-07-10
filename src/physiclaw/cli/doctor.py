"""``physiclaw doctor`` — read-only health check.

Server-aware: if a PhysiClaw server is running, the live ``/api/status``
response wins over local probes (the server holds the serial port and
camera, so re-probing them would either fail with "busy" or break the
server). When the server is offline, doctor actively probes the GRBL
arm and enumerates cameras.
"""

import logging
import platform
import shutil
import sys
from typing import Annotated

import typer

from physiclaw import __version__, paths, runtime_state
from physiclaw.cli._format import info as _fmt_info
from physiclaw.cli._format import ok as _fmt_ok
from physiclaw.cli._format import section as _fmt_section
from physiclaw.cli._format import warn as _fmt_warn
from physiclaw.cli._update_check import maybe_print_update_banner


def _opencv_import_error() -> str | None:
    """Return a formatted hint if ``import cv2`` fails, else None.

    The platform backend augments with OS-specific remediation (e.g. the
    libGL/glib system libs the manylinux wheel needs on minimal Linux).
    """
    try:
        import cv2  # noqa: F401
    except ImportError as e:
        from physiclaw.core import platform as os_platform

        line = f"OpenCV (cv2) import failed — {e}"
        extra = os_platform.opencv_import_hint(e)
        if extra:
            line += extra
        return _fmt_warn(line)
    return None


def _list_cameras(max_index: int = 4) -> list[int]:
    # Each failed index on macOS can block 1–3s in AVFoundation, so stop
    # after a couple of consecutive misses.
    import cv2

    from physiclaw.core.hardware.camera import silenced_stderr

    found: list[int] = []
    misses = 0
    with silenced_stderr():
        for i in range(max_index + 1):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                found.append(i)
                misses = 0
            else:
                misses += 1
            cap.release()
            if misses >= 2:
                break
    return found


def _probe_server(live: dict | None) -> tuple[str, int, bool, dict | None]:
    """GET /api/status from the server at ``live['host']:live['port']``,
    falling back to ``config.toml`` when no live state is passed.

    Returns ``(display_host, port, bind_all, status)``. ``bind_all`` is
    True when the configured host was the 0.0.0.0 wildcard — display_host
    is rewritten to "localhost" in that case so output reads cleanly,
    but the flag lets doctor surface the exposure warning separately.

    Caller-supplied ``live`` so doctor runs one ``read_live()`` per
    invocation — used twice (server probe + Provider section).
    """
    # Lazy: httpx import is ~100ms — paid only when doctor actually runs,
    # not on every `physiclaw --help`.
    import httpx

    from physiclaw.core import platform

    if live:
        host, port = live["host"], live["port"]
    else:
        from physiclaw.config import CONFIG

        host, port = CONFIG.server.host, CONFIG.server.port
    bind_all = host == "0.0.0.0"
    if bind_all:
        host = "localhost"
    try:
        r = httpx.get(
            f"http://{host}:{port}/api/status",
            timeout=1.0,
            trust_env=platform.TRUST_PROXY_ENV,
        )
        return (host, port, bind_all, r.json())
    except (httpx.HTTPError, ValueError):
        return (host, port, bind_all, None)


# --- Deep probes (only run with --deep). ------------------------------------
# Each returns a pre-formatted line ready for typer.echo. Failures are
# non-fatal — doctor reports them and moves on.


def _probe_vision_model_deep() -> str:
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(paths.omniparser_onnx()), providers=["CPUExecutionProvider"]
        )
        shape = sess.get_inputs()[0].shape
        return _fmt_ok(f"vision model: loads OK (input {shape})")
    except (OSError, RuntimeError, ValueError, ImportError) as e:
        return _fmt_warn(f"vision model: load failed — {type(e).__name__}: {e}")


def _probe_camera_frame(index: int) -> str:
    """cap.read() catches the macOS-TCC-denied case (isOpened lies).

    Reproduces the server's real capture negotiation (backend + FOURCC +
    size + exposure via `configure_capture`) so the frame it meters is
    the frame sessions will see, then judges it with `quality.assess` —
    catching a blown/blurry rig at doctor time instead of mid-task."""
    import cv2

    from physiclaw.config import CONFIG
    from physiclaw.core import platform as os_platform
    from physiclaw.core.hardware.camera import configure_capture, silenced_stderr

    with silenced_stderr():
        cap = cv2.VideoCapture(index, os_platform.camera_backend())
        try:
            configure_capture(
                cap,
                exposure_auto=CONFIG.camera.auto_exposure,
                exposure=CONFIG.camera.exposure,
            )
            frame = None
            # ~10 reads: the first frames are pre-auto-exposure garbage.
            for _ in range(10):
                ok, got = cap.read()
                if ok and got is not None:
                    frame = got
            if frame is None:
                return _fmt_warn(
                    f"camera {index}: opens but no frame "
                    f"({os_platform.camera_denied_hint()})"
                )
        finally:
            cap.release()
    h, w = frame.shape[:2]
    good, clause = _frame_quality_verdict(frame)
    line = f"camera {index}: frame OK ({w}x{h}) — {clause}"
    return _fmt_ok(line) if good else _fmt_warn(line)


def _frame_quality_verdict(frame) -> tuple[bool, str]:
    """One quality clause for the camera line, cropped to the phone screen
    when the calibration bundle allows it (rotated-frame size must match
    the bundle's cam_size, else the crop coords don't apply)."""
    import cv2

    from physiclaw.core.calibration.state import Calibration
    from physiclaw.core.vision import quality
    from physiclaw.core.vision.util import crop_to_phone_screen

    scope = "whole frame — no calibration crop"
    cal = Calibration.load()  # None on missing/unreadable bundle, logged
    if cal is not None and cal.transforms_ready:
        if cal.effective_rotation() != -1:
            frame = cv2.rotate(frame, cal.effective_rotation())
        h, w = frame.shape[:2]
        if (w, h) == tuple(cal.cam_size):
            frame = crop_to_phone_screen(frame, cal.transforms())
            scope = "phone-screen crop"
        else:
            scope = (
                f"whole frame — size {w}x{h} != calibrated "
                f"{cal.cam_size[0]}x{cal.cam_size[1]}"
            )
    r = quality.assess(frame)
    stats = (
        f"sharpness {r.sharpness:.0f}, clip {r.clip_pct:.0%}, "
        f"median {r.median_luma:.0f} ({scope})"
    )
    if r.blown:
        return False, f"quality BLOWN — overexposed/glare: {stats}"
    if r.blurry:
        return False, f"quality BLURRY — check focus: {stats}"
    return True, f"quality ok: {stats}"


def _probe_calibration_deep() -> str:
    # `complete` is a @property on `Calibration`, not a serialized field,
    # so `data.get("complete")` was always None — bundles that finished
    # the full setup were still reported as "partial". Reconstruct the
    # dataclass and check the live property instead.
    from physiclaw.core.calibration.state import Calibration

    data = paths.load_calibration_bundle()
    if data is None:
        return _fmt_warn("calibration: parse failed or not a JSON object")
    try:
        cal = Calibration.from_dict(data)
    except (TypeError, ValueError, KeyError) as e:
        return _fmt_warn(f"calibration: bundle unreadable — {e}")
    if cal.complete:
        return _fmt_ok("calibration: bundle complete")
    missing = [k for k, v in data.items() if v is None]
    return _fmt_warn(
        f"calibration: partial (missing: {missing})" if missing
        else f"calibration: partial (keys: {sorted(data.keys())})"
    )


def _probe_bridge_deep(host: str, port: int) -> str:
    import httpx  # cached after _probe_server's first import

    from physiclaw.core import platform

    try:
        r = httpx.get(
            f"http://{host}:{port}/api/bridge/state",
            timeout=1.0,
            trust_env=platform.TRUST_PROXY_ENV,
        )
        connected = r.json().get("connected", False)
    except (httpx.HTTPError, ValueError) as e:
        # Network/parse failure is a real problem — keep the warn.
        return _fmt_warn(f"bridge: {type(e).__name__}: {e}")
    if connected:
        return _fmt_ok("bridge: phone connected")
    # "not paired yet" is a transient state (phone hasn't opened /bridge),
    # not a broken configuration — surface the fact without `!`.
    return _fmt_info("bridge: phone not paired yet")


def _skills_lines() -> list[str]:
    """One formatted line per installed skill. CLI-installed skills carry
    a ``.installed-from`` JSON marker (see ``cli/skills.py``); user-authored
    dirs are valid too and render as ``(local)``."""
    from physiclaw.cli.skills import (
        PROVENANCE_FILE,
        installed_skill_dirs,
        read_provenance,
    )

    entries = installed_skill_dirs()
    if not entries:
        return [f"  (none installed in {paths.skills_dir()})"]
    out: list[str] = []
    for d in entries:
        prov = read_provenance(d)
        if prov is not None:
            ref = prov.get("ref") or (prov.get("sha", "") or "")[:7] or "?"
            out.append(_fmt_ok(f"{d.name}  ← {prov.get('source', '?')} @ {ref}"))
        elif (d / PROVENANCE_FILE).exists():
            # Marker present but unparseable — rare, worth flagging
            # separately from user-authored so the user can fix it.
            out.append(_fmt_warn(f"{d.name}: provenance unreadable"))
        else:
            out.append(_fmt_ok(f"{d.name}  (local)"))
    return out


# 96x24 PNG, "PhysiClaw" black-on-white in 14pt Arial Bold. ~1.1 KB.
# The probe asks the model to read the word back — confirms not just
# that the wire accepts image_url, but that the model can actually see
# (and OCR) what it received.
_PROBE_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAGAAAAAYCAIAAACKi2/DAAAEQ0lEQVR42u2Y2yt0exjHv2stLcyi"
    "aCQih2kKpZkmkVvmDpFElEYM01zIP+BwxYUYSU5RbswIFw45xIU7KXKDnJXcGKdmxiGnMfPsi197"
    "7dnjfXdv70vZmu/d+v2eeQ6f+f2e9bQ4IkJAPxcfQBAAFAAUABQA9P8BFBQUxHEcx3E8zwcHB6tU"
    "qp6eHrbFcVxUVNSHRPVzdXBwUFFRER0dLYpiSkpKe3u7x+P58KC/Kfq3BEF4bzM5OcmmAaVSSR+t"
    "jY2N8PBwv4g1NTVs95OC/rp+DOj+/t7r9drt9tLSUgCVlZVyrisrK2q1WpKkuro6t9tNRHq9HsDC"
    "wgLzUF9fD2BoaIiIRkZGUlNTQ0JCoqOjDQbD7e3t+7K1Wi2A/Pz8o6Mjl8vV1NTEGO3t7flZLi4u"
    "6nS60NDQsLAwvV5/fHxMRBkZGYIguFwuOXRLSwsRHR4eAsjNzf0sQB6P5/LysrCw0BdQSEiI779t"
    "s9mIaHh4GIDJZGIekpKSBEG4vr7e3NzkOM73XBgMBj9A+/v7ACRJcjqdcg55eXnd3d1PT0++lna7"
    "PTg42NdbXl4eETU3NwOYmZkhIo1GA0Cv1xNRf38/AIvF8imA/DQ/Py8P3G1tbQ6Hw2g0AmhoaCAi"
    "h8MhimJsbKzX693e3pZTHB8fB2A0Gl9fX/2j/l327OwsAJ1O99P83l0xr9e7u7sLQK1WE9Ha2hrL"
    "xOl08jzPcZwkSW63u6SkhHW3zwLEcZxCodBqtaOjo3Kuoii+vb0R0eTkJIDa2lq2VVBQAGB9fb2t"
    "rQ3A4OAgEd3c3CQnJ7MK2aG4v7/3K3t6ehqAVqv9FUB3d3dWq9VsNqelpQGIj48nIo/Ho1Qq09PT"
    "5+bmABQVFQHY2NhQKpUqleoTe9B/58oKMxqN7NFmswFobGzMzs4WBOHy8pKtu1yuoaGh8vLypKQk"
    "ABkZGX6utra2ACgUCofDIQcqKSnp6Oh4fHz0tTw/P1er1Tqdbnx8/PT0lOf5hIQEZl9RUQGgqqqK"
    "47j19XV2l+UD/iUAPTw8KBSKxMREnudzcnLe/9Zut0dERABgfVp25fV6U1NTARQUFJycnNzd3bW2"
    "trLzu7u762tpsVgAVFdX397eDgwMAEhMTGTOR0dHAQiCoNFoiCgmJoZVsby8/FUAEVFZWRm7m/39"
    "/Wylr6/Pr5fJvcbX1erqqkKh8LM0m81+lmNjY/8MbzwfFhYme7i6uuJ5HkB9fT0RsTevJEnPz89f"
    "CBBb4Xn+4uJCXuzu7k5LS2Ov+dLS0rOzsx+23p2dneLi4sjIyNDQ0PT09K6uLo/H42f59vZmNpsj"
    "IiLi4uKsVmtxcTGAo6MjZpaVlQVgYmKCiHp7ewEUFhZ+yhz021paWgKQmZlJ30tBfz6LPz09ud3u"
    "zs5OAPJF+z76c8ZTU1OiKEZGRppMppeXl292grjAJ9fA544AoACgAKCvq78Atz3+gKHHygsAAAAA"
    "SUVORK5CYII="
)


_PROBE_IMAGE_WORD = "PhysiClaw"


def _probe_provider_deep(provider_id: str, model_id: str) -> str:
    """One round-trip with text + a tiny inline image — exercises the
    full path PhysiClaw needs (network, auth, billing, vision input,
    text output). PhysiClaw requires vision on every peek, so a text-
    only probe wouldn't add coverage.

    The probe asks the model to read the word in the image (`PhysiClaw`)
    and validates the reply contains it. "Non-empty reply" alone isn't
    enough: some endpoints (e.g. DashScope's compat shim for text-only
    Qwen models) accept image_url parts and silently drop them server-
    side, then hallucinate an answer. Demanding the actual word back
    catches that case.
    """
    import asyncio
    import time

    from physiclaw.agent.engine.dto import (
        ImageBlock,
        SystemMessage,
        TextBlock,
        UserMessage,
    )
    from physiclaw.agent.engine.trace import brief
    from physiclaw.agent.provider import make_provider

    try:
        prov = make_provider(provider_id, model_id)
    except (ValueError, RuntimeError) as e:
        return _fmt_warn(f"{provider_id}/{model_id}: setup — {e}")

    history = [
        SystemMessage(content="You are a one-word reply bot."),
        UserMessage(content=[
            TextBlock(text="What word is in this image? Reply with just the word."),
            ImageBlock(media_type="image/png", data_b64=_PROBE_IMAGE_B64),
        ]),
    ]

    async def _run():
        try:
            return await prov.chat(history, tools=[])
        finally:
            await prov.aclose()

    t0 = time.perf_counter()
    try:
        asst = asyncio.run(_run())
    except Exception as e:
        return _fmt_warn(
            f"{provider_id}/{model_id}: {type(e).__name__}: {brief(str(e), 140)}"
        )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    reply = (asst.content or "").strip().replace("\n", " ")
    if not reply:
        return _fmt_warn(
            f"{provider_id}/{model_id}: empty reply "
            f"(finish={asst.finish_reason}, {elapsed_ms}ms)"
        )
    if _PROBE_IMAGE_WORD.lower() not in reply.lower():
        return _fmt_warn(
            f"{provider_id}/{model_id}: vision check failed — expected "
            f"{_PROBE_IMAGE_WORD!r} in reply, got {brief(reply, 60)!r} "
            f"({elapsed_ms}ms). Likely a text-only model that silently "
            "drops images."
        )
    u = asst.usage
    usage_str = (
        f"{u.prompt_tokens}p+{u.completion_tokens}c"
        if u.prompt_tokens else "no usage"
    )
    return _fmt_ok(
        f"{provider_id}/{model_id}: reply={brief(reply, 40)!r} "
        f"({elapsed_ms}ms, {usage_str})"
    )


def doctor(
    deep: Annotated[
        bool,
        typer.Option(
            "--deep",
            help="Run active probes (model load, camera frame, API call). "
            "May take several seconds.",
        ),
    ] = False,
) -> None:
    """Check environment, hardware, and assets. Report what's missing."""
    paths.ensure_dirs()

    typer.echo(_fmt_section("PhysiClaw doctor"))
    typer.echo(f"  physiclaw:  {__version__}")
    typer.echo(f"  python:     {sys.version.split()[0]} ({sys.executable})")
    typer.echo(
        f"  platform:   {platform.system()} {platform.release()} "
        f"({platform.machine()})"
    )
    uv = shutil.which("uv")
    typer.echo(f"  uv:         {uv or '(not found — not required for runtime)'}")

    typer.echo()
    typer.echo(_fmt_section("Paths"))
    typer.echo(f"  home: {paths.HOME}")

    typer.echo()
    typer.echo(_fmt_section("Config"))
    # If config.toml failed to parse, importing physiclaw.config at the top
    # of this module would have raised — so reaching here means the file is
    # either absent or valid.
    from physiclaw import config as _cfg

    cp = _cfg.config_path()
    if cp.exists():
        typer.echo(_fmt_ok(f"config.toml: {cp}"))
    else:
        typer.echo(_fmt_warn(
            f"no config yet — using built-in defaults. Edit via `physiclaw config edit` ({cp})"
        ))

    typer.echo()
    typer.echo(_fmt_section("Assets"))
    model = paths.omniparser_onnx()
    if model.exists():
        size_mb = model.stat().st_size / 1024 / 1024
        typer.echo(_fmt_ok(f"vision model: {model}  ({size_mb:.1f} MB)"))
        if deep:
            typer.echo(_probe_vision_model_deep())
    else:
        typer.echo(_fmt_warn(
            f"vision model missing: {model}\n"
            "    Run: physiclaw setup local-vision-model"
        ))

    typer.echo()
    typer.echo(_fmt_section("Hardware"))
    # One read_live() per doctor run — reused by _probe_server and the
    # Provider section below.
    live = runtime_state.read_live()
    host, port, bind_all, status = _probe_server(live)
    if status is not None:
        # Server is up — its view of arm/camera/calibration is authoritative.
        typer.echo(_fmt_ok(f"server: running on {host}:{port}"))
        if bind_all:
            # Bind mode is a config decision the user already made; surface
            # the fact without the yellow `!` since a healthy server isn't
            # a warning state. Security notes live in the README.
            typer.echo(_fmt_info("bind: 0.0.0.0 (LAN-reachable; intended for the phone bridge)"))
        for label in ("arm", "camera", "calibrated", "ready"):
            if status.get(label):
                typer.echo(_fmt_ok(f"{label}: yes"))
            else:
                typer.echo(_fmt_warn(f"{label}: no"))
        if deep:
            typer.echo(_probe_bridge_deep(host, port))
    else:
        # Server offline — safe to probe hardware ourselves.
        typer.echo(_fmt_warn("server: not running"))
        typer.echo("  Probing serial ports for GRBL (active $I query, ~2s/port)…")
        from physiclaw.core.hardware.grbl import candidate_ports, detect_grbl

        # detect_grbl logs its own narration; doctor speaks for itself.
        grbl_logger = logging.getLogger("physiclaw.core.hardware.grbl")
        prev_level = grbl_logger.level
        grbl_logger.setLevel(logging.CRITICAL)
        try:
            grbl_port = detect_grbl()
        finally:
            grbl_logger.setLevel(prev_level)
        if grbl_port:
            typer.echo(_fmt_ok(f"GRBL arm: {grbl_port}"))
        else:
            relevant = candidate_ports()
            if relevant:
                typer.echo(_fmt_warn(
                    f"no GRBL detected (saw {len(relevant)} candidate port(s): "
                    f"{', '.join(relevant)})"
                ))
            else:
                typer.echo(_fmt_warn(
                    "no serial ports detected — connect the arm and re-run."
                ))
        from physiclaw.core import platform as os_platform

        cv2_err = _opencv_import_error()
        if cv2_err is not None:
            typer.echo(cv2_err)
        else:
            cams = _list_cameras()
            if cams:
                typer.echo(_fmt_ok(f"cameras: {len(cams)} detected"))
                if deep:
                    for idx in cams:
                        typer.echo(_probe_camera_frame(idx))
            else:
                typer.echo(_fmt_warn(
                    f"cameras: none detected ({os_platform.camera_denied_hint()})"
                ))
        for hint in os_platform.hardware_permission_hints():
            typer.echo(_fmt_warn(hint))

    typer.echo()
    typer.echo(_fmt_section("Calibration"))
    bundle = paths.calibration_bundle()
    if bundle.exists():
        typer.echo(_fmt_ok(f"calibration bundle: {bundle}"))
        if deep:
            typer.echo(_probe_calibration_deep())
    else:
        typer.echo(_fmt_warn(
            f"no calibration yet: {bundle}\n"
            "    Run: physiclaw  (starts the server and opens the hardware-setup wizard)"
        ))

    typer.echo()
    typer.echo(_fmt_section("Engine"))

    # Prefer the live server's recorded choice — resolving here would pull
    # from the current shell's env, which may be missing PHYSICLAW_MODEL
    # even when the server has it set. Fall back to a fresh resolve when no
    # server is running.
    from physiclaw.agent.runtime.launcher import engine_label, resolve
    from physiclaw.config import parse_model_ref

    live_ref = live.get("model_ref") if live else None
    active_ref: str | None = None
    if live_ref:
        active_ref = live_ref
        ref_source = f"live server, {live.get('model_source') or '?'}"
        typer.echo(_fmt_ok(f"{engine_label(active_ref)} ({ref_source})"))
    else:
        try:
            active_ref, ref_source = resolve()
            typer.echo(_fmt_ok(f"{engine_label(active_ref)} (from {ref_source})"))
        except RuntimeError as e:
            typer.echo(_fmt_warn(f"engine: invalid — {e}"))

    active_provider, active_model = (None, None)
    if active_ref:
        try:
            active_provider, active_model = parse_model_ref(active_ref)
        except ValueError:
            pass

    # Surface the active provider's key status. In `--deep` mode, send a
    # real `chat()` round-trip on the active provider (proves network +
    # auth + billing + model response).
    from physiclaw.agent.provider import CLAUDE_CODE_ID, provider_key_status
    if active_provider and active_provider != CLAUDE_CODE_ID:
        masked, source = provider_key_status(active_provider)
        if masked is None:
            typer.echo(_fmt_warn(
                f"{active_provider} api key: (unset) — required for the active model. "
                f"Set via `physiclaw models key {active_provider}` or env var."
            ))
        else:
            typer.echo(_fmt_ok(f"{active_provider} api key: {masked} ({source})"))
            if deep and active_model:
                typer.echo(_probe_provider_deep(active_provider, active_model))

    typer.echo()
    typer.echo(_fmt_section("Skills"))
    for line in _skills_lines():
        typer.echo(line)

    typer.echo()
    typer.echo(_fmt_section("Next steps"))
    steps = []
    if not model.exists():
        steps.append("physiclaw setup local-vision-model")
    if not (status and status.get("ready")):
        if status is None:
            steps.append(
                "physiclaw   (starts the server and opens the hardware-setup wizard)"
            )
        else:
            steps.append(
                "finish setup in the browser wizard (or run: physiclaw setup hardware)"
            )
    for i, step in enumerate(steps, 1):
        typer.echo(f"  {i}. {step}")
    if not steps:
        typer.echo("  All set.")

    maybe_print_update_banner()
