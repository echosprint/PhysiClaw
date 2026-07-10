"""``physiclaw setup local-vision-model`` — install the OmniParser icon detector.

By default we fetch a pre-converted ONNX from the PhysiClaw release mirror —
no Hugging Face download, no conversion deps, no uv. If every mirror is
unreachable (or fails its checksum), fall back to converting from upstream:
download the PyTorch weights and export to ONNX in a throwaway ``uv run
--with`` env (ultralytics + onnx + onnxslim, ~500 MB) so those heavy deps
never enter the physiclaw install — runtime inference only needs
``onnxruntime``, already a core dep. ``--build`` forces the from-source path.

Model: ``microsoft/OmniParser-v2.0`` ``icon_detect`` (AGPL-3.0; a finetuned
YOLOv8). See THIRD_PARTY_NOTICES.md.
"""

import base64
import hashlib
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated

import typer

from physiclaw import paths
from physiclaw.cli._http import http_get, stream
from physiclaw.cli._format import ok

_PT_URL = (
    "https://huggingface.co/microsoft/OmniParser-v2.0/"
    "resolve/main/icon_detect/model.pt"
)

# physiclaw.ai serves the prebuilt zip as base64 parts: Cloudflare Pages caps
# each static asset at 25 MiB, and base64(zip) split four ways stays under it.
# The CLI concatenates the parts and base64-decodes them back to the zip. The
# GitHub release carries the whole zip (no size cap) as a fallback.
_PREBUILT_PARTS = 4
_PREBUILT_PARTS_URL = (
    "https://physiclaw.ai/downloads/local_vision_model.zip.b64.{i:02d}"
)
_PREBUILT_ZIP_URL = (
    "https://github.com/physiclaw/PhysiClaw/releases/download/"
    "local-vision-model/local_vision_model.zip"
)

# sha256 of the extracted model.onnx — pinned so a corrupted, truncated, or
# tampered download is rejected (a missing part can't yield a half model). We
# hash the model, not the archive, so it stays valid across re-zipping /
# re-splitting. Bump when the model is re-released.
_PREBUILT_ONNX_SHA256 = (
    "a0f977a4674d11074341895331ac523ce1372ee0a4ae97001219e50876cd1b7c"
)

# Bumping these is a deliberate decision — the ONNX export contract has
# shifted between ultralytics minor versions before.
_CONVERT_DEPS = (
    "ultralytics>=8.4",
    "onnx>=1.18",
    "onnxslim>=0.1.71",
)

# Filenames inside convert/. Ultralytics derives the .onnx name from the
# .pt name, so the two must stay in lockstep — pinning both as constants
# makes that linkage explicit.
_PT_NAME = "model.pt"
_ONNX_NAME = "model.onnx"
_SCRIPT_NAME = "convert.py"

# Real file (not -c) so it stays in convert/ for debugging on failure.
_CONVERT_SCRIPT = f"""\
from ultralytics import YOLO
YOLO("{_PT_NAME}").export(format="onnx", imgsz=1280)
"""


def _abort_kept_scratch(convert_dir: Path, reason: str) -> None:
    typer.echo(typer.style(
        f"{reason} Scratch dir kept for debugging: {convert_dir}",
        fg=typer.colors.RED, bold=True,
    ))
    raise typer.Abort()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_prebuilt_zip(dest: Path) -> bool:
    """Write the prebuilt zip to ``dest``.

    Tries the base64 parts on physiclaw.ai first (concatenate + decode), then
    the whole-zip GitHub release. Returns False on any network failure so the
    caller can fall back to converting from source.
    """
    try:
        b64 = bytearray()
        for i in range(_PREBUILT_PARTS):
            with http_get(_PREBUILT_PARTS_URL.format(i=i)) as r:
                stream(r, b64.extend, f"  vision model part {i + 1}/{_PREBUILT_PARTS}")
        dest.write_bytes(base64.b64decode(b64))
        return True
    except OSError as e:
        typer.echo(typer.style(
            f"  CDN parts unavailable ({e}) — trying the release.",
            fg=typer.colors.YELLOW,
        ))
    try:
        with http_get(_PREBUILT_ZIP_URL) as r, open(dest, "wb") as f:
            stream(r, f.write, "  vision model")
        return True
    except OSError as e:
        typer.echo(typer.style(f"  release unavailable ({e}).", fg=typer.colors.YELLOW))
        return False


def _try_prebuilt(onnx: Path) -> bool:
    """Install the pre-converted ONNX from the release mirror, verifying its
    sha256.

    Returns True on success, False (quietly) on any network / archive /
    integrity failure so the caller can fall back to converting from source.
    Needs no uv and no conversion deps.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        if not _download_prebuilt_zip(tmp / "model.zip"):
            return False
        try:
            with zipfile.ZipFile(tmp / "model.zip") as z:
                z.extract(_ONNX_NAME, tmp)
        except (zipfile.BadZipFile, KeyError) as e:
            typer.echo(typer.style(
                f"  bad archive ({e}) — skipping prebuilt.",
                fg=typer.colors.YELLOW,
            ))
            return False
        extracted = tmp / _ONNX_NAME
        digest = _sha256(extracted)
        if digest != _PREBUILT_ONNX_SHA256:
            typer.echo(typer.style(
                f"  checksum mismatch ({digest[:12]}…) — skipping prebuilt.",
                fg=typer.colors.YELLOW,
            ))
            return False
        onnx.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), onnx)
        return True


def _report_ready(onnx: Path, label: str) -> None:
    """Shared success trailer for both the prebuilt and from-source paths."""
    typer.echo(f"  {onnx.stat().st_size / 1024 / 1024:.1f} MB at {onnx}")
    typer.echo(ok(label))


def _abort_download(exc: OSError) -> None:
    """Friendly message for a failed weights download — no raw traceback.

    ``http_get`` raises ``urllib.error.HTTPError`` / ``URLError`` (both
    ``OSError`` subclasses) on a 403/blocked-host/offline fetch; surface the
    reason plus the likely fix instead of a stack trace.
    """
    typer.echo(typer.style(
        "Couldn't download the OmniParser vision model.",
        fg=typer.colors.RED, bold=True,
    ))
    typer.echo(f"  Reason: {exc}")
    typer.echo(
        f"\n  The weights are hosted on Hugging Face:\n    {_PT_URL}\n"
        "\n  Likely causes:\n"
        "    - No internet connection\n"
        "    - A proxy, VPN, or firewall is blocking huggingface.co\n"
        "    - Hugging Face is temporarily unavailable\n"
        "\n  Fix the connection, then re-run:\n"
        "    physiclaw setup local-vision-model\n"
    )
    raise typer.Abort()


def vision(
    force: Annotated[
        bool,
        typer.Option(
            "--force", help="Re-install even if the ONNX already exists."
        ),
    ] = False,
    build: Annotated[
        bool,
        typer.Option(
            "--build",
            help="Convert from upstream (PyTorch→ONNX) instead of fetching "
            "the prebuilt model.",
        ),
    ] = False,
) -> None:
    """Install the local vision model (prebuilt by default; --build to convert)."""
    onnx = paths.omniparser_onnx()
    if onnx.exists() and not force:
        typer.echo(f"Already present: {onnx}")
        return

    # Fast path: a pre-converted ONNX from the release mirror — no Hugging
    # Face download, no ultralytics conversion, no uv. Falls through to the
    # from-source build when every mirror is unreachable or fails its checksum.
    if not build:
        if _try_prebuilt(onnx):
            _report_ready(onnx, "vision model ready (prebuilt)")
            return
        typer.echo("Prebuilt model unavailable — converting from source …")

    if shutil.which("uv") is None:
        typer.echo(typer.style(
            "`uv` is required to convert the vision model.",
            fg=typer.colors.YELLOW, bold=True,
        ))
        typer.echo(
            "The conversion runs the heavy deps (ultralytics + onnx + onnxslim, "
            "~500 MB) in an ephemeral uv environment so they never enter the "
            "physiclaw install. Install uv first, then re-run:\n\n"
            "    curl -fsSL https://astral.sh/uv/install.sh | sh    # macOS / Linux\n"
            '    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows\n'
            "    physiclaw setup local-vision-model\n"
        )
        raise typer.Abort()

    convert_dir = onnx.parent / "convert"
    if force and convert_dir.exists():
        # --force means re-fetch from upstream — drop any stale .pt cached
        # from a partial prior run.
        shutil.rmtree(convert_dir)
    convert_dir.mkdir(parents=True, exist_ok=True)
    pt_path = convert_dir / _PT_NAME
    script_path = convert_dir / _SCRIPT_NAME
    onnx_in_scratch = convert_dir / _ONNX_NAME

    if not pt_path.exists():
        try:
            with http_get(_PT_URL) as r, open(pt_path, "wb") as f:
                stream(r, f.write, "  OmniParser weights")
        except OSError as e:
            # A failed fetch can leave a partial file behind; drop it so a retry
            # re-fetches from scratch instead of skipping the download.
            pt_path.unlink(missing_ok=True)
            _abort_download(e)

    script_path.write_text(_CONVERT_SCRIPT)

    typer.echo("Converting to ONNX in ephemeral uv env …")
    cmd = [
        "uv", "run",
        "--python", "3.12",
        "--no-project",
        *(arg for dep in _CONVERT_DEPS for arg in ("--with", dep)),
        "python", _SCRIPT_NAME,
    ]
    result = subprocess.run(cmd, cwd=convert_dir)
    if result.returncode != 0:
        _abort_kept_scratch(
            convert_dir, f"Conversion failed (uv exit {result.returncode}).",
        )
    if not onnx_in_scratch.exists():
        _abort_kept_scratch(
            convert_dir, f"Conversion finished but {onnx_in_scratch} not found.",
        )

    shutil.move(onnx_in_scratch, onnx)
    shutil.rmtree(convert_dir)

    _report_ready(onnx, "vision model ready")
