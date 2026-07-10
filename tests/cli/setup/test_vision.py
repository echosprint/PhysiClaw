"""Tests for `physiclaw.cli.setup.vision` — local vision model download."""
from __future__ import annotations

import base64
import importlib
import io
import subprocess
from pathlib import Path

import typer
from typer.testing import CliRunner

vision_mod = importlib.import_module("physiclaw.cli.setup.vision")
http_mod = importlib.import_module("physiclaw.cli._http")


class _BytesCtx:
    """Stand-in for an ``urlopen`` response: a context manager backed by a
    BytesIO, supporting ``.read(n)`` and ``.getheader("Content-Length")`` so it
    works with the chunked ``stream`` reader."""

    def __init__(self, data: bytes, *, content_length: str | None = None) -> None:
        self._buf = io.BytesIO(data)
        self._len = content_length

    def __enter__(self) -> _BytesCtx:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def read(self, *args: object) -> bytes:
        return self._buf.read(*args)

    def getheader(self, name: str, default: object = None) -> object:
        if name.lower() == "content-length":
            return self._len if self._len is not None else default
        return default

runner = CliRunner()
app = typer.Typer()
app.command()(vision_mod.vision)


def _stub_download(mocker, payload: bytes = b"PT") -> None:
    """Replace the HF fetch (http_get -> urlopen) with one returning payload."""
    mocker.patch.object(
        http_mod.urllib.request, "urlopen",
        side_effect=lambda *_a, **_k: _BytesCtx(payload),
    )


def _stub_uv_run(mocker, *, onnx_bytes: bytes = b"ONNX") -> object:
    """Make `uv run` write convert/<onnx> and return 0. Returns the spy."""

    def _fake_run(cmd, cwd, **_kw):
        (Path(cwd) / vision_mod._ONNX_NAME).write_bytes(onnx_bytes)
        return subprocess.CompletedProcess(cmd, 0)

    return mocker.patch.object(vision_mod.subprocess, "run", side_effect=_fake_run)


def _zip_bytes(onnx_bytes: bytes) -> bytes:
    """A zip archive (as bytes) holding model.onnx=onnx_bytes."""
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(vision_mod._ONNX_NAME, onnx_bytes)
    return buf.getvalue()


def _stub_prebuilt(mocker, onnx_bytes: bytes, *, pin_hash: bool = True) -> None:
    """Make `_download_prebuilt_zip` write a zip holding model.onnx=onnx_bytes;
    optionally pin the expected sha256 so the install succeeds."""
    import hashlib

    def _fake_download(dest):
        Path(dest).write_bytes(_zip_bytes(onnx_bytes))
        return True

    mocker.patch.object(
        vision_mod, "_download_prebuilt_zip", side_effect=_fake_download
    )
    if pin_hash:
        mocker.patch.object(
            vision_mod, "_PREBUILT_ONNX_SHA256",
            hashlib.sha256(onnx_bytes).hexdigest(),
        )


def test_vision_prebuilt_installs_without_uv(tmp_path: Path, mocker) -> None:
    # The default path fetches the prebuilt ONNX — no uv, no conversion.
    onnx = tmp_path / "models" / "omniparser_icon_detect" / "model.onnx"
    mocker.patch.object(vision_mod.paths, "omniparser_onnx", return_value=onnx)
    _stub_prebuilt(mocker, onnx_bytes=b"PREBUILT-ONNX")
    uv_run = mocker.patch.object(vision_mod.subprocess, "run")

    result = runner.invoke(app, [])

    assert result.exit_code == 0, result.output
    assert onnx.read_bytes() == b"PREBUILT-ONNX"
    assert "prebuilt" in result.output
    uv_run.assert_not_called()


def test_vision_prebuilt_checksum_mismatch_falls_back_to_build(
    tmp_path: Path, mocker,
) -> None:
    onnx = tmp_path / "models" / "omniparser_icon_detect" / "model.onnx"
    mocker.patch.object(vision_mod.paths, "omniparser_onnx", return_value=onnx)
    mocker.patch.object(vision_mod.shutil, "which", return_value="/usr/bin/uv")

    # Prebuilt zip downloads fine but its onnx hash won't match the pinned one.
    _stub_prebuilt(mocker, onnx_bytes=b"TAMPERED", pin_hash=False)
    _stub_download(mocker)  # mock the HF fetch the convert fallback uses
    _stub_uv_run(mocker)

    result = runner.invoke(app, [])

    assert result.exit_code == 0, result.output
    assert "checksum mismatch" in result.output
    assert "converting from source" in result.output
    assert onnx.read_bytes() == b"ONNX"  # came from the convert fallback


def test_download_prebuilt_zip_reassembles_base64_parts(
    tmp_path: Path, mocker,
) -> None:
    payload = _zip_bytes(b"PARTED-ONNX")
    b64 = base64.b64encode(payload)
    n = len(b64)
    cuts = [b64[0:n // 4], b64[n // 4:n // 2], b64[n // 2:3 * n // 4], b64[3 * n // 4:]]
    seen: list[str] = []

    def _fake_urlopen(req, **_kw):
        seen.append(req.full_url)
        return _BytesCtx(cuts[len(seen) - 1])

    mocker.patch.object(
        http_mod.urllib.request, "urlopen", side_effect=_fake_urlopen
    )

    dest = tmp_path / "out.zip"
    assert vision_mod._download_prebuilt_zip(dest) is True
    assert dest.read_bytes() == payload
    assert len(seen) == vision_mod._PREBUILT_PARTS


def test_download_prebuilt_zip_retries_a_truncated_part(
    tmp_path: Path, mocker,
) -> None:
    # A CDN dropping the connection mid-part makes read() end early with no
    # error (the install.sh field failure) — stream()'s Content-Length check
    # turns that into a retry, and the retry's clean bytes are used; the
    # failed attempt's partial bytes must never reach the concatenation.
    payload = _zip_bytes(b"PARTED-ONNX")
    b64 = base64.b64encode(payload)
    n = len(b64)
    cuts = [b64[0:n // 4], b64[n // 4:n // 2], b64[n // 2:3 * n // 4], b64[3 * n // 4:]]
    truncated_once = {"done": False}

    def _fake_urlopen(req, **_kw):
        i = int(req.full_url[-2:])
        if i == 1 and not truncated_once["done"]:
            truncated_once["done"] = True
            # Body stops short of the declared length — no exception.
            return _BytesCtx(cuts[1][: len(cuts[1]) // 2],
                             content_length=str(len(cuts[1])))
        return _BytesCtx(cuts[i])

    mocker.patch.object(
        http_mod.urllib.request, "urlopen", side_effect=_fake_urlopen
    )

    dest = tmp_path / "out.zip"
    assert vision_mod._download_prebuilt_zip(dest) is True
    assert dest.read_bytes() == payload


def test_download_prebuilt_zip_corrupt_b64_falls_back_to_release(
    tmp_path: Path, mocker,
) -> None:
    # Garbage that survives the length check (no Content-Length header, so
    # stream can't verify) must not escape as a b64decode traceback — it
    # falls through to the release like any other CDN failure. The parts
    # concatenate to 17 base64 chars (≡1 mod 4), an impossible length.
    def _fake_urlopen(req, **_kw):
        if "b64" in req.full_url:
            return _BytesCtx(b"abcde" if req.full_url.endswith("00") else b"abcd")
        return _BytesCtx(b"WHOLE-ZIP")

    mocker.patch.object(
        http_mod.urllib.request, "urlopen", side_effect=_fake_urlopen
    )

    dest = tmp_path / "out.zip"
    assert vision_mod._download_prebuilt_zip(dest) is True
    assert dest.read_bytes() == b"WHOLE-ZIP"


def test_download_prebuilt_zip_falls_back_to_release(
    tmp_path: Path, mocker,
) -> None:
    # CDN parts fail; the whole-zip release (also fetched via http_get) wins.
    def _fake_urlopen(req, **_kw):
        if "b64" in req.full_url:
            raise OSError("CDN down")
        return _BytesCtx(b"WHOLE-ZIP")

    mocker.patch.object(
        http_mod.urllib.request, "urlopen", side_effect=_fake_urlopen
    )

    dest = tmp_path / "out.zip"
    assert vision_mod._download_prebuilt_zip(dest) is True
    assert dest.read_bytes() == b"WHOLE-ZIP"


def test_download_prebuilt_zip_sets_user_agent(tmp_path: Path, mocker) -> None:
    # Cloudflare 403s the default Python-urllib UA — every request must set one.
    seen_ua: list[str] = []

    def _fake_urlopen(req, **_kw):
        seen_ua.append(req.get_header("User-agent"))
        return _BytesCtx(b"")  # body irrelevant — we only assert the header

    mocker.patch.object(
        http_mod.urllib.request, "urlopen", side_effect=_fake_urlopen
    )
    vision_mod._download_prebuilt_zip(tmp_path / "out.zip")
    assert seen_ua and all(ua == http_mod.USER_AGENT for ua in seen_ua)


def test_vision_already_present_no_op(tmp_path: Path, mocker) -> None:
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"x")
    mocker.patch.object(vision_mod.paths, "omniparser_onnx", return_value=onnx)

    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert "Already present" in result.output


def test_vision_build_aborts_when_uv_missing(tmp_path: Path, mocker) -> None:
    # --build forces the from-source path, which needs uv.
    onnx = tmp_path / "missing.onnx"
    mocker.patch.object(vision_mod.paths, "omniparser_onnx", return_value=onnx)
    mocker.patch.object(vision_mod.shutil, "which", return_value=None)

    result = runner.invoke(app, ["--build"])

    assert result.exit_code == 1
    assert "uv" in result.output
    assert "physiclaw setup local-vision-model" in result.output


def test_vision_downloads_converts_and_cleans_up(
    tmp_path: Path, mocker,
) -> None:
    onnx = tmp_path / "models" / "omniparser_icon_detect" / "model.onnx"
    mocker.patch.object(vision_mod.paths, "omniparser_onnx", return_value=onnx)
    mocker.patch.object(vision_mod.shutil, "which", return_value="/usr/bin/uv")

    _stub_download(mocker, payload=b"PT-WEIGHTS")
    spy = _stub_uv_run(mocker)

    result = runner.invoke(app, ["--build"])

    assert result.exit_code == 0, result.output
    assert onnx.exists()
    assert onnx.read_bytes() == b"ONNX"
    assert "vision model ready" in result.output
    assert not (onnx.parent / "convert").exists()

    cmd = spy.call_args.args[0]
    assert cmd[0] == "uv"
    assert "run" in cmd
    assert "--no-project" in cmd
    assert "--python" in cmd and "3.12" in cmd
    assert vision_mod._SCRIPT_NAME in cmd
    with_args = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--with"]
    assert any(a.startswith("ultralytics") for a in with_args)
    assert any(a.startswith("onnx>=") for a in with_args)
    assert any(a.startswith("onnxslim") for a in with_args)


def test_vision_force_redownloads(tmp_path: Path, mocker) -> None:
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"old")
    mocker.patch.object(vision_mod.paths, "omniparser_onnx", return_value=onnx)
    mocker.patch.object(vision_mod.shutil, "which", return_value="/usr/bin/uv")

    _stub_download(mocker)
    _stub_uv_run(mocker, onnx_bytes=b"newly converted")

    result = runner.invoke(app, ["--force", "--build"])

    assert result.exit_code == 0, result.output
    assert onnx.read_bytes() == b"newly converted"


def test_vision_force_purges_stale_scratch(tmp_path: Path, mocker) -> None:
    """--force must drop any stale convert/model.pt and re-download."""
    onnx = tmp_path / "models" / "omniparser_icon_detect" / "model.onnx"
    onnx.parent.mkdir(parents=True)
    onnx.write_bytes(b"old onnx")
    mocker.patch.object(vision_mod.paths, "omniparser_onnx", return_value=onnx)
    mocker.patch.object(vision_mod.shutil, "which", return_value="/usr/bin/uv")

    convert_dir = onnx.parent / "convert"
    convert_dir.mkdir()
    (convert_dir / vision_mod._PT_NAME).write_bytes(b"stale PT")

    download_spy = mocker.patch.object(
        http_mod.urllib.request, "urlopen",
        side_effect=lambda *_a, **_k: _BytesCtx(b"FRESH PT"),
    )
    _stub_uv_run(mocker)

    result = runner.invoke(app, ["--force", "--build"])

    assert result.exit_code == 0, result.output
    download_spy.assert_called_once()


def test_vision_skips_download_when_pt_already_exists(
    tmp_path: Path, mocker,
) -> None:
    onnx = tmp_path / "models" / "omniparser_icon_detect" / "model.onnx"
    mocker.patch.object(vision_mod.paths, "omniparser_onnx", return_value=onnx)
    mocker.patch.object(vision_mod.shutil, "which", return_value="/usr/bin/uv")

    convert_dir = onnx.parent / "convert"
    convert_dir.mkdir(parents=True)
    (convert_dir / vision_mod._PT_NAME).write_bytes(b"existing PT")

    download_spy = mocker.patch.object(http_mod.urllib.request, "urlopen")
    _stub_uv_run(mocker)

    result = runner.invoke(app, ["--build"])

    assert result.exit_code == 0, result.output
    download_spy.assert_not_called()


def test_vision_keeps_scratch_on_uv_failure(tmp_path: Path, mocker) -> None:
    onnx = tmp_path / "models" / "omniparser_icon_detect" / "model.onnx"
    mocker.patch.object(vision_mod.paths, "omniparser_onnx", return_value=onnx)
    mocker.patch.object(vision_mod.shutil, "which", return_value="/usr/bin/uv")
    _stub_download(mocker)

    mocker.patch.object(
        vision_mod.subprocess, "run",
        return_value=subprocess.CompletedProcess([], returncode=2),
    )

    result = runner.invoke(app, ["--build"])

    assert result.exit_code == 1
    assert "Conversion failed" in result.output
    assert (onnx.parent / "convert").exists()
    assert not onnx.exists()


def test_vision_keeps_scratch_when_onnx_not_produced(
    tmp_path: Path, mocker,
) -> None:
    onnx = tmp_path / "models" / "omniparser_icon_detect" / "model.onnx"
    mocker.patch.object(vision_mod.paths, "omniparser_onnx", return_value=onnx)
    mocker.patch.object(vision_mod.shutil, "which", return_value="/usr/bin/uv")
    _stub_download(mocker)

    mocker.patch.object(
        vision_mod.subprocess, "run",
        return_value=subprocess.CompletedProcess([], returncode=0),
    )

    result = runner.invoke(app, ["--build"])

    assert result.exit_code == 1
    assert "not found" in result.output
    assert (onnx.parent / "convert").exists()
