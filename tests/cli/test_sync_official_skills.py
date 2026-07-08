"""Tests for `physiclaw.cli.sync_official_skills` — the `skills sync official`
client.

The network is mocked at the module's `http_get` seam: a routing fake maps
each URL to canned bytes (or raises a 404 HTTPError for anything unmapped),
so no test hits the wire. Packs are built in-memory as real STORED zips so
extraction, zip-slip, and checksum paths run for real.
"""
from __future__ import annotations

import hashlib
import importlib
import io
import json
import urllib.error
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

osk = importlib.import_module("physiclaw.cli.sync_official_skills")
skills_mod = importlib.import_module("physiclaw.cli.skills")

BASE = "https://site.test/downloads/official-skills"
LATEST_URL = f"{BASE}/latest.json"
ZIP_URL = f"{BASE}/physiclaw_official_skills.zip"
SHA_URL = f"{ZIP_URL}.sha256"

runner = CliRunner()


# ---------- fixtures / helpers ----------


class _FakeResp:
    """Supports both whole-body `read()` (latest.json / sha256) and the
    chunked `read(n)` loop the streaming zip download uses."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *_a: object) -> bool:
        return False

    def getheader(self, _name: str) -> str | None:
        # No Content-Length → `_download.stream` quietly streams (no bar).
        return None

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            chunk = self._data[self._pos:]
        else:
            chunk = self._data[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk


def _make_http_get(routes: dict[str, bytes], calls: list[str] | None = None):
    def _get(url: str, timeout: int = 120):
        if calls is not None:
            calls.append(url)
        if url in routes:
            return _FakeResp(routes[url])
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    return _get


def _skill_md(name: str) -> bytes:
    return f"---\nname: {name}\n---\n# {name}\n".encode()


def _skill_hash(skill_md: bytes) -> str:
    """The site builder's per-skill digest for a single-file `SKILL.md` skill:
    sha256(relpath + NUL + bytes), prefixed `sha256:`."""
    h = hashlib.sha256()
    h.update(b"SKILL.md")
    h.update(b"\x00")
    h.update(skill_md)
    return f"sha256:{h.hexdigest()}"


def _build_pack(
    commit: str,
    skill_names: list[str],
    *,
    declared: int | None = None,
    bad_hash: tuple[str, ...] = (),
):
    """A valid STORED pack: source.json + skills/<name>/SKILL.md each, with the
    manifest carrying each shipped skill's REAL content hash (so the happy path
    verifies clean). `declared` pads the manifest with un-shipped ghost skills
    (no hash → exercises count drift); `bad_hash` names shipped skills whose
    manifest hash is deliberately wrong (exercises per-skill mismatch)."""
    contents = {n: _skill_md(n) for n in skill_names}
    skills = [
        {"name": n, "path": f"skills/{n}", "description": "d",
         "hash": ("sha256:" + "f" * 64) if n in bad_hash else _skill_hash(contents[n])}
        for n in skill_names
    ]
    if declared is not None:  # pad with ghosts the pack doesn't actually ship
        skills += [
            {"name": f"ghost{i}", "path": f"skills/ghost{i}", "description": "d"}
            for i in range(declared - len(skill_names))
        ]
    src = {
        "schemaVersion": 1,
        "package": "physiclaw-official-skills",
        "commit": commit,
        "repo": "physiclaw/PhysiClaw",
        "subdir": "skills",
        "builtAt": "2026-07-08T09:46:19Z",
        "skills": skills,
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("source.json", json.dumps(src))
        for n in skill_names:
            zf.writestr(f"skills/{n}/SKILL.md", contents[n])
    data = buf.getvalue()
    return data, hashlib.sha256(data).hexdigest()


def _routes(
    commit: str,
    skill_names: list[str],
    *,
    declared: int | None = None,
    bad_hash: tuple[str, ...] = (),
):
    data, hexd = _build_pack(commit, skill_names, declared=declared, bad_hash=bad_hash)
    return {
        LATEST_URL: json.dumps(
            {"schemaVersion": 1, "commit": commit, "builtAt": "x",
             "skillCount": len(skill_names)}
        ).encode(),
        ZIP_URL: data,
        SHA_URL: f"{hexd}  physiclaw_official_skills.zip\n".encode(),
    }


@pytest.fixture
def official_home(tmp_path: Path, mocker) -> Path:
    d = tmp_path / "official"
    mocker.patch.object(osk.paths, "official_dir", return_value=d)
    mocker.patch.object(
        osk, "_load_config",
        return_value=SimpleNamespace(skills=SimpleNamespace(official_base_url=BASE)),
    )
    return d


def _patch_net(mocker, routes: dict[str, bytes], calls: list[str] | None = None):
    mocker.patch.object(osk, "http_get", _make_http_get(routes, calls))


# ---------- freshness / not-published ----------


def test_not_published_is_not_an_error(official_home: Path, mocker, capsys) -> None:
    _patch_net(mocker, {})  # latest.json 404s

    osk.sync()  # no raise

    assert "no official skills published yet" in capsys.readouterr().out
    assert not (official_home / "skills").exists()
    assert not (official_home / osk.SYNC_STATE_FILE).exists()


def test_latest_missing_commit_errors(official_home: Path, mocker, capsys) -> None:
    _patch_net(mocker, {LATEST_URL: json.dumps({"schemaVersion": 1}).encode()})

    with pytest.raises(typer.Exit) as e:
        osk.sync()

    assert e.value.exit_code == 1
    assert "missing 'commit'" in capsys.readouterr().err


# ---------- fresh install ----------


def test_fresh_install_mounts_and_records_state(
    official_home: Path, mocker, capsys
) -> None:
    _, hexd = _build_pack("a" * 40, ["jd", "wechat"])
    _patch_net(mocker, _routes("a" * 40, ["jd", "wechat"]))

    osk.sync()

    assert (official_home / "skills" / "jd" / "SKILL.md").is_file()
    assert (official_home / "skills" / "wechat" / "SKILL.md").is_file()
    assert (official_home / "source.json").is_file()

    state = json.loads((official_home / osk.SYNC_STATE_FILE).read_text())
    assert state["commit"] == "a" * 40
    assert state["skillCount"] == 2
    assert state["zipHash"] == f"sha256:{hexd}"
    assert state["url"] == ZIP_URL
    assert state["syncedAt"].endswith("Z")
    # No staging dir left behind.
    assert not (official_home / osk.STAGING_DIR).exists()
    assert "synced 2 official skill(s)" in capsys.readouterr().out


# ---------- idempotence / force ----------


def test_up_to_date_skips_download(official_home: Path, mocker, capsys) -> None:
    official_home.mkdir(parents=True)
    (official_home / osk.SYNC_STATE_FILE).write_text(
        json.dumps({"commit": "a" * 40})
    )
    calls: list[str] = []
    _patch_net(mocker, _routes("a" * 40, ["jd"]), calls)

    osk.sync()

    assert "already up to date" in capsys.readouterr().out
    assert ZIP_URL not in calls  # freshness shortcut — never fetched the pack


def test_force_re_syncs_when_up_to_date(official_home: Path, mocker, capsys) -> None:
    official_home.mkdir(parents=True)
    (official_home / osk.SYNC_STATE_FILE).write_text(json.dumps({"commit": "a" * 40}))
    calls: list[str] = []
    _patch_net(mocker, _routes("a" * 40, ["jd"]), calls)

    osk.sync(force=True)

    assert ZIP_URL in calls
    assert (official_home / "skills" / "jd").is_dir()


def test_changed_commit_replaces_and_drops_removed_skills(
    official_home: Path, mocker
) -> None:
    # First sync: two skills at commit A.
    _patch_net(mocker, _routes("a" * 40, ["jd", "wechat"]))
    osk.sync()
    assert (official_home / "skills" / "wechat").is_dir()

    # Second sync: commit B ships only jd — wechat must disappear.
    _patch_net(mocker, _routes("b" * 40, ["jd"]))
    osk.sync()

    assert (official_home / "skills" / "jd").is_dir()
    assert not (official_home / "skills" / "wechat").exists()
    state = json.loads((official_home / osk.SYNC_STATE_FILE).read_text())
    assert state["commit"] == "b" * 40
    assert state["skillCount"] == 1


# ---------- integrity ----------


def test_checksum_mismatch_aborts_untouched(
    official_home: Path, mocker, capsys
) -> None:
    routes = _routes("a" * 40, ["jd"])
    routes[SHA_URL] = f"{'f' * 64}  physiclaw_official_skills.zip\n".encode()  # wrong
    _patch_net(mocker, routes)

    with pytest.raises(typer.Exit) as e:
        osk.sync()

    assert e.value.exit_code == 1
    assert "checksum mismatch" in capsys.readouterr().err
    assert not (official_home / "skills").exists()
    assert not (official_home / osk.SYNC_STATE_FILE).exists()


def test_malformed_sha256_line_aborts(official_home: Path, mocker, capsys) -> None:
    routes = _routes("a" * 40, ["jd"])
    routes[SHA_URL] = b"not-a-checksum\n"
    _patch_net(mocker, routes)

    with pytest.raises(typer.Exit):
        osk.sync()

    assert "not a valid sha256sum" in capsys.readouterr().err


# ---------- zip-slip / layout ----------


def test_zip_slip_entry_is_rejected(official_home: Path, mocker, capsys) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("source.json", json.dumps({"commit": "a" * 40, "skills": []}))
        zf.writestr("../escape.txt", "pwned")
    data = buf.getvalue()
    routes = {
        LATEST_URL: json.dumps({"commit": "a" * 40}).encode(),
        ZIP_URL: data,
        SHA_URL: f"{hashlib.sha256(data).hexdigest()}  x\n".encode(),
    }
    _patch_net(mocker, routes)

    with pytest.raises(typer.Exit):
        osk.sync()

    assert "zip-slip" in capsys.readouterr().err.lower()
    assert not (official_home / "skills").exists()


def test_unexpected_layout_rejected(official_home: Path, mocker, capsys) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("source.json", json.dumps({"commit": "a" * 40, "skills": []}))
        # missing skills/ ; stray file instead
        zf.writestr("README.txt", "hi")
    data = buf.getvalue()
    routes = {
        LATEST_URL: json.dumps({"commit": "a" * 40}).encode(),
        ZIP_URL: data,
        SHA_URL: f"{hashlib.sha256(data).hexdigest()}  x\n".encode(),
    }
    _patch_net(mocker, routes)

    with pytest.raises(typer.Exit):
        osk.sync()

    assert "unexpected package layout" in capsys.readouterr().err


# ---------- per-skill hash verification ----------


def test_skill_hash_matches_no_warning(official_home: Path, mocker, capsys) -> None:
    # Happy path: manifest carries each skill's real content hash.
    _patch_net(mocker, _routes("a" * 40, ["jd", "wechat"]))

    osk.sync()

    out = capsys.readouterr().out
    assert "skill hash mismatch" not in out
    assert (official_home / "skills" / "jd").is_dir()


def test_skill_hash_mismatch_warns_but_mounts(
    official_home: Path, mocker, capsys
) -> None:
    # jd's manifest hash is wrong; wechat's is right.
    _patch_net(mocker, _routes("a" * 40, ["jd", "wechat"], bad_hash=("jd",)))

    osk.sync()  # warns, does NOT abort (per-skill hash is not integrity)

    out = capsys.readouterr().out
    mismatch_lines = [ln for ln in out.splitlines() if "skill hash mismatch" in ln]
    assert len(mismatch_lines) == 1  # only jd, not wechat
    assert "jd:" in mismatch_lines[0]
    # The skill still mounts — the zip .sha256 already vouched for the bytes.
    assert (official_home / "skills" / "jd" / "SKILL.md").is_file()
    state = json.loads((official_home / osk.SYNC_STATE_FILE).read_text())
    assert state["skillCount"] == 2


def test_hash_recipe_matches_builder(tmp_path: Path) -> None:
    # Lock the recipe to the site builder's hashSkill (relpath + NUL + bytes).
    d = tmp_path / "jd"
    d.mkdir()
    (d / "SKILL.md").write_bytes(_skill_md("jd"))
    assert osk._hash_skill(d) == _skill_hash(_skill_md("jd"))


# ---------- reconcile drift ----------


def test_manifest_drift_warns(official_home: Path, mocker, capsys) -> None:
    # Manifest declares 3 skills but only 1 dir is shipped.
    _patch_net(mocker, _routes("a" * 40, ["jd"], declared=3))

    osk.sync()

    out = capsys.readouterr().out
    assert "synced 1 official skill(s)" in out
    assert "manifest lists 3 skill(s) but 1 landed" in out


# ---------- dry-run ----------


def test_dry_run_reports_install_without_downloading(
    official_home: Path, mocker, capsys
) -> None:
    calls: list[str] = []
    _patch_net(mocker, _routes("a" * 40, ["jd"]), calls)

    osk.sync(dry_run=True)

    out = capsys.readouterr().out
    assert "sync would run" in out and "install" in out
    assert calls == [LATEST_URL]  # only the freshness check
    assert not (official_home / "skills").exists()


def test_dry_run_reports_up_to_date(official_home: Path, mocker, capsys) -> None:
    official_home.mkdir(parents=True)
    (official_home / osk.SYNC_STATE_FILE).write_text(json.dumps({"commit": "a" * 40}))
    _patch_net(mocker, _routes("a" * 40, ["jd"]))

    osk.sync(dry_run=True)

    assert "up to date" in capsys.readouterr().out


# ---------- CLI wiring ----------


def test_cli_sync_official_routes_flags(mocker) -> None:
    called = mocker.patch.object(osk, "sync")

    result = runner.invoke(skills_mod.skills_app, ["sync", "official", "--dry-run"])

    assert result.exit_code == 0
    called.assert_called_once_with(force=False, dry_run=True)


def test_cli_sync_no_target_shows_help() -> None:
    result = runner.invoke(skills_mod.skills_app, ["sync"])
    # no_args_is_help → non-zero exit with usage listing the `official` command.
    assert "official" in result.output


# ---------- maybe_auto_sync (server-startup hook, backgrounded) ----------


@pytest.fixture
def auto_env(mocker, monkeypatch):
    """Baseline for maybe_auto_sync: sync_auto on, not CI, no live server.
    Patches threading.Thread so the guards can be tested without actually
    spawning the background sync. Yields the Thread mock."""
    monkeypatch.delenv("CI", raising=False)
    mocker.patch.object(
        osk, "_load_config",
        return_value=SimpleNamespace(skills=SimpleNamespace(sync_auto=True)),
    )
    mocker.patch("physiclaw.runtime_state.read_live", return_value=None)
    return mocker.patch.object(osk.threading, "Thread")


def test_auto_sync_spawns_background_daemon_thread(auto_env) -> None:
    osk.maybe_auto_sync()
    # The sync runs off the startup critical path, in a daemon thread.
    auto_env.assert_called_once_with(
        target=osk._run_sync_quiet, name="physiclaw-auto-sync", daemon=True
    )
    auto_env.return_value.start.assert_called_once_with()


def test_auto_sync_skipped_when_disabled(mocker, monkeypatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    mocker.patch.object(
        osk, "_load_config",
        return_value=SimpleNamespace(skills=SimpleNamespace(sync_auto=False)),
    )
    thread = mocker.patch.object(osk.threading, "Thread")

    osk.maybe_auto_sync()

    thread.assert_not_called()


def test_auto_sync_skipped_under_ci(auto_env, monkeypatch) -> None:
    monkeypatch.setenv("CI", "true")

    osk.maybe_auto_sync()

    auto_env.assert_not_called()  # no thread spawned


def test_auto_sync_skipped_when_server_live(auto_env, mocker) -> None:
    mocker.patch("physiclaw.runtime_state.read_live", return_value={"pid": 123})

    osk.maybe_auto_sync()

    auto_env.assert_not_called()  # don't race another server's swap


# ---------- _run_sync_quiet (the daemon-thread body, fail-soft) ----------


def test_run_sync_quiet_calls_sync(mocker) -> None:
    called = mocker.patch.object(osk, "sync")
    osk._run_sync_quiet()
    called.assert_called_once_with()


def test_run_sync_quiet_swallows_typer_exit(mocker) -> None:
    mocker.patch.object(osk, "sync", side_effect=typer.Exit(code=1))
    osk._run_sync_quiet()  # a hard sync abort must never escape the thread


def test_run_sync_quiet_swallows_unexpected_error(mocker) -> None:
    mocker.patch.object(osk, "sync", side_effect=RuntimeError("boom"))
    osk._run_sync_quiet()  # any error is non-fatal
