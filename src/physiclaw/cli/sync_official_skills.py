"""``physiclaw skills sync official`` — pull the curated official skill pack
from the site and mount it at ``~/.physiclaw/official/``.

Two identities, never conflated:

  * ``commit`` — the version + freshness key. We compare ``latest.json``'s
    commit to the stored one; equal → up to date, no download. It's stable
    across site rebuilds (a rebuild of the same source doesn't churn it), so
    an unchanged pack never re-downloads.
  * ``zipHash`` — integrity only, computed *here* over the downloaded bytes
    and checked against the published ``.sha256`` before anything on disk is
    touched. The per-skill ``hash`` in ``source.json`` is change-detection,
    not integrity (a hash shipped inside the zip can't prove the zip wasn't
    tampered — the ``.sha256`` compare is what does).

The swap is package-owned and same-filesystem: staging lives under
``official/`` so the final ``os.replace`` is atomic, and only ``official/skills``
+ ``official/source.json`` are replaced. The zip is the authoritative full
set, so a skill dropped upstream disappears here automatically.

Integrity here guards corruption, not a coordinated MITM — the checksum
travels the same channel as the zip. Signature verification (the site
publishing a signature) would be the next step for tamper-proofing.

Skill discovery snapshots per agent session (in the separate runtime process),
so the mount takes effect at the next session/wake — a running server picks up
a change without a restart, and the atomic swap means a session always reads
either the whole old or whole new tree.
"""

import datetime as dt
import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import urllib.error
import zipfile
from pathlib import Path
from typing import Callable

import click
import typer

from physiclaw.common import paths
from physiclaw.cli._http import http_get, stream
from physiclaw.cli._format import exit_error, info, ok, warn
from physiclaw.common.config import load as _load_config
from physiclaw.common.text import read_text, write_text

log = logging.getLogger(__name__)

SYNC_STATE_FILE = ".sync-state.json"
STAGING_DIR = ".sync-staging"
ZIP_NAME = "physiclaw_official_skills.zip"
SCHEMA_VERSION = 1


class _NotPublished(Exception):
    """``latest.json`` (or the pack) 404s — nothing published yet, not an
    error the caller should surface as a failure."""


def _require_https(url: str) -> None:
    if not url.startswith("https://"):
        exit_error(f"refusing non-HTTPS official-skills URL: {url}")


def _urls(base: str) -> dict[str, str]:
    base = base.rstrip("/")
    return {
        "latest": f"{base}/latest.json",
        "zip": f"{base}/{ZIP_NAME}",
        "sha256": f"{base}/{ZIP_NAME}.sha256",
    }


def _fetch_bytes(url: str) -> bytes:
    """GET ``url`` over HTTPS, returning the raw body. 404 → ``_NotPublished``;
    any other transport/HTTP error is a clean ``Exit(1)`` (no traceback)."""
    _require_https(url)
    try:
        with http_get(url) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise _NotPublished from e
        exit_error(f"GET {url} failed: HTTP {e.code}")
    except urllib.error.URLError as e:
        exit_error(f"GET {url} failed: {e.reason}")


def _fetch_json(url: str) -> dict:
    try:
        data = json.loads(_fetch_bytes(url).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        exit_error(f"{url} did not return valid JSON: {e}")
    if not isinstance(data, dict):
        exit_error(f"{url} did not return a JSON object.")
    return data


def _read_sync_state() -> dict | None:
    """Parse ``official/.sync-state.json``; None on missing/corrupt (a corrupt
    state just forces a re-sync, never a crash)."""
    p = paths.official_dir() / SYNC_STATE_FILE
    if not p.exists():
        return None
    try:
        data = json.loads(read_text(p))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _parse_sha256(text: str) -> str | None:
    """``<hex>  <filename>\\n`` → the lowercased 64-char hex, or None if the
    line isn't a valid sha256sum record."""
    parts = text.split()
    if not parts:
        return None
    hexd = parts[0].strip().lower()
    if len(hexd) == 64 and all(c in "0123456789abcdef" for c in hexd):
        return hexd
    return None


def _download_to_temp(url: str) -> tuple[Path, str]:
    """Stream ``url`` (HTTPS only) to a temp file, hashing as we go so we never
    hold the whole pack in memory. Returns ``(temp_path, sha256_hex)``; the
    caller is responsible for unlinking the path. The temp file lives in the
    system temp dir (``/tmp`` on Linux) — only the later staging→install
    ``os.replace`` needs to be same-filesystem, not this download."""
    _require_https(url)
    digest = hashlib.sha256()
    fd, name = tempfile.mkstemp(prefix="physiclaw-official-", suffix=".zip")
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as f, http_get(url) as resp:
            # Reuse the shared chunked reader (64 KiB chunks), hashing each
            # chunk on the way to disk — constant memory. progress=False: the
            # skills pack is small, so no progress bar — just the synced result.
            def _write(chunk: bytes) -> None:
                f.write(chunk)
                digest.update(chunk)

            stream(resp, _write, "  official skills", progress=False)
    except urllib.error.URLError as e:
        tmp.unlink(missing_ok=True)
        reason = f"HTTP {e.code}" if isinstance(e, urllib.error.HTTPError) else e.reason
        exit_error(f"GET {url} failed: {reason}")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return tmp, digest.hexdigest()


def _extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract the zip file into ``dest`` with zip-slip protection: every
    member's resolved path must stay within ``dest`` or the whole extract is
    refused (a malicious ``../`` entry can't escape staging)."""
    dest_real = dest.resolve()
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                target = (dest / member).resolve()
                if target != dest_real and not target.is_relative_to(dest_real):
                    exit_error(
                        f"zip entry escapes staging dir: {member!r} — "
                        "aborting (possible zip-slip)."
                    )
            zf.extractall(dest)
    except zipfile.BadZipFile as e:
        exit_error(f"downloaded pack is not a valid zip: {e}")


def _sanity_check_layout(staging: Path) -> None:
    """The extracted top level must be exactly ``source.json`` + ``skills/``.
    Anything else (stray files, ``__MACOSX``, a missing half) means a
    malformed pack — refuse rather than mount it."""
    entries = {p.name for p in staging.iterdir()}
    if entries != {"source.json", "skills"} or not (staging / "skills").is_dir():
        exit_error(
            f"unexpected package layout {sorted(entries)} — expected "
            "exactly source.json + skills/."
        )


def _count_skills(skills_dir: Path) -> int:
    """Landed skill count — direct ``<name>/`` subdirs, dot-dirs excluded."""
    if not skills_dir.is_dir():
        return 0
    return sum(
        1 for d in skills_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )


def _hash_skill(skill_dir: Path) -> str:
    """Content digest of one skill dir. Mirrors the site builder's ``hashSkill``
    (PhysiClaw-site ``scripts/build-official-skills.mjs``) byte-for-byte so the
    digests line up: sha256 over each file's ``/``-joined path RELATIVE to the
    skill dir, then a NUL, then the file bytes — files sorted by that relpath.
    Prefixed ``sha256:`` so the algorithm can migrate later."""
    files = sorted(
        p.relative_to(skill_dir).as_posix() for p in skill_dir.rglob("*") if p.is_file()
    )
    h = hashlib.sha256()
    for rel in files:
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update((skill_dir / rel).read_bytes())
    return f"sha256:{h.hexdigest()}"


def _verify_skill_hashes(skills_root: Path, manifest_skills: list) -> list[str]:
    """Recompute each manifest skill's digest from the extracted dir and compare
    to its declared ``hash``. Returns human-readable mismatch notes ([] = all
    consistent). Per-skill hash is change-detection, NOT integrity (the zip
    ``.sha256`` is integrity, checked earlier) — so the caller WARNS on a
    mismatch rather than aborting: the bytes already passed the zip checksum, so
    a divergence means an internally-inconsistent manifest, not a bad download."""
    notes: list[str] = []
    for entry in manifest_skills:
        if not isinstance(entry, dict):
            continue
        name, declared = entry.get("name"), entry.get("hash")
        if not name or not declared:
            continue
        d = skills_root / name
        if not d.is_dir():
            notes.append(f"{name}: in manifest but not extracted")
            continue
        actual = _hash_skill(d)
        if actual != declared:
            notes.append(f"{name}: {actual[:19]}… ≠ manifest {str(declared)[:19]}…")
    return notes


def _write_sync_state(url: str, zip_hex: str, commit: str, count: int) -> None:
    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "url": url,
        "zipHash": f"sha256:{zip_hex}",
        "commit": commit,
        "skillCount": count,
        "syncedAt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    write_text(
        paths.official_dir() / SYNC_STATE_FILE, json.dumps(payload, indent=2) + "\n"
    )


def sync(
    *,
    force: bool = False,
    dry_run: bool = False,
    emit: Callable[[str], None] = typer.echo,
) -> None:
    """Sync the official skill pack. Idempotent: a matching commit is a no-op
    unless ``--force``. ``--dry-run`` does the freshness check only and downloads
    nothing.

    ``emit`` is the outcome printer: the interactive CLI keeps the default
    (styled ``✓``/``i`` lines on stdout); the server's background auto-sync
    passes a logger-backed emitter so its lines land in the timestamped
    ``[physiclaw]`` stream instead of interleaving raw mid-log."""
    base = _load_config().skills.official_base_url
    urls = _urls(base)

    # 1. Freshness — one small GET; 404 means nothing's published.
    try:
        latest = _fetch_json(urls["latest"])
    except _NotPublished:
        emit(info("no official skills published yet — nothing to sync."))
        return

    remote_commit = str(latest.get("commit") or "")
    if not remote_commit:
        exit_error(f"{urls['latest']} is missing 'commit'.")

    local_commit = str((_read_sync_state() or {}).get("commit") or "")
    up_to_date = bool(local_commit) and local_commit == remote_commit

    if dry_run:
        if up_to_date:
            tail = "; --force would re-sync." if force else " — no sync needed."
            emit(info(f"up to date at {remote_commit[:7]}{tail}"))
        elif local_commit:
            emit(
                info(
                    f"sync would run: update {local_commit[:7]} → {remote_commit[:7]}."
                )
            )
        else:
            emit(info(f"sync would run: install {remote_commit[:7]} (none installed)."))
        return

    if up_to_date and not force:
        emit(ok(f"official skills already up to date ({remote_commit[:7]})."))
        return

    official = paths.official_dir()
    official.mkdir(parents=True, exist_ok=True)

    # 2. Download the pack to a temp file (/tmp), hashing as we stream so the
    #    whole pack never sits in memory. Quiet (no progress bar / announce
    #    line) — the pack is small; only the final synced result is shown.
    zip_path, actual = _download_to_temp(urls["zip"])
    src: dict = {}  # the pack's manifest, parsed once from staging
    hash_notes: list[str] = []
    try:
        # 3. Integrity — compare our digest to the published .sha256 BEFORE we
        #    touch the install. Mismatch = corrupt/tampered → abort untouched.
        expected = _parse_sha256(
            _fetch_bytes(urls["sha256"]).decode("utf-8", "replace")
        )
        if expected is None:
            exit_error(f"{urls['sha256']} is not a valid sha256sum line.")
        if actual != expected:
            exit_error(
                f"checksum mismatch — expected {expected[:12]}…, got "
                f"{actual[:12]}…. Aborting; the install was not touched."
            )

        # 4. Extract to staging (same filesystem as the target) with zip-slip
        #    guard, then verify the shape.
        staging = official / STAGING_DIR
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            _extract_zip(zip_path, staging)
            _sanity_check_layout(staging)

            # 5. Per-skill hash check — each extracted skill dir must match the
            #    digest the manifest declares. Change-detection, not integrity
            #    (the zip .sha256 already vouched for the bytes), so we collect
            #    mismatches and warn below rather than abort. Parse source.json
            #    once here (pre-swap) and reuse it for the reconcile in step 6.
            src = _read_source_json(staging / "source.json")
            hash_notes = _verify_skill_hashes(
                staging / "skills", src.get("skills") or []
            )

            # 6. Atomic swap of package-owned paths only, on the same
            #    filesystem; the full zip is the set, so removed skills drop.
            #    os.replace (NOT Path.rename): Windows' os.rename refuses to
            #    overwrite an existing target (WinError 183), so source.json —
            #    renamed straight onto the prior file — threw on every re-sync,
            #    leaving source.json + .sync-state.json stale (the state write
            #    below never ran) and re-syncing on every startup. os.replace
            #    overwrites atomically on Windows and POSIX alike.
            dst_skills = paths.official_skills_dir()
            if dst_skills.exists():
                shutil.rmtree(dst_skills)
            os.replace(staging / "skills", dst_skills)
            os.replace(staging / "source.json", official / "source.json")
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    finally:
        zip_path.unlink(missing_ok=True)

    # 7. Record state + reconcile the manifest against what actually landed.
    commit = str(src.get("commit") or remote_commit)
    landed = _count_skills(paths.official_skills_dir())
    _write_sync_state(urls["zip"], actual, commit, landed)

    emit(ok(f"synced {landed} official skill(s) @ {commit[:7]}."))
    for note in hash_notes:
        emit(warn(f"skill hash mismatch — {note}"))
    declared = len(src.get("skills", [])) if isinstance(src.get("skills"), list) else 0
    if declared and declared != landed:
        emit(warn(f"manifest lists {declared} skill(s) but {landed} landed on disk."))


def _read_source_json(p: Path) -> dict:
    """Parse the extracted ``source.json``; {} if unreadable (state/reconcile
    then fall back to the ``latest.json`` commit and an on-disk count)."""
    try:
        data = json.loads(read_text(p))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def maybe_auto_sync() -> None:
    """Kick off a best-effort official-skills sync at ``physiclaw server``
    startup, in a background daemon thread so it NEVER blocks startup — the
    network I/O happens off the critical path while the server begins serving.

    The cheap guards run synchronously here (so ``read_live`` sees only OTHER
    servers, before this one records itself live); only the sync itself is
    backgrounded. Silent no-op when disabled (``[skills] sync_auto = false``),
    under CI (unattended network work a pipeline never asked for), or another
    live server exists (its concurrent sync would race the ``official/skills``
    swap). Idempotent: an unchanged commit is a no-op with no download.

    Mutating ``official/skills`` mid-run is safe: the swap is an atomic
    ``os.replace`` and skill discovery snapshots per session (in the separate
    runtime process), so a session sees either the whole old or whole new tree
    and picks up the change at its next wake."""
    if not _load_config().skills.sync_auto:
        return
    if os.environ.get("CI", "").strip().lower() in ("1", "true", "yes"):
        return
    # Imported lazily — keeps this module import-light for the plain CLI path.
    from physiclaw.common import runtime_state

    if runtime_state.read_live():
        return  # another instance owns official/ — don't race its swap
    threading.Thread(
        target=_run_sync_quiet, name="physiclaw-auto-sync", daemon=True
    ).start()


def _run_sync_quiet() -> None:
    """The background-thread body: run ``sync`` fully fail-soft — a hard abort
    (``typer.Exit``, already on stderr) or any unexpected error is swallowed so
    the daemon thread dies quietly and the server is never affected."""
    try:
        sync(emit=_log_emit)
    except typer.Exit:
        pass
    except Exception:
        log.exception("auto-sync: official skills sync failed (non-fatal)")


def _log_emit(msg: str) -> None:
    """Outcome printer for the background sync: unstyle (no ANSI codes in
    the daily log file) and route through the server's logger so the line
    gets the ``HH:MM [physiclaw]`` shape like every other startup message."""
    log.info("%s", click.unstyle(msg))


__all__ = ["sync", "maybe_auto_sync"]
