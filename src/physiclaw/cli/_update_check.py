"""Soft check for a newer ``physiclaw`` on PyPI.

Prints a single-line nudge after a diagnostic command (``doctor`` /
``status``) if a newer version is available. Never blocks the command;
fails silently on any error. Cache lives at
``~/.physiclaw/run/version-check.json`` with a 7-day TTL so we don't hit
PyPI more than weekly.

Disabled by setting ``PHYSICLAW_DISABLE_UPDATE_CHECK=1`` (matches Claude
Code's ``DISABLE_UPDATES`` convention). Skipped when stdout is not a TTY
so CI / piped output never sees the banner.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from physiclaw import __version__ as _pkg_version
from physiclaw import paths
from physiclaw.text import read_text, write_text

_PYPI_URL = "https://pypi.org/pypi/physiclaw/json"
_HTTP_TIMEOUT_SECONDS = 2.0
_CACHE_TTL_DAYS = 7
_DISABLE_ENV = "PHYSICLAW_DISABLE_UPDATE_CHECK"


def _cache_file() -> Path:
    return paths.HOME / "run" / "version-check.json"


def _read_cache() -> dict | None:
    p = _cache_file()
    if not p.exists():
        return None
    try:
        data = json.loads(read_text(p))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(latest_version: str) -> None:
    p = _cache_file()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        write_text(p, json.dumps({
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "latest_version": latest_version,
        }))
    except OSError:
        pass  # cache write failure is non-fatal — we'll re-fetch next time


def _cache_is_fresh(checked_at: str) -> bool:
    try:
        ts = datetime.fromisoformat(checked_at)
    except (TypeError, ValueError):
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - ts < timedelta(days=_CACHE_TTL_DAYS)


def _fetch_pypi_version() -> str | None:
    try:
        with urllib.request.urlopen(_PYPI_URL, timeout=_HTTP_TIMEOUT_SECONDS) as r:
            data = json.loads(r.read())
        v = data.get("info", {}).get("version")
        return v if isinstance(v, str) and v else None
    except Exception:
        return None


def _is_newer(current: str, latest: str) -> bool:
    """Return True if ``latest`` is strictly newer than ``current``.

    Compares dotted-integer versions as int tuples (so ``0.0.10`` beats
    ``0.0.5``). Falls back to a conservative ``False`` if either side
    has non-integer parts (pre/post/dev releases) — we'd rather miss a
    nudge than push a bad upgrade hint.
    """
    try:
        cur = tuple(int(p) for p in current.split("."))
        lat = tuple(int(p) for p in latest.split("."))
    except (ValueError, AttributeError):
        return False
    return lat > cur


def _resolve_latest() -> str | None:
    """Return the latest known version, from a fresh cache or PyPI."""
    cache = _read_cache()
    if cache and _cache_is_fresh(cache.get("checked_at", "")):
        cached = cache.get("latest_version")
        if isinstance(cached, str) and cached:
            return cached
    latest = _fetch_pypi_version()
    if latest:
        _write_cache(latest)
    return latest


def _disabled_via_env() -> bool:
    val = os.environ.get(_DISABLE_ENV, "").strip().lower()
    return val in ("1", "true", "yes")


def maybe_print_update_banner() -> None:
    """Print a one-line update nudge if PyPI advertises a newer version.

    Silent (returns without output) if:
      - ``PHYSICLAW_DISABLE_UPDATE_CHECK=1`` is set
      - ``sys.stdout`` is not a TTY (CI, piped output)
      - The cache is missing/stale and PyPI is unreachable
      - The current version is already at or beyond ``latest_version``
    """
    if _disabled_via_env():
        return
    if not sys.stdout.isatty():
        return
    latest = _resolve_latest()
    if not latest:
        return
    if not _is_newer(_pkg_version, latest):
        return
    print(
        f"\n! physiclaw {_pkg_version} → {latest} available. "
        "Update with: uv tool upgrade physiclaw",
        flush=True,
    )
