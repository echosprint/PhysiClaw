"""Top-level pytest fixtures and import-time isolation.

Two non-obvious pieces:

1. **`PHYSICLAW_HOME` is read once at `physiclaw.paths` import time** (line:
   `HOME = Path(os.environ.get("PHYSICLAW_HOME", "~/.physiclaw")).expanduser()`).
   We set it at module load, before any test or fixture runs, so the first
   `import physiclaw` resolves `paths.HOME` to a session-scoped tmp dir
   instead of the real `~/.physiclaw`.

2. **Per-test isolation** — the autouse `physiclaw_home` fixture re-points
   `paths.HOME` and `paths.LOG_DIR` to a per-test tmp dir, so tests that
   write under HOME don't bleed state across each other.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import pytest

# ---- module-load isolation (must run before any `import physiclaw`) ----

_SESSION_HOME = Path(tempfile.mkdtemp(prefix="physiclaw-test-session-"))
os.environ["PHYSICLAW_HOME"] = str(_SESSION_HOME)


# ---- per-test fixtures ----


@pytest.fixture(autouse=True)
def physiclaw_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test isolated `~/.physiclaw`. Re-points `paths.HOME` / `paths.LOG_DIR`.

    Yields the path so tests that need to inspect on-disk state can.
    """
    home = tmp_path / "physiclaw_home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PHYSICLAW_HOME", str(home))

    # `paths` was imported once at session start with the session tmp dir;
    # patch the live module attributes so callers in this test see the
    # per-test path.
    from physiclaw import paths as _paths

    monkeypatch.setattr(_paths, "HOME", home)
    monkeypatch.setattr(_paths, "LOG_DIR", home / "log")
    return home


@pytest.fixture(autouse=True)
def scrub_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip proxy env vars so the suite is hermetic on proxied shells.

    A VPN client's `all_proxy=socks://…` leaking in from the developer's
    shell once produced 97 bogus failures: every test constructing a
    `trust_env=True` httpx client (providers, mcp_tool, poll) died in
    the constructor on the unsupported scheme. Tests that exercise proxy
    behavior set the vars they need via monkeypatch, which runs after
    this scrub.
    """
    for key in list(os.environ):
        if key.lower() in ("all_proxy", "http_proxy", "https_proxy", "no_proxy"):
            monkeypatch.delenv(key)


@pytest.fixture
def silenced_log() -> None:
    """Suppress all `physiclaw.*` logger output for the duration of a test.

    Use when a unit test deliberately exercises an error path that emits
    `log.warning` / `log.error` and the noise drowns the test report.
    Don't apply blindly — silence by exception, not by default.
    """
    physiclaw_logger = logging.getLogger("physiclaw")
    prior_level = physiclaw_logger.level
    physiclaw_logger.setLevel(logging.CRITICAL + 1)
    try:
        yield
    finally:
        physiclaw_logger.setLevel(prior_level)
