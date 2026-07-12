"""Live-server state file: ``~/.physiclaw/run/server.json``.

Written by ``physiclaw server`` when it binds, removed on shutdown.
``physiclaw doctor`` reads this to learn the *actual* host/port the server
is on — which can differ from ``config.toml`` when ``--port`` was passed
at startup.

Stale files (server crashed without cleanup) are detected via a pid liveness
check in :func:`read_live`.
"""

import json
import os
import sys
import time

from physiclaw.common import paths
from physiclaw.common.text import read_text, write_text


def write(
    host: str,
    port: int,
    *,
    model_ref: str | None = None,
    model_source: str | None = None,
) -> None:
    """Record this process as the running server. Overwrites any prior file.

    ``model_ref`` / ``model_source`` capture the `provider/model` ref the
    server resolved at startup so ``physiclaw doctor`` in a different
    shell can report the live choice instead of re-running ``resolve()``
    against a shell that doesn't have the env var set. ``None`` marks
    "not recorded"; readers gate on truthiness so the JSON-``null``
    round-trip works unchanged.
    """
    p = paths.runtime_state_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    write_text(
        p,
        json.dumps(
            {
                "pid": os.getpid(),
                "host": host,
                "port": port,
                "model_ref": model_ref,
                "model_source": model_source,
                "started_at": time.time(),
            }
        ),
    )


def clear() -> None:
    """Best-effort delete of this process's state file. Refuses to remove
    a file written by a different pid so stray debug scripts can't kill
    a live server's recorded state (the server re-writes only on its
    own startup, not per-request, so a mis-clear would persist until
    restart).
    """
    p = paths.runtime_state_file()
    try:
        state = json.loads(read_text(p))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return
    if state.get("pid") != os.getpid():
        return
    try:
        p.unlink()
    except FileNotFoundError:
        pass


def read_live() -> dict | None:
    """Return ``{pid, host, port, started_at}`` if a live server is recorded.

    Returns ``None`` for: missing file, malformed JSON, or stale pid.
    Doesn't delete stale files — the next ``server`` start overwrites them.
    """
    p = paths.runtime_state_file()
    if not p.exists():
        return None
    try:
        state = json.loads(read_text(p))
    except (OSError, json.JSONDecodeError):
        return None
    pid = state.get("pid")
    if not isinstance(pid, int) or not _pid_alive(pid):
        return None
    return state


def _pid_alive(pid: int) -> bool:
    """True if a process with this pid currently exists."""
    if sys.platform == "win32":
        # os.kill(pid, 0) is NOT a liveness probe on Windows: any signal
        # other than CTRL_C/BREAK_EVENT unconditionally TerminateProcess()es
        # the target — a doctor/status/update check would KILL the live
        # server. Probe with a query-only process handle instead.
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        ERROR_ACCESS_DENIED = 5
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        # Exists but owned by another user — still alive (mirrors the
        # PermissionError case below).
        return ctypes.get_last_error() == ERROR_ACCESS_DENIED
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but owned by another user — still alive
        return True
    except OSError:
        return False
    return True
