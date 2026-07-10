"""Retention for daily log files — shared by engine / runtime / claude dirs.

Session artifacts have their own purge (`agent.engine.trace._purge_old`);
this covers only the `<prefix>-YYYY-MM-DD.log` dailies, which are tiny by
comparison and so keep a longer window (CONFIG.retention.log_days).
"""
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)


def purge_daily_logs(dir: Path, prefix: str, days: int) -> None:
    """Delete `<prefix>-*.log` files in `dir` with mtime older than `days`.

    mtime beats filename-date parsing: it tolerates clock skew and files
    appended to long after creation. Fail-open — retention must never
    take down the process doing the logging."""
    cutoff = time.time() - days * 86400
    removed = 0
    try:
        entries = list(dir.glob(f"{prefix}-*.log"))
    except OSError:
        return
    for path in entries:
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            pass
    if removed:
        log.info(
            "purged %d %s daily log(s) older than %d days", removed, prefix, days,
        )
