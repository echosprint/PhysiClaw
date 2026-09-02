"""The suspension file — the one thing a walk leaves behind.

``playbooks/suspended.json`` holds a walk that asked the user something
and stepped out of the way rather than burning a session waiting: a
tell was sent, or an ask ran out of patience polling for a reply. It carries the walk's whole position — cursor, agent outputs, the
gate's consent numbers — so the next wake picks up mid-purchase
instead of starting over.

One-shot: consumed on load, whatever the outcome. A crash mid-resume
loses the suspension and the wake runs plain, never a loop.

This is the ONLY cross-wake state the conductor writes. (There used to
be a second, ``armed.json`` — a standing order naming the playbook to
run next. The overture retired it: a playbook on disk is the grant, so
there is nothing left to pre-declare.)
"""

import json
import logging
from pathlib import Path

from physiclaw.common import paths
from physiclaw.common.text import read_text

log = logging.getLogger(__name__)

SUSPENDED_FILENAME = "suspended.json"
SUSPENDED_SCHEMA = 1


def suspended_path() -> Path:
    return paths.playbooks_dir() / SUSPENDED_FILENAME


def clear_suspended() -> bool:
    p = suspended_path()
    if not p.exists():
        return False
    p.unlink()
    return True


def suspended_ref() -> tuple[str, str] | None:
    """The suspended walk's ``(app, playbook)`` per the file, without
    validating anything else. None when nothing is suspended /
    unreadable."""
    p = suspended_path()
    try:
        if not p.exists():
            return None
        data = json.loads(read_text(p))
        return str(data["app"]), str(data["playbook"])
    except Exception:
        return None
