"""Debug wake hook — the e2e harness's trigger, one-shot and gated.

Fires when `debug/wake.json` exists AND the server runs in debug mode
(`physiclaw debug` exports DEBUG_ENV_VAR; this runtime inherits it) —
the
file alone must never wake a production runtime. Consumed on read (the
`suspended.json` idiom), so a stale file cannot re-wake the agent a
session later. The `physiclaw debug` CLI writes it; the wake then flows
through the same hook loop, ready gate, and session lifecycle as a real
phone trigger.
"""

import json
import logging
import os

from physiclaw.agent.runtime.hook import Trigger, register
from physiclaw.common import paths
from physiclaw.common.config import DEBUG_ENV_VAR
from physiclaw.common.text import read_text

log = logging.getLogger(__name__)


def wake_path():
    return paths.debug_dir() / "wake.json"


@register
def debug_wake() -> Trigger | None:
    if not os.environ.get(DEBUG_ENV_VAR):
        return None
    p = wake_path()
    if not p.exists():
        return None
    description = "debug wake"
    try:
        description = json.loads(read_text(p)).get("description") or description
    except Exception:
        log.warning("debug wake file unreadable — waking with the default reason")
    p.unlink(missing_ok=True)  # one-shot: consumed on ANY read outcome
    log.info("debug wake fired: %s", description)
    return Trigger(description=description, source="debug")
