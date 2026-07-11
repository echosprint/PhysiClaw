"""Live model-list cache.

Each provider's `/v1/models` response is JSON-cached at
`~/.physiclaw/discovered/<provider>.json` so subsequent commands can
answer "does the user actually have access to this model?" without
hitting the network. Refreshed by `physiclaw models key` (after a key
write) and `physiclaw models discover` (explicit re-fetch).

Cache shape:
    {
      "fetched_at":  "2026-04-27T11:23:45Z",
      "models":      [{"id": "...", ...}, ...]
    }

The cache is the source of truth for `models use` validation —
PhysiClaw doesn't ship a curated model list, so anything the
provider's API returns is fair game.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path

from physiclaw import paths
from physiclaw.text import read_text, write_text

log = logging.getLogger(__name__)

_DIR = paths.HOME / "discovered"


def cache_path(provider_id: str) -> Path:
    return _DIR / f"{provider_id}.json"


def save(provider_id: str, models: list[dict]) -> None:
    """Write the live list with a fetched-at timestamp. Best-effort —
    a write failure is logged but never raised; discovery still
    completes (the user got the printed list)."""
    _DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "models": models,
    }
    try:
        write_text(cache_path(provider_id), json.dumps(payload, indent=2))
    except OSError as e:
        log.warning("failed to write discovered cache for %s: %s", provider_id, e)


def load(provider_id: str) -> list[dict]:
    """Return the cached `models` list, or [] if the cache doesn't
    exist or is unreadable."""
    try:
        payload = json.loads(read_text(cache_path(provider_id)))
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as e:
        log.warning("failed to read discovered cache for %s: %s", provider_id, e)
        return []
    return payload.get("models") or []


def model_ids(provider_id: str) -> set[str]:
    return {m.get("id", "") for m in load(provider_id)} - {""}


def is_cached(provider_id: str, model_id: str) -> bool:
    return model_id in model_ids(provider_id)
