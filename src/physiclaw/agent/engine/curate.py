"""Post-session pitfalls curator — a separate lightweight LLM pass that
consolidates the learned-pitfalls list and enforces its cap.

The agent only *appends* traps (0–3/session, append-only), so over time the list
accumulates near-duplicates, over-specific one-offs, and stale entries. Once a
session that added traps closes, `curate(...)` runs one throwaway LLM call over the
whole list with a fixed prompt: merge duplicates, tighten wording, drop stale /
low-value entries, and return **≤`max_items`** items — while PRESERVING every
distinct trap (Hermes' hard lesson: a wholesale rewrite is where good lines get
lost, so the prompt forbids dropping information). The result is written back
through `pitfalls.replace`, which also hard-cuts to the cap and snapshots
history, so a prune is always recoverable.

Reuses the session's still-open provider, makes NO tool calls, and is fully
fail-open: a parse error or empty result leaves the list untouched.
"""
import json
import logging

from physiclaw.agent.engine import pitfalls
from physiclaw.agent.engine.dto import SystemMessage, UserMessage
from physiclaw.config import CONFIG

log = logging.getLogger(__name__)


def _system() -> str:
    cap = CONFIG.pitfalls.max_items
    return (
        "You curate a list of learned pitfalls — one-line traps an agent hit "
        "while driving phone apps, each `app: trap → avoid/fix`. Consolidate "
        "the list, but PRESERVE EVERY DISTINCT TRAP: merge near-duplicates into "
        "one line that keeps the nuance, tighten wording, drop only stale or "
        "low-value entries. When unsure whether two are the same trap, keep "
        "both. Keep each line terse (lead with the app) and keep the most "
        "useful nearer the top. Do NOT invent traps.\n"
        f"Return ONLY a JSON array of strings, at most {cap} items, no prose, "
        "no code fence."
    )


def _parse(text: str) -> list[str] | None:
    """Parse the model's JSON array reply into a list of strings. Robust to a
    ```json fence and surrounding prose (parses the outermost `[...]` span).
    None on anything malformed, so the caller skips and the list is untouched."""
    s = text or ""
    start, end = s.find("["), s.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        arr = json.loads(s[start : end + 1])
    except (ValueError, TypeError):
        return None
    if not isinstance(arr, list):
        return None
    return [x for x in arr if isinstance(x, str) and x.strip()]


async def curate(provider, *, tr=None) -> bool:
    """Consolidate the pitfalls list in place. Returns True if it was rewritten.
    Fail-open: any error / unparseable / empty reply leaves the list as-is."""
    current = pitfalls.read()
    if not current:
        return False
    try:
        asst = await provider.chat(
            [SystemMessage(content=_system()),
             UserMessage(content="\n".join(f"- {x}" for x in current))],
            [],
        )
        items = _parse(asst.content or "")
        if not items:
            log.warning("curate: unparseable/empty pitfalls reply — skipped")
            return False
        res = pitfalls.replace(items)
        # Surface the change (never silent) — the pre-curate snapshot is one
        # history entry back, so a lossy consolidation is recoverable.
        log.info(
            "curate pitfalls: %d → %d (%d dropped over cap)",
            len(current), res["total"], res["dropped"],
        )
        if tr is not None:
            tr.write({"event": "curated_pitfalls", "before": len(current),
                      "after": res["total"], "dropped": res["dropped"]})
        return True
    except Exception:
        log.exception("curate: pitfalls pass failed (non-fatal)")
        return False
