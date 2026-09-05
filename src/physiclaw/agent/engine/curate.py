"""Post-session pitfalls curator — a separate lightweight LLM pass that
consolidates the learned-pitfalls list and enforces its cap.

The agent authors traps mid-run (0–3/session, append-only) and their quality
varies, so over time the list accumulates near-duplicates, over-specific one-offs,
vague/low-signal lines, and stale entries. Once a session that added traps closes,
`curate(...)` runs one throwaway LLM call over the whole list with a fixed prompt:
merge duplicates, tighten wording, drop the weak/stale entries, and return
**≤`max_items`** items ordered best-first (most durable, broadly-useful traps on
top). Merging (not wholesale rewriting) guards the Hermes lesson — a from-scratch
rewrite is where good lines get lost, so the prompt keeps both when two might be
distinct and forbids *inventing* traps. The best-first order is load-bearing:
`pitfalls.replace` hard-cuts to the cap from the bottom, so the lowest-quality
lines are the ones dropped. It also snapshots history, so a prune is recoverable.

Reuses the session's still-open provider, makes NO tool calls, and is fully
fail-open: a parse error or empty result leaves the list untouched.
"""

import logging

from physiclaw.agent.engine import pitfalls
from physiclaw.common.config import CONFIG
from physiclaw.common.text import json_span
from physiclaw.contract.dto import USAGE_CALL_CURATE, SystemMessage, UserMessage

log = logging.getLogger(__name__)


def _system() -> str:
    cap = CONFIG.pitfalls.max_items
    return (
        "You curate a list of learned pitfalls — one-line traps an agent hit "
        "while driving phone apps, each `app: trap → avoid/fix`. The agent "
        "authors these mid-run, so quality varies: some are sharp and reusable, "
        "others are vague, over-specific to one screen, or restate the obvious. "
        "Your job is to consolidate the list and surface the best.\n"
        "Rules:\n"
        f"- Return AT MOST {cap} items — a hard cap, not a target. Fewer is fine.\n"
        "- ORDER BY QUALITY, best first: put the most durable, broadly-useful, "
        "actionable traps at the top; the weakest last. This ordering matters — "
        "if the list is trimmed to the cap, the lowest ones are dropped.\n"
        "- Merge near-duplicates into one line that keeps the nuance; when unsure "
        "whether two are the same trap, keep both (as separate lines).\n"
        "- Drop stale, vague, one-off, or low-signal entries rather than pad to "
        "the cap. Tighten wording; lead each line with the app; keep it terse.\n"
        "- Do NOT invent traps or add information not already present.\n"
        "Return ONLY a JSON array of strings, no prose, no code fence."
    )


def _parse(text: str) -> list[str] | None:
    """Parse the model's JSON array reply into a list of strings (the
    shared `json_span` idiom: fence/prose-tolerant outermost `[...]`).
    None on anything malformed, so the caller skips and the list is
    untouched."""
    arr = json_span(text or "", "[", "]")
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
            [
                SystemMessage(content=_system()),
                UserMessage(content="\n".join(f"- {x}" for x in current)),
            ],
            [],
            purpose=USAGE_CALL_CURATE,
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
            len(current),
            res["total"],
            res["dropped"],
        )
        if tr is not None:
            tr.write(
                {
                    "event": "curated_pitfalls",
                    "before": len(current),
                    "after": res["total"],
                    "dropped": res["dropped"],
                }
            )
        return True
    except Exception:
        log.exception("curate: pitfalls pass failed (non-fatal)")
        return False
