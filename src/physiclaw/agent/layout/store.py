"""The layout store: schema, validation, persistence, rendering, the
`record` tool entry point, and the first-run tail reminder.

The agent captures three pages one at a time — Spotlight search, the IM chat
with the keyboard down, and the same chat with it up. `screenshot` already
returns every on-screen element with its bbox, so the AGENT reads the input /
keyboard / Send boxes off that list (tapping and `peek`-ing to confirm the
keyboard is up), then calls `report_screen_layout(page, field, bbox, app)` once
per box with the coordinates it measured. This module does NOT detect anything
— it only sanity-checks each box (valid geometry, in the rough region that
field belongs to) and, if it passes, merges it into the running layout and
writes
``~/.physiclaw/screen-layout/``:

  layout.json  the structured bboxes (source of truth, built up per page),
               plus a `layout_learned` bool so out-of-module readers (the core
               orchestrator's /api/status) can tell setup is done without
               importing this module or knowing its page/field schema
  layout.md    the rendered card that `prompt._render_screen_layout` injects

`is_learned()` (all pages captured) / `missing_pages()` / `load_layout_md()`
are the read side the prompt builder uses to decide first-run setup vs.
inject-the-layout.
"""

import json
import logging
from typing import NotRequired, TypedDict

from physiclaw.common import paths
from physiclaw.common.bbox import inside
from physiclaw.common.text import read_text, write_text
from physiclaw.contract.dto import Message, UserMessage

log = logging.getLogger(__name__)

# Slack around a learned box before a press counts as inside it — the
# package's tolerance for bbox re-transcription jitter, consumed by the
# lint and the keyboard belief. `stuck.MATCH_TOLERANCE` is deliberately
# a separate knob (see its comment).
BOX_MARGIN = 0.02


def inside_learned(center: tuple[float, float], box: list) -> bool:
    """`bbox.inside` with the learned-box slack applied — the one form
    this package tests containment with."""
    return inside(center, box, margin=BOX_MARGIN)


# Any chat app is supported — the agent passes whichever it's in. This just
# nice-cases a few common ones for the layout card; unknown apps are used
# verbatim (the agent may pass proper casing itself).
_APP_LABELS = {
    "wechat": "WeChat",
    "whatsapp": "WhatsApp",
    "telegram": "Telegram",
    "signal": "Signal",
    "messenger": "Messenger",
    "imessage": "iMessage",
    "line": "LINE",
    "wcom": "WeCom",
    "qq": "QQ",
    "kakaotalk": "KakaoTalk",
}


def _app_label(app: str) -> str:
    """Display name for the IM app — nice-cased if known, else as passed."""
    return _APP_LABELS.get(app.strip().lower(), app.strip())


# The three capture pages, and the layout fields each one contributes. Used to
# report what's still missing and to decide when setup is complete. Pages whose
# name starts with "chat" are IM-app-specific and require `app`.
PAGES = ("spotlight", "chat-no-keyboard", "chat-keyboard")
_PAGE_FIELDS = {
    # Spotlight owns the shared keyboard landmarks (space, backspace, and the
    # search/return key) — its keyboard is up and the search flow needs them.
    "spotlight": ("spotlight_input", "spotlight_paste", "backspace", "return", "space"),
    "chat-no-keyboard": ("chat_input_kb_hidden",),
    "chat-keyboard": ("chat_input_kb_visible", "send", "chat_paste"),
}
# Every field name, in capture order — the tool records one per call.
ALL_FIELDS = tuple(f for fields in _PAGE_FIELDS.values() for f in fields)


def _is_chat_page(page: str) -> bool:
    """Chat pages are IM-app-specific (need `app`); Spotlight is system-wide."""
    return page.startswith("chat")


def _load() -> dict:
    """Read the accumulated layout, or {} if none/unreadable yet."""
    p = paths.screen_layout_json()
    if not p.exists():
        return {}
    try:
        data = json.loads(read_text(p))
    except (OSError, ValueError):  # ValueError covers JSON + UTF-8 decode errors
        return {}
    return data if isinstance(data, dict) else {}


def missing_pages() -> list[str]:
    """Pages whose fields aren't all captured yet, in capture order."""
    d = _load()
    return [p for p in PAGES if not all(f in d for f in _PAGE_FIELDS[p])]


def is_learned() -> bool:
    """True once every page has been captured — keys the prompt's first-run
    branch off. ``~/.physiclaw/screen-layout/`` is otherwise incomplete."""
    return not missing_pages()


# Keys the user legitimately presses many times in a row (backspace-clearing
# a field, spacing through text). The stuck guard exempts these targets from
# same-target counting — repeated presses there are typing, not looping.
REPEATABLE_KEY_FIELDS = ("backspace", "space", "return")


def repeatable_key_boxes() -> list[list[float]]:
    """Bboxes of the learned keyboard keys that are safe to press
    repeatedly. Empty while the layout is unlearned."""
    d = _load()
    return [d[f] for f in REPEATABLE_KEY_FIELDS if isinstance(d.get(f), list)]


def load_layout_md() -> str:
    """The rendered layout card, or '' if nothing captured yet."""
    p = paths.screen_layout_md()
    return read_text(p).strip() if p.exists() else ""


# ─── First-run tail reminder ──────────────────────────────────


def tail_reminder() -> str:
    """A first-run nudge listing, per page, which fields are still missing and
    what to do — pinned at the request tail so it's fresh and reflects capture
    progress. '' once every field is captured (nothing to append)."""
    d = _load()
    missing = {p: [f for f in _PAGE_FIELDS[p] if f not in d] for p in PAGES}
    missing = {p: fs for p, fs in missing.items() if fs}
    if not missing:
        return ""
    done = [p for p in PAGES if p not in missing]
    lines = [
        "[First-run setup needed]",
        "You can't reliably open apps by Spotlight search or send IM messages "
        "until the screen layout is learned.",
    ]
    if done:
        lines.append(f"Pages fully captured: {', '.join(done)}.")
    lines.append("Still to capture (page → missing fields):")
    lines += [f"  • {p}: {', '.join(fs)}" for p, fs in missing.items()]
    lines.append(
        "Follow the `screen-layout` skill (under `## Built-in Skills` when "
        "inlined in your system prompt; otherwise load it with the Skill "
        "tool) to capture each field above."
    )
    return "\n".join(lines)


def inject_tail(messages: list[Message]) -> list[Message]:
    """Append the first-run setup reminder as the LAST message while the
    screen layout isn't fully learned; return `messages` unchanged once it
    is. Wake/turn-volatile (reflects capture progress), so it rides the tail
    rather than the cached SYSTEM prompt. Original list is not mutated."""
    reminder = tail_reminder()
    return messages + [UserMessage(content=reminder)] if reminder else messages


# ─── Validation ───────────────────────────────────────────────
#
# The agent measures the boxes; we only reject ones that are clearly wrong.
# Each field has a rough region (center within a band, minimum width) — coarse
# on purpose, just enough to catch a mis-picked element before it's persisted.


class _Region(TypedDict):
    """Rough region for one field — any key may be absent."""

    cx: NotRequired[tuple[float, float]]  # (lo, hi) band for the center x
    cy: NotRequired[tuple[float, float]]  # (lo, hi) band for the center y
    wmin: NotRequired[float]  # minimum width


_REGION: dict[str, _Region] = {
    # Bands are calibrated from real WeChat/Spotlight pages, with margin for
    # other apps/devices. Spotlight's field sits LOW — just above the keyboard
    # (~y 0.60); the chat input bar is at the very bottom with the keyboard down
    # (~y 0.94) and just above the keyboard with it up (~y 0.61).
    "spotlight_input": {"cy": (0.50, 0.70), "wmin": 0.70},
    "chat_input_kb_hidden": {"cy": (0.80, 0.99), "wmin": 0.40},
    "chat_input_kb_visible": {"cy": (0.50, 0.70), "wmin": 0.20},
    # Paste is the edit-menu button raised by long-pressing the input box — a
    # small bubble that floats just ABOVE the field (~y 0.53). No width floor
    # (it's a button, not a bar); the floor allows for the callout riding a
    # little higher above a tall input box.
    "spotlight_paste": {"cy": (0.40, 0.72)},
    "chat_paste": {"cy": (0.40, 0.72)},
    # Keyboard: space bar and bottom-right key (~y 0.90), backspace one row up
    # (~y 0.84), all on the right for the two corner keys.
    "space": {"cy": (0.80, 1.00), "wmin": 0.20},
    "backspace": {"cy": (0.70, 1.00), "cx": (0.80, 1.00)},
    "return": {"cy": (0.80, 1.00), "cx": (0.80, 1.00)},
    # Send sits in one of two places, app-depending: a keyboard key at the
    # bottom (WeChat-style) OR a button on the input bar's right (WhatsApp /
    # Telegram / most apps). Either way it's on the right and in the lower part
    # of the screen, so one broad region covers every app.
    "send": {"cy": (0.50, 1.00), "cx": (0.80, 1.00)},
}


def _validate_box(field: str, raw) -> str | None:
    """Return an error string if `raw` is not a plausible bbox for `field`,
    else None. Checks geometry then the field's rough region."""
    try:
        bbox = [float(v) for v in raw]
    except (TypeError, ValueError):
        return f"{field}: bbox must be four numbers [left, top, right, bottom]."
    if len(bbox) != 4:
        return f"{field}: expected 4 numbers, got {len(bbox)}."
    left, top, right, bottom = bbox
    if not all(0.0 <= v <= 1.0 for v in bbox):
        return f"{field}: coordinates must be normalized 0–1 ({bbox})."
    if left >= right or top >= bottom:
        return f"{field}: need left<right and top<bottom ({bbox})."

    region: _Region = _REGION.get(field, {})
    cx, cy, width = (left + right) / 2, (top + bottom) / 2, right - left
    if "cx" in region and not region["cx"][0] <= cx <= region["cx"][1]:
        return f"{field}: looks off — expected it toward x∈{region['cx']} (got center x={cx:.2f})."
    if "cy" in region and not region["cy"][0] <= cy <= region["cy"][1]:
        return f"{field}: looks off — expected it toward y∈{region['cy']} (got center y={cy:.2f})."
    if "wmin" in region and width < region["wmin"]:
        return f"{field}: too narrow for an input bar (width {width:.2f} < {region['wmin']})."
    return None


# ─── Rendering ────────────────────────────────────────────────


def _fmt(bbox) -> str:
    if not bbox:
        return "(not captured yet)"
    return "[" + ", ".join(f"{v:.3f}" for v in bbox) + "]"


def _render_md(layout: dict) -> str:
    """Render the accumulated layout as the markdown injected into SYSTEM.
    Partial-aware — uncaptured boxes show `(not captured yet)`."""
    im_app = layout.get("im_app")
    app = f"{im_app}'s" if im_app else "per-IM-app"
    return "\n".join(
        [
            "Key input boxes and keyboard landmarks, measured from your phone — "
            "`[left, top, right, bottom]`, 0–1. Trust these over generic priors; "
            "re-ground from the current view if a tap misses.",
            "",
            f"**Chat input + Send below are {app}.** For any other IM app, ignore "
            "them and ground live. Spotlight + keyboard keys are system-wide.",
            "",
            "### Input boxes",
            "",
            f"- Spotlight search {_fmt(layout.get('spotlight_input'))} — paste an app name to open it",
            f"- Chat input, keyboard hidden {_fmt(layout.get('chat_input_kb_hidden'))} — tap to raise the keyboard",
            f"- Chat input, keyboard visible {_fmt(layout.get('chat_input_kb_visible'))} — long-press to Paste",
            f"- Send {_fmt(layout.get('send'))} — tap to send an IM",
            "",
            "### Paste buttons",
            "",
            "Only exist AFTER long-pressing the input box (above):",
            "",
            f"- Spotlight search → Paste at {_fmt(layout.get('spotlight_paste'))}",
            f"- Chat input (keyboard visible) → Paste at {_fmt(layout.get('chat_paste'))}",
            "",
            "### Keyboard keys",
            "",
            f"- backspace ⌫ {_fmt(layout.get('backspace'))}",
            f"- return / search / send {_fmt(layout.get('return'))}",
            f"- space {_fmt(layout.get('space'))}",
            "",
            "Bottom-right key: label varies (`Send` / `Return` / `Search` / `搜索` / "
            "`前往`), bbox stable. Tap not registering → check the current view: is "
            "the keyboard actually visible?",
        ]
    )


# ─── Record (the tool entry point) ────────────────────────────


def record(page: str, field: str, bbox, app: str | None = None) -> str:
    """Validate ONE box the agent measured — `field` (e.g. 'send') with its
    `bbox` on `page` — and, if it passes, merge it into layout.json and
    re-render layout.md. `app` (any chat app) is required for the chat pages:
    it labels the chat boxes as belonging to that app. Returns a message for
    the agent — the layout so far plus what's still to capture, or a re-measure
    instruction on failure."""
    if page not in PAGES:
        return f"unknown page {page!r}; expected one of: {', '.join(PAGES)}."
    allowed = _PAGE_FIELDS[page]
    if field not in allowed:
        return (
            f"{field!r} is not a {page} field; expected one of: {', '.join(allowed)}."
        )
    if _is_chat_page(page) and not (app and app.strip()):
        return (
            f"{page} needs the IM app: pass app='<the chat app you're in>' "
            "(e.g. 'wechat', 'whatsapp', 'telegram', ...) so the chat boxes are "
            "labelled with their app."
        )
    err = _validate_box(field, bbox)
    if err:
        return (
            f"Layout not saved — {err} Re-read the box off the screenshot "
            "(copy the element's bbox verbatim) and call again."
        )

    layout = _load()
    # Reporting the same field again is fine — it OVERWRITES (the re-measure /
    # correction path). `was_complete` distinguishes the call that finishes
    # setup (→ restart) from a later correction to an already-complete layout.
    was_complete = not any(f not in layout for f in ALL_FIELDS)
    layout[field] = [round(float(v), 3) for v in bbox]
    if _is_chat_page(page) and app and app.strip():
        layout["im_app"] = _app_label(app)
    still = [f for f in ALL_FIELDS if f not in layout]
    # Persist a plain `learned` flag alongside the boxes so readers outside
    # this module (e.g. the core orchestrator's /api/status) can tell setup is
    # done by reading the file — without importing this module or knowing its
    # page/field schema. `is_learned()` derives the same truth from the schema.
    layout["layout_learned"] = not still
    md = _render_md(layout)
    paths.screen_layout_dir().mkdir(parents=True, exist_ok=True)
    write_text(paths.screen_layout_json(), json.dumps(layout, indent=2))
    write_text(paths.screen_layout_md(), md + "\n")

    if still:
        # The per-turn first-run notice (tail_reminder) already lists which
        # fields remain and points at the skill — don't repeat it here.
        note = f"{len(still)} box(es) still to capture — see the first-run notice."
    elif not was_complete:
        note = (
            "All boxes captured — setup is done. If you woke for a task, the "
            "session restarts with the layout loaded to finish it; otherwise it "
            "just ends (the layout is saved and loads on the next wake)."
        )
    else:
        note = (
            "The layout was already complete; this updated one box. No restart "
            "needed — carry on with the request."
        )
    return f"Saved `{field}` = {layout[field]}. {note}"
