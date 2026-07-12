"""Context compaction — two jobs, both about keeping image payload small:

  1. `drop_stale_screens` — "latest screen wins": only the most recent
     image-bearing tool_result keeps its image + full listing. Views
     arrive from `peek`/`screenshot` AND from every gesture (the fused
     post-action view), so staleness is keyed on ImageBlock presence,
     not tool names. Earlier view results are stubbed down to a marker
     line (`(superseded <tool>) — labels only, in order`) plus their
     surviving text: the action result line (verdict included — the
     retry history must stay legible) and the listing reduced to LABELS
     ONLY — each `[text]` row's id index, kind tag, bbox and confidence
     are dropped, leaving just the label text, one per line, in the
     original top-to-bottom order and set off from the action line by a
     blank line. Once the screen is gone those fields
     are dead weight: you can't tap a bbox on a page you've left or
     reference an old element's id, and a past detection's confidence is
     moot — the label is the only content worth remembering (and it's
     ~20% of a row's tokens; the rest is the 80% we drop). Icon rows are
     label-less and dropped; the listing header is dropped too. The
     assistant message and its tool_calls stay intact, so the decision
     history ("I called peek here, tapped there") is preserved.

  2. `scale_image_bytes` — ingress re-encode: normalize every incoming
     tool-result image to JPEG with long edge ≤ MAX_IMAGE_EDGE. Drops PNG
     transparency (fine for screenshots), typically cuts payload 3–10×.

`drop_stale_screens` operates on `Message` DTOs: it replaces each stale
`ImageBlock`-bearing `ToolResultMessage`'s content with the stub string
AND sets `is_superseded=True` on the new DTO. Providers find the latest
stub via the typed flag — no string parsing across modules.
"""

import json
import logging
import re

import cv2
import numpy as np

from physiclaw.agent.engine import memory
from physiclaw.agent.engine.dto import (
    AssistantMessage,
    ContentBlock,
    ImageBlock,
    Message,
    TextBlock,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from physiclaw.common.config import CONFIG

log = logging.getLogger(__name__)

MAX_IMAGE_EDGE = CONFIG.compact.max_image_edge_px
JPEG_QUALITY = CONFIG.compact.jpeg_quality

# Human-readable lead-in for the stubbed content. Operators reading raw
# logs see `(superseded peek)` / `(superseded tap)` and immediately know
# what happened. Cache-marker code does NOT parse this — providers find
# stubs via the `is_superseded` flag on `ToolResultMessage`.
STUB_PREFIX = "(superseded "
# Suffix on the marker line — reminds the model this view was reduced to
# on-screen labels only, in their original order, so it doesn't mistake a
# bare label list for the current screen or try to re-tap a vanished bbox.
STUB_NOTE = " — labels only, in order"

# The element-listing shape produced by `core.vision.util.format_elements`
# — keep the header string and the row regexes in sync with that formatter.
_LISTING_HEADER = 'id [kind] "label" [left,top,right,bottom] conf'
# Prefix matcher — "is this line an element row?" (kept for the non-row
# line strategy in tests; `_stub_body` itself uses the full-shape regexes
# below so its output is stable under re-application).
_ROW_RE = re.compile(r"^\d+ \[(icon|text)\] ")
# Full-shape matchers. `_TEXT_ROW_RE` captures the label (group 1); the
# greedy `.*` plus a `]`-free bbox class lets a label carry quotes and
# brackets (`He said "hi" [ok]`) yet still peel off the trailing bbox +
# confidence. `_ICON_ROW_RE` requires the empty-label icon shape so a
# stubbed label that merely *looks* icon-ish isn't re-dropped on a second
# pass (idempotency).
_TEXT_ROW_RE = re.compile(r'^\d+ \[text\] "(.*)" \[[^\]]*\] [0-9.]+\s*$')
_ICON_ROW_RE = re.compile(r'^\d+ \[icon\] "" \[[^\]]*\] [0-9.]+\s*$')


# ---------- summary collapse (turn-age pruning) ----------

# Three pre-allocated slots, allocated at session bootstrap right after
# `[system, trigger]`, so `collapse_old_turns` always knows where to
# write — no "find existing slot" lookup, no shifting positions when
# the slots fill:
#
#   messages[2]  summary slot     `note(summary=...)` bullets
#   messages[3]  memory slot      `read_memory` / `read_logs` results
#   messages[4]  skills slot      `Skill(...)` results (bodies + refs)
#
# Memory and skill loads are durable artifacts the agent loaded
# specifically for persistent reference — summarizing them would
# defeat the load. They get carried through collapse as full
# tool_call args + result text. Multiple loads concat with `\n\n`
# (no dedup; if the agent reloaded the same skill, both copies
# survive — keeps the implementation predictable).
SUMMARY_HEADER = "[earlier turns]"
SUMMARY_INITIAL = f"{SUMMARY_HEADER}\n(none yet)"

MEMORY_HEADER = "[memory loads]"
MEMORY_INITIAL = f"{MEMORY_HEADER}\n(none yet)"

SKILLS_HEADER = "[loaded skills]"
SKILLS_INITIAL = f"{SKILLS_HEADER}\n(none yet)"

# Tool calls whose results are durable artifacts. Their tool_results
# would otherwise be dropped by the slice; the collapse harvests them
# into the dedicated slots so subsequent turns can still see what the
# agent loaded earlier.
MEMORY_TOOL_NAMES = frozenset({"read_memory", "read_logs"})
SKILL_TOOL_NAMES = frozenset({"Skill"})

# First message index after the three pre-allocated slots. The collapse
# range is `[_FIRST_TURN_INDEX:cut]`.
_FIRST_TURN_INDEX = 5

# Three knobs for collapse behavior, all on the `Provider` class:
#
#   F = COLLAPSE_FIRST_AT_TURN     first collapse fires at this turn
#   K = KEEP_RECENT_TURNS          recent turns kept intact per collapse
#   I = COLLAPSE_INTERVAL_TURNS    cadence between subsequent collapses
#
# Defaults (F=30, K=10, I=20) are an EOQ optimum for vendors with
# anchored caches (Anthropic/Qwen). Moonshot accepts the same I=20
# despite its whole-prefix cache invalidation — the EOQ analysis
# alone would suggest I≈30 there, but the tighter prompt at I=20
# wins on long-session quality. See `MoonshotProvider` for the
# trade-off rationale.


def new_summary_placeholder() -> UserMessage:
    """The pre-allocated summary slot. Engine bootstrap puts this at
    `messages[2]`; `collapse_old_turns` mutates it in place once enough
    turns accumulate. Empty-state body keeps the position stable from
    turn 0 — no "first collapse inserts a new message and shifts all
    indices" surprise."""
    return UserMessage(content=SUMMARY_INITIAL)


def new_memory_placeholder() -> UserMessage:
    """The pre-allocated memory-load slot at `messages[3]`. Pre-populated
    at bootstrap with the latest log entries as a synthetic `read_logs`
    artifact so recent activity is in context from turn 0; falls back to
    `(none yet)` when no log files exist."""
    log_text = memory.load_recent_entries(memory.BOOTSTRAP_LOG_ENTRIES)
    if not log_text:
        return UserMessage(content=MEMORY_INITIAL)
    entry = _format_artifact_text(
        "read_logs",
        {"entries": memory.BOOTSTRAP_LOG_ENTRIES},
        log_text,
    )
    return UserMessage(content=_render_slot(MEMORY_HEADER, [entry], sep="\n\n"))


def new_skills_placeholder() -> UserMessage:
    """The pre-allocated skills-load slot at `messages[4]`. Holds the
    full body of any `Skill(...)` result that would otherwise be
    dropped by collapse. The skill body is the workflow doctrine the
    agent is following; a one-line summary cannot replace it."""
    return UserMessage(content=SKILLS_INITIAL)


def _trigger_threshold(
    messages: list[Message], *, first_at: int, interval: int, keep: int
) -> int | None:
    """The complete-turn count at which a collapse fires, or None when
    the pre-allocated slots are missing. ONE source of truth shared by
    `collapse_pending` and `collapse_old_turns` — the predictor and the
    trigger can never disagree on the firing turn."""
    if (
        len(messages) < _FIRST_TURN_INDEX
        or not isinstance(messages[2], UserMessage)
        or not isinstance(messages[3], UserMessage)
        or not isinstance(messages[4], UserMessage)
    ):
        return None
    is_first = messages[2].content == SUMMARY_INITIAL
    return first_at if is_first else keep + interval


def collapse_pending(
    messages: list[Message],
    *,
    first_at: int,
    interval: int,
    keep: int,
) -> bool:
    """True when the turn ABOUT to run will, once complete, trigger
    `collapse_old_turns` — i.e. this is the model's LAST turn with the
    full history in context. The engine uses it to post the checkpoint
    tail and require a `note(scratchpad=...)` write-down before the
    folding erases everything but summary lines.

    Same threshold as the collapse itself; the +1 counts the upcoming
    turn. Missing slots → False (collapse would refuse too)."""
    threshold = _trigger_threshold(
        messages,
        first_at=first_at,
        interval=interval,
        keep=keep,
    )
    if threshold is None:
        return False
    turns = sum(isinstance(m, AssistantMessage) for m in messages)
    return turns + 1 >= threshold


def inject_checkpoint_tail(messages: list[Message], *, keep: int) -> list[Message]:
    """Append the pre-compression checkpoint notice — the engine adds it
    on `collapse_pending` turns only, as one of the LAST things the model
    sees. Original list is not mutated."""
    return messages + [
        UserMessage(
            content=(
                f"⚠ Context compresses after this turn: every turn except the "
                f"newest {keep} folds to its note summary line — screens, "
                "listings and results there are erased. Your `note` this turn "
                "REQUIRES `scratchpad`: rewrite it (full replacement) with what "
                "your future self needs — facts and values collected (IDs, "
                "prices, counts), the current situation, and what remains. "
                "Scratchpad and plan survive compression in full."
            )
        )
    ]


def collapse_old_turns(
    messages: list[Message],
    *,
    first_at: int,
    interval: int,
    keep: int,
) -> None:
    """Fold turns older than `keep` into three slots:
      - `messages[2]` — `note(summary=...)` bullets
      - `messages[3]` — `read_memory` / `read_logs` results in full
      - `messages[4]` — `Skill(...)` bodies + references in full

    Cuts at turn boundaries (an `AssistantMessage` starts a turn) so
    every collapsed `tool_use` block has its matching `tool_result`
    collapsed too — no API-rejecting orphans.

    Trigger:
      - First collapse: when complete-turn count reaches `first_at`.
      - Subsequent: when count reaches `keep + interval`.

    All three knobs come from the active provider's class attributes
    (`COLLAPSE_FIRST_AT_TURN`, `KEEP_RECENT_TURNS`, `COLLAPSE_INTERVAL_TURNS`).
    The engine threads them through; this function stays vendor-
    agnostic.

    "First vs subsequent" is detected by the summary slot's content —
    the placeholder body persists until the first collapse rewrites it.

    Source material:
      - summary: each turn's `note(summary=...)` — string concat,
        flattened to one line per bullet, no model call.
      - memory / skills: paired (tool_use, tool_result) carried
        verbatim. These are durable artifacts the agent loaded
        specifically for persistent reference; a summary cannot
        replace them. Multiple loads concat with `\\n\\n`. No dedup —
        if the agent reloaded the same skill, both copies survive.

    Each collapse mutates the prefix bytes between system and the next
    stub anchor → triggers one cache_creation event. The vendor's
    defaults amortize this tax against the bounded-prompt savings
    (EOQ analysis in this file's module-level comment).
    """
    threshold = _trigger_threshold(
        messages,
        first_at=first_at,
        interval=interval,
        keep=keep,
    )
    if threshold is None:
        log.warning("collapse_old_turns: missing summary/memory/skill slots")
        return

    turn_starts = [i for i, m in enumerate(messages) if isinstance(m, AssistantMessage)]
    if len(turn_starts) < threshold:
        return

    cut = turn_starts[-keep]

    # Carry forward existing slot bodies so running history stays
    # continuous across multiple collapse events.
    summary_lines = _carry_items(messages[2].content, SUMMARY_HEADER, sep="\n")
    memory_entries = _carry_items(messages[3].content, MEMORY_HEADER, sep="\n\n")
    skill_entries = _carry_items(messages[4].content, SKILLS_HEADER, sep="\n\n")

    # Pair tool_call ids with their results within the collapsed range.
    # The cut is at a turn boundary, so every AssistantMessage in
    # [..cut] has its ToolResultMessage in [..cut].
    result_by_id: dict[str, ToolResultMessage] = {
        m.tool_call_id: m
        for m in messages[_FIRST_TURN_INDEX:cut]
        if isinstance(m, ToolResultMessage)
    }

    for i in range(_FIRST_TURN_INDEX, cut):
        m = messages[i]
        if not isinstance(m, AssistantMessage):
            continue
        for tc in m.tool_calls:
            if tc.name == "note":
                s = (tc.arguments.get("summary") or "").replace("\n", " ").strip()
                if s:
                    summary_lines.append(f"- {s}")
            elif tc.name in MEMORY_TOOL_NAMES:
                entry = _format_artifact(tc, result_by_id.get(tc.id))
                if entry:
                    memory_entries.append(entry)
            elif tc.name in SKILL_TOOL_NAMES:
                entry = _format_artifact(tc, result_by_id.get(tc.id))
                if entry:
                    skill_entries.append(entry)

    if not (summary_lines or memory_entries or skill_entries):
        return  # nothing salvageable; leave bytes in place

    messages[2:cut] = [
        UserMessage(content=_render_slot(SUMMARY_HEADER, summary_lines, sep="\n")),
        UserMessage(content=_render_slot(MEMORY_HEADER, memory_entries, sep="\n\n")),
        UserMessage(content=_render_slot(SKILLS_HEADER, skill_entries, sep="\n\n")),
    ]


def _carry_items(existing: str | list, header: str, *, sep: str) -> list[str]:
    """Pull `sep`-separated items out of an existing slot body. Used
    to round-trip a slot's content across consecutive collapses so
    the running history stays continuous."""
    if not isinstance(existing, str) or not existing.startswith(header):
        return []
    body = existing.removeprefix(header).lstrip()
    if not body or body == "(none yet)":
        return []
    return [item for item in body.split(sep) if item]


def _render_slot(header: str, items: list[str], *, sep: str) -> str:
    body = sep.join(items) if items else "(none yet)"
    return f"{header}\n{body}"


def _format_artifact(tc: ToolCall, result: ToolResultMessage | None) -> str:
    """Format one (tool_call, tool_result) pair as a slot entry. Skips
    error results and non-string content (image-bearing tool_results
    aren't memory/skill loads anyway)."""
    if result is None or result.is_error:
        return ""
    if not isinstance(result.content, str):
        return ""
    return _format_artifact_text(tc.name, tc.arguments, result.content)


def _format_artifact_text(name: str, arguments: dict, content: str) -> str:
    """Render the slot-entry text for a (tool_name, args, result) triple.
    Shared by real artifact harvesting and synthetic bootstrap entries
    so both surfaces produce identical formatting."""
    args_str = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    return f"{name}({args_str}) →\n{content}"


def scale_image_bytes(raw: bytes) -> tuple[bytes, str]:
    """Decode, scale long-edge to MAX_IMAGE_EDGE if larger, re-encode JPEG.

    Returns (bytes, mime_type). On decode/encode failure returns the input
    unchanged with a generic mime so the caller still has something to
    forward — context bloat is preferable to a dropped screenshot.
    """
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        log.warning("scale_image_bytes: decode failed on %d bytes", len(raw))
        return raw, "application/octet-stream"
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge > MAX_IMAGE_EDGE:
        scale = MAX_IMAGE_EDGE / long_edge
        new_size = (int(round(w * scale)), int(round(h * scale)))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        log.warning("scale_image_bytes: encode failed")
        return raw, "application/octet-stream"
    return buf.tobytes(), "image/jpeg"


def _content_to_text(content: str | list[ContentBlock]) -> str:
    """Pull the first text out of a tool_result content — a plain string
    or the first `TextBlock` in a multipart list."""
    if isinstance(content, str):
        return content
    for block in content:
        if isinstance(block, TextBlock):
            return block.text
    return ""


def _has_image(content: str | list[ContentBlock]) -> bool:
    """True iff `content` carries at least one `ImageBlock` — i.e. the
    content is multipart and includes a fresh screen capture. Stubbed
    content (plain string) returns False so re-passes are no-ops."""
    if isinstance(content, str):
        return False
    return any(isinstance(b, ImageBlock) for b in content)


def _stub_body(text: str) -> str:
    """Reduce a stubbed view's element listing to bare labels (see the
    module docstring for why the other row fields are dropped). Each
    `[text]` row yields just its label, one per line in listing order;
    icon rows and the header are dropped. Every non-listing line — the
    action result with its verdict, stuck warnings, sequence step lines —
    survives verbatim as decision history, and a blank line sets it off
    from the label block.

    Format coupling: `_LISTING_HEADER` / `_TEXT_ROW_RE` / `_ICON_ROW_RE`
    mirror the shape produced by `physiclaw.core.vision.util.format_elements`;
    sequence step lines (`1 tap ok — …`) lack a quoted label + bbox and are
    kept. Output is stable under re-application — a bare label can't match a
    full-shape row regex unless it is itself a complete row, which OCR never
    produces.
    """
    pre: list[str] = []  # action result + verdict + sequence steps
    labels: list[str] = []  # surviving [text] labels, in listing order
    for line in text.splitlines():
        if line == _LISTING_HEADER:
            continue  # names columns that no longer exist
        m = _TEXT_ROW_RE.match(line)
        if m:
            label = m.group(1)
            if label:  # drop empty-label text rows — no content to keep
                labels.append(label)
            continue
        if _ICON_ROW_RE.match(line):
            continue  # opaque without the image, and label-less
        pre.append(line)
    pre_s = "\n".join(pre).strip()
    lbl_s = "\n".join(labels).strip()
    if pre_s and lbl_s:
        return f"{pre_s}\n\n{lbl_s}"  # blank line sets the label list apart
    return pre_s or lbl_s


def drop_stale_screens(messages: list[Message]) -> None:
    """Stub earlier image-bearing tool_results; keep asst + tool_call
    history intact.

    A "view" is any `ToolResultMessage` carrying an `ImageBlock` —
    `peek`, `screenshot`, and every gesture's fused post-action view.
    All but the newest are replaced with a marker line
    (`"(superseded <tool>)"`) followed by `_stub_body` of their text:
    the action result (verdict included) and the listing's text rows.
    The newest view is untouched — its pixels and full listing are the
    agent's live view.

    Runs in place. Idempotency is carried by the `_has_image` guard:
    once stubbed, content is a plain string (and `is_superseded=True`)
    so further passes skip it.
    """
    view_indices = [
        i
        for i, m in enumerate(messages)
        if isinstance(m, ToolResultMessage) and _has_image(m.content)
    ]
    if len(view_indices) <= 1:
        return

    name_by_id = {
        tc.id: tc.name
        for m in messages
        if isinstance(m, AssistantMessage)
        for tc in m.tool_calls
        if tc.id
    }

    for i in view_indices[:-1]:
        tr = messages[i]
        assert isinstance(tr, ToolResultMessage)  # guaranteed by the filter
        name = name_by_id.get(tr.tool_call_id, "view")
        body = _stub_body(_content_to_text(tr.content))
        head = f"{STUB_PREFIX}{name}){STUB_NOTE}"
        messages[i] = ToolResultMessage(
            tool_call_id=tr.tool_call_id,
            content=f"{head}\n{body}" if body else head,
            is_error=tr.is_error,
            is_superseded=True,
        )
