"""The virtual thread — `debug/thread.json` and its listing renderer.

The file is one scripted conversation: the bubbles so far, plus the
user's `staged` replies waiting for their moment. `physiclaw debug
--task --reply` seeds both; the interceptor appends the agent's own
sends (`record_send`) and reads the thread per peek (`peek_bubbles`),
which is where timing lives: a staged reply enters the thread only when
the newest bubble is the agent's — after an ask, never before one, and
never into the ask's own baseline snapshot (the gate baselines the SEND
result; the release happens on the following peek). One rule covers the
in-session poll, the suspended resume, and multi-ask scripts alike.

`render_listing` draws the thread in the element-listing grammar the
whole system reads, against the REAL channel pack's fingerprint: every
declared anchor of the `thread` page is rendered at its learned position
(else its region band, else spread across the top), so `match_screen`
scores the render like a genuine reading rather than being bypassed.
Bubble geometry carries the semantics `reply.new_incoming` keys on:
incoming bubbles centered left of `INCOMING_MAX_CX`, our own right of
it, newest at the bottom, long texts split into wrapped-line rows.

Block builders mirror the server's wire shapes (`core/server/tools.py`):
a view reply is `[image, listing]` — block 0 must NOT be text, or
`verdict.action_text` would misread the listing as composed action text
— and a gesture reply is `[action text, image, listing]`.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from physiclaw.common import paths
from physiclaw.common.listing import Element, format_elements
from physiclaw.common.logger import write_json_atomic
from physiclaw.common.text import read_text
from physiclaw.conductor.pages import (
    CHANNEL_APP,
    REGIONS,
    THREAD_PAGE,
    PagePrint,
    prints_for_app,
)

log = logging.getLogger(__name__)

THREAD_SCHEMA = 1
USER = "user"
AGENT = "agent"

# Bubble geometry, chosen against `reply.INCOMING_MAX_CX` (0.45): user
# bubbles center at 0.25 (incoming), the agent's at 0.75 (own). Bands
# start below the anchor chrome and stack downward like a real thread
# scrolled to its tail.
_USER_BBOX = (0.06, 0.44)
_AGENT_BBOX = (0.56, 0.94)
_BUBBLE_TOP = 0.20  # first bubble's center y, below the anchor chrome
_BUBBLE_STEP = 0.06
_BUBBLE_BOTTOM = 0.95

# Anchor extents are cosmetic (the matcher verifies centers, not
# widths), so one fixed half-width serves every label.
_ANCHOR_HALF_W = 0.06

# A valid 1×1 JPEG. A view reply's block 0 must be an image
# (`verdict.screen_text` drops block 0 only when it is action text), and
# post-handover the model may re-read the faked turn, so the block has
# to decode. Content is irrelevant — the conductor reads listings only.
TINY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRof"
    "Hh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAAB"
    "AAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AVN//2Q=="
)


@dataclass(frozen=True)
class Bubble:
    sender: str  # USER | AGENT
    text: str


@dataclass
class Thread:
    """The whole scripted conversation: what has been said, and the
    user's replies still waiting for an ask to answer."""

    bubbles: list[Bubble] = field(default_factory=list)
    staged: list[str] = field(default_factory=list)


def thread_path() -> Path:
    return paths.debug_dir() / "thread.json"


def load() -> Thread:
    """The virtual thread, fail-open: missing or unreadable → empty (the
    render then shows a bare thread and the walk behaves accordingly)."""
    p = thread_path()
    if not p.exists():
        return Thread()
    try:
        data = json.loads(read_text(p))
        if data.get("schema") != THREAD_SCHEMA:
            log.warning("debug thread %s: unknown schema — ignoring", p)
            return Thread()
        return Thread(
            bubbles=[
                Bubble(
                    sender=AGENT if b.get("from") == AGENT else USER,
                    text=str(b.get("text") or ""),
                )
                for b in data.get("bubbles") or []
            ],
            staged=[str(s) for s in data.get("staged") or []],
        )
    except Exception:
        log.warning("debug thread %s unreadable — ignoring", p, exc_info=True)
        return Thread()


def save(thread: Thread) -> None:
    paths.debug_dir().mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        thread_path(),
        {
            "schema": THREAD_SCHEMA,
            "bubbles": [{"from": b.sender, "text": b.text} for b in thread.bubbles],
            "staged": list(thread.staged),
        },
    )


def seed(task: str, replies: list[str]) -> None:
    """Reset the conversation to one user message plus the staged
    replies — the `physiclaw debug --task --reply` script."""
    save(Thread(bubbles=[Bubble(sender=USER, text=task)], staged=list(replies)))


def stage(replies: list[str]) -> None:
    """Append staged replies to the running script (`--reply` without
    `--task`) — released by the same ask-then-peek rule."""
    thread = load()
    thread.staged.extend(replies)
    save(thread)


def record_send(message: str) -> list[Bubble]:
    """The agent's own send, appended and persisted — the ask a gate
    quotes stays visible (and excludable) across a suspension exactly
    like a real sent message. Deliberately releases nothing: the send's
    result screen is the gate's baseline snapshot, and a reply released
    into it would be baselined away."""
    thread = load()
    thread.bubbles.append(Bubble(sender=AGENT, text=message))
    save(thread)
    return thread.bubbles


def peek_bubbles() -> list[Bubble]:
    """The thread as a peek sees it — where reply timing lives. A staged
    reply is released iff the newest bubble is the agent's: an ask is
    the latest word, so the reply lands below it, fresh against the
    baseline, on this read. Covers the in-session poll, the suspended
    resume, and replies staged mid-run alike; with no ask outstanding,
    staged replies wait."""
    thread = load()
    if thread.staged and thread.bubbles and thread.bubbles[-1].sender == AGENT:
        released = thread.staged.pop(0)
        thread.bubbles.append(Bubble(sender=USER, text=released))
        save(thread)
        log.info("debug thread: released staged reply %r", released)
    return thread.bubbles


# ---------- rendering ----------


def thread_print() -> PagePrint | None:
    """The channel pack's thread fingerprint, or None (anchorless
    render). Loaded fresh here; per-session callers cache it — the pack
    cannot change mid-session."""
    try:
        prints = prints_for_app(CHANNEL_APP)
    except Exception:
        log.warning("channel pack unreadable — rendering an anchorless thread")
        return None
    return next((p for p in prints if p.decl.name == THREAD_PAGE), None)


def _element(idx: int, label: str, cx: float, cy: float, half_w: float) -> Element:
    return Element(
        id=idx,
        kind="text",
        label=label,
        bbox=(
            max(cx - half_w, 0.0),
            max(cy - 0.015, 0.0),
            min(cx + half_w, 1.0),
            min(cy + 0.015, 1.0),
        ),
        conf=0.99,
    )


def _anchor_elements(pp: PagePrint | None) -> list[Element]:
    """One element per declared thread anchor: learned position first
    (the matcher verifies against it), else the region band's center,
    else spread across the top chrome."""
    if pp is None:
        return []
    out: list[Element] = []
    for i, a in enumerate(pp.decl.anchors):
        la = pp.learned.anchors.get(a.text) if pp.learned else None
        if la is not None:
            cx, cy = la.cx, la.cy
        elif a.region is not None:
            band = REGIONS[a.region]
            cx, cy = (band[0] + band[2]) / 2, (band[1] + band[3]) / 2
        else:
            cx, cy = 0.5, 0.03 + i * 0.045
        out.append(_element(i, a.text, cx, cy, _ANCHOR_HALF_W))
    return out


def render_listing(bubbles: list[Bubble], pp: PagePrint | None) -> str:
    """The virtual thread as one element listing: the channel pack's
    anchors, then the newest bubbles that fit, oldest first — a thread
    scrolled to its tail, the way every real peek reads it."""
    elements = _anchor_elements(pp)
    lines: list[tuple[str, str]] = [
        (b.sender, line.strip())
        for b in bubbles
        for line in b.text.splitlines()
        if line.strip()
    ]
    fit = int((_BUBBLE_BOTTOM - _BUBBLE_TOP) / _BUBBLE_STEP)
    y = _BUBBLE_TOP
    for sender, line in lines[-fit:]:
        left, right = _AGENT_BBOX if sender == AGENT else _USER_BBOX
        cx, half_w = (left + right) / 2, (right - left) / 2
        elements.append(_element(len(elements), line, cx, y, half_w))
        y += _BUBBLE_STEP
    return format_elements(elements)


# ---------- wire shapes ----------


def _image_block() -> dict:
    return {"type": "image", "data": TINY_JPEG_B64, "mimeType": "image/jpeg"}


def view_blocks(listing: str) -> list[dict]:
    """A faked view reply — `[image, listing]`, the peek shape."""
    return [_image_block(), {"type": "text", "text": listing}]


def gesture_blocks(action_text: str, listing: str) -> list[dict]:
    """A faked gesture reply — `[action text, image, listing]`, the
    run_macro shape. The caller composes `action_text` (verdict marker
    included, via `common.verdict.attach`)."""
    return [{"type": "text", "text": action_text}, *view_blocks(listing)]
