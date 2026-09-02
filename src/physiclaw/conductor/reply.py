"""Gate reply reading — the deterministic reading of the ask's reply.

Ruled order: (0) there must be a NEW incoming message at all — no new
bubble, no check; (1) exact word match on the WHOLE message, normalized,
against the words the ASK ITSELF declares (`yes:` / `no:`) — the
conductor holds no word list of its own. Whole-message equality is the
discipline: "ok, but make it two boxes" contains "ok" yet carries a
qualifier that changes the order — anything the declared words do not
cover is the model's to read off the thread (the walk hands over).

New-message detection is positional first, set-difference second:
incoming bubbles sit left of center in every listing-shaped IM layout
(ours sit right); a bubble BELOW our visible ask is newer than the ask
by construction and counts whatever it says (a reply that repeats a
word already on screen must not vanish); only when position cannot
tell — the ask scrolled off, or a sweep above it — does the baseline
(the label set snapshotted when our ask landed) decide what is new.
"""

import unicodedata
from collections.abc import Set as AbstractSet

from physiclaw.common.bbox import center_of
from physiclaw.common.listing import Element

# Incoming bubbles' centers sit left of this; our own sit right, and
# centered system rows (timestamps, at ~0.5) fall OUTSIDE it — they
# would otherwise read as a reply after a suspension.
INCOMING_MAX_CX = 0.45

# A right-side row is read as a wrapped LINE of our own ask only above
# this length: a short fragment could be anything.
_OWN_FRAGMENT_MIN = 5
# The deny sweep skips the just-sent ask's own band, extended by one
# line below its last recognized line: a short wrapped tail ("不用") is
# too short to be recognized as ours and can OCR left of center.
_WRAP_GAP = 0.04


def normalize(text: str) -> str:
    """The comparison space for whole-message matching: NFKC (folds
    full-width forms), casefold, ALL whitespace removed, punctuation and
    symbols stripped from both ends (。！!?～ and friends — a trailing
    exclamation mark must not defeat 好的)."""
    t = "".join(unicodedata.normalize("NFKC", text).casefold().split())
    start, end = 0, len(t)
    while start < end and unicodedata.category(t[start])[0] in ("P", "S"):
        start += 1
    while end > start and unicodedata.category(t[end - 1])[0] in ("P", "S"):
        end -= 1
    return t[start:end]


def classify(text: str, yes: AbstractSet[str], no: AbstractSet[str]) -> str | None:
    """The verdict for ONE message: "confirm", "deny", or None (the
    declared words do not cover it). Whole-message equality only —
    `yes`/`no` are the ask's words already in `normalize` space (the
    parser normalizes them once)."""
    norm = normalize(text)
    if norm in no:
        return "deny"
    if norm in yes:
        return "confirm"
    return None


def classify_all(
    messages: list[str], yes: AbstractSet[str], no: AbstractSet[str]
) -> str | None:
    """The verdict over every new message of one check round. Deny wins
    over confirm (a 不要 anywhere is a stop, whatever else was said);
    any unclassifiable message alongside a confirm also defers —
    partial understanding must not open a money gate."""
    verdicts = [classify(m, yes, no) for m in messages]
    if "deny" in verdicts:
        return "deny"
    if verdicts and all(v == "confirm" for v in verdicts):
        return "confirm"
    return None


def new_incoming(
    rows: tuple[Element, ...],
    baseline: AbstractSet[str],
    own_text: str,
    *,
    after_ask: bool = True,
) -> list[str]:
    """The user's new bubbles since the baseline snapshot, in screen
    order. Incoming = left of center (our own lines sit right, so they
    never enter the candidate set).

    `after_ask` (the default) reads THIS ask's reply: when the ask
    bubble is visible, rows above it are older than the ask (a keyboard
    hides bubbles without disturbing the page's anchors — when it
    dismisses, pre-ask history resurfaces) and rows below it are newer
    by construction, whatever their label — a reply repeating a word
    already on screen is still a reply. Only when the ask has scrolled
    off the top does the baseline decide what is new.

    `after_ask=False` is the deny sweep at a later send's landing: it
    reads bubbles ABOVE the just-sent ask, skipping the ask's own band,
    and the baseline (the previous send's snapshot) decides."""
    ask_top: float | None = None
    ask_bottom: float | None = None
    if own_text:
        for row in rows:
            label = row.label.strip()
            if not label:
                continue
            c = center_of(row.bbox)
            if c is None or c[0] <= INCOMING_MAX_CX:
                continue  # ask lines render as OUR bubbles, right of center
            if own_text in label or (
                len(label) >= _OWN_FRAGMENT_MIN and label in own_text
            ):
                ask_top = c[1] if ask_top is None else min(ask_top, c[1])
                ask_bottom = c[1] if ask_bottom is None else max(ask_bottom, c[1])
    out: list[str] = []
    for row in rows:
        label = row.label.strip()
        if not label:
            continue
        c = center_of(row.bbox)
        if c is None or c[0] > INCOMING_MAX_CX:
            continue
        if ask_top is not None and ask_bottom is not None:
            if after_ask:
                if c[1] <= ask_bottom:
                    continue  # older than the ask — never its reply
                out.append(label)  # below the ask: newer by construction
                continue
            if ask_top <= c[1] <= ask_bottom + _WRAP_GAP:
                continue  # the just-sent ask's own band (tail included)
        if label in baseline:
            continue
        out.append(label)
    return out
