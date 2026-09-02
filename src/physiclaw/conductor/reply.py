"""Gate reply reading — the deterministic reading of the ask's reply.

Ruled order: (0) there must be a NEW incoming message at all — no new
bubble, no check; (1) exact word match on the WHOLE message, normalized,
against the words the ASK ITSELF declares (`yes:` / `no:`) — the
conductor holds no word list of its own. Whole-message equality is the
discipline: "ok, but make it two boxes" contains "ok" yet carries a
qualifier that changes the order — anything the declared words do not
cover is the model's to read off the thread (the walk hands over).

New-message detection is positional + set-difference: incoming bubbles
sit left of center in every listing-shaped IM layout (ours sit right),
and the baseline is the label set snapshotted when our ask landed —
wrapped bubbles split into one row per line, which set difference
handles for free.
"""

import unicodedata
from collections.abc import Set as AbstractSet

from physiclaw.common.bbox import center_of
from physiclaw.common.listing import Element

# Incoming bubbles' centers sit left of this; our own sit right, and
# centered system rows (timestamps, at ~0.5) fall OUTSIDE it — they
# would otherwise read as a reply after a suspension.
INCOMING_MAX_CX = 0.45

# A row is treated as a wrapped LINE of our own ask only above this
# length. Shorter fragments must never be excluded: the ask itself
# quotes the confirm words ("回复 好的 确认…"), so a bare 好的 reply IS a
# substring of it — the very reply the check exists to catch.
_OWN_FRAGMENT_MIN = 5


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
    words: AbstractSet[str] = frozenset(),
) -> list[str]:
    """The user's new bubbles since the baseline snapshot, in screen
    order. Incoming = left of center; new = label not in the baseline
    set; our own ask is excluded (whole, or as a wrapped line — but only
    fragments long enough to BE lines, see _OWN_FRAGMENT_MIN), except a
    verbatim declared reply word (`words`), which is always a reply.

    `after_ask` (the default): when the ask bubble is visible, only rows
    BELOW it count. The baseline is a snapshot of one screen, and a
    keyboard hides bubbles without disturbing the thread page's anchors
    — when it dismisses, pre-ask history reappears as "not in baseline"
    and a stale confirm word from another conversation would open the
    gate. A
    real reply is chronologically after the ask, so it renders below it;
    if the ask has scrolled off the top, everything visible is newer
    than it and the filter stands down. Pass False for a sweep that must
    see bubbles ABOVE a just-sent ask (the re-ask deny sweep)."""
    ask_bottom: float | None = None
    if own_text and after_ask:
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
                ask_bottom = c[1] if ask_bottom is None else max(ask_bottom, c[1])
    out: list[str] = []
    for row in rows:
        label = row.label.strip()
        if not label or label in baseline:
            continue
        c = center_of(row.bbox)
        if c is None or c[0] > INCOMING_MAX_CX:
            continue
        if ask_bottom is not None and c[1] <= ask_bottom:
            # Above the visible ask = older than the ask — stale history
            # resurfacing, never this ask's reply.
            continue
        norm = normalize(label)
        if norm in words:
            # A verbatim declared reply word is ALWAYS a reply, even when
            # the ask
            # quotes it ("reply confirm to pay" + reply "confirm") — the
            # own-line exclusion must never swallow the very words the
            # check exists to catch.
            out.append(label)
            continue
        if own_text and (
            own_text in label or (len(label) >= _OWN_FRAGMENT_MIN and label in own_text)
        ):
            continue
        out.append(label)
    return out
