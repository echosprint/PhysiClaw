"""Gate reply reading — the deterministic tiers of the HUMAN_GATE check.

Ruled order: (0) there must be a NEW incoming message at all — no new
bubble, no check, no budget spent; (1) exact word match on the WHOLE
message, normalized — obvious confirmations and rejections never cost a
model call; (2) only what neither tier can classify goes to the
`confirm_reply` micro-call. Whole-message equality is the discipline:
"ok, but make it two boxes" contains "ok" yet carries a qualifier that
changes the order — anything longer than a listed token falls through
to the LLM.

Holds (等等 / wait / hold on) are deliberately in NEITHER list: a hold is
not a yes and not a no, so it rides the re-ask cycle as "unclear".

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
# would otherwise burn an LLM check per fresh timestamp after a
# suspension.
INCOMING_MAX_CX = 0.45

# A row is treated as a wrapped LINE of our own ask only above this
# length. Shorter fragments must never be excluded: the ask itself
# quotes the confirm words ("回复 好的 确认…"), so a bare 好的 reply IS a
# substring of it — the very reply the check exists to catch.
_OWN_FRAGMENT_MIN = 5

_CONFIRM_RAW = (
    # zh
    "好",
    "好的",
    "好啊",
    "好呀",
    "好嘞",
    "行",
    "行的",
    "可以",
    "可以的",
    "没问题",
    "确认",
    "确定",
    "同意",
    "嗯",
    "嗯嗯",
    "对",
    "对的",
    "是",
    "是的",
    "买",
    "买吧",
    "买它",
    "付",
    "付吧",
    "付款",
    "支付",
    "去支付",
    "下单",
    "下单吧",
    "就这个",
    "就它了",
    "继续",
    "冲",
    "走起",
    "ok的",
    "好的ok",
    # en
    "ok",
    "okay",
    "k",
    "kk",
    "yes",
    "yep",
    "yeah",
    "yup",
    "sure",
    "go",
    "go on",
    "go ahead",
    "proceed",
    "confirm",
    "confirmed",
    "do it",
    "buy",
    "buy it",
    "pay",
    "pay it",
    "approve",
    "approved",
    "sounds good",
    "yes please",
)

_DENY_RAW = (
    # zh
    "不",
    "不要",
    "不用",
    "不行",
    "不买",
    "不付",
    "别",
    "别买",
    "别付",
    "算了",
    "取消",
    "停",
    "停止",
    "先不",
    "先不买",
    "先别",
    # en
    "no",
    "nope",
    "don't",
    "dont",
    "do not",
    "stop",
    "cancel",
    "cancel it",
    "abort",
    "no thanks",
    "not now",
    "hold off",
    "don't buy",
    "don't pay",
)


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


CONFIRM_WORDS = frozenset(normalize(w) for w in _CONFIRM_RAW)
DENY_WORDS = frozenset(normalize(w) for w in _DENY_RAW)


def classify(text: str) -> str | None:
    """Tier-1 verdict for ONE message: "confirm", "deny", or None (the
    LLM tier's jurisdiction). Whole-message equality only."""
    norm = normalize(text)
    if norm in DENY_WORDS:
        return "deny"
    if norm in CONFIRM_WORDS:
        return "confirm"
    return None


def classify_all(messages: list[str]) -> str | None:
    """Tier-1 verdict over every new message of one check round. Deny
    wins over confirm (a 不要 anywhere is a stop, whatever else was
    said); any unclassifiable message alongside a confirm also defers to
    the LLM — partial understanding must not open a money gate."""
    verdicts = [classify(m) for m in messages]
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
    order. Incoming = left of center; new = label not in the baseline
    set; our own ask is excluded (whole, or as a wrapped line — but only
    fragments long enough to BE lines, see _OWN_FRAGMENT_MIN).

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
        if norm in CONFIRM_WORDS or norm in DENY_WORDS:
            # A verbatim reply word is ALWAYS a reply, even when the ask
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
