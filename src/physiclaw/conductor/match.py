"""Open-set page matcher — which known page is this screen, if any.

Matches in listing space, never pixels: the camera pipeline is already
rectified to screen 0–1 coords by calibration, and pixel matchers have
catastrophic recall on same-screen detection (fleet-measured: same-page
label sets overlap at Jaccard ≈0.97 with bbox jitter p99 0.007, while
different pages sit at p90 0.13 — listing space separates by an order
of magnitude).

The decision is open-set and fail-closed on action: a page is reported
only above its own threshold AND with a margin over the runner-up;
anything else is `unknown` (the LLM's jurisdiction) or `occluded` (a
known page under an overlay band — keyboard, dialog, popup — which is an
event on the page, not a new page). The matcher's most important output
is a reliable `unknown`.

Text comparison builds on the shared base rule (`common.listing.
label_hit`: single-char = whole-label equality, else substring) with
normalization and fuzzy tiers layered on top for OCR noise. This runs
per gesture against ~10 candidate pages × ~40 rows, so the tiers carry
cheap exact prefilters — the fuzzy math only runs where it could
mathematically pass.
"""

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from physiclaw.common.bbox import center_of, inside, near
from physiclaw.common.listing import Element, Screen, label_hit
from physiclaw.conductor.pages import (
    DEFAULT_MARGIN,
    REGIONS,
    AnchorDecl,
    LearnedAnchor,
    PagePrint,
)

# Fuzzy-tier floors (industrial practice: fuzzy text is reliable only
# paired with anchors/structure — UiPath's 0.5–0.6 band). Short anchors
# (≤ SHORT_ANCHOR_LEN chars — the dominant CJK chrome class: 综合, 搜索,
# 结算) instead allow ONE substituted character against any same-length
# window of the label: bigram/ratio tiers mathematically cannot admit a
# single-char confusion in a 2-char string, and single-char visual
# confusion is the dominant CJK OCR error.
BIGRAM_DICE_MIN = 0.5
EDIT_RATIO_MAX = 0.3
SHORT_ANCHOR_LEN = 4

# Δy scroll voting: bin height in 0–1 screen coords.
DY_BIN = 0.02

# Overlay hypothesis: mid-band score floor, the band's max height, and how
# many unexpected labels the band must hold to read as an overlay.
OCCLUDED_FLOOR = 0.3
OVERLAY_MAX_HEIGHT = 0.6
OVERLAY_MIN_LABELS = 3
# The overlay extends beyond the missing anchors it hides — pad the band
# by this much when counting the overlay's own labels.
OVERLAY_PAD = 0.08

# The one spelling of "a ¥/￥ amount": the matcher class-tokenizes with
# it, the money predicates (program.py) extract its group.
PRICE_RE = re.compile(r"[¥￥]\s*(\d+(?:\.\d+)?)")

# Volatile-content class tokens — "the clock is still a clock".
# `TIME_TOKEN` is named because `reads_as_cover` tests a row against it:
# one spelling, so the tokenizer and its reader cannot drift apart.
TIME_TOKEN = "<TIME>"
_NORM_SUBS: tuple[tuple[re.Pattern, str], ...] = (
    (PRICE_RE, "<PRICE>"),
    (re.compile(r"\b\d{1,2}:\d{2}\b"), TIME_TOKEN),
    (re.compile(r"\d+"), "<NUM>"),
)


@lru_cache(maxsize=8192)
def normalize(text: str) -> str:
    """NFKC (folds full-width forms), casefold, whitespace collapsed,
    volatile spans class-tokenized. Applied identically to anchor texts
    and screen labels — normalize once, compare in one space. Memoized:
    per screen the same ~40 labels are compared against every anchor of
    every candidate page, and anchor texts recur across screens forever."""
    t = unicodedata.normalize("NFKC", text).casefold()
    t = "".join(t.split())
    for pat, token in _NORM_SUBS:
        t = pat.sub(token, t)
    return t


def bigram_dice(a: str, b: str) -> float:
    """Dice coefficient over character bigrams (falls back to unigrams for
    1-char strings). CJK OCR errors are mostly substitutions, which
    bigram overlap tolerates better than exact matching."""
    if not a or not b:
        return 0.0
    ga = {a[i : i + 2] for i in range(len(a) - 1)} or {a}
    gb = {b[i : i + 2] for i in range(len(b) - 1)} or {b}
    return 2 * len(ga & gb) / (len(ga) + len(gb))


def levenshtein(a: str, b: str) -> int:
    """Plain edit distance. Inputs are anchor-sized (≤80 chars), so
    O(len²) is fine on the paths that still reach it."""
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def edit_ratio(a: str, b: str) -> float:
    """Levenshtein distance / max length — 0.0 identical, 1.0 disjoint."""
    if not a or not b:
        return 0.0 if a == b else 1.0
    return levenshtein(a, b) / max(len(a), len(b))


def label_matches(anchor_norm: str, label_norm: str, variants: tuple[str, ...]) -> bool:
    """The tiered text match. `anchor_norm`/`variants` are pre-normalized."""
    if label_norm in variants:
        return True
    if label_hit(anchor_norm, label_norm):
        return True
    if len(anchor_norm) == 1:
        return False  # single char: the whole-label base rule is final
    if len(anchor_norm) <= SHORT_ANCHOR_LEN:
        return _window_match(anchor_norm, label_norm)
    # Length prefilters: both fuzzy tiers are mathematically unable to
    # pass when the lengths are far apart — skip the O(n²)/set work.
    la, lb = len(anchor_norm), len(label_norm)
    ga, gb = max(la - 1, 1), max(lb - 1, 1)
    if (
        2 * min(ga, gb) / (ga + gb) >= BIGRAM_DICE_MIN
        and bigram_dice(anchor_norm, label_norm) >= BIGRAM_DICE_MIN
    ):
        return True
    if abs(la - lb) <= EDIT_RATIO_MAX * max(la, lb):
        return edit_ratio(anchor_norm, label_norm) <= EDIT_RATIO_MAX
    return False


def _window_match(anchor_norm: str, label_norm: str) -> bool:
    """Short-anchor tier: some same-length window of the label is within
    one SUBSTITUTION of the anchor — catches 综合→综台 both standalone and
    buried inside a longer row. Equal-length edit distance ≤1 is exactly
    Hamming ≤1 (an insert/delete changes length), so no DP is needed; the
    `any(ch in ...)` prefilter is a C-level scan that rejects most rows
    (Hamming ≤1 over k≥2 chars implies ≥1 anchor char survives)."""
    k = len(anchor_norm)
    if len(label_norm) < k:
        return levenshtein(anchor_norm, label_norm) <= 1
    if not any(ch in label_norm for ch in anchor_norm):
        return False
    for i in range(len(label_norm) - k + 1):
        bad = 0
        for j in range(k):
            if anchor_norm[j] != label_norm[i + j]:
                bad += 1
                if bad > 1:
                    break
        if bad <= 1:
            return True
    return False


# ---------- the cover, which has no page ----------

# The iOS lock screen cannot be declared as a page: measured on-device
# across every state it can be put in — resting Always-On Display, woken
# and fully lit, and after a swipe — iOS prints NO "swipe up to unlock"
# hint, so a text anchor has nothing to match. (`unlock_phone` never
# identified it by text either: `core/orchestration/unlock.py` wakes,
# swipes, and then OCR-polls for the passcode keypad.)
#
# What it does have is a SHAPE: the hero clock, which is enormous. Its
# WIDTH is the signal, not the row count — a cover showing notifications
# has as many rows as a app screen, so counting them missed exactly the
# frames a busy phone produces.
#
# Measured over 8 on-device readings: the cover's clock spans 0.686-0.753
# of the screen, an app's status-bar clock 0.108-0.122. The floor sits in
# that 5.6x gap, ~2.8x clear of the widest status bar and ~1.7x under the
# narrowest hero — wide enough that OCR clipping a digit cannot cross it.
COVER_CLOCK_MIN_WIDTH = 0.4


def reads_as_cover(screen: Screen) -> bool:
    """True when a screen shows the iOS cover's hero clock — locked,
    asleep, or Always-On Display, a state no `PagePrint` can describe.

    A bare `<TIME>` row alone is NOT enough: every app screen carries a
    status-bar clock that normalizes identically. Width separates them,
    and it holds whether the display is dim or fully lit — the camera's
    own brightness verdict does not (a woken cover raises no dark
    warning), which is why that cannot be the signal.

    Deliberately not fail-closed. A full-screen clock or timer app would
    read as cover here, and the caller then spends an `unlock_phone` that
    does nothing on an unlocked phone — seconds, no state change.
    Cheaper than the alternative it exists to prevent: driving a recovery
    macro's taps into a screen that is not awake to receive them.
    """
    for row in screen.rows:
        if row.kind != "text" or normalize(row.label) != TIME_TOKEN:
            continue
        if row.bbox[2] - row.bbox[0] >= COVER_CLOCK_MIN_WIDTH:
            return True
    return False


def reads_as_locked(screen: Screen) -> bool:
    """The cover OR the passcode keypad — everything `unlock_phone` can
    handle. The keypad state (post-swipe, hero clock gone) prints the
    one hint line the cover never does; its spelling follows the
    device's system locale, so the shape read stays the primary signal
    and the text is the keypad-state belt."""
    return reads_as_cover(screen) or "Enter Passcode" in screen.content


# ---------- per-page scoring ----------


@dataclass(frozen=True)
class AnchorHit:
    anchor: str
    center: tuple[float, float]
    weight: float


@dataclass(frozen=True)
class PageScore:
    print_: PagePrint
    score: float
    hits: tuple[AnchorHit, ...]
    missing: tuple[str, ...]  # expected-visible anchors not found on screen
    dy: float  # chosen scroll offset (0.0 for non-scrollable / unlearned)
    forbidden: bool

    @property
    def page_id(self) -> str:
        return self.print_.page_id


def candidate_rows(
    anchor: AnchorDecl, rows: tuple[Element, ...], variants: tuple[str, ...]
) -> list[Element]:
    """Rows satisfying one anchor. ANY of its declared readings matches
    (`AnchorDecl.readings` — the canonical text plus authored alts),
    exactly as any mined `variant` does; the anchor still counts once, so
    declaring two spellings of one label never halves the page's score."""
    anchor_norms = [normalize(t) for t in anchor.readings]
    sole = anchor_norms[0] if len(anchor_norms) == 1 else None
    # Hoisted out of the row loop: the band is per anchor, not per row.
    band = list(REGIONS[anchor.region]) if anchor.region is not None else None
    out = []
    for row in rows:
        if not row.label:
            continue
        if band is not None:
            # Geometry BEFORE text: a bare centre-in-band test is far
            # cheaper than normalize + the fuzzy tiers, and both filters
            # are ANDed — so rejecting here skips the expensive half.
            c = center_of(row.bbox)
            if c is None or not inside(c, band, margin=0.0):
                continue
        label_norm = normalize(row.label)
        if sole is not None:
            # The overwhelmingly common shape (one declared reading) —
            # kept off the generator to stay a plain call per row.
            if not label_matches(sole, label_norm, variants):
                continue
        elif not any(label_matches(a, label_norm, variants) for a in anchor_norms):
            continue
        out.append(row)
    return out


def score_page(pp: PagePrint, screen: Screen) -> PageScore:
    """Score one candidate page against one screen reading."""
    if pp.decl.forbid:
        content_norm = normalize(screen.content)
        for term in pp.decl.forbid:
            if normalize(term) in content_norm:
                return PageScore(pp, 0.0, (), (), 0.0, forbidden=True)

    learned = pp.learned
    anchors = pp.decl.anchors
    las = [learned.anchors.get(a.text) if learned else None for a in anchors]
    rows_per = [
        candidate_rows(a, screen.rows, la.variants if la else ())
        for a, la in zip(anchors, las)
    ]

    dy = 0.0
    if learned is not None and pp.decl.scrollable:
        dy = _vote_dy(anchors, las, rows_per)

    hits: list[AnchorHit] = []
    missing: list[str] = []
    total_weight = 0.0
    hit_weight = 0.0
    for a, la, rows in zip(anchors, las, rows_per):
        # Expected-visible: with a scroll offset, an anchor whose learned
        # position moved off-screen is neither expected nor missing.
        if la is not None and pp.decl.scrollable and not (0.0 <= la.cy + dy <= 1.0):
            continue
        weight = la.weight if la else 1.0
        total_weight += weight
        center = _verify_position(la, rows, dy)
        if center is not None:
            hit_weight += weight
            hits.append(AnchorHit(anchor=a.text, center=center, weight=weight))
        else:
            missing.append(a.text)
    score = hit_weight / total_weight if total_weight else 0.0
    return PageScore(pp, score, tuple(hits), tuple(missing), dy, forbidden=False)


def _verify_position(
    la: LearnedAnchor | None,
    rows: list[Element],
    dy: float,
) -> tuple[float, float] | None:
    """The first matched row consistent with the learned geometry (or any
    row when unlearned — text is all we have). Returns its center.
    `pos_tol` is already floored by capture, so it is used as-is."""
    for row in rows:
        c = center_of(row.bbox)
        if c is None:
            continue
        if la is None or near(c, (la.cx, la.cy + dy), tolerance=la.pos_tol):
            return c
    return None


def _vote_dy(
    anchors: tuple[AnchorDecl, ...],
    las: list[LearnedAnchor | None],
    rows_per: list[list[Element]],
) -> float:
    """Hough-style vote: each (matched row, learned position) pair proposes
    dy = observed_cy − learned_cy; the fullest DY_BIN wins. Anchors with a
    region hint are chrome — they vote for 0 implicitly by their absolute
    check and are excluded here."""
    votes: dict[int, list[float]] = {}
    for a, la, rows in zip(anchors, las, rows_per):
        if la is None or a.region is not None:
            continue
        for row in rows:
            c = center_of(row.bbox)
            if c is None:
                continue
            d = c[1] - la.cy
            votes.setdefault(round(d / DY_BIN), []).append(d)
    if not votes:
        return 0.0
    best = max(votes.values(), key=len)
    return sorted(best)[len(best) // 2]


# ---------- open-set decision ----------


@dataclass(frozen=True)
class Verdict:
    kind: str  # "match" | "occluded" | "unknown"
    page_id: str | None
    score: float
    runner_up: float
    dy: float
    detail: str
    # The padded (lo, hi) y-band the overlay hypothesis found — set only
    # on "occluded", where it is what the rescue ladder's dismissal tier
    # restricts its candidates to (`rescue.find_dismiss`).
    overlay_band: tuple[float, float] | None = None

    def matches(self, expected_id: str) -> bool:
        """Is this a confident read of exactly `expected_id` (`app.page`)?
        The ONE spelling of that question — pack pages, the channel
        thread, and OS states are all judged by it, so no caller
        re-derives "match AND the right page"."""
        return self.kind == "match" and self.page_id == expected_id


def match_screen(screen: Screen, candidates: list[PagePrint]) -> Verdict:
    """The three-way open-set decision over one app-scoped candidate set.

    Every candidate is scored — the margin rule needs the runner-up, so
    there is no early exit that preserves its semantics."""
    if not screen.readable or not candidates:
        return Verdict("unknown", None, 0.0, 0.0, 0.0, "unreadable or no candidates")

    scores = sorted(
        (score_page(pp, screen) for pp in candidates),
        key=lambda s: s.score,
        reverse=True,
    )
    best = scores[0]
    runner = scores[1].score if len(scores) > 1 else 0.0

    if best.forbidden or best.score <= 0.0:
        return Verdict("unknown", None, 0.0, runner, 0.0, "no candidate scored")

    if best.score >= best.print_.threshold and (best.score - runner) >= DEFAULT_MARGIN:
        return Verdict(
            "match",
            best.page_id,
            best.score,
            runner,
            best.dy,
            f"{len(best.hits)}/{len(best.hits) + len(best.missing)} anchors",
        )

    band = _overlay_band(best, screen) if best.score >= OCCLUDED_FLOOR else None
    if band is not None:
        return Verdict(
            "occluded",
            best.page_id,
            best.score,
            runner,
            best.dy,
            f"missing anchors cluster in one band: {', '.join(best.missing)}",
            overlay_band=band,
        )

    return Verdict(
        "unknown",
        None,
        best.score,
        runner,
        best.dy,
        f"best {best.page_id} below threshold {best.print_.threshold:.2f} or margin",
    )


def _overlay_band(best: PageScore, screen: Screen) -> tuple[float, float] | None:
    """Overlay test (keyboard/dialog/scroll are events on a page, not new
    pages): the page's MISSING learned anchors all fall inside one
    horizontal band (≤ OVERLAY_MAX_HEIGHT tall) that also holds unexpected
    labels — a dialog or keyboard over a known page. Needs learned
    geometry; declaration-only pages can't hypothesize overlays. Returns
    the padded (lo, hi) band — the overlay's own extent, which the
    rescue dismissal tier scopes to — or None."""
    learned = best.print_.learned
    if learned is None or not best.missing:
        return None
    ys = [learned.anchors[t].cy + best.dy for t in best.missing if t in learned.anchors]
    if not ys:
        return None
    lo, hi = min(ys), max(ys)
    if (hi - lo) > OVERLAY_MAX_HEIGHT:
        return None
    lo, hi = lo - OVERLAY_PAD, hi + OVERLAY_PAD
    hit_norms = {normalize(h.anchor) for h in best.hits}
    unexpected = 0
    for row in screen.rows:
        if not row.label:
            continue
        c = center_of(row.bbox)
        if c is None or not (lo <= c[1] <= hi):
            continue
        if normalize(row.label) not in hit_norms:
            unexpected += 1
    return (lo, hi) if unexpected >= OVERLAY_MIN_LABELS else None
