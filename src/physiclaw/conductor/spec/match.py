"""Open-set page matcher — which known page is this screen, if any.

Matches in listing space, never pixels: the camera pipeline is already
rectified to screen 0–1 coords by calibration, and pixel matchers have
catastrophic recall on same-screen detection (fleet-measured: same-page
label sets overlap at Jaccard ≈0.97 with bbox jitter p99 0.007, while
different pages sit at p90 0.13 — listing space separates by an order
of magnitude).

The decision is open-set, boolean, and fail-closed on action: a page
is this page when EVERY declared anchor shows and no forbid term does,
exactly the `require`/`forbid` rule a macro step's checks use; a screen
that satisfies exactly one page is a match, none is `unknown` (the
LLM's jurisdiction, with each page's missing anchor named), two or
more is an ambiguous `unknown`. A calibrated page whose missing anchors
sit under one overlay band reads `occluded` — a keyboard, dialog, or
popup over a known page is an event on the page, not a new page. No
score decides anything: the author chooses few, unmistakable anchors
and gives each its alternate readings. The matcher's most important
output is a reliable `unknown`.

Text comparison builds on the shared base rule (`common.listing.
label_hit`: single-char = whole-label equality, else substring) with
normalization and fuzzy tiers layered on top for OCR noise. This runs
per gesture against ~10 candidate pages × ~40 rows, so the tiers carry
cheap exact prefilters — the fuzzy math only runs where it could
mathematically pass.
"""

import re
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache

from physiclaw.common.bbox import center_of, inside, near
from physiclaw.common.listing import Element, Screen, label_hit
from physiclaw.common.text import fold
from physiclaw.conductor.spec.conventions import LOCKED_ID, PRICE_RE
from physiclaw.conductor.spec.pages import AnchorDecl, LearnedAnchor, PagePrint

# Fuzzy-tier floors (industrial practice: fuzzy text is reliable only
# paired with anchors/structure — UiPath's 0.5–0.6 band). Short anchors
# (SHORT_ANCHOR_MIN..SHORT_ANCHOR_MAX chars — CJK chrome like 搜索框, 购物车)
# instead allow ONE substituted character against any same-length
# window of the label: bigram/ratio tiers mathematically cannot admit a
# single-char confusion in a short string, and single-char visual
# confusion is the dominant CJK OCR error. Below SHORT_ANCHOR_MIN the
# tier is off: one substitution in two characters is half the anchor,
# and on a real order sheet it read a 热销 banner as 销量 — a two-char
# anchor must be read exactly (or pinned to a band and calibrated so
# its OCR variants are mined: capture's `loose` reading lowers the
# floor to LOOSE_ANCHOR_MIN at a spot the exact readings vouch for).
BIGRAM_DICE_MIN = 0.5
EDIT_RATIO_MAX = 0.3
SHORT_ANCHOR_MIN = 3
SHORT_ANCHOR_MAX = 4
LOOSE_ANCHOR_MIN = 2

# Δy scroll voting: bin height in 0–1 screen coords.
DY_BIN = 0.02

# Overlay hypothesis: the band's max height, and how many unexpected
# labels the band must hold to read as an overlay.
OVERLAY_MAX_HEIGHT = 0.6
OVERLAY_MIN_LABELS = 3
# The overlay extends beyond the missing anchors it hides — pad the band
# by this much when counting the overlay's own labels.
OVERLAY_PAD = 0.08

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
    t = fold(text)
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


def label_matches(
    anchor_norm: str,
    label_norm: str,
    variants: tuple[str, ...],
    *,
    loose: bool = False,
) -> bool:
    """The tiered text match. `anchor_norm`/`variants` are pre-normalized.
    `loose` is capture's mining tier: the one-substitution window opens
    from LOOSE_ANCHOR_MIN, so a two-character confusion at a vouched-for
    spot can be learned; a single character stays exact everywhere."""
    if label_norm in variants:
        return True
    if label_hit(anchor_norm, label_norm):
        return True
    if len(anchor_norm) < (LOOSE_ANCHOR_MIN if loose else SHORT_ANCHOR_MIN):
        return False  # too short for a substitution: the base rule is final
    if len(anchor_norm) <= SHORT_ANCHOR_MAX:
        return window_match(anchor_norm, label_norm)
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


def window_match(anchor_norm: str, label_norm: str) -> bool:
    """Short-anchor tier (and capture's confusion mining for anchors too
    short to get it at run time): some same-length window of the label is within
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
    read as cover here, and the page's `locked:` hand then spends an
    `unlock_phone` that does nothing on an unlocked phone — seconds, no
    state change. Cheaper than the alternative it exists to prevent:
    driving a recovery macro's taps into a screen that is not awake to
    receive them.
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
class PageScore:
    """One page read against one screen: which anchors showed, which
    expected-visible ones did not, and the forbid term that showed."""

    print_: PagePrint
    hits: tuple[str, ...]  # anchor texts found (canonical spelling)
    missing: tuple[str, ...]  # expected-visible anchors not found on screen
    dy: float  # chosen scroll offset (0.0 for non-scrollable / unlearned)
    forbid_term: str | None = None

    @property
    def page_id(self) -> str:
        return self.print_.page_id

    @property
    def passes(self) -> bool:
        """The page's whole rule: every anchor shows, no forbid term does."""
        return self.forbid_term is None and bool(self.hits) and not self.missing

    def gap(self) -> str:
        """What kept this page out — the clause after its name on the
        unknown line, and the handover reason for the page a route
        expected."""
        if self.forbid_term is not None:
            return f"forbids {self.forbid_term}"
        if self.missing:
            return f"missing {', '.join(self.missing)}"
        return "shows no anchor"


def candidate_rows(
    anchor: AnchorDecl,
    rows: tuple[Element, ...],
    variants: tuple[str, ...],
    *,
    loose: bool = False,
) -> list[Element]:
    """Rows satisfying one anchor. ANY of its declared readings matches
    (`AnchorDecl.readings` — the canonical text plus authored alts),
    exactly as any mined `variant` does; the anchor still counts once, so
    declaring two spellings of one label never halves the page's score."""
    anchor_norms = [normalize(t) for t in anchor.readings]
    sole = anchor_norms[0] if len(anchor_norms) == 1 else None
    # Hoisted out of the row loop: the band is per anchor, not per row.
    band = list(anchor.within) if anchor.within is not None else None
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
            if not label_matches(sole, label_norm, variants, loose=loose):
                continue
        elif not any(
            label_matches(a, label_norm, variants, loose=loose) for a in anchor_norms
        ):
            continue
        out.append(row)
    return out


def score_page(pp: PagePrint, screen: Screen) -> PageScore:
    """Read one candidate page against one screen."""
    if pp.decl.forbid:
        # Row labels only, one row at a time: a macro result's step log
        # rides `screen.content` too, and a term must never straddle two
        # labels once whitespace is collapsed.
        labels_norm = [normalize(r.label) for r in screen.rows if r.label]
        for term in pp.decl.forbid:
            t = normalize(term)
            if any(t in label for label in labels_norm):
                return PageScore(pp, (), (), 0.0, forbid_term=term)

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

    hits: list[str] = []
    missing: list[str] = []
    for a, la, rows in zip(anchors, las, rows_per):
        # Chrome (a region-pinned anchor) does not scroll with the
        # content: it is checked at its absolute position, exactly as
        # `_vote_dy` leaves it out of the vote.
        adj = 0.0 if a.within is not None else dy
        # Expected-visible: with a scroll offset, an anchor whose learned
        # position moved off-screen is neither expected nor missing.
        if la is not None and pp.decl.scrollable and not (0.0 <= la.cy + adj <= 1.0):
            continue
        if _verified(la, rows, adj):
            hits.append(a.text)
        else:
            missing.append(a.text)
    return PageScore(pp, tuple(hits), tuple(missing), dy)


def _verified(la: LearnedAnchor | None, rows: list[Element], dy: float) -> bool:
    """Some matched row is consistent with the learned geometry (or any
    row when unlearned — text is all we have). `pos_tol` is already
    floored by capture, so it is used as-is."""
    for row in rows:
        c = center_of(row.bbox)
        if c is None:
            continue
        if la is None or near(c, (la.cx, la.cy + dy), tolerance=la.pos_tol):
            return True
    return False


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
        if la is None or a.within is not None:
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


class Reading(StrEnum):
    """The three ways a screen reads. A `StrEnum`, so a verdict still
    compares equal to its spelling in logs and tests, while every
    branch on it can be checked exhaustively."""

    MATCH = "match"  # a confident read of one page
    OCCLUDED = "occluded"  # that page, under an overlay band
    UNKNOWN = "unknown"  # the model's jurisdiction


@dataclass(frozen=True)
class Verdict:
    kind: Reading
    page_id: str | None
    dy: float
    detail: str
    # On an unknown read: every candidate's gap, by page id — the walk
    # names the one page it expected; `describe` joins them all.
    gaps: dict[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        """One log line's worth: the reading, the page, and the matcher's
        own detail — the anchor count on a match, the gap of every page
        on an unknown — what a walk logs after every screen so "why did
        it think it was elsewhere?" is answered on the spot."""
        if self.kind is Reading.UNKNOWN:
            return f"unknown — {self.detail}"
        return f"{self.kind} {self.page_id} ({self.detail})"

    def matches(self, expected_id: str) -> bool:
        """Is this a confident read of exactly `expected_id` (`app.page`)?
        The ONE spelling of that question — pack pages, the channel
        thread, and OS states are all judged by it, so no caller
        re-derives "match AND the right page"."""
        return self.kind is Reading.MATCH and self.page_id == expected_id

    def occludes(self, expected_id: str) -> bool:
        """Is this exactly `expected_id` under an overlay — the page
        itself, with a sheet or popup over it (the reading a page's
        `covered:` hand is declared for)?"""
        return self.kind is Reading.OCCLUDED and self.page_id == expected_id


def match_screen(screen: Screen, candidates: list[PagePrint]) -> Verdict:
    """The open-set decision over one app-scoped candidate set.

    The lock screen is read FIRST and by shape (`reads_as_locked`),
    whatever the candidates: it is the one OS state every walk must
    tell apart from "a screen I don't recognize", because the two
    demand opposite hands — and a real cover prints no text an anchor
    could find, so no `PagePrint` can describe it. The declared
    `ios.locked` page, when it is a candidate, is the sharper belt for
    a device that does print a hint.

    Then every candidate is read whole (`PageScore.passes`): exactly
    one passing page is the match; two or more is ambiguous; none is
    unknown unless exactly one calibrated page has its missing anchors
    under an overlay band, which reads occluded."""
    if reads_as_locked(screen):
        return Verdict(Reading.MATCH, LOCKED_ID, 0.0, "the lock screen")
    if not screen.readable or not candidates:
        return Verdict(Reading.UNKNOWN, None, 0.0, "unreadable or no candidates")

    reads = [score_page(pp, screen) for pp in candidates]
    passing = [r for r in reads if r.passes]
    if len(passing) == 1:
        (best,) = passing
        n = len(best.hits)
        return Verdict(
            Reading.MATCH, best.page_id, best.dy, f"{n} anchor{'s' if n != 1 else ''}"
        )
    if passing:
        names = ", ".join(r.print_.decl.name for r in passing)
        return Verdict(Reading.UNKNOWN, None, 0.0, f"ambiguous: {names} all read whole")
    covered = [r for r in reads if r.hits and _under_overlay(r, screen)]
    if len(covered) == 1:
        (best,) = covered
        return Verdict(
            Reading.OCCLUDED,
            best.page_id,
            best.dy,
            f"missing anchors cluster in one band: {', '.join(best.missing)}",
        )
    return Verdict(
        Reading.UNKNOWN,
        None,
        0.0,
        "; ".join(f"{r.print_.decl.name} {r.gap()}" for r in reads),
        gaps={r.page_id: r.gap() for r in reads},
    )


def _under_overlay(read: PageScore, screen: Screen) -> bool:
    """Overlay test (keyboard/dialog/scroll are events on a page, not new
    pages): the page's MISSING learned anchors all fall inside one
    horizontal band (≤ OVERLAY_MAX_HEIGHT tall) that, padded by
    OVERLAY_PAD, also holds unexpected labels — a dialog or keyboard
    over a known page. Needs learned geometry; declaration-only pages
    can't hypothesize overlays."""
    learned = read.print_.learned
    if learned is None or not read.missing:
        return False
    ys = [learned.anchors[t].cy + read.dy for t in read.missing if t in learned.anchors]
    if not ys:
        return False
    lo, hi = min(ys), max(ys)
    if (hi - lo) > OVERLAY_MAX_HEIGHT:
        return False
    lo, hi = lo - OVERLAY_PAD, hi + OVERLAY_PAD
    hit_norms = {normalize(t) for t in read.hits}
    unexpected = 0
    for row in screen.rows:
        if not row.label:
            continue
        c = center_of(row.bbox)
        if c is None or not (lo <= c[1] <= hi):
            continue
        if normalize(row.label) not in hit_norms:
            unexpected += 1
    return unexpected >= OVERLAY_MIN_LABELS
