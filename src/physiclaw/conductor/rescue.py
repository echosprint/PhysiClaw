"""The recovery ladder's policy — one deviation, one next action.

Pure policy over the walk's senses, the `reconcile.plan` shape: given a
check that needed page P and did not get it, name the ONE next rescue
action, or Exhausted and the walk hands over. The walk (`program.py`)
owns the state: it increments the counters this module consults,
synthesizes the action, and re-checks on the action's own result screen
(the reconcile rule: the tap's result view IS the re-read).

The ladder, safest-and-cheapest first:

  locked (the cover or the passcode keypad) → ``unlock_phone``. Read
      before anything else: taps do not reach a sleeping phone, so
      every other rung is inert against it (the overture's lesson).
  occluded (a popup band over a known page) → tap ONE code-owned
      dismissal inside the band: a vocabulary word (关闭 / 我知道了 /
      以后再说 / skip / not now …), the close glyph, or the band's
      top-right icon — deny-listed so nothing money-shaped is ever
      tappable, whatever the popup says.
  anything else (wandered deeper: a live-stream room, a product page,
      an ad landing; or unknown) → ``go_back``, which only pops the
      navigation stack and cannot mutate app state.

Everything is bounded twice — per rung and by one global budget — and
the counters only ever go up (the rebinding-attack lesson: recovery
must not be farmable into a loop). Handover stays the floor: whatever
this ladder cannot restore, the model inherits with the brief.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Mapping

from physiclaw.common import paths
from physiclaw.common.bbox import center_of
from physiclaw.common.listing import Element, Screen
from physiclaw.common.logger import write_json_atomic
from physiclaw.common.text import read_text
from physiclaw.conductor.match import PRICE_RE, Verdict, reads_as_locked
from physiclaw.conductor.micro import (
    CLEAR_OVERLAY,
    DecisionRequest,
    build_request,
)
from physiclaw.conductor.reply import normalize

log = logging.getLogger(__name__)

# Fixed, not authorable — the GATE_MAX_CHECKS discipline. The global
# budget bounds the whole walk's rescue spend; the per-rung caps stop
# any one rung from eating it.
GLOBAL_BUDGET = 6
DISMISS_TRIES = 2
UNLOCK_TRIES = 2
BACK_TRIES = 2

# Rung names — the tries-counter keys `plan` reads and the walk writes
# (also the tail of each `rescue-<rung>` pending kind). Named so a typo
# on either side of the module boundary fails a pin, not a budget.
RUNG_SETTLE = "settle"
RUNG_DISMISS = "dismiss"
RUNG_UNLOCK = "unlock"
RUNG_BACK = "back"
RUNG_MICRO = "micro"
RUNG_RESET = "reset"

# Resume modes — how the walk continues once the target page is
# restored. Validated at State construction (the walklog OUTCOMES
# pattern): a typo'd mode must fail loudly, never silently resume as
# "enter".
MODE_ENTER = "enter"  # re-check the enter, then run the leg
MODE_VERIFY = "verify"  # the macro already ran — verify is satisfied
MODE_RECONCILE = "reconcile"  # re-read the cart
MODES = (MODE_ENTER, MODE_VERIFY, MODE_RECONCILE)

# Dismissal vocabulary, in priority order (first present wins — every
# entry is a pure dismissal, so priority is about determinism, not
# safety). Whole-label equality in `reply.normalize` space: "ok, but…"
# style qualifiers never match, exactly the word-tier discipline.
_DISMISS_RAW = (
    # zh
    "关闭",
    "我知道了",
    "知道了",
    "以后再说",
    "稍后再说",
    "暂不",
    "暂不需要",
    "不用了",
    "跳过",
    "取消",
    # en
    "close",
    "got it",
    "skip",
    "not now",
    "later",
    "maybe later",
    "no thanks",
    "dismiss",
)
DISMISS_WORDS = tuple(normalize(w) for w in _DISMISS_RAW)

# The close glyph as OCR actually reads it — matched on the RAW stripped
# label (normalize strips symbols, which would erase exactly these).
_GLYPHS = frozenset({"×", "✕", "☓", "x", "X"})

# The deny-list: a band row whose label carries any of these is NEVER a
# dismissal candidate — popups are an injection surface, and a "popup"
# whose button says "tap to pay" must be untappable by rescue, full
# stop. Containment on the raw label (buttons read 立即领取, 去支付…),
# plus any ¥ amount via the shared PRICE_RE.
_DENY_TERMS = (
    "支付",
    "付款",
    "购买",
    "开通",
    "领取",
    "订阅",
    "下单",
    "确认",
    "同意",
    "pay",
    "buy",
    "subscribe",
    "confirm",
    "agree",
    "checkout",
)

# The top-right icon fallback: the canonical close-X position when the
# popup's dismissal is an unlabeled icon. Upper slice of the band, right
# edge of the screen.
_ICON_TOP_FRACTION = 1 / 3
_ICON_MIN_CX = 0.75


@dataclass(frozen=True)
class Dismiss:
    """Tap this element to dismiss the overlay; `note` names why it was
    chosen (the journal line's material)."""

    el: Element
    note: str


@dataclass(frozen=True)
class AskDismiss:
    """No code-owned dismissal in sight — spend ONE micro call asking
    which band control is a pure dismissal (`clear_overlay`). The walk
    brokers the request; a pick becomes a Dismiss-shaped tap, and its
    label is LEARNED so the free tier handles this popup next time."""

    request: DecisionRequest


@dataclass(frozen=True)
class Settle:
    """One FREE re-peek before any recovery action is spent: a landing
    judged off a gesture's immediate fused view may still be animating —
    the flakiness rule every mature harness encodes (Maestro's settle
    logic, XCTest's idle wait). Fires only on the transition-shaped
    verdicts (unknown / wrong known page) — a popup or a lock screen is
    a stable read, not an animation."""


@dataclass(frozen=True)
class Unlock:
    pass


@dataclass(frozen=True)
class Back:
    pass


@dataclass(frozen=True)
class Reset:
    """The big hammer, once per WALK: force_quit the app and re-enter
    through the pack's own `open` macro, then locate from the top — the
    resume path a killed session already takes, so everything it can
    re-run is what a next wake would re-run anyway (I2)."""


@dataclass(frozen=True)
class Exhausted:
    """No rung can act — the walk hands over with this reason."""

    reason: str


Step = Settle | Dismiss | AskDismiss | Unlock | Back | Reset | Exhausted


@dataclass
class State:
    """One rescue in flight: the page the frozen cursor requires, how
    the walk resumes once it is restored, and the counters `plan`
    consults. Owned by the walk; cleared only on resume or handover.
    `reason` is the ORIGINAL check failure — the exhausted handover
    reports what actually went wrong, not the last rung's miss.
    `micro_pending` marks an outstanding clear_overlay request (its
    resolve owns the next step); `pending_learn` is the micro pick's
    label, written to the learned file only once the page is RESTORED."""

    target: str  # full `app.page` id the interrupted check needs
    mode: str  # one of MODES — the resume rule
    reason: str = ""
    actions: int = 0
    tries: dict = field(default_factory=dict)
    micro_pending: bool = False
    pending_learn: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown rescue mode {self.mode!r}")

    def note(self) -> str:
        """The RECOVERY rungs used, terse — for journal lines and the
        brief. The settle re-peek is verification hygiene, not a
        recovery action, so it is not reported as one."""
        return ", ".join(
            f"{k}×{v}" for k, v in self.tries.items() if v and k != RUNG_SETTLE
        )


def plan(
    verdict: Verdict,
    screen: Screen,
    tries: Mapping[str, int],
    actions: int,
    learned: tuple[str, ...] = (),
    can_reset: bool = False,
) -> Step:
    """The next rescue action for this reading of the screen. Read-only
    over the walk's counters (the `reconcile.plan` contract). `learned`
    is the app's mined dismissal labels — an extra vocabulary tier.
    `can_reset` says the walk still has its once-per-walk reset (a pack
    `open` macro exists and the hammer is unswung)."""
    if actions >= GLOBAL_BUDGET:
        return Exhausted(f"rescue budget ({GLOBAL_BUDGET} actions) spent")
    if reads_as_locked(screen):
        if tries.get(RUNG_UNLOCK, 0) >= UNLOCK_TRIES:
            return Exhausted(f"phone still locked after {UNLOCK_TRIES} unlock attempts")
        return Unlock()
    if (
        verdict.kind == "occluded"
        and verdict.overlay_band is not None
        and tries.get(RUNG_DISMISS, 0) < DISMISS_TRIES
    ):
        step = find_dismiss(screen, verdict.overlay_band, learned)
        if step is not None:
            return step
        if not tries.get(RUNG_MICRO, 0):
            # The micro tier, once: ask which band control is a pure
            # dismissal. Candidates deny-filtered HERE, so nothing
            # money-shaped ever reaches the model's answer space (I5) —
            # and an all-denied band leaves nothing to ask about.
            req = overlay_request(screen, verdict.overlay_band)
            if req.candidates:
                return AskDismiss(request=req)
        # No safe dismissal in the band — fall through: go_back closes
        # many sheets too, and is state-safe either way.
    if tries.get(RUNG_BACK, 0) >= BACK_TRIES:
        if can_reset:
            return Reset()
        return Exhausted(
            f"screen still reads as {verdict.kind} "
            f"({verdict.page_id or 'no known page'}) after {BACK_TRIES} back attempts"
        )
    if not tries.get(RUNG_SETTLE, 0):
        # Before the first back: the wrong reading may just be a page
        # mid-transition — re-peek once, free, and judge the settled
        # screen instead of spending a real recovery action on it.
        return Settle()
    return Back()


def overlay_request(screen: Screen, band: tuple[float, float]) -> DecisionRequest:
    """The clear_overlay request over the band's SAFE rows only — a
    sub-screen of deny-filtered, in-band labeled rows, through the one
    request assembler."""
    lo, hi = band
    safe = tuple(
        r
        for r in screen.rows
        if r.kind == "text"
        and r.label.strip()
        and _in_band(r, lo, hi)
        and not _denied(r.label)
    )
    sub = Screen(text="", content="", rows=safe)
    return build_request(CLEAR_OVERLAY, "rescue", (), {}, sub)


def find_dismiss(
    screen: Screen, band: tuple[float, float], learned: tuple[str, ...] = ()
) -> Dismiss | None:
    """The one safe dismissal inside the overlay band, or None. Tiers:
    the built-in vocabulary by priority order, then this app's LEARNED
    labels (mined from past clear_overlay picks), then the close glyph,
    then the band's top-right unlabeled icon. Every text candidate
    passes the deny-list first — including learned entries, so a
    poisoned learned file still cannot tap money (I5)."""
    lo, hi = band
    rows = [r for r in screen.rows if _in_band(r, lo, hi)]
    texts = [r for r in rows if r.kind == "text" and not _denied(r.label)]
    by_norm: dict[str, Element] = {}
    for r in texts:
        by_norm.setdefault(normalize(r.label), r)
    for word in DISMISS_WORDS + tuple(normalize(w) for w in learned):
        row = by_norm.get(word)
        if row is not None:
            return Dismiss(el=row, note=f"tapping {row.label.strip()!r}")
    for r in texts:
        if r.label.strip() in _GLYPHS:
            return Dismiss(el=r, note="tapping the close glyph")
    icon_hi = lo + (hi - lo) * _ICON_TOP_FRACTION
    for r in rows:
        if r.kind != "icon":
            continue
        c = center_of(r.bbox)
        if c is not None and c[0] >= _ICON_MIN_CX and c[1] <= icon_hi:
            return Dismiss(el=r, note="tapping the band's top-right icon")
    return None


def _in_band(el: Element, lo: float, hi: float) -> bool:
    c = center_of(el.bbox)
    return c is not None and lo <= c[1] <= hi


def _denied(label: str) -> bool:
    """True when a label may never be tapped by rescue — money-shaped
    text or any deny term, case-folded containment."""
    if PRICE_RE.search(label):
        return True
    folded = label.casefold()
    return any(term in folded for term in _DENY_TERMS)


# ---------- learned dismissals ----------

# Labels per app file — a popup zoo bigger than this is a pack problem.
MAX_LEARNED = 20


def load_dismiss(app: str) -> tuple[str, ...]:
    """This app's mined dismissal labels — fail-open: missing or
    unreadable loads as empty, never takes a walk down."""
    p = paths.learned_dismiss_dir() / f"{app}.json"
    if not p.exists():
        return ()
    try:
        data = json.loads(read_text(p))
        return tuple(str(x) for x in data.get("labels", []))
    except Exception:
        log.warning("learned dismiss %s unreadable; ignoring", p, exc_info=True)
        return ()


def learn_dismiss(app: str, label: str) -> None:
    """Append one label the micro tier successfully dismissed with —
    the micro tier teaching the free tier. Deny-gated at write as a
    belt (use-time gating is the real fence), deduped, oldest dropped
    past MAX_LEARNED. Best-effort like every learned store."""
    label = label.strip()
    if not label or _denied(label):
        return
    labels = [x for x in load_dismiss(app) if x != label] + [label]
    try:
        paths.learned_dismiss_dir().mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            paths.learned_dismiss_dir() / f"{app}.json",
            {"app": app, "labels": labels[-MAX_LEARNED:]},
        )
    except OSError:
        log.warning("learned dismiss write failed", exc_info=True)
