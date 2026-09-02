"""The overture — drive to the user's thread, read the intent there.

What plays before the main work. At wake, if any enabled playbook
exists, the conductor takes the wheel: it navigates to the chat content
page deterministically, fires ONE `parse_task` micro-call over the
thread, and either hands the program the baton or goes quiet with the
model already standing on the thread.

The invariant is not new — ``context/AGENT.md`` has always said *"open
the user's chat thread every wake"*. What is new is who executes it.
Left to the model that is one full provider round-trip per gesture
(peek, recognize, home, tap dock, tap contact, peek); here it is
synthesized turns and no provider calls at all. And intent parsing stops
being conditional on the model happening to navigate somewhere: before,
`Activation` could only fire if the latest tool result was already the
thread, so on a wake where the model took another route it never fired.

Three rules earn their place, all learned from recorded sessions:

  - **A verdict gates the confirmation, never the attempt.** The
    recovery action (`channel/open`) begins with `home_screen` and
    recovers from anywhere, so an unrecognized screen is no reason to
    give up — act, then judge the landing. The walk's instinct (unknown
    ⇒ hand over) is wrong here, and inherited it would have made the
    overture quit on the most common verdict.
  - **A blocked or errored call spends no budget.** The dominant real
    failure is a dead phone bridge answering every call in a
    millisecond; retrying against it just burns tokens proving the phone
    is gone.
  - **A sleeping phone is recognized by shape, not by text.** The rule
    above is right about acting on an unknown screen, but a locked one
    is the exception the field found: taps do not reach it at all, so
    the recovery is not merely unhelpful, it is inert. The cover carries
    no anchorable hint on a real iPhone, so `match.reads_as_cover` reads
    its shape (a clock and nothing else) and routes to `unlock_phone`
    before the recovery arm gets it.

Everything is bounded and every failure is a hand-over — the overture
owns no files, writes nothing, and can end no session. It issues five
kinds of action and **never chooses a tap target**: `peek`,
`home_screen` (inside the macro), the user's own rehearsed
`channel/open`, `unlock_phone` (which grounds its own keypad taps in
`core/orchestration/unlock.py`), and the fixed-band history swipe the
`scroll_up` escape mints. The worst case of a misfire is a press of the
home button on the wrong screen.

Without a live `channel/open` macro there is nothing to drive WITH, so
the overture stands by instead: it watches the transcript and reads the
intent the moment the model reaches the thread — exactly what
`Activation` did before, so a channel without `open` loses nothing.
"""

import logging
from dataclasses import replace

from physiclaw.common import gesture_vocab
from physiclaw.common.listing import Screen
from physiclaw.conductor import brief, views
from physiclaw.conductor.channel import Channel
from physiclaw.conductor.match import Verdict, match_screen, reads_as_cover
from physiclaw.conductor.micro import SCROLL_UP, DecisionRequest, MicroOutcome
from physiclaw.conductor.pages import LOCKED_ID, THREAD_ID, PagePrint
from physiclaw.conductor.program import Program
from physiclaw.conductor.turns import SCROLL_BBOX, Turnsmith
from physiclaw.contract.dto import AssistantMessage, Message

log = logging.getLogger(__name__)

# Bounds — fixed, not authorable, the same discipline as GATE_MAX_CHECKS.
# `unlock_phone` races the passcode keypad and its own doctrine says to
# retry once or twice; the open macro is deterministic, so a second miss
# means the world is not what the pack describes.
# Together these ARE the turn budget: the boot mints one opening peek and
# then only unlocks, opens, and scrolls for history, so it can never
# exceed 1 + 2 + 2 + 2 turns (plus the one [note, peek] quit brief when
# it gives up).
UNLOCK_TRIES = 2
OPEN_TRIES = 2
# How many times parse_task's `scroll_up` escape may scroll the thread
# for older messages before the cautious read (no full request in view →
# no activation) stands.
HISTORY_SCROLLS = 2


class Overture:
    """One boot to the thread, constructed per session. Spent once in
    drive mode; in stand-by it may watch the whole session without ever
    being spent, which is what `done` distinguishes."""

    def __init__(
        self,
        *,
        channel: Channel,
        activation,
        prints: list[PagePrint],
    ):
        # Fixed at construction: `Channel` is frozen, so whether there is
        # a hand to navigate with cannot change mid-session.
        self.open_macro = channel.open
        # The baton: a Program when parse_task matched a playbook, else
        # None. Read by the conductor once `done` is set.
        self.program: Program | None = None
        # False while the overture may still act on a later turn — the
        # stand-by mode below leans on it.
        self.done = False
        self._activation = activation
        self._prints = prints
        self._turns = Turnsmith("boot")
        self._unlocks = 0
        self._opens = 0
        self._screen: Screen | None = None
        # The scroll-for-history accumulation: how many scroll_up rounds
        # ran, and the merged thread labels (oldest first) the re-ask
        # reads — None until the first scroll.
        self._scrolls = 0
        self._merged: list[str] | None = None

    def advance(
        self, history: list[Message]
    ) -> "AssistantMessage | DecisionRequest | None":
        """The next synthesized turn, a DecisionRequest for the conductor
        to broker, or None. None means "not mine": check `done` to tell a
        finished overture (hand the baton over, or go quiet) from one
        still standing by for a later turn.

        Never raises — a bug here degrades to a plain model session."""
        try:
            return self._advance(history)
        except Exception:
            log.exception("conductor: overture crashed — handing over to the model")
            self.done = True
            return None

    def resolve(
        self, outcome: MicroOutcome | None
    ) -> "AssistantMessage | DecisionRequest | None":
        """Finish with the parse_task answer — or scroll for the rest of
        it: a `scroll_up` answer in drive mode (bounded) swipes the
        thread up and re-asks over the accumulated listing. Any other
        answer spends the overture: a matched playbook becomes
        `program`, anything else leaves the model the thread it is
        already standing on."""
        if outcome is not None and outcome.out == SCROLL_UP:
            if self.open_macro is not None and self._scrolls < HISTORY_SCROLLS:
                self._scrolls += 1
                if self._merged is None and self._screen is not None:
                    self._merged = _labels(self._screen)
                return self._turns.synth(
                    "scroll",
                    "conductor: the request sits above the fold — scrolling up",
                    gesture_vocab.SWIPE,
                    {"bbox": list(SCROLL_BBOX), "direction": "down"},
                    channel=True,
                )
            # Stand-by (no hand to scroll with) or budget spent: the
            # cautious read — a request we cannot fully see activates
            # nothing, exactly the `not_a_task` disposition.
            outcome = None
        self.done = True
        try:
            self.program = self._activation.build(outcome)
        except Exception:
            log.exception("conductor: activation failed — continuing without it")
            self.program = None
        if self.program is not None:
            log.info(
                "conductor: overture hands over to %s/%s",
                self.program.app,
                self.program.spec.name,
            )
        else:
            log.info("conductor: no playbook covers the thread — the model takes it")
        return None

    # ---- one advance ----

    def _advance(
        self, history: list[Message]
    ) -> "AssistantMessage | DecisionRequest | None":
        if self.done:
            # The quit brief was the boot's last word; its peek result is
            # ordinary history. Quiet from here.
            return None
        if self.open_macro is None:
            return self._stand_by(history)
        if self._turns.pending is None:
            return self._turns.synth(
                "peek", "conductor: looking for the user's thread", "peek", {}
            )
        _pending, result, failed = self._turns.settle(history)
        if failed is not None:
            # No retry budget against a blocked or dead call: the phone
            # bridge dying is the field's most common failure, and every
            # extra round is tokens spent proving the phone is gone.
            return self._quit(failed)
        assert result is not None  # settle: exactly one of result/failed
        return self._on_screen(result)

    def _on_screen(self, result) -> "AssistantMessage | DecisionRequest | None":
        """A landed view → where it sends us. Shared by both modes: the
        boot reads its own action's result, the watcher reads the
        model's."""
        screen = views.screen_of(result)
        self._screen = screen
        return self._route(match_screen(screen, self._prints), screen)

    def _route(
        self, verdict: Verdict, screen: Screen
    ) -> "AssistantMessage | DecisionRequest | None":
        """Where the screen sends us. Only the thread and a locked phone
        are actionable; EVERYTHING else — a known app page, occluded,
        unknown, unreadable — takes the recovery arm, because the
        recovery is unconditional and an unrecognized screen is not a
        reason to quit.

        "Locked" is read two ways, and the SHAPE is the one that fires in
        practice. A declared `ios.locked` page is honoured first, but on
        a real iPhone it never matches: the cover prints no hint text for
        an anchor to find (`match.reads_as_cover`). Without the shape
        read, a sleeping phone scored `unknown` and fell to the recovery
        arm, which spent both `OPEN_TRIES` driving a macro's taps into a
        screen that was not awake to receive them — twice measured at
        ~45s per attempt, then a hand-over, on a wake whose only real
        obstacle was the lock."""
        if verdict.matches(THREAD_ID):
            return self._read_intent()
        if verdict.matches(LOCKED_ID) or reads_as_cover(screen):
            return self._unlock()
        return self._open(verdict)

    def _unlock(self) -> "AssistantMessage | None":
        self._unlocks += 1
        if self._unlocks > UNLOCK_TRIES:
            # unlock_phone's own doctrine for this case: the phone needs
            # passcode 111111 or auto-lock off. The model reads that from
            # the tool description and closes the session properly.
            return self._quit(
                f"phone still locked after {UNLOCK_TRIES} unlock attempts"
            )
        return self._turns.synth(
            "unlock",
            "conductor: phone is locked — unlocking",
            gesture_vocab.UNLOCK_PHONE,
            {},
        )

    def _open(self, verdict: Verdict) -> "AssistantMessage | None":
        assert self.open_macro is not None  # drive mode only
        self._opens += 1
        if self._opens > OPEN_TRIES:
            return self._quit(
                f"could not reach the user's thread in {OPEN_TRIES} attempts "
                f"(screen reads as {verdict.kind}: "
                f"{verdict.page_id or 'no known page'})"
            )
        return self._turns.synth(
            "open",
            f"conductor: opening the user's thread via {self.open_macro}",
            gesture_vocab.RUN_MACRO,
            {"name": self.open_macro},
            channel=True,
        )

    def _read_intent(self) -> "AssistantMessage | DecisionRequest | None":
        """On the thread: the one micro-call this whole walk exists for.
        After a scroll round the request reads the ACCUMULATED listing —
        the newly revealed older labels seamed onto the already-seen
        ones — not just the current viewport."""
        assert self._screen is not None
        req = self._activation.request(self._screen)
        if self._merged is not None:
            self._merged = _merge_labels(_labels(self._screen), self._merged)
            req = replace(req, listing="\n".join(self._merged))
        return req

    def _stand_by(
        self, history: list[Message]
    ) -> "AssistantMessage | DecisionRequest | None":
        """No `channel/open` to navigate with: watch instead of drive.
        Read the intent the moment the MODEL lands on the thread — the
        behaviour `Activation` had before the overture existed, kept so a
        channel missing its `open` macro is no worse off than it was."""
        result = views.last_result(history)
        if result is None or result.is_error:
            return None
        self._screen = views.screen_of(result)
        # Watch only: an unrecognized screen is the model's business, so
        # the drive-mode recovery arm must not fire here.
        if not match_screen(self._screen, self._prints).matches(THREAD_ID):
            return None
        return self._read_intent()

    # ---- synthesis ----

    def _quit(self, reason: str) -> "AssistantMessage | None":
        """The boot's exit (`Program._handover`'s idiom): one final
        synthesized [note, peek] brief turn so the reason reaches the
        model instead of only the log, then spent — the next advance is
        None and the conductor drops the overture. The model takes the
        session from wherever the peek shows the phone to be."""
        self.done = True
        log.warning("conductor: overture handing over to the model — %s", reason)
        return self._turns.synth(
            "brief",
            brief.boot_brief(reason),
            gesture_vocab.PEEK,
            {},
        )


def _labels(screen: Screen) -> list[str]:
    return [r.label for r in screen.rows if r.label.strip()]


def _merge_labels(newer: list[str], previous: list[str]) -> list[str]:
    """Seam two thread readings. Two-step because chrome pins: the
    contact-name header tops BOTH readings, so first strip the common
    PREFIX (rows the scroll did not move — chrome, or an unmoved top),
    then seam the scrollable remainders: after a scroll UP the screen
    shows OLDER content whose bottom overlaps the previous remainder's
    top — the longest suffix-equals-prefix run is the seam. Overlap
    matching (not set dedup) on purpose: a thread legitimately repeats
    labels, and dedup would eat the second of two identical bubbles."""
    p = 0
    while p < min(len(newer), len(previous)) and newer[p] == previous[p]:
        p += 1
    head, newer_rest, prev_rest = newer[:p], newer[p:], previous[p:]
    for k in range(min(len(newer_rest), len(prev_rest)), 0, -1):
        if newer_rest[-k:] == prev_rest[:k]:
            return head + newer_rest + prev_rest[k:]
    return head + newer_rest + prev_rest
