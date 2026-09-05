"""One playbook mid-walk — the walk's core.

A `Program` is one playbook being executed. The conductor asks it for
each turn; it answers with a synthesized ``[note, one-other]`` turn, a
``DecisionRequest`` to broker through the micro-caller (fed back via
``resolve()``), or ``None`` — spent for good: the transcript of its
turns is the model's handoff, and every handover or completion mints
ONE last ``[note, peek]`` brief turn first (`brief.py`).

This file is the walk alone: the cursor and phase, the one action in
flight (`turns.py`), the page verdict every step judges against, the
declared recovery toward a page (`recover.py`), the money state an ask
binds and a payment move spends, the terminal moments, and the record
they write (`record.py`). What each STEP does is its executor's
(`step.py` is the contract), one per route entry kind: `step_do`,
`step_agent`, `step_ask`, `step_tell`, `step_activate`.

What the playbook declares is what runs: the walk opens with one peek,
starts at the route's first unsettled node (never below a resumed
suspension's cursor), runs a page's own `recover:` hand on a failed
check or hands over, and hands over on every blocked or errored call.
Money never recovers.
"""

import logging
from collections import Counter
from collections.abc import Callable
from enum import StrEnum

from physiclaw.common import gesture_vocab
from physiclaw.common.listing import Screen
from physiclaw.common.logger import write_json_atomic
from physiclaw.conductor.spec.channel import Channel
from physiclaw.conductor.spec.conventions import LOCKED_ID, owned_by, page_id, page_name
from physiclaw.conductor.spec.match import Reading, Verdict, match_screen
from physiclaw.conductor.spec.model import (
    READING_COVERED,
    READING_ELSEWHERE,
    READING_LOCKED,
    ActivateNode,
    AgentNode,
    AskNode,
    Checked,
    DoNode,
    Node,
    Playbook,
    PlaybookError,
    TellNode,
)
from physiclaw.conductor.spec.pack import qualified_macro
from physiclaw.conductor.spec.pages import Landmark, PagePrint
from physiclaw.conductor.walk import brief, recover, views
from physiclaw.conductor.walk.gate import Gate
from physiclaw.conductor.walk.micro import MicroOutcome
from physiclaw.conductor.walk.record import Record
from physiclaw.conductor.walk.step import Activator, Paused, Step, Turn
from physiclaw.conductor.walk.step_activate import ActivateStep
from physiclaw.conductor.walk.step_agent import AgentStep
from physiclaw.conductor.walk.step_ask import AskStep
from physiclaw.conductor.walk.step_do import DoStep
from physiclaw.conductor.walk.step_tell import TellStep
from physiclaw.conductor.walk.suspension import (
    SUSPENDED_SCHEMA,
    clear_suspended,
    suspended_path,
)
from physiclaw.conductor.walk.turns import Turnsmith
from physiclaw.conductor.walk.walklog import Outcome
from physiclaw.contract.dto import AssistantMessage, Message
from physiclaw.contract.plugin import EventSink
from physiclaw.macros.model import Macro

log = logging.getLogger(__name__)

# runtime.sentinel.WAIT, spelled literally: the conductor may not import
# engine runtimes; a test pins the two equal.
SUSPEND_STATUS = "WAIT"

# The one recovery action's pending kind — the declared hand's landing.
KIND_RECOVER = "recover-hand"
# A resumed walk's one infrastructure action: the unlock before its
# opening reading.
KIND_UNLOCK = "resume-unlock"


class Phase(StrEnum):
    """Where a walk stands in its life. One value replaces the latches
    it grew up with (started / done / paused): every transition is a
    named event, and the terminal rules read off one field."""

    FRESH = "fresh"  # constructed, never advanced
    OPENING = "opening"  # the first reading: unlock, gate resume, the opening peek
    LIVE = "live"  # a step at the cursor: stepping, recovering, gated
    PAUSED = "paused"  # a stepping run ended with the cursor moved on
    DONE = "done"  # the last word is minted: brief, or crash — quiet from here


# The executor for each route entry kind (`Node` is a closed union).
_STEP_FOR: dict[type, type[Step]] = {
    DoNode: DoStep,
    AgentNode: AgentStep,
    AskNode: AskStep,
    TellNode: TellStep,
    ActivateNode: ActivateStep,
}


# How a checked node names itself in a handover reason.
_CHECKED_KIND: dict[type, str] = {
    DoNode: "move",
    AgentNode: "agent",
    ActivateNode: "select",
}


class Program:
    """One playbook mid-walk. Constructed per session (the conductor
    plugin's wake setup builds it), so the cursor state lives for
    exactly one attempt; persistence across wakes is the suspension file
    (`suspend`), and only that."""

    def __init__(
        self,
        *,
        spec: Playbook,
        values: dict[str, str],
        pack_macros: dict[str, Macro],
        prints: list[PagePrint],
        channel: Channel | None = None,
        suspended: dict | None = None,
        position: dict | None = None,
        landmarks: "dict[str, Landmark] | None" = None,
        dry: bool = False,
        activation: "Activator | None" = None,
        events: "EventSink | None" = None,
    ) -> None:
        self.app = spec.app
        # A dry walk (`replay.py`) leaves no trace: no runs.jsonl line,
        # no daily-log entry, no suspension file. Everything else runs
        # exactly as live.
        self.spec = spec
        self.values = values
        # The `inputs.<name>` half of the ref-resolution dict, built once:
        # `values` never mutates after construction.
        self._input_vals = {f"inputs.{k}": v for k, v in values.items()}
        # The program's own qualified dispatch contribution — merged into
        # the wake registry by session_setup.
        self.pack_macros = pack_macros
        self.prints = prints
        # The user-channel infrastructure (constructor-injected via
        # build.build_program). None degrades to hand-over at the first
        # ask.
        self.channel = channel
        # The pack's declared fixed spots — recover hands tap them, agent
        # episodes are granted them by name.
        self.landmarks: dict[str, Landmark] = landmarks or {}
        # The boot's activation (menu, parse_task, build) — what its
        # `select` step runs; None on every other walk — and the
        # baton that step hands on: the program the conductor drives
        # once this walk goes quiet.
        self.activation = activation
        self.baton: "Program | None" = None
        self.idx = 0
        # Turn minting + the one action in flight (`turns.py`). The scope
        # is the playbook ref: two walks in one session (the boot, then
        # the program it activates) mint under different names, so a
        # call id can never find the other walk's stale result.
        self.turns = Turnsmith(f"{spec.app}-{spec.name}")
        self.gate = Gate()
        # Recorded agent outputs (`{node.field}` refs read them).
        self.outputs: dict[str, str] = {}
        # The screen/verdict the current step works from — every path
        # observes one before acting.
        self.screen: Screen | None = None
        self.verdict: Verdict | None = None
        # The amount a payment move actually fired with (consent is
        # consumed at fire, so this is the only place it survives to the
        # completed history line), and whether its daily-log line landed.
        self.paid: float | None = None
        self._paid_logged = False
        # The step executor at the cursor (a resume pre-step rides the
        # same slot before the walk proper opens).
        self._step: Step | None = None
        # Recovery (`recover.py`): the hand in flight and the actions
        # spent per target page (their sum is the walk-wide count).
        self._recovery: recover.State | None = None
        self._page_recoveries: Counter[str] = Counter()
        # A restored walk's stored cursor and whether it is a wake's
        # suspension — `_restore` says what each buys.
        self._resume_at = 0
        self._from_suspension = False
        self._unlocked = False  # the resume unlock, once
        # Telemetry (`walklog`): decision outcomes brokered to this walk.
        self._micros = 0
        # The record (`record.py`): the runs.jsonl line, the session's
        # `walk` event, the daily-log entries a terminal moment writes,
        # and the outcome latch.
        self.record = Record(self.app, spec.name, dry, events)
        self.phase = Phase.FRESH
        # The journal line the next synthesized note carries
        # (record-don't-replay: the transcript carries what happened).
        self._journal: str | None = None
        # Stepping: a rehearsal that wants ONE node sets `step_one`; the
        # walk runs the first node it opens and answers `Paused` the
        # moment the cursor stands anywhere else — forward when the node
        # settles, backward when a recover hand restarted the route —
        # WITHOUT opening the next step, so nothing of it (a payment's
        # consent, an ask's numbers) is spent. (The opening peek may
        # move the cursor past a settled prefix first; the node opened
        # after it is the one.)
        self.step_one = False
        self._stepped: int | None = None
        if suspended is not None and position is not None:
            raise ValueError("a walk resumes a suspension OR a position, not both")
        restored = suspended if suspended is not None else position
        if restored is not None:
            # A restored walk is ALSO whole at construction: the
            # projection overlays the fresh state right here, so no
            # caller ever patches a program up afterwards.
            self._restore(restored, resumed=suspended is not None)
        if suspended is not None:
            log.info(
                "conductor: resuming suspended %s/%s at node %d (%s)",
                self.app,
                spec.name,
                self.idx + 1,
                "awaiting reply" if self.gate.awaiting else "walk",
            )

    @property
    def node(self) -> "Node | None":
        """The node at the cursor — None once the walk is past the last."""
        nodes = self.spec.nodes
        return nodes[self.idx] if self.idx < len(nodes) else None

    @property
    def _recoveries(self) -> int:
        return sum(self._page_recoveries.values())

    # ---- suspension ----

    def state(self) -> dict:
        """The walk's position as the suspension projection — what a
        later wake, or a stepping rehearsal's next invocation, rebuilds
        the walk from (`suspended=` or `position=` at
        construction): walk state here,
        gate state via `Gate.to_suspended` — each field list lives
        beside its fields."""
        return {
            "schema": SUSPENDED_SCHEMA,
            "app": self.app,
            "playbook": self.spec.name,
            "idx": self.idx,
            "values": self.values,
            "outputs": self.outputs,
            **self.gate.to_suspended(),
        }

    def _restore(self, data: dict, *, resumed: bool) -> None:
        """Overlay one projection (`state()`'s shape). The stored cursor
        is where the walk opens either way (the next node's own checks
        judge whether the world still fits). `resumed` marks a WAKE's
        suspension, which buys two more things: the cursor is also the
        floor no recovery restarts below (the nodes before it ran on an
        earlier wake — an ask answered — and re-running them per wake
        would loop across wakes with a fresh recovery budget each time),
        and the opening read may unlock a locked phone once. A stepping
        position keeps the fresh walk's rules: a recover hand that does
        not restore the page walks again from the top — exactly what
        the author steps to see — and the tool's own preamble wakes the
        phone."""
        idx = int(data["idx"])
        if not (0 <= idx <= len(self.spec.nodes)):
            # The spec changed under the suspension (edited shorter) — a
            # stale cursor must not fake a completion. Raising drops the
            # suspension (load_suspended is fail-open).
            raise PlaybookError(f"suspended idx {idx} is outside the playbook")
        self.idx = idx
        self._resume_at = idx
        self._from_suspension = resumed
        self.outputs = {str(k): str(v) for k, v in (data.get("outputs") or {}).items()}
        self.gate = Gate.from_suspended(data)

    @property
    def _floor(self) -> int:
        """The cursor no recovery restarts below (`_restore`)."""
        return self._resume_at if self._from_suspension else 0

    def suspend(self) -> AssistantMessage:
        """Write the suspended state (`state()`, the cursor where it
        stands) and close the session WAIT. No job is synthesized: a
        WAIT without a session-created job auto-schedules the follow-up
        alarm (`contract.drive`) — the file, not the job, is what
        resumes the walk on ANY next wake. The one caller is an ask out
        of patience, which already holds the gate open."""
        assert self.gate.awaiting, "only an ask awaiting its reply suspends"
        if not self.dry:
            write_json_atomic(suspended_path(), self.state())
        recap = f"waiting for the user's reply on {self.app}/{self.spec.name}"
        self._record_run(Outcome.SUSPENDED, recap)
        # The close-routine's log line, harness-written: the conductor
        # synthesizes end_session directly, so the model never runs this
        # wake — with no per-step logs either, the suspension would be
        # invisible to the next wake's memory window.
        self.log_day(
            f"conductor: {self.app}/{self.spec.name} suspended — {recap}; "
            "any wake resumes it"
        )
        return self.synth(
            "suspend",
            f"conductor: suspending — {recap}",
            "end_session",
            {"status": SUSPEND_STATUS, "recap": recap},
        )

    # ---- the conductor's two calls ----

    def advance(self, history: list[Message]) -> Turn:
        """The next synthesized turn; a DecisionRequest for the conductor
        to broker (feed the outcome back via ``resolve``); `Paused` when
        a stepping run is over; or None — spent, hand over to the model.
        A spec-level failure (a ref with no value) hands over with its
        reason. A program bug RAISES: the one caller that must keep a
        session alive (`Conductor._drive`) catches it and calls `crash`;
        every tool and test sees the traceback."""
        if self.phase is Phase.FRESH:
            self.phase = Phase.OPENING
        return self._guarded(lambda: self._advance(history))

    def resolve(self, outcome: MicroOutcome | None) -> Turn:
        """Continue the walk with a micro-call's outcome (None = the call
        failed or was under-confident — hand over). Same return contract
        and same guarantees as ``advance``."""
        self._micros += 1
        return self._guarded(lambda: self._resolve(outcome))

    def _guarded(self, run: "Callable[[], Turn]") -> Turn:
        try:
            return run()
        except PlaybookError as e:
            return self.handover(str(e))

    def crash(self) -> None:
        """The walk died of a program bug (the conductor caught it):
        quiet from here whatever else fails, and recorded as crashed
        when the record can be written. The transcript so far is the
        model's hand-off."""
        self.phase = Phase.DONE
        try:
            self._record_run(Outcome.CRASHED, "program crashed")
        except Exception:
            log.exception("conductor: crash record failed — ignored")

    def _advance(self, history: list[Message]) -> Turn:
        if self.phase is Phase.DONE:
            # The brief turn was the walk's last word; its peek result is
            # ordinary history. Quiet from here — the conductor drops us.
            return None
        if self.phase is Phase.PAUSED:
            return Paused()
        if self.turns.pending is None:
            return self._opening()
        pending, result, failed = self.turns.settle(history)
        kind = pending.kind
        if kind == "suspend":
            # The suspension file is already written; whether the
            # end_session was blocked, its result never arrived, or the
            # session simply ran on, a dead walk must not resurrect on
            # the next wake.
            if not self.dry:
                clear_suspended()
            return self.handover(
                f"{failed or 'suspend end_session returned'} — suspension dropped"
            )
        if failed is not None:
            # Nothing retries in the background: what the playbook did
            # not declare, the model decides. A fired payment is logged
            # first — money may have moved even though the call failed.
            self.log_purchase()
            return self.handover(failed)
        assert result is not None  # settle: a result or a failure
        # One reading, one verdict: channel-facing actions (declared at
        # the synth site) match against the thread; everything else
        # against the pack's own pages.
        self.screen = views.screen_of(result)
        self.verdict = match_screen(
            self.screen,
            self.channel.prints if pending.channel and self.channel else self.prints,
        )
        log.info(
            "conductor: %s/%s read after %s — %s",
            self.app,
            self.spec.name,
            kind,
            self.verdict.describe(),
        )
        if (
            self.phase is Phase.OPENING
            and self._from_suspension
            and not self._unlocked
            and self.verdict.matches(LOCKED_ID)
        ):
            # The resumed walk's first reading is the cover: wake the
            # phone once, then open again exactly as before.
            self._unlocked = True
            return self.synth(
                KIND_UNLOCK,
                "conductor: phone is locked on resume — unlocking",
                gesture_vocab.UNLOCK_PHONE,
                {},
            )
        if kind == KIND_UNLOCK:
            return self._opening()
        if kind == "peek":
            # Past the settled pure-text prefix, never below a restored
            # walk's stored cursor (`_restore`).
            self.phase = Phase.LIVE
            self.idx = max(self.spec.first_unsettled(self.outputs), self._resume_at)
            return self.next()
        if kind == KIND_RECOVER:
            return self._recover_landed()
        if self._step is not None and kind in self._step.kinds:
            return self._step.landed(kind)
        # A typo'd kind at a synth site must fail loudly, never silently
        # walk the next node.
        return self.handover(f"unknown pending action kind {kind!r}")

    def _resolve(self, outcome: MicroOutcome | None) -> Turn:
        if self._step is None:
            return self.handover("a decision arrived with no step in flight")
        return self._step.resolve(outcome)

    def _opening(self) -> Turn:
        """The walk's first turn (fresh, or after a resume): the ask's
        reply check when suspended at a gate, else the plain opening
        peek."""
        if self.gate.awaiting:
            # Suspended-at-gate resume: the ask was sent before the
            # suspension — the ask step picks up at its reply check.
            node = self.node
            if not isinstance(node, AskNode):
                return self.handover(
                    "suspended awaiting a reply, but no ask at the cursor"
                )
            self._step = AskStep(self, node)
            return self._step.open()
        # Observe before acting: the first page check needs a screen.
        return self.peek()

    # ---- the cursor ----

    def next(self) -> Turn:
        """Walk the node at the cursor: open its step executor, or finish.
        Works from the stored screen/verdict — every path here observed
        one first."""
        if self.verdict is None:
            return self.handover("no screen observed yet")
        nodes = self.spec.nodes
        if self.idx >= len(nodes):
            log.info(
                "conductor: playbook %s/%s complete — handing over",
                self.app,
                self.spec.name,
            )
            self._end(Outcome.COMPLETED)
            return self.synth(
                "brief",
                brief.completion_brief(self.app, self.spec.name, len(nodes)),
                gesture_vocab.PEEK,
                {},
            )
        if self.step_one:
            if self._stepped is None:
                self._stepped = self.idx
            elif self.idx != self._stepped:
                # The cursor left the one node this run was for. Pause
                # here, before the next step opens and spends anything.
                log.info(
                    "conductor: stepping pause — cursor moved from node %d to %d",
                    self._stepped + 1,
                    self.idx + 1,
                )
                self.phase = Phase.PAUSED
                return Paused()
        node = nodes[self.idx]
        self._step = _STEP_FOR[type(node)](self, node)
        return self._step.open()

    def advance_cursor(self) -> Turn:
        """The step at the cursor is done — walk the next node. The ONE
        way the cursor moves forward."""
        self.idx += 1
        self._step = None
        return self.next()

    def _node_id(self) -> str | None:
        node = self.node
        return node.id if node is not None else None

    # ---- what the steps read and call ----

    def ref_values(self) -> dict[str, str]:
        """Ref-resolution values, keyed by the dotted ref spellings: the
        walk's inputs under `inputs.<name>` and every recorded agent
        output under `node.field`. The roots can never collide: the
        parser reserves `inputs` as a move id."""
        return {**self._input_vals, **self.outputs}

    def mismatch(self, verdict: Verdict, expected_id: str) -> str | None:
        """None when the verdict is a match on the full `expected_id`
        (`app.page`); else a short reason — pack pages and the channel
        thread judged by one spelling."""
        if verdict.matches(expected_id):
            return None
        return f"screen reads as {verdict.describe()}"

    def money_page_block(self, what: str) -> str | None:
        """A payment move fires only off a VERIFIED own-pack page: the
        ask left the phone on the IM thread, and an unverified screen
        could satisfy the predicates with the conductor's own ask
        bubble. None when the current verdict is such a page; else the
        handover reason. (The ask itself reads its total off the exact
        waypoint before it — `AskNode.enter`.)"""
        v = self.verdict
        if (
            v is not None
            and v.kind is Reading.MATCH
            and owned_by(v.page_id or "", self.app)
        ):
            return None
        return (
            f"{what}: current screen is not a verified {self.app} page — "
            "money never reads or fires blind"
        )

    def spend_consent(self) -> None:
        """A payment move fires: consent is consumed, the amount survives
        into the history line and the purchase log. A later action of
        the same payment episode finds the consent already spent and
        leaves the record alone."""
        amount = self.gate.spend()
        if amount is not None:
            self.paid = amount
            self._paid_logged = False

    def log_purchase(self) -> None:
        """The doctrine's purchase line, harness-written ONCE as soon as
        a fired payment's result lands, fails, or the session dies —
        whatever the next check says, money may have moved, and the
        daily log is the cross-wake record. Idempotent: nothing new to
        log is a no-op."""
        if self.paid is None or self._paid_logged:
            return
        self._paid_logged = True
        self.log_day(
            f"conductor: {self.app}: payment ¥{self.paid:g} fired "
            f"(playbook {self.app}/{self.spec.name}) — verify the order before "
            "paying again"
        )

    def enter_gate(self, node: Checked) -> Turn:
        """The one enter-page guard moves, acting agents, and the boot's
        activate share: None when the node has no enter (a `start` runs
        unconditionally) or the page reads; else the recovery/handover
        step."""
        if not node.enter:
            return None
        assert self.verdict is not None
        expected = page_id(self.app, node.enter)
        wrong = self.mismatch(self.verdict, expected)
        if wrong is None:
            return None
        kind = _CHECKED_KIND[type(node)]
        return self.recover_or_handover(
            node,
            expected,
            recover.Mode.ENTER,
            f"{kind} {node.id!r} expects page {node.enter!r} ({wrong})",
        )

    def journal(self, text: str) -> None:
        """What the next synthesized note carries beside its own summary."""
        self._journal = text

    def peek(self) -> AssistantMessage:
        return self.synth(
            "peek",
            f"conductor: observing the screen before walking {self.app}/{self.spec.name}",
            gesture_vocab.PEEK,
            {},
        )

    def synth(
        self, kind: str, summary: str, tool: str, args: dict, *, channel: bool = False
    ) -> AssistantMessage:
        """The walk's synthesized turn: fold in anything journaled, then
        mint it (`turns.py` owns the shape and the call-id convention)."""
        if self._journal is not None:
            summary = f"{summary} | {self._journal}"
            self._journal = None
        return self.turns.synth(kind, summary, tool, args, channel=channel)

    def conclude(self, reason: str) -> None:
        """The walk's quiet completion — recorded, DONE, and no brief
        turn: the boot's ending is the next walk's beginning (its
        `select` step set the baton) or the model's own thread (it
        set none), and either way there is nothing to report to a
        model that is not about to speak from a brief."""
        log.info("conductor: %s/%s concluded — %s", self.app, self.spec.name, reason)
        self._end(Outcome.COMPLETED, reason)

    def handover(self, reason: str) -> AssistantMessage:
        """The walk's exit: log and record, then mint the ONE final
        synthesized [note, peek] brief turn (`brief.walk_brief`) — the
        distilled report the model resumes from. `_done` makes the NEXT
        advance the permanent None the conductor drops the program on."""
        log.warning(
            "conductor: handing %s/%s over to the model — %s",
            self.app,
            self.spec.name,
            reason,
        )
        self._end(Outcome.HANDOVER, reason)
        return self.synth(
            "brief",
            brief.walk_brief(
                reason,
                app=self.app,
                playbook=self.spec.name,
                node=self._node_id(),
                idx=self.idx,
                nodes=len(self.spec.nodes),
                outputs=self.outputs,
                consented=self.gate.consented,
                paid=self.paid,
            ),
            gesture_vocab.PEEK,
            {},
        )

    # ---- recovery ----

    def recover_or_handover(
        self, node: Checked, expected_id: str, mode: recover.Mode, reason: str
    ) -> Turn:
        """The page's declared hand before the model: a deviation is
        recovered toward the page the frozen cursor already requires —
        the cursor, outputs, and consent are untouched throughout. Never
        with consent bound, mid-gate, for an irreversible move, or once
        a payment fired: money keeps the hard handover (a restart from
        the top would walk back into the ask and pay again)."""
        if (
            self.gate.consented is not None
            or self.gate.awaiting
            or node.irreversible
            or self.paid is not None
        ):
            return self.handover(reason)
        if not owned_by(expected_id, self.app):
            # Recovery covers this pack's own pages only — a reserved or
            # channel target has no hand to declare.
            return self.handover(reason)
        st = recover.State(target=expected_id, mode=mode, reason=reason)
        v = self.verdict
        # The reading the page declared its hands for: the lock screen
        # (taps do not land there — the matcher reads it by shape), the
        # page itself under an overlay, or any other screen.
        reading = READING_ELSEWHERE
        if v is not None and v.matches(LOCKED_ID):
            reading = READING_LOCKED
        elif v is not None and v.occludes(expected_id):
            reading = READING_COVERED
        step = recover.plan(
            self._recoveries,
            self.spec.recovers.get(page_name(expected_id)),
            self._page_recoveries[expected_id],
            reading=reading,
        )
        if isinstance(step, recover.Exhausted):
            return self.handover(f"{reason} — {step.reason}")
        # The page's DECLARED hand — the planner decides WHETHER, the
        # walk interprets WHAT: a bare gesture, a landmark tap
        # (label-healed), or an argument-less macro.
        hand = step.hand
        note = (
            f"conductor: recovering toward {expected_id} via its declared hand "
            f"({reading})"
        )
        if hand.macro is not None:
            return self._recover_act(
                st,
                note,
                gesture_vocab.RUN_MACRO,
                {"name": qualified_macro(self.app, hand.macro)},
            )
        if hand.tool == "tap":
            landmark = self.landmarks.get(hand.landmark or "")
            if landmark is None:
                return self.handover(
                    f"{reason} (recover landmark {hand.landmark!r} undeclared)"
                )
            assert self.screen is not None
            bbox, located = recover.locate_landmark(landmark, self.screen)
            return self._recover_act(st, note + located, "tap", {"bbox": list(bbox)})
        assert hand.tool is not None
        return self._recover_act(st, note, hand.tool, {})

    def _recover_act(
        self, st: recover.State, note: str, tool: str, args: dict
    ) -> AssistantMessage:
        """The hand's turn: the engagement goes in flight, the page's
        limit and the walk-wide ceiling both count it, and the landing
        dispatch reads `KIND_RECOVER`."""
        self._recovery = st
        self._page_recoveries[st.target] += 1
        return self.synth(KIND_RECOVER, note, tool, args)

    def _recover_landed(self) -> Turn:
        """The hand's result view, judged. Restored → resume exactly
        where the walk stood: an interrupted enter re-checks and runs its
        move; an interrupted verify is satisfied by the restored page
        (the macro already ran — never re-run it). Still off → walk the
        route again from its first unsettled node (the start re-runs,
        which is a force_quit hand's whole point)."""
        st = self._recovery
        assert st is not None and self.verdict is not None
        self._recovery = None
        self._step = None
        if self.verdict.matches(st.target):
            self.journal(f"recovered {st.target} via its declared hand")
            if st.mode is recover.Mode.VERIFY:
                return self.advance_cursor()
            return self.next()
        self.journal(f"recover hand ran — walking again toward {st.target}")
        self.idx = max(self.spec.first_unsettled(self.outputs), self._floor)
        return self.next()

    # ---- the record ----

    def abandon(self) -> None:
        """The session ended with this walk mid-flight (the plugin's
        teardown): one runs.jsonl line so the escalation KPI counts it,
        plus a daily-log breadcrumb when the walk actually moved the
        phone. Latched like every terminal moment — a walk that already
        closed is a no-op, and so is one that never started."""
        if self.phase in (Phase.FRESH, Phase.PAUSED) or self.outcome is not None:
            # Never started, already closed, or a stepping pause (the
            # walk continues from its persisted position next run).
            return
        self.log_purchase()  # a fired payment outlives the session
        node = self._node_id() or "(end)"
        self._record_run(Outcome.ABANDONED, "session ended mid-walk")
        self.log_day(
            f"conductor: {self.app}/{self.spec.name} cut short mid-walk "
            f"at node {node} — the next wake starts the route over"
        )

    @property
    def dry(self) -> bool:
        """A dry walk leaves no trace: no runs row, no daily-log entry,
        no suspension file (`Record`)."""
        return self.record.dry

    @property
    def outcome(self) -> Outcome | None:
        """How the walk ended, None while it runs (`Record`)."""
        return self.record.outcome

    def log_day(self, entry: str) -> None:
        self.record.day(entry)

    def _end(self, outcome: Outcome, reason: str = "") -> None:
        """A terminal moment that also closes the walk: recorded, and
        DONE — the next advance is the permanent None."""
        self._record_run(outcome, reason)
        self.phase = Phase.DONE

    def _record_run(self, outcome: Outcome, reason: str = "") -> None:
        """The walk's runs.jsonl line, with the walk's own numbers
        (`Record.run` keeps the first terminal moment)."""
        self.record.run(
            outcome,
            idx=self.idx,
            nodes=len(self.spec.nodes),
            node=self._node_id(),
            reason=reason,
            micros=self._micros,
            rescues=self._recoveries,
            values=self.values,
            total=self.paid,
        )
