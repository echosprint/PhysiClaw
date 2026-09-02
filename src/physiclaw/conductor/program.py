"""One playbook mid-walk — the walk's core.

A `Program` is one playbook being executed. The conductor asks it for
each turn; it answers with a synthesized ``[note, one-other]`` assistant
turn, with a ``DecisionRequest`` the conductor brokers through the
micro-caller and feeds back via ``resolve()``, or with ``None`` — "I
hand over". ``None`` is permanent for the session: the conductor goes
quiet and the model takes over with the transcript as the handoff,
because every synthesized turn and every tool result is ordinary
history the model can read. Every hand-over (and completion) is
preceded by ONE final synthesized ``[note, peek]`` turn carrying the
brief (`brief.py`) — the distilled report of why the walk stopped and
where its state stands — so the model never resumes blind.

This file is the walk alone: the cursor, the one action in flight, the
page verdict every step judges against, recovery toward a page, the
money state an ask binds and a payment move spends, and the terminal
moments (handover, completion, suspension). What each STEP does is its
executor's (`step.py` is the contract) — one per route entry kind, the
runner the GitHub-Actions analogy asks for:

  - `step_do.py`     a `do`/`start` move: enter check, the macro, verify
  - `step_agent.py`  an `agent` step: the pure-text call, or the episode
  - `step_ask.py`    `ask` (send, hold, judge the reply, bind consent)
                     and `tell` (send, suspend, read a cancel on resume)

The opening peek doubles as resume: a killed session's next wake
fast-forwards past every `do` whose verify page already matches the
screen, so completed gestures are never replayed. A mechanical
deviation at a check — a popup band, a locked phone, a wandered-into
page — goes to recovery (`recover.py`: unlock, one settle re-peek, then
the page's own declared hand); only what recovery cannot restore — and
every blocked or errored call — hands over. Money never recovers: with
consent bound or on an irreversible move, a deviation is the model's.
"""

import logging
from dataclasses import dataclass, field

from physiclaw.common import daylog, gesture_vocab
from physiclaw.common.listing import Screen
from physiclaw.common.logger import write_json_atomic
from physiclaw.conductor import brief, recover, views, walklog
from physiclaw.conductor.channel import Channel
from physiclaw.conductor.match import Verdict, match_screen
from physiclaw.conductor.micro import MicroOutcome
from physiclaw.conductor.pages import (
    Landmark,
    PagePrint,
    owned_by,
    page_id,
    page_name,
)
from physiclaw.conductor.playbook import (
    AgentNode,
    AskNode,
    DoNode,
    Playbook,
    PlaybookError,
    TellNode,
    qualified_macro,
)
from physiclaw.conductor.step import Step, Turn
from physiclaw.conductor.step_agent import AgentStep
from physiclaw.conductor.step_ask import AskStep, TellResume, TellStep
from physiclaw.conductor.step_do import DoStep
from physiclaw.conductor.suspension import (
    SUSPENDED_SCHEMA,
    clear_suspended,
    suspended_path,
)
from physiclaw.conductor.turns import Turnsmith
from physiclaw.contract.dto import AssistantMessage, Message
from physiclaw.macros.model import Macro

log = logging.getLogger(__name__)

# runtime.sentinel.WAIT, spelled literally: the conductor may not import
# engine runtimes; a test pins the two equal.
SUSPEND_STATUS = "WAIT"

# The executor for each route entry kind (`Node` is a closed union).
_STEP_FOR: dict[type, type[Step]] = {
    DoNode: DoStep,
    AgentNode: AgentStep,
    AskNode: AskStep,
    TellNode: TellStep,
}


@dataclass
class Gate:
    """The ask-and-hold state — one object, one suspension projection. The
    walk owns the cursor; this owns everything between "ask sent" and
    "consent bound": the ask text, the thread snapshot it is diffed
    against, the money numbers, the two bounded counters, and the
    LLM-tier handshake (`llm_batch` non-empty = a confirm_reply request
    is outstanding and owns the next resolve()). Consent lives here,
    not on the ask step, because the payment move AFTER the ask spends
    it — and a suspension in between must carry it."""

    ask: str = ""
    baseline: set[str] = field(default_factory=set)
    quoted: float | None = None
    cap: float | None = None
    consented: float | None = None
    silence: int = 0
    checks: int = 0
    awaiting: bool = False  # ask sent, polling for the reply
    # The batch behind an outstanding LLM judgment, held UNBASELINED
    # until the judgment actually arrives — a provider failure must
    # leave the reply visible to the next round ("judged once,
    # eventually"), never swallowed.
    llm_batch: list[str] = field(default_factory=list)
    tried_open: bool = False

    def spend(self) -> float | None:
        """Consent is CONSUMED by firing: a later payment needs its own
        gate's fresh confirm, never this one's leftovers. Returns the
        amount that fired."""
        amount, self.consented, self.quoted = self.consented, None, None
        return amount

    def to_suspended(self) -> dict:
        """The persisted projection — the one field list, beside the
        fields. Counters and the in-flight handshake deliberately reset
        on resume; `consented` persists so a post-consent suspension can
        never resume into a refused payment."""
        return {
            "ask_text": self.ask,
            "baseline": sorted(self.baseline),
            "quoted": self.quoted,
            "cap": self.cap,
            "consented": self.consented,
            "awaiting": self.awaiting,
        }

    @classmethod
    def from_suspended(cls, data: dict) -> "Gate":
        return cls(
            ask=str(data.get("ask_text") or ""),
            baseline=set(data.get("baseline") or []),
            quoted=data.get("quoted"),
            cap=data.get("cap"),
            consented=data.get("consented"),
            awaiting=bool(data.get("awaiting")),
        )


class Program:
    """One playbook mid-walk. Constructed per session (the conductor
    plugin's wake setup builds it), so the cursor state lives for
    exactly one attempt; persistence across wakes is the locate peek,
    not saved state."""

    def __init__(
        self,
        *,
        spec: Playbook,
        values: dict[str, str],
        pack_macros: dict[str, Macro],
        prints: list[PagePrint],
        channel: Channel | None = None,
        suspended: dict | None = None,
        landmarks: "dict[str, Landmark] | None" = None,
    ):
        self.app = spec.app
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
        # setup.build_program). None degrades to hand-over at the first
        # ask.
        self.channel = channel
        # The pack's declared fixed spots — recover hands tap them, agent
        # episodes are granted them by name.
        self.landmarks: dict[str, Landmark] = landmarks or {}
        self.idx = 0
        # Turn minting + the one action in flight (`turns.py`).
        self.turns = Turnsmith("walk")
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
        # Moves whose failed run already earned their single retry this
        # walk (an abort under an overlay, a locked phone).
        self.retried_moves: set[str] = set()
        # The step executor at the cursor (a resume pre-step rides the
        # same slot before the walk proper opens).
        self._step: Step | None = None
        # Recovery (`recover.py`): the one engagement in flight and the
        # walk-lifetime action count that budgets it.
        self._recovery: recover.State | None = None
        self._recoveries = 0
        self._resumed = False
        # A suspended tell's resume owes the user one thread read before
        # the walk continues: the wake that resumes it may BE their
        # cancel reply. Set only by _restore_suspended.
        self._confirm_check = False
        # True once any device-changing action was synthesized (peeks
        # don't count). A walk that "completes" without ever acting hit
        # a coincidental page match — that must not count as done.
        self._acted = False
        # The telemetry pair (`walklog`): decision outcomes brokered to
        # this walk, and the one-shot latch so exactly one runs.jsonl
        # line lands per walk whatever terminal path fires.
        self._micros = 0
        self._run_recorded = False
        # True once advance() was ever called — `abandon` records only
        # walks that actually started.
        self._started = False
        # Terminal: the brief turn (handover or completion) was minted.
        # The NEXT advance is the permanent None the conductor drops the
        # program on.
        self._done = False
        # The journal line the next synthesized note carries
        # (record-don't-replay: the transcript carries what happened).
        self._journal: str | None = None
        if suspended is not None:
            # A resumed walk is ALSO whole at construction: the suspended
            # projection overlays the fresh state right here, so no
            # caller ever patches a program up afterwards.
            self._restore_suspended(suspended)
            log.info(
                "conductor: resuming suspended %s/%s at node %d (%s)",
                self.app,
                spec.name,
                self.idx + 1,
                "awaiting reply" if self.gate.awaiting else "walk",
            )

    # ---- suspension ----

    def _suspended_dict(self, resume_idx: int) -> dict:
        """The suspension projection: walk state here, gate state via
        `Gate.to_suspended` — each field list lives beside its fields."""
        return {
            "schema": SUSPENDED_SCHEMA,
            "app": self.app,
            "playbook": self.spec.name,
            "idx": resume_idx,
            "values": self.values,
            "outputs": self.outputs,
            **self.gate.to_suspended(),
        }

    def _restore_suspended(self, data: dict) -> None:
        idx = int(data["idx"])
        if not (0 <= idx <= len(self.spec.nodes)):
            # The spec changed under the suspension (edited shorter) — a
            # stale cursor must not fake a completion. Raising drops the
            # suspension (load_suspended is fail-open).
            raise PlaybookError(f"suspended idx {idx} is outside the playbook")
        self.idx = idx
        self._acted = True  # the suspended session acted; completion is real
        self.outputs = {str(k): str(v) for k, v in (data.get("outputs") or {}).items()}
        self.gate = Gate.from_suspended(data)
        self._resumed = True
        # A suspended tell (message away, not awaiting a reply) resumes
        # mid-walk — read the thread for a cancel before continuing.
        self._confirm_check = (
            not self.gate.awaiting
            and bool(self.gate.ask)
            and self.idx < len(self.spec.nodes)
        )

    def suspend(self, *, resume_idx: int, awaiting: bool) -> AssistantMessage:
        """Write the suspended state and close the session WAIT. No job is
        synthesized: a WAIT without a session-created job auto-schedules
        the follow-up alarm (`contract.drive`) — the file, not the job,
        is what resumes the walk on ANY next wake."""
        self.gate.awaiting = awaiting
        write_json_atomic(suspended_path(), self._suspended_dict(resume_idx))
        recap = f"waiting for the user's reply on {self.app}/{self.spec.name}"
        self._record_run("suspended", recap)
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
        to broker (feed the outcome back via ``resolve``); or None — hand
        over to the model. Never raises: a spec-level failure (a ref with
        no value) hands over with its reason, and a program bug degrades
        to a handover too, not a crashed session (the model finishes the
        task either way)."""
        self._started = True
        return self._guarded(lambda: self._advance(history))

    def resolve(self, outcome: MicroOutcome | None) -> Turn:
        """Continue the walk with a micro-call's outcome (None = the call
        failed or was under-confident — hand over). Same return contract
        and same never-raises guarantee as ``advance``."""
        self._micros += 1
        return self._guarded(lambda: self._resolve(outcome))

    def _guarded(self, run) -> Turn:
        try:
            return run()
        except PlaybookError as e:
            return self.handover(str(e))
        except Exception:
            log.exception("conductor: program crashed — handing over to the model")
            self._record_run("crashed", "program crashed")
            return None

    def _advance(self, history: list[Message]) -> Turn:
        if self._done:
            # The brief turn was the walk's last word; its peek result is
            # ordinary history. Quiet from here — the conductor drops us.
            return None
        if self.turns.pending is None:
            if self.gate.awaiting:
                # Suspended-at-gate resume: the ask was sent before the
                # suspension — the ask step picks up at its reply check.
                node = (
                    self.spec.nodes[self.idx]
                    if self.idx < len(self.spec.nodes)
                    else None
                )
                if not isinstance(node, AskNode):
                    return self.handover(
                        "suspended awaiting a reply, but no ask at the cursor"
                    )
                self._step = AskStep(self, node)
                return self._step.open()
            if self._confirm_check and self.channel is not None:
                # Suspended-tell resume: this wake may BE the user's
                # cancel reply — read the thread before the walk acts.
                self._confirm_check = False
                self._step = TellResume(self, None)
                return self._step.open()
            # Observe before acting. The peek is also how a killed
            # session resumes: _locate fast-forwards past moves whose
            # verify page already matches, so completed gestures are
            # never replayed.
            return self.peek()
        pending, result, failed = self.turns.settle(history)
        kind = pending.kind
        if kind == "suspend":
            # The suspension file is already written; whether the
            # end_session was blocked, its result never arrived, or the
            # session simply ran on, a dead walk must not resurrect on
            # the next wake.
            clear_suspended()
            return self.handover(
                f"{failed or 'suspend end_session returned'} — suspension dropped"
            )
        if failed is not None:
            if self._step is not None and kind in self._step.kinds:
                retry = self._step.failed(kind, failed, result)
                if retry is not None:
                    return retry
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
        if kind == "peek":
            if self._resumed:
                # A suspended walk trusts its stored cursor — the next
                # node's own checks judge whether the world still fits.
                self._resumed = False
            else:
                self.idx = self._locate(self.verdict)
            return self.next()
        if kind.startswith("recover-"):
            return self._recover_landed(kind)
        if self._step is not None and kind in self._step.kinds:
            return self._step.landed(kind)
        # A typo'd kind at a synth site must fail loudly, never silently
        # walk the next node.
        return self.handover(f"unknown pending action kind {kind!r}")

    def _resolve(self, outcome: MicroOutcome | None) -> Turn:
        if self._step is None:
            return self.handover("a decision arrived with no step in flight")
        return self._step.resolve(outcome)

    # ---- the cursor ----

    def next(self) -> Turn:
        """Walk the node at the cursor: open its step executor, or finish.
        Works from the stored screen/verdict — every path here observed
        one first."""
        if self.verdict is None:
            return self.handover("no screen observed yet")
        nodes = self.spec.nodes
        if self.idx >= len(nodes):
            if not self._acted:
                # The opening locate fast-forwarded past EVERYTHING off
                # one page match — a coincidence (common pages recur),
                # not evidence the task ran. Hand over; the model verifies.
                return self.handover(
                    "screen already reads as the final page but this walk "
                    "did nothing — verify the task state before trusting it"
                )
            log.info(
                "conductor: playbook %s/%s complete — handing over",
                self.app,
                self.spec.name,
            )
            self._record_run("completed")
            self._done = True
            return self.synth(
                "brief",
                brief.completion_brief(self.app, self.spec.name, len(nodes)),
                gesture_vocab.PEEK,
                {},
            )
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
        nodes = self.spec.nodes
        return nodes[self.idx].id if self.idx < len(nodes) else None

    def _locate(self, verdict: Verdict) -> int:
        """Cursor for the current screen: just past the LAST move of the
        playbook's LEADING `do` RUN whose verify page matches it (that
        page proves the move's outcome holds), else the top. The scan
        stops at the first non-do node: a page match proves navigation
        happened, never that an agent step or an ask ran — fast-forwarding
        past work off one page coincidence would quote a gate total for
        a cart that was never filled."""
        nodes = self.spec.nodes
        # A COMPLETED pure-text agent never re-runs: its outputs are
        # recorded, and re-deriving them mid-walk (a recover hand's
        # walk-from-the-top, a resume) could silently change them.
        base = 0
        while base < len(nodes) and self._text_agent_settled(nodes[base]):
            base += 1
        if verdict.kind != "match":
            log.info("conductor: screen is %s — starting from the top", verdict.kind)
            return base
        resume = base
        for i in range(base, len(nodes)):
            node = nodes[i]
            if not isinstance(node, DoNode):
                break
            if page_id(self.app, node.verify) == verdict.page_id:
                resume = i + 1
        if resume:
            log.info(
                "conductor: screen already on %s — resuming at node %d/%d",
                verdict.page_id,
                resume + 1,
                len(self.spec.nodes),
            )
        return resume

    def _text_agent_settled(self, node) -> bool:
        """A pure-text agent whose outputs are already recorded — the
        locate scan skips it rather than re-spending its call."""
        return (
            isinstance(node, AgentNode)
            and not node.tools
            and all(f"{node.id}.{f}" in self.outputs for f in node.return_fields)
        )

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
        seen = verdict.page_id or "no known page"
        return f"screen reads as {verdict.kind}: {seen} — {verdict.detail}"

    def money_page_block(self, what: str) -> str | None:
        """Money reads and fires only off a VERIFIED own-pack page: an ask
        is sent on the IM thread and quotes the consented total, so an
        unverified screen could satisfy the predicates with the
        conductor's own ask bubble. None when the current verdict is
        such a page; else the handover reason."""
        v = self.verdict
        if v is not None and v.kind == "match" and owned_by(v.page_id or "", self.app):
            return None
        return (
            f"{what}: current screen is not a verified {self.app} page — "
            "money never reads or fires blind"
        )

    def spend_consent(self) -> None:
        """A payment move fires: consent is consumed, the amount survives
        into the history line and the purchase log."""
        self.paid = self.gate.spend()
        self._paid_logged = False

    def log_purchase(self) -> None:
        """The doctrine's purchase line, harness-written ONCE the moment
        a payment move's result lands — whatever the next check says,
        money may have moved, and the daily log is the cross-wake
        record. Idempotent: a landing with nothing new to log is a no-op."""
        if self.paid is None or self._paid_logged:
            return
        self._paid_logged = True
        self.log_day(
            f"conductor: {self.app}: paid ¥{self.paid:g} "
            f"(playbook {self.app}/{self.spec.name})"
        )

    def enter_gate(self, node: "DoNode | AgentNode") -> Turn:
        """The one enter-page guard moves and acting agents share: None
        when the node has no enter (a `start` runs unconditionally) or
        the page reads; else the recovery/handover step."""
        if not node.enter:
            return None
        assert self.verdict is not None
        expected = page_id(self.app, node.enter)
        wrong = self.mismatch(self.verdict, expected)
        if wrong is None:
            return None
        kind = "agent" if isinstance(node, AgentNode) else "move"
        return self.recover_or_handover(
            node,
            expected,
            recover.MODE_ENTER,
            f"{kind} {node.id!r} expects page {node.enter!r} ({wrong})",
        )

    def journal(self, text: str) -> None:
        """What the next synthesized note carries beside its own summary."""
        self._journal = text

    def peek(self) -> AssistantMessage:
        return self.synth(
            "peek",
            f"conductor: observing the screen to locate {self.app}/{self.spec.name}",
            gesture_vocab.PEEK,
            {},
        )

    def synth(
        self, kind: str, summary: str, tool: str, args: dict, *, channel: bool = False
    ) -> AssistantMessage:
        """The walk's synthesized turn: fold in anything journaled, note
        whether this one touched the phone, then mint it (`turns.py`
        owns the shape and the call-id convention). Any non-peek tool
        marks the walk as having acted — a "completion" that never
        acted must not count."""
        if self._journal is not None:
            summary = f"{summary} | {self._journal}"
            self._journal = None
        if tool != gesture_vocab.PEEK:
            self._acted = True
        return self.turns.synth(kind, summary, tool, args, channel=channel)

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
        self._record_run("handover", reason)
        self._done = True
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
            ),
            gesture_vocab.PEEK,
            {},
        )

    # ---- recovery ----

    def recover_or_handover(
        self, node: "DoNode | AgentNode", expected_id: str, mode: str, reason: str
    ) -> Turn:
        """Try recovery before the model: a mechanical deviation (popup,
        lock, wandered deeper) is recoverable in code toward the page the
        frozen cursor already requires — the cursor, outputs, and consent
        are untouched throughout. Never with consent bound, mid-gate, or
        for an irreversible move: money keeps the hard handover."""
        if self.gate.consented is not None or self.gate.awaiting or node.irreversible:
            return self.handover(reason)
        if not owned_by(expected_id, self.app):
            # Recovery covers this pack's own pages only — a reserved or
            # channel target has no hand to declare.
            return self.handover(reason)
        if self._recovery is None:
            self._recovery = recover.State(target=expected_id, mode=mode, reason=reason)
        return self._recover_step()

    def _recover_step(self) -> Turn:
        """One action: ask the planner (`recover.plan`) for the next one
        and synthesize it; Exhausted → handover carrying the ORIGINAL
        check failure plus what was tried (the brief quotes it)."""
        st = self._recovery
        assert st is not None
        assert self.verdict is not None and self.screen is not None
        hand = self.spec.recovers.get(page_name(st.target))
        step = recover.plan(self.verdict, self.screen, st.tries, self._recoveries, hand)
        if isinstance(step, recover.Settle):
            # Free — verification hygiene, not recovery: no action
            # budget, just the one re-peek marker.
            st.tries[recover.RUNG_SETTLE] = 1
            return self.synth(
                f"recover-{recover.RUNG_SETTLE}",
                f"conductor: re-reading a possibly mid-transition screen "
                f"before recovering toward {st.target}",
                gesture_vocab.PEEK,
                {},
            )
        if isinstance(step, recover.Exhausted):
            note = st.note()
            self._recovery = None
            return self.handover(
                f"{st.reason} — {step.reason}"
                + (f" (recovery tried: {note})" if note else "")
            )
        if isinstance(step, recover.Unlock):
            return self._recover_act(
                recover.RUNG_UNLOCK,
                "conductor: phone locked mid-walk — unlocking",
                gesture_vocab.UNLOCK_PHONE,
                {},
            )
        assert isinstance(step, recover.Hand)
        # The page's DECLARED hand — the planner decides WHEN, the walk
        # interprets WHAT: a bare gesture, a landmark tap (label-healed),
        # or an argument-less macro.
        hand = step.hand
        note = f"conductor: recovering toward {st.target} via its declared hand"
        if hand.macro is not None:
            return self._recover_act(
                recover.RUNG_HAND,
                note,
                gesture_vocab.RUN_MACRO,
                {"name": qualified_macro(self.app, hand.macro)},
            )
        if hand.tool == "tap":
            landmark = self.landmarks.get(hand.landmark or "")
            if landmark is None:
                self._recovery = None
                return self.handover(
                    f"{st.reason} (recover landmark {hand.landmark!r} undeclared)"
                )
            bbox, located = recover.locate_landmark(landmark, self.screen)
            return self._recover_act(
                recover.RUNG_HAND, note + located, "tap", {"bbox": list(bbox)}
            )
        assert hand.tool is not None
        return self._recover_act(recover.RUNG_HAND, note, hand.tool, {})

    def _recover_act(
        self, rung: str, note: str, tool: str, args: dict
    ) -> AssistantMessage:
        """One recovery action's shared bookkeeping — the rung counter,
        the walk-wide budget — then the synthesized turn under the
        `recover-<rung>` kind the landing dispatch reads."""
        st = self._recovery
        assert st is not None
        st.tries[rung] = st.tries.get(rung, 0) + 1
        self._recoveries += 1
        return self.synth(f"recover-{rung}", note, tool, args)

    def _recover_landed(self, kind: str) -> Turn:
        """The recovery action's result view, judged. Restored → resume
        exactly where the walk stood: an interrupted enter re-checks and
        runs its move; an interrupted verify is satisfied by the restored
        page (the macro already ran — never re-run it). Still off after
        the declared hand → re-locate on the route and walk again (the
        start re-runs when the locate lands at the top, which is a
        force_quit hand's whole point). Still off otherwise → the next
        planned action."""
        st = self._recovery
        assert st is not None and self.verdict is not None
        if self.verdict.matches(st.target):
            note = st.note()
            self.journal(f"recovered {st.target}" + (f" ({note})" if note else ""))
            self._recovery = None
            if st.mode == recover.MODE_VERIFY:
                return self.advance_cursor()
            self._step = None
            return self.next()
        if kind == f"recover-{recover.RUNG_HAND}":
            self.journal(f"recover hand ran — walking again toward {st.target}")
            self._recovery = None
            self._step = None
            self.idx = self._locate(self.verdict)
            return self.next()
        return self._recover_step()

    # ---- the record ----

    def abandon(self) -> None:
        """The session ended with this walk mid-flight (the plugin's
        teardown): one runs.jsonl line so the escalation KPI counts it,
        plus a daily-log breadcrumb when the walk actually moved the
        phone. Latched like every terminal moment — a walk that already
        closed is a no-op, and so is one that never started."""
        if not self._started or self._run_recorded:
            return
        node = self._node_id() or "(end)"
        self._record_run("abandoned", "session ended mid-walk")
        if self._acted:
            self.log_day(
                f"conductor: {self.app}/{self.spec.name} cut short mid-walk "
                f"at node {node} — the next wake re-locates from the screen"
            )

    def log_day(self, entry: str) -> None:
        """One daily-log line in the agent's own convention
        (`common.daylog`), stamped, fail-open — the walk's activity must
        land in the same record the model reads at wake, and a logging
        failure must never take the walk down."""
        try:
            daylog.append_log(daylog.stamped(entry))
        except OSError:
            log.warning("conductor daily-log write failed", exc_info=True)

    def _record_run(self, outcome: str, reason: str = "") -> None:
        """The walk's one runs.jsonl line (`walklog`) — first terminal
        moment wins, so a suspension whose end_session is later blocked
        stays recorded as suspended."""
        if self._run_recorded:
            return
        self._run_recorded = True
        walklog.record(
            app=self.app,
            playbook=self.spec.name,
            outcome=outcome,
            idx=self.idx,
            nodes=len(self.spec.nodes),
            node=self._node_id(),
            reason=reason,
            micros=self._micros,
            rescues=self._recoveries,
            values=self.values,
            total=self.paid,
        )
