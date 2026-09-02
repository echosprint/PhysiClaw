"""One playbook mid-walk — the conductor's synthesized turns.

A `Program` is one playbook being executed. The conductor asks it for
each turn; it answers with a synthesized ``[note, one-other]`` assistant
turn (a LEG as ``run_macro``; the opening ``peek``; a decision's own
tap/swipe primitive), with a ``DecisionRequest`` the conductor brokers
through the micro-caller and feeds back via ``resolve()``, or with
``None`` — "I hand over". ``None`` is permanent for the session: the
conductor goes quiet and the model takes over with the transcript as the
handoff, because every synthesized turn and every tool result is
ordinary history the model can read. Every hand-over (and completion) is
preceded by ONE final synthesized ``[note, peek]`` turn carrying the
brief (`brief.py`) — the distilled report of why the walk stopped and
where its state stands — so the model never resumes blind.

The walk executes every node type, strictly:

  - Before a leg: its ``enter:`` page (when declared — a ``start`` leg
    has none and runs unconditionally) must match the current screen.
    After a leg: its ``verify:`` page must match the screen the macro
    result carries. A mechanical deviation — a popup band, a locked
    phone, a wandered-into page — goes to recovery first: in DECLARED
    mode (any ``recover:`` in the playbook) the page's own declared
    hand after an implicit unlock/settle, else the legacy rescue ladder
    (`rescue.py`); only what recovery cannot restore — and every
    blocked or errored call, and any reserved ``ios.*`` page ref —
    hands over.
  - An AGENT step is the model's, inside the author's fence: no tools =
    one pure-text call filling its declared ``returns``; tools = an
    EPISODE — each turn one constrained call (append-only context, so
    every request's prefix is byte-identical to the previous one) whose
    answer the walk grounds to a live bbox; ``done`` is audited against
    the adjacent verify page, and a payment episode re-runs the money
    predicates before EVERY tap.
  - A DECIDE becomes one micro-call over the decide-time screen. A pick
    is acted on by a conductor tap primitive at the chosen row; a
    ``scroll`` routed back to the same node swipes and re-asks, bounded
    by ``max_visits``. A failed or under-confident call hands over.
    Deterministic calls (``next_item``) are answered by the program
    itself — no prompt.
  - The opening peek doubles as resume: a killed session's next wake
    fast-forwards past every node whose ``verify`` page already matches
    the screen, so completed gestures are never replayed.
  - CONFIRM and HUMAN_GATE send the playbook's ``message:`` VERBATIM
    over the channel pack (only the author knows the user's language),
    then suspend (CONFIRM) or hold for the tiered reply check (the gate).
    Suspends write ``playbooks/suspended.json``; ANY next wake resumes.
  - Money runs in code: the parser guarantees a payment leg directly
    follows its ``gate: payment`` HUMAN_GATE; consent binds the quoted
    total; the fire-time predicates re-read the sheet.
  - The ledger walk: the deterministic ``next_item`` loop shops each
    item, RECONCILE converges the cart on the list in code (steppers,
    re-shop, unclaimed rows left alone), and a gate ``revise:`` turns a
    "yes, but change it" reply into a list rewrite and a fresh ask.

How a Program comes to exist (the overture's activation, a resumed
suspension, the CLI rehearsal) lives in `setup.py`; the suspension file
in `suspension.py`. The decision kernels this walk applies are pure
functions next door — `money.py` (the payment predicates),
`reconcile.py` (the cart step planner), `ledger.py` (list state, cart
readers, revision merge) — with `channel.py`, `memory.py`, `reply.py`,
`views.py` as the remaining senses. This file is the state machine
alone: cursor, pending action, and who does what next.
"""

import json
import logging
from dataclasses import dataclass, field

from physiclaw.common import daylog, gesture_vocab
from physiclaw.common.listing import Screen
from physiclaw.common.logger import write_json_atomic
from physiclaw.conductor import (
    brief,
    ledger,
    memory,
    money,
    reconcile,
    reply,
    rescue,
    views,
    walklog,
)
from physiclaw.conductor.calls import (
    ACT_BACK,
    ACT_SCROLL_DOWN,
    ACT_SCROLL_UP,
    AGENT_DONE,
    AGENT_TOOL_VERBS,
    CALLS,
    ESCALATE,
    LEDGER_FIELDS,
    NEXT_ITEM,
)
from physiclaw.conductor.channel import Channel
from physiclaw.conductor.match import (
    Verdict,
    match_screen,
    reads_as_locked,
)
from physiclaw.conductor.micro import (
    ACT_ARM,
    AGENT_ACT,
    AGENT_FIELDS,
    CONFIRM_REPLY,
    REVISE_LIST,
    Candidate,
    DecisionRequest,
    MicroOutcome,
    act_block,
    act_candidates,
    build_request,
    canonical_reply,
)
from physiclaw.conductor.pages import (
    OPEN_MACRO,
    THREAD_ID,
    Landmark,
    PagePrint,
    page_id,
)
from physiclaw.conductor.playbook import (
    GATE_MAX_CHECKS,
    GATE_MAX_REVISIONS,
    AgentNode,
    ConfirmNode,
    DecideNode,
    HumanGateNode,
    LegNode,
    Playbook,
    PlaybookError,
    ReconcileNode,
    fill_refs,
    qualified_macro,
)
from physiclaw.conductor.suspension import (
    SUSPENDED_SCHEMA,
    clear_suspended,
    suspended_path,
)
from physiclaw.conductor.turns import Turnsmith
from physiclaw.contract.dto import AssistantMessage, Message, ToolResultMessage
from physiclaw.macros.model import NO_GESTURES_NOTE, Macro

log = logging.getLogger(__name__)

# Where the scroll-for-more swipe originates: a mid-content band, clear
# of top chrome and the tab bar, so the drag scrolls the list rather
# than dismissing or paging anything. Stylus up → page scrolls down.
SCROLL_BBOX = (0.2, 0.35, 0.8, 0.65)

# HUMAN_GATE ask-and-hold bounds: in-session reply polling cadence, and
# how many silent rounds before the session suspends for the next wake.
# (Unclear-reply rounds are bounded separately by GATE_MAX_CHECKS.)
GATE_WAIT_SECONDS = 45
SILENCE_ROUNDS = 3

# runtime.sentinel.WAIT, spelled literally: the conductor may not import
# engine runtimes; a test pins the two equal.
SUSPEND_STATUS = "WAIT"


@dataclass
class _Gate:
    """The ask-and-hold state — one object, one suspension projection. The
    walk owns the cursor; this owns everything between "ask sent" and
    "consent bound": the ask text, the thread snapshot it is diffed
    against, the money numbers, the two bounded counters, and the
    LLM-tier handshake (`llm_reply` non-None = a confirm_reply request
    is outstanding and owns the next resolve())."""

    ask: str = ""
    baseline: set[str] = field(default_factory=set)
    quoted: float | None = None
    cap: float | None = None
    consented: float | None = None
    silence: int = 0
    checks: int = 0
    awaiting: bool = False  # ask sent, polling for the reply
    llm_reply: str | None = None
    # The batch behind llm_reply, held UNBASELINED until a judgment
    # actually arrives — a provider failure must leave the reply visible
    # to the next round ("judged once, eventually"), never swallowed.
    llm_batch: list[str] = field(default_factory=list)
    tried_open: bool = False
    # Bounded "yes, but change it" cycles (persists — a suspended revision
    # spree must not reset its budget); True while a revise_list request
    # is outstanding and owns the next resolve().
    revisions: int = 0
    revise_pending: bool = False

    def to_suspended(self) -> dict:
        """The persisted projection — the one field list, beside the
        fields. Counters and the in-flight handshake deliberately reset
        on resume; `consented` persists so a post-consent suspension can never
        resume into a refused payment, `revisions` so a suspension cannot
        refill the revision budget."""
        return {
            "ask_text": self.ask,
            "baseline": sorted(self.baseline),
            "quoted": self.quoted,
            "cap": self.cap,
            "consented": self.consented,
            "awaiting": self.awaiting,
            "revisions": self.revisions,
        }

    @classmethod
    def from_suspended(cls, data: dict) -> "_Gate":
        return cls(
            ask=str(data.get("ask_text") or ""),
            baseline=set(data.get("baseline") or []),
            quoted=data.get("quoted"),
            cap=data.get("cap"),
            consented=data.get("consented"),
            awaiting=bool(data.get("awaiting")),
            revisions=int(data.get("revisions") or 0),
        )


@dataclass
class _Episode:
    """One acting agent step mid-flight. `history` is the append-only
    transcript of settled (user block, model reply) pairs — replayed
    verbatim on every call so each request's prefix is byte-identical
    to the previous one; `block` is the pending user block the next
    call sends. `consented` is a payment episode's bound, stashed at
    open so the per-tap predicates keep checking after the gate's
    consent is consumed by the first fire."""

    node: AgentNode
    block: str = ""
    history: list[tuple[str, str]] = field(default_factory=list)
    candidates: tuple[Candidate, ...] = ()
    calls: int = 0
    scrolls: int = 0
    consented: float | None = None
    spent: bool = False  # the first tap fired — consent consumed
    logged: bool = False  # the purchase daylog line was written
    pending_desc: str = ""


class Program:
    """One playbook mid-walk. Constructed per session (the conductor
    plugin's wake setup builds it), so the cursor state lives for
    exactly one attempt; persistence across wakes is the locate peek,
    not saved state."""

    def __init__(
        self,
        *,
        app: str,
        spec: Playbook,
        values: dict[str, str],
        pack_macros: dict[str, Macro],
        prints: list[PagePrint],
        channel: Channel | None = None,
        suspended: dict | None = None,
        landmarks: "dict[str, Landmark] | None" = None,
    ):
        self.app = app
        self.spec = spec
        self.values = values
        # The `inputs.<name>` half of the ref-resolution dict, built once:
        # `values` never mutates after construction.
        self._input_vals = {f"inputs.{k}": v for k, v in values.items()}
        # The program's own qualified dispatch contribution — merged into
        # the wake registry by session_setup.
        self.pack_macros = pack_macros
        self._prints = prints
        self._ids = {n.id: i for i, n in enumerate(spec.nodes)}
        # The user-channel infrastructure (constructor-injected via
        # setup._build_program). None degrades to hand-over at the first
        # ask.
        self.channel = channel
        self._idx = 0
        # Turn minting + the one action in flight (`turns.py`) — the walk
        # keeps its own vocabulary (journal, acted) around it.
        self._turns = Turnsmith("walk")
        self._gate = _Gate()
        self._resumed = False
        # True once any device-changing action was synthesized (peeks
        # don't count). A walk that "completes" without ever acting hit
        # a coincidental page match — that must not consume the arm.
        self._acted = False
        # The telemetry pair (`walklog`): decision outcomes brokered to
        # this walk, and the one-shot latch so exactly one runs.jsonl
        # line lands per walk whatever terminal path fires.
        self._micros = 0
        self._run_recorded = False
        # The rescue ladder (`rescue.py`): the one rescue in flight (a
        # failed page check being recovered) and the walk-lifetime action
        # count the telemetry reports.
        self._rescue: rescue.State | None = None
        self._rescues_total = 0
        self._dismiss_labels: "tuple[str, ...] | None" = None  # lazy, per walk
        # Agent steps: the acting episode in flight (a pure-text call
        # holds no state — its outcome resolves against the node at the
        # cursor). Declared recovery holds NO state of its own either:
        # `spec.recovers` non-empty flips `rescue.plan` into declared
        # mode, riding the one rescue State/step/landed path.
        self._episode: _Episode | None = None
        # The pack's declared app chrome (back / dismiss) — the rescue
        # ladder's author-trusted prior knowledge.
        self._landmarks: dict[str, Landmark] = landmarks or {}
        # The reset rung's hands and its once-per-WALK latch: the start
        # page's own `open:` body first (it launches to exactly the page
        # the route starts at), else the pack's directory `open` macro
        # (enabled), else None — absent means the rung is skipped, the
        # ladder degrades, never breaks. The walk's boot init IS this
        # ladder: the first move's derived enter is the start page, so a
        # wrong screen at wake climbs back → force_quit → open before
        # anything else runs.
        self._open_macro: "str | None" = None
        for cand in filter(None, (spec.open_macro, OPEN_MACRO)):
            found = pack_macros.get(qualified_macro(app, cand))
            if found is not None and found.enabled:
                self._open_macro = qualified_macro(app, cand)
                break
        self._reset_used = False
        # The I6 one-shot: legs whose zero-gesture abort already earned
        # their single rescue-and-retry this walk.
        self._leg_retries: set[str] = set()
        # True once advance() was ever called — `abandon` records only
        # walks that actually started (a built-but-never-driven program
        # is not a run).
        self._started = False
        # The amount a payment leg actually fired with (consent is
        # consumed at fire, so this is the only place it survives to the
        # completed history line), and the once-read runs.jsonl rows for
        # the "usual brand" decide context (the memory-sections idiom:
        # nothing rewrites the file while the walk drives).
        self._paid: float | None = None
        self._history_rows: "list[dict] | None" = None
        # Terminal: the brief turn (handover or completion) was minted.
        # The NEXT advance is the permanent None the conductor drops the
        # program on — "quiet is permanent" now happens one turn later,
        # with the report in the transcript.
        self._done = False
        # A suspended CONFIRM's resume owes the user one thread read before
        # walk continues: the wake that resumes it may BE their cancel
        # reply (the IM banner wakes the device), and barreling on would
        # ignore it. Set only by _restore_suspended.
        self._confirm_check = False
        # Reconcile re-shop bound, per item index: one re-shop is a
        # failed add, a second means the pick's label will never match
        # its cart row — hand over instead of duplicating adds forever.
        self._reshops: dict[int, int] = {}
        # Decision state: recorded outputs (`{node.field}` refs read
        # them), per-node ask counts (max_visits bound), the journal line
        # the next synthesized note carries (record-don't-replay), the
        # decide-time screen/verdict the current step works from, and the
        # once-parsed memory.md sections (the file cannot change mid-walk
        # — the model isn't running while the program synthesizes turns).
        self._outputs: dict[str, str] = {}
        self._visits: dict[str, int] = {}
        self._journal: str | None = None
        self._screen: Screen | None = None
        self._verdict: Verdict | None = None
        self._memory: memory.Sections | None = None
        # The ledger walk: desired state + the loop cursor + the
        # reconciler's action budget. A value that fails to parse
        # degrades to None (fail-open) — the next_item node then hands
        # over; arm/activation validated theirs already.
        self._ledger: list[ledger.LedgerItem] | None = None
        self._item = 0
        self._rec_actions = 0
        inp = spec.ledger_input
        if inp is not None:
            try:
                self._ledger = ledger.parse_ledger(values.get(inp.name, ""))
            except PlaybookError as e:
                log.warning("ledger input %r unusable (%s)", inp.name, e)
        # The loop BODY head — where re-shop re-enters (the closer would
        # re-bind a stale pick label).
        self._loop_body_id: str | None = None
        loop = spec.loop
        if loop is not None:
            arm = CALLS[loop.call].loop_arm
            assert arm is not None
            self._loop_body_id = loop.routes[arm]
        if suspended is not None:
            # A resumed walk is ALSO whole at construction: the suspended
            # projection overlays the fresh state right here, so no
            # caller ever patches a program up afterwards.
            self._restore_suspended(suspended)
            log.info(
                "conductor: resuming suspended %s/%s at node %d (%s)",
                app,
                spec.name,
                self._idx + 1,
                "awaiting reply" if self._gate.awaiting else "walk",
            )

    def _suspended_dict(self, resume_idx: int) -> dict:
        """The suspension projection: walk state here, gate state via
        `_Gate.to_suspended` — each field list lives beside its fields."""
        return {
            "schema": SUSPENDED_SCHEMA,
            "app": self.app,
            "playbook": self.spec.name,
            "idx": resume_idx,
            "values": self.values,
            "outputs": self._outputs,
            "visits": self._visits,
            "ledger": (
                [it.to_dict() for it in self._ledger]
                if self._ledger is not None
                else None
            ),
            "item": self._item,
            **self._gate.to_suspended(),
        }

    def _restore_suspended(self, data: dict) -> None:
        idx = int(data["idx"])
        if not (0 <= idx <= len(self.spec.nodes)):
            # The spec changed under the suspension (edited shorter) — a stale
            # cursor must not fake a completion. Raising drops the suspension
            # (load_suspended is fail-open).
            raise PlaybookError(f"suspended idx {idx} is outside the playbook")
        self._idx = idx
        self._acted = True  # the suspended session acted; completion is real
        self._outputs = {str(k): str(v) for k, v in (data.get("outputs") or {}).items()}
        self._visits = {str(k): int(v) for k, v in (data.get("visits") or {}).items()}
        if data.get("ledger"):  # non-empty — [] would leave no valid cursor
            # The suspended ledger (statuses, labels, revisions applied)
            # supersedes the arm-value parse from __init__. The cursor
            # clamps HERE — external input — so every reader downstream
            # may index without bounds checks.
            self._ledger = [ledger.LedgerItem.from_dict(d) for d in data["ledger"]]
            self._item = min(max(int(data.get("item") or 0), 0), len(self._ledger) - 1)
        self._gate = _Gate.from_suspended(data)
        self._resumed = True
        # A suspended CONFIRM (message away, not awaiting a reply) resumes
        # mid-walk — read the thread for a cancel before continuing.
        self._confirm_check = (
            not self._gate.awaiting
            and bool(self._gate.ask)
            and self._idx < len(self.spec.nodes)
        )

    def advance(
        self, history: list[Message]
    ) -> "AssistantMessage | DecisionRequest | None":
        """The next synthesized turn; a DecisionRequest for the conductor
        to broker (feed the outcome back via ``resolve``); or None — hand
        over to the model. Never raises: a program bug degrades to a
        handover, not a crashed session (the model finishes the task
        either way)."""
        self._started = True
        try:
            step = self._advance(history)
        except Exception:
            log.exception("conductor: program crashed — handing over to the model")
            self._record_run("crashed", "program crashed")
            step = None
        return step

    def resolve(
        self, outcome: MicroOutcome | None
    ) -> "AssistantMessage | DecisionRequest | None":
        """Continue the walk with a micro-call's outcome (None = the call
        failed or was under-confident — hand over). Same return contract
        and same never-raises guarantee as ``advance``."""
        self._micros += 1
        try:
            step = self._resolve(outcome)
        except Exception:
            log.exception("conductor: program crashed — handing over to the model")
            self._record_run("crashed", "program crashed")
            step = None
        return step

    # ---- one advance ----

    def _advance(
        self, history: list[Message]
    ) -> "AssistantMessage | DecisionRequest | None":
        if self._done:
            # The brief turn was the walk's last word; its peek result is
            # ordinary history. Quiet from here — the conductor drops us.
            return None
        if self._turns.pending is None:
            if self._gate.awaiting:
                # Suspended-at-gate resume: go straight to reading the
                # thread — the ask was sent before the suspension.
                return self._synth_gate_peek()
            if self._confirm_check and self.channel is not None:
                # Suspended-CONFIRM resume: this wake may BE the user's
                # cancel reply — read the thread before the walk acts.
                self._confirm_check = False
                return self._synth(
                    "confirm-peek",
                    "conductor: checking the thread for a reply before continuing",
                    gesture_vocab.PEEK,
                    {},
                    channel=True,
                )
            # Observe before acting. The peek is also how a killed
            # session resumes: _locate fast-forwards past legs whose
            # verify page already matches, so completed gestures are
            # never replayed.
            return self._synth(
                "peek",
                "conductor: observing the screen to locate "
                f"{self.app}/{self.spec.name}",
                gesture_vocab.PEEK,
                {},
            )
        pending, result, failed = self._turns.settle(history)
        kind = pending.kind
        if failed is not None:
            if kind == "suspend":
                # The suspension file is already written; whether the
                # end_session was blocked or its result never arrived,
                # the session may run on — a dead walk must not
                # resurrect on the next wake.
                clear_suspended()
                return self._handover(f"{failed} — suspension dropped")
            if kind == "leg":
                view = self._failed_leg_view(history)
                if view is not None:
                    retry = self._retry_leg_after_abort(*view)
                    if retry is None:
                        retry = self._retry_leg_locked(*view)
                    if retry is not None:
                        return retry
            return self._handover(failed)
        assert result is not None  # settle: exactly one of result/failed
        # One reading, one verdict: channel-facing actions (declared at
        # the synth site) match against the thread; everything else
        # against the pack's own pages.
        in_channel = pending.channel
        self._screen = views.screen_of(result)
        self._verdict = match_screen(
            self._screen,
            self.channel.prints if in_channel and self.channel else self._prints,
        )
        if kind == "peek":
            if self._resumed:
                # A suspended walk trusts its stored cursor — the next
                # node's own checks judge whether the world still fits.
                self._resumed = False
            else:
                self._idx = self._locate(self._verdict)
        elif kind == "leg":
            node = self.spec.nodes[self._idx]
            assert isinstance(node, LegNode)
            if node.irreversible == "payment" and self._paid is not None:
                # The doctrine's purchase line (PERSISTENCE § Format),
                # harness-written the moment the payment leg's result
                # lands — whatever the verify check says next, money may
                # have moved, and the daily log is the cross-wake record.
                self._log_day(self._purchase_line())
            wrong = self._mismatch(self._verdict, page_id(self.app, node.verify))
            if wrong is not None:
                return self._rescue_or_handover(
                    node,
                    page_id(self.app, node.verify),
                    rescue.MODE_VERIFY,
                    f"leg {node.id!r} did not land on {node.verify!r} ({wrong})",
                )
            self._idx += 1
        elif kind == "swipe":
            # The scroll-for-more between re-asks: same decide, fresh
            # screen, bounded by max_visits in _request.
            node = self.spec.nodes[self._idx]
            assert isinstance(node, DecideNode)
            return self._request(node)
        elif kind in ("gate-sent", "confirm-sent"):
            wrong = self._thread_mismatch()
            if wrong is not None:
                return self._handover(
                    f"channel send did not land on the thread ({wrong})"
                )
            if self._gate.baseline:
                # A previous ask baselined this thread; anything the user
                # sent while the walk was off in the app (a revision
                # cycle, an earlier gate) lands here as new. A deny among
                # it must stop the walk NOW — overwriting the baseline
                # would swallow it forever. Deny only: an old confirm
                # word above the fresh ask is never treated as consent.
                stale = reply.new_incoming(
                    self._screen.rows,
                    self._gate.baseline,
                    self._gate.ask,
                    after_ask=False,
                )
                if reply.classify_all(stale) == "deny":
                    return self._deny()
            self._gate.baseline = {
                r.label.strip() for r in self._screen.rows if r.label.strip()
            }
            if kind == "confirm-sent":
                # CONFIRM: message away → suspend; the walk continues past
                # this node on the resuming wake.
                return self._suspend(resume_idx=self._idx + 1, awaiting=False)
            self._gate.awaiting = True
            self._gate.silence = 0
            self._gate.checks = 0
            return self._synth_wait()
        elif kind == "gate-wait":
            return self._synth_gate_peek()
        elif kind in ("gate-peek", "gate-open"):
            return self._gate_check()
        elif kind in ("confirm-peek", "confirm-open"):
            return self._confirm_check_step()
        elif kind in ("rec-peek", "rec-tap"):
            # The tap's own result screen is the re-read — no extra peek.
            return self._reconcile_step()
        elif kind == "leg-unlock":
            # The locked-leg retry's unlock landed — re-run the node at
            # the unchanged cursor; its own checks judge the fresh world.
            return self._next()
        elif kind in ("agent-tap", "agent-swipe"):
            # An episode action landed: its own result view is the next
            # turn's screen block.
            return self._episode_landed()
        elif kind.startswith("rescue-"):
            # Every rescue kind is `rescue-<rung>` minted by _rescue_act
            # (plus the reset pair's follow-up "rescue-open") — one
            # prefix, so a new rung cannot miss this dispatch arm. Same
            # rule as reconcile: the action's own result view is the
            # re-read the ladder judges.
            return self._rescue_landed(kind)
        elif kind == "suspend":
            # end_session was blocked/failed — the suspension file is already
            # written and would resurrect this walk after the model
            # finishes the task in-session. Drop it.
            clear_suspended()
            return self._handover(
                "suspend end_session was blocked — suspension dropped"
            )
        elif kind not in ("tap", "gate-return"):
            # A typo'd kind at a _synth site must fail loudly, never
            # silently walk the next node.
            return self._handover(f"unknown pending action kind {kind!r}")
        # "tap" / "gate-return": the cursor was routed at resolve (or
        # confirm) time; the target node's own checks judge the landing
        # in _next.
        return self._next()

    def _next(self) -> "AssistantMessage | DecisionRequest | None":
        """Walk the node at the cursor, holding this phase's line: legs
        and decisions only, own-pack pages only, enter verified when
        declared. Works from the stored screen/verdict — every path here
        observed one first."""
        verdict = self._verdict
        if verdict is None:
            return self._handover("no screen observed yet")
        nodes = self.spec.nodes
        if self._idx >= len(nodes):
            if not self._acted:
                # The opening locate fast-forwarded past EVERYTHING off
                # one page match — a coincidence (common pages recur),
                # not evidence the task ran. Hand over without consuming
                # the arm; the model verifies.
                return self._handover(
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
            return self._synth(
                "brief",
                brief.completion_brief(
                    self.app, self.spec.name, len(nodes), self._ledger
                ),
                gesture_vocab.PEEK,
                {},
            )
        node = nodes[self._idx]
        if isinstance(node, AgentNode):
            if not node.tools:
                # A pure-text call: no screen, no page contract — prompt
                # in, declared fields out.
                return self._agent_fields_request(node)
            gate = self._enter_gate(node)
            if gate is not None:
                return gate
            return self._episode_start(node)
        if isinstance(node, DecideNode):
            if CALLS[node.call].deterministic:
                # The program answers it itself — no prompt, no
                # confidence, no visit budget (the finite ledger is the
                # bound).
                return self._advance_item(node)
            return self._request(node)
        if isinstance(node, ReconcileNode):
            return self._start_reconcile(node)
        if isinstance(node, (ConfirmNode, HumanGateNode)):
            # Both ask nodes share the one precondition: a live send.
            if self.channel is None or self.channel.send is None:
                return self._handover(
                    "no channel send macro — record playbooks/channel to enable asks"
                )
            if isinstance(node, ConfirmNode):
                return self._start_confirm(node)
            return self._start_gate(node)
        if not isinstance(node, LegNode):
            return self._handover(
                f"node {node.id!r} is a {type(node).__name__} — not executable"
            )
        if "." in node.verify or "." in node.enter:
            return self._handover(
                f"leg {node.id!r} references a reserved built-in page — "
                "not supported in this phase"
            )
        gate = self._enter_gate(node)
        if gate is not None:
            return gate
        if node.irreversible == "payment":
            # Money fires only off a VERIFIED own-pack page: the ask was
            # sent on the IM thread and quotes the consented total, so an
            # unverified screen could satisfy the predicates with the
            # conductor's own ask bubble.
            if verdict.kind != "match" or not (verdict.page_id or "").startswith(
                f"{self.app}."
            ):
                return self._handover(
                    f"payment leg {node.id!r}: current screen is not a "
                    f"verified {self.app} page — money never fires blind"
                )
            # The fire-time money predicates — in code, on the current
            # screen, after the human's consent (`money.py` owns the
            # rules): staleness and the bound.
            assert self._screen is not None  # a matched verdict was read off it
            blocked = money.fire_block(
                consented=self._gate.consented,
                cap=self._gate.cap,
                screen=self._screen,
            )
            if blocked is not None:
                return self._handover(f"payment leg {node.id!r}: {blocked}")
        try:
            vals = self._values()
            inputs = {
                k: fill_refs(v, vals, where=f"leg {node.id!r} `with.{k}`")
                for k, v in node.args.items()
            }
        except PlaybookError as e:
            return self._handover(str(e))
        args: dict = {"name": qualified_macro(self.app, node.macro)}
        if inputs:
            args["inputs"] = inputs
        if node.irreversible == "payment":
            # Consent is CONSUMED by firing: a later payment leg needs
            # its own gate's fresh confirm, never this one's leftovers.
            # The amount survives only into the history line.
            self._paid = self._gate.consented
            self._gate.consented = None
            self._gate.quoted = None
        return self._synth(
            "leg",
            f"conductor: leg {node.id} ({self._idx + 1}/{len(nodes)}) — "
            f"macro {node.macro}, verify {node.verify}",
            gesture_vocab.RUN_MACRO,
            args,
        )

    # ---- decisions ----

    def _values(self) -> dict[str, str]:
        """Ref-resolution values, keyed by the dotted ref spellings: the
        walk's inputs under `inputs.<name>`, every recorded decision
        output under `node.field`, plus the loop-scoped `{item.*}` slots
        when a ledger walk has a current item. The roots can never
        collide: the parser reserves `inputs`/`item` as node ids."""
        vals = {**self._input_vals, **self._outputs}
        if self._ledger is not None:
            it = self._ledger[self._item]  # cursor valid by construction
            for f in LEDGER_FIELDS:
                vals[f"item.{f}"] = str(getattr(it, f))
        return vals

    def _request(self, node: DecideNode) -> "AssistantMessage | DecisionRequest | None":
        """One micro-call request off the decide-time screen — or a
        handover (None, typed as the shared step result) when the ask
        budget is spent or the walk has nothing to decide over."""
        # In a ledger walk the loop body's decide legitimately runs once
        # per item — the budget is per (node, item), or an 8-item list
        # could never finish under the default max_visits.
        key = node.id if self._ledger is None else f"{node.id}@{self._item}"
        self._visits[key] = self._visits.get(key, 0) + 1
        if self._visits[key] > node.max_visits:
            return self._handover(
                f"decide {node.id!r} exceeded max_visits ({node.max_visits})"
            )
        if self._screen is None:
            return self._handover(f"decide {node.id!r} has no screen to decide over")
        try:
            vals = self._values()
            args = {
                k: str(fill_refs(v, vals, where=f"decide {node.id!r}"))
                for k, v in node.args.items()
            }
        except PlaybookError as e:
            return self._handover(str(e))
        return build_request(
            node.call, node.id, node.outcomes, args, self._screen, self._context(node)
        )

    def _context(self, node: DecideNode) -> str:
        """The declared context slices, assembled least-privilege:
        `inputs.*` from the walk's values; `memory.<slug>` pulls ONLY the
        matching `## <slug>` section of memory.md (see `memory.py` for
        the fail-closed contract). A ledger walk's current item rides as
        DEFAULT context — authors kept forgetting to template
        `{item.query}` into `criteria`, the exact starved-subagent
        failure, so the walk supplies it itself."""
        parts: list[str] = []
        if self._ledger is not None:
            it = self._ledger[self._item]  # cursor valid by construction
            parts.append(f"current buying-list item: {it.query} (want {it.qty})")
        if self._history_rows is None:
            self._history_rows = walklog.load()
        prior = walklog.last_picks(self._history_rows, self.app, self.spec.name)
        if prior:
            picked = "; ".join(f"{q} → {label}" for q, label in prior.items())
            parts.append(f"previously picked (last completed run): {picked}")
        memory_slices: list[str] = []
        for entry in node.context:
            root, _, member = entry.partition(".")
            if root == "inputs":
                parts.append(f"{member}: {self.values.get(member, '')}")
            else:
                memory_slices.append(member)
        if memory_slices:
            if self._memory is None:
                # Parsed once for the walk: nothing can rewrite the file
                # mid-program (the model isn't running).
                self._memory = memory.read_sections()
            sliced = memory.match_sections(self._memory, memory_slices)
            if sliced:
                parts.append(f"memory (for {', '.join(memory_slices)}):\n{sliced}")
            else:
                log.info(
                    "memory slice(s) %s: no matching `## <slug>` section in "
                    "memory.md — decision runs without memory context",
                    memory_slices,
                )
        return "\n".join(parts)

    def _resolve(
        self, outcome: MicroOutcome | None
    ) -> "AssistantMessage | DecisionRequest | None":
        if self._rescue is not None and self._rescue.micro_pending:
            return self._resolve_rescue(outcome)
        if self._gate.revise_pending:
            return self._apply_revision(outcome)
        if self._gate.llm_reply is not None:
            # The gate's LLM tier came back. None/unclear → keep waiting
            # (this round's check was already counted in _gate_check).
            judged = self._gate.llm_reply
            batch = self._gate.llm_batch
            self._gate.llm_reply = None
            self._gate.llm_batch = []
            if outcome is not None:
                # Baseline only what was actually judged: a provider
                # failure leaves the batch visible to the next round —
                # the user's reply is never silently swallowed.
                self._gate.baseline |= set(batch)
            if outcome is not None and outcome.out == "deny":
                return self._deny()
            if outcome is not None and outcome.out == "revise":
                node = self.spec.nodes[self._idx]
                assert isinstance(node, HumanGateNode)
                if node.revise is not None and self._ledger is not None:
                    # A ledger playbook absorbs the change itself:
                    # revise_list rewrites the desired state, the loop
                    # shops the additions, RECONCILE fixes the rest,
                    # and the gate re-asks with the fresh total.
                    return self._start_revision(node, judged)
                # No ledger: a change request ("ok, but make it two
                # boxes") or a question is a TASK change, not a gate
                # verdict — the model takes over with the thread in the
                # transcript and acts on the user's words.
                return self._handover(
                    f"user asked for changes ({judged!r}) — read the "
                    "thread and adjust the order before any payment"
                )
            if outcome is not None and outcome.out == "confirm":
                return self._gate_confirmed()
            return self._synth_wait()
        if self._episode is not None:
            return self._episode_resolve(outcome)
        node = self.spec.nodes[self._idx]
        if isinstance(node, AgentNode) and not node.tools:
            return self._agent_fields_done(node, outcome)
        assert isinstance(node, DecideNode)
        if outcome is None:
            return self._handover(
                f"decide {node.id!r}: micro-call failed or under-confident"
            )
        self._journal = (
            f"decided {node.id}: {outcome.out} — {outcome.reason} "
            f"({outcome.confidence:.2f})"
        )
        target = node.routes[outcome.out]
        if target == ESCALATE:
            return self._handover(
                f"decide {node.id!r} routed {outcome.out!r} → escalate "
                f"({outcome.reason})"
            )
        decl = CALLS[node.call]
        if outcome.picked is not None:
            # Act on the pick: record the declared payload, then the
            # conductor's own tap at the chosen row — the one gesture a
            # rehearsed macro cannot know (its bbox exists only on the
            # decide-time screen). The routed node's enter check judges
            # the landing.
            for fld in decl.payload:
                self._outputs[f"{node.id}.{fld}"] = outcome.picked.key
            self._idx = self._ids[target]
            return self._synth(
                "tap",
                f"conductor: tapping picked {outcome.picked.key!r}",
                "tap",
                {"bbox": list(outcome.picked.bbox)},
            )
        if outcome.out == decl.reask_arm and target == node.id:
            # The sanctioned self-loop (the parser lints self-routes to
            # exactly this arm): swipe up (page scrolls down), then
            # re-ask over whatever the fresh screen shows.
            return self._synth_swipe(
                "swipe", "conductor: scrolling for more candidates", "up"
            )
        self._idx = self._ids[target]
        return self._next()

    # ---- the ledger loop + reconciliation ----

    def _advance_item(
        self, node: DecideNode
    ) -> "AssistantMessage | DecisionRequest | None":
        """The next_item closer: record the pick for the item just
        shopped, move the cursor to the next pending item (`next`, the
        sanctioned backward edge) or fall through spent (`done`)."""
        if self._ledger is None:
            return self._handover(f"{NEXT_ITEM} {node.id!r}: no usable ledger")
        cur = self._ledger[self._item]
        if cur.status == "pending":
            # Idempotent: a re-entry (re-shop, revision) whose current
            # item is already picked must not re-bind a stale label.
            try:
                picked = fill_refs(
                    node.args.get("picked", ""),
                    self._values(),
                    where=f"{NEXT_ITEM} {node.id!r} `with.picked`",
                )
            except PlaybookError as e:
                return self._handover(str(e))
            cur.status, cur.label = "picked", picked.strip() or cur.query
        loop_arm = CALLS[node.call].loop_arm
        assert loop_arm is not None
        done_arm = next(o for o in node.outcomes if o != loop_arm)
        nxt = next(
            (
                i
                for i, it in enumerate(self._ledger)
                if it.status == "pending" and it.qty > 0
            ),
            None,
        )
        shopped = sum(1 for it in self._ledger if it.status == "picked")
        self._journal = f"ledger: {shopped}/{len(self._ledger)} items shopped"
        if nxt is None:
            target = node.routes[done_arm]
        else:
            self._item = nxt
            target = node.routes[loop_arm]
        if target == ESCALATE:
            return self._handover(f"{NEXT_ITEM} {node.id!r} routed to escalate")
        self._idx = self._ids[target]
        return self._next()

    # ---- agent steps (pure-text calls and acting episodes) ----

    def _agent_fields_request(
        self, node: AgentNode
    ) -> "DecisionRequest | AssistantMessage | None":
        """A no-tools agent: one call — the authored prompt (refs filled
        NOW, then frozen), the declared return fields as the contract."""
        try:
            prompt = str(
                fill_refs(
                    node.prompt, self._values(), where=f"agent {node.id!r} `prompt`"
                )
            )
        except PlaybookError as e:
            return self._handover(str(e))
        return DecisionRequest(
            call=AGENT_FIELDS,
            node_id=node.id,
            outcomes=(),
            args={
                "prompt": prompt,
                "fields": "\n".join(f"- {n}: {d}" for n, d in node.returns),
            },
            candidates=(),
            listing="",
            context="",
        )

    def _agent_fields_done(
        self, node: AgentNode, outcome: MicroOutcome | None
    ) -> "AssistantMessage | DecisionRequest | None":
        if outcome is None:
            return self._handover(f"agent {node.id!r}: call failed or under-confident")
        if outcome.out != AGENT_DONE:
            return self._handover(f"agent {node.id!r} escalated: {outcome.reason}")
        return self._agent_close(node, outcome)

    def _agent_close(
        self, node: AgentNode, outcome: MicroOutcome, *, calls: int = 0
    ) -> "AssistantMessage | DecisionRequest | None":
        """The one done tail both agent forms share: record the returns,
        journal, advance the cursor."""
        err = self._agent_returns(node, outcome)
        if err is not None:
            return self._handover(err)
        after = f" after {calls} calls" if calls else ""
        self._journal = f"agent {node.id}: done{after} — {outcome.reason}"
        self._idx += 1
        return self._next()

    def _agent_returns(self, node: AgentNode, outcome: MicroOutcome) -> str | None:
        """Record the declared return fields off a done outcome into the
        walk's outputs (`{node.field}` refs read them). The reason a
        missing field hands over instead of retrying: escalation is the
        default, never a guess."""
        payload = outcome.payload or {}
        missing = [n for n in node.return_fields if not payload.get(n, "").strip()]
        if missing:
            return (
                f"agent {node.id!r} answered done without return field(s) "
                f"{', '.join(missing)}"
            )
        for n in node.return_fields:
            self._outputs[f"{node.id}.{n}"] = payload[n].strip()
        return None

    def _episode_start(
        self, node: AgentNode
    ) -> "AssistantMessage | DecisionRequest | None":
        """Open an acting episode on the current (enter-verified) screen.
        The prompt's refs fill ONCE here; a payment episode additionally
        gets {ask.total} — the consented amount its adjacent gate bound —
        and stashes that bound for the per-tap predicates."""
        vals = self._values()
        consented: float | None = None
        if node.irreversible == "payment":
            if self._gate.consented is None:
                return self._handover(
                    f"agent {node.id!r}: payment episode without bound consent"
                )
            consented = self._gate.consented
            vals = {**vals, "ask.total": f"{consented:g}"}
        try:
            prompt = str(
                fill_refs(node.prompt, vals, where=f"agent {node.id!r} `prompt`")
            )
        except PlaybookError as e:
            return self._handover(str(e))
        self._episode = _Episode(node=node, consented=consented)
        self._episode_screen_block(prompt)
        return self._episode_request()

    def _episode_screen_block(self, prefix: str) -> None:
        """Rebuild the episode's answerable candidates off the CURRENT
        screen and set the pending user block. When the episode may tap,
        granted landmarks ride as candidates too (declared bbox,
        label-healed now) unless a live row already reads their name —
        the fresher bbox wins; with no tap tool there is nothing to
        answer with, so no candidates (and no locate work) at all."""
        ep = self._episode
        assert ep is not None and self._screen is not None
        rows = act_candidates(self._screen.rows)
        can_tap = "tap" in ep.node.tools
        give = ep.node.give if can_tap else ()
        row_keys = {r.key for r in rows}
        grants = tuple(
            Candidate(
                key=n,
                bbox=rescue.locate_landmark(self._landmarks[n], self._screen)[0],
            )
            for n in give
            if n in self._landmarks and n not in row_keys
        )
        ep.candidates = grants + rows if can_tap else ()
        parts = [prefix] if prefix else []
        parts.append(act_block("Current screen", rows))
        if give:
            parts.append("Granted landmarks (answer by name): " + ", ".join(give))
        ep.block = "\n".join(parts)

    def _episode_request(self) -> "AssistantMessage | DecisionRequest | None":
        ep = self._episode
        assert ep is not None
        node = ep.node
        ep.calls += 1
        if ep.calls > node.max_calls:
            self._episode = None
            return self._handover(
                f"agent {node.id!r} exceeded its call limit ({node.max_calls})"
            )
        verbs = [AGENT_DONE, ESCALATE]
        for tool, tool_verbs in AGENT_TOOL_VERBS.items():
            if tool in node.tools:
                verbs.extend(tool_verbs)
        return DecisionRequest(
            call=AGENT_ACT,
            node_id=node.id,
            outcomes=tuple(verbs),
            args={"block": ep.block},
            candidates=ep.candidates,
            listing="",
            context="",
            history=tuple(ep.history),
        )

    def _episode_resolve(
        self, outcome: MicroOutcome | None
    ) -> "AssistantMessage | DecisionRequest | None":
        ep = self._episode
        assert ep is not None
        node = ep.node
        if outcome is None:
            self._episode = None
            return self._handover(f"agent {node.id!r}: call failed or under-confident")
        # Settle the turn into the append-only history: the block that
        # asked, then the reply in the contract's canonical spelling
        # (micro re-serializes it, so repair-retry noise never enters
        # the replayed prefix).
        ep.history.append(("user", ep.block))
        ep.history.append(("assistant", canonical_reply(outcome)))
        if outcome.out == ESCALATE:
            self._episode = None
            return self._handover(f"agent {node.id!r} escalated: {outcome.reason}")
        if outcome.out == ACT_BACK:
            # The OS back edge-swipe — the reliable pop on iOS (a corner
            # chevron tap misses too often to trust the stylus with).
            ep.pending_desc = "went back"
            return self._synth(
                "agent-swipe",
                f"conductor: agent {node.id} — went back",
                gesture_vocab.GO_BACK,
                {},
            )
        if outcome.out in (ACT_SCROLL_DOWN, ACT_SCROLL_UP):
            ep.scrolls += 1
            if ep.scrolls > node.max_scrolls:
                self._episode = None
                return self._handover(
                    f"agent {node.id!r} exceeded its scroll limit ({node.max_scrolls})"
                )
            # scroll_down = see content further down = the swipe goes up.
            down = outcome.out == ACT_SCROLL_DOWN
            ep.pending_desc = "scrolled down" if down else "scrolled up"
            return self._synth_swipe(
                "agent-swipe",
                f"conductor: agent {node.id} — {ep.pending_desc}",
                "up" if down else "down",
            )
        if outcome.out == AGENT_DONE:
            # The exit contract is the matcher's, never the model's: done
            # counts only on the adjacent verify page. A rejection costs
            # one call and the episode continues.
            wrong = (
                self._mismatch(self._verdict, page_id(self.app, node.verify))
                if self._verdict is not None
                else "no screen observed"
            )
            if wrong is not None:
                ep.block = (
                    f"done rejected: the walk must be on {node.verify!r} — "
                    f"{wrong}. Continue toward the goal, or answer escalate."
                )
                return self._episode_request()
            self._episode = None
            return self._agent_close(node, outcome, calls=ep.calls)
        assert outcome.out == ACT_ARM and outcome.picked is not None
        if node.irreversible == "payment":
            # The purse stays with the walker: BOTH predicates re-run
            # before every tap the model proposes — a tap while no
            # visible amount equals the consented total, or with any
            # amount above it, is refused.
            assert self._screen is not None
            blocked = money.fire_block(
                consented=ep.consented, cap=self._gate.cap, screen=self._screen
            )
            if blocked is not None:
                self._episode = None
                return self._handover(f"payment agent {node.id!r}: {blocked}")
            if not ep.spent:
                # Consent is consumed by the FIRST fire — a later payment
                # needs its own gate (the leg rule, episode-shaped). The
                # amount survives into the history line via _paid.
                ep.spent = True
                self._paid = ep.consented
                self._gate.consented = None
                self._gate.quoted = None
        ep.pending_desc = f"tapped {outcome.picked.key!r}"
        return self._synth(
            "agent-tap",
            f"conductor: agent {node.id} — {ep.pending_desc}",
            "tap",
            {"bbox": list(outcome.picked.bbox)},
        )

    def _episode_landed(self) -> "AssistantMessage | DecisionRequest | None":
        """An episode action's own result view becomes the next turn's
        screen block — no extra peek."""
        ep = self._episode
        if ep is None:
            return self._handover("agent action landed with no episode in flight")
        node = ep.node
        assert self._screen is not None
        if node.irreversible == "payment" and ep.spent and not ep.logged:
            # The doctrine's purchase line, written the moment the first
            # consented tap's result lands — whatever happens next, money
            # may have moved, and the daily log is the cross-wake record.
            ep.logged = True
            self._log_day(self._purchase_line())
        if reads_as_locked(self._screen):
            self._episode = None
            return self._handover(f"agent {node.id!r}: phone locked mid-episode")
        self._episode_screen_block(f"[you {ep.pending_desc}]")
        return self._episode_request()

    def _start_reconcile(self, node: ReconcileNode) -> AssistantMessage:
        self._rec_actions = 0
        return self._synth(
            "rec-peek",
            f"conductor: reconciling the cart against the list ({node.id})",
            gesture_vocab.PEEK,
            {},
        )

    def _reconcile_step(self) -> "AssistantMessage | DecisionRequest | None":
        """One divergence, one action, re-read — the tap's own result
        screen is the verification (there is no other). The rules live
        in `reconcile.plan`; the walk owns the action budget, the
        re-shop counts, and the cursor."""
        node = self.spec.nodes[self._idx]
        assert isinstance(node, ReconcileNode)
        assert self._verdict is not None and self._screen is not None
        wrong = self._mismatch(self._verdict, page_id(self.app, node.page))
        if wrong is not None:
            return self._rescue_or_handover(
                node,
                page_id(self.app, node.page),
                rescue.MODE_RECONCILE,
                f"reconcile {node.id!r} expects the cart ({wrong})",
            )
        if self._ledger is None:
            return self._handover(f"reconcile {node.id!r}: no usable ledger")
        self._rec_actions += 1
        step = reconcile.plan(
            self._screen, self._ledger, self._reshops, self._rec_actions
        )
        if isinstance(step, reconcile.Blocked):
            return self._handover(f"reconcile {node.id!r}: {step.reason}")
        if isinstance(step, reconcile.Reshop):
            # Back into the loop for THIS item — the body head, not the
            # closer (the closer would re-bind a stale pick label).
            item = self._ledger[step.item_idx]
            self._reshops[step.item_idx] = self._reshops.get(step.item_idx, 0) + 1
            item.status, item.label = "pending", None
            self._item = step.item_idx
            self._journal = (
                f"reconcile: {item.query!r} missing from the cart — re-shopping"
            )
            assert self._loop_body_id is not None
            self._idx = self._ids[self._loop_body_id]
            return self._next()
        if isinstance(step, reconcile.Tap):
            return self._synth_rec_tap(step.el, step.note)
        if step is not None:
            # A new Step variant must fail loudly, never silently read
            # as convergence (same rule as unknown pending kinds).
            return self._handover(f"unknown reconcile step {step!r}")
        self._journal = "reconcile: cart matches the list"
        self._idx += 1
        return self._next()

    def _synth_rec_tap(self, el, note: str) -> AssistantMessage:
        return self._synth(
            "rec-tap",
            f"conductor: cart stepper — {note}",
            "tap",
            {"bbox": list(el.bbox)},
        )

    def _start_revision(
        self, node: HumanGateNode, judged: str
    ) -> "AssistantMessage | DecisionRequest | None":
        """Hand the quoted reply and the current list to revise_list —
        bounded: a user still revising after GATE_MAX_REVISIONS rounds
        is negotiating, and negotiation is the model's job."""
        assert self._ledger is not None
        self._gate.revisions += 1
        if self._gate.revisions > GATE_MAX_REVISIONS:
            return self._handover(
                f"gate {node.id!r}: {self._gate.revisions - 1} revisions "
                "already applied — read the thread and settle the order "
                "with the user"
            )
        self._gate.revise_pending = True
        assert self._screen is not None
        return build_request(
            REVISE_LIST,
            node.id,
            (),
            {
                "ask": self._gate.ask,
                "reply": judged,
                "ledger": json.dumps(
                    [{"query": it.query, "qty": it.qty} for it in self._ledger],
                    ensure_ascii=False,
                ),
            },
            self._screen,
        )

    def _apply_revision(
        self, outcome: MicroOutcome | None
    ) -> "AssistantMessage | DecisionRequest | None":
        """The revised desired state, code-validated, merged onto the
        walk's ledger: known queries keep their shopping progress (the
        reconciler moves their quantities), new queries enter pending,
        dropped queries go to qty 0 (the reconciler steps them out).
        Then back into the loop via the gate's revise target — consent
        is void until the gate re-asks over the new total."""
        node = self.spec.nodes[self._idx]
        assert isinstance(node, HumanGateNode) and node.revise is not None
        assert self._ledger is not None
        self._gate.revise_pending = False
        if outcome is None or outcome.out != "updated" or not outcome.payload:
            return self._handover(
                "the revision reply could not be applied — read the thread "
                "and adjust the order before any payment"
            )
        try:
            revised = ledger.parse_ledger(outcome.payload["ledger"], allow_zero=True)
        except PlaybookError as e:
            return self._handover(f"revised list unusable ({e}) — read the thread")
        if all(r.qty == 0 for r in revised):
            # "Remove everything" is a cancellation, not a revision.
            return self._deny()
        # The merge rules (progress kept, drops to 0, rephrases matched,
        # and why there is no `_query_present` second key here) live with
        # the list: `ledger.merge_revision`.
        ledger.merge_revision(self._ledger, revised)
        self._gate.consented = None  # the old ask no longer covers the order
        self._gate.awaiting = False
        self._journal = f"revision applied: {ledger.describe(self._ledger)}"
        self._idx = self._ids[node.revise]
        # The reply was read on the IM thread — re-enter the app before
        # the loop shops (the parser lints `revise` requires `return:`).
        if node.return_macro is not None:
            return self._synth(
                "gate-return",
                f"conductor: back to the app via {node.return_macro}",
                gesture_vocab.RUN_MACRO,
                {"name": qualified_macro(self.app, node.return_macro)},
            )
        return self._next()

    # ---- asks, the gate, suspending ----

    def _start_confirm(self, node: ConfirmNode) -> "AssistantMessage | None":
        # `message:` verbatim, refs filled — the playbook owns every
        # word the user reads (only its author knows their language),
        # and a CONFIRM has no consent contract to protect (any wake
        # resumes the walk).
        try:
            text = str(
                fill_refs(
                    node.message,
                    self._values(),
                    where=f"confirm {node.id!r} `message`",
                )
            )
        except PlaybookError as e:
            return self._handover(str(e))
        self._gate.ask = text
        return self._synth_send("confirm-sent", text)

    def _start_gate(self, node: HumanGateNode) -> "AssistantMessage | None":
        # All prose is the playbook's; code fills the consent slots the
        # parser required. For a payment ask: read the sheet NOW — the
        # ask quotes it via {ask.total}, and consent binds to exactly
        # this number (the ask IS the consent record).
        template, values = node.message, self._values()
        if node.gate == "payment":
            # The ask IS the consent record — its total must be read off
            # a VERIFIED own-pack page (the sheet the lints put before
            # the gate), never whatever screen happened to come last.
            verdict = self._verdict
            if (
                verdict is None
                or verdict.kind != "match"
                or not (verdict.page_id or "").startswith(f"{self.app}.")
            ):
                return self._handover(
                    f"gate {node.id!r}: the total must be quoted off a "
                    f"verified {self.app} page — refusing to ask blind"
                )
            cap = None
            if self.spec.mandate is not None:
                # A `budget:` is optional now — without one the consented
                # total IS the bound (`money.fire_block` treats cap None
                # exactly so), and there is no over-budget branch.
                cap = money.mandate_cap(self.spec.mandate, self.values)
                if cap is None:
                    return self._handover(
                        f"gate {node.id!r}: mandate cap could not be resolved"
                    )
            amts = money.amounts(self._screen) if self._screen else []
            if not amts:
                return self._handover(
                    f"gate {node.id!r}: no total readable on the sheet"
                )
            quoted = max(amts)
            self._gate.quoted, self._gate.cap = quoted, cap
            values = {**values, "ask.total": f"{quoted:g}"}
            if cap is not None:
                values["ask.cap"] = f"{cap:g}"
                if quoted > cap:
                    # Over-cap (ruled): the SAME plain-consent reply opens
                    # the gate, but the breach must be disclosed — the
                    # parser guarantees this template quotes total AND cap.
                    assert node.over_message is not None
                    template = node.over_message
        try:
            text = str(fill_refs(template, values, where=f"gate {node.id!r} `message`"))
        except PlaybookError as e:
            return self._handover(str(e))
        self._gate.ask = text
        self._gate.tried_open = False
        return self._synth_send("gate-sent", text)

    def _gate_check(self) -> "AssistantMessage | DecisionRequest | None":
        """One reply-evaluation round over the freshly peeked thread —
        the ruled tiers: new-message precondition, word match, LLM."""
        node = self.spec.nodes[self._idx]
        assert isinstance(node, HumanGateNode)
        wrong = self._thread_mismatch()
        if wrong is not None:
            if self.channel and self.channel.open and not self._gate.tried_open:
                self._gate.tried_open = True
                return self._synth(
                    "gate-open",
                    "conductor: reopening the user thread",
                    gesture_vocab.RUN_MACRO,
                    {"name": self.channel.open},
                    channel=True,
                )
            return self._handover(f"cannot reach the user thread ({wrong})")
        assert self._screen is not None
        new = reply.new_incoming(self._screen.rows, self._gate.baseline, self._gate.ask)
        if not new:
            self._gate.silence += 1
            if self._gate.silence >= SILENCE_ROUNDS:
                return self._suspend(resume_idx=self._idx, awaiting=True)
            return self._synth_wait()
        verdict = reply.classify_all(new)
        if verdict == "deny":
            return self._deny()
        if verdict == "confirm":
            return self._gate_confirmed()
        # Unclear → the LLM tier, bounded; a batch is baselined ONLY
        # when a judgment actually arrives (see _resolve) — the round
        # that exhausts the budget suspends with the batch still unread,
        # and a failed judgment leaves it for the next round, so the
        # reply is judged once, eventually.
        self._gate.checks += 1
        if self._gate.checks > GATE_MAX_CHECKS:
            return self._suspend(resume_idx=self._idx, awaiting=True)
        joined = " / ".join(new)
        self._gate.llm_reply = joined
        self._gate.llm_batch = new
        # Empty outcomes: confirm_reply's answer space is fixed whole in its
        # _SPECS row, never caller-supplied.
        return build_request(
            CONFIRM_REPLY,
            node.id,
            (),
            {"ask": self._gate.ask, "reply": joined},
            self._screen,
        )

    def _deny(self) -> "AssistantMessage | None":
        """The one deny disposition — both reply tiers land here: no
        re-asks, and no second chance this session."""
        return self._handover(
            "user declined the ask — acknowledge them, back out of any "
            "open checkout or cart state this task created, and wrap up"
        )

    def _gate_confirmed(self) -> "AssistantMessage | DecisionRequest | None":
        node = self.spec.nodes[self._idx]
        assert isinstance(node, HumanGateNode)
        # Informed consent binds the money predicates: the quoted total
        # becomes the consented one (over-cap included — ruled).
        self._gate.consented = self._gate.quoted
        self._gate.awaiting = False
        self._journal = f"user confirmed {node.gate}"
        self._idx += 1
        if node.return_macro is not None:
            return self._synth(
                "gate-return",
                f"conductor: back to the app via {node.return_macro}",
                gesture_vocab.RUN_MACRO,
                {"name": qualified_macro(self.app, node.return_macro)},
            )
        return self._next()

    def _confirm_check_step(self) -> "AssistantMessage | None":
        """The suspended CONFIRM resume's one thread read: a deny among the
        replies since the confirm's baseline ends the walk (the user
        cancelled into a fire-and-forget message — honor it); anything
        else falls through to the normal resume peek. Word tier only —
        a CONFIRM has no consent contract, so an unclear reply is the
        model's to read from the transcript if the walk hands over."""
        wrong = self._thread_mismatch()
        if wrong is None:
            assert self._screen is not None
            new = reply.new_incoming(
                self._screen.rows, self._gate.baseline, self._gate.ask
            )
            if reply.classify_all(new) == "deny":
                return self._deny()
        elif self.channel and self.channel.open and not self._gate.tried_open:
            self._gate.tried_open = True
            return self._synth(
                "confirm-open",
                "conductor: reopening the user thread",
                gesture_vocab.RUN_MACRO,
                {"name": self.channel.open},
                channel=True,
            )
        else:
            log.info("conductor: confirm reply check skipped — %s", wrong)
        # No cancel (or no thread): resume the walk with a fresh plain
        # peek — _resumed is still set, so the stored cursor is trusted.
        return self._synth(
            "peek",
            f"conductor: observing the screen to locate {self.app}/{self.spec.name}",
            gesture_vocab.PEEK,
            {},
        )

    def _thread_mismatch(self) -> str | None:
        if self.channel is None:
            return "no channel pack"
        assert self._verdict is not None
        return self._mismatch(self._verdict, THREAD_ID)

    def _synth_send(self, kind: str, text: str) -> AssistantMessage:
        assert self.channel is not None and self.channel.send is not None
        return self._synth(
            kind,
            "conductor: messaging the user",
            gesture_vocab.RUN_MACRO,
            {"name": self.channel.send, "inputs": {"message": text}},
            channel=True,
        )

    def _synth_swipe(self, kind: str, note: str, direction: str) -> AssistantMessage:
        """The one scroll gesture — a mid-screen swipe in `direction` —
        both re-ask forms ride: the decide's scroll-for-more and an
        episode's scroll verb."""
        return self._synth(
            kind,
            note,
            gesture_vocab.SWIPE,
            {"bbox": list(SCROLL_BBOX), "direction": direction},
        )

    def _synth_wait(self) -> AssistantMessage:
        return self._synth(
            "gate-wait",
            "conductor: waiting for the user's reply",
            "wait",
            {"seconds": GATE_WAIT_SECONDS},
        )

    def _synth_gate_peek(self) -> AssistantMessage:
        return self._synth(
            "gate-peek",
            "conductor: checking for a reply",
            gesture_vocab.PEEK,
            {},
            channel=True,
        )

    def _suspend(self, *, resume_idx: int, awaiting: bool) -> AssistantMessage:
        """Write the suspended state and close the session WAIT. No job is
        synthesized: a WAIT without a session-created job auto-schedules
        the follow-up alarm (`contract.drive`) — the file, not the job,
        is what resumes the walk on ANY next wake."""
        self._gate.awaiting = awaiting
        write_json_atomic(suspended_path(), self._suspended_dict(resume_idx))
        recap = f"waiting for the user's reply on {self.app}/{self.spec.name}"
        self._record_run("suspended", recap)
        # The close-routine's log line, harness-written (the
        # `log_external_stop` precedent): the conductor synthesizes
        # end_session directly, so the model never runs this wake — with
        # no per-step logs either, the suspension would be invisible to
        # the next wake's memory window.
        self._log_day(
            f"conductor: {self.app}/{self.spec.name} suspended — {recap}; "
            "any wake resumes it"
        )
        return self._synth(
            "suspend",
            f"conductor: suspending — {recap}",
            "end_session",
            {"status": SUSPEND_STATUS, "recap": recap},
        )

    # ---- the rescue ladder ----

    def _rescue_or_handover(
        self, node, expected_id: str, mode: str, reason: str
    ) -> "AssistantMessage | DecisionRequest | None":
        """Try the ladder before the model: a mechanical deviation
        (popup, lock, wandered deeper) is recoverable in code toward the
        page the frozen cursor already requires — the cursor, decision
        outputs, ledger, and consent are untouched throughout (I1).
        Never with consent bound, mid-gate, or for an irreversible leg
        (I3): money keeps the hard handover."""
        if (
            self._gate.consented is not None
            or self._gate.awaiting
            or (isinstance(node, (LegNode, AgentNode)) and node.irreversible)
        ):
            return self._handover(reason)
        if self._declared and not expected_id.startswith(f"{self.app}."):
            # Declared mode covers this pack's own pages only — a
            # reserved/channel target has no hand to declare.
            return self._handover(reason)
        if self._rescue is None:
            self._rescue = rescue.State(target=expected_id, mode=mode, reason=reason)
        return self._rescue_step()

    def _rescue_step(self) -> "AssistantMessage | DecisionRequest | None":
        """One rung: ask the ladder (`rescue.plan`) for the next action
        and synthesize it — or broker its ONE clear_overlay micro call;
        Exhausted → handover carrying the ORIGINAL check failure plus
        what was tried (the brief quotes it)."""
        st = self._rescue
        assert st is not None
        assert self._verdict is not None and self._screen is not None
        declared_mode = self._declared
        if self._dismiss_labels is None and not declared_mode:
            # The dismiss vocabulary feeds the ladder's overlay rung,
            # which declared mode never reaches — skip the disk read.
            self._dismiss_labels = rescue.load_dismiss(self.app)
        step = rescue.plan(
            self._verdict,
            self._screen,
            st.tries,
            # Declared mode budgets the WALK, not the engagement: a hand
            # that ran and re-located clears its State, so the fresh
            # engagement's own counter restarts at zero — the lifetime
            # count is what stops a relaunch loop (a splash ad on every
            # cold launch, say) from running forever.
            self._rescues_total if declared_mode else st.actions,
            self._dismiss_labels or (),
            can_reset=self._open_macro is not None and not self._reset_used,
            landmarks=self._landmarks,
            declared_mode=declared_mode,
            declared=self.spec.recovers.get(st.target.partition(".")[2]),
        )
        if isinstance(step, rescue.Settle):
            # Free — verification hygiene, not recovery: no action
            # budget, no rescue count, just the one re-peek marker.
            st.tries[rescue.RUNG_SETTLE] = 1
            return self._synth(
                f"rescue-{rescue.RUNG_SETTLE}",
                f"conductor: re-reading a possibly mid-transition screen "
                f"before rescuing toward {st.target}",
                gesture_vocab.PEEK,
                {},
            )
        if isinstance(step, rescue.AskDismiss):
            st.micro_pending = True
            st.tries[rescue.RUNG_MICRO] = 1  # a call, not a phone action
            return step.request
        if isinstance(step, rescue.Exhausted):
            note = st.note()
            self._rescue = None
            return self._handover(
                f"{st.reason} — {step.reason}"
                + (f" (rescue tried: {note})" if note else "")
            )
        if isinstance(step, rescue.Dismiss):
            return self._rescue_act(
                rescue.RUNG_DISMISS,
                f"conductor: overlay on {st.target} — {step.note}",
                "tap",
                {"bbox": list(step.bbox)},
            )
        if isinstance(step, rescue.Unlock):
            return self._rescue_act(
                rescue.RUNG_UNLOCK,
                "conductor: phone locked mid-walk — unlocking",
                gesture_vocab.UNLOCK_PHONE,
                {},
            )
        if isinstance(step, rescue.Reset):
            self._reset_used = True
            return self._rescue_act(
                rescue.RUNG_RESET,
                f"conductor: {self.app} unrecoverable in place — force "
                "quitting to restart it",
                gesture_vocab.FORCE_QUIT,
                {},
            )
        if isinstance(step, rescue.Hand):
            # The page's DECLARED hand (declared mode's one recovery
            # action) — the planner decides WHEN, the walk interprets
            # WHAT: a bare gesture, a landmark tap (label-healed), or
            # an argument-less macro.
            hand = step.hand
            note = f"conductor: recovering toward {st.target} via its declared hand"
            if hand.macro is not None:
                return self._rescue_act(
                    rescue.RUNG_HAND,
                    note,
                    gesture_vocab.RUN_MACRO,
                    {"name": qualified_macro(self.app, hand.macro)},
                )
            if hand.tool == "tap":
                landmark = self._landmarks.get(hand.landmark or "")
                if landmark is None:
                    self._rescue = None
                    return self._handover(
                        f"{st.reason} (recover landmark {hand.landmark!r} undeclared)"
                    )
                bbox, located = rescue.locate_landmark(landmark, self._screen)
                return self._rescue_act(
                    rescue.RUNG_HAND, note + located, "tap", {"bbox": list(bbox)}
                )
            assert hand.tool is not None
            return self._rescue_act(rescue.RUNG_HAND, note, hand.tool, {})
        assert isinstance(step, rescue.Back)
        if step.landmark is not None:
            # The app's OWN back affordance, located by its label first
            # (the labeled-target heal, conductor-side) — more precise
            # than the generic gesture on apps whose chrome we know.
            bbox, located = rescue.locate_landmark(step.landmark, self._screen)
            return self._rescue_act(
                rescue.RUNG_BACK,
                f"conductor: backing out via the app's back landmark toward "
                f"{st.target}{located}",
                "tap",
                {"bbox": list(bbox)},
            )
        return self._rescue_act(
            rescue.RUNG_BACK,
            f"conductor: backing out to reach {st.target}",
            gesture_vocab.GO_BACK,
            {},
        )

    def _rescue_act(
        self, rung: str, note: str, tool: str, args: dict
    ) -> AssistantMessage:
        """One rescue action's shared bookkeeping — the rung counter,
        both budgets, and the `rescue-<rung>` pending kind, in one home
        so no arm can forget a counter."""
        st = self._rescue
        assert st is not None
        st.tries[rung] = st.tries.get(rung, 0) + 1
        st.actions += 1
        self._rescues_total += 1
        return self._synth(f"rescue-{rung}", note, tool, args)

    def _failed_leg_view(
        self, history: list[Message]
    ) -> "tuple[LegNode, ToolResultMessage, Screen] | None":
        """A failed leg's own result view, harvested ONCE for the two
        one-shot retries below: the leg at the cursor (not yet retried
        this walk), its raw result, and the screen that result carries.
        None = no retry can apply; the caller hands over."""
        node = self.spec.nodes[self._idx]
        if not isinstance(node, LegNode) or node.id in self._leg_retries:
            return None
        pending = self._turns.pending
        raw = views.result_for(history, pending.call_id) if pending else None
        if raw is None:
            return None
        return node, raw, views.screen_of(raw)

    def _retry_leg_after_abort(
        self, node: LegNode, raw: ToolResultMessage, screen: Screen
    ) -> "AssistantMessage | DecisionRequest | None":
        """The I6 one-shot: a leg whose macro aborted BEFORE any gesture
        ran (the runner's marker) moved nothing — when the abort's own
        view reads as an overlay on a known page, rescue the page and
        re-run the leg, once per leg per walk. Everything else keeps the
        hard handover. None = not this case.

        The marker is read off the RAW result text, not the settle
        failure string — that one is clipped (`turns.MAX_ERROR_CHARS`)
        and the marker sits late in a header that grows with macro name
        and view notes."""
        if NO_GESTURES_NOTE not in views.text_of(raw):
            return None
        verdict = match_screen(screen, self._prints)
        if verdict.kind != "occluded" or verdict.page_id is None:
            return None
        self._leg_retries.add(node.id)
        self._turns.pending = None  # the abort is handled, not retried blind
        self._screen, self._verdict = screen, verdict
        return self._rescue_or_handover(
            node,
            verdict.page_id,
            rescue.MODE_ENTER,
            f"leg {node.id!r} aborted before any gesture under an overlay "
            f"on {verdict.page_id}",
        )

    def _retry_leg_locked(
        self, node: LegNode, raw: ToolResultMessage, screen: Screen
    ) -> "AssistantMessage | DecisionRequest | None":
        """The I6 retry's sibling: a leg whose macro FAILED while the
        screen reads locked (hint text or the cover's hero clock) moved
        nothing — a locked phone swallows gestures. Unlock and re-run
        the leg, once per leg per walk. The field case: the phone
        auto-locks during a long model call before the first gesture (a
        route opening with a pure-text agent reasons for tens of
        seconds). None = not this case. (The unlock's own result view
        is the next reading, so nothing here is worth matching.)"""
        if not reads_as_locked(screen):
            return None
        self._leg_retries.add(node.id)
        self._turns.pending = None  # the failure is handled, not retried blind
        self._journal = f"leg {node.id} ran against a locked phone"
        return self._synth(
            "leg-unlock",
            f"conductor: phone locked during leg {node.id} — unlocking to retry",
            gesture_vocab.UNLOCK_PHONE,
            {},
        )

    def _resolve_rescue(
        self, outcome: MicroOutcome | None
    ) -> "AssistantMessage | DecisionRequest | None":
        """clear_overlay came back. A pick becomes the dismissal tap
        (its label held for learning until the page actually restores);
        none_safe or a failed call continues the ladder — the micro try
        is spent, so `plan` falls through to the back rung."""
        st = self._rescue
        assert st is not None
        st.micro_pending = False
        if outcome is not None and outcome.picked is not None:
            st.pending_learn = outcome.picked.key
            st.tries["dismiss"] = st.tries.get("dismiss", 0) + 1
            st.actions += 1
            self._rescues_total += 1
            return self._synth(
                "rescue-dismiss",
                f"conductor: overlay on {st.target} — "
                f"tapping {outcome.picked.key!r} (micro pick)",
                "tap",
                {"bbox": list(outcome.picked.bbox)},
            )
        return self._rescue_step()

    def _rescue_landed(self, kind: str) -> "AssistantMessage | DecisionRequest | None":
        """The rescue action's result view, judged. Restored → resume
        exactly where the walk stood: an interrupted `enter:` re-checks
        and runs its leg; an interrupted `verify:` is satisfied by the
        restored page (the macro already ran — never re-run it, I6); an
        interrupted reconcile re-reads the cart. Still off → next rung.
        The reset pair is different: force_quit's landing (the
        springboard) is never judged — the open macro follows — and the
        open's landing restarts the walk via `_locate` from the top,
        byte-for-byte the killed-session resume path (I2)."""
        st = self._rescue
        assert st is not None and self._verdict is not None
        if kind == f"rescue-{rescue.RUNG_RESET}":
            assert self._open_macro is not None
            return self._synth(
                "rescue-open",
                f"conductor: reopening {self.app} via {self._open_macro}",
                gesture_vocab.RUN_MACRO,
                {"name": self._open_macro},
            )
        if kind == "rescue-open":
            self._journal = f"rescue: {self.app} reset ({st.note()})"
            self._rescue = None
            self._idx = self._locate(self._verdict)
            return self._next()
        if not self._verdict.matches(st.target):
            if kind == f"rescue-{rescue.RUNG_HAND}":
                # The declared hand ran and the page still does not read:
                # re-locate on the route and walk from wherever reads —
                # the start move re-runs when the locate lands at the
                # top, which is a force_quit hand's whole point.
                # Recurrence is bounded by the walk-wide budget.
                self._journal = f"recover hand ran — walking again toward {st.target}"
                self._rescue = None
                self._idx = self._locate(self._verdict)
                return self._next()
            st.pending_learn = None  # that pick did not restore — unlearned
            return self._rescue_step()
        if st.pending_learn:
            # The micro tier's pick restored the page — the free tier
            # handles this popup next time (deny-gated at use anyway).
            rescue.learn_dismiss(self.app, st.pending_learn)
            self._dismiss_labels = None  # re-read on the next rescue
        self._journal = f"rescue: restored {st.target} ({st.note()})"
        self._rescue = None
        if st.mode == rescue.MODE_VERIFY:
            self._idx += 1
        if st.mode == rescue.MODE_RECONCILE:
            return self._reconcile_step()
        return self._next()

    # ---- page identity ----

    def _text_agent_settled(self, node) -> bool:
        """A pure-text agent whose outputs are already recorded — the
        locate scan skips it rather than re-spending its call."""
        return (
            isinstance(node, AgentNode)
            and not node.tools
            and all(f"{node.id}.{f}" in self._outputs for f in node.return_fields)
        )

    @property
    def _declared(self) -> bool:
        """The playbook-with-`recover:` regime — `rescue.plan`'s declared
        mode, where a page's own hand replaces the hidden ladder."""
        return bool(self.spec.recovers)

    def _enter_gate(
        self, node: "LegNode | AgentNode"
    ) -> "AssistantMessage | DecisionRequest | None":
        """The one enter-page guard legs and acting agents share: None
        when the node has no enter (a `start` leg runs unconditionally)
        or the page reads; else the recovery/handover step."""
        if not node.enter:
            return None
        kind = "agent" if isinstance(node, AgentNode) else "leg"
        assert self._verdict is not None
        expected = page_id(self.app, node.enter)
        wrong = self._mismatch(self._verdict, expected)
        if wrong is None:
            return None
        return self._rescue_or_handover(
            node,
            expected,
            rescue.MODE_ENTER,
            f"{kind} {node.id!r} expects page {node.enter!r} ({wrong})",
        )

    def _mismatch(self, verdict: Verdict, expected_id: str) -> str | None:
        """None when the verdict is a match on the full `expected_id`
        (`app.page`); else a short reason — pack pages and the channel
        thread judged by one spelling."""
        if verdict.kind == "match" and verdict.page_id == expected_id:
            return None
        seen = verdict.page_id or "no known page"
        return f"screen reads as {verdict.kind}: {seen} — {verdict.detail}"

    def _locate(self, verdict: Verdict) -> int:
        """Cursor for the current screen: just past the LAST leg of the
        playbook's LEADING LEG RUN whose verify page matches it (that
        page proves the leg's outcome holds), else the top. The scan
        stops at the first non-leg node: a page match proves navigation
        happened, never that a decision, loop pass, reconcile, or ask
        ran — fast-forwarding past work off one page coincidence would
        quote a gate total for a list that was never shopped."""
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
            if not isinstance(node, LegNode):
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

    # ---- synthesis ----

    def _synth(
        self, kind: str, summary: str, tool: str, args: dict, *, channel: bool = False
    ) -> AssistantMessage:
        """The walk's synthesized turn: fold in anything journaled, note
        whether this one touched the phone, then mint it (`turns.py`
        owns the shape and the call-id convention).

        A fresh decision outcome journals into this note's summary
        (record-don't-replay: the transcript carries the decision, never
        a re-ask), and any non-peek tool marks the walk as having acted —
        a "completion" that never acted must not consume the arm."""
        if self._journal is not None:
            summary = f"{summary} | {self._journal}"
            self._journal = None
        if tool != gesture_vocab.PEEK:
            self._acted = True
        return self._turns.synth(kind, summary, tool, args, channel=channel)

    def _handover(self, reason: str) -> AssistantMessage | None:
        """The walk's exit: log and record, then mint the ONE final
        synthesized [note, peek] brief turn (`brief.walk_brief`) — the
        distilled report the model resumes from, instead of the log-only
        reason it used to never see. `_done` makes the NEXT advance the
        permanent None the conductor drops the program on. Call sites
        keep reading `return self._handover(...)`."""
        log.warning(
            "conductor: handing %s/%s over to the model — %s",
            self.app,
            self.spec.name,
            reason,
        )
        self._record_run("handover", reason)
        self._done = True
        nodes = self.spec.nodes
        return self._synth(
            "brief",
            brief.walk_brief(
                reason,
                app=self.app,
                playbook=self.spec.name,
                node=nodes[self._idx].id if self._idx < len(nodes) else None,
                idx=self._idx,
                nodes=len(nodes),
                outputs=self._outputs,
                ledger_items=self._ledger,
                consented=self._gate.consented,
            ),
            gesture_vocab.PEEK,
            {},
        )

    def abandon(self) -> None:
        """The killed-session record — `log_external_stop`'s twin for
        walks. The engine's breadcrumb needs a drafted plan, which a
        walk never has (its plan is the playbook and its recovery is
        the locate peek), so the plugin calls this from session
        teardown instead: one telemetry row, plus the daily-log
        breadcrumb when the walk actually moved the phone. Latched like
        every terminal moment — a walk that already closed is a no-op,
        and so is one that never started."""
        if not self._started or self._run_recorded:
            return
        nodes = self.spec.nodes
        node = nodes[self._idx].id if self._idx < len(nodes) else "(end)"
        self._record_run("abandoned", "session ended mid-walk")
        if self._acted:
            self._log_day(
                f"conductor: {self.app}/{self.spec.name} cut short mid-walk "
                f"at node {node} — the next wake re-locates from the screen"
            )

    def _purchase_line(self) -> str:
        """The purchase entry, doctrine format ("merchant, brand, spec,
        qty, price"): the app is the merchant, the ledger's picked
        labels are the brands, the consented amount the price."""
        assert self._paid is not None
        picked = ", ".join(
            f"{it.label or it.query} ×{it.qty}"
            for it in (self._ledger or [])
            if it.status == "picked"
        )
        return (
            f"conductor: {self.app}: paid ¥{self._paid:g} "
            f"(playbook {self.app}/{self.spec.name})"
            + (f" — {picked}" if picked else "")
        )

    def _log_day(self, entry: str) -> None:
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
        nodes = self.spec.nodes
        picks = {
            it.query: it.label
            for it in (self._ledger or [])
            if it.status == "picked" and it.label
        }
        walklog.record(
            app=self.app,
            playbook=self.spec.name,
            outcome=outcome,
            idx=self._idx,
            nodes=len(nodes),
            node=nodes[self._idx].id if self._idx < len(nodes) else None,
            reason=reason,
            micros=self._micros,
            rescues=self._rescues_total,
            values=self.values,
            total=self._paid,
            picks=picks or None,
        )
