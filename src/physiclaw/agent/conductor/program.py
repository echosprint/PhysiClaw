"""One playbook mid-walk — the conductor's synthesized turns.

A `Program` is one playbook being executed. The conductor asks it for
each turn; it answers with a synthesized ``[note, one-other]`` assistant
turn (a LEG as ``run_macro``; the opening ``peek``; a decision's own
tap/swipe primitive), with a ``DecisionRequest`` the conductor brokers
through the micro-caller and feeds back via ``resolve()``, or with
``None`` — "I hand over". ``None`` is permanent for the session: the
conductor goes quiet and the model takes over with the transcript as the
handoff, because every synthesized turn and every tool result is
ordinary history the model can read.

The walk executes every node type, strictly:

  - Before a leg: its ``enter:`` page (when declared) must match the
    current screen. After a leg: its ``verify:`` page must match the
    screen the macro result carries. Anything else — wrong page,
    occluded, unknown, a blocked or errored call, a reserved ``ios.*``
    page — hands over. No retries, no recovery legs yet.
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
in `suspension.py`; the domain helpers in `channel.py`, `ledger.py`,
`memory.py`, `views.py`.
"""

import json
import logging
from dataclasses import dataclass, field

from physiclaw.agent.conductor import ledger, memory, reply, views
from physiclaw.agent.conductor.calls import CALLS, ESCALATE, NEXT_ITEM
from physiclaw.agent.conductor.channel import Channel
from physiclaw.agent.conductor.match import (
    PRICE_RE,
    Verdict,
    label_matches,
    match_screen,
    normalize,
)
from physiclaw.agent.conductor.micro import (
    CONFIRM_REPLY,
    REVISE_LIST,
    DecisionRequest,
    MicroOutcome,
    build_request,
)
from physiclaw.agent.conductor.pages import THREAD_ID, PagePrint, page_id
from physiclaw.agent.conductor.playbook import (
    GATE_MAX_CHECKS,
    GATE_MAX_REVISIONS,
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
from physiclaw.agent.conductor.suspension import (
    SUSPENDED_SCHEMA,
    clear_suspended,
    suspended_path,
)
from physiclaw.agent.conductor.turns import Turnsmith
from physiclaw.agent.engine.dto import AssistantMessage, Message
from physiclaw.agent.macros.model import Macro
from physiclaw.common import gesture_vocab
from physiclaw.common.listing import Screen
from physiclaw.common.logger import write_json_atomic

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

# The reconciler's action budget: every divergence costs at least one
# tap-and-reread action, so a converging cart finishes well under it —
# hitting it means the cart is NOT converging.
RECONCILE_MAX_ACTIONS = 16


def _amounts(screen: Screen) -> list[float]:
    """Every ¥/￥ amount visible on the screen — `match.PRICE_RE`, the
    one spelling of a currency amount, run over raw row labels; never a
    model."""
    out: list[float] = []
    for row in screen.rows:
        out.extend(float(m) for m in PRICE_RE.findall(row.label))
    return out


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


class Program:
    """One playbook mid-walk. Constructed per session (the engine builds
    it at wake), so the cursor state lives for exactly one attempt;
    persistence across wakes is the locate peek, not saved state."""

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
    ):
        self.app = app
        self.spec = spec
        self.values = values
        # The program's own qualified dispatch contribution — merged into
        # the wake registry by session_setup.
        self.pack_macros = pack_macros
        self._prints = prints
        self._ids = {n.id: i for i, n in enumerate(spec.nodes)}
        # Read by the engine: prompted decisions AND gate reply checks
        # ride the micro channel — either kind of node means wiring one
        # up. Deterministic calls (next_item) never prompt, so a
        # legs+loop-only playbook must not pay for a client.
        self.needs_micro = any(
            (isinstance(n, DecideNode) and not CALLS[n.call].deterministic)
            or isinstance(n, HumanGateNode)
            for n in spec.nodes
        )
        # The user-channel infrastructure (constructor-injected via
        # setup._build_program). None degrades to hand-over at the first
        # ask.
        self.channel = channel
        self._idx = 0
        # Turn minting + the one action in flight (`turns.py`) — the walk
        # keeps its own vocabulary (journal, acted) around it.
        self._turns = Turnsmith()
        self._gate = _Gate()
        self._resumed = False
        # True once any device-changing action was synthesized (peeks
        # don't count). A walk that "completes" without ever acting hit
        # a coincidental page match — that must not consume the arm.
        self._acted = False
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
            self._loop_body_id = loop.on[arm]
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
        try:
            step = self._advance(history)
        except Exception:
            log.exception("conductor: program crashed — handing over to the model")
            step = None
        return step

    def resolve(
        self, outcome: MicroOutcome | None
    ) -> "AssistantMessage | DecisionRequest | None":
        """Continue the walk with a micro-call's outcome (None = the call
        failed or was under-confident — hand over). Same return contract
        and same never-raises guarantee as ``advance``."""
        try:
            step = self._resolve(outcome)
        except Exception:
            log.exception("conductor: program crashed — handing over to the model")
            step = None
        return step

    # ---- one advance ----

    def _advance(
        self, history: list[Message]
    ) -> "AssistantMessage | DecisionRequest | None":
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
            wrong = self._mismatch(self._verdict, page_id(self.app, node.verify))
            if wrong is not None:
                return self._handover(
                    f"leg {node.id!r} did not land on {node.verify!r} ({wrong})"
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
            return None
        node = nodes[self._idx]
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
        if "." in node.verify or (node.enter is not None and "." in node.enter):
            return self._handover(
                f"leg {node.id!r} references a reserved built-in page — "
                "not supported in this phase"
            )
        if node.enter is not None:
            wrong = self._mismatch(verdict, page_id(self.app, node.enter))
            if wrong is not None:
                return self._handover(
                    f"leg {node.id!r} expects page {node.enter!r} ({wrong})"
                )
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
            # screen, after the human's consent: staleness (the sheet
            # must still show the consented total) and the bound.
            blocked = self._money_block(node)
            if blocked is not None:
                return self._handover(blocked)
        try:
            inputs = {
                k: fill_refs(v, self._values(), where=f"leg {node.id!r} `with.{k}`")
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
        """Ref-resolution values: the walk's inputs plus every recorded
        decision output (keyed `node.field`, exactly the dotted ref),
        plus the loop-scoped `{item.*}` slots when a ledger walk has a
        current item (last wins — the same shadowing the parser
        documents)."""
        vals = {**self.values, **self._outputs}
        if self._ledger is not None:
            it = self._ledger[self._item]  # cursor valid by construction
            vals["item.query"] = it.query
            vals["item.qty"] = str(it.qty)
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
            args = {
                k: str(fill_refs(v, self._values(), where=f"decide {node.id!r}"))
                for k, v in node.args.items()
            }
        except PlaybookError as e:
            return self._handover(str(e))
        return build_request(
            node.call, node.id, node.outs, args, self._screen, self._context(node)
        )

    def _context(self, node: DecideNode) -> str:
        """The declared context slices, assembled least-privilege:
        `inputs.*` from the walk's values; `memory.<slug>` pulls ONLY the
        matching `## <slug>` section of memory.md (see `memory.py` for
        the fail-closed contract)."""
        parts: list[str] = []
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
        node = self.spec.nodes[self._idx]
        assert isinstance(node, DecideNode)
        if outcome is None:
            return self._handover(
                f"decide {node.id!r}: micro-call failed or under-confident"
            )
        self._journal = (
            f"decided {node.id}: {outcome.out} — {outcome.reason} "
            f"({outcome.confidence:.2f})"
        )
        target = node.on[outcome.out]
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
            return self._synth(
                "swipe",
                "conductor: scrolling for more candidates",
                gesture_vocab.SWIPE,
                {"bbox": list(SCROLL_BBOX), "direction": "up"},
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
        done_arm = next(o for o in node.outs if o != loop_arm)
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
            target = node.on[done_arm]
        else:
            self._item = nxt
            target = node.on[loop_arm]
        if target == ESCALATE:
            return self._handover(f"{NEXT_ITEM} {node.id!r} routed to escalate")
        self._idx = self._ids[target]
        return self._next()

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
        screen is the verification (there is no other). Converged =
        every picked item reads at its wanted quantity; cart rows the
        ledger does not claim are LEFT ALONE (they may be the user's
        own — the conductor never destroys what it cannot attribute to
        itself)."""
        node = self.spec.nodes[self._idx]
        assert isinstance(node, ReconcileNode)
        assert self._verdict is not None and self._screen is not None
        wrong = self._mismatch(self._verdict, page_id(self.app, node.page))
        if wrong is not None:
            return self._handover(f"reconcile {node.id!r} expects the cart ({wrong})")
        if self._ledger is None:
            return self._handover(f"reconcile {node.id!r}: no usable ledger")
        self._rec_actions += 1
        if self._rec_actions > RECONCILE_MAX_ACTIONS:
            return self._handover(
                f"reconcile {node.id!r}: cart not converging after "
                f"{RECONCILE_MAX_ACTIONS} actions — list: "
                f"{ledger.describe(self._ledger)}"
            )
        assigned = ledger.assign_rows(self._screen, self._ledger)
        for idx, item in enumerate(self._ledger):
            if item.status != "picked":
                continue  # pending items are the loop's, reached via re-shop
            row = assigned[idx]
            if row is None:
                if item.qty <= 0:
                    continue  # removed and absent — converged
                # Picked but not in the cart: back into the loop for THIS
                # item — the body head, not the closer (the closer would
                # re-bind a stale pick label). Bounded per item: a second
                # miss means the pick's label will never read as a cart
                # row, and re-shopping again just duplicates real adds.
                self._reshops[idx] = self._reshops.get(idx, 0) + 1
                if self._reshops[idx] > 1:
                    return self._handover(
                        f"reconcile {node.id!r}: {item.query!r} still missing "
                        f"after a re-shop — list: {ledger.describe(self._ledger)}"
                    )
                item.status, item.label = "pending", None
                self._item = idx
                self._journal = (
                    f"reconcile: {item.query!r} missing from the cart — re-shopping"
                )
                assert self._loop_body_id is not None
                self._idx = self._ids[self._loop_body_id]
                return self._next()
            found = ledger.row_qty(self._screen, row)
            if found is None:
                return self._handover(
                    f"reconcile {node.id!r}: no readable qty/steppers beside "
                    f"{row.label!r} — list: {ledger.describe(self._ledger)}"
                )
            have, minus, plus = found
            if have < item.qty:
                return self._synth_rec_tap(plus, f"{item.query} {have}→{item.qty}")
            if have > item.qty:
                return self._synth_rec_tap(minus, f"{item.query} {have}→{item.qty}")
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
        # Old items keep their shopping progress (the reconciler moves
        # quantities; a query the revision dropped goes to 0); genuinely
        # new queries append pending. Old-ledger order is preserved —
        # friendlier to the `_item` cursor, and nothing reads revised
        # order. Matching is assign_rows' two passes — exact normalized
        # first, then fuzzy against the query OR the picked label:
        # revise_list is asked to echo unchanged items verbatim, but a
        # rephrase ("eggs" → "fresh eggs") must land on the existing
        # item, not zero out a correct cart row and re-shop the same
        # product. It does NOT carry assign_rows' `_query_present` second
        # key, on purpose: there both sides are screen text, so a rival's
        # row could be claimed outright, while here the revision's own
        # wording IS the thing being matched and demanding it contain the
        # old query would defeat the rephrase case above. A cross-match
        # here costs a wrong quantity, not a wrong tap, and the gate
        # re-earns consent (`_gate.consented = None`) before any payment.
        remaining = list(revised)
        matched: dict[int, ledger.LedgerItem] = {}
        for i, it in enumerate(self._ledger):  # pass 1: exact
            want = normalize(it.query)
            for r in remaining:
                if normalize(r.query) == want:
                    matched[i] = r
                    remaining.remove(r)
                    break
        for i, it in enumerate(self._ledger):  # pass 2: fuzzy
            if i in matched:
                continue
            for r in remaining:
                rn = normalize(r.query)
                if label_matches(normalize(it.query), rn, ()) or (
                    it.label and label_matches(normalize(it.label), rn, ())
                ):
                    matched[i] = r
                    remaining.remove(r)
                    break
        for i, it in enumerate(self._ledger):
            hit = matched.get(i)
            it.qty = hit.qty if hit is not None else 0
        self._ledger.extend(remaining)
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
        # parser required. For a payment gate: read the sheet NOW — the
        # ask quotes it via {gate.total}, and consent binds to exactly
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
            cap = self._mandate_cap()
            if cap is None:
                return self._handover(
                    f"gate {node.id!r}: mandate cap could not be resolved"
                )
            amts = _amounts(self._screen) if self._screen else []
            if not amts:
                return self._handover(
                    f"gate {node.id!r}: no total readable on the sheet"
                )
            quoted = max(amts)
            self._gate.quoted, self._gate.cap = quoted, cap
            values = {**values, "gate.total": f"{quoted:g}", "gate.cap": f"{cap:g}"}
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

    def _mandate_cap(self) -> float | None:
        m = self.spec.mandate
        if m is None:
            return None
        if isinstance(m.max_amount, float):
            return m.max_amount
        try:
            return float(self.values[m.max_amount.name])
        except (KeyError, ValueError):
            return None

    def _money_block(self, node: LegNode) -> str | None:
        """The two fire-time predicates. None = pay; else the logged
        reason to block and hand over."""
        if self._gate.consented is None:
            return f"payment leg {node.id!r} reached without a confirmed total"
        assert self._screen is not None
        consented = self._gate.consented
        amts = _amounts(self._screen)
        if not any(abs(a - consented) < 0.01 for a in amts):
            return (
                f"sheet changed after consent: confirmed ¥{consented:g}, "
                f"now sees {amts or 'no amounts'}"
            )
        bound = max(self._gate.cap or 0.0, consented)
        over = [a for a in amts if a > bound + 0.005]
        if over:
            return f"amount(s) {over} exceed the consented bound ¥{bound:g}"
        return None

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
        # Empty outs: confirm_reply's answer space is fixed whole in its
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
        return self._synth(
            "suspend",
            f"conductor: suspending — {recap}",
            "end_session",
            {"status": SUSPEND_STATUS, "recap": recap},
        )

    # ---- page identity ----

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
        if verdict.kind != "match":
            log.info("conductor: screen is %s — starting from the top", verdict.kind)
            return 0
        resume = 0
        for i, node in enumerate(self.spec.nodes):
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
        """Always None — typed as the advance result so call sites read
        `return self._handover(...)`."""
        log.warning(
            "conductor: handing %s/%s over to the model — %s",
            self.app,
            self.spec.name,
            reason,
        )
        return None
