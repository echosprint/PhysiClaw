"""The `agent` step — the model's move, inside the author's fence.

No tools = one pure-text call filling the declared `returns` (prompt
in, fields out, no screen). Tools = an EPISODE: each turn one
constrained call (append-only context, so every request's prefix is
byte-identical to the previous one and the provider cache pays for all
but the newest block) whose answer the walk grounds to a live bbox — a
screen row, a granted landmark, a scroll or back verb, `done`, or
`escalate` — or runs by name (a granted pack macro); never
coordinates. A landmark scoped to a page is offered only while that
page is the verified reading. `done` is audited against the adjacent
verify page by the matcher, never trusted, and a payment episode
re-runs the money predicates before EVERY tap or macro the model
proposes.
"""

from physiclaw.common import gesture_vocab
from physiclaw.conductor import context, money, recover
from physiclaw.conductor.calls import (
    ACT_BACK,
    ACT_SCROLL_DOWN,
    ACT_SCROLL_UP,
    AGENT_DONE,
    AGENT_TOOL_VERBS,
    ESCALATE,
)
from physiclaw.conductor.match import reads_as_locked
from physiclaw.conductor.micro import (
    ACT_ARM,
    AGENT_ACT,
    AGENT_FIELDS,
    CAND_LANDMARK,
    CAND_MACRO,
    Candidate,
    DecisionRequest,
    MicroOutcome,
    act_block,
    act_candidates,
    canonical_reply,
    data_block,
    return_fields,
)
from physiclaw.conductor.pages import page_id
from physiclaw.conductor.playbook import AgentNode, fill_refs, qualified_macro
from physiclaw.conductor.step import Step, Turn, Walk
from physiclaw.conductor.turns import SCROLL_BBOX

KIND_TAP = "agent-tap"
KIND_SWIPE = "agent-swipe"
KIND_MACRO = "agent-macro"


class AgentStep(Step[AgentNode]):
    kinds = frozenset({KIND_TAP, KIND_SWIPE, KIND_MACRO})

    def __init__(self, walk: Walk, node: AgentNode) -> None:
        super().__init__(walk, node)
        # Episode state. `history` is the append-only transcript of
        # settled (user block, model reply) pairs — replayed verbatim on
        # every call; `block` is the pending user block the next call
        # sends. `consented` is a payment episode's bound, stashed at
        # open so the per-tap predicates keep checking after the gate's
        # consent is consumed by the first fire.
        self.block = ""
        self.history: list[tuple[str, str]] = []
        self.candidates: tuple[Candidate, ...] = ()
        self.calls = 0
        self.scrolls = 0
        self.consented: float | None = None
        self.seen: tuple[float, ...] = ()
        self.pending_desc = ""

    def open(self) -> Turn:
        if not self.node.tools:
            # A pure-text call: no screen, no page contract — prompt in,
            # declared fields out.
            return self._fields_request()
        gate = self.walk.enter_gate(self.node)
        if gate is not None:
            return gate
        return self._episode_start()

    def landed(self, kind: str) -> Turn:
        # An episode action landed: its own result view is the next
        # turn's screen block.
        return self._episode_landed()

    def resolve(self, outcome: MicroOutcome | None) -> Turn:
        if self.node.tools:
            return self._episode_resolve(outcome)
        return self._fields_done(outcome)

    # ---- shared ----

    def _prompt(self, vals: dict[str, str]) -> str:
        """The authored prompt with its refs filled ONCE — then frozen."""
        return str(
            fill_refs(self.node.prompt, vals, where=f"agent {self.node.id!r} `prompt`")
        )

    def _fields(self) -> str:
        return "\n".join(f"- {n}: {d}" for n, d in self.node.returns)

    def _context(self) -> str:
        """What the author declared beside the prompt (`context:`),
        loaded now — nothing else of the agent's memory travels."""
        return context.load(self.node.context)

    def _close(self, outcome: MicroOutcome, *, calls: int = 0) -> Turn:
        """The one done tail both forms share: record the returns,
        journal, advance the cursor."""
        node, walk = self.node, self.walk
        payload = outcome.payload or {}
        missing = [n for n in node.return_fields if not payload.get(n, "").strip()]
        if missing:
            # A missing field hands over instead of retrying: escalation
            # is the default, never a guess.
            return walk.handover(
                f"agent {node.id!r} answered done without return field(s) "
                f"{', '.join(missing)}"
            )
        for n in node.return_fields:
            walk.outputs[f"{node.id}.{n}"] = payload[n].strip()
        after = f" after {calls} calls" if calls else ""
        walk.journal(f"agent {node.id}: done{after} — {outcome.reason}")
        return walk.advance_cursor()

    # ---- the pure-text call ----

    def _fields_request(self) -> Turn:
        node = self.node
        return DecisionRequest(
            call=AGENT_FIELDS,
            node_id=node.id,
            outcomes=(),
            args={
                "prompt": self._prompt(self.walk.ref_values()),
                "fields": self._fields(),
            },
            context=self._context(),
        )

    def _fields_done(self, outcome: MicroOutcome | None) -> Turn:
        node = self.node
        if outcome is None:
            return self.walk.handover(
                f"agent {node.id!r}: call failed or under-confident"
            )
        if outcome.out != AGENT_DONE:
            return self.walk.handover(f"agent {node.id!r} escalated: {outcome.reason}")
        return self._close(outcome)

    # ---- the episode ----

    def _episode_start(self) -> Turn:
        """Open an acting episode on the current (enter-verified) screen.
        The prompt's refs fill ONCE here — the brief (prompt, return
        fields, declared context) heads the first block and rides the
        replayed history verbatim; a payment episode additionally gets
        {ask.total} — the consented amount its adjacent gate bound — and
        stashes that bound for the per-tap predicates."""
        node, walk = self.node, self.walk
        vals = walk.ref_values()
        if node.irreversible == "payment":
            if walk.gate.consented is None:
                return walk.handover(
                    f"agent {node.id!r}: payment episode without bound consent"
                )
            self.consented = walk.gate.consented
            self.seen = walk.gate.seen
            vals = {**vals, "ask.total": f"{self.consented:g}"}
        brief = [self._prompt(vals)]
        if self.node.returns:
            brief.append(return_fields(self._fields()))
        loaded = self._context()
        if loaded:
            brief.append(data_block("Context", loaded))
        self._screen_block("\n\n".join(brief))
        return self._request()

    def _screen_block(self, prefix: str) -> None:
        """Rebuild the episode's answerable candidates off the CURRENT
        screen and set the pending user block. When the episode may tap,
        granted landmarks ride as candidates too (their declared bbox —
        healed to the live label only if picked) unless a live row
        already reads their name — the fresher row wins — and a
        page-scoped landmark rides only while its page is the verified
        reading; with no tap tool there is nothing to tap, so no rows.
        Granted macros are always answerable, by name."""
        walk, node = self.walk, self.node
        assert walk.screen is not None
        macros = tuple(Candidate(key=n, kind=CAND_MACRO) for n in node.macros)
        rows = tuple(
            c for c in act_candidates(walk.screen.rows) if c.key not in node.macros
        )
        can_tap = "tap" in node.tools
        give = tuple(n for n in node.give if self._granted(n)) if can_tap else ()
        row_keys = {r.key for r in rows}
        grants = tuple(
            Candidate(key=n, bbox=walk.landmarks[n].bbox, kind=CAND_LANDMARK)
            for n in give
            if n not in row_keys
        )
        self.candidates = macros + (grants + rows if can_tap else ())
        parts = [prefix] if prefix else []
        parts.append(act_block("Current screen", rows))
        if give:
            parts.append("Granted landmarks (answer by name): " + ", ".join(give))
        if node.macros:
            parts.append(
                "Granted macros (answer by name to run one): " + ", ".join(node.macros)
            )
        self.block = "\n".join(parts)

    def _granted(self, name: str) -> bool:
        """Whether a landmark is on offer NOW: declared, and either
        unscoped or scoped to the page the current verdict reads."""
        walk = self.walk
        landmark = walk.landmarks.get(name)
        if landmark is None:
            return False
        if landmark.page is None:
            return True
        return walk.verdict is not None and walk.verdict.matches(
            page_id(walk.app, landmark.page)
        )

    def _request(self) -> Turn:
        node = self.node
        self.calls += 1
        if self.calls > node.max_calls:
            return self.walk.handover(
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
            # `tools`/`give` shape the legend — fixed for the episode.
            args={
                "block": self.block,
                "tools": " ".join(node.tools),
                "give": ", ".join(node.give),
                "macros": ", ".join(node.macros),
            },
            candidates=self.candidates,
            history=tuple(self.history),
        )

    def _episode_resolve(self, outcome: MicroOutcome | None) -> Turn:
        node, walk = self.node, self.walk
        if outcome is None:
            return walk.handover(f"agent {node.id!r}: call failed or under-confident")
        # Settle the turn into the append-only history: the block that
        # asked, then the reply in the contract's canonical spelling
        # (micro re-serializes it, so repair-retry noise never enters
        # the replayed prefix).
        self.history.append(("user", self.block))
        self.history.append(("assistant", canonical_reply(outcome)))
        if outcome.out == ESCALATE:
            return walk.handover(f"agent {node.id!r} escalated: {outcome.reason}")
        if outcome.out == ACT_BACK:
            # The OS back edge-swipe — the reliable pop on iOS (a corner
            # chevron tap misses too often to trust the stylus with).
            self.pending_desc = "went back"
            return walk.synth(
                KIND_SWIPE,
                f"conductor: agent {node.id} — went back",
                gesture_vocab.GO_BACK,
                {},
            )
        if outcome.out in (ACT_SCROLL_DOWN, ACT_SCROLL_UP):
            self.scrolls += 1
            if self.scrolls > node.max_scrolls:
                return walk.handover(
                    f"agent {node.id!r} exceeded its scroll limit ({node.max_scrolls})"
                )
            # scroll_down = see content further down = the swipe goes up.
            down = outcome.out == ACT_SCROLL_DOWN
            self.pending_desc = "scrolled down" if down else "scrolled up"
            return walk.synth(
                KIND_SWIPE,
                f"conductor: agent {node.id} — {self.pending_desc}",
                gesture_vocab.SWIPE,
                {"bbox": list(SCROLL_BBOX), "direction": "up" if down else "down"},
            )
        if outcome.out == AGENT_DONE:
            # The exit contract is the matcher's, never the model's: done
            # counts only on the adjacent verify page. A rejection costs
            # one call and the episode continues.
            wrong = (
                walk.mismatch(walk.verdict, page_id(walk.app, node.verify))
                if walk.verdict is not None
                else "no screen observed"
            )
            if wrong is not None:
                self.block = (
                    f"done rejected: the walk must be on {node.verify!r} — "
                    f"{wrong}. Continue toward the goal, or answer escalate."
                )
                return self._request()
            return self._close(outcome, calls=self.calls)
        assert outcome.out == ACT_ARM and outcome.picked is not None
        assert walk.screen is not None
        if node.irreversible == "payment":
            # The purse stays with the walker: BOTH predicates re-run
            # before every tap or macro the model proposes — one while
            # no visible amount equals the consented total, or with any
            # amount above it, is refused.
            blocked = money.fire_block(
                consented=self.consented, seen=self.seen, screen=walk.screen
            )
            if blocked is not None:
                return walk.handover(f"payment agent {node.id!r}: {blocked}")
            # Consent is consumed by the FIRST fire — a later payment
            # needs its own gate (the move rule, episode-shaped).
            walk.spend_consent()
        picked = outcome.picked
        if picked.kind == CAND_MACRO:
            # A granted pack macro: the recorded gesture sequence runs
            # whole, argument-less; its result view is the next screen.
            self.pending_desc = f"ran macro {picked.key!r}"
            return walk.synth(
                KIND_MACRO,
                f"conductor: agent {node.id} — {self.pending_desc}",
                gesture_vocab.RUN_MACRO,
                {"name": qualified_macro(walk.app, picked.key)},
            )
        bbox = picked.bbox
        assert bbox is not None  # rows and landmarks carry one
        if picked.kind == CAND_LANDMARK:
            # A granted landmark: label-healed to where its text sits
            # NOW (the one search the macro heal rides too).
            bbox = recover.locate_landmark(walk.landmarks[picked.key], walk.screen)[0]
        self.pending_desc = f"tapped {picked.key!r}"
        return walk.synth(
            KIND_TAP,
            f"conductor: agent {node.id} — {self.pending_desc}",
            "tap",
            {"bbox": list(bbox)},
        )

    def _episode_landed(self) -> Turn:
        """An episode action's own result view becomes the next turn's
        screen block — no extra peek."""
        node, walk = self.node, self.walk
        assert walk.screen is not None
        # Whatever happens next, money may have moved: the purchase line
        # lands the moment the first consented tap's result does.
        walk.log_purchase()
        if reads_as_locked(walk.screen):
            return walk.handover(f"agent {node.id!r}: phone locked mid-episode")
        self._screen_block(f"[you {self.pending_desc}]")
        return self._request()
