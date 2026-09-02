"""The `ask` and `tell` steps — the walk's two ways of speaking to the
user, both over the channel pack (`channel.py`): the playbook's
`message:` goes VERBATIM (only the author knows the user's language).

`ask` sends and HOLDS: it polls the thread for a reply (bounded silent
rounds, then the session suspends for the next wake), reads the reply
through the ruled tiers — the deterministic word lists (`reply.py`)
first, ONE bounded LLM check (`confirm_reply`) for anything unclear —
and on consent binds the money numbers the following payment move
spends. A deny ends the walk. A payment ask reads the sheet total off
a VERIFIED own-pack page and quotes it in the message: the ask IS the
consent record.

`tell` sends and suspends; ANY next wake resumes the walk past it, and
that resume first reads the thread for a cancel the user may have sent
meanwhile (`TellResume`).
"""

import logging

from physiclaw.common import gesture_vocab
from physiclaw.conductor import money, reply
from physiclaw.conductor.micro import CONFIRM_REPLY, MicroOutcome, build_request
from physiclaw.conductor.pages import THREAD_ID
from physiclaw.conductor.playbook import (
    GATE_MAX_CHECKS,
    AskNode,
    TellNode,
    fill_refs,
    qualified_macro,
)
from physiclaw.conductor.step import Step, Turn
from physiclaw.contract.dto import AssistantMessage

log = logging.getLogger(__name__)

# Ask-and-hold bounds: in-session reply polling cadence, and how many
# silent rounds before the session suspends for the next wake.
# (Unclear-reply rounds are bounded separately by GATE_MAX_CHECKS.)
GATE_WAIT_SECONDS = 45
SILENCE_ROUNDS = 3

KIND_ASK_SENT = "ask-sent"
KIND_ASK_WAIT = "ask-wait"
KIND_ASK_PEEK = "ask-peek"
KIND_ASK_OPEN = "ask-open"
KIND_ASK_RESUME = "ask-resume"
KIND_TELL_SENT = "tell-sent"
KIND_TELL_PEEK = "tell-peek"
KIND_TELL_OPEN = "tell-open"


def _thread_mismatch(walk) -> str | None:
    if walk.channel is None:
        return "no channel pack"
    assert walk.verdict is not None
    return walk.mismatch(walk.verdict, THREAD_ID)


def _send(walk, kind: str, text: str) -> Turn:
    """The one door to the user: the channel's send macro with the
    authored text — or the handover when there is no send to run."""
    if walk.channel is None or walk.channel.send is None:
        return walk.handover(
            "no channel send macro — record playbooks/channel to enable asks"
        )
    walk.gate.ask = text
    walk.gate.tried_open = False
    return walk.synth(
        kind,
        "conductor: messaging the user",
        gesture_vocab.RUN_MACRO,
        {"name": walk.channel.send, "inputs": {"message": text}},
        channel=True,
    )


def _reopen(walk, kind: str) -> AssistantMessage | None:
    """The thread is not on screen: reopen it ONCE per ask via the
    channel's open macro — None when there is none, or it already ran."""
    if not (walk.channel and walk.channel.open) or walk.gate.tried_open:
        return None
    walk.gate.tried_open = True
    return walk.synth(
        kind,
        "conductor: reopening the user thread",
        gesture_vocab.RUN_MACRO,
        {"name": walk.channel.open},
        channel=True,
    )


def _new_replies(walk, *, after_ask: bool = True) -> list[str]:
    """The user's messages since the gate's baseline (`reply.py` owns
    the diff), off the current thread screen."""
    assert walk.screen is not None
    return reply.new_incoming(
        walk.screen.rows, walk.gate.baseline, walk.gate.ask, after_ask=after_ask
    )


def _deny(walk) -> Turn:
    """The one deny disposition — every tier lands here: no re-asks, and
    no second chance this session."""
    return walk.handover(
        "user declined the ask — acknowledge them, back out of any "
        "open checkout or cart state this task created, and wrap up"
    )


def _sent_landed(walk) -> Turn:
    """A send's landing, shared by ask and tell: it must be on the
    thread; anything the user sent while the walk was off in the app (an
    earlier ask's baseline) lands here as new, and a deny among it must
    stop the walk NOW — overwriting the baseline would swallow it
    forever (deny only: an old confirm word above the fresh ask is never
    treated as consent). Then the thread is baselined. None = carry on."""
    wrong = _thread_mismatch(walk)
    if wrong is not None:
        return walk.handover(f"channel send did not land on the thread ({wrong})")
    gate = walk.gate
    if (
        gate.baseline
        and reply.classify_all(_new_replies(walk, after_ask=False)) == "deny"
    ):
        return _deny(walk)
    assert walk.screen is not None
    gate.baseline = {r.label.strip() for r in walk.screen.rows if r.label.strip()}
    return None


class AskStep(Step[AskNode]):
    kinds = frozenset(
        {KIND_ASK_SENT, KIND_ASK_WAIT, KIND_ASK_PEEK, KIND_ASK_OPEN, KIND_ASK_RESUME}
    )

    def open(self) -> Turn:
        if self.walk.gate.awaiting:
            # Suspended-at-gate resume: the ask was sent before the
            # suspension — go straight to reading the thread.
            return self._peek()
        return self._start()

    def landed(self, kind: str) -> Turn:
        walk = self.walk
        if kind == KIND_ASK_SENT:
            stop = _sent_landed(walk)
            if stop is not None:
                return stop
            walk.gate.awaiting = True
            walk.gate.silence = 0
            walk.gate.checks = 0
            return self._wait()
        if kind == KIND_ASK_WAIT:
            return self._peek()
        if kind == KIND_ASK_RESUME:
            # The resume macro landed; the next node's own checks judge
            # the landing.
            return walk.advance_cursor()
        return self._check()  # KIND_ASK_PEEK / KIND_ASK_OPEN

    def resolve(self, outcome: MicroOutcome | None) -> Turn:
        """The LLM tier came back. None/unclear → keep waiting (this
        round's check was already counted in `_check`)."""
        gate = self.walk.gate
        if not gate.llm_batch:
            return self.walk.handover("a reply judgment arrived with none outstanding")
        batch, gate.llm_batch = gate.llm_batch, []
        if outcome is not None:
            # Baseline only what was actually judged: a provider failure
            # leaves the batch visible to the next round — the user's
            # reply is never silently swallowed.
            gate.baseline |= set(batch)
        if outcome is not None and outcome.out == "deny":
            return _deny(self.walk)
        if outcome is not None and outcome.out == "revise":
            # A change request ("ok, but make it two boxes") or a
            # question is a TASK change, not a gate verdict — the model
            # takes over with the thread in the transcript and acts on
            # the user's words.
            return self.walk.handover(
                f"user asked for changes ({' / '.join(batch)!r}) — read the "
                "thread and adjust the order before any payment"
            )
        if outcome is not None and outcome.out == "confirm":
            return self._confirmed()
        return self._wait()

    def _start(self) -> Turn:
        """Send the ask. A payment ask reads the sheet total off a
        VERIFIED own-pack page (the sheet the lints put before the ask),
        never whatever screen happened to come last, and quotes it —
        the message IS the consent record."""
        node, walk = self.node, self.walk
        values = walk.ref_values()
        template = node.message
        if node.approve == "payment":
            blocked = walk.money_page_block(f"ask {node.id!r}")
            if blocked is not None:
                return walk.handover(f"{blocked} — refusing to ask blind")
            cap = None
            if walk.spec.mandate is not None:
                # A `budget:` is optional — without one the consented
                # total IS the bound (`money.fire_block` treats cap None
                # exactly so), and there is no over-budget branch.
                cap = money.mandate_cap(walk.spec.mandate, walk.values)
                if cap is None:
                    return walk.handover(
                        f"ask {node.id!r}: mandate cap could not be resolved"
                    )
            amts = money.amounts(walk.screen) if walk.screen else []
            if not amts:
                return walk.handover(f"ask {node.id!r}: no total readable on the sheet")
            quoted = max(amts)
            walk.gate.quoted, walk.gate.cap = quoted, cap
            values = {**values, "ask.total": f"{quoted:g}"}
            if cap is not None:
                values["ask.cap"] = f"{cap:g}"
                if quoted > cap:
                    # Over-cap (ruled): the SAME plain-consent reply opens
                    # the gate, but the breach must be disclosed — the
                    # parser guarantees this template quotes total AND cap.
                    assert node.over_message is not None
                    template = node.over_message
        text = str(fill_refs(template, values, where=f"ask {node.id!r} `message`"))
        return _send(walk, KIND_ASK_SENT, text)

    def _check(self) -> Turn:
        """One reply-evaluation round over the freshly peeked thread —
        the ruled tiers: new-message precondition, word match, LLM."""
        node, walk = self.node, self.walk
        gate = walk.gate
        wrong = _thread_mismatch(walk)
        if wrong is not None:
            reopen = _reopen(walk, KIND_ASK_OPEN)
            if reopen is not None:
                return reopen
            return walk.handover(f"cannot reach the user thread ({wrong})")
        new = _new_replies(walk)
        if not new:
            gate.silence += 1
            if gate.silence >= SILENCE_ROUNDS:
                return walk.suspend(resume_idx=walk.idx, awaiting=True)
            return self._wait()
        verdict = reply.classify_all(new)
        if verdict == "deny":
            return _deny(walk)
        if verdict == "confirm":
            return self._confirmed()
        # Unclear → the LLM tier, bounded; a batch is baselined ONLY
        # when a judgment actually arrives (see `resolve`) — the round
        # that exhausts the budget suspends with the batch still unread,
        # and a failed judgment leaves it for the next round, so the
        # reply is judged once, eventually.
        gate.checks += 1
        if gate.checks > GATE_MAX_CHECKS:
            return walk.suspend(resume_idx=walk.idx, awaiting=True)
        gate.llm_batch = new
        # Empty outcomes: confirm_reply's answer space is fixed whole in
        # its _SPECS row, never caller-supplied.
        assert walk.screen is not None
        return build_request(
            CONFIRM_REPLY,
            node.id,
            (),
            {"ask": gate.ask, "reply": " / ".join(new)},
            walk.screen,
        )

    def _confirmed(self) -> Turn:
        node, walk = self.node, self.walk
        # Informed consent binds the money predicates: the quoted total
        # becomes the consented one (over-cap included — ruled).
        walk.gate.consented = walk.gate.quoted
        walk.gate.awaiting = False
        walk.journal(f"user confirmed {node.approve}")
        if node.resume is not None:
            return walk.synth(
                KIND_ASK_RESUME,
                f"conductor: back to the app via {node.resume}",
                gesture_vocab.RUN_MACRO,
                {"name": qualified_macro(walk.app, node.resume)},
            )
        return walk.advance_cursor()

    def _wait(self) -> AssistantMessage:
        return self.walk.synth(
            KIND_ASK_WAIT,
            "conductor: waiting for the user's reply",
            "wait",
            {"seconds": GATE_WAIT_SECONDS},
        )

    def _peek(self) -> AssistantMessage:
        return self.walk.synth(
            KIND_ASK_PEEK,
            "conductor: checking for a reply",
            gesture_vocab.PEEK,
            {},
            channel=True,
        )


class TellStep(Step[TellNode]):
    kinds = frozenset({KIND_TELL_SENT})

    def open(self) -> Turn:
        node, walk = self.node, self.walk
        # `message:` verbatim, refs filled — a tell has no consent
        # contract to protect (any wake resumes the walk).
        text = str(
            fill_refs(
                node.message, walk.ref_values(), where=f"tell {node.id!r} `message`"
            )
        )
        return _send(walk, KIND_TELL_SENT, text)

    def landed(self, kind: str) -> Turn:
        walk = self.walk
        stop = _sent_landed(walk)
        if stop is not None:
            return stop
        # Message away → suspend; the walk continues past this node on
        # the resuming wake.
        return walk.suspend(resume_idx=walk.idx + 1, awaiting=False)


class TellResume(Step[None]):
    """A suspended tell's resume: one thread read before the walk acts —
    a deny among the replies since the tell's baseline ends the walk
    (the user cancelled into a fire-and-forget message — honor it);
    anything else falls through to the normal resume peek. Word tier
    only — a tell has no consent contract, so an unclear reply is the
    model's to read from the transcript if the walk hands over."""

    kinds = frozenset({KIND_TELL_PEEK, KIND_TELL_OPEN})

    def open(self) -> Turn:
        return self.walk.synth(
            KIND_TELL_PEEK,
            "conductor: checking the thread for a reply before continuing",
            gesture_vocab.PEEK,
            {},
            channel=True,
        )

    def landed(self, kind: str) -> Turn:
        walk = self.walk
        wrong = _thread_mismatch(walk)
        if wrong is None:
            if reply.classify_all(_new_replies(walk)) == "deny":
                return _deny(walk)
        else:
            reopen = _reopen(walk, KIND_TELL_OPEN)
            if reopen is not None:
                return reopen
            log.info("conductor: tell reply check skipped — %s", wrong)
        # No cancel (or no thread): the walk proper starts with its plain
        # peek — the stored cursor is trusted.
        return walk.peek()
