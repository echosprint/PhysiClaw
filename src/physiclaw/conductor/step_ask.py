"""The `ask` and `tell` steps — the walk's two ways of speaking to the
user, both over the channel pack (`channel.py`): the playbook's
`message:` goes VERBATIM (only the author knows the user's language).

`ask` sends and HOLDS: it polls the thread for a reply (the ask's own
`wait:` cadence and silent rounds, then the session suspends for the
next wake), reads the reply
against the words the ask itself declares (`yes:` / `no:`, `reply.py`),
and on consent binds the money numbers the following payment move
spends. A deny ends the walk; a reply the declared words do not cover
hands over — the model reads the thread, the conductor never guesses.
A payment ask reads the sheet total off the waypoint before it, once
that page reads — the amount beside the label its `total:` names —
and quotes it in the message: the ask IS the consent record. Every
amount on that sheet is remembered as what the user saw.

`tell` sends and suspends; ANY next wake resumes the walk past it, and
when the tell declares `no:` words that resume first reads the thread
for a cancel the user may have sent meanwhile (`TellResume`).
"""

import logging

from physiclaw.common import gesture_vocab
from physiclaw.conductor import money, reply
from physiclaw.conductor.pages import THREAD_ID, page_id
from physiclaw.conductor.playbook import AskNode, TellNode, fill_refs, qualified_macro
from physiclaw.conductor.step import Step, Turn, Walk
from physiclaw.contract.dto import AssistantMessage

log = logging.getLogger(__name__)

KIND_ASK_SENT = "ask-sent"
KIND_ASK_WAIT = "ask-wait"
KIND_ASK_PEEK = "ask-peek"
KIND_ASK_OPEN = "ask-open"
KIND_ASK_RESUME = "ask-resume"
KIND_TELL_SENT = "tell-sent"
KIND_TELL_PEEK = "tell-peek"
KIND_TELL_OPEN = "tell-open"


def _thread_mismatch(walk: Walk) -> str | None:
    if walk.channel is None:
        return "no channel pack"
    assert walk.verdict is not None
    return walk.mismatch(walk.verdict, THREAD_ID)


def _send(
    walk: Walk, kind: str, text: str, yes: tuple[str, ...], no: tuple[str, ...]
) -> Turn:
    """The one door to the user: the channel's send macro with the
    authored text — or the handover when there is no send to run.
    `yes`/`no` are this send's declared reply words; they take over
    from the previous send's when this one LANDS (`_sent_landed`), so
    the sweep for replies to the previous send judges by its words."""
    if walk.channel is None or walk.channel.send is None:
        return walk.handover(
            "no channel send macro — record playbooks/channel to enable asks"
        )
    walk.gate.ask = text
    walk.gate.next_words = (yes, no)
    walk.gate.tried_open = False
    return walk.synth(
        kind,
        "conductor: messaging the user",
        gesture_vocab.RUN_MACRO,
        {"name": walk.channel.send, "inputs": {"message": text}},
        channel=True,
    )


def _reopen(walk: Walk, kind: str) -> AssistantMessage | None:
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


def _new_replies(walk: Walk, *, after_ask: bool = True) -> list[str]:
    """The user's messages since the gate's baseline (`reply.py` owns
    the diff), off the current thread screen."""
    assert walk.screen is not None
    gate = walk.gate
    return reply.new_incoming(
        walk.screen.rows, gate.baseline, gate.ask, after_ask=after_ask
    )


def _verdict(walk: Walk, messages: list[str]) -> reply.Answer | None:
    """The gate's own words over `messages` — deny wins over anything
    else said, an uncovered message defers."""
    return reply.classify_all(
        messages, frozenset(walk.gate.yes), frozenset(walk.gate.no)
    )


def _deny(walk: Walk) -> Turn:
    """The one deny disposition: no re-asks, and no second chance this
    session."""
    return walk.handover(
        "user declined the ask — acknowledge them, back out of any "
        "open checkout or cart state this task created, and wrap up"
    )


def _sent_landed(walk: Walk) -> Turn:
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
        and _verdict(walk, _new_replies(walk, after_ask=False)) is reply.Answer.DENY
    ):
        return _deny(walk)
    # The send landed: its words and its thread snapshot take over together.
    gate.yes, gate.no = gate.next_words
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
            return self._wait()
        if kind == KIND_ASK_WAIT:
            return self._peek()
        if kind == KIND_ASK_RESUME:
            # The resume macro landed; the next node's own checks judge
            # the landing.
            return walk.advance_cursor()
        return self._check()  # KIND_ASK_PEEK / KIND_ASK_OPEN

    def _start(self) -> Turn:
        """Send the ask. A payment ask reads the sheet total off the
        waypoint before it, once that page reads — never whatever screen
        happened to come last — the amount beside the label its `total:`
        declares — and quotes it: the message IS the consent record."""
        node, walk = self.node, self.walk
        values = walk.ref_values()
        if node.approve == "payment":
            # The sheet is the waypoint before the ask — that page, not
            # any own-pack page the phone happens to show.
            assert walk.verdict is not None
            wrong = walk.mismatch(walk.verdict, page_id(walk.app, node.enter))
            if wrong is not None:
                return walk.handover(
                    f"ask {node.id!r} reads its total off {node.enter!r} "
                    f"({wrong}) — refusing to ask blind"
                )
            assert walk.screen is not None  # a verified verdict was read off it
            walk.gate.seen = tuple(money.amounts(walk.screen))
            total = money.declared_total(walk.screen, node.total)
            if total is None:
                return walk.handover(
                    f"ask {node.id!r}: no amount readable beside "
                    f"{' / '.join(node.total)} on the sheet"
                )
            # The number the user will see is the number they consent
            # to — and, at fire time, the bound.
            walk.gate.quoted = total
            values = {**values, "ask.total": f"{walk.gate.quoted:g}"}
        text = str(fill_refs(node.message, values, where=f"ask {node.id!r} `message`"))
        return _send(walk, KIND_ASK_SENT, text, node.yes, node.no)

    def _check(self) -> Turn:
        """One reply-evaluation round over the freshly peeked thread:
        new-message precondition, then the ask's own words."""
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
            if gate.silence >= node.silence_rounds:
                return walk.suspend(resume_idx=walk.idx, awaiting=True)
            return self._wait()
        verdict = _verdict(walk, new)
        if verdict is reply.Answer.DENY:
            return _deny(walk)
        if verdict is reply.Answer.CONFIRM:
            return self._confirmed()
        # The declared words do not cover it ("ok, but make it two
        # boxes", a question, a hold): the model reads the thread from
        # the transcript and decides — the conductor never guesses.
        return walk.handover(
            f"ask {node.id!r}: reply {' / '.join(new)!r} matches none of its "
            "yes/no words — read the thread and decide before any payment"
        )

    def _confirmed(self) -> Turn:
        node, walk = self.node, self.walk
        # Informed consent binds the money predicates: the quoted total
        # becomes the consented one.
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
            {"seconds": self.node.wait_seconds},
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
        return _send(walk, KIND_TELL_SENT, text, (), node.no)

    def landed(self, kind: str) -> Turn:
        walk = self.walk
        stop = _sent_landed(walk)
        if stop is not None:
            return stop
        if walk.idx + 1 >= len(walk.spec.nodes):
            # A trailing tell: the message is the walk's last word, so
            # it completes now rather than suspending for a wake that
            # would only mint the completion.
            return walk.advance_cursor()
        # Message away → suspend; the walk continues past this node on
        # the resuming wake — reading the thread for a cancel first when
        # the tell declared deny words.
        walk.gate.told = bool(walk.gate.no)
        return walk.suspend(resume_idx=walk.idx + 1, awaiting=False)


class TellResume(Step[None]):
    """A suspended tell's resume, when the tell declared `no:` words:
    one thread read before the walk acts — a declared deny among the
    replies since the tell's baseline ends the walk (the user cancelled
    into a fire-and-forget message — honor it); anything else falls
    through to the normal resume peek."""

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
            if _verdict(walk, _new_replies(walk)) is reply.Answer.DENY:
                return _deny(walk)
        else:
            reopen = _reopen(walk, KIND_TELL_OPEN)
            if reopen is not None:
                return reopen
            log.info("conductor: tell reply check skipped — %s", wrong)
        # No cancel (or no thread): the read is spent — the tell's words
        # and flag are cleared so it never repeats — and the walk proper
        # starts with its plain peek (the stored cursor is trusted).
        walk.gate.no = ()
        walk.gate.told = False
        return walk.peek()
