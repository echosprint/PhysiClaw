"""The `ask` step — message the user and HOLD for approval.

The playbook's `message:` goes verbatim over the channel (`speak.py`
is the shared voice); the step polls the thread for a reply (the ask's
own `wait:` cadence and silent rounds, then the session suspends for
the next wake), reads the reply against the words the ask itself
declares (`yes:` / `no:`, `reply.py`), and on consent binds the money
numbers the following payment move spends. A deny ends the walk; a
reply the declared words do not cover hands over — the model reads the
thread, the conductor never guesses. A payment ask reads the sheet
total off the waypoint before it, once that page reads — the amount
beside the label its `total:` names — and quotes it in the message:
the ask IS the consent record. Every amount on that sheet is
remembered as what the user saw.
"""

from physiclaw.common import gesture_vocab
from physiclaw.conductor.spec import reply
from physiclaw.conductor.spec.conventions import page_id
from physiclaw.conductor.spec.model import AskNode
from physiclaw.conductor.spec.pack import qualified_macro
from physiclaw.conductor.spec.refs import fill_refs
from physiclaw.conductor.walk import money, speak
from physiclaw.conductor.walk.step import Step, Turn
from physiclaw.contract.dto import AssistantMessage

KIND_ASK_SENT = "ask-sent"
KIND_ASK_WAIT = "ask-wait"
KIND_ASK_PEEK = "ask-peek"
KIND_ASK_OPEN = "ask-open"
KIND_ASK_RESUME = "ask-resume"


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
            stop = speak.sent_landed(walk)
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
        return speak.send(walk, KIND_ASK_SENT, text, node.yes, node.no)

    def _check(self) -> Turn:
        """One reply-evaluation round over the freshly peeked thread:
        new-message precondition, then the ask's own words."""
        node, walk = self.node, self.walk
        gate = walk.gate
        wrong = speak.thread_mismatch(walk)
        if wrong is not None:
            reopen = self._reopen()
            if reopen is not None:
                return reopen
            return walk.handover(f"cannot reach the user thread ({wrong})")
        new = speak.new_replies(walk)
        if not new:
            gate.silence += 1
            if gate.silence >= node.silence_rounds:
                return walk.suspend()
            return self._wait()
        verdict = speak.verdict(walk, new)
        if verdict is reply.Answer.DENY:
            return speak.deny(walk)
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

    def _reopen(self) -> AssistantMessage | None:
        """The thread is not on screen: reopen it ONCE per ask via the
        channel's open macro — None when there is none, or it already
        ran."""
        walk = self.walk
        if not (walk.channel and walk.channel.open) or walk.gate.tried_open:
            return None
        walk.gate.tried_open = True
        return walk.synth(
            KIND_ASK_OPEN,
            "conductor: reopening the user thread",
            gesture_vocab.RUN_MACRO,
            {"name": walk.channel.open},
            channel=True,
        )

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
