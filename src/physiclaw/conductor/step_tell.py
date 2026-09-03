"""The `tell` step — message the user, and the walk moves on.

The playbook's `message:` goes verbatim over the channel (`speak.py`
is the shared voice); once the send lands on the thread the step is
done and the next node runs — nothing waits, nothing suspends. A
trailing tell is the walk's last word, so the walk completes right
after it. Whatever the user writes back is the next wake's business:
the boot reads the thread then and parse_task decides whether it is a
new request.
"""

from physiclaw.conductor import speak
from physiclaw.conductor.playbook import TellNode, fill_refs
from physiclaw.conductor.step import Step, Turn

KIND_TELL_SENT = "tell-sent"


class TellStep(Step[TellNode]):
    kinds = frozenset({KIND_TELL_SENT})

    def open(self) -> Turn:
        node, walk = self.node, self.walk
        # `message:` verbatim, refs filled — a tell has no reply to read,
        # so it declares no words.
        text = str(
            fill_refs(
                node.message, walk.ref_values(), where=f"tell {node.id!r} `message`"
            )
        )
        return speak.send(walk, KIND_TELL_SENT, text)

    def landed(self, kind: str) -> Turn:
        stop = speak.sent_landed(self.walk)
        if stop is not None:
            return stop
        return self.walk.advance_cursor()
