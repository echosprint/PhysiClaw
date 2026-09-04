"""The walk's voice — how a step speaks to the user and reads what
comes back, over the channel pack (`channel.py`).

Both `ask` (`step_ask.py`) and `tell` (`step_tell.py`) send the
playbook's `message:` VERBATIM (only the author knows the user's
language) through the channel's send macro, land on the thread, and
baseline it; an ask later reads the bubbles that arrived since against
its declared words (`reply.py`), and any later send sweeps them for a
deny first. What is shared lives here once — the send door, the
landing rule with its deny sweep, the thread check, the new-reply
read, the verdict, the deny disposition — so the two steps cannot
word one rule two ways.
"""

from physiclaw.common import gesture_vocab
from physiclaw.conductor.spec import reply
from physiclaw.conductor.spec.conventions import THREAD_ID
from physiclaw.conductor.walk.step import Turn, Walk


def thread_mismatch(walk: Walk) -> str | None:
    """None when the current verdict IS the user thread; else why not."""
    if walk.channel is None:
        return "no channel pack"
    assert walk.verdict is not None
    return walk.mismatch(walk.verdict, THREAD_ID)


def send(
    walk: Walk,
    kind: str,
    text: str,
    yes: tuple[str, ...] = (),
    no: tuple[str, ...] = (),
) -> Turn:
    """The one door to the user: the channel's send macro with the
    authored text — or the handover when there is no send to run.
    `yes`/`no` are this send's declared reply words (an ask's; a tell
    reads no reply and declares none); they take over from the
    previous send's when this one LANDS (`sent_landed`), so the sweep
    for replies to the previous send judges by its words."""
    if walk.channel is None or walk.channel.send is None:
        return walk.handover(
            "no channel send macro — record playbooks/channel to enable asks and tells"
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


def new_replies(walk: Walk, *, after_ask: bool = True) -> list[str]:
    """The user's messages since the gate's baseline (`reply.py` owns
    the diff), off the current thread screen."""
    assert walk.screen is not None
    gate = walk.gate
    return reply.new_incoming(
        walk.screen.rows, gate.baseline, gate.ask, after_ask=after_ask
    )


def verdict(walk: Walk, messages: list[str]) -> reply.Answer | None:
    """The gate's own words over `messages` — deny wins over anything
    else said, an uncovered message defers."""
    return reply.classify_all(
        messages, frozenset(walk.gate.yes), frozenset(walk.gate.no)
    )


def deny(walk: Walk) -> Turn:
    """The one deny disposition: no re-asks, and no second chance this
    session."""
    return walk.handover(
        "user declined the ask — acknowledge them, back out of any "
        "open checkout or cart state this task created, and wrap up"
    )


def sent_landed(walk: Walk) -> Turn:
    """A send's landing, shared by ask and tell: it must be on the
    thread; anything the user sent while the walk was off in the app (an
    earlier ask's baseline) lands here as new, and a deny among it must
    stop the walk NOW — overwriting the baseline would swallow it
    forever (deny only: an old confirm word above the fresh ask is never
    treated as consent). Then the thread is baselined. None = carry on."""
    wrong = thread_mismatch(walk)
    if wrong is not None:
        return walk.handover(f"channel send did not land on the thread ({wrong})")
    gate = walk.gate
    # The sweep needs a baseline to diff against and deny words to judge
    # by — after a tell (which declares none) there is nothing to read.
    if (
        gate.baseline
        and gate.no
        and verdict(walk, new_replies(walk, after_ask=False)) is reply.Answer.DENY
    ):
        return deny(walk)
    # The send landed: its words and its thread snapshot take over together.
    gate.yes, gate.no = gate.next_words
    assert walk.screen is not None
    gate.baseline = {label.strip() for label in walk.screen.labels}
    return None
