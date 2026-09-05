"""Tests for the boot — `channel/boot.yml` walked by the ordinary
`Program`, with `step_activate` as its last node: routing, the
declared bounds, and the rules the field data forced (act on an
unknown screen; spend nothing on a blocked call; recognize a sleeping
phone by shape, since it has no text to anchor)."""

from __future__ import annotations

import pytest
from conductor_fakes import (
    CHANNEL_OPEN,
    ELSEWHERE,
    FLOW,
    Sink,
    make_screen,
    thread_screen,
    write_channel,
    write_pack,
)
from conductor_fakes import feed as _feed
from conductor_fakes import finish as _finish
from conductor_fakes import history as _history

from physiclaw.common import gesture_vocab, paths
from physiclaw.common.listing import LISTING_HEADER
from physiclaw.conductor.drive import setup
from physiclaw.conductor.walk import walklog
from physiclaw.conductor.walk.micro import SCROLL_UP, DecisionRequest, MicroOutcome
from physiclaw.conductor.walk.program import Program
from physiclaw.conductor.walk.walklog import Outcome

THREAD = thread_screen(("买牛奶", 0.25, 0.4))
# y=0.93: where an iPhone actually prints the hint — the bottom band, not
# the top. A fixture inside `within: top` agreed with a wrong declaration
# and hid it from every test until a live boot met a real lock screen.
LOCKED = make_screen(("Swipe up for Face ID or Enter Passcode", 0.5, 0.93)).text
# Verbatim from a failed wake's own trace: a real cover prints no hint,
# only the hero clock — its WIDTH is the signal.
COVER = (
    f"{LISTING_HEADER}\n"
    '0 [text] "Thu Aug 27" [0.335,0.065,0.597,0.099] 0.94\n'
    '1 [text] "21:34" [0.101,0.084,0.842,0.239] 0.97'
)


def _boot(boot_yml: str | None = None) -> Program:
    """The wake's boot program off on-disk fixtures: the channel pack
    (thread page, open macro), one enabled playbook to offer, and the
    boot file — the scaffold's stub unless the test writes its own."""
    write_channel(CHANNEL_OPEN)
    write_pack(playbooks={"flow": FLOW})
    if boot_yml is not None:
        (paths.playbooks_dir() / "channel" / "boot.yml").write_text(
            boot_yml, encoding="utf-8"
        )
    program, _ = setup.session_setup()
    assert program is not None and program.spec.name == "boot"
    return program


def _matched(**payload: str) -> MicroOutcome:
    return MicroOutcome(out="demo/flow", reason="task", confidence=0.9, payload=payload)


# ---------- the happy paths ----------


def test_already_on_the_thread_asks_straight_away() -> None:
    o = _boot()
    h = _history()

    peek = o.advance(h)
    assert peek.tool_names() == ["note", "peek"]
    _feed(h, peek, THREAD)

    req = o.advance(h)

    assert isinstance(req, DecisionRequest)
    assert req.call == "parse_task" and req.node_id == "parse"
    assert "买牛奶" in req.listing and "demo/flow" in req.args["menu"]
    assert o.outcome is None  # not spent until the answer comes back


def test_unknown_screen_opens_the_thread_then_asks() -> None:
    """The rule the recorded sessions forced: an unrecognized screen is
    NOT a reason to quit, because `channel/open` starts at home_screen
    and recovers from anywhere. The boot's route says so: `elsewhere`
    runs the open hand, then the walk judges the landing."""
    o = _boot()
    h = _history()

    _feed(h, o.advance(h), ELSEWHERE)
    opening = o.advance(h)

    assert opening.tool_names() == ["note", gesture_vocab.RUN_MACRO]
    assert opening.tool_calls[1].arguments == {"name": "channel/open"}
    assert "(elsewhere)" in opening.tool_calls[0].arguments["summary"]

    _feed(h, opening, THREAD)
    assert isinstance(o.advance(h), DecisionRequest)


def test_locked_screen_unlocks_then_continues() -> None:
    o = _boot()
    h = _history()

    _feed(h, o.advance(h), LOCKED)
    unlock = o.advance(h)

    assert unlock.tool_names() == ["note", gesture_vocab.UNLOCK_PHONE]
    assert "(locked)" in unlock.tool_calls[0].arguments["summary"]

    # Unlocked onto some app screen → the walk restarts the route from
    # the top and the thread's `elsewhere` hand opens it.
    _feed(h, unlock, ELSEWHERE)
    assert o.advance(h).tool_calls[1].arguments == {"name": "channel/open"}


def test_unlock_landing_on_the_thread_skips_the_open_macro() -> None:
    # The phone locked while WeChat was foregrounded, so unlocking
    # restores the thread. The hand's own result view is the re-check —
    # no extra peek, and no reason to run the open macro just to arrive
    # where we are.
    o = _boot()
    h = _history()
    _feed(h, o.advance(h), LOCKED)
    unlock = o.advance(h)
    _feed(h, unlock, THREAD)

    req = o.advance(h)
    assert isinstance(req, DecisionRequest)
    assert not any(
        c.name == gesture_vocab.RUN_MACRO
        for m in h
        if getattr(m, "tool_calls", None)
        for c in m.tool_calls
    )


def test_a_clock_only_screen_unlocks_without_a_declared_anchor() -> None:
    # The failure this exists to stop, measured on the rig: a real iPhone
    # cover prints no hint text, so `ios.locked` scored 0.00, the screen
    # read `unknown`, and the boot spent BOTH open attempts driving a
    # macro's taps into a phone that was not awake to receive them —
    # ~45s each — before handing over. The matcher reads the cover by
    # shape, so the `locked` hand fires.
    o = _boot()
    h = _history()

    _feed(h, o.advance(h), COVER)
    step = o.advance(h)

    assert step.tool_calls[1].name == gesture_vocab.UNLOCK_PHONE


# ---------- resolution ----------


def test_a_matched_playbook_becomes_the_baton_and_leaves_no_trace() -> None:
    o = _boot()
    h = _history()
    _feed(h, o.advance(h), THREAD)
    o.advance(h)

    step = o.resolve(_matched(keyword="milk"))

    assert step is None
    assert o.outcome is Outcome.COMPLETED
    assert o.baton is not None and o.baton.spec.name == "flow"
    assert o.baton.values == {"keyword": "milk"}
    assert o.outputs == {"parse.playbook": "demo/flow"}
    assert o.advance(h) is None  # quiet for good
    assert walklog.load() == []  # dry: the boot records no row


@pytest.mark.parametrize(
    "outcome",
    [None, MicroOutcome(out="not_a_task", reason="chat", confidence=0.9)],
)
def test_no_playbook_leaves_the_model_on_the_thread(outcome) -> None:
    o = _boot()
    h = _history()
    _feed(h, o.advance(h), THREAD)
    o.advance(h)

    assert o.resolve(outcome) is None
    assert o.outcome is Outcome.COMPLETED and o.baton is None


# ---------- bounds and failure ----------


def test_a_blocked_call_spends_no_retry_budget() -> None:
    """The field's dominant failure: a dead bridge answers every call in
    a millisecond. One error ends the boot — retrying just burns tokens
    proving the phone is gone."""
    o = _boot()
    h = _history()

    _feed(h, o.advance(h), "peek failed: Session terminated", error=True)

    summary = _finish(o, h, o.advance(h))
    assert "blocked or failed" in summary
    assert o.outcome is Outcome.HANDOVER and o.baton is None


def test_recover_attempts_are_bounded_by_the_declared_limit() -> None:
    # unlocks and opens share the thread page's one `limit:` — the
    # scaffold's 4 covers today's worst case (two keypad races, two
    # opens) and the author may change it.
    o = _boot()
    h = _history()
    _feed(h, o.advance(h), ELSEWHERE)

    for _ in range(4):
        step = o.advance(h)
        assert step.tool_calls[1].arguments == {"name": "channel/open"}
        _feed(h, step, ELSEWHERE)  # never lands on the thread

    summary = _finish(o, h, o.advance(h))
    assert "recover tries (4) spent" in summary
    assert o.outcome is Outcome.HANDOVER


def test_a_missing_result_ends_the_boot() -> None:
    o = _boot()
    h = _history()
    h.append(o.advance(h))  # the turn, but never its result

    summary = _finish(o, h, o.advance(h))
    assert "never arrived" in summary


# ---------- scroll-for-history ----------

# The thread scrolled UP one notch: the older request appears above,
# and the bottom of this reading overlaps the top of the previous one
# (the nudge bubble) — the seam `merge_labels` finds.
OLDER_THREAD = thread_screen(("买两盒鸡蛋", 0.25, 0.3), ("继续", 0.25, 0.5))
NUDGE_THREAD = thread_screen(("继续", 0.25, 0.4))
SCROLL = MicroOutcome(out=SCROLL_UP, reason="request above", confidence=0.9)


def test_scroll_up_swipes_the_thread_and_reasks_merged() -> None:
    o = _boot()
    h = _history()
    _feed(h, o.advance(h), NUDGE_THREAD)
    assert isinstance(o.advance(h), DecisionRequest)

    swipe = o.resolve(SCROLL)
    assert swipe is not None and swipe.tool_names() == ["note", "swipe"]
    assert swipe.tool_calls[1].arguments["direction"] == "down"
    assert o.outcome is None  # not spent — the re-ask is coming

    _feed(h, swipe, OLDER_THREAD)
    req = o.advance(h)

    assert isinstance(req, DecisionRequest)
    # Seamed at the overlapping rows, oldest first, duplicates intact.
    assert req.listing == "MyChat\n买两盒鸡蛋\n继续"


def test_scroll_budget_spent_resolves_as_no_task() -> None:
    o = _boot()
    h = _history()
    _feed(h, o.advance(h), NUDGE_THREAD)
    o.advance(h)
    for _ in range(2):  # the stub's `limit: {scrolls: 2}`
        swipe = o.resolve(SCROLL)
        _feed(h, swipe, NUDGE_THREAD)
        o.advance(h)

    step = o.resolve(SCROLL)  # third — budget spent

    assert step is None
    assert o.outcome is Outcome.COMPLETED and o.baton is None


# ---------- the file is the user's ----------


def test_the_boot_route_is_authorable() -> None:
    # A user who wants no unlock and one open: the hands and limits
    # are theirs, and the walk runs exactly what the file says.
    o = _boot(
        "name: boot\n"
        "description: mine\n"
        "route:\n"
        "  - page: thread\n"
        "    recover: {elsewhere: {macro: open}}\n    tries: 1\n"
        "  - select: read\n"
        "    limit: {scrolls: 0}\n"
    )
    h = _history()
    _feed(h, o.advance(h), LOCKED)

    summary = _finish(o, h, o.advance(h))
    assert "declares no `locked` recover hand" in summary

    o2 = _boot()  # the file stays: written once, never touched
    assert o2.spec.nodes[-1].id == "read"


def test_a_disabled_boot_means_a_plain_session() -> None:
    write_channel(CHANNEL_OPEN)
    write_pack(playbooks={"flow": FLOW})
    (paths.playbooks_dir() / "channel" / "boot.yml").write_text(
        "name: boot\ndescription: off\nenabled: false\nroute:\n"
        "  - page: thread\n  - select: parse\n",
        encoding="utf-8",
    )

    program, hidden = setup.session_setup()

    assert program is None and "channel/open" in hidden


# ---------- what the log says ----------


def test_every_reading_logs_its_verdict(caplog: pytest.LogCaptureFixture) -> None:
    # The number-one debugging question of a walk is "what did it think
    # the screen was?" — answered in the runtime log after every
    # reading, with the page and both scores, not only on a mismatch.
    caplog.set_level("INFO", logger="physiclaw.conductor")
    o = _boot()
    h = _history()
    _feed(h, o.advance(h), THREAD)

    o.advance(h)

    (line,) = [
        r.getMessage()
        for r in caplog.records
        if " read after peek — " in r.getMessage()
    ]
    assert line.startswith(
        "conductor: channel/boot read after peek — match channel.thread (score "
    )


def test_the_boot_hands_on_as_a_walk_event_in_the_session() -> None:
    # Dry as it is, the boot's conclusion lands in events.jsonl, so the
    # summary's `walks` list opens with which playbook it picked.
    write_channel(CHANNEL_OPEN)
    write_pack(playbooks={"flow": FLOW})
    sink = Sink()
    program, _ = setup.session_setup(events=sink)
    assert program is not None
    h = _history()
    _feed(h, program.advance(h), THREAD)
    program.advance(h)

    program.resolve(_matched(keyword="milk"))

    assert [
        (e["event"], e["app"], e["playbook"], e["outcome"], e["reason"])
        for e in sink.events
    ] == [("walk", "channel", "boot", "completed", "hands over to demo/flow")]
    # …and the baton records into the same session.
    assert program.baton is not None and program.baton.record.events is sink
