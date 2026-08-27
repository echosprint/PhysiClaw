"""Tests for `physiclaw.agent.conductor.overture` — the boot to the
user's thread: routing, bounds, and the two rules the field data forced
(act on an unknown screen; spend nothing on a blocked call)."""

from __future__ import annotations

import pytest
from conductor_fakes import ELSEWHERE, make_screen, thread_screen
from conductor_fakes import feed as _feed
from conductor_fakes import history as _history

from physiclaw.agent.conductor import overture as ov
from physiclaw.agent.conductor.channel import Channel
from physiclaw.agent.conductor.micro import DecisionRequest, MicroOutcome
from physiclaw.agent.conductor.pages import AnchorDecl, PageDecl, PagePrint
from physiclaw.agent.conductor.playbook import Pack, Playbook
from physiclaw.agent.conductor.setup import Activation
from physiclaw.agent.engine.dto import (
    ToolResultMessage,
)
from physiclaw.common import gesture_vocab

THREAD = thread_screen(("买牛奶", 0.25, 0.4))
LOCKED = make_screen(("Swipe up for Face ID or Enter Passcode", 0.5, 0.19)).text


def _prints() -> list[PagePrint]:
    return [
        PagePrint(
            app="channel",
            decl=PageDecl(name="thread", anchors=(AnchorDecl("MyChat"),)),
        ),
        PagePrint(
            app="ios",
            decl=PageDecl(
                name="locked",
                anchors=(
                    AnchorDecl("Swipe up for Face ID or Enter Passcode", region="top"),
                ),
            ),
        ),
    ]


def _spec() -> Playbook:
    return Playbook(
        app="demo",
        name="flow",
        description="d",
        enabled=True,
        inputs=(),
        mandate=None,
        nodes=(),
    )


def _overture(*, open_macro: str | None = "channel/open") -> ov.Overture:
    macros = {}
    if open_macro is not None:
        from physiclaw.agent.macros.model import Macro

        macros["channel/open"] = Macro(
            name="open", description="d", enabled=True, inputs=(), steps=()
        )
    channel = Channel(prints=_prints()[:1], macros=macros)
    activation = Activation(
        entries={"demo/flow": (_spec(), Pack("demo", {}, {}, {}))}, channel=channel
    )
    return ov.Overture(channel=channel, activation=activation, prints=_prints())


# ---------- the happy paths ----------


def test_already_on_the_thread_asks_straight_away() -> None:
    o = _overture()
    h = _history()

    peek = o.advance(h)
    assert peek.tool_names() == ["note", "peek"]
    _feed(h, peek, THREAD)

    req = o.advance(h)

    assert isinstance(req, DecisionRequest)
    assert "买牛奶" in req.listing
    assert o.done is False  # not spent until the answer comes back


def test_unknown_screen_opens_the_thread_then_asks() -> None:
    """The rule the recorded sessions forced: an unrecognized screen is
    NOT a reason to quit, because `channel/open` starts at home_screen
    and recovers from anywhere. Act, then judge the landing."""
    o = _overture()
    h = _history()

    _feed(h, o.advance(h), ELSEWHERE)
    opening = o.advance(h)

    assert opening.tool_names() == ["note", gesture_vocab.RUN_MACRO]
    assert opening.tool_calls[1].arguments == {"name": "channel/open"}

    _feed(h, opening, THREAD)
    assert isinstance(o.advance(h), DecisionRequest)


def test_locked_screen_unlocks_then_continues() -> None:
    o = _overture()
    h = _history()

    _feed(h, o.advance(h), LOCKED)
    unlock = o.advance(h)

    assert unlock.tool_names() == ["note", gesture_vocab.UNLOCK_PHONE]

    # Unlocked onto some app screen → recover via the open macro.
    _feed(h, unlock, ELSEWHERE)
    assert o.advance(h).tool_calls[1].arguments == {"name": "channel/open"}


def test_unlock_landing_on_the_thread_skips_the_open_macro() -> None:
    # The phone locked while WeChat was foregrounded, so unlocking
    # restores the thread. `unlock_phone` replies with its own attached
    # view, so the landing is already in hand — no extra peek, and no
    # reason to run the open macro just to arrive where we are.
    o = _overture()
    h = _history()
    _feed(h, o.advance(h), LOCKED)
    unlock = o.advance(h)
    _feed(h, unlock, THREAD)

    assert isinstance(o.advance(h), DecisionRequest)
    assert o._opens == 0


def test_unlock_is_not_spent_on_an_unlocked_phone() -> None:
    o = _overture()
    h = _history()

    _feed(h, o.advance(h), ELSEWHERE)
    o.advance(h)

    assert o._unlocks == 0  # 20-40s spent only on a positive lock match


# ---------- resolution ----------


def test_a_matched_playbook_becomes_the_program() -> None:
    o = _overture()
    h = _history()
    _feed(h, o.advance(h), THREAD)
    o.advance(h)

    step = o.resolve(
        MicroOutcome(out="demo/flow", reason="task", confidence=0.9, payload={})
    )

    assert step is None
    assert o.done is True
    assert o.program is not None and o.program.spec.name == "flow"


@pytest.mark.parametrize(
    "outcome",
    [None, MicroOutcome(out="not_a_task", reason="chat", confidence=0.9)],
)
def test_no_playbook_leaves_the_model_on_the_thread(outcome) -> None:
    o = _overture()
    h = _history()
    _feed(h, o.advance(h), THREAD)
    o.advance(h)

    assert o.resolve(outcome) is None
    assert o.done is True and o.program is None


# ---------- bounds and failure ----------


def test_a_blocked_call_spends_no_retry_budget() -> None:
    """The field's dominant failure: a dead bridge answers every call in
    a millisecond. One error ends the boot — retrying just burns tokens
    proving the phone is gone."""
    o = _overture()
    h = _history()

    _feed(h, o.advance(h), "peek failed: Session terminated", error=True)

    assert o.advance(h) is None
    assert o.done is True and o.program is None
    assert o._opens == 0 and o._unlocks == 0


def test_open_attempts_are_bounded() -> None:
    o = _overture()
    h = _history()
    _feed(h, o.advance(h), ELSEWHERE)

    for _ in range(ov.OPEN_TRIES):
        step = o.advance(h)
        assert step is not None
        _feed(h, step, ELSEWHERE)  # never lands on the thread

    assert o.advance(h) is None
    assert o.done is True


def test_unlock_attempts_are_bounded() -> None:
    o = _overture()
    h = _history()
    _feed(h, o.advance(h), LOCKED)

    for _ in range(ov.UNLOCK_TRIES):
        step = o.advance(h)
        assert step is not None
        _feed(h, step, LOCKED)  # unlock keeps losing the keypad race

    assert o.advance(h) is None
    assert o.done is True


def test_a_missing_result_ends_the_boot() -> None:
    o = _overture()
    h = _history()
    h.append(o.advance(h))  # the turn, but never its result

    assert o.advance(h) is None
    assert o.done is True


def test_a_crash_hands_over_instead_of_raising(mocker) -> None:
    o = _overture()
    mocker.patch.object(ov, "match_screen", side_effect=RuntimeError("boom"))
    h = _history()
    _feed(h, o.advance(h), THREAD)

    assert o.advance(h) is None
    assert o.done is True


# ---------- stand-by (no `open` macro to drive with) ----------


def test_without_an_open_macro_it_watches_instead_of_driving() -> None:
    o = _overture(open_macro=None)
    h = _history()

    # No turn synthesized, and NOT done — it may still act later.
    assert o.advance(h) is None
    assert o.done is False

    h.append(ToolResultMessage(tool_call_id="t1", content=ELSEWHERE))
    assert o.advance(h) is None
    assert o.done is False

    # The model reaches the thread on its own — now it asks.
    h.append(ToolResultMessage(tool_call_id="t2", content=THREAD))
    assert isinstance(o.advance(h), DecisionRequest)
