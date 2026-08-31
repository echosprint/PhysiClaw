"""Tests for `physiclaw.conductor.program` — arming (validation +
the arm file), fail-open loading, and the walk: synthesized turns,
fingerprint checks, resume-by-locate, and hand-over on everything this
phase refuses to guess about."""

from __future__ import annotations

import json

import pytest
from conductor_fakes import (
    ELSEWHERE,
    LEDGERED,
    make_screen,
    write_pack,
)
from conductor_fakes import (
    feed as _feed,
)
from conductor_fakes import (
    finish as _finish,
)
from conductor_fakes import (
    history as _history,
)
from conductor_fakes import (
    thread_screen as _thread,
)

from physiclaw.common import paths
from physiclaw.conductor import channel, memory, program, reconcile, setup, suspension
from physiclaw.conductor.playbook import GATE_MAX_REVISIONS, PlaybookError

# The shared fixture pages are three DISTINCT pages so `_locate`
# (resume past matching verifies) never fast-forwards a freshly booted
# walk: the start page matches no move's landing.
FLOW = """\
description: two moves
inputs:
  keyword:
    description: what to search
route:
  - page: home
  - do: open
    macro: open-app
    with: {message: "{inputs.keyword}"}
  - page: results
  - do: search
    macro: add-cart
    with: {message: "go"}
  - page: done
"""

# Same moves, then a decide.
BRANCH = (
    FLOW
    + """\
  - decide: choose
    uses: choose_item
    with: {criteria: "cheapest"}
    routes: {pick: escalate, scroll: escalate, none_fit: escalate, escalate: escalate}
"""
)

HOME = make_screen(("Files", 0.5, 0.1)).text
RESULTS = make_screen(("综合", 0.5, 0.1)).text
DONE = make_screen(("AllDone", 0.5, 0.1)).text


def _program(app: str = "demo", name: str = "flow", **values) -> program.Program:
    """Build the walk the way `playbooks run` does — the factory that
    replaced arming as the way to get a Program without a wake."""
    spec, pack = setup.load_spec(app, name, require_live=False)
    return setup.build_program(
        app, spec, pack, setup.resolve_inputs(spec, values), channel.load_channel()
    )


# ---------- the walk ----------


def test_walk_runs_both_legs_then_completes() -> None:
    write_pack(playbooks={"flow": FLOW})
    p = _program(keyword="milk")
    h = _history()

    peek = p.advance(h)
    assert peek is not None and peek.synthesized
    assert peek.tool_names() == ["note", "peek"]

    _feed(h, peek, HOME)  # the start page — the walk begins
    leg1 = p.advance(h)
    assert leg1 is not None and leg1.tool_names() == ["note", "run_macro"]
    assert leg1.tool_calls[1].arguments == {
        "name": "demo/open-app",
        "inputs": {"message": "milk"},  # {inputs.keyword} resolved from the arm
    }

    _feed(h, leg1, RESULTS)  # landed on `results` → next move (enter holds too)
    leg2 = p.advance(h)
    assert leg2 is not None
    assert leg2.tool_calls[1].arguments["name"] == "demo/add-cart"

    _feed(h, leg2, DONE)  # landed on `done` → playbook complete
    summary = _finish(p, h, p.advance(h))
    assert "walk demo/flow completed" in summary


def test_locate_resumes_past_completed_legs() -> None:
    # A killed session's next wake: the screen already shows leg 1's
    # verify page, so the walk resumes at leg 2 — no repeated gestures.
    write_pack(playbooks={"flow": FLOW})
    p = _program(keyword="milk")
    h = _history()

    peek = p.advance(h)
    assert peek is not None
    _feed(h, peek, RESULTS)  # leg 1's landing page — its outcome holds
    nxt = p.advance(h)

    assert nxt is not None
    assert nxt.tool_calls[1].arguments["name"] == "demo/add-cart"


def test_verify_mismatch_rescues_then_hands_over() -> None:
    # A wrong known page at verify is a mechanical deviation: the back
    # rung tries to pop out of it (bounded) before the model is woken.
    write_pack(playbooks={"flow": FLOW})
    p = _program(keyword="milk")
    h = _history()

    _feed(h, p.advance(h), HOME)
    leg1 = p.advance(h)
    assert leg1 is not None
    _feed(h, leg1, HOME)  # landed on the WRONG known page

    settle = p.advance(h)  # rung 0: maybe just mid-transition
    assert settle is not None and settle.tool_names() == ["note", "peek"]
    _feed(h, settle, HOME)  # settled — genuinely the wrong page
    back1 = p.advance(h)
    assert back1 is not None and back1.tool_names() == ["note", "go_back"]
    _feed(h, back1, HOME)
    back2 = p.advance(h)
    assert back2 is not None and back2.tool_names() == ["note", "go_back"]
    _feed(h, back2, HOME)

    summary = _finish(p, h, p.advance(h))
    assert "did not land" in summary and "rescue tried: back×2" in summary


def test_leg_verifying_a_builtin_page_hands_over() -> None:
    # `ios` pages LOAD now — the conductor matches against them itself —
    # but a playbook still may not ACT on one: legs run this pack's
    # macros and land on this pack's pages. The loader change made this
    # guarantee easy to lose by accident, so pin it.
    write_pack(
        playbooks={
            "flow": FLOW.replace("  - page: results\n", "  - page: ios.locked\n")
        }
    )
    p = _program(keyword="milk")
    h = _history()

    _feed(h, p.advance(h), HOME)

    _finish(p, h, p.advance(h))


def test_error_result_hands_over() -> None:
    write_pack(playbooks={"flow": FLOW})
    p = _program(keyword="milk")
    h = _history()

    _feed(h, p.advance(h), HOME)
    leg1 = p.advance(h)
    assert leg1 is not None
    _feed(h, leg1, "BLOCKED — not executed", error=True)

    summary = _finish(p, h, p.advance(h))
    assert "blocked or failed" in summary


def test_decide_yields_a_request_and_an_unresolved_one_hands_over() -> None:
    from physiclaw.conductor.micro import DecisionRequest

    write_pack(playbooks={"branch": BRANCH})
    p = _program(name="branch", **{"keyword": "milk"})
    h = _history()

    _feed(h, p.advance(h), HOME)  # the start page
    _feed(h, p.advance(h), RESULTS)  # move open landed
    _feed(h, p.advance(h), DONE)  # move search landed

    req = p.advance(h)
    assert isinstance(req, DecisionRequest)  # the conductor brokers it
    _finish(p, h, p.resolve(None))  # a failed micro-call hands over


def test_program_advance_never_raises() -> None:
    write_pack(playbooks={"flow": FLOW})
    p = _program(keyword="milk")

    # A malformed history (no pending result will ever match) must
    # degrade to a hand-over, not an exception.
    p.advance(_history())
    step = p.advance(_history())  # missing result → the handover brief
    assert step is not None and step.tool_names() == ["note", "peek"]
    assert p.advance(_history()) is None  # then permanently quiet


# ---------- decisions ----------

PICKY = """\
description: pick flow
inputs:
  keyword:
    description: what
route:
  - page: home
  - do: open
    macro: open-app
    with: {message: "{inputs.keyword}"}
  - page: results
  - decide: choose
    uses: choose_item
    with: {criteria: "cheapest {inputs.keyword}"}
    routes: {pick: use, scroll: choose, none_fit: escalate, escalate: escalate}
  - do: use
    macro: add-cart
    with: {message: "{choose.pick}"}
  - page: done
"""


def _at_decision():
    """Arm PICKY and walk it to the choose move's DecisionRequest."""
    from physiclaw.conductor.micro import DecisionRequest

    write_pack(playbooks={"picky": PICKY})
    p = _program(name="picky", **{"keyword": "milk"})
    h = _history()
    _feed(h, p.advance(h), HOME)  # the start page
    _feed(h, p.advance(h), RESULTS)  # move open landed on results
    req = p.advance(h)
    assert isinstance(req, DecisionRequest), req
    return p, h, req


def test_decide_emits_a_request_off_the_decide_time_screen() -> None:
    _, _, req = _at_decision()

    assert req.node_id == "choose" and req.call == "choose_item"
    assert req.args == {"criteria": "cheapest milk"}  # {inputs.keyword} resolved
    assert [c.key for c in req.candidates] == ["综合"]  # the RESULTS row


def test_pick_taps_the_row_then_output_feeds_the_next_leg() -> None:
    from physiclaw.conductor.micro import MicroOutcome

    p, h, req = _at_decision()
    picked = req.candidates[0]

    tap = p.resolve(
        MicroOutcome(out="pick", reason="cheapest", confidence=0.9, picked=picked)
    )
    assert tap is not None and tap.tool_names() == ["note", "tap"]
    assert tap.tool_calls[1].arguments == {"bbox": list(picked.bbox)}
    assert "decided choose: pick" in tap.tool_calls[0].arguments["summary"]

    _feed(h, tap, RESULTS)  # post-tap screen still reads results — `use` enters
    leg = p.advance(h)
    assert leg is not None
    assert leg.tool_calls[1].arguments["inputs"] == {"message": "综合"}


def test_scroll_self_loop_swipes_and_reasks_until_max_visits() -> None:
    from physiclaw.conductor.micro import DecisionRequest, MicroOutcome

    p, h, req = _at_decision()  # visit 1
    scroll = MicroOutcome(out="scroll", reason="maybe below", confidence=0.9)

    for visit in (2, 3):
        swipe = p.resolve(scroll)
        assert swipe is not None and swipe.tool_names() == ["note", "swipe"]
        _feed(h, swipe, RESULTS)
        req = p.advance(h)
        assert isinstance(req, DecisionRequest), (visit, req)

    # The default max_visits (3) is spent — the next re-ask hands over.
    swipe = p.resolve(scroll)
    assert swipe is not None
    _feed(h, swipe, RESULTS)
    summary = _finish(p, h, p.advance(h))
    assert "max_visits" in summary


def test_escalate_routed_outcome_hands_over() -> None:
    from physiclaw.conductor.micro import MicroOutcome

    p, h, _ = _at_decision()

    step = p.resolve(MicroOutcome(out="none_fit", reason="nothing", confidence=0.9))

    summary = _finish(p, h, step)
    assert "escalate" in summary


def test_failed_micro_call_hands_over() -> None:
    p, h, _ = _at_decision()

    summary = _finish(p, h, p.resolve(None))
    assert "failed or under-confident" in summary


# ---------- the gate, suspending, activation ----------

CHANNEL_PAGES = """\
thread:
  anchors: ["MyChat"]
"""

CHANNEL_SEND = """\
name: send
description: send to the user
inputs:
  message:
    description: text
steps:
  - name: clip
    tool: send_to_clipboard
    with: {text: "{message}"}
"""

CHANNEL_OPEN = """\
name: open
description: open the thread
steps:
  - name: go
    tool: tap
    with: {label: t, bbox: [0.1, 0.1, 0.2, 0.2]}
"""

GATED = """\
description: 买牛奶
inputs:
  keyword:
    description: what
budget: 100
route:
  - page: home
  - do: open
    macro: open-app
    with: {message: "{inputs.keyword}"}
  - page: results
  - ask: gate
    approve: payment
    message: "已选好{inputs.keyword}，合计 ¥{ask.total}。回复 好的 确认支付，或 不用 取消。"
    over_budget_message: "已选好{inputs.keyword}，合计 ¥{ask.total}，已超出预算 ¥{ask.cap}。回复 好的 确认支付，或 不用 取消。"
    return: open-app
  - do: pay
    macro: add-cart
    with: {message: "pay"}
    irreversible: payment
  - page: home
"""


def _write_channel() -> None:
    from conductor_fakes import compose_pack_doc

    root = paths.playbooks_dir() / "channel"
    (root / "macros" / "send").mkdir(parents=True, exist_ok=True)
    (root / "macros" / "open").mkdir(parents=True, exist_ok=True)
    (root / "PLAYBOOK.yml").write_text(
        compose_pack_doc("channel", CHANNEL_PAGES), encoding="utf-8"
    )
    (root / "macros" / "send" / "MACRO.yml").write_text(CHANNEL_SEND, encoding="utf-8")
    (root / "macros" / "open" / "MACRO.yml").write_text(CHANNEL_OPEN, encoding="utf-8")


def _sheet(total: str = "¥45") -> str:
    return make_screen(("综合", 0.5, 0.1), (f"合计 {total}", 0.5, 0.5)).text


def _at_gate(total: str = "¥45", playbook: str = GATED):
    """Arm the gated playbook (with a channel pack) and walk to the sent
    ask; `total` is what the payment sheet shows."""
    _write_channel()
    write_pack(playbooks={"pay": playbook})
    p = _program(name="pay", keyword="milk")
    assert p.channel is not None and p.channel.send == "channel/send"
    h = _history()
    _feed(h, p.advance(h), HOME)  # the start page
    _feed(h, p.advance(h), _sheet(total))  # move open landed on the sheet
    send = p.advance(h)
    assert send is not None and send.tool_names() == ["note", "run_macro"]
    assert send.tool_calls[1].arguments["name"] == "channel/send"
    return p, h, send


def _reply_arrives(p, h, send, bubble: str):
    """Drive one wait+peek round ending with `bubble` as the new reply;
    returns the walk's next step."""
    ask = send.tool_calls[1].arguments["inputs"]["message"]
    _feed(h, send, _thread((ask, 0.75, 0.3)))
    _feed(h, p.advance(h), "waited")
    peek = p.advance(h)
    assert peek.tool_names() == ["note", "peek"]
    _feed(h, peek, _thread((ask, 0.75, 0.3), (bubble, 0.25, 0.5)))
    return p.advance(h)


def _suspend_via_silence(p, h, send) -> str:
    """Drive the full silent-round cycle to the suspension; returns the ask."""
    ask = send.tool_calls[1].arguments["inputs"]["message"]
    thread = _thread((ask, 0.75, 0.3))
    _feed(h, send, thread)
    step = p.advance(h)
    for _ in range(program.SILENCE_ROUNDS):
        assert step.tool_names() == ["note", "wait"]
        _feed(h, step, "waited")
        peek = p.advance(h)
        _feed(h, peek, thread)
        step = p.advance(h)
    assert step.tool_names() == ["note", "end_session"]
    assert step.tool_calls[1].arguments["status"] == "WAIT"
    return ask


def test_gate_ask_quotes_the_total_within_budget() -> None:
    _, _, send = _at_gate()

    ask = send.tool_calls[1].arguments["inputs"]["message"]
    assert "¥45" in ask and "好的" in ask
    assert "超出预算" not in ask


def test_gate_over_cap_ask_discloses_the_breach() -> None:
    _, _, send = _at_gate(total="¥200")

    ask = send.tool_calls[1].arguments["inputs"]["message"]
    assert "超出预算" in ask and "¥200" in ask and "¥100" in ask
    assert "好的" in ask  # plain consent still opens it — ruled


def test_arm_warns_when_a_gate_ask_quotes_no_deny_word() -> None:
    # Advisory, never blocking: an ask in a language the word lists
    # don't cover still arms and works — every reply just rides the
    # LLM tier. The author is told the cost, not refused.
    quiet = GATED.replace("回复 好的 确认支付，或 不用 取消", "veuillez répondre")
    _write_channel()
    write_pack(playbooks={"pay": quiet})

    spec, _ = setup.load_spec("demo", "pay", require_live=False)
    warnings = setup.readiness_warnings(spec)

    assert len(warnings) == 2  # message and over_message alike
    assert all("LLM check" in w for w in warnings)


def test_gate_confirm_returns_and_pays_under_the_predicates() -> None:
    p, h, send = _at_gate()

    back = _reply_arrives(p, h, send, "好的")  # confirmed → the return macro
    assert back.tool_names() == ["note", "run_macro"]
    assert back.tool_calls[1].arguments["name"] == "demo/open-app"
    assert "user confirmed" in back.tool_calls[0].arguments["summary"]

    _feed(h, back, _sheet())  # back on the sheet, same total
    pay = p.advance(h)
    assert pay is not None and pay.tool_calls[1].arguments["name"] == "demo/add-cart"


def test_gate_blocks_when_the_sheet_changed_after_consent() -> None:
    p, h, send = _at_gate()
    back = _reply_arrives(p, h, send, "好的")

    _feed(h, back, _sheet("¥69"))  # the total drifted after the confirm

    summary = _finish(p, h, p.advance(h))  # staleness predicate → hand over
    assert "sheet changed after consent" in summary


def test_gate_deny_hands_over_without_reasking() -> None:
    p, h, send = _at_gate()

    step = _reply_arrives(p, h, send, "不用")

    summary = _finish(p, h, step)
    assert "user declined" in summary and "back out" in summary


def test_gate_unclear_reply_goes_to_the_llm_tier() -> None:
    from physiclaw.conductor.micro import (
        CONFIRM_REPLY,
        DecisionRequest,
        MicroOutcome,
    )

    p, h, send = _at_gate()

    req = _reply_arrives(p, h, send, "那就来一份吧")
    assert isinstance(req, DecisionRequest) and req.call == CONFIRM_REPLY
    assert "那就来一份吧" in req.args["reply"]

    back = p.resolve(MicroOutcome(out="confirm", reason="yes", confidence=0.9))
    assert back.tool_calls[1].arguments["name"] == "demo/open-app"


def test_gate_silence_suspends_and_resumes_on_next_wake() -> None:
    p, h, send = _at_gate()
    ask = _suspend_via_silence(p, h, send)

    # Next wake: the suspended walk resumes straight into a reply check.
    resumed = setup.load_suspended()
    assert resumed is not None and setup.load_suspended() is None  # one-shot
    h2 = _history()
    peek = resumed.advance(h2)
    assert peek.tool_names() == ["note", "peek"]
    _feed(h2, peek, _thread((ask, 0.75, 0.3), ("好的", 0.25, 0.5)))
    back = resumed.advance(h2)
    assert back.tool_calls[1].arguments["name"] == "demo/open-app"


def test_gate_resume_off_thread_reopens_via_channel_open() -> None:
    p, h, send = _at_gate()
    ask = _suspend_via_silence(p, h, send)

    resumed = setup.load_suspended()
    h2 = _history()
    peek = resumed.advance(h2)
    _feed(h2, peek, ELSEWHERE)  # user left the phone on another app

    reopen = resumed.advance(h2)
    assert reopen.tool_calls[1].arguments["name"] == "channel/open"
    _feed(h2, reopen, _thread((ask, 0.75, 0.3), ("好的", 0.25, 0.5)))
    back = resumed.advance(h2)
    assert back.tool_calls[1].arguments["name"] == "demo/open-app"


CONFIRMING = """\
description: 汇报进展
inputs:
  keyword:
    description: what
route:
  - page: home
  - do: open
    macro: open-app
    with: {message: "{inputs.keyword}"}
  - page: results
  - tell: tell
    message: "已下单{inputs.keyword}，稍后汇报进度"
  - do: wrap
    macro: add-cart
    with: {message: "done"}
  - page: done
"""


def test_confirm_sends_suspends_and_resumes_past_itself() -> None:
    _write_channel()
    write_pack(playbooks={"notify": CONFIRMING})
    p = _program(name="notify", keyword="milk")
    h = _history()
    _feed(h, p.advance(h), HOME)  # the start page
    _feed(h, p.advance(h), RESULTS)  # move open landed

    send = p.advance(h)
    assert send.tool_calls[1].arguments["name"] == "channel/send"
    text = send.tool_calls[1].arguments["inputs"]["message"]
    # `message:` IS the sent text — refs filled, nothing code-appended.
    assert text == "已下单milk，稍后汇报进度"

    _feed(h, send, _thread((text, 0.75, 0.3)))
    susp = p.advance(h)
    assert susp.tool_names() == ["note", "end_session"]
    assert susp.tool_calls[1].arguments["status"] == "WAIT"

    # Resume: one thread read first (the wake may BE a cancel reply),
    # then the walk continues PAST the confirm node at its stored idx.
    resumed = setup.load_suspended()
    assert resumed is not None
    h2 = _history()
    check = resumed.advance(h2)
    assert check.tool_names() == ["note", "peek"]
    _feed(h2, check, _thread((text, 0.75, 0.3)))  # nothing new since the send
    peek = resumed.advance(h2)
    _feed(h2, peek, RESULTS)  # `wrap` enters where the walk left off
    leg = resumed.advance(h2)
    assert leg.tool_calls[1].arguments["name"] == "demo/add-cart"


def test_suspended_confirm_resume_reads_a_cancel() -> None:
    # The user replies "cancel" to the CONFIRM's message; the reply
    # itself wakes the device. The resumed walk must read it and stop —
    # not barrel on into the remaining legs.
    _write_channel()
    write_pack(playbooks={"notify": CONFIRMING})
    p = _program(name="notify", keyword="milk")
    h = _history()
    _feed(h, p.advance(h), HOME)
    _feed(h, p.advance(h), RESULTS)
    send = p.advance(h)
    text = send.tool_calls[1].arguments["inputs"]["message"]
    _feed(h, send, _thread((text, 0.75, 0.3)))
    p.advance(h)  # suspend

    resumed = setup.load_suspended()
    assert resumed is not None
    h2 = _history()
    check = resumed.advance(h2)
    _feed(h2, check, _thread((text, 0.75, 0.3), ("cancel", 0.25, 0.5)))
    summary = _finish(resumed, h2, resumed.advance(h2))  # deny — the walk stops
    assert "user declined" in summary


def test_suspended_confirm_resume_off_thread_reopens_then_continues() -> None:
    # Banner wake: the screen is some other app. The check reopens the
    # thread once; with no cancel there, the walk resumes normally.
    _write_channel()
    write_pack(playbooks={"notify": CONFIRMING})
    p = _program(name="notify", keyword="milk")
    h = _history()
    _feed(h, p.advance(h), HOME)
    _feed(h, p.advance(h), RESULTS)
    send = p.advance(h)
    text = send.tool_calls[1].arguments["inputs"]["message"]
    _feed(h, send, _thread((text, 0.75, 0.3)))
    p.advance(h)  # suspend

    resumed = setup.load_suspended()
    h2 = _history()
    check = resumed.advance(h2)
    _feed(h2, check, HOME)  # not the thread
    reopen = resumed.advance(h2)
    assert reopen.tool_calls[1].arguments["name"] == "channel/open"
    _feed(h2, reopen, _thread((text, 0.75, 0.3)))
    peek = resumed.advance(h2)
    _feed(h2, peek, RESULTS)
    leg = resumed.advance(h2)
    assert leg.tool_calls[1].arguments["name"] == "demo/add-cart"


# ---------- activation / session_setup ----------


def test_session_setup_builds_the_overture_and_hidden_registry() -> None:
    _write_channel()
    write_pack(playbooks={"flow": FLOW})

    prog, overture, hidden = setup.session_setup()

    assert prog is None
    assert overture is not None
    activation = overture._activation
    assert tuple(activation.entries) == ("demo/flow",)
    menu = activation._menu()
    assert "demo/flow" in menu and "keyword" in menu
    assert set(hidden) == {
        "channel/send",
        "channel/open",
        "demo/open-app",
        "demo/add-cart",
    }


def test_session_setup_prefers_a_suspended_walk_over_the_overture() -> None:
    from physiclaw.common.logger import write_json_atomic

    _write_channel()
    write_pack(playbooks={"flow": FLOW})
    write_json_atomic(
        suspension.suspended_path(),
        {
            "schema": suspension.SUSPENDED_SCHEMA,
            "app": "demo",
            "playbook": "flow",
            "idx": 0,
            "values": {"keyword": "milk"},
        },
    )

    prog, overture, hidden = setup.session_setup()

    assert prog is not None and prog.channel is not None
    assert overture is None


def test_activation_builds_a_request_over_the_thread_screen() -> None:
    from physiclaw.common.listing import Screen
    from physiclaw.conductor.micro import PARSE_TASK, MicroOutcome

    _write_channel()
    write_pack(playbooks={"flow": FLOW})
    _, overture, _ = setup.session_setup()
    assert overture is not None
    activation = overture._activation

    # The caller establishes the screen IS the thread (the overture drove
    # there, or watched the model arrive) — this turns it into the call.
    req = activation.request(Screen.read(_thread(("买牛奶", 0.25, 0.4))))
    assert req is not None and req.call == PARSE_TASK
    assert "买牛奶" in req.listing and "demo/flow" in req.args["menu"]

    prog = activation.build(
        MicroOutcome(
            out="demo/flow",
            reason="purchase task",
            confidence=0.9,
            payload={"keyword": "牛奶"},
        )
    )
    assert prog is not None
    assert prog.values == {"keyword": "牛奶"} and prog.channel is activation.channel


def test_activation_menu_renders_the_input_example() -> None:
    # The authored `example:` is the extraction hint — it must reach the
    # parse_task menu, or a keyword input has only prose to shape its
    # value ("五常大米 5kg" beats any rule about quantity words).
    _write_channel()
    spec = FLOW.replace(
        "    description: what to search\n",
        "    description: what to search\n    example: rice 5kg\n",
    )
    write_pack(playbooks={"flow": spec})
    _, overture, _ = setup.session_setup()
    assert overture is not None

    menu = overture._activation._menu()

    assert "keyword (what to search; e.g. rice 5kg)" in menu


def test_activation_rejects_unresolvable_inputs_and_not_a_task() -> None:
    from physiclaw.conductor.micro import MicroOutcome

    _write_channel()
    write_pack(playbooks={"flow": FLOW})
    _, overture, _ = setup.session_setup()
    assert overture is not None
    activation = overture._activation

    assert activation.build(None) is None
    assert (
        activation.build(MicroOutcome(out="not_a_task", reason="chat", confidence=0.9))
        is None
    )
    # keyword is required; an empty extraction cannot activate.
    assert (
        activation.build(
            MicroOutcome(out="demo/flow", reason="task", confidence=0.9, payload={})
        )
        is None
    )


def test_scaffolded_channel_pack_parses_and_loads_disabled() -> None:
    from physiclaw.conductor import scaffold

    scaffold.init_pack("channel")

    ch = channel.load_channel()
    # Pages parse and the thread page exists; the macros are scaffolded
    # DISABLED (rehearse, then enable), so the sends stay unavailable.
    assert ch is not None
    assert ch.send is None and ch.open is None
    from physiclaw.conductor.playbook import load_pack

    pack = load_pack("channel")
    assert set(pack.macros) == {"send", "open"} and not pack.macro_errors


def test_gate_revise_hands_over_with_the_request() -> None:
    # "好的，但是买两盒" is a change request, not a confirmation — the LLM
    # tier answers revise and the model takes over to adjust the order.
    from physiclaw.conductor.micro import DecisionRequest, MicroOutcome

    p, h, send = _at_gate()

    req = _reply_arrives(p, h, send, "好的，但是买两盒")
    assert isinstance(req, DecisionRequest)

    handed = p.resolve(MicroOutcome(out="revise", reason="wants two", confidence=0.9))
    summary = _finish(p, h, handed)  # hand over — the model adjusts the order
    assert "asked for changes" in summary


def test_gate_ask_is_the_filled_template_exactly() -> None:
    # The playbook owns every word; the conductor owns only the slots:
    # the ask is `message:` with {inputs.keyword} and {gate.total} filled —
    # nothing appended, nothing reworded.
    _, _, send = _at_gate()

    ask = send.tool_calls[1].arguments["inputs"]["message"]
    assert ask == "已选好milk，合计 ¥45。回复 好的 确认支付，或 不用 取消。"


def test_suspension_persists_consent_across_the_wake() -> None:
    # A post-consent suspension must not resume into a refused payment: the
    # consented total rides suspended.json (the _Gate.to_suspended projection).
    p, h, send = _at_gate()
    _reply_arrives(p, h, send, "好的")
    assert p._gate.consented == 45.0
    p._suspend(resume_idx=p._idx, awaiting=False)

    resumed = setup.load_suspended()

    assert resumed is not None and resumed._gate.consented == 45.0


def test_suspend_status_literal_matches_the_sentinel() -> None:
    # program.py spells WAIT literally (the conductor may not import
    # engine runtimes); this pins its constant to the sentinel's spelling
    # (_suspend emits SUSPEND_STATUS; the suspend tests assert the emitted value).
    from physiclaw.agent.runtime.sentinel import WAIT

    assert program.SUSPEND_STATUS == WAIT


def _sections(text: str, slugs: list[str]) -> str:
    return memory.match_sections(memory.split_sections(text), slugs)


def test_memory_context_slices_fail_closed_with_token_match() -> None:
    # Least-privilege: only the declared section rides the micro-call —
    # the rest of memory.md never travels to the (possibly third-party)
    # micro tier. Token-exact heading match (no substring bleed), and
    # NO match means NO memory context: a privacy boundary must never
    # silently widen to the whole file.
    structured = (
        "## shopping_prefs 购物偏好\n- 只买伊利\n\n## shopping_blacklist\n- 三无"
    )
    sliced = _sections(structured, ["shopping_prefs"])
    assert "只买伊利" in sliced and "三无" not in sliced
    # `shopping` is a substring of both headings but a token of neither.
    assert _sections(structured, ["shopping"]) == ""

    unstructured = "只买伊利，不要临期"
    assert _sections(unstructured, ["shopping_prefs"]) == ""


# ---------- the ledger: loop, reconcile, revise ----------


LEDGER_ITEMS = '[{"query": "eggs", "qty": 2}, {"query": "chips", "qty": 1}]'

PRODUCTS = make_screen(
    ("综合", 0.5, 0.1), ("farm eggs", 0.5, 0.3), ("lays chips", 0.5, 0.4)
).text


def _cart_screen(*items: tuple[str, int]) -> str:
    """A cart listing in the real layout: per item a far-left selection
    CHECKBOX icon, the label, then the qty numeral flanked by the two
    stepper ICONS — `_row_qty` must attribute minus/plus to the flanking
    pair, never to the checkbox. Anchored on 综合 so it matches page
    `results`."""
    from physiclaw.common.listing import Element, Screen, format_elements

    els = [
        Element(id=0, kind="text", label="综合", bbox=(0.45, 0.06, 0.55, 0.1), conf=0.9)
    ]
    y = 0.3
    for label, qty in items:
        els.append(
            Element(
                id=len(els),
                kind="icon",  # the selection checkbox — NOT a stepper
                label="",
                bbox=(0.01, y - 0.015, 0.04, y + 0.015),
                conf=0.9,
            )
        )
        els.append(
            Element(
                id=len(els),
                kind="text",
                label=label,
                bbox=(0.06, y - 0.02, 0.45, y + 0.02),
                conf=0.9,
            )
        )
        els.append(
            Element(
                id=len(els),
                kind="icon",
                label="",
                bbox=(0.6, y - 0.015, 0.65, y + 0.015),
                conf=0.9,
            )
        )
        els.append(
            Element(
                id=len(els),
                kind="text",
                label=str(qty),
                bbox=(0.68, y - 0.015, 0.72, y + 0.015),
                conf=0.9,
            )
        )
        els.append(
            Element(
                id=len(els),
                kind="icon",
                label="",
                bbox=(0.75, y - 0.015, 0.8, y + 0.015),
                conf=0.9,
            )
        )
        y += 0.1
    return Screen.read(format_elements(els)).text


def _arm_ledger(items: str = LEDGER_ITEMS):
    _write_channel()
    write_pack(playbooks={"shop": LEDGERED})
    p = _program(name="shop", **{"items": items})
    assert p is not None and p.channel is not None
    return p


def _shop_item(p, h, req, label: str):
    """Resolve one choose_item pick and drive tap + add-cart; returns
    the step after the loop closer routed (next search, or rec-peek)."""
    from physiclaw.conductor.micro import MicroOutcome

    cand = next(c for c in req.candidates if c.key == label)
    tap = p.resolve(MicroOutcome(out="pick", reason="r", confidence=0.9, picked=cand))
    _feed(h, tap, PRODUCTS)
    add = p.advance(h)
    assert add.tool_calls[1].arguments["name"] == "demo/add-cart"
    _feed(h, add, PRODUCTS)
    return p.advance(h)


def _to_reconcile(p):
    """Drive a fresh ledger walk to the reconcile peek; returns (h, peek)."""
    from physiclaw.conductor.micro import DecisionRequest

    h = _history()
    _feed(h, p.advance(h), HOME)  # the start page
    _feed(h, p.advance(h), PRODUCTS)  # `goto` landed on results
    search = p.advance(h)
    assert search.tool_calls[1].arguments["inputs"]["message"] == "eggs"
    _feed(h, search, PRODUCTS)
    req = p.advance(h)
    assert isinstance(req, DecisionRequest)
    step = _shop_item(p, h, req, "farm eggs")
    # The sanctioned backward edge: the loop re-enters `search`.
    assert step.tool_calls[1].arguments["inputs"]["message"] == "chips"
    _feed(h, step, PRODUCTS)
    req2 = p.advance(h)
    peek = _shop_item(p, h, req2, "lays chips")
    assert peek.tool_names() == ["note", "peek"]  # ledger spent → reconcile
    return h, peek


def _at_ledger_gate():
    """A converged cart driven to the gate's sent ask."""
    p = _arm_ledger()
    h, peek = _to_reconcile(p)
    _feed(h, peek, _cart_screen(("farm eggs", 2), ("lays chips", 1)))
    sheet = p.advance(h)  # converged → the to-sheet leg
    assert sheet.tool_calls[1].arguments["inputs"]["message"] == "sheet"
    _feed(h, sheet, _sheet())
    send = p.advance(h)
    assert send.tool_calls[1].arguments["name"] == "channel/send"
    return p, h, send


def test_ledger_walk_shops_reconciles_and_asks() -> None:
    p, _, send = _at_ledger_gate()

    ask = send.tool_calls[1].arguments["inputs"]["message"]
    assert "¥45" in ask and "ok" in ask
    assert [it.status for it in p._ledger] == ["picked", "picked"]
    assert p._ledger[0].label == "farm eggs"


def test_reconcile_steps_quantity_to_the_list() -> None:
    p = _arm_ledger()
    h, peek = _to_reconcile(p)

    _feed(h, peek, _cart_screen(("farm eggs", 1), ("lays chips", 1)))  # egg short
    tap = p.advance(h)
    assert tap.tool_names() == ["note", "tap"]
    assert tap.tool_calls[1].arguments["bbox"][0] > 0.7  # the PLUS (right) icon

    _feed(h, tap, _cart_screen(("farm eggs", 2), ("lays chips", 1)))
    nxt = p.advance(h)  # converged on the tap's own result screen
    assert nxt.tool_calls[1].arguments["inputs"]["message"] == "sheet"


def test_reconcile_missing_item_reshops_it() -> None:
    p = _arm_ledger()
    h, peek = _to_reconcile(p)

    _feed(h, peek, _cart_screen(("farm eggs", 2)))  # 薯片 vanished
    step = p.advance(h)

    # Back into the loop BODY for that item — a fresh search, not a tap.
    assert step.tool_calls[1].arguments["inputs"]["message"] == "chips"
    assert p._ledger[1].status == "pending"


def test_ledger_decide_carries_default_item_context() -> None:
    # The current item rides as DEFAULT context — authors kept
    # forgetting to template {item.query} into criteria, the exact
    # starved-subagent failure the context plan names.
    from physiclaw.conductor.micro import DecisionRequest

    p = _arm_ledger()
    h = _history()
    _feed(h, p.advance(h), HOME)
    _feed(h, p.advance(h), PRODUCTS)  # `goto` landed
    search = p.advance(h)
    _feed(h, search, PRODUCTS)

    req = p.advance(h)

    assert isinstance(req, DecisionRequest)
    assert "current buying-list item: eggs (want 2)" in req.context


def test_gate_revise_rewrites_the_ledger_and_reasks() -> None:
    from physiclaw.conductor.micro import (
        CONFIRM_REPLY,
        REVISE_LIST,
        DecisionRequest,
        MicroOutcome,
    )

    p, h, send = _at_ledger_gate()

    req = _reply_arrives(p, h, send, "ok, but one egg is enough")
    assert isinstance(req, DecisionRequest) and req.call == CONFIRM_REPLY
    req2 = p.resolve(MicroOutcome(out="revise", reason="fewer eggs", confidence=0.9))
    assert isinstance(req2, DecisionRequest) and req2.call == REVISE_LIST
    assert "eggs" in req2.args["ledger"]

    step = p.resolve(
        MicroOutcome(
            out="updated",
            reason="one egg",
            confidence=0.9,
            payload={
                "ledger": '[{"query": "eggs", "qty": 1}, {"query": "chips", "qty": 1}]'
            },
        )
    )
    # The reply was read on the IM thread — the walk returns to the
    # app FIRST (the gate's return macro), then reconciles.
    assert step.tool_names() == ["note", "run_macro"]
    assert step.tool_calls[1].arguments["name"] == "demo/open-app"
    assert p._gate.consented is None  # the old ask no longer covers the order

    _feed(h, step, RESULTS)  # back in the app
    peek = p.advance(h)
    # Quantity-only change: nothing pending, straight back to reconcile.
    assert peek.tool_names() == ["note", "peek"]
    _feed(h, peek, _cart_screen(("farm eggs", 2), ("lays chips", 1)))
    tap = p.advance(h)
    # The MINUS: the icon flanking the numeral — NOT the far-left checkbox.
    assert 0.5 < tap.tool_calls[1].arguments["bbox"][0] < 0.7
    _feed(h, tap, _cart_screen(("farm eggs", 1), ("lays chips", 1)))
    sheet = p.advance(h)
    _feed(h, sheet, _sheet("¥30"))
    resend = p.advance(h)

    ask = resend.tool_calls[1].arguments["inputs"]["message"]
    assert "¥30" in ask  # fresh consent binds the NEW total


def test_gate_revise_added_item_reenters_the_loop() -> None:
    from physiclaw.conductor.micro import MicroOutcome

    p, h, send = _at_ledger_gate()
    _reply_arrives(p, h, send, "ok, and add a bottle of oil")
    p.resolve(MicroOutcome(out="revise", reason="add oil", confidence=0.9))

    step = p.resolve(
        MicroOutcome(
            out="updated",
            reason="r",
            confidence=0.9,
            payload={
                "ledger": '[{"query": "eggs", "qty": 2}, '
                '{"query": "chips", "qty": 1}, {"query": "oil", "qty": 1}]'
            },
        )
    )
    # Return to the app first, then the loop re-enters `search` for the
    # new pending item.
    assert step.tool_calls[1].arguments["name"] == "demo/open-app"
    h2 = h
    _feed(h2, step, RESULTS)
    search = p.advance(h2)

    assert search.tool_calls[1].arguments["inputs"]["message"] == "oil"
    assert [it.qty for it in p._ledger] == [2, 1, 1]


def test_gate_revisions_are_bounded() -> None:
    from physiclaw.conductor.micro import MicroOutcome

    p, h, send = _at_ledger_gate()
    p._gate.revisions = GATE_MAX_REVISIONS

    _reply_arrives(p, h, send, "change it again, no chips this time")
    handed = p.resolve(MicroOutcome(out="revise", reason="again", confidence=0.9))

    summary = _finish(p, h, handed)  # budget spent — the model settles the order
    assert "revisions" in summary


def test_suspension_persists_the_ledger() -> None:
    p, _, _ = _at_ledger_gate()
    p._suspend(resume_idx=p._idx, awaiting=True)

    resumed = setup.load_suspended()

    assert resumed is not None
    assert [it.query for it in resumed._ledger] == ["eggs", "chips"]
    assert resumed._ledger[0].status == "picked"
    assert resumed._ledger[0].label == "farm eggs"


def test_a_bad_ledger_value_is_rejected_at_the_input_seam() -> None:
    from physiclaw.conductor.ledger import check_ledger_value

    write_pack(playbooks={"shop": LEDGERED})
    spec, _ = setup.load_spec("demo", "shop", require_live=False)

    for bad, fragment in (
        ("not json", "items"),
        ('[{"query": "egg", "qty": 0}]', "qty"),
        ('[{"query": "eggs", "qty": 1}, {"query": "eggs", "qty": 2}]', "appears twice"),
    ):
        with pytest.raises(PlaybookError, match=fragment):
            check_ledger_value(spec, setup.resolve_inputs(spec, {"items": bad}))


# (`needs_micro` was deleted with the `_wire_micro` relaxation: the
# micro CLIENT is lazily built on first call, so "who needs one" no
# longer has a consumer — the rescue ladder means any walk can.)


# ---------- bug-hunt regressions ----------


def test_payment_never_fires_off_an_unverified_screen() -> None:
    # The gate's own ask bubble quotes the consented total — if the
    # return macro fails to leave the thread, the money predicates must
    # not be satisfied by our own message.
    p, h, send = _at_gate()
    back = _reply_arrives(p, h, send, "好的")  # confirmed → return macro

    # The return macro FAILS to leave the messenger: the walk sees the
    # chat list (¥ amounts may sit in previews) — no verified demo page.
    _feed(
        h,
        back,
        make_screen(("Weixin", 0.5, 0.05), ("合计 ¥45 昨天", 0.3, 0.2)).text,
    )

    summary = _finish(p, h, p.advance(h))  # blind money refused, hand over
    assert "leg 'pay' expects page" in summary
    # Consent was bound but never consumed — the brief must say so.
    assert "consented to ¥45" in summary and "NOT been made" in summary


def test_pay_consumes_the_consent() -> None:
    p, h, send = _at_gate()
    back = _reply_arrives(p, h, send, "好的")
    _feed(h, back, _sheet())
    pay = p.advance(h)

    assert pay.tool_calls[1].arguments["name"] == "demo/add-cart"
    assert p._gate.consented is None  # spent at fire — no leftovers


def test_budget_suspend_does_not_swallow_the_final_batch() -> None:
    # The 4th unclear message suspends WITHOUT being baselined, so the
    # resuming wake still sees it (judged once, eventually).
    from physiclaw.conductor.micro import MicroOutcome

    p, h, send = _at_gate()
    ask = send.tool_calls[1].arguments["inputs"]["message"]
    _feed(h, send, _thread((ask, 0.75, 0.3)))
    step = p.advance(h)  # first wait
    for i in range(3):  # three unclear rounds spend the checks budget
        _feed(h, step, "waited")
        peek = p.advance(h)
        _feed(h, peek, _thread((ask, 0.75, 0.3), (f"hmm what about {i}", 0.25, 0.5)))
        req = p.advance(h)
        step = p.resolve(MicroOutcome(out="unclear", reason="?", confidence=0.9))
    _feed(h, step, "waited")
    peek = p.advance(h)
    _feed(h, peek, _thread((ask, 0.75, 0.3), ("make it two boxes then", 0.25, 0.5)))
    susp = p.advance(h)  # budget spent → suspend

    assert susp.tool_calls[1].arguments["status"] == "WAIT"
    assert "make it two boxes then" not in p._gate.baseline  # NOT swallowed

    resumed = setup.load_suspended()
    h2 = _history()
    peek2 = resumed.advance(h2)
    _feed(h2, peek2, _thread((ask, 0.75, 0.3), ("make it two boxes then", 0.25, 0.5)))
    req = resumed.advance(h2)  # the message reaches the LLM tier now

    assert "make it two boxes then" in req.args["reply"]


def test_suspended_idx_outside_the_spec_drops_the_suspension() -> None:
    p, h, send = _at_gate()
    _suspend_via_silence(p, h, send)
    import json as _json

    susp_file = suspension.suspended_path()
    data = _json.loads(susp_file.read_text(encoding="utf-8"))
    data["idx"] = 99  # spec edited shorter between wakes
    susp_file.write_text(_json.dumps(data), encoding="utf-8")

    assert setup.load_suspended() is None  # dropped, not a fake completion


def test_blocked_suspend_end_session_drops_the_suspension() -> None:
    p, h, send = _at_gate()
    ask = send.tool_calls[1].arguments["inputs"]["message"]
    thread = _thread((ask, 0.75, 0.3))
    _feed(h, send, thread)
    step = p.advance(h)
    for _ in range(program.SILENCE_ROUNDS):  # silent rounds → suspend turn
        _feed(h, step, "waited")
        peek = p.advance(h)
        _feed(h, peek, thread)
        step = p.advance(h)
    assert step.tool_names() == ["note", "end_session"]

    _feed(h, step, "BLOCKED", error=True)  # end_session refused

    summary = _finish(p, h, p.advance(h))
    assert "suspension dropped" in summary
    assert setup.load_suspended() is None  # stale suspension not resurrected


def test_all_zero_revision_is_a_deny() -> None:
    from physiclaw.conductor.micro import MicroOutcome

    p, h, send = _at_ledger_gate()
    _reply_arrives(p, h, send, "drop everything, remove it all")
    p.resolve(MicroOutcome(out="revise", reason="cancel all", confidence=0.9))

    handed = p.resolve(
        MicroOutcome(
            out="updated",
            reason="all zero",
            confidence=0.9,
            payload={
                "ledger": '[{"query": "eggs", "qty": 0}, {"query": "chips", "qty": 0}]'
            },
        )
    )

    summary = _finish(p, h, handed)
    assert "user declined" in summary


def test_reconciler_never_cross_matches_similar_items() -> None:
    # "coke" must not claim "coke zero"'s row when both are converged.
    from physiclaw.common.listing import Screen
    from physiclaw.conductor.ledger import LedgerItem, assign_rows

    screen_text = _cart_screen(("coke zero", 1), ("coke", 2))
    screen = Screen.read(screen_text)
    items = [
        LedgerItem(query="coke zero", qty=1, status="picked", label="coke zero"),
        LedgerItem(query="coke", qty=2, status="picked", label="coke"),
    ]

    rows = assign_rows(screen, items)

    assert rows[0] is not None and rows[0].label == "coke zero"
    assert rows[1] is not None and rows[1].label == "coke"


def test_reconciler_never_claims_another_products_row() -> None:
    # `label_matches` is recall-oriented, and two brands of one carton
    # size agree on every word that is not the brand — Dice 0.74. With
    # the first item's row absent from the cart and the second's
    # TRUNCATED (so neither claims exactly), the first went first in the
    # fuzzy pass and took the second's row: the reconciler would have
    # stepped a rival's quantity from 3 down to 1, in a cart about to be
    # paid. An unassigned item must read as missing and re-enter the
    # loop. Note this one is NOT a class-token artifact — strip every
    # `<NUM>` and it still scores 0.65 (see the total-row test below for
    # the case that is).
    from physiclaw.common.listing import Screen
    from physiclaw.conductor.ledger import LedgerItem, assign_rows

    screen = Screen.read(_cart_screen(("yolo fresh milk 250ml*16", 3)))
    items = [
        LedgerItem(
            query="acme", qty=1, status="picked", label="acme fresh milk 250ml*16 case"
        ),
        LedgerItem(
            query="yolo", qty=3, status="picked", label="yolo fresh milk 250ml*16 case"
        ),
    ]

    rows = assign_rows(screen, items)

    assert rows[0] is None  # not in this cart — say so
    assert rows[1] is not None and rows[1].label == "yolo fresh milk 250ml*16"


def test_reconciler_never_claims_the_sheets_total_row() -> None:
    # The other victim, and this one IS a class-token artifact:
    # `match.normalize` collapses an amount to the 7-character `<PRICE>`,
    # so a short total row is mostly token once normalized
    # (`total<PRICE>`, 7 of 12 characters) and ANY pick label ending in a
    # price scores 0.57 against it. Claiming the total row turned a
    # recoverable "item missing from the cart" into a hand-over
    # reporting unreadable steppers.
    from physiclaw.conductor.ledger import LedgerItem, assign_rows

    # bbox is irrelevant here — `assign_rows` reads only kind and label.
    screen = make_screen(("total ¥104.7", 0.25, 0.875))
    items = [LedgerItem(query="cola", qty=2, status="picked", label="cola ¥3.5")]

    assert assign_rows(screen, items) == [None]


# ---------- deep-sweep regressions: state-machine holes ----------


def test_locate_never_skips_work_nodes() -> None:
    # A fresh wake whose screen matches a LATE leg's verify page must
    # not fast-forward past the loop/reconcile/gate: a page match proves
    # navigation, never that the work ran — the gate would quote ¥45
    # for a never-shopped list.
    from physiclaw.conductor.micro import DecisionRequest

    p = _arm_ledger()
    h = _history()
    peek = p.advance(h)
    _feed(h, peek, _sheet())  # reads as `results` — the sheet leg's verify

    step = p.advance(h)

    # The cursor stops at the end of the LEADING leg run: the choose
    # DECIDE — never the gate.
    assert isinstance(step, DecisionRequest) and step.call == "choose_item"
    assert all(it.status == "pending" for it in p._ledger)


def _write_suspended(playbook: str, idx: int, **over) -> None:
    """suspended.json with the boilerplate defaulted — tests pass only
    their deltas. The production shape is `Program._suspended_dict`; a
    schema change is edited here once beside the tests that fake it."""
    from physiclaw.common.logger import write_json_atomic

    data = {
        "schema": suspension.SUSPENDED_SCHEMA,
        "app": "demo",
        "playbook": playbook,
        "idx": idx,
        "values": {},
        "outputs": {},
        "visits": {},
        "ledger": None,
        "item": 0,
        "ask_text": "",
        "baseline": [],
        "quoted": None,
        "cap": None,
        "consented": None,
        "awaiting": False,
        "revisions": 0,
    }
    data.update(over)
    write_json_atomic(suspension.suspended_path(), data)


def test_payment_gate_total_is_quoted_only_off_a_verified_page() -> None:
    # A suspended resume straight onto the gate with an unknown screen must
    # hand over, not quote max(¥) off whatever the camera saw.
    _write_channel()
    write_pack(playbooks={"shop": LEDGERED})
    spec, _ = setup.load_spec("demo", "shop")
    gate_idx = next(
        i for i, n in enumerate(spec.nodes) if type(n).__name__ == "HumanGateNode"
    )
    _write_suspended(
        "shop",
        gate_idx,
        values={"items": LEDGER_ITEMS},
        ledger=[
            {"query": "eggs", "qty": 2, "status": "picked", "label": "farm eggs"},
            {"query": "chips", "qty": 1, "status": "picked", "label": "lays chips"},
        ],
        item=1,
    )
    p = setup.load_suspended(channel.load_channel())
    assert p is not None
    h = _history()
    _feed(h, p.advance(h), ELSEWHERE)  # unknown screen at the gate

    summary = _finish(p, h, p.advance(h))  # handover — no ask was sent
    assert "refusing to ask blind" in summary


def test_reask_send_reads_a_deny_sent_meanwhile() -> None:
    # The user cancels while the walk is off shopping a revision; the
    # "cancel" sits on the thread when the re-ask lands. Overwriting the
    # baseline would swallow it forever — the send must read it first.
    from physiclaw.conductor.micro import MicroOutcome

    p, h, send = _at_ledger_gate()
    _reply_arrives(p, h, send, "ok, but one egg is enough")
    p.resolve(MicroOutcome(out="revise", reason="fewer", confidence=0.9))
    step = p.resolve(
        MicroOutcome(
            out="updated",
            reason="one egg",
            confidence=0.9,
            payload={
                "ledger": '[{"query": "eggs", "qty": 1}, {"query": "chips", "qty": 1}]'
            },
        )
    )
    _feed(h, step, RESULTS)  # gate-return landed
    peek = p.advance(h)
    _feed(h, peek, _cart_screen(("farm eggs", 1), ("lays chips", 1)))
    sheet = p.advance(h)  # converged → the to-sheet leg
    _feed(h, sheet, _sheet("¥30"))
    resend = p.advance(h)
    ask2 = resend.tool_calls[1].arguments["inputs"]["message"]
    _feed(h, resend, _thread((ask2, 0.75, 0.7), ("cancel", 0.25, 0.5)))

    summary = _finish(p, h, p.advance(h))
    assert "user declined" in summary


def test_failed_reply_judgment_keeps_the_batch_visible() -> None:
    # A provider failure on the LLM tier must NOT baseline the reply —
    # the next round re-reads and re-judges it ("judged once,
    # eventually"), instead of suspending on false silence forever.
    from physiclaw.conductor.micro import CONFIRM_REPLY, DecisionRequest

    p, h, send = _at_gate()
    req = _reply_arrives(p, h, send, "ok, buy it now??")
    assert isinstance(req, DecisionRequest)

    wait = p.resolve(None)  # the micro-call failed
    assert wait.tool_names() == ["note", "wait"]
    assert "ok, buy it now??" not in p._gate.baseline

    _feed(h, wait, "waited")
    peek = p.advance(h)
    ask = send.tool_calls[1].arguments["inputs"]["message"]
    _feed(h, peek, _thread((ask, 0.75, 0.3), ("ok, buy it now??", 0.25, 0.5)))
    again = p.advance(h)

    assert isinstance(again, DecisionRequest) and again.call == CONFIRM_REPLY


def test_revision_rephrase_lands_on_the_shopped_item() -> None:
    # revise_list is asked to echo unchanged items verbatim, but a
    # rephrase ("eggs" → "fresh eggs") must merge onto the existing item
    # — not zero a correct cart row and re-shop the same product.
    from physiclaw.conductor.micro import MicroOutcome

    p, h, send = _at_ledger_gate()
    _reply_arrives(p, h, send, "make it fresh eggs instead")
    p.resolve(MicroOutcome(out="revise", reason="rephrase", confidence=0.9))
    p.resolve(
        MicroOutcome(
            out="updated",
            reason="r",
            confidence=0.9,
            payload={
                "ledger": '[{"query": "fresh eggs", "qty": 2}, '
                '{"query": "chips", "qty": 1}]'
            },
        )
    )

    assert [it.qty for it in p._ledger] == [2, 1]  # no phantom third item
    assert p._ledger[0].status == "picked" and p._ledger[0].label == "farm eggs"


def test_ledger_longer_than_max_visits_completes() -> None:
    # The loop body's decide legitimately runs once per item — the visit
    # budget is per (node, item), or a 4-item list could never finish
    # under the default max_visits of 3.
    from physiclaw.conductor.micro import DecisionRequest

    items = json.dumps([{"query": q, "qty": 1} for q in ("a1", "b2", "c3", "d4")])
    p = _arm_ledger(items)
    h = _history()
    _feed(h, p.advance(h), HOME)
    _feed(h, p.advance(h), PRODUCTS)  # `goto` landed
    step = p.advance(h)
    for _ in range(4):
        assert step.tool_names() == ["note", "run_macro"]  # the search leg
        _feed(h, step, PRODUCTS)
        req = p.advance(h)
        assert isinstance(req, DecisionRequest), "visit budget must be per item"
        step = _shop_item(p, h, req, "farm eggs")

    assert step.tool_names() == ["note", "peek"]  # all four shopped → reconcile


def test_second_reshop_of_an_item_hands_over() -> None:
    # An item still missing after one re-shop will never read as a cart
    # row — hand over instead of duplicating real adds forever.
    p = _arm_ledger()
    h, peek = _to_reconcile(p)
    _feed(h, peek, _cart_screen(("farm eggs", 2)))  # chips missing
    step = p.advance(h)  # re-shop #1
    assert step.tool_calls[1].arguments["inputs"]["message"] == "chips"
    _feed(h, step, PRODUCTS)
    req = p.advance(h)
    peek2 = _shop_item(p, h, req, "lays chips")
    _feed(h, peek2, _cart_screen(("farm eggs", 2)))  # STILL missing

    summary = _finish(p, h, p.advance(h))
    assert "still missing after a re-shop" in summary


def _suspend_a_confirm() -> None:
    """Drive the CONFIRMING playbook to its written suspension."""
    _write_channel()
    write_pack(playbooks={"notify": CONFIRMING})
    p = _program(name="notify", keyword="milk")
    h = _history()
    _feed(h, p.advance(h), HOME)
    _feed(h, p.advance(h), RESULTS)
    send = p.advance(h)
    text = send.tool_calls[1].arguments["inputs"]["message"]
    _feed(h, send, _thread((text, 0.75, 0.3)))
    p.advance(h)  # suspension written
    assert suspension.suspended_path().exists()


def test_rearming_the_same_playbook_drops_its_stale_suspension() -> None:
    # Re-arm with new inputs: the suspended old-values walk must not
    # resume first and then consume the fresh standing order.
    _suspend_a_confirm()

    assert suspension.clear_suspended() is True
    assert not suspension.suspended_path().exists()


def test_missing_suspend_result_drops_the_suspension_file() -> None:
    # The suspension file is written before end_session; if that result never
    # lands, the session may run on — a dead walk must not resurrect.
    _write_channel()
    write_pack(playbooks={"notify": CONFIRMING})
    p = _program(name="notify", keyword="milk")
    h = _history()
    _feed(h, p.advance(h), HOME)
    _feed(h, p.advance(h), RESULTS)
    send = p.advance(h)
    text = send.tool_calls[1].arguments["inputs"]["message"]
    _feed(h, send, _thread((text, 0.75, 0.3)))
    susp = p.advance(h)
    assert susp.tool_names() == ["note", "end_session"]
    assert suspension.suspended_path().exists()
    h.append(susp)  # the end_session result never arrives

    summary = _finish(p, h, p.advance(h))
    assert "suspension dropped" in summary
    assert not suspension.suspended_path().exists()


# ---------- decide context slices ----------


def _context_decide_program(context: tuple[str, ...]) -> program.Program:
    """A one-decision walk with declared context, built directly — the
    context assembly is program behavior, not parser behavior, so no
    pack on disk is needed. `criteria` IS declared: the parser rejects a
    context entry naming an undeclared input, so the hand-built spec
    must stay a shape the parser could have produced."""
    from physiclaw.conductor.calls import CALLS
    from physiclaw.conductor.playbook import DecideNode, Playbook, PlaybookInput

    spec = Playbook(
        app="demo",
        name="ctx",
        description="d",
        enabled=True,
        inputs=(PlaybookInput(name="criteria", description="pick rule"),),
        mandate=None,
        nodes=(
            DecideNode(
                id="choose",
                call="choose_item",
                args={"criteria": "cheapest"},
                context=context,
                outcomes=CALLS["choose_item"].outcomes,
                routes={
                    "pick": "escalate",
                    "scroll": "choose",
                    "none_fit": "escalate",
                    "escalate": "escalate",
                },
                max_visits=3,
            ),
        ),
    )
    return program.Program(
        app="demo",
        spec=spec,
        values={"criteria": "cheapest"},
        pack_macros={},
        prints=[],
    )


def _write_memory(text: str) -> None:
    paths.memory_dir().mkdir(parents=True, exist_ok=True)
    paths.memory_file().write_text(text, encoding="utf-8")


def test_decide_context_assembles_inputs_and_memory_slices() -> None:
    from physiclaw.conductor.micro import DecisionRequest

    _write_memory("## shopping\nbuy oat milk\n\n## other\nunrelated\n")
    p = _context_decide_program(("inputs.criteria", "memory.shopping"))
    h = _history()
    _feed(h, p.advance(h), make_screen(("牛奶", 0.5, 0.3)).text)

    req = p.advance(h)

    assert isinstance(req, DecisionRequest)
    assert "criteria: cheapest" in req.context
    assert "buy oat milk" in req.context
    assert "unrelated" not in req.context  # only the matching section


def test_decide_memory_slice_without_match_stays_fail_closed(caplog) -> None:
    # No `## shopping` heading anywhere: the decision runs WITHOUT
    # memory context (never the whole file — the privacy boundary must
    # not silently widen), and the degradation is logged.
    from physiclaw.conductor.micro import DecisionRequest

    _write_memory("## other\nsecret fact\n")
    p = _context_decide_program(("memory.shopping",))
    h = _history()
    _feed(h, p.advance(h), make_screen(("牛奶", 0.5, 0.3)).text)

    with caplog.at_level("INFO"):
        req = p.advance(h)

    assert isinstance(req, DecisionRequest)
    assert "secret fact" not in req.context
    assert "without memory context" in caplog.text


# ---------- payment gate: cap and total edges ----------


def test_gate_over_cap_consent_binds_to_the_quoted_total() -> None:
    # Quoted ¥145 against the mandate's ¥100. The over_message WORDING
    # pin lives in `test_gate_over_cap_ask_discloses_the_breach`; this
    # one drives the half no test owned: the SAME plain-consent reply
    # opens an over-cap gate, and consent binds to the QUOTED total —
    # never the cap.
    p, h, send = _at_gate("¥145")
    assert "已超出预算" in send.tool_calls[1].arguments["inputs"]["message"]

    _reply_arrives(p, h, send, "好的")

    assert p._gate.consented == 145.0


def test_gate_hands_over_when_no_total_is_readable(caplog) -> None:
    # The sheet page verified but shows no ¥ amount: the ask IS the
    # consent record, so with nothing to quote the gate refuses to ask.
    _write_channel()
    write_pack(playbooks={"pay": GATED})
    p = _program(name="pay", keyword="milk")
    h = _history()
    _feed(h, p.advance(h), HOME)  # the start page
    _feed(h, p.advance(h), RESULTS)  # the verified results page — no ¥ on it

    with caplog.at_level("INFO"):
        step = p.advance(h)  # handover, no ask sent
    # THIS guard, not a sibling: the brief's reason tells them apart.
    assert "no total readable" in _finish(p, h, step)


def test_gate_hands_over_when_the_cap_cannot_resolve(caplog) -> None:
    # A `{inputs.cap}` mandate whose input turns out non-numeric: no cap means
    # no over-budget rule, so the gate hands over rather than asking
    # with an unenforceable mandate.
    ref_cap = GATED.replace("budget: 100", 'budget: "{inputs.cap}"').replace(
        "inputs:\n", "inputs:\n  cap:\n    description: budget\n"
    )
    _write_channel()
    write_pack(playbooks={"pay": ref_cap})
    p = _program(name="pay", keyword="milk", cap="oops")
    h = _history()
    _feed(h, p.advance(h), HOME)
    _feed(h, p.advance(h), _sheet())

    with caplog.at_level("INFO"):
        step = p.advance(h)  # handover, no ask sent
    assert "cap could not be resolved" in _finish(p, h, step)


def test_payment_leg_without_consent_hands_over(caplog) -> None:
    # A resume landing directly ON the payment leg with no consent
    # recorded (the gate never confirmed): money never fires. This is
    # `money.fire_block`'s first predicate — the last line of defense
    # if every earlier guard were somehow skipped.
    _write_channel()
    write_pack(playbooks={"pay": GATED})
    spec, _ = setup.load_spec("demo", "pay")
    pay_idx = next(
        i for i, n in enumerate(spec.nodes) if getattr(n, "irreversible", None)
    )
    _write_suspended(
        "pay",
        pay_idx,
        values={"keyword": "milk"},
        quoted=45.0,
        cap=100.0,
        # consented stays None — the gate never opened.
    )
    p = setup.load_suspended(channel.load_channel())
    assert p is not None
    h = _history()
    _feed(h, p.advance(h), _sheet())  # verified sheet at the pay leg

    with caplog.at_level("INFO"):
        step = p.advance(h)  # money never fires blind
    assert "without a confirmed total" in _finish(p, h, step)


def test_reconcile_hands_over_when_the_cart_never_converges(caplog) -> None:
    # A stepper whose taps never change the read qty (sticky UI, wrong
    # element) must exhaust `reconcile.MAX_ACTIONS` and hand over — the
    # runaway backstop, not an infinite tap loop.
    p = _arm_ledger()
    h, step = _to_reconcile(p)

    stuck = _cart_screen(("farm eggs", 1), ("lays chips", 1))  # egg stays short
    with caplog.at_level("INFO"):
        for _ in range(reconcile.MAX_ACTIONS + 1):
            _feed(h, step, stuck)
            step = p.advance(h)
            if step.tool_names() != ["note", "tap"]:  # kept stepping until spent
                break
    assert "cart not converging" in _finish(p, h, step)


# ---------- the rescue ladder ----------

# Both walk pages carry learned geometry (the occluded verdict needs
# it); `done` keeps the route's last landing distinct so `_locate`
# never fast-forwards a booted walk.
RESCUE_PAGES = """\
home:
  anchors: ["Files", "Recent"]
results:
  anchors: ["综合", "销量"]
done:
  anchors: ["AllDone"]
"""

# Screens with both anchors visible — match against learned geometry.
HOME2 = make_screen(("Files", 0.5, 0.10), ("Recent", 0.5, 0.50)).text
RESULTS2 = make_screen(("综合", 0.5, 0.10), ("销量", 0.5, 0.50)).text

# A popup band over a page: the top anchor still visible, the lower one
# hidden under four unexpected labels clustered at its learned position
# — reads `occluded` with the band around cy 0.50 (± OVERLAY_PAD).
POPUP_OVER_HOME = make_screen(
    ("Files", 0.5, 0.10),
    ("new user gift pack", 0.4, 0.44),
    ("mega deal inside", 0.6, 0.48),
    ("one free with one", 0.5, 0.52),
    ("以后再说", 0.5, 0.56),
).text

POPUP_OVER_RESULTS = make_screen(
    ("综合", 0.5, 0.10),
    ("new user gift pack", 0.4, 0.44),
    ("mega deal inside", 0.6, 0.48),
    ("one free with one", 0.5, 0.52),
    ("以后再说", 0.5, 0.56),
).text

LOCKED_MID = make_screen(("Enter Passcode", 0.5, 0.5)).text


def _learn_pages() -> None:
    from physiclaw.conductor.pages import LearnedAnchor, LearnedPage, save_learned

    save_learned(
        "demo",
        {
            "home": LearnedPage(
                anchors={
                    "Files": LearnedAnchor("Files", 0.5, 0.10, 0.05, 1.0, 1.0),
                    "Recent": LearnedAnchor("Recent", 0.5, 0.50, 0.05, 1.0, 1.0),
                },
                threshold=0.9,
                observations=4,
            ),
            "results": LearnedPage(
                anchors={
                    "综合": LearnedAnchor("综合", 0.5, 0.10, 0.05, 1.0, 1.0),
                    "销量": LearnedAnchor("销量", 0.5, 0.50, 0.05, 1.0, 1.0),
                },
                threshold=0.9,
                observations=4,
            ),
        },
    )


def _rescue_walk(flow: str = FLOW, *, macros: tuple = ("open-app", "add-cart")):
    """A FLOW walk over pages with learned geometry (the occluded
    verdict needs it), driven past the opening peek to move 1."""
    write_pack(pages=RESCUE_PAGES, macros=macros, playbooks={"flow": flow})
    _learn_pages()
    p = _program(keyword="milk")
    h = _history()
    _feed(h, p.advance(h), HOME2)  # the start page
    leg1 = p.advance(h)
    assert leg1 is not None and leg1.tool_names() == ["note", "run_macro"]
    return p, h, leg1


def test_popup_at_verify_is_dismissed_and_the_walk_continues() -> None:
    p, h, leg1 = _rescue_walk()
    _feed(h, leg1, POPUP_OVER_RESULTS)  # the landing check blocked by an overlay

    dismiss = p.advance(h)
    assert dismiss is not None and dismiss.tool_names() == ["note", "tap"]
    assert "以后再说" in dismiss.tool_calls[0].arguments["summary"]

    _feed(h, dismiss, RESULTS2)  # overlay gone — results restored, landing holds
    leg2 = p.advance(h)
    assert leg2 is not None
    assert leg2.tool_calls[1].arguments["name"] == "demo/add-cart"
    assert "rescue: restored demo.results" in leg2.tool_calls[0].arguments["summary"]


def test_wrong_page_on_resume_goes_back_then_runs_the_move() -> None:
    # A resumed walk stands on `home` while its move expects `results` —
    # wandered relative to the derived precondition. go_back pops, the
    # enter check re-runs on the restored page, and the move fires: the
    # cursor never moved (I1).
    write_pack(pages=RESCUE_PAGES, playbooks={"flow": FLOW})
    _learn_pages()
    _write_suspended("flow", 1, values={"keyword": "milk"})
    p = setup.load_suspended(channel.load_channel())
    assert p is not None
    h = _history()
    _feed(h, p.advance(h), HOME2)  # `search` enters at results — wrong page

    settle = p.advance(h)  # rung 0: the free settle re-peek
    assert settle is not None and settle.tool_names() == ["note", "peek"]
    _feed(h, settle, HOME2)  # settled — still the wrong page
    back = p.advance(h)
    assert back is not None and back.tool_names() == ["note", "go_back"]

    _feed(h, back, RESULTS2)
    leg2 = p.advance(h)
    assert leg2 is not None
    assert leg2.tool_calls[1].arguments["name"] == "demo/add-cart"


def test_locked_mid_walk_unlocks_then_continues() -> None:
    p, h, leg1 = _rescue_walk()
    _feed(h, leg1, LOCKED_MID)

    unlock = p.advance(h)
    assert unlock is not None and unlock.tool_names() == ["note", "unlock_phone"]

    _feed(h, unlock, RESULTS2)
    leg2 = p.advance(h)
    assert leg2 is not None
    assert leg2.tool_calls[1].arguments["name"] == "demo/add-cart"


# A popup whose buttons match NO vocabulary word — rung 1 has nothing,
# the micro tier gets asked. Four unexpected labels around the lower
# anchor's learned position keep the occluded verdict firing.
WORDLESS_POPUP = make_screen(
    ("综合", 0.5, 0.10),
    ("limited offer", 0.4, 0.44),
    ("mega deal inside", 0.6, 0.48),
    ("one free with one", 0.5, 0.52),
    ("continue browsing", 0.5, 0.56),
).text


def test_wordless_popup_asks_the_micro_tier_then_taps_and_learns() -> None:
    from physiclaw.conductor import rescue
    from physiclaw.conductor.micro import (
        CLEAR_OVERLAY,
        DISMISS_ARM,
        DecisionRequest,
        MicroOutcome,
    )

    p, h, leg1 = _rescue_walk()
    _feed(h, leg1, WORDLESS_POPUP)

    req = p.advance(h)
    assert isinstance(req, DecisionRequest) and req.call == CLEAR_OVERLAY
    picked = next(c for c in req.candidates if c.key == "continue browsing")

    tap = p.resolve(
        MicroOutcome(
            out=DISMISS_ARM, reason="pure close", confidence=0.9, picked=picked
        )
    )
    assert tap is not None and tap.tool_names() == ["note", "tap"]

    _feed(h, tap, RESULTS2)  # dismissed — results restored
    leg2 = p.advance(h)
    assert leg2 is not None
    assert leg2.tool_calls[1].arguments["name"] == "demo/add-cart"
    # The micro tier taught the free tier: the label is learned.
    assert rescue.load_dismiss("demo") == ("continue browsing",)


def test_micro_none_safe_continues_the_ladder_with_back() -> None:
    from physiclaw.conductor.micro import NONE_SAFE, DecisionRequest, MicroOutcome

    p, h, leg1 = _rescue_walk()
    _feed(h, leg1, WORDLESS_POPUP)
    req = p.advance(h)
    assert isinstance(req, DecisionRequest)

    step = p.resolve(MicroOutcome(out=NONE_SAFE, reason="nothing safe", confidence=0.9))

    assert step is not None and step.tool_names() == ["note", "peek"]  # settle first
    _feed(h, step, WORDLESS_POPUP)  # settled — the popup is genuinely there
    nxt = p.advance(h)
    assert nxt is not None and nxt.tool_names() == ["note", "go_back"]


def test_zero_gesture_leg_abort_under_an_overlay_retries_once() -> None:
    # The I6 one-shot: the leg's macro guard-failed BEFORE any gesture
    # (the runner's marker) with a popup over the page — dismiss it and
    # re-run the leg. Nothing was burned engine-side either (the
    # zero-gesture rule), so the retry actually dispatches.
    from physiclaw.macros.model import NO_GESTURES_NOTE

    p, h, leg1 = _rescue_walk()
    abort_text = (
        "macro demo/open-app: ABORTED at step 1/1 (guard_failed) — no steps "
        f"executed by this run. {NO_GESTURES_NOTE} — the phone did not move.\n"
        + POPUP_OVER_HOME
    )
    _feed(h, leg1, abort_text, error=True)

    dismiss = p.advance(h)
    assert dismiss is not None and dismiss.tool_names() == ["note", "tap"]
    assert "以后再说" in dismiss.tool_calls[0].arguments["summary"]

    _feed(h, dismiss, HOME2)  # popup gone — the page the guard needed
    retry = p.advance(h)
    assert retry is not None and retry.tool_names() == ["note", "run_macro"]
    assert retry.tool_calls[1].arguments["name"] == "demo/open-app"  # SAME move

    _feed(h, retry, RESULTS2)  # this time the move lands
    leg2 = p.advance(h)
    assert leg2 is not None
    assert leg2.tool_calls[1].arguments["name"] == "demo/add-cart"


def test_acted_leg_abort_still_hands_over() -> None:
    # No marker = gestures ran = the one-strike world stands: hard
    # handover, never a blind retry (I6).
    p, h, leg1 = _rescue_walk()
    _feed(
        h,
        leg1,
        "macro demo/open-app: ABORTED at step 2/3 (guard_failed) — "
        "steps 1–1 already executed. Do NOT re-run.",
        error=True,
    )

    summary = _finish(p, h, p.advance(h))
    assert "blocked or failed" in summary


def test_reset_rung_force_quits_reopens_and_relocates() -> None:
    # The big hammer: back budget spent with a pack `open` macro on
    # hand → force_quit, reopen, then locate from the top — the
    # killed-session resume path. The open lands on the START page, so
    # the walk restarts from move 1.
    p, h, leg1 = _rescue_walk(macros=("open-app", "add-cart", "open"))
    _feed(h, leg1, ELSEWHERE)  # verify fails on an unknown screen
    settle = p.advance(h)
    assert settle.tool_names() == ["note", "peek"]  # rung 0
    _feed(h, settle, ELSEWHERE)
    back1 = p.advance(h)
    assert back1.tool_names() == ["note", "go_back"]
    _feed(h, back1, ELSEWHERE)
    back2 = p.advance(h)
    _feed(h, back2, ELSEWHERE)

    quit_ = p.advance(h)
    assert quit_ is not None and quit_.tool_names() == ["note", "force_quit"]

    _feed(h, quit_, ELSEWHERE)  # the springboard — never judged
    reopen = p.advance(h)
    assert reopen is not None and reopen.tool_names() == ["note", "run_macro"]
    assert reopen.tool_calls[1].arguments == {"name": "demo/open"}

    _feed(h, reopen, HOME2)  # the open landed on the START page
    redo = p.advance(h)  # locate from the top — the walk restarts cleanly
    assert redo is not None
    assert redo.tool_calls[1].arguments["name"] == "demo/open-app"
    assert "rescue: demo reset" in redo.tool_calls[0].arguments["summary"]


def test_rescue_back_budget_exhausts_into_a_handover_naming_the_rungs() -> None:
    from physiclaw.conductor import walklog

    p, h, leg1 = _rescue_walk()
    _feed(h, leg1, ELSEWHERE)  # unknown — settle first, then the back rung
    settle = p.advance(h)
    assert settle is not None and settle.tool_names() == ["note", "peek"]
    _feed(h, settle, ELSEWHERE)
    back1 = p.advance(h)
    assert back1 is not None and back1.tool_names() == ["note", "go_back"]
    _feed(h, back1, ELSEWHERE)
    back2 = p.advance(h)
    assert back2 is not None and back2.tool_names() == ["note", "go_back"]
    _feed(h, back2, ELSEWHERE)

    summary = _finish(p, h, p.advance(h))

    assert "rescue tried: back×2" in summary
    (row,) = walklog.load()
    assert row["rescues"] == 2 and row["outcome"] == "handover"


# ---------- walk telemetry (runs.jsonl) ----------


def test_completed_walk_records_one_completed_run_line() -> None:
    from physiclaw.conductor import walklog

    write_pack(playbooks={"flow": FLOW})
    p = _program(keyword="milk")
    h = _history()
    _feed(h, p.advance(h), HOME)
    _feed(h, p.advance(h), RESULTS)
    _feed(h, p.advance(h), DONE)

    _finish(p, h, p.advance(h))

    (row,) = walklog.load()
    assert row["outcome"] == "completed"
    assert (row["app"], row["playbook"]) == ("demo", "flow")
    assert row["node"] is None  # cursor past the last node
    assert row["micros"] == 0


def test_handover_records_run_line_at_the_failing_node() -> None:
    from physiclaw.conductor import walklog

    write_pack(playbooks={"flow": FLOW})
    p = _program(keyword="milk")
    h = _history()
    _feed(h, p.advance(h), HOME)
    _feed(h, p.advance(h), HOME)  # move 1 landed on the WRONG page
    _feed(h, p.advance(h), HOME)  # rung 0 settle re-peek — still wrong
    _feed(h, p.advance(h), HOME)  # rescue back #1 — still wrong
    _feed(h, p.advance(h), HOME)  # rescue back #2 — still wrong

    _finish(p, h, p.advance(h))

    (row,) = walklog.load()
    assert row["outcome"] == "handover"
    assert row["node"] == "open"
    assert "did not land" in row["reason"]
    assert row["rescues"] == 2


def test_completed_payment_walk_records_history_fields() -> None:
    # The completed line carries the structured fields (telemetry +
    # last_picks): inputs, the fired total (consent is consumed at fire —
    # this is where it survives), picks.
    from physiclaw.conductor import walklog

    p, h, send = _at_ledger_gate()
    back = _reply_arrives(p, h, send, "ok")
    _feed(h, back, _sheet())
    pay = p.advance(h)
    assert pay.tool_calls[1].arguments["name"] == "demo/add-cart"
    _feed(h, pay, HOME)  # the pay leg's verify page

    _finish(p, h, p.advance(h))

    (row,) = walklog.load()
    assert row["outcome"] == "completed"
    assert row["total"] == 45.0
    assert row["picks"] == {"eggs": "farm eggs", "chips": "lays chips"}
    assert "items" in row["values"]


def test_payment_fire_writes_the_doctrine_purchase_log_line() -> None:
    # PERSISTENCE § When to write: "append_log after every major step
    # (purchase…)" — the conductor is the one doing the purchasing, so
    # it writes the line itself (the log_external_stop precedent), the
    # moment the payment leg's result lands.
    from physiclaw.common import daylog

    p, h, send = _at_ledger_gate()
    back = _reply_arrives(p, h, send, "ok")
    _feed(h, back, _sheet())
    pay = p.advance(h)
    _feed(h, pay, HOME)
    p.advance(h)  # the settle that judges the pay leg — and logs

    entries = daylog.load_recent_entries(5)

    assert "conductor: demo: paid ¥45 (playbook demo/shop)" in entries
    assert "farm eggs ×2, lays chips ×1" in entries


def test_suspend_writes_the_close_routine_log_line() -> None:
    # A conductor-suspended wake never runs the model, and the walk has
    # no per-step logs — without this line the suspension is invisible
    # to the next wake's memory window.
    from physiclaw.common import daylog

    p, h, send = _at_gate()
    _suspend_via_silence(p, h, send)

    entries = daylog.load_recent_entries(5)

    assert "conductor: demo/pay suspended" in entries
    assert "any wake resumes it" in entries


def test_ledger_decide_context_quotes_the_previous_picks() -> None:
    from physiclaw.conductor import walklog
    from physiclaw.conductor.micro import DecisionRequest

    walklog.record(
        app="demo",
        playbook="shop",
        outcome="completed",
        idx=9,
        nodes=9,
        node=None,
        picks={"eggs": "farm eggs 30ct"},
    )
    p = _arm_ledger()
    h = _history()
    _feed(h, p.advance(h), HOME)
    _feed(h, p.advance(h), PRODUCTS)  # `goto` landed
    _feed(h, p.advance(h), PRODUCTS)  # `search` landed
    req = p.advance(h)

    assert isinstance(req, DecisionRequest)
    assert (
        "previously picked (last completed run): eggs → farm eggs 30ct" in req.context
    )


def test_session_setup_assembles_the_activation_context() -> None:
    # The agent's OWN memory convention feeds parse_task: declared
    # memory slices (fail-closed) + the recent daily-log window — the
    # same record the model reads at wake, never a conductor-private
    # store (runs.jsonl stays telemetry).
    from physiclaw.common import daylog

    _write_channel()
    _write_memory("## shopping\nprefers oat milk\n\n## other\nsecret\n")
    daylog.append_log("[11:02] demo: bought milk ¥45 — reported to the user")
    spec = FLOW.replace(
        "description: two moves\n",
        "description: two moves\ncontext: [memory.shopping]\n",
    )
    write_pack(playbooks={"flow": spec})

    _, overture, _ = setup.session_setup()

    assert overture is not None
    ctx = overture._activation.context
    assert "prefers oat milk" in ctx and "secret" not in ctx
    assert "Recent daily-log entries" in ctx
    assert "bought milk ¥45" in ctx


def test_abandon_records_a_mid_flight_walk_and_breadcrumbs_it() -> None:
    # The killed-session path (`log_external_stop`'s twin): the plugin's
    # teardown abandons a walk cut short — one telemetry row plus the
    # daily-log breadcrumb, since this walk had acted.
    from physiclaw.common import daylog
    from physiclaw.conductor import walklog

    write_pack(playbooks={"flow": FLOW})
    p = _program(keyword="milk")
    h = _history()
    _feed(h, p.advance(h), HOME)
    p.advance(h)  # move 1 synthesized — the walk acted, then the session dies

    p.abandon()

    (row,) = walklog.load()
    assert row["outcome"] == "abandoned"
    assert row["node"] == "open"
    assert "cut short mid-walk at node open" in daylog.load_recent_entries(5)


def test_abandon_is_a_no_op_for_unstarted_and_closed_walks() -> None:
    from physiclaw.conductor import walklog

    write_pack(playbooks={"flow": FLOW})
    fresh = _program(keyword="milk")
    fresh.abandon()  # never advanced — not a run
    assert walklog.load() == []

    p = _program(keyword="milk")
    h = _history()
    _feed(h, p.advance(h), HOME)
    _feed(h, p.advance(h), RESULTS)
    _feed(h, p.advance(h), DONE)
    _finish(p, h, p.advance(h))  # completed — latched

    p.abandon()

    (row,) = walklog.load()
    assert row["outcome"] == "completed"  # still exactly one row


def test_failed_decision_records_handover_with_micro_count() -> None:
    from physiclaw.conductor import walklog
    from physiclaw.conductor.micro import DecisionRequest

    write_pack(playbooks={"branch": BRANCH})
    p = _program(name="branch", keyword="milk")
    h = _history()
    _feed(h, p.advance(h), HOME)
    _feed(h, p.advance(h), RESULTS)
    _feed(h, p.advance(h), DONE)
    step = p.advance(h)
    assert isinstance(step, DecisionRequest)

    _finish(p, h, p.resolve(None))  # the brokered call failed → handover

    (row,) = walklog.load()
    assert row["outcome"] == "handover"
    assert row["node"] == "choose"
    assert row["micros"] == 1


# ---------- inline macros (a LEG's embedded body) ----------


# FLOW's first leg with the body embedded in place of `macro: open-app`.
INLINE_FLOW = FLOW.replace(
    "    macro: open-app\n",
    "    macro:\n"
    "      inputs:\n"
    "        message: {description: the text}\n"
    "      steps:\n"
    "        - {name: go, tool: home_screen}\n",
)


def test_inline_leg_dispatches_under_its_synthesized_name() -> None:
    # The whole wiring in one walk: parse synthesizes `flow.open`,
    # build_program merges it into the dispatch registry, and the leg's
    # run_macro turn is name-keyed like any pack macro — refs filled
    # through the node's `with:` exactly as on the directory path.
    write_pack(playbooks={"flow": INLINE_FLOW})
    p = _program(keyword="milk")
    h = _history()
    _feed(h, p.advance(h), HOME)  # the start page

    leg = p.advance(h)

    assert leg is not None and leg.tool_calls[1].arguments == {
        "name": "demo/flow.open",
        "inputs": {"message": "milk"},
    }
    assert "demo/flow.open" in p.pack_macros


def test_session_setup_hidden_registry_carries_inline_macros() -> None:
    # The registry is handed to the engine ONCE at wake — an activation
    # mid-session dispatches inline legs out of `hidden`, so they must
    # ride it beside the directory macros.
    _write_channel()
    write_pack(playbooks={"flow": INLINE_FLOW})

    _, overture, hidden = setup.session_setup()

    assert overture is not None
    assert "demo/flow.open" in hidden
