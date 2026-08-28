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
    history as _history,
)
from conductor_fakes import (
    thread_screen as _thread,
)

from physiclaw.common import paths
from physiclaw.conductor import channel, memory, program, reconcile, setup, suspension
from physiclaw.conductor.playbook import GATE_MAX_REVISIONS, PlaybookError

FLOW = """\
name: flow
description: two legs
inputs:
  keyword:
    description: what to search
nodes:
  - id: open
    type: LEG
    macro: open-app
    with: {message: "{keyword}"}
    verify: home
  - id: search
    type: LEG
    macro: add-cart
    with: {message: "go"}
    enter: home
    verify: results
"""

# Same legs, then a DECIDE.
BRANCH = (
    FLOW.replace("name: flow", "name: branch")
    + """\
  - id: choose
    type: DECIDE
    call: choose_item
    with: {criteria: "cheapest"}
    on: {pick: escalate, scroll: escalate, none_fit: escalate, escalate: escalate}
"""
)

HOME = make_screen(("Files", 0.5, 0.1)).text
RESULTS = make_screen(("综合", 0.5, 0.1)).text


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

    _feed(h, peek, ELSEWHERE)  # unknown page → start from the top
    leg1 = p.advance(h)
    assert leg1 is not None and leg1.tool_names() == ["note", "run_macro"]
    assert leg1.tool_calls[1].arguments == {
        "name": "demo/open-app",
        "inputs": {"message": "milk"},  # {keyword} resolved from the arm
    }

    _feed(h, leg1, HOME)  # verify: home holds → next leg (enter: home holds too)
    leg2 = p.advance(h)
    assert leg2 is not None
    assert leg2.tool_calls[1].arguments["name"] == "demo/add-cart"

    _feed(h, leg2, RESULTS)  # verify: results holds → playbook complete
    assert p.advance(h) is None


def test_locate_resumes_past_completed_legs() -> None:
    # A killed session's next wake: the screen already shows leg 1's
    # verify page, so the walk resumes at leg 2 — no repeated gestures.
    write_pack(playbooks={"flow": FLOW})
    p = _program(keyword="milk")
    h = _history()

    peek = p.advance(h)
    assert peek is not None
    _feed(h, peek, HOME)
    nxt = p.advance(h)

    assert nxt is not None
    assert nxt.tool_calls[1].arguments["name"] == "demo/add-cart"


def test_verify_mismatch_hands_over() -> None:
    write_pack(playbooks={"flow": FLOW})
    p = _program(keyword="milk")
    h = _history()

    _feed(h, p.advance(h), ELSEWHERE)
    leg1 = p.advance(h)
    assert leg1 is not None
    _feed(h, leg1, RESULTS)  # landed on the WRONG known page

    assert p.advance(h) is None


def test_leg_verifying_a_builtin_page_hands_over() -> None:
    # `ios` pages LOAD now — the conductor matches against them itself —
    # but a playbook still may not ACT on one: legs run this pack's
    # macros and land on this pack's pages. The loader change made this
    # guarantee easy to lose by accident, so pin it.
    write_pack(playbooks={"flow": FLOW.replace("verify: home", "verify: ios.locked")})
    p = _program(keyword="milk")
    h = _history()

    _feed(h, p.advance(h), ELSEWHERE)

    assert p.advance(h) is None


def test_error_result_hands_over() -> None:
    write_pack(playbooks={"flow": FLOW})
    p = _program(keyword="milk")
    h = _history()

    _feed(h, p.advance(h), ELSEWHERE)
    leg1 = p.advance(h)
    assert leg1 is not None
    _feed(h, leg1, "BLOCKED — not executed", error=True)

    assert p.advance(h) is None


def test_decide_yields_a_request_and_an_unresolved_one_hands_over() -> None:
    from physiclaw.conductor.micro import DecisionRequest

    write_pack(playbooks={"branch": BRANCH})
    p = _program(name="branch", **{"keyword": "milk"})
    h = _history()

    _feed(h, p.advance(h), ELSEWHERE)
    _feed(h, p.advance(h), HOME)  # leg open verified
    _feed(h, p.advance(h), RESULTS)  # leg search verified

    req = p.advance(h)
    assert isinstance(req, DecisionRequest)  # the conductor brokers it
    assert p.resolve(None) is None  # a failed micro-call hands over


def test_program_advance_never_raises() -> None:
    write_pack(playbooks={"flow": FLOW})
    p = _program(keyword="milk")

    # A malformed history (no pending result will ever match) must
    # degrade to a hand-over, not an exception.
    p.advance(_history())
    assert p.advance(_history()) is None


# ---------- decisions ----------

PICKY = """\
name: picky
description: pick flow
inputs:
  keyword:
    description: what
nodes:
  - id: open
    type: LEG
    macro: open-app
    with: {message: "{keyword}"}
    verify: results
  - id: choose
    type: DECIDE
    call: choose_item
    with: {criteria: "cheapest {keyword}"}
    on: {pick: use, scroll: choose, none_fit: escalate, escalate: escalate}
  - id: use
    type: LEG
    macro: add-cart
    with: {message: "{choose.pick}"}
    verify: home
"""


def _at_decision():
    """Arm PICKY and walk it to the choose node's DecisionRequest."""
    from physiclaw.conductor.micro import DecisionRequest

    write_pack(playbooks={"picky": PICKY})
    p = _program(name="picky", **{"keyword": "milk"})
    h = _history()
    _feed(h, p.advance(h), ELSEWHERE)  # locate: unknown → top
    _feed(h, p.advance(h), RESULTS)  # leg open verified on results
    req = p.advance(h)
    assert isinstance(req, DecisionRequest), req
    return p, h, req


def test_decide_emits_a_request_off_the_decide_time_screen() -> None:
    _, _, req = _at_decision()

    assert req.node_id == "choose" and req.call == "choose_item"
    assert req.args == {"criteria": "cheapest milk"}  # {keyword} resolved
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

    _feed(h, tap, HOME)  # post-tap screen; `use` has no enter check
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
    assert p.advance(h) is None


def test_escalate_routed_outcome_hands_over() -> None:
    from physiclaw.conductor.micro import MicroOutcome

    p, _, _ = _at_decision()

    assert (
        p.resolve(MicroOutcome(out="none_fit", reason="nothing", confidence=0.9))
        is None
    )


def test_failed_micro_call_hands_over() -> None:
    p, _, _ = _at_decision()

    assert p.resolve(None) is None


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
    with: {bbox: [0.1, 0.1, 0.2, 0.2]}
"""

GATED = """\
name: pay
description: 买牛奶
inputs:
  keyword:
    description: what
mandate:
  max_amount: 100
nodes:
  - id: open
    type: LEG
    macro: open-app
    with: {message: "{keyword}"}
    verify: results
  - id: gate
    type: HUMAN_GATE
    gate: payment
    compose: payment-request
    message: "已选好{keyword}，合计 ¥{gate.total}。回复 好的 确认支付，或 不用 取消。"
    over_message: "已选好{keyword}，合计 ¥{gate.total}，已超出预算 ¥{gate.cap}。回复 好的 确认支付，或 不用 取消。"
    return: open-app
  - id: pay
    type: LEG
    macro: add-cart
    with: {message: "pay"}
    enter: results
    verify: home
    irreversible: payment
"""


def _write_channel() -> None:
    root = paths.playbooks_dir() / "channel"
    (root / "macros" / "send").mkdir(parents=True, exist_ok=True)
    (root / "macros" / "open").mkdir(parents=True, exist_ok=True)
    (root / "pages.yml").write_text(CHANNEL_PAGES, encoding="utf-8")
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
    _feed(h, p.advance(h), ELSEWHERE)  # locate → top
    _feed(h, p.advance(h), _sheet(total))  # leg open verified on the sheet
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

    assert p.advance(h) is None  # staleness predicate → hand over


def test_gate_deny_hands_over_without_reasking() -> None:
    p, h, send = _at_gate()

    assert _reply_arrives(p, h, send, "不用") is None


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
name: notify
description: 汇报进展
inputs:
  keyword:
    description: what
nodes:
  - id: open
    type: LEG
    macro: open-app
    with: {message: "{keyword}"}
    verify: home
  - id: tell
    type: CONFIRM
    compose: status-update
    message: "已下单{keyword}，稍后汇报进度"
  - id: wrap
    type: LEG
    macro: add-cart
    with: {message: "done"}
    verify: results
"""


def test_confirm_sends_suspends_and_resumes_past_itself() -> None:
    _write_channel()
    write_pack(playbooks={"notify": CONFIRMING})
    p = _program(name="notify", keyword="milk")
    h = _history()
    _feed(h, p.advance(h), ELSEWHERE)
    _feed(h, p.advance(h), HOME)  # leg open verified

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
    _feed(h2, peek, HOME)  # wherever the phone is; stored cursor is trusted
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
    _feed(h, p.advance(h), ELSEWHERE)
    _feed(h, p.advance(h), HOME)
    send = p.advance(h)
    text = send.tool_calls[1].arguments["inputs"]["message"]
    _feed(h, send, _thread((text, 0.75, 0.3)))
    p.advance(h)  # suspend

    resumed = setup.load_suspended()
    assert resumed is not None
    h2 = _history()
    check = resumed.advance(h2)
    _feed(h2, check, _thread((text, 0.75, 0.3), ("cancel", 0.25, 0.5)))
    assert resumed.advance(h2) is None  # deny — the walk stops


def test_suspended_confirm_resume_off_thread_reopens_then_continues() -> None:
    # Banner wake: the screen is some other app. The check reopens the
    # thread once; with no cancel there, the walk resumes normally.
    _write_channel()
    write_pack(playbooks={"notify": CONFIRMING})
    p = _program(name="notify", keyword="milk")
    h = _history()
    _feed(h, p.advance(h), ELSEWHERE)
    _feed(h, p.advance(h), HOME)
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
    _feed(h2, peek, HOME)
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
    assert handed is None  # hand over — the model adjusts the order


def test_gate_ask_is_the_filled_template_exactly() -> None:
    # The playbook owns every word; the conductor owns only the slots:
    # the ask is `message:` with {keyword} and {gate.total} filled —
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
    _feed(h, p.advance(h), ELSEWHERE)
    _feed(h, p.advance(h), HOME)  # leg open
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

    assert handed is None  # budget spent — the model settles the order


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


def test_loop_only_ledger_playbook_needs_no_micro() -> None:
    # next_item is deterministic — a ledger walk with no prompted
    # decision and no gate must not make the engine wire a micro client.
    loop_only = """\
name: fetch
description: shop the list, fixed picks
inputs:
  items:
    description: the buying list
    kind: list
nodes:
  - id: open
    type: LEG
    macro: open-app
    with: {message: "cart"}
    verify: home
  - id: search
    type: LEG
    macro: open-app
    with: {message: "{item.query}"}
    verify: results
  - id: add
    type: LEG
    macro: add-cart
    with: {message: "add"}
    verify: results
  - id: advance
    type: DECIDE
    call: next_item
    with: {picked: "{item.query}"}
    on: {next: search, done: fix}
  - id: fix
    type: RECONCILE
    page: results
"""
    write_pack(playbooks={"fetch": loop_only})
    p = _program(name="fetch", items='[{"query": "eggs", "qty": 1}]')

    assert p.needs_micro is False


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

    assert p.advance(h) is None  # blind money refused, hand over


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

    assert p.advance(h) is None
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

    assert handed is None


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

    assert p.advance(h) is None  # handover — no ask was sent


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

    assert p.advance(h) is None


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
    _feed(h, p.advance(h), ELSEWHERE)
    _feed(h, p.advance(h), HOME)
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

    assert p.advance(h) is None


def _suspend_a_confirm() -> None:
    """Drive the CONFIRMING playbook to its written suspension."""
    _write_channel()
    write_pack(playbooks={"notify": CONFIRMING})
    p = _program(name="notify", keyword="milk")
    h = _history()
    _feed(h, p.advance(h), ELSEWHERE)
    _feed(h, p.advance(h), HOME)
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
    _feed(h, p.advance(h), ELSEWHERE)
    _feed(h, p.advance(h), HOME)
    send = p.advance(h)
    text = send.tool_calls[1].arguments["inputs"]["message"]
    _feed(h, send, _thread((text, 0.75, 0.3)))
    susp = p.advance(h)
    assert susp.tool_names() == ["note", "end_session"]
    assert suspension.suspended_path().exists()
    h.append(susp)  # the end_session result never arrives

    assert p.advance(h) is None
    assert not suspension.suspended_path().exists()


def test_check_warns_when_a_gate_reenters_blind() -> None:
    # Non-payment gates: no `return:` and a fall-through leg without
    # `enter:` means the first post-consent action runs off the IM
    # thread — advisory (the payment case is a parse error).
    _write_channel()
    text = """\
name: handoff
description: gate then blind leg
inputs:
  keyword:
    description: what
nodes:
  - id: open
    type: LEG
    macro: open-app
    with: {message: "{keyword}"}
    verify: home
  - id: addr
    type: HUMAN_GATE
    gate: address
    compose: addr-check
    message: "address ok? reply ok or no"
  - id: finish
    type: LEG
    macro: add-cart
    with: {message: "x"}
    verify: results
"""
    write_pack(playbooks={"handoff": text})

    spec, _ = setup.load_spec("demo", "handoff", require_live=False)

    assert any("off the IM thread" in w for w in setup.readiness_warnings(spec))


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
                outs=CALLS["choose_item"].outs,
                on={
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
    _feed(h, p.advance(h), ELSEWHERE)  # locate → top
    _feed(h, p.advance(h), RESULTS)  # the verified results page — no ¥ on it

    with caplog.at_level("INFO"):
        assert p.advance(h) is None  # handover, no ask sent
    # THIS guard, not a sibling: every handover returns None, so the
    # reason is what tells them apart.
    assert "no total readable" in caplog.text


def test_gate_hands_over_when_the_cap_cannot_resolve(caplog) -> None:
    # A `{cap}` mandate whose input turns out non-numeric: no cap means
    # no over-budget rule, so the gate hands over rather than asking
    # with an unenforceable mandate.
    ref_cap = GATED.replace("max_amount: 100", 'max_amount: "{cap}"').replace(
        "inputs:\n", "inputs:\n  cap:\n    description: budget\n"
    )
    _write_channel()
    write_pack(playbooks={"pay": ref_cap})
    p = _program(name="pay", keyword="milk", cap="oops")
    h = _history()
    _feed(h, p.advance(h), ELSEWHERE)
    _feed(h, p.advance(h), _sheet())

    with caplog.at_level("INFO"):
        assert p.advance(h) is None  # handover, no ask sent
    assert "cap could not be resolved" in caplog.text


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
        assert p.advance(h) is None  # money never fires blind
    assert "without a confirmed total" in caplog.text


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
            if step is None:
                break
            assert step.tool_names() == ["note", "tap"]  # keeps stepping until spent
    assert step is None
    assert "cart not converging" in caplog.text
