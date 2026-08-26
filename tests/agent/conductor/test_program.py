"""Tests for `physiclaw.agent.conductor.program` — arming (validation +
the arm file), fail-open loading, and the walk: synthesized turns,
fingerprint checks, resume-by-locate, and hand-over on everything this
phase refuses to guess about."""

from __future__ import annotations

import json

import pytest
from conductor_fakes import PACK_MACRO, make_screen, write_pack

from physiclaw.agent.conductor import program
from physiclaw.agent.conductor.playbook import PlaybookError
from physiclaw.agent.engine.dto import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from physiclaw.common import paths

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
ELSEWHERE = make_screen(("Nothing known", 0.5, 0.5)).text


def _feed(
    history: list[Message],
    turn: AssistantMessage,
    text: str = "",
    *,
    error: bool = False,
) -> None:
    """Append the synthesized turn plus its action's tool result — the
    loop's contract (one result per call, in the very next messages)."""
    history.append(turn)
    history.append(
        ToolResultMessage(
            tool_call_id=turn.tool_calls[1].id, content=text, is_error=error
        )
    )


def _armed_program() -> program.Program:
    p = program.load_armed()
    assert p is not None
    return p


def _history() -> list[Message]:
    return [SystemMessage(content="sys"), UserMessage(content="wake")]


# ---------- arming ----------


def test_arm_writes_file_and_load_armed_builds_program() -> None:
    write_pack(playbooks={"flow": FLOW})

    spec, warnings = program.arm("demo", "flow", {"keyword": "milk"})

    assert len(spec.nodes) == 2 and warnings == []
    assert program.armed_ref() == ("demo", "flow")
    p = _armed_program()
    assert p.app == "demo"
    assert p.values == {"keyword": "milk"}
    assert set(p.pack_macros) == {"demo/open-app", "demo/add-cart"}


def test_disarm_removes_the_file() -> None:
    write_pack(playbooks={"flow": FLOW})
    program.arm("demo", "flow", {"keyword": "milk"})

    assert program.disarm() is True
    assert program.disarm() is False
    assert program.armed_ref() is None
    assert program.load_armed() is None


@pytest.mark.parametrize(
    "app, name, inputs, fragment",
    [
        ("demo", "ghost", {}, "no playbook"),
        ("demo", "flow", {}, "missing required input"),
        ("demo", "flow", {"keyword": "x", "typo": "y"}, "unknown input"),
    ],
)
def test_arm_rejections(app, name, inputs, fragment) -> None:
    write_pack(playbooks={"flow": FLOW})

    with pytest.raises(PlaybookError, match=fragment):
        program.arm(app, name, inputs)


def test_arm_refuses_a_disabled_playbook() -> None:
    write_pack(playbooks={"flow": FLOW + "enabled: false\n"})

    with pytest.raises(PlaybookError, match="disabled"):
        program.arm("demo", "flow", {"keyword": "milk"})


def test_arm_refuses_disabled_leg_macros() -> None:
    write_pack(playbooks={"flow": FLOW})
    macro = paths.playbooks_dir() / "demo" / "macros" / "open-app" / "MACRO.yml"
    macro.write_text(
        PACK_MACRO.format(name="open-app") + "enabled: false\n", encoding="utf-8"
    )

    with pytest.raises(PlaybookError, match="disabled pack macro"):
        program.arm("demo", "flow", {"keyword": "milk"})


def test_load_armed_fail_open() -> None:
    # No file at all.
    assert program.load_armed() is None
    # Corrupt file.
    paths.playbooks_dir().mkdir(parents=True, exist_ok=True)
    (paths.playbooks_dir() / program.ARMED_FILENAME).write_text(
        "{not json", encoding="utf-8"
    )
    assert program.load_armed() is None
    # A file naming a playbook that no longer exists.
    write_pack(playbooks={"flow": FLOW})
    (paths.playbooks_dir() / program.ARMED_FILENAME).write_text(
        json.dumps({"schema": 1, "app": "demo", "playbook": "gone", "inputs": {}}),
        encoding="utf-8",
    )
    assert program.load_armed() is None


# ---------- the walk ----------


def test_walk_runs_both_legs_then_completes() -> None:
    write_pack(playbooks={"flow": FLOW})
    program.arm("demo", "flow", {"keyword": "milk"})
    p = _armed_program()
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
    program.arm("demo", "flow", {"keyword": "milk"})
    p = _armed_program()
    h = _history()

    peek = p.advance(h)
    assert peek is not None
    _feed(h, peek, HOME)
    nxt = p.advance(h)

    assert nxt is not None
    assert nxt.tool_calls[1].arguments["name"] == "demo/add-cart"


def test_verify_mismatch_hands_over() -> None:
    write_pack(playbooks={"flow": FLOW})
    program.arm("demo", "flow", {"keyword": "milk"})
    p = _armed_program()
    h = _history()

    _feed(h, p.advance(h), ELSEWHERE)
    leg1 = p.advance(h)
    assert leg1 is not None
    _feed(h, leg1, RESULTS)  # landed on the WRONG known page

    assert p.advance(h) is None


def test_error_result_hands_over() -> None:
    write_pack(playbooks={"flow": FLOW})
    program.arm("demo", "flow", {"keyword": "milk"})
    p = _armed_program()
    h = _history()

    _feed(h, p.advance(h), ELSEWHERE)
    leg1 = p.advance(h)
    assert leg1 is not None
    _feed(h, leg1, "BLOCKED — not executed", error=True)

    assert p.advance(h) is None


def test_decide_yields_a_request_and_an_unresolved_one_hands_over() -> None:
    from physiclaw.agent.conductor.micro import DecisionRequest

    write_pack(playbooks={"branch": BRANCH})
    program.arm("demo", "branch", {"keyword": "milk"})
    p = _armed_program()
    h = _history()

    _feed(h, p.advance(h), ELSEWHERE)
    _feed(h, p.advance(h), HOME)  # leg open verified
    _feed(h, p.advance(h), RESULTS)  # leg search verified

    req = p.advance(h)
    assert isinstance(req, DecisionRequest)  # the conductor brokers it
    assert p.resolve(None) is None  # a failed micro-call hands over


def test_program_advance_never_raises() -> None:
    write_pack(playbooks={"flow": FLOW})
    program.arm("demo", "flow", {"keyword": "milk"})
    p = _armed_program()

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
    from physiclaw.agent.conductor.micro import DecisionRequest

    write_pack(playbooks={"picky": PICKY})
    program.arm("demo", "picky", {"keyword": "milk"})
    p = _armed_program()
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
    from physiclaw.agent.conductor.micro import MicroOutcome

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
    from physiclaw.agent.conductor.micro import DecisionRequest, MicroOutcome

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
    from physiclaw.agent.conductor.micro import MicroOutcome

    p, _, _ = _at_decision()

    assert (
        p.resolve(MicroOutcome(out="none_fit", reason="nothing", confidence=0.9))
        is None
    )


def test_failed_micro_call_hands_over() -> None:
    p, _, _ = _at_decision()

    assert p.resolve(None) is None


# ---------- the gate, parking, activation ----------

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


def _thread(*bubbles: tuple) -> str:
    return make_screen(("MyChat", 0.5, 0.05), *bubbles).text


def _at_gate(total: str = "¥45", playbook: str = GATED):
    """Arm the gated playbook (with a channel pack) and walk to the sent
    ask; `total` is what the payment sheet shows."""
    _write_channel()
    write_pack(playbooks={"pay": playbook})
    program.arm("demo", "pay", {"keyword": "milk"})
    p = program.load_armed(program.load_channel())
    assert p is not None
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


def _park_via_silence(p, h, send) -> str:
    """Drive the full silent-round cycle to the park; returns the ask."""
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

    _, warnings = program.arm("demo", "pay", {"keyword": "milk"})

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
    from physiclaw.agent.conductor.micro import (
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


def test_gate_silence_parks_and_resumes_on_next_wake() -> None:
    p, h, send = _at_gate()
    ask = _park_via_silence(p, h, send)

    # Next wake: the parked walk resumes straight into a reply check.
    resumed = program.load_parked()
    assert resumed is not None and program.load_parked() is None  # one-shot
    h2 = _history()
    peek = resumed.advance(h2)
    assert peek.tool_names() == ["note", "peek"]
    _feed(h2, peek, _thread((ask, 0.75, 0.3), ("好的", 0.25, 0.5)))
    back = resumed.advance(h2)
    assert back.tool_calls[1].arguments["name"] == "demo/open-app"


def test_gate_resume_off_thread_reopens_via_channel_open() -> None:
    p, h, send = _at_gate()
    ask = _park_via_silence(p, h, send)

    resumed = program.load_parked()
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


def test_confirm_sends_parks_and_resumes_past_itself() -> None:
    _write_channel()
    write_pack(playbooks={"notify": CONFIRMING})
    program.arm("demo", "notify", {"keyword": "milk"})
    p = program.load_armed(program.load_channel())
    h = _history()
    _feed(h, p.advance(h), ELSEWHERE)
    _feed(h, p.advance(h), HOME)  # leg open verified

    send = p.advance(h)
    assert send.tool_calls[1].arguments["name"] == "channel/send"
    text = send.tool_calls[1].arguments["inputs"]["message"]
    # `message:` IS the sent text — refs filled, nothing code-appended.
    assert text == "已下单milk，稍后汇报进度"

    _feed(h, send, _thread((text, 0.75, 0.3)))
    park = p.advance(h)
    assert park.tool_names() == ["note", "end_session"]
    assert park.tool_calls[1].arguments["status"] == "WAIT"

    # Resume: the walk continues PAST the confirm node at its stored idx.
    resumed = program.load_parked()
    assert resumed is not None
    h2 = _history()
    peek = resumed.advance(h2)
    _feed(h2, peek, HOME)  # wherever the phone is; stored cursor is trusted
    leg = resumed.advance(h2)
    assert leg.tool_calls[1].arguments["name"] == "demo/add-cart"


# ---------- activation / session_setup ----------


def test_session_setup_builds_activation_and_hidden_registry() -> None:
    _write_channel()
    write_pack(playbooks={"flow": FLOW})

    prog, activation, hidden = program.session_setup()

    assert prog is None
    assert activation is not None
    assert tuple(activation.entries) == ("demo/flow",)
    menu = activation._menu()
    assert "demo/flow" in menu and "keyword" in menu
    assert set(hidden) == {
        "channel/send",
        "channel/open",
        "demo/open-app",
        "demo/add-cart",
    }


def test_session_setup_prefers_armed_program_over_activation() -> None:
    _write_channel()
    write_pack(playbooks={"flow": FLOW})
    program.arm("demo", "flow", {"keyword": "milk"})

    prog, activation, hidden = program.session_setup()

    assert prog is not None and prog.channel is not None
    assert activation is None


def test_activation_fires_once_on_the_thread_page() -> None:
    from physiclaw.agent.conductor.micro import PARSE_TASK, MicroOutcome
    from physiclaw.agent.engine.dto import ToolResultMessage

    _write_channel()
    write_pack(playbooks={"flow": FLOW})
    _, activation, _ = program.session_setup()

    h = _history()
    h.append(ToolResultMessage(tool_call_id="t1", content=ELSEWHERE))
    assert activation.request(h) is None  # not the thread — free

    h.append(
        ToolResultMessage(tool_call_id="t2", content=_thread(("买牛奶", 0.25, 0.4)))
    )
    req = activation.request(h)
    assert req is not None and req.call == PARSE_TASK
    assert "买牛奶" in req.listing and "demo/flow" in req.args["menu"]
    assert activation.request(h) is None  # once per session

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
    from physiclaw.agent.conductor.micro import MicroOutcome

    _write_channel()
    write_pack(playbooks={"flow": FLOW})
    _, activation, _ = program.session_setup()

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
    from physiclaw.agent.conductor import scaffold

    scaffold.init_pack("channel")

    ch = program.load_channel()
    # Pages parse and the thread page exists; the macros are scaffolded
    # DISABLED (rehearse, then enable), so the sends stay unavailable.
    assert ch is not None
    assert ch.send is None and ch.open is None
    from physiclaw.agent.conductor.playbook import load_pack

    pack = load_pack("channel")
    assert set(pack.macros) == {"send", "open"} and not pack.macro_errors


def test_gate_revise_hands_over_with_the_request() -> None:
    # "好的，但是买两盒" is a change request, not a confirmation — the LLM
    # tier answers revise and the model takes over to adjust the order.
    from physiclaw.agent.conductor.micro import DecisionRequest, MicroOutcome

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


def test_park_persists_consent_across_the_wake() -> None:
    # A post-consent park must not resume into a refused payment: the
    # consented total rides parked.json (the _Gate.to_park projection).
    p, h, send = _at_gate()
    _reply_arrives(p, h, send, "好的")
    assert p._gate.consented == 45.0
    p._park(resume_idx=p._idx, awaiting=False)

    resumed = program.load_parked()

    assert resumed is not None and resumed._gate.consented == 45.0


def test_park_status_literal_matches_the_sentinel() -> None:
    # program.py spells WAIT literally (the conductor may not import
    # engine runtimes); this pins its constant to the sentinel's spelling
    # (_park emits PARK_STATUS; the park tests assert the emitted value).
    from physiclaw.agent.runtime.sentinel import WAIT

    assert program.PARK_STATUS == WAIT


def test_memory_context_slices_fail_closed_with_token_match() -> None:
    # Least-privilege: only the declared section rides the micro-call —
    # the rest of memory.md never travels to the (possibly third-party)
    # micro tier. Token-exact heading match (no substring bleed), and
    # NO match means NO memory context: a privacy boundary must never
    # silently widen to the whole file.
    structured = (
        "## shopping_prefs 购物偏好\n- 只买伊利\n\n## shopping_blacklist\n- 三无"
    )
    sliced = program.memory_sections(structured, ["shopping_prefs"])
    assert "只买伊利" in sliced and "三无" not in sliced
    # `shopping` is a substring of both headings but a token of neither.
    assert program.memory_sections(structured, ["shopping"]) == ""

    unstructured = "只买伊利，不要临期"
    assert program.memory_sections(unstructured, ["shopping_prefs"]) == ""
