"""Tests for `physiclaw.agent.conductor.program` — arming (validation +
the arm file), fail-open loading, and the walk: synthesized turns,
fingerprint checks, resume-by-locate, and hand-over on everything this
phase refuses to guess about."""

from __future__ import annotations

import json

import pytest
from conductor_fakes import LEDGERED, PACK_MACRO, make_screen, write_pack

from physiclaw.agent.conductor import arming, channel, memory, program, setup
from physiclaw.agent.conductor.playbook import GATE_MAX_REVISIONS, PlaybookError
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
    p = setup.load_armed()
    assert p is not None
    return p


def _history() -> list[Message]:
    return [SystemMessage(content="sys"), UserMessage(content="wake")]


# ---------- arming ----------


def test_arm_writes_file_and_load_armed_builds_program() -> None:
    write_pack(playbooks={"flow": FLOW})

    spec, warnings = arming.arm("demo", "flow", {"keyword": "milk"})

    assert len(spec.nodes) == 2 and warnings == []
    assert arming.armed_ref() == ("demo", "flow")
    p = _armed_program()
    assert p.app == "demo"
    assert p.values == {"keyword": "milk"}
    assert set(p.pack_macros) == {"demo/open-app", "demo/add-cart"}


def test_disarm_removes_the_file() -> None:
    write_pack(playbooks={"flow": FLOW})
    arming.arm("demo", "flow", {"keyword": "milk"})

    assert arming.disarm() is True
    assert arming.disarm() is False
    assert arming.armed_ref() is None
    assert setup.load_armed() is None


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
        arming.arm(app, name, inputs)


def test_arm_refuses_a_disabled_playbook() -> None:
    write_pack(playbooks={"flow": FLOW + "enabled: false\n"})

    with pytest.raises(PlaybookError, match="disabled"):
        arming.arm("demo", "flow", {"keyword": "milk"})


def test_arm_refuses_disabled_leg_macros() -> None:
    write_pack(playbooks={"flow": FLOW})
    macro = paths.playbooks_dir() / "demo" / "macros" / "open-app" / "MACRO.yml"
    macro.write_text(
        PACK_MACRO.format(name="open-app") + "enabled: false\n", encoding="utf-8"
    )

    with pytest.raises(PlaybookError, match="disabled pack macro"):
        arming.arm("demo", "flow", {"keyword": "milk"})


def test_load_armed_fail_open() -> None:
    # No file at all.
    assert setup.load_armed() is None
    # Corrupt file.
    paths.playbooks_dir().mkdir(parents=True, exist_ok=True)
    (paths.playbooks_dir() / arming.ARMED_FILENAME).write_text(
        "{not json", encoding="utf-8"
    )
    assert setup.load_armed() is None
    # A file naming a playbook that no longer exists.
    write_pack(playbooks={"flow": FLOW})
    (paths.playbooks_dir() / arming.ARMED_FILENAME).write_text(
        json.dumps({"schema": 1, "app": "demo", "playbook": "gone", "inputs": {}}),
        encoding="utf-8",
    )
    assert setup.load_armed() is None


# ---------- the walk ----------


def test_walk_runs_both_legs_then_completes() -> None:
    write_pack(playbooks={"flow": FLOW})
    arming.arm("demo", "flow", {"keyword": "milk"})
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
    arming.arm("demo", "flow", {"keyword": "milk"})
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
    arming.arm("demo", "flow", {"keyword": "milk"})
    p = _armed_program()
    h = _history()

    _feed(h, p.advance(h), ELSEWHERE)
    leg1 = p.advance(h)
    assert leg1 is not None
    _feed(h, leg1, RESULTS)  # landed on the WRONG known page

    assert p.advance(h) is None


def test_error_result_hands_over() -> None:
    write_pack(playbooks={"flow": FLOW})
    arming.arm("demo", "flow", {"keyword": "milk"})
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
    arming.arm("demo", "branch", {"keyword": "milk"})
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
    arming.arm("demo", "flow", {"keyword": "milk"})
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
    arming.arm("demo", "picky", {"keyword": "milk"})
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
    arming.arm("demo", "pay", {"keyword": "milk"})
    p = setup.load_armed(channel.load_channel())
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

    _, warnings = arming.arm("demo", "pay", {"keyword": "milk"})

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
    resumed = setup.load_parked()
    assert resumed is not None and setup.load_parked() is None  # one-shot
    h2 = _history()
    peek = resumed.advance(h2)
    assert peek.tool_names() == ["note", "peek"]
    _feed(h2, peek, _thread((ask, 0.75, 0.3), ("好的", 0.25, 0.5)))
    back = resumed.advance(h2)
    assert back.tool_calls[1].arguments["name"] == "demo/open-app"


def test_gate_resume_off_thread_reopens_via_channel_open() -> None:
    p, h, send = _at_gate()
    ask = _park_via_silence(p, h, send)

    resumed = setup.load_parked()
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
    arming.arm("demo", "notify", {"keyword": "milk"})
    p = setup.load_armed(channel.load_channel())
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
    resumed = setup.load_parked()
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

    prog, activation, hidden = setup.session_setup()

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
    arming.arm("demo", "flow", {"keyword": "milk"})

    prog, activation, hidden = setup.session_setup()

    assert prog is not None and prog.channel is not None
    assert activation is None


def test_activation_fires_once_on_the_thread_page() -> None:
    from physiclaw.agent.conductor.micro import PARSE_TASK, MicroOutcome
    from physiclaw.agent.engine.dto import ToolResultMessage

    _write_channel()
    write_pack(playbooks={"flow": FLOW})
    _, activation, _ = setup.session_setup()

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
    _, activation, _ = setup.session_setup()

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

    ch = channel.load_channel()
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

    resumed = setup.load_parked()

    assert resumed is not None and resumed._gate.consented == 45.0


def test_park_status_literal_matches_the_sentinel() -> None:
    # program.py spells WAIT literally (the conductor may not import
    # engine runtimes); this pins its constant to the sentinel's spelling
    # (_park emits PARK_STATUS; the park tests assert the emitted value).
    from physiclaw.agent.runtime.sentinel import WAIT

    assert program.PARK_STATUS == WAIT


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
    arming.arm("demo", "shop", {"items": items})
    p = setup.load_armed(channel.load_channel())
    assert p is not None and p.channel is not None
    return p


def _shop_item(p, h, req, label: str):
    """Resolve one choose_item pick and drive tap + add-cart; returns
    the step after the loop closer routed (next search, or rec-peek)."""
    from physiclaw.agent.conductor.micro import MicroOutcome

    cand = next(c for c in req.candidates if c.key == label)
    tap = p.resolve(MicroOutcome(out="pick", reason="r", confidence=0.9, picked=cand))
    _feed(h, tap, PRODUCTS)
    add = p.advance(h)
    assert add.tool_calls[1].arguments["name"] == "demo/add-cart"
    _feed(h, add, PRODUCTS)
    return p.advance(h)


def _to_reconcile(p):
    """Drive a fresh ledger walk to the reconcile peek; returns (h, peek)."""
    from physiclaw.agent.conductor.micro import DecisionRequest

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
    from physiclaw.agent.conductor.micro import (
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
    # Quantity-only change: nothing pending, straight back to reconcile.
    assert step.tool_names() == ["note", "peek"]
    assert p._gate.consented is None  # the old ask no longer covers the order

    _feed(h, step, _cart_screen(("farm eggs", 2), ("lays chips", 1)))
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
    from physiclaw.agent.conductor.micro import MicroOutcome

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

    # The new item is pending → the loop re-enters `search` for it.
    assert step.tool_calls[1].arguments["inputs"]["message"] == "oil"
    assert [it.qty for it in p._ledger] == [2, 1, 1]


def test_gate_revisions_are_bounded() -> None:
    from physiclaw.agent.conductor.micro import MicroOutcome

    p, h, send = _at_ledger_gate()
    p._gate.revisions = GATE_MAX_REVISIONS

    _reply_arrives(p, h, send, "change it again, no chips this time")
    handed = p.resolve(MicroOutcome(out="revise", reason="again", confidence=0.9))

    assert handed is None  # budget spent — the model settles the order


def test_park_persists_the_ledger() -> None:
    p, _, _ = _at_ledger_gate()
    p._park(resume_idx=p._idx, awaiting=True)

    resumed = setup.load_parked()

    assert resumed is not None
    assert [it.query for it in resumed._ledger] == ["eggs", "chips"]
    assert resumed._ledger[0].status == "picked"
    assert resumed._ledger[0].label == "farm eggs"


def test_arm_rejects_a_bad_ledger_value() -> None:
    write_pack(playbooks={"shop": LEDGERED})

    with pytest.raises(PlaybookError, match="items"):
        arming.arm("demo", "shop", {"items": "not json"})
    with pytest.raises(PlaybookError, match="qty"):
        arming.arm("demo", "shop", {"items": '[{"query": "egg", "qty": 0}]'})


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
    arming.arm("demo", "fetch", {"items": '[{"query": "eggs", "qty": 1}]'})

    p = setup.load_armed()

    assert p is not None and p.needs_micro is False


# ---------- arm lifecycle (terminal outcomes consume the arm) ----------


def test_completed_walk_is_terminal_and_consumes_the_arm() -> None:
    write_pack(playbooks={"flow": FLOW})
    arming.arm("demo", "flow", {"keyword": "milk"})
    p = _armed_program()
    h = _history()
    _feed(h, p.advance(h), ELSEWHERE)
    _feed(h, p.advance(h), HOME)
    _feed(h, p.advance(h), RESULTS)

    assert p.advance(h) is None and p.finished == "complete"
    assert p.origin == "armed"
    # retire happens AT the quiet — no separate call.
    assert arming.armed_ref() is None  # done deals never re-walk


def test_gate_deny_is_terminal_and_consumes_the_arm() -> None:
    p, h, send = _at_gate()

    assert _reply_arrives(p, h, send, "不用") is None
    assert p.finished == "deny"
    assert arming.armed_ref() is None


def test_incidental_handover_keeps_the_arm() -> None:
    # A wrong-page landing is a retry-next-wake case, not a done deal.
    write_pack(playbooks={"flow": FLOW})
    arming.arm("demo", "flow", {"keyword": "milk"})
    p = _armed_program()
    h = _history()
    _feed(h, p.advance(h), ELSEWHERE)
    leg = p.advance(h)
    _feed(h, leg, RESULTS)  # verify: home expected — wrong page

    assert p.advance(h) is None and p.finished is None
    assert arming.armed_ref() == ("demo", "flow")


def test_parked_walk_deny_consumes_the_arm() -> None:
    # armed.json outlives the park; a terminal outcome on the RESUMED
    # walk must still consume it.
    p, h, send = _at_gate()
    ask = _park_via_silence(p, h, send)

    resumed = setup.load_parked()
    assert resumed is not None and resumed.origin == "parked"
    h2 = _history()
    peek = resumed.advance(h2)
    _feed(h2, peek, _thread((ask, 0.75, 0.3), ("不用", 0.25, 0.5)))

    assert resumed.advance(h2) is None and resumed.finished == "deny"
    assert arming.armed_ref() is None
