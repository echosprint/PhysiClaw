"""Tests for `physiclaw.conductor.walk.program` and its step executors — the
walk: opening peek and locate, moves with their enter/verify checks,
declared recovery, the ask (send, hold, judge, consent), the tell
(send, move on), suspension, money, and the walk's telemetry."""

from __future__ import annotations

import json

import pytest
from conductor_fakes import (
    DONE,
    ELSEWHERE,
    FLOW,
    HOME,
    RESULTS,
    make_screen,
    write_channel,
    write_pack,
)
from conductor_fakes import build_program as _program
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

from physiclaw.common import paths as paths_mod
from physiclaw.conductor.drive import build, setup
from physiclaw.conductor.spec import channel, limits, lints
from physiclaw.conductor.spec.model import PlaybookError
from physiclaw.conductor.walk import program, suspension

# ---------- the walk ----------


def test_walk_runs_both_moves_then_completes() -> None:
    write_pack(playbooks={"flow": FLOW})
    p = _program(keyword="milk")
    h = _history()

    peek = p.advance(h)
    assert peek is not None and peek.synthesized
    assert peek.tool_names() == ["note", "peek"]

    _feed(h, peek, HOME)  # the start page — the walk begins
    move1 = p.advance(h)
    assert move1 is not None and move1.tool_names() == ["note", "run_macro"]
    assert move1.tool_calls[1].arguments == {
        "name": "demo/open-app",
        "inputs": {"message": "milk"},  # {inputs.keyword} resolved from the arm
    }

    _feed(h, move1, RESULTS)  # landed on `results` → next move (enter holds too)
    move2 = p.advance(h)
    assert move2 is not None
    assert move2.tool_calls[1].arguments["name"] == "demo/add-cart"

    _feed(h, move2, DONE)  # landed on `done` → playbook complete
    summary = _finish(p, h, p.advance(h))
    assert "walk demo/flow completed" in summary


def test_walk_starts_at_the_top_whatever_the_screen_reads() -> None:
    # The screen already shows move 1's landing page — the walk still
    # begins at move 1, which expects `home`: a page match proves
    # nothing about the moves before it, and nothing fast-forwards
    # undeclared. With no recover on home the walk hands over.
    write_pack(playbooks={"flow": FLOW})
    p = _program(keyword="milk")
    h = _history()

    peek = p.advance(h)
    assert peek is not None
    _feed(h, peek, RESULTS)

    summary = _finish(p, h, p.advance(h))
    assert "move 'open' expects page 'home'" in summary


def test_verify_mismatch_hands_over_without_a_recover() -> None:
    # A wrong known page at verify: a page declaring no recover hands
    # over on the spot — what you declare is what runs, nothing hidden
    # re-peeks, waits, or taps around.
    write_pack(playbooks={"flow": FLOW})
    p = _program(keyword="milk")
    h = _history()

    _feed(h, p.advance(h), HOME)
    move1 = p.advance(h)
    assert move1 is not None
    _feed(h, move1, HOME)  # landed on the WRONG known page

    summary = _finish(p, h, p.advance(h))
    assert "did not land" in summary and "declares no recover" in summary


def test_move_verifying_a_builtin_page_is_refused_at_parse() -> None:
    # `ios` pages LOAD (the conductor matches against them itself) but a
    # playbook may not ACT on one: moves run this pack's macros and land
    # on this pack's pages — refused at the pack door, never at run time.
    write_pack(
        playbooks={
            "flow": FLOW.replace("  - page: results\n", "  - page: ios.locked\n")
        }
    )

    with pytest.raises(PlaybookError, match="reserved built-in"):
        build.load_spec("demo", "flow", require_live=False)


def test_error_result_hands_over() -> None:
    write_pack(playbooks={"flow": FLOW})
    p = _program(keyword="milk")
    h = _history()

    _feed(h, p.advance(h), HOME)
    move1 = p.advance(h)
    assert move1 is not None
    _feed(h, move1, "BLOCKED — not executed", error=True)

    summary = _finish(p, h, p.advance(h))
    assert "blocked or failed" in summary


def test_program_advance_never_raises() -> None:
    write_pack(playbooks={"flow": FLOW})
    p = _program(keyword="milk")

    # A malformed history (no pending result will ever match) must
    # degrade to a hand-over, not an exception.
    p.advance(_history())
    step = p.advance(_history())  # missing result → the handover brief
    assert step is not None and step.tool_names() == ["note", "peek"]
    assert p.advance(_history()) is None  # then permanently quiet


# A pure-text agent opens the route — the one step that brokers a
# model call in these walks.
AGENT_FLOW = """\
description: parse then walk
inputs:
  keyword:
    description: what to search
route:
  - agent: parse
    prompt: "Turn this into a search term: {inputs.keyword}"
    returns:
      term: the search term
  - page: home
  - do: open
    macro: open-app
    with: {message: "{parse.term}"}
  - page: results
"""


def test_failed_agent_call_hands_over() -> None:
    from physiclaw.conductor.walk.micro import DecisionRequest

    write_pack(playbooks={"flow": AGENT_FLOW})
    p = _program(keyword="milk")
    h = _history()
    _feed(h, p.advance(h), HOME)

    req = p.advance(h)
    assert isinstance(req, DecisionRequest)  # the conductor brokers it
    summary = _finish(p, h, p.resolve(None))  # a failed call hands over
    assert "call failed or under-confident" in summary


def test_a_late_page_match_never_skips_work_nodes() -> None:
    # A fresh wake whose screen matches a LATE move's verify page must
    # not skip the agent step before it: the walk starts at the top.
    flow = """\
description: agent in the middle
inputs:
  keyword:
    description: what
route:
  - page: home
  - do: open
    macro: open-app
    with: {message: "{inputs.keyword}"}
  - page: results
  - agent: choose
    prompt: "pick one"
    tools: [tap]
    returns:
      pick: the pick
  - page: done
  - do: wrap
    macro: add-cart
    with: {message: "x"}
  - page: done
"""
    write_pack(playbooks={"flow": flow})
    p = _program(keyword="milk")
    h = _history()
    _feed(h, p.advance(h), DONE)  # reads as the LAST move's landing

    # The cursor stayed at the top: the leading `do` run ended at the
    # agent, so the walk starts over at `open` (whose enter fails here)
    # instead of jumping to `wrap` off the coincidental page match.
    summary = _finish(p, h, p.advance(h))
    assert "move 'open' expects page 'home'" in summary


# ---------- the gate, suspending, activation ----------

CHANNEL_OPEN = """\
name: open
description: open the thread
steps:
  - tap: t
    at: [0.1, 0.1, 0.2, 0.2]
"""

GATED = """\
description: 买牛奶
inputs:
  keyword:
    description: what
route:
  - page: home
  - do: open
    macro: open-app
    with: {message: "{inputs.keyword}"}
  - page: results
  - ask: gate
    approve: payment
    total_label: "合计"
    message: "已选好{inputs.keyword}，合计 ¥{ask.total}。回复 好的 确认支付，或 不用 取消。"
    yes: ["好的"]
    no: ["不用"]
    resume: open-app
  - do: pay
    macro: add-cart
    with: {message: "pay"}
    irreversible: payment
  - page: home
"""


def _sheet(total: str = "¥45") -> str:
    return make_screen(("综合", 0.5, 0.1), (f"合计 {total}", 0.5, 0.5)).text


def _at_gate(total: str = "¥45", playbook: str = GATED):
    """Arm the gated playbook (with a channel pack) and walk to the sent
    ask; `total` is what the payment sheet shows."""
    write_channel(CHANNEL_OPEN)
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
    for _ in range(limits.DEFAULT_ASK_ROUNDS):
        assert step.tool_names() == ["note", "wait"]
        _feed(h, step, "waited")
        peek = p.advance(h)
        _feed(h, peek, thread)
        step = p.advance(h)
    assert step.tool_names() == ["note", "end_session"]
    assert step.tool_calls[1].arguments["status"] == "WAIT"
    return ask


def test_gate_ask_quotes_the_sheet_total() -> None:
    _, _, send = _at_gate()

    ask = send.tool_calls[1].arguments["inputs"]["message"]
    assert "¥45" in ask and "好的" in ask


def test_check_warns_when_a_gate_ask_quotes_no_deny_word() -> None:
    # Advisory, never blocking: an ask whose message quotes none of its
    # own words still works — a reply in other words just hands over.
    quiet = GATED.replace("回复 好的 确认支付，或 不用 取消", "veuillez répondre")
    write_channel(CHANNEL_OPEN)
    write_pack(playbooks={"pay": quiet})

    spec, pack = build.load_spec("demo", "pay", require_live=False)
    warnings = [w for w in lints.readiness_warnings(spec, pack) if "yes/no" in w]

    (warning,) = warnings
    assert "hands the walk over" in warning


def test_check_warns_about_thinly_anchored_route_pages() -> None:
    # The fixture pages declare one anchor each and no geometry is
    # learned: one OCR miss reads them unknown, so the checker says so
    # for every page the route checks — advisory, never a refusal.
    write_pack(playbooks={"flow": FLOW})
    spec, pack = build.load_spec("demo", "flow", require_live=False)

    warnings = [w for w in lints.readiness_warnings(spec, pack) if "anchor" in w]

    assert [w.split("'")[1] for w in warnings] == ["done", "home", "results"]
    assert all("one OCR miss" in w for w in warnings)


def test_gate_confirm_resumes_and_pays_under_the_predicates() -> None:
    p, h, send = _at_gate()

    back = _reply_arrives(p, h, send, "好的")  # confirmed → the resume macro
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


def test_gate_patience_is_the_asks_own() -> None:
    # `wait:` and `rounds:` — the poll cadence and the silent rounds
    # before the session suspends are the ask's declaration, not code.
    patient = GATED.replace(
        '    yes: ["好的"]\n', '    yes: ["好的"]\n    wait: 10\n    rounds: 1\n'
    )
    p, h, send = _at_gate(playbook=patient)
    ask = send.tool_calls[1].arguments["inputs"]["message"]
    thread = _thread((ask, 0.75, 0.3))
    _feed(h, send, thread)

    wait = p.advance(h)
    assert wait.tool_names() == ["note", "wait"]
    assert wait.tool_calls[1].arguments == {"seconds": 10}
    _feed(h, wait, "waited")
    _feed(h, p.advance(h), thread)  # the one silent peek
    assert p.advance(h).tool_names() == ["note", "end_session"]


def test_gate_deny_hands_over_without_reasking() -> None:
    p, h, send = _at_gate()

    step = _reply_arrives(p, h, send, "不用")

    summary = _finish(p, h, step)
    assert "user declined" in summary and "back out" in summary


def test_gate_reply_outside_the_declared_words_hands_over() -> None:
    # "那就来一份吧" is a yes in spirit, but the ask declared 好的/不用: the
    # conductor never guesses — the model reads the thread and decides.
    p, h, send = _at_gate()

    step = _reply_arrives(p, h, send, "那就来一份吧")

    summary = _finish(p, h, step)
    assert "matches none of its yes/no words" in summary
    assert "那就来一份吧" in summary


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


TELLING = """\
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


def test_tell_sends_then_the_walk_moves_on() -> None:
    # Fire-and-forget: the send lands on the thread and the cursor moves
    # to the next node in the same session — nothing suspends, nothing
    # waits for a reply. (The next move's enter check then reads the
    # thread the send left the phone on — the readiness advisory's
    # case — and hands over with no recover declared.)
    write_channel(CHANNEL_OPEN)
    write_pack(playbooks={"notify": TELLING})
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

    summary = _finish(p, h, p.advance(h))

    assert "move 'wrap' expects page 'results'" in summary
    assert not suspension.suspended_path().exists()
    assert p.idx == 2  # the cursor stood on `wrap`, past the tell


def test_tell_landing_off_the_thread_hands_over() -> None:
    write_channel(CHANNEL_OPEN)
    write_pack(playbooks={"notify": TELLING})
    p = _program(name="notify", keyword="milk")
    h = _history()
    _feed(h, p.advance(h), HOME)
    _feed(h, p.advance(h), RESULTS)
    send = p.advance(h)
    _feed(h, send, HOME)  # the send macro did not reach the thread

    summary = _finish(p, h, p.advance(h))
    assert "did not land on the thread" in summary


# ---------- activation / session_setup ----------


def test_session_setup_builds_the_boot_and_hidden_registry() -> None:
    # The boot is the channel pack's own playbook, materialized beside
    # a channel that has none: a dry Program over its route, carrying
    # the activation (the menu of enabled playbooks) for its `activate`
    # step, and the whole dispatch table beside it.
    from physiclaw.common import paths
    from physiclaw.conductor.spec.model import ActivateNode

    write_channel(CHANNEL_OPEN)
    write_pack(playbooks={"flow": FLOW})

    prog, hidden = setup.session_setup()

    assert prog is not None
    assert prog.app == "channel" and prog.spec.name == "boot" and prog.dry
    assert (paths.playbooks_dir() / "channel" / "boot.yml").exists()
    assert isinstance(prog.spec.nodes[-1], ActivateNode)
    activation = prog.activation
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


def test_session_setup_builds_no_boot_without_an_open_macro() -> None:
    # The boot names `open` as its hand; a channel that cannot open the
    # thread has no live boot: a plain model session, and no pack
    # discovery paid for nothing.
    write_channel()  # send only
    write_pack(
        playbooks={
            "flow": FLOW.replace(
                "description: two moves", "description: two moves\nenabled: true"
            )
        }
    )

    prog, hidden = setup.session_setup()

    assert prog is None
    assert set(hidden) == {"channel/send"}


def test_session_setup_builds_no_boot_without_an_enabled_playbook() -> None:
    # Nothing to offer, nothing to boot for — no wake pays the boot's
    # turns to read a thread no playbook could answer.
    write_channel(CHANNEL_OPEN)
    write_pack(playbooks={"flow": FLOW.replace("enabled: true", "enabled: false")})
    (paths_mod.playbooks_dir() / "demo" / "flow.yml").write_text(
        "name: flow\nenabled: false\n" + FLOW, encoding="utf-8"
    )

    prog, hidden = setup.session_setup()

    assert prog is None
    assert "channel/open" in hidden and "demo/open-app" in hidden


def test_session_setup_prefers_a_suspended_walk_over_the_boot() -> None:
    from physiclaw.common.logger import write_json_atomic

    write_channel(CHANNEL_OPEN)
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

    prog, hidden = setup.session_setup()

    assert prog is not None and prog.channel is not None
    assert prog.spec.name == "flow"  # the suspension, not the boot


def test_activation_builds_a_request_over_the_thread_screen() -> None:
    from physiclaw.common.listing import Screen
    from physiclaw.conductor.walk.micro import PARSE_TASK, MicroOutcome

    write_channel(CHANNEL_OPEN)
    write_pack(playbooks={"flow": FLOW})
    boot, _ = setup.session_setup()
    assert boot is not None and boot.activation is not None
    activation = boot.activation

    # The caller establishes the screen IS the thread (the boot's enter
    # check read it) — this turns it into the call.
    req = activation.request(Screen.read(_thread(("买牛奶", 0.25, 0.4))), "parse")
    assert req is not None and req.call == PARSE_TASK and req.node_id == "parse"
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
    write_channel(CHANNEL_OPEN)
    spec = FLOW.replace(
        "    description: what to search\n",
        "    description: what to search\n    example: rice 5kg\n",
    )
    write_pack(playbooks={"flow": spec})
    boot, _ = setup.session_setup()
    assert boot is not None and boot.activation is not None

    menu = boot.activation._menu()

    assert "keyword (what to search; e.g. rice 5kg)" in menu


def test_activation_rejects_unresolvable_inputs_and_not_a_task() -> None:
    from physiclaw.conductor.walk.micro import MicroOutcome

    write_channel(CHANNEL_OPEN)
    write_pack(playbooks={"flow": FLOW})
    boot, _ = setup.session_setup()
    assert boot is not None and boot.activation is not None
    activation = boot.activation

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
    from physiclaw.conductor.spec import scaffold

    scaffold.init_pack("channel")

    ch = channel.load_channel()
    # Pages parse and the thread page exists; the macros are scaffolded
    # DISABLED (rehearse, then enable), so the sends stay unavailable —
    # and the boot, which names `open`, is not live until it is.
    assert ch is not None
    assert ch.send is None and ch.open is None
    assert ch.boot is None and ch.pack is not None
    from physiclaw.conductor.spec.pack import load_pack

    pack = load_pack("channel")
    assert set(pack.macros) == {"send", "open"} and not pack.macro_errors


def test_gate_ask_is_the_filled_template_exactly() -> None:
    # The playbook owns every word; the conductor owns only the slots:
    # the ask is `message:` with {inputs.keyword} and {ask.total} filled —
    # nothing appended, nothing reworded.
    _, _, send = _at_gate()

    ask = send.tool_calls[1].arguments["inputs"]["message"]
    assert ask == "已选好milk，合计 ¥45。回复 好的 确认支付，或 不用 取消。"


def test_suspension_persists_consent_across_the_wake() -> None:
    # The consented total rides the projection (the Gate half of
    # `Program.state`), so a walk restored from it can never resume into
    # a refused payment.
    from physiclaw.common.logger import write_json_atomic

    p, h, send = _at_gate()
    _reply_arrives(p, h, send, "好的")
    assert p.gate.consented == 45.0
    write_json_atomic(suspension.suspended_path(), p.state())

    resumed = setup.load_suspended()

    assert resumed is not None and resumed.gate.consented == 45.0


def test_suspend_status_literal_matches_the_sentinel() -> None:
    # program.py spells WAIT literally (the conductor may not import
    # engine runtimes); this pins its constant to the sentinel's spelling.
    from physiclaw.agent.runtime.sentinel import WAIT

    assert program.SUSPEND_STATUS == WAIT


# ---------- regressions: money and suspension holes ----------


def test_payment_never_fires_off_an_unverified_screen() -> None:
    # The gate's own ask bubble quotes the consented total — if the
    # resume macro fails to leave the thread, the money predicates must
    # not be satisfied by our own message.
    p, h, send = _at_gate()
    back = _reply_arrives(p, h, send, "好的")  # confirmed → resume macro

    # The resume macro FAILS to leave the messenger: the walk sees the
    # chat list (¥ amounts may sit in previews) — no verified demo page.
    _feed(
        h,
        back,
        make_screen(("Weixin", 0.5, 0.05), ("合计 ¥45 昨天", 0.3, 0.2)).text,
    )

    summary = _finish(p, h, p.advance(h))  # blind money refused, hand over
    assert "move 'pay' expects page" in summary
    # Consent was bound but never consumed — the brief must say so.
    assert "consented to ¥45" in summary and "NOT been made" in summary


def test_pay_consumes_the_consent() -> None:
    p, h, send = _at_gate()
    back = _reply_arrives(p, h, send, "好的")
    _feed(h, back, _sheet())
    pay = p.advance(h)

    assert pay.tool_calls[1].arguments["name"] == "demo/add-cart"
    assert p.gate.consented is None  # spent at fire — no leftovers


def test_suspended_idx_outside_the_spec_drops_the_suspension() -> None:
    p, h, send = _at_gate()
    _suspend_via_silence(p, h, send)

    susp_file = suspension.suspended_path()
    data = json.loads(susp_file.read_text(encoding="utf-8"))
    data["idx"] = 99  # spec edited shorter between wakes
    susp_file.write_text(json.dumps(data), encoding="utf-8")

    assert setup.load_suspended() is None  # dropped, not a fake completion


def test_blocked_suspend_end_session_drops_the_suspension() -> None:
    p, h, send = _at_gate()
    ask = send.tool_calls[1].arguments["inputs"]["message"]
    thread = _thread((ask, 0.75, 0.3))
    _feed(h, send, thread)
    step = p.advance(h)
    for _ in range(limits.DEFAULT_ASK_ROUNDS):  # silent rounds → suspend turn
        _feed(h, step, "waited")
        peek = p.advance(h)
        _feed(h, peek, thread)
        step = p.advance(h)
    assert step.tool_names() == ["note", "end_session"]

    _feed(h, step, "BLOCKED", error=True)  # end_session refused

    summary = _finish(p, h, p.advance(h))
    assert "suspension dropped" in summary
    assert setup.load_suspended() is None  # stale suspension not resurrected


def _suspended_ask():
    """Walk GATED to its suspension turn (the ask ran out of patience);
    returns (program, history, the end_session turn — not yet fed)."""
    p, h, send = _at_gate()
    ask = send.tool_calls[1].arguments["inputs"]["message"]
    thread = _thread((ask, 0.75, 0.3))
    _feed(h, send, thread)
    step = p.advance(h)
    for _ in range(limits.DEFAULT_ASK_ROUNDS):
        _feed(h, step, "waited")
        peek = p.advance(h)
        _feed(h, peek, thread)
        step = p.advance(h)
    assert step.tool_names() == ["note", "end_session"]
    return p, h, step


def test_missing_suspend_result_drops_the_suspension_file() -> None:
    # The suspension file is written before end_session; if that result
    # never lands, the session may run on — a dead walk must not
    # resurrect.
    p, h, susp = _suspended_ask()
    assert suspension.suspended_path().exists()
    h.append(susp)  # the end_session result never arrives

    summary = _finish(p, h, p.advance(h))
    assert "suspension dropped" in summary
    assert not suspension.suspended_path().exists()


def test_clear_suspended_drops_a_suspension_file() -> None:
    _suspended_ask()

    assert suspension.clear_suspended() is True
    assert not suspension.suspended_path().exists()


def _write_suspended(playbook: str, idx: int, **over) -> None:
    """suspended.json with the boilerplate defaulted — tests pass only
    their deltas. The production shape is `Program.state`; a
    schema change is edited here once beside the tests that fake it."""
    from physiclaw.common.logger import write_json_atomic

    data = {
        "schema": suspension.SUSPENDED_SCHEMA,
        "app": "demo",
        "playbook": playbook,
        "idx": idx,
        "values": {},
        "outputs": {},
        "ask_text": "",
        "baseline": [],
        "quoted": None,
        "consented": None,
        "seen": [],
        "awaiting": False,
    }
    data.update(over)
    write_json_atomic(suspension.suspended_path(), data)


def test_payment_gate_total_is_quoted_only_off_a_verified_page() -> None:
    # A suspended resume straight onto the gate with an unknown screen
    # must hand over, not quote max(¥) off whatever the camera saw.
    write_channel(CHANNEL_OPEN)
    write_pack(playbooks={"pay": GATED})
    _write_suspended("pay", 1, values={"keyword": "milk"})  # idx 1 = the ask
    p = setup.load_suspended(channel.load_channel())
    assert p is not None
    h = _history()
    _feed(h, p.advance(h), ELSEWHERE)  # unknown screen at the gate

    summary = _finish(p, h, p.advance(h))  # handover — no ask was sent
    assert "refusing to ask blind" in summary


TWO_ASKS = """\
description: two asks
inputs:
  keyword:
    description: what
route:
  - page: home
  - do: open
    macro: open-app
    with: {message: "{inputs.keyword}"}
  - page: results
  - ask: address
    approve: address
    message: "地址没变吧？回复 好的 或 不用"
    yes: ["好的"]
    no: ["不用", "cancel"]
    resume: open-app
  - ask: handoff
    approve: handoff
    message: "现在下单吗？回复 好的 或 不用"
    yes: ["好的"]
    no: ["不用", "cancel"]
"""


def test_second_ask_reads_a_deny_sent_meanwhile() -> None:
    # The user cancels while the walk is off in the app between two
    # asks; the "cancel" sits on the thread when the second send lands.
    # Overwriting the baseline would swallow it forever — the send must
    # read it first.
    write_channel(CHANNEL_OPEN)
    write_pack(playbooks={"two": TWO_ASKS})
    p = _program(name="two", keyword="milk")
    h = _history()
    _feed(h, p.advance(h), HOME)
    _feed(h, p.advance(h), RESULTS)
    send = p.advance(h)
    back = _reply_arrives(p, h, send, "好的")  # the first ask confirmed
    assert back.tool_calls[1].arguments["name"] == "demo/open-app"
    _feed(h, back, RESULTS)
    resend = p.advance(h)
    ask1 = send.tool_calls[1].arguments["inputs"]["message"]
    ask2 = resend.tool_calls[1].arguments["inputs"]["message"]
    _feed(
        h, resend, _thread((ask1, 0.75, 0.2), ("cancel", 0.25, 0.4), (ask2, 0.75, 0.6))
    )

    summary = _finish(p, h, p.advance(h))
    assert "user declined" in summary


def test_second_ask_reads_a_yes_that_repeats_the_first() -> None:
    # Both asks take "好的"; the first reply is still on screen (and in
    # the second ask's baseline) when the user says "好的" again.
    write_channel(CHANNEL_OPEN)
    write_pack(playbooks={"two": TWO_ASKS})
    p = _program(name="two", keyword="milk")
    h = _history()
    _feed(h, p.advance(h), HOME)
    _feed(h, p.advance(h), RESULTS)
    send = p.advance(h)
    back = _reply_arrives(p, h, send, "好的")
    _feed(h, back, RESULTS)
    resend = p.advance(h)
    ask1 = send.tool_calls[1].arguments["inputs"]["message"]
    ask2 = resend.tool_calls[1].arguments["inputs"]["message"]
    thread = [(ask1, 0.75, 0.2), ("好的", 0.25, 0.35), (ask2, 0.75, 0.5)]
    _feed(h, resend, _thread(*thread))
    _feed(h, p.advance(h), "waited")
    _feed(h, p.advance(h), _thread(*thread, ("好的", 0.25, 0.7)))

    summary = _finish(p, h, p.advance(h))
    assert "completed" in summary


def test_gate_seen_amounts_survive_the_suspension() -> None:
    gate = program.Gate(quoted=45.0, consented=45.0, seen=(45.0, 79.0))

    restored = program.Gate.from_suspended(gate.to_suspended())

    assert restored.seen == (45.0, 79.0) and restored.consented == 45.0


# ---------- payment gate: total edges ----------


def test_consent_binds_to_the_quoted_total() -> None:
    # Whatever the sheet says is what the user is asked about and what
    # they consent to — there is no other bound.
    p, h, send = _at_gate("¥145")
    assert "¥145" in send.tool_calls[1].arguments["inputs"]["message"]

    _reply_arrives(p, h, send, "好的")

    assert p.gate.consented == 145.0


def test_gate_hands_over_when_no_total_is_readable() -> None:
    # The sheet page verified but shows no ¥ amount: the ask IS the
    # consent record, so with nothing to quote the gate refuses to ask.
    write_channel(CHANNEL_OPEN)
    write_pack(playbooks={"pay": GATED})
    p = _program(name="pay", keyword="milk")
    h = _history()
    _feed(h, p.advance(h), HOME)  # the start page
    _feed(h, p.advance(h), RESULTS)  # the verified results page — no ¥ on it

    step = p.advance(h)  # handover, no ask sent
    assert "no amount readable beside 合计" in _finish(p, h, step)


def test_payment_move_without_consent_hands_over() -> None:
    # A resume landing directly ON the payment move with no consent
    # recorded (the gate never confirmed): money never fires. This is
    # `money.fire_block`'s first predicate — the last line of defense
    # if every earlier guard were somehow skipped.
    write_channel(CHANNEL_OPEN)
    write_pack(playbooks={"pay": GATED})
    spec, _ = build.load_spec("demo", "pay")
    pay_idx = next(
        i for i, n in enumerate(spec.nodes) if getattr(n, "irreversible", None)
    )
    _write_suspended(
        "pay",
        pay_idx,
        values={"keyword": "milk"},
        quoted=45.0,
        # consented stays None — the gate never opened.
    )
    p = setup.load_suspended(channel.load_channel())
    assert p is not None
    h = _history()
    _feed(h, p.advance(h), _sheet())  # verified sheet at the pay move

    step = p.advance(h)  # money never fires blind
    assert "without a confirmed total" in _finish(p, h, step)


# ---------- declared recovery ----------


LOCKED_MID = make_screen(("Enter Passcode", 0.5, 0.5)).text

LANDMARKS = """\
back:
  label: "back chevron"
  at: [0.02, 0.05, 0.10, 0.10]
"""

# FLOW with declared hands: home force-quits, results pops back with
# the OS gesture.
RECOVERING = FLOW.replace(
    "  - page: home\n",
    "  - page: home\n    recover: force_quit\n",
).replace(
    "  - page: results\n",
    "  - page: results\n    recover: go_back\n",
)


def _recovering_walk(flow: str = RECOVERING):
    """A walk with declared hands, driven past the opening peek to move 1."""
    write_pack(playbooks={"flow": flow}, landmarks=LANDMARKS)
    p = _program(keyword="milk")
    h = _history()
    _feed(h, p.advance(h), HOME)  # the start page
    move1 = p.advance(h)
    assert move1 is not None and move1.tool_names() == ["note", "run_macro"]
    return p, h, move1


def test_wrong_page_on_resume_runs_the_hand_then_the_move() -> None:
    # A resumed walk stands on `home` while its move expects `results`:
    # the page's declared hand (go_back) runs, the enter check re-runs
    # on the restored page, and the move fires — the cursor never moved.
    write_pack(playbooks={"flow": RECOVERING}, landmarks=LANDMARKS)
    _write_suspended("flow", 1, values={"keyword": "milk"})
    p = setup.load_suspended(channel.load_channel())
    assert p is not None
    h = _history()
    _feed(h, p.advance(h), HOME)  # `search` enters at results — wrong page
    back = p.advance(h)
    assert back is not None and back.tool_names() == ["note", "go_back"]
    assert "declared hand" in back.tool_calls[0].arguments["summary"]

    _feed(h, back, RESULTS)
    move2 = p.advance(h)
    assert move2 is not None
    assert move2.tool_calls[1].arguments["name"] == "demo/add-cart"
    assert "recovered demo.results" in move2.tool_calls[0].arguments["summary"]


def test_a_stepping_position_restarts_below_its_cursor_a_suspension_never() -> None:
    # The rig, 2026-09-03: `step --at search` with the phone off the app
    # ran the home page's force_quit hand twice and hit its limit —
    # the position had been overlaid as a suspension, whose stored
    # cursor is a floor no recovery restarts below. A checkpoint keeps
    # the fresh walk's rule: the hand ran, walk again from the top.
    write_pack(playbooks={"flow": RECOVERING}, landmarks=LANDMARKS)
    spec, pack = build.load_spec("demo", "flow", require_live=False)
    at_search = {**_program(keyword="milk").state(), "idx": 1}

    stepped = build.build_program(
        spec, pack, {"keyword": "milk"}, None, position=at_search, dry=True
    )
    h = _history()
    _feed(h, stepped.advance(h), ELSEWHERE)
    back = stepped.advance(h)
    assert back is not None and back.tool_names() == ["note", "go_back"]
    _feed(h, back, ELSEWHERE)  # the hand did not restore results
    relaunch = stepped.advance(h)
    assert relaunch is not None  # …so the walk restarted at move 1
    assert (
        stepped.idx == 0
        and "walking again" in relaunch.tool_calls[0].arguments["summary"]
    )

    resumed = build.build_program(
        spec, pack, {"keyword": "milk"}, None, suspended=at_search, dry=True
    )
    h2 = _history()
    _feed(h2, resumed.advance(h2), ELSEWHERE)
    _feed(h2, resumed.advance(h2), ELSEWHERE)
    again = resumed.advance(h2)
    assert again is not None and again.tool_names() == ["note", "go_back"]
    assert resumed.idx == 1  # the floor: never below the stored cursor


def test_declared_unlock_hand_wakes_the_phone_then_continues() -> None:
    # Nothing unlocks in the background: the page declares the
    # `unlock_phone` hand, and only then does a locked phone get woken.
    flow = FLOW.replace(
        "  - page: results\n", "  - page: results\n    recover: unlock_phone\n"
    )
    p, h, move1 = _recovering_walk(flow)
    _feed(h, move1, LOCKED_MID)

    unlock = p.advance(h)
    assert unlock is not None and unlock.tool_names() == ["note", "unlock_phone"]

    _feed(h, unlock, RESULTS)
    move2 = p.advance(h)
    assert move2 is not None
    assert move2.tool_calls[1].arguments["name"] == "demo/add-cart"


def test_failed_move_hands_over() -> None:
    # A macro that fails hands over — nothing re-runs in the background,
    # whatever the failure text says.
    p, h, move1 = _recovering_walk()
    _feed(
        h,
        move1,
        "macro demo/open-app: ABORTED at step 2/3 (guard_failed) — "
        "steps 1–1 already executed. Do NOT re-run.",
        error=True,
    )

    summary = _finish(p, h, p.advance(h))
    assert "blocked or failed" in summary


def test_recover_tap_hand_falls_back_to_the_declared_bbox() -> None:
    # A tap hand whose label is not on screen presses the declared spot.
    flow = FLOW.replace(
        "  - page: results\n",
        "  - page: results\n    recover: {tap: landmarks.back}\n",
    )
    p, h, move1 = _recovering_walk(flow)
    _feed(h, move1, ELSEWHERE)  # unknown landing

    back = p.advance(h)

    assert back is not None and back.tool_names() == ["note", "tap"]
    assert back.tool_calls[1].arguments == {"bbox": [0.02, 0.05, 0.10, 0.10]}
    assert "declared hand" in back.tool_calls[0].arguments["summary"]


def test_hand_that_does_not_restore_relocates_from_the_top() -> None:
    # After the hand the page still does not read: the walk starts the
    # route over from its first unsettled node.
    p, h, move1 = _recovering_walk()
    _feed(h, move1, ELSEWHERE)
    back = p.advance(h)
    assert back.tool_names() == ["note", "go_back"]

    _feed(h, back, HOME)  # popped all the way to home

    again = p.advance(h)
    assert (
        again is not None and again.tool_calls[1].arguments["name"] == "demo/open-app"
    )
    assert "walking again" in again.tool_calls[0].arguments["summary"]


OCCLUDABLE_PAGES = """\
home:
  anchors: ["Files"]
results:
  anchors: ["综合", "销量"]
done:
  anchors: ["AllDone"]
"""

KEYED = FLOW.replace(
    "  - page: results\n",
    "  - page: results\n    recover:\n"
    "      covered: {tap: landmarks.back}\n"
    "      elsewhere: go_back\n",
)


def _learn_results() -> None:
    """Calibrated geometry for `results` — an overlay verdict needs it."""
    from physiclaw.conductor.spec import pages

    def anchor(text, cy):
        return pages.LearnedAnchor(
            text=text, cx=0.5, cy=cy, pos_tol=0.03, freq=1.0, weight=1.0
        )

    pages.save_learned(
        "demo",
        {
            "results": pages.LearnedPage(
                anchors={"综合": anchor("综合", 0.1), "销量": anchor("销量", 0.2)},
                threshold=0.75,
                observations=6,
            )
        },
    )


# `results` with its second anchor hidden under a coupon sheet: three
# unexpected labels in the band where 销量 should be.
COVERED_RESULTS = make_screen(
    ("综合", 0.5, 0.1),
    ("领券", 0.5, 0.18),
    ("消费券", 0.5, 0.2),
    ("马上去用", 0.5, 0.22),
).text


def test_keyed_recover_runs_the_hand_for_the_reading() -> None:
    # The page itself under an overlay → its `covered` hand (the scrim
    # tap); any other screen → its `elsewhere` hand (go_back).
    write_pack(playbooks={"flow": KEYED}, pages=OCCLUDABLE_PAGES, landmarks=LANDMARKS)
    _learn_results()
    p = _program(keyword="milk")
    h = _history()
    _feed(h, p.advance(h), HOME)
    move1 = p.advance(h)
    _feed(h, move1, COVERED_RESULTS)

    dismiss = p.advance(h)
    assert dismiss is not None and dismiss.tool_names() == ["note", "tap"]
    assert "(covered)" in dismiss.tool_calls[0].arguments["summary"]
    assert dismiss.tool_calls[1].arguments == {"bbox": [0.02, 0.05, 0.10, 0.10]}

    # The sheet is gone: both anchors read where they were learned.
    _feed(h, dismiss, make_screen(("综合", 0.5, 0.1), ("销量", 0.5, 0.2)).text)
    move2 = p.advance(h)
    assert move2 is not None and move2.tool_names() == ["note", "run_macro"]
    assert "recovered demo.results" in move2.tool_calls[0].arguments["summary"]

    # A fresh walk landing on an unknown screen takes the other hand.
    p2 = _program(keyword="milk")
    h2 = _history()
    _feed(h2, p2.advance(h2), HOME)
    _feed(h2, p2.advance(h2), ELSEWHERE)
    back = p2.advance(h2)
    assert back is not None and back.tool_names() == ["note", "go_back"]
    assert "(elsewhere)" in back.tool_calls[0].arguments["summary"]


def test_a_locked_phone_takes_the_locked_hand_and_no_other() -> None:
    # The matcher reads the lock screen by shape, so a page's hands key
    # it as `locked`: with none declared for it, the walk hands over
    # saying so (a go_back hand would land nowhere), and the reading
    # names the OS page rather than "no known page".
    keyed = FLOW.replace(
        "  - page: results\n",
        "  - page: results\n    recover:\n      elsewhere: go_back\n",
    )
    p, h, move1 = _recovering_walk(keyed)
    _feed(h, move1, LOCKED_MID)

    summary = _finish(p, h, p.advance(h))
    assert "declares no `locked` recover hand" in summary
    assert "ios.locked" in summary

    with_lock = keyed.replace(
        "      elsewhere: go_back\n",
        "      elsewhere: go_back\n      locked: unlock_phone\n",
    )
    p2, h2, m1 = _recovering_walk(with_lock)
    _feed(h2, m1, LOCKED_MID)
    unlock = p2.advance(h2)
    assert unlock is not None and unlock.tool_names() == ["note", "unlock_phone"]
    assert "(locked)" in unlock.tool_calls[0].arguments["summary"]


def test_keyed_recover_without_a_hand_for_the_reading_hands_over() -> None:
    only_occluded = FLOW.replace(
        "  - page: results\n",
        "  - page: results\n    recover:\n      covered: {tap: landmarks.back}\n",
    )
    p, h, move1 = _recovering_walk(only_occluded)
    _feed(h, move1, ELSEWHERE)

    summary = _finish(p, h, p.advance(h))
    assert "declares no `elsewhere` recover hand" in summary


def test_failed_payment_call_logs_the_purchase_and_briefs_it() -> None:
    # Consent is spent when the pay macro is dispatched; an error result
    # means money MAY have moved: the purchase line lands and the brief
    # says the payment fired with an unverified result.
    from physiclaw.common import daylog

    p, h, send = _at_gate()
    back = _reply_arrives(p, h, send, "好的")
    _feed(h, back, _sheet())
    pay = p.advance(h)
    assert pay.tool_calls[1].arguments["name"] == "demo/add-cart"
    _feed(h, pay, "macro demo/add-cart: ABORTED at step 1/1", error=True)

    summary = _finish(p, h, p.advance(h))

    assert "A payment of ¥45 was FIRED" in summary
    assert "payment ¥45 fired" in daylog.load_recent_entries(5)


def test_recovery_never_restarts_once_a_payment_fired() -> None:
    # A restart from the top would walk back into the ask and pay again:
    # after the payment move, a deviation is the model's even where the
    # page declares a hand.
    gated = GATED.replace(
        "  - page: home\n  - do: open",
        "  - page: home\n    recover: go_back\n  - do: open",
    )
    p, h, send = _at_gate(playbook=gated)
    back = _reply_arrives(p, h, send, "好的")
    _feed(h, back, _sheet())
    pay = p.advance(h)
    _feed(h, pay, ELSEWHERE)  # the pay macro did not land on home

    summary = _finish(p, h, p.advance(h))
    assert "did not land on 'home'" in summary


def test_resumed_walk_never_restarts_below_its_cursor() -> None:
    # The nodes before the stored cursor ran on an earlier wake; a hand
    # that does not restore walks again from the cursor, not the top —
    # so the page's limit is what ends it, never a cross-wake loop.
    write_pack(playbooks={"flow": RECOVERING}, landmarks=LANDMARKS)
    _write_suspended("flow", 1, values={"keyword": "milk"})
    p = setup.load_suspended(channel.load_channel())
    assert p is not None
    h = _history()
    _feed(h, p.advance(h), HOME)  # `search` expects results
    back = p.advance(h)
    assert back.tool_names() == ["note", "go_back"]
    _feed(h, back, HOME)  # not restored

    again = p.advance(h)
    assert again.tool_names() == ["note", "go_back"]  # NOT `open` re-run
    _feed(h, again, HOME)

    summary = _finish(p, h, p.advance(h))
    assert "recover tries (2) spent" in summary


def test_resumed_walk_unlocks_a_locked_phone_before_reading() -> None:
    # A wake minutes later meets the cover: one unlock, then the opening
    # exactly as before — no walk-level hand can run before it reads.
    write_pack(playbooks={"flow": FLOW})
    _write_suspended("flow", 1, values={"keyword": "milk"})
    p = setup.load_suspended(channel.load_channel())
    assert p is not None
    h = _history()
    _feed(h, p.advance(h), LOCKED_MID)

    unlock = p.advance(h)
    assert unlock.tool_names() == ["note", "unlock_phone"]
    _feed(h, unlock, RESULTS)
    peek = p.advance(h)
    assert peek.tool_names() == ["note", "peek"]
    _feed(h, peek, RESULTS)

    move2 = p.advance(h)
    assert move2.tool_calls[1].arguments["name"] == "demo/add-cart"


def test_trailing_tell_completes_the_walk() -> None:
    write_channel()
    write_pack(playbooks={"flow": FLOW + '  - tell: finish\n    message: "all done"\n'})
    p = _program(keyword="milk")
    h = _history()
    _feed(h, p.advance(h), HOME)
    _feed(h, p.advance(h), RESULTS)
    _feed(h, p.advance(h), DONE)
    send = p.advance(h)
    assert send.tool_calls[1].arguments["name"] == "channel/send"
    _feed(h, send, _thread(("all done", 0.75, 0.3)))

    summary = _finish(p, h, p.advance(h))
    assert "completed" in summary
    assert not suspension.suspended_path().exists()


def test_check_warns_when_an_ask_without_resume_precedes_a_screen_move() -> None:
    flow = TWO_ASKS.replace("    resume: open-app\n", "") + (
        "  - page: results\n  - do: wrap\n    macro: add-cart\n"
        '    with: {message: "x"}\n  - page: done\n'
    )
    write_channel(CHANNEL_OPEN)
    write_pack(playbooks={"two": flow})
    spec, pack = build.load_spec("demo", "two", require_live=False)

    warnings = [w for w in lints.readiness_warnings(spec, pack) if "resume" in w]

    assert len(warnings) == 1 and "ask 'handoff'" in warnings[0]


def test_recovery_never_runs_with_consent_bound() -> None:
    # Money keeps the hard handover: a deviation after the user consented
    # is the model's, never a hand's.
    gated = GATED.replace(
        "  - page: results\n", "  - page: results\n    recover: go_back\n"
    )
    p, h, send = _at_gate(playbook=gated)
    back = _reply_arrives(p, h, send, "好的")
    _feed(h, back, HOME)  # the resume macro landed on the wrong page

    summary = _finish(p, h, p.advance(h))
    assert "move 'pay' expects page" in summary


# ---------- walk telemetry (runs.jsonl) ----------


def test_completed_walk_records_one_completed_run_line() -> None:
    from physiclaw.conductor.walk import walklog

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
    from physiclaw.conductor.walk import walklog

    p, h, move1 = _recovering_walk()
    _feed(h, move1, HOME)  # move 1 landed on the WRONG page
    _feed(h, p.advance(h), HOME)  # the declared hand — still wrong → route top
    _feed(h, p.advance(h), HOME)  # move 1 re-runs, lands wrong again
    _feed(h, p.advance(h), HOME)  # hand again

    step = p.advance(h)
    for _ in range(40):  # relaunch attempts until the walk budget is spent
        if "conductor handing over" in step.tool_calls[0].arguments["summary"]:
            break
        _feed(h, step, HOME)
        step = p.advance(h)
    _finish(p, h, step)

    (row,) = walklog.load()
    assert row["outcome"] == "handover"
    assert row["node"] == "open"
    assert row["rescues"] >= 2


def test_completed_payment_walk_records_history_fields() -> None:
    # The completed line carries the structured fields: inputs and the
    # fired total (consent is consumed at fire — this is where it
    # survives).
    from physiclaw.conductor.walk import walklog

    p, h, send = _at_gate()
    back = _reply_arrives(p, h, send, "好的")
    _feed(h, back, _sheet())
    pay = p.advance(h)
    assert pay.tool_calls[1].arguments["name"] == "demo/add-cart"
    _feed(h, pay, HOME)  # the pay move's verify page

    _finish(p, h, p.advance(h))

    (row,) = walklog.load()
    assert row["outcome"] == "completed"
    assert row["total"] == 45.0
    assert row["values"] == {"keyword": "milk"}


def test_payment_fire_writes_the_doctrine_purchase_log_line() -> None:
    # The conductor is the one doing the purchasing, so it writes the
    # daily-log line itself, the moment the payment move's result lands.
    from physiclaw.common import daylog

    p, h, send = _at_gate()
    back = _reply_arrives(p, h, send, "好的")
    _feed(h, back, _sheet())
    pay = p.advance(h)
    _feed(h, pay, HOME)
    p.advance(h)  # the landing that judges the pay move — and logs

    entries = daylog.load_recent_entries(5)

    assert "conductor: demo: payment ¥45 fired (playbook demo/pay)" in entries


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


def test_session_setup_assembles_the_activation_context() -> None:
    # The agent's OWN memory convention feeds parse_task: the recent
    # daily-log window — the same record the model reads at wake, never
    # a conductor-private store (runs.jsonl stays telemetry).
    from physiclaw.common import daylog

    write_channel(CHANNEL_OPEN)
    daylog.append_log("[11:02] demo: bought milk ¥45 — reported to the user")
    write_pack(playbooks={"flow": FLOW})

    boot, _ = setup.session_setup()

    assert boot is not None and boot.activation is not None
    ctx = boot.activation.context
    assert "Recent daily-log entries" in ctx
    assert "bought milk ¥45" in ctx


def test_abandon_records_a_mid_flight_walk_and_breadcrumbs_it() -> None:
    # The killed-session path: the plugin's teardown abandons a walk cut
    # short — one telemetry row plus the daily-log breadcrumb, since
    # this walk had acted.
    from physiclaw.common import daylog
    from physiclaw.conductor.walk import walklog

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
    from physiclaw.conductor.walk import walklog

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


def test_failed_agent_call_records_handover_with_micro_count() -> None:
    from physiclaw.conductor.walk import walklog
    from physiclaw.conductor.walk.micro import DecisionRequest

    write_pack(playbooks={"flow": AGENT_FLOW})
    p = _program(keyword="milk")
    h = _history()
    _feed(h, p.advance(h), HOME)
    step = p.advance(h)
    assert isinstance(step, DecisionRequest)

    _finish(p, h, p.resolve(None))  # the brokered call failed → handover

    (row,) = walklog.load()
    assert row["outcome"] == "handover"
    assert row["node"] == "parse"
    assert row["micros"] == 1


# ---------- inline macros (a move's embedded body) ----------


# FLOW's first move with the body embedded in place of `macro: open-app`.
INLINE_FLOW = FLOW.replace(
    "    macro: open-app\n",
    "    macro:\n"
    "      inputs:\n"
    "        message: {description: the text}\n"
    "      steps:\n"
    "        - home_screen\n",
)


def test_inline_move_dispatches_under_its_synthesized_name() -> None:
    # The whole wiring in one walk: parse synthesizes `flow.open`,
    # build_program merges it into the dispatch registry, and the
    # run_macro turn is name-keyed like any pack macro — refs filled
    # through the node's `with:` exactly as on the directory path.
    write_pack(playbooks={"flow": INLINE_FLOW})
    p = _program(keyword="milk")
    h = _history()
    _feed(h, p.advance(h), HOME)  # the start page

    move = p.advance(h)

    assert move is not None and move.tool_calls[1].arguments == {
        "name": "demo/flow.open",
        "inputs": {"message": "milk"},
    }
    assert "demo/flow.open" in p.pack_macros


def test_session_setup_hidden_registry_carries_inline_macros() -> None:
    # The registry is handed to the engine ONCE at wake — an activation
    # mid-session dispatches inline moves out of `hidden`, so they must
    # ride it beside the directory macros.
    write_channel(CHANNEL_OPEN)
    write_pack(playbooks={"flow": INLINE_FLOW})

    boot, hidden = setup.session_setup()

    assert boot is not None
    assert "demo/flow.open" in hidden


def test_suspended_walk_with_a_broken_spec_is_dropped() -> None:
    # load_suspended is fail-open: a spec that no longer parses drops the
    # suspension instead of taking the wake down.
    _suspended_ask()
    write_pack(playbooks={"pay": "description: broken\nroute: []"})

    assert setup.load_suspended() is None
    with pytest.raises(PlaybookError):
        build.load_spec("demo", "pay")
