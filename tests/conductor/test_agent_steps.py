"""Tests for the agent-step grammar and its walk — `agent` moves (pure
text and acting episodes), the `start` move, per-page `recover:` hands,
`landmarks`, and the anchors clause forms."""

from __future__ import annotations

import pytest
from conductor_fakes import (
    ELSEWHERE,
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

from physiclaw.conductor.drive import build
from physiclaw.conductor.spec import pack as pb
from physiclaw.conductor.spec import pages
from physiclaw.conductor.spec.calls import ACT_SCROLL_DOWN, AGENT_DONE
from physiclaw.conductor.spec.model import AgentNode, DoNode, PlaybookError
from physiclaw.conductor.walk.micro import (
    ACT_ARM,
    AGENT_ACT,
    AGENT_FIELDS,
    DecisionRequest,
    MicroOutcome,
)

BACK_LANDMARK = """\
back:
  label: "back"
  bbox: [0.0, 0.0, 0.1, 0.1]
"""

# The new-grammar walk: a pure-text agent derives a value before the
# phone is touched, `start` cold-launches unconditionally, pages declare
# their own recovery, and an acting episode carries the judgment stretch.
AGENTED = """\
description: agent-driven flow
inputs:
  user_said:
    description: verbatim ask
route:
  - agent: parse
    prompt: |
      Derive the keyword.
      Said: "{inputs.user_said}"
    returns:
      keyword: the search keyword
  - start: app
    macro:
      steps:
        - name: launch
          tool: home_screen
  - page: home
    recover:
      tool: force_quit
  - do: search
    macro: open-app
    with: {message: "{parse.keyword}"}
  - page: results
    recover:
      tool: tap
      with: landmarks.back
  - agent: pick
    prompt: |
      Add the right item to the cart, then finish on the done page.
    tools: [tap, scroll]
    give: [landmarks.back]
    returns:
      total: the audited total
    limit: {calls: 5, scrolls: 1}
  - page: done
"""

HOME = make_screen(("Files", 0.5, 0.1)).text
RESULTS = make_screen(("综合", 0.5, 0.1), ("Milk 5kg", 0.5, 0.4)).text
DONE = make_screen(("AllDone", 0.5, 0.1)).text


def _write(playbook: str = AGENTED, name: str = "walk"):
    write_pack(playbooks={name: playbook}, landmarks=BACK_LANDMARK)


def _done_outcome(**payload) -> MicroOutcome:
    return MicroOutcome(out=AGENT_DONE, reason="ok", confidence=0.9, payload=payload)


# ---------- parsing ----------


def test_parse_the_agented_playbook() -> None:
    _write()
    spec, _ = build.load_spec("demo", "walk", require_live=False)

    kinds = [type(n).__name__ for n in spec.nodes]
    assert kinds == ["AgentNode", "DoNode", "DoNode", "AgentNode"]
    parse, start, search, pick = spec.nodes
    assert isinstance(parse, AgentNode) and parse.tools == ()
    assert parse.return_fields == ("keyword",)
    assert isinstance(start, DoNode) and start.enter == "" and start.verify == "home"
    assert isinstance(pick, AgentNode)
    assert pick.enter == "results" and pick.verify == "done"
    assert pick.give == ("back",) and pick.max_calls == 5 and pick.max_scrolls == 1
    assert spec.recovers["home"].elsewhere.tool == "force_quit"
    assert spec.recovers["home"].occluded is spec.recovers["home"].elsewhere
    assert spec.recovers["results"].elsewhere.landmark == "back"


def _parse(text: str):
    _write(text)
    return build.load_spec("demo", "walk", require_live=False)[0]


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        # An agent with neither hands nor fields can do nothing.
        (("    returns:\n      keyword: the search keyword\n", ""), "can do nothing"),
        # An acting agent must be framed by pages.
        (("  - page: done\n", ""), "followed by the page"),
        # start must sit immediately before the first page.
        (
            (
                "  - start: app\n",
                "  - page: home\n  - start: app\n",
            ),
            "immediately before the first page",
        ),
        # A screen-touching move cannot precede the first page.
        (
            (
                "  - agent: parse\n",
                "  - do: open-app\n  - agent: parse\n",
            ),
            "precede the first page",
        ),
    ],
)
def test_route_shape_lints(mutate, fragment) -> None:
    old, new = mutate
    text = AGENTED.replace(old, new)
    with pytest.raises(PlaybookError, match=fragment):
        _parse(text)


def test_give_may_grant_a_pack_macro() -> None:
    spec = _parse(
        AGENTED.replace(
            "give: [landmarks.back]", "give: [landmarks.back, macros.add-cart]"
        )
    )
    pick = spec.nodes[3]
    assert isinstance(pick, AgentNode)
    assert pick.give == ("back",) and pick.macros == ("add-cart",)
    assert pb.disabled_macros(spec, pb.load_pack("demo")) == []


@pytest.mark.parametrize(
    "grant, fragment",
    [
        ("macros.nope", "not found in this pack"),
        ("macros.done", "fixed episode answer"),
        ("gestures.back", "must look like"),
    ],
)
def test_give_grants_are_checked(grant, fragment) -> None:
    write_pack(
        playbooks={
            "walk": AGENTED.replace("give: [landmarks.back]", f"give: [{grant}]")
        },
        landmarks=BACK_LANDMARK,
        macros=("open-app", "add-cart", "done"),
    )
    with pytest.raises(PlaybookError, match=fragment):
        build.load_spec("demo", "walk", require_live=False)


def test_give_is_optional() -> None:
    _parse(AGENTED.replace("    give: [landmarks.back]\n", ""))


def test_agent_give_must_name_a_declared_landmark() -> None:
    text = AGENTED.replace("give: [landmarks.back]", "give: [landmarks.cart]")
    with pytest.raises(PlaybookError, match="not declared under\n?.*`landmarks`"):
        _parse(text)


def test_recover_tap_requires_a_landmark_target() -> None:
    text = AGENTED.replace(
        "      tool: tap\n      with: landmarks.back\n", "      tool: tap\n"
    )
    with pytest.raises(PlaybookError, match="landmarks.<name>"):
        _parse(text)


def test_recover_rejects_an_unknown_tool() -> None:
    text = AGENTED.replace("      tool: force_quit\n", "      tool: dance\n")
    with pytest.raises(PlaybookError, match="recover.*tool"):
        _parse(text)


def test_agent_prompt_refs_are_validated() -> None:
    text = AGENTED.replace("{inputs.user_said}", "{inputs.nope}")
    with pytest.raises(PlaybookError, match="not declared under `inputs`"):
        _parse(text)


def test_landmarks_section_is_the_open_spelling() -> None:
    root = write_pack(playbooks={"walk": AGENTED})
    doc = (root / "PLAYBOOK.yml").read_text(encoding="utf-8")
    (root / "PLAYBOOK.yml").write_text(
        doc + 'landmarks:\n  back:\n    label: "back"\n    bbox: [0.0, 0.0, 0.1, 0.1]\n'
        '  cart-tab:\n    label: "cart"\n    bbox: [0.6, 0.9, 0.8, 1.0]\n',
        encoding="utf-8",
    )
    pack = pb.load_pack("demo")
    assert sorted(pack.landmarks) == ["back", "cart-tab"]


def test_controls_is_not_a_pack_section() -> None:
    # The fixed-spot section has ONE spelling; the earlier `controls:`
    # is an unknown key, refused at the pack door.
    root = write_pack(playbooks={"walk": AGENTED}, landmarks=BACK_LANDMARK)
    doc = (root / "PLAYBOOK.yml").read_text(encoding="utf-8")
    (root / "PLAYBOOK.yml").write_text(
        doc.replace("landmarks:", "controls:"), encoding="utf-8"
    )

    with pytest.raises(PlaybookError, match="controls"):
        pb.load_pack("demo")


def test_anchor_clause_forms_parse() -> None:
    decls = pages.parse_pages(
        """\
home:
  anchors: {or: ["推荐", "关注"], region: top}
results:
  anchors: {and: ["综合", {or: ["销量", "销售"], region: top}]}
paid:
  anchors: "支付成功"
""",
        "demo",
    )
    home = decls["home"].anchors
    assert len(home) == 1 and home[0].readings == ("推荐", "关注")
    results = decls["results"].anchors
    assert len(results) == 2 and results[1].readings == ("销量", "销售")
    assert decls["paid"].anchors[0].text == "支付成功"


def test_anchor_clause_rejects_nested_and() -> None:
    with pytest.raises(pages.PagesError, match="does not nest"):
        pages.parse_pages('p:\n  anchors: {and: [{and: ["a"]}]}', "demo")


# ---------- the walk: pure-text agent + start ----------


def _boot():
    """Walk AGENTED to the parse agent's request (the opening peek lands
    on an unknown screen — the text agent needs none)."""
    _write()
    p = _program(name="walk", user_said="买牛奶")
    h = _history()
    _feed(h, p.advance(h), ELSEWHERE)
    req = p.advance(h)
    assert isinstance(req, DecisionRequest) and req.call == AGENT_FIELDS
    assert "买牛奶" in req.args["prompt"]
    return p, h, req


def test_text_agent_fills_outputs_then_start_runs_unconditionally() -> None:
    p, h, req = _boot()
    assert "keyword" in req.args["fields"]

    start = p.resolve(_done_outcome(keyword="milk"))
    assert start is not None and start.tool_names() == ["note", "run_macro"]
    # The start leg has no enter: it runs from the unknown screen.
    assert start.tool_calls[1].arguments["name"] == "demo/walk.app"

    _feed(h, start, HOME)  # verified by the page that follows it
    search = p.advance(h)
    assert search is not None
    assert search.tool_calls[1].arguments == {
        "name": "demo/open-app",
        "inputs": {"message": "milk"},  # {parse.keyword} resolved
    }


def test_declared_context_rides_the_brief_and_nothing_else_does() -> None:
    from physiclaw.common import daylog, paths
    from physiclaw.common.text import write_text

    f = paths.memory_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    write_text(f, "## shopping\nprefers oat milk\n\n## other\nsecret\n")
    daylog.append_log("[11:02] demo: bought milk ¥45")
    _write(
        AGENTED.replace(
            "      keyword: the search keyword\n",
            "      keyword: the search keyword\n    context: [memory.shopping]\n",
        )
    )
    p = _program(name="walk", user_said="买牛奶")
    h = _history()
    _feed(h, p.advance(h), ELSEWHERE)

    req = p.advance(h)

    assert isinstance(req, DecisionRequest) and req.call == AGENT_FIELDS
    assert "prefers oat milk" in req.context
    assert "secret" not in req.context and "bought milk" not in req.context


def test_text_agent_escalate_hands_over() -> None:
    p, h, _ = _boot()
    step = p.resolve(MicroOutcome(out="escalate", reason="no product", confidence=0.9))
    summary = _finish(p, h, step)
    assert "escalated" in summary


def test_text_agent_missing_return_field_hands_over() -> None:
    p, h, _ = _boot()
    step = p.resolve(_done_outcome())
    summary = _finish(p, h, step)
    assert "without return field" in summary


# ---------- the walk: the acting episode ----------


def _at_episode():
    """Walk to the pick episode's first request (standing on results)."""
    p, h, _ = _boot()
    start = p.resolve(_done_outcome(keyword="milk"))
    _feed(h, start, HOME)
    search = p.advance(h)
    _feed(h, search, RESULTS)
    req = p.advance(h)
    assert isinstance(req, DecisionRequest) and req.call == AGENT_ACT
    return p, h, req


def test_episode_offers_rows_grants_and_verbs() -> None:
    _, _, req = _at_episode()

    keys = [c.key for c in req.candidates]
    assert "back" in keys  # the granted landmark, by name
    assert "Milk 5kg" in keys  # a live screen row
    assert AGENT_DONE in req.outcomes and ACT_SCROLL_DOWN in req.outcomes
    assert "Granted landmarks" in req.args["block"]


def test_episode_tap_grounds_and_history_is_append_only() -> None:
    p, h, req = _at_episode()
    row = next(c for c in req.candidates if c.key == "Milk 5kg")

    tap = p.resolve(
        MicroOutcome(out=ACT_ARM, reason="fits", confidence=0.9, picked=row)
    )
    assert tap is not None and tap.tool_names() == ["note", "tap"]
    assert tap.tool_calls[1].arguments["bbox"] == list(row.bbox)

    _feed(h, tap, RESULTS)
    req2 = p.advance(h)
    assert isinstance(req2, DecisionRequest)
    # Append-only: the settled turn rides verbatim before the new block.
    assert len(req2.history) == 2
    assert req2.history[0] == ("user", req.args["block"])
    assert '"answer": "Milk 5kg"' in req2.history[1][1]
    assert "[you tapped 'Milk 5kg']" in req2.args["block"]


def test_episode_runs_a_granted_macro_by_name() -> None:
    from physiclaw.conductor.walk.step_agent import KIND_MACRO

    _write(AGENTED.replace("give: [landmarks.back]", "give: [macros.add-cart]"))
    p = _program(name="walk", user_said="买牛奶")
    h = _history()
    _feed(h, p.advance(h), ELSEWHERE)
    assert isinstance(p.advance(h), DecisionRequest)
    _feed(h, p.resolve(_done_outcome(keyword="milk")), HOME)
    _feed(h, p.advance(h), RESULTS)
    req = p.advance(h)
    assert isinstance(req, DecisionRequest)
    assert "Granted macros" in req.args["block"] and req.args["macros"] == "add-cart"
    macro = next(c for c in req.candidates if c.key == "add-cart")
    assert macro.bbox is None

    run = p.resolve(
        MicroOutcome(out=ACT_ARM, reason="add", confidence=0.9, picked=macro)
    )

    assert run is not None and run.tool_names() == ["note", "run_macro"]
    assert run.tool_calls[1].arguments == {"name": "demo/add-cart"}
    assert "ran macro 'add-cart'" in run.tool_calls[0].arguments["summary"]
    _feed(h, run, RESULTS)  # its result view is the next turn's screen
    req2 = p.advance(h)
    assert isinstance(req2, DecisionRequest)
    assert req2.args["block"].startswith("[you ran macro 'add-cart']")
    assert KIND_MACRO in p._step.kinds


@pytest.mark.parametrize("page, offered", [("results", True), ("home", False)])
def test_page_scoped_landmark_is_offered_only_on_its_page(page, offered) -> None:
    scoped = BACK_LANDMARK.rstrip("\n") + f"\n  page: {page}\n"
    write_pack(playbooks={"walk": AGENTED}, landmarks=scoped)
    p = _program(name="walk", user_said="买牛奶")
    h = _history()
    _feed(h, p.advance(h), ELSEWHERE)
    assert isinstance(p.advance(h), DecisionRequest)
    _feed(h, p.resolve(_done_outcome(keyword="milk")), HOME)
    _feed(h, p.advance(h), RESULTS)  # the episode opens on results

    req = p.advance(h)

    assert isinstance(req, DecisionRequest)
    assert any(c.key == "back" for c in req.candidates) is offered
    assert ("Granted landmarks" in req.args["block"]) is offered


def test_episode_done_is_audited_against_the_verify_page() -> None:
    p, h, req = _at_episode()

    # done while still on results → rejected, costs a call, continues.
    retry = p.resolve(_done_outcome(total="45"))
    assert isinstance(retry, DecisionRequest)
    assert "done rejected" in retry.args["block"]

    # Move to the verify page, then done sticks and records the returns.
    row = next(c for c in retry.candidates if c.key == "Milk 5kg")
    tap = p.resolve(MicroOutcome(out=ACT_ARM, reason="go", confidence=0.9, picked=row))
    _feed(h, tap, DONE)
    req3 = p.advance(h)
    assert isinstance(req3, DecisionRequest)
    step = p.resolve(_done_outcome(total="45"))
    summary = _finish(p, h, step)
    assert "completed" in summary


def test_episode_call_limit_hands_over() -> None:
    p, h, req = _at_episode()
    step: object = req
    for _ in range(10):
        if not isinstance(step, DecisionRequest):
            break
        step = p.resolve(_done_outcome(total="45"))  # rejected off-page each time
    summary = _finish(p, h, step)
    assert "call limit" in summary


def test_episode_scroll_limit_hands_over() -> None:
    p, h, _ = _at_episode()

    swipe = p.resolve(MicroOutcome(out=ACT_SCROLL_DOWN, reason="more", confidence=0.9))
    assert swipe is not None and swipe.tool_names() == ["note", "swipe"]
    _feed(h, swipe, RESULTS)
    assert isinstance(p.advance(h), DecisionRequest)

    step = p.resolve(MicroOutcome(out=ACT_SCROLL_DOWN, reason="more", confidence=0.9))
    summary = _finish(p, h, step)
    assert "scroll limit" in summary


# ---------- declared recovery ----------


def test_declared_recover_hand_runs_then_walk_resumes() -> None:
    p, h, _ = _boot()
    start = p.resolve(_done_outcome(keyword="milk"))
    _feed(h, start, HOME)
    search = p.advance(h)
    _feed(h, search, HOME)  # search did NOT land on results

    hand = p.advance(h)  # results' declared hand: tap landmarks.back — no re-peek first
    assert hand is not None and hand.tool_names() == ["note", "tap"]
    _feed(h, hand, RESULTS)  # the hand restored the page

    req = p.advance(h)  # VERIFY satisfied → the episode opens
    assert isinstance(req, DecisionRequest) and req.call == AGENT_ACT


def test_page_without_recover_hands_over_in_declared_mode() -> None:
    # `done` declares no recover; failing to reach it (episode fence
    # aside) → the walk hands over instead of climbing a hidden ladder.
    no_recover = AGENTED.replace(
        "  - page: results\n    recover:\n      tool: tap\n      with: landmarks.back\n",
        "  - page: results\n",
    )
    _write(no_recover)
    p = _program(name="walk", user_said="买牛奶")
    h = _history()
    _feed(h, p.advance(h), ELSEWHERE)
    assert isinstance(p.advance(h), DecisionRequest)
    start = p.resolve(_done_outcome(keyword="milk"))
    _feed(h, start, HOME)
    search = p.advance(h)
    _feed(h, search, HOME)  # wrong page; results has no recover now

    summary = _finish(p, h, p.advance(h))
    assert "declares no recover" in summary


def test_recover_relaunch_loop_is_bounded_by_the_walk_budget() -> None:
    # A hand that runs and restores its page clears its recovery State, so the
    # budget must count the WALK's spend, not the engagement's — else a
    # splash ad on every cold launch loops force_quit forever.
    from physiclaw.conductor.spec.limits import MAX_RECOVER_ACTIONS

    _write(
        AGENTED.replace(
            "    recover:\n      tool: force_quit\n",
            "    recover: {tool: force_quit, limit: 6}\n",
        )
    )
    p = _program(name="walk", user_said="买牛奶")
    h = _history()
    _feed(h, p.advance(h), ELSEWHERE)
    assert isinstance(p.advance(h), DecisionRequest)
    step = p.resolve(_done_outcome(keyword="milk"))
    quits = 0
    for _ in range(60):
        assert step is not None and step.synthesized
        if step.tool_names() == ["note", "force_quit"]:
            quits += 1
        _feed(h, step, ELSEWHERE)  # home never reads
        nxt = p.advance(h)
        if nxt is None:  # the brief landed — the walk is over
            break
        step = nxt
    else:
        pytest.fail("the relaunch loop never terminated")
    assert 1 <= quits <= MAX_RECOVER_ACTIONS
    assert "budget" in step.tool_calls[0].arguments["summary"]


def test_page_recover_limit_stops_the_relaunch_before_the_walk_budget() -> None:
    # The page's own `limit:` (default 2) is spent first; the handover
    # names it, so the author sees which bound fired.
    from physiclaw.conductor.spec.limits import MAX_RECOVER_ACTIONS

    _write()
    p = _program(name="walk", user_said="买牛奶")
    h = _history()
    _feed(h, p.advance(h), ELSEWHERE)
    assert isinstance(p.advance(h), DecisionRequest)
    step = p.resolve(_done_outcome(keyword="milk"))
    quits = 0
    for _ in range(20):
        assert step is not None and step.synthesized
        if step.tool_names() == ["note", "force_quit"]:
            quits += 1
        _feed(h, step, ELSEWHERE)
        nxt = p.advance(h)
        if nxt is None:
            break
        step = nxt
    assert quits == 2 < MAX_RECOVER_ACTIONS
    assert "recover limit (2) spent" in step.tool_calls[0].arguments["summary"]


def test_recover_force_quit_then_walk_restarts_from_the_top() -> None:
    _write()
    p = _program(name="walk", user_said="买牛奶")
    h = _history()
    _feed(h, p.advance(h), ELSEWHERE)
    assert isinstance(p.advance(h), DecisionRequest)
    start = p.resolve(_done_outcome(keyword="milk"))
    _feed(h, start, ELSEWHERE)  # the launch did NOT reach home

    hand = p.advance(h)  # home's declared hand: force_quit
    assert hand is not None and hand.tool_names() == ["note", "force_quit"]
    _feed(h, hand, ELSEWHERE)  # springboard — home still does not read

    relaunch = p.advance(h)  # still off → route top → start runs again
    assert relaunch is not None
    assert relaunch.tool_calls[1].arguments["name"] == "demo/walk.app"


# ---------- the payment episode ----------

AGENT_PAY = """\
description: gated agent pay
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
    total: "合计"
    message: "合计 ¥{ask.total}。回复 好的 确认支付，或 不用 取消。"
    yes: ["好的"]
    no: ["不用"]
    resume:
      macro: open-app
  - agent: pay
    irreversible: payment
    prompt: |
      Pay exactly ¥{ask.total}, then finish on the done page.
    tools: [tap]
    limit: {calls: 3}
  - page: done
"""

SHEET = make_screen(("综合", 0.5, 0.1), ("合计 ¥45", 0.5, 0.5), ("支付", 0.5, 0.8)).text
SHEET_CHANGED = make_screen(("综合", 0.5, 0.1), ("合计 ¥60", 0.5, 0.5)).text


def _at_pay_episode(resume_screen: str = SHEET):
    """Walk AGENT_PAY through the confirmed gate to the pay episode's
    first request."""
    write_channel()
    write_pack(playbooks={"pay": AGENT_PAY})
    p = _program(name="pay", keyword="milk")
    h = _history()
    _feed(h, p.advance(h), HOME)
    _feed(h, p.advance(h), SHEET)  # `open` landed on the sheet (results)
    send = p.advance(h)
    assert send is not None and send.tool_calls[1].arguments["name"] == "channel/send"
    ask = send.tool_calls[1].arguments["inputs"]["message"]
    assert "¥45" in ask  # the sheet total, quoted
    _feed(h, send, _thread((ask, 0.75, 0.3)))
    _feed(h, p.advance(h), "waited")
    peek = p.advance(h)
    _feed(h, peek, _thread((ask, 0.75, 0.3), ("好的", 0.25, 0.5)))
    back = p.advance(h)  # confirmed → resume macro re-enters the app
    assert back is not None and back.tool_calls[1].arguments["name"] == "demo/open-app"
    _feed(h, back, resume_screen)
    return p, h, p.advance(h)


def test_payment_episode_taps_under_consent_then_completes() -> None:
    p, h, req = _at_pay_episode()
    assert isinstance(req, DecisionRequest)
    assert "¥45" in req.args["block"]  # {ask.total} filled into the prompt

    row = next(c for c in req.candidates if c.key == "支付")
    tap = p.resolve(MicroOutcome(out=ACT_ARM, reason="pay", confidence=0.9, picked=row))
    assert tap is not None and tap.tool_names() == ["note", "tap"]

    _feed(h, tap, DONE)
    req2 = p.advance(h)
    assert isinstance(req2, DecisionRequest)
    step = p.resolve(_done_outcome())
    summary = _finish(p, h, step)
    assert "completed" in summary


def test_payment_episode_second_tap_keeps_the_paid_record() -> None:
    # Consent is spent on the first tap; a later tap of the same episode
    # finds nothing to spend and must not erase the amount that fired.
    p, h, req = _at_pay_episode()
    row = next(c for c in req.candidates if c.key == "支付")
    tap = p.resolve(MicroOutcome(out=ACT_ARM, reason="pay", confidence=0.9, picked=row))
    _feed(h, tap, SHEET)  # the sheet still shows ¥45 — a confirm step
    req2 = p.advance(h)
    row2 = next(c for c in req2.candidates if c.key == "支付")
    tap2 = p.resolve(
        MicroOutcome(out=ACT_ARM, reason="ok", confidence=0.9, picked=row2)
    )
    assert tap2 is not None and tap2.tool_names() == ["note", "tap"]
    _feed(h, tap2, DONE)
    assert isinstance(p.advance(h), DecisionRequest)
    summary = _finish(p, h, p.resolve(_done_outcome()))

    assert "completed" in summary and p.paid == 45.0


@pytest.mark.parametrize(
    "old, new, fragment",
    [
        # A granted name can never be spelled like a fixed answer.
        ("give: [landmarks.back]", "give: [landmarks.done]", "fixed episode answer"),
        # A return field cannot reuse the reply contract's own fields.
        ("      keyword: the search keyword\n", "      answer: the pick\n", "contract"),
        # scroll granted with no scroll budget would hand over at once.
        ("limit: {calls: 5, scrolls: 1}", "limit: {calls: 5, scrolls: 0}", "scrolls"),
        # Landmarks are pressed; without tap there is nothing to press with.
        ("    tools: [tap, scroll]\n", "    tools: [scroll]\n", "without `tap`"),
    ],
)
def test_episode_grammar_lints(old, new, fragment) -> None:
    write_pack(
        playbooks={"walk": AGENTED.replace(old, new)},
        landmarks=BACK_LANDMARK
        + 'done:\n  label: "done"\n  bbox: [0.5, 0.5, 0.6, 0.6]\n',
    )
    with pytest.raises(PlaybookError, match=fragment):
        build.load_spec("demo", "walk", require_live=False)


def test_give_refuses_one_name_as_both_landmark_and_macro() -> None:
    write_pack(
        playbooks={
            "walk": AGENTED.replace(
                "give: [landmarks.back]", "give: [landmarks.back, macros.back]"
            )
        },
        landmarks=BACK_LANDMARK,
        macros=("open-app", "add-cart", "back"),
    )
    with pytest.raises(PlaybookError, match="both a landmark and a macro"):
        build.load_spec("demo", "walk", require_live=False)


def test_payment_ask_before_a_screen_move_needs_resume() -> None:
    text = AGENT_PAY.replace("    resume:\n      macro: open-app\n", "")
    write_channel()
    write_pack(playbooks={"pay": text})
    with pytest.raises(PlaybookError, match="declare `resume:`"):
        build.load_spec("demo", "pay", require_live=False)


def test_payment_ask_reads_the_page_before_it() -> None:
    # A reserved built-in cannot be the sheet a payment ask reads.
    text = AGENT_PAY.replace(
        "  - page: results\n  - ask: gate\n",
        "  - page: results\n  - page: ios.locked\n  - ask: gate\n",
    )
    write_channel()
    write_pack(playbooks={"pay": text})
    with pytest.raises(PlaybookError, match="reads its total off the page before"):
        build.load_spec("demo", "pay", require_live=False)


def test_payment_episode_blocks_a_tap_when_the_sheet_changed() -> None:
    p, h, req = _at_pay_episode(resume_screen=SHEET_CHANGED)
    assert isinstance(req, DecisionRequest)

    row = req.candidates[0]
    step = p.resolve(
        MicroOutcome(out=ACT_ARM, reason="pay", confidence=0.9, picked=row)
    )
    summary = _finish(p, h, step)
    assert "sheet changed after consent" in summary
