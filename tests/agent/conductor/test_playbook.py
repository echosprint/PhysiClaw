"""Tests for `physiclaw.agent.conductor.playbook` — the PLAYBOOK.yml
grammar, its lints, and pack loading. Every rejection must name the
exact field and rule; the graph/money lints are the safety substance."""

from __future__ import annotations

import pytest
from conductor_fakes import write_pack

from physiclaw.agent.conductor import playbook as pb
from physiclaw.agent.conductor.playbook import PlaybookError
from physiclaw.common import paths

VALID = """\
name: buy
description: test playbook
enabled: false
inputs:
  keyword:
    description: what to search
mandate:
  max_amount: 100
nodes:
  - id: open
    type: LEG
    macro: open-app
    with: {message: "{keyword}"}
    verify: home
  - id: choose
    type: DECIDE
    call: choose_item
    with: {criteria: "cheapest {keyword}"}
    context: [memory.shopping_prefs, inputs.keyword]
    on: {pick: to-cart, scroll: choose, none_fit: escalate, escalate: escalate}
  - id: to-cart
    type: LEG
    macro: add-cart
    with: {message: "{choose.pick}"}
    enter: results
    verify: results
    irreversible: send_message
  - id: confirm
    type: CONFIRM
    compose: order-summary
  - id: pay
    type: HUMAN_GATE
    gate: payment
    compose: payment-request
"""


def _pack(app: str = "demo"):
    write_pack(app)
    return pb.load_pack(app)


# ---------- happy path ----------


def test_parse_valid_playbook() -> None:
    p = pb.parse_playbook(VALID, "buy", _pack())

    assert p.name == "buy" and p.enabled is False
    assert p.mandate is not None and p.mandate.max_amount == 100.0
    kinds = [type(n).__name__ for n in p.nodes]
    assert kinds == ["LegNode", "DecideNode", "LegNode", "ConfirmNode", "HumanGateNode"]
    choose = p.nodes[1]
    assert choose.outs == ("pick", "scroll", "none_fit", "escalate")
    assert choose.on["scroll"] == "choose"  # bounded self-loop
    # send_message needs no gate; only `payment` does.
    assert p.nodes[2].irreversible == "send_message"
    assert p.nodes[4].gate == "payment" and p.nodes[4].compose == "payment-request"


def test_scan_playbooks_reads_pack_files() -> None:
    root = write_pack()
    (root / "buy.yml").write_text(VALID, encoding="utf-8")
    (root / "broken.yml").write_text("name: broken\nnodes: []", encoding="utf-8")

    entries = {e.name: e for e in pb.scan_playbooks("demo")}

    assert entries["buy"].spec is not None
    assert entries["broken"].spec is None and entries["broken"].error
    assert "pages" not in entries  # pages.yml excluded


# ---------- rejection lints ----------


def _mutate(old: str, new: str) -> str:
    # count==1 pins each entry's blast radius: replace() substitutes every
    # occurrence, so a repeated fragment would mutate more than intended.
    assert VALID.count(old) == 1, old
    return VALID.replace(old, new)


@pytest.mark.parametrize(
    "text, fragment",
    [
        # name/filename mismatch
        (_mutate("name: buy", "name: other"), "filename stem"),
        # unknown top key
        (VALID + "bogus: 1\n", "unknown key"),
        # duplicate node id
        (_mutate("id: confirm", "id: open"), "duplicate node id"),
        # unknown node type
        (_mutate("type: CONFIRM", "type: NOPE"), "`type` must be one of"),
        # unroutable target
        (_mutate("on: {pick: to-cart,", "on: {pick: nowhere,"), "unknown node"),
        # non-total on:
        (_mutate("none_fit: escalate, ", ""), "route EVERY out"),
        # unknown macro
        (_mutate("macro: add-cart", "macro: ghost"), "not found in this pack"),
        # unknown page
        (_mutate("verify: home", "verify: mars"), "not declared"),
        # foreign app page
        (_mutate("verify: home", "verify: jd.home"), "reserved namespaces"),
        # undeclared placeholder
        (
            _mutate('with: {message: "{keyword}"}', 'with: {message: "{typo}"}'),
            "not declared under `inputs`",
        ),
        # dotted ref to a non-earlier node
        (
            _mutate('with: {message: "{keyword}"}', 'with: {message: "{choose.pick}"}'),
            "EARLIER decide node",
        ),
        # dotted ref to unknown payload field
        (_mutate("{choose.pick}", "{choose.nope}"), "no output"),
        # LEG with: key not a macro input
        (
            _mutate('with: {message: "{keyword}"}', 'with: {bogus: "x", message: "m"}'),
            "inputs of macro",
        ),
        # choose_item may not author outs
        (
            _mutate("call: choose_item", "call: choose_item\n    outs: [a, escalate]"),
            "declares its outs itself",
        ),
        # unknown irreversible class
        (
            _mutate("irreversible: send_message", "irreversible: nuclear"),
            "`irreversible` must be one of",
        ),
        # stray brace
        (_mutate('"cheapest {keyword}"', '"cheapest {Keyword}"'), "stray"),
        # reserved routing target as a node id
        (_mutate("id: confirm", "id: escalate"), "reserved routing target"),
        # context inputs.* typo (with: refs are strict; context must be too)
        (_mutate("inputs.keyword", "inputs.keywrod"), "not declared"),
        # a self-route on anything but the call's re-ask arm can never
        # converge (same screen, same prompt) — parse-time rejection
        (_mutate("none_fit: escalate", "none_fit: choose"), "re-ask arm"),
    ],
)
def test_rejections_name_the_rule(text: str, fragment: str) -> None:
    pack = _pack()

    with pytest.raises(PlaybookError, match=fragment):
        pb.parse_playbook(text, "buy", pack)


def test_decide_call_requires_authored_outs_with_escalate() -> None:
    pack = _pack()
    text = _mutate("call: choose_item", "call: decide").replace(
        'with: {criteria: "cheapest {keyword}"}', 'with: {question: "which?"}'
    )

    with pytest.raises(PlaybookError, match="escalate"):
        # decide with no outs at all
        pb.parse_playbook(text, "buy", pack)


def test_money_requires_gate_on_every_path() -> None:
    pack = _pack()
    # Make to-cart a payment node: pay gate sits AFTER it → unguarded.
    text = _mutate("irreversible: send_message", "irreversible: payment")

    with pytest.raises(PlaybookError, match="HUMAN_GATE"):
        pb.parse_playbook(text, "buy", pack)


def test_money_requires_mandate() -> None:
    pack = _pack()
    text = _mutate("irreversible: send_message", "irreversible: payment")
    text = text.replace("mandate:\n  max_amount: 100\n", "")

    with pytest.raises(PlaybookError, match="mandate"):
        pb.parse_playbook(text, "buy", pack)


def test_scaffolded_pack_parses_clean() -> None:
    from physiclaw.agent.conductor import scaffold
    from physiclaw.common.text import write_text

    root = paths.playbooks_dir() / "newapp"
    (root / pb.PACK_MACROS_DIRNAME / scaffold.EXAMPLE_MACRO).mkdir(parents=True)
    write_text(root / "pages.yml", scaffold.render_pages_stub())
    write_text(
        root / f"{scaffold.EXAMPLE_PLAYBOOK}.yml",
        scaffold.render_playbook_stub("newapp"),
    )
    write_text(
        root / pb.PACK_MACROS_DIRNAME / scaffold.EXAMPLE_MACRO / "MACRO.yml",
        scaffold.render_example_macro(),
    )

    (entry,) = pb.scan_playbooks("newapp")

    assert entry.error is None, entry.error
    assert entry.spec is not None and entry.spec.enabled is False


def test_cycle_beyond_self_loop_rejected() -> None:
    pack = _pack()
    # Route none_fit back to the EARLIER open node → a wide cycle.
    text = _mutate("none_fit: escalate", "none_fit: open")

    with pytest.raises(PlaybookError, match="cycle"):
        pb.parse_playbook(text, "buy", pack)


def test_unreachable_node_rejected() -> None:
    pack = _pack()
    # DECIDE routes explicitly (no fall-through), so routing `pick` past
    # to-cart leaves to-cart with no incoming edge at all.
    text = _mutate("on: {pick: to-cart,", "on: {pick: confirm,")

    with pytest.raises(PlaybookError, match="unreachable"):
        pb.parse_playbook(text, "buy", pack)


def test_payment_leg_behind_gate_parses() -> None:
    # The whole point of the gate: a confirmed HUMAN_GATE falls through,
    # so the conductor itself executes the payment leg under the mandate.
    pack = _pack()
    text = (
        VALID
        + """  - id: do-pay
    type: LEG
    macro: add-cart
    with: {message: "pay"}
    verify: results
    irreversible: payment
"""
    )

    p = pb.parse_playbook(text, "buy", pack)

    assert p.nodes[-1].irreversible == "payment"


def test_leg_missing_required_macro_input_rejected() -> None:
    pack = _pack()
    text = _mutate('with: {message: "{keyword}"}\n    verify: home', "verify: home")

    with pytest.raises(PlaybookError, match="requires input"):
        pb.parse_playbook(text, "buy", pack)


# ---------- pack loading ----------


def test_load_pack_carries_macro_errors() -> None:
    root = write_pack()
    bad = root / "macros" / "broken"
    bad.mkdir()
    (bad / "MACRO.yml").write_text("name: mismatch\n", encoding="utf-8")

    pack = pb.load_pack("demo")

    assert set(pack.macros) == {"open-app", "add-cart"}
    assert "broken" in pack.macro_errors


def test_load_pack_bad_pages_raises_playbook_error() -> None:
    root = paths.playbooks_dir() / "demo"
    root.mkdir(parents=True)
    (root / "pages.yml").write_text("Bad Name:\n  anchors: ['x']", encoding="utf-8")

    with pytest.raises(PlaybookError, match="pages.yml"):
        pb.load_pack("demo")


def test_list_apps_finds_packs() -> None:
    write_pack("demo")

    assert pb.list_apps() == ["demo"]


# ---------- mandate ----------


def test_mandate_accepts_input_ref() -> None:
    pack = _pack()
    text = VALID.replace("max_amount: 100", 'max_amount: "{keyword}"')

    p = pb.parse_playbook(text, "buy", pack)

    assert p.mandate is not None
    assert p.mandate.max_amount == pb.InputRef(name="keyword")


@pytest.mark.parametrize(
    "amount, fragment",
    [
        ("max_amount: -5", "positive"),
        ("max_amount: true", "number or exactly one"),
        ('max_amount: "up to {keyword} yuan"', "exactly one"),
        ('max_amount: "{typo}"', "not declared"),
    ],
)
def test_mandate_rejections(amount: str, fragment: str) -> None:
    pack = _pack()

    with pytest.raises(PlaybookError, match=fragment):
        pb.parse_playbook(VALID.replace("max_amount: 100", amount), "buy", pack)
