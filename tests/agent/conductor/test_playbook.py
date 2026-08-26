"""Tests for `physiclaw.agent.conductor.playbook` — the PLAYBOOK.yml
grammar, its lints, and pack loading. Every rejection must name the
exact field and rule; the graph/money lints are the safety substance."""

from __future__ import annotations

import pytest
from conductor_fakes import LEDGERED, write_pack

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
    message: "Added to the cart, ordering soon"
  - id: pay
    type: HUMAN_GATE
    gate: payment
    compose: payment-request
    message: "Total ¥{gate.total}, reply ok to pay, or no to cancel"
    over_message: "Total ¥{gate.total}, over budget ¥{gate.cap}, reply ok to pay, or no to cancel"
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


def test_payment_leg_must_directly_follow_its_gate() -> None:
    # Reachability is not enough: the conductor reads the sheet AT the
    # gate and fires the leg as its fall-through — a node in between
    # desynchronizes consent from the sheet, so the parser rejects it.
    pack = _pack()
    text = (
        VALID
        + """  - id: detour
    type: LEG
    macro: add-cart
    with: {message: "x"}
    verify: results
  - id: do-pay
    type: LEG
    macro: add-cart
    with: {message: "pay"}
    verify: results
    irreversible: payment
"""
    )

    with pytest.raises(PlaybookError, match="DIRECTLY follow"):
        pb.parse_playbook(text, "buy", pack)


@pytest.mark.parametrize(
    "old, new, fragment",
    [
        # Asks are REQUIRED — the conductor composes no user-facing
        # prose (only the author knows the user's language).
        ('    message: "Added to the cart, ordering soon"\n', "", "is required"),
        # A payment gate's ask must quote the sheet total…
        (
            '    message: "Total ¥{gate.total}, reply ok to pay, or no to cancel"\n',
            '    message: "reply ok to pay, or no to cancel"\n',
            "must quote the sheet",
        ),
        # …and carry the over-cap variant…
        (
            '    over_message: "Total ¥{gate.total}, over budget ¥{gate.cap}, '
            'reply ok to pay, or no to cancel"\n',
            "",
            "needs `over_message`",
        ),
        # …which must disclose the breached budget too.
        (", over budget ¥{gate.cap}", "", "total AND the budget"),
        # over_message is meaningless outside a payment gate.
        (
            "gate: payment\n    compose: payment-request\n"
            '    message: "Total ¥{gate.total}, reply ok to pay, or no to cancel"',
            "gate: handoff\n    compose: payment-request\n"
            '    message: "reply ok to continue, or no to cancel"',
            "only for `gate: payment`",
        ),
    ],
)
def test_ask_template_lints(old, new, fragment) -> None:
    pack = _pack()

    with pytest.raises(PlaybookError, match=fragment):
        pb.parse_playbook(_mutate(old, new), "buy", pack)


def test_ledger_playbook_parses_with_the_sanctioned_backward_edge() -> None:
    p = pb.parse_playbook(LEDGERED, "shop", _pack())

    assert p.inputs[0].kind == "list"
    loop = p.nodes[4]
    assert loop.call == "next_item" and loop.on == {"next": "search", "done": "fix"}
    assert type(p.nodes[5]).__name__ == "ReconcileNode" and p.nodes[5].page == "results"
    assert p.nodes[7].revise == "advance"


@pytest.mark.parametrize(
    "old, new, fragment",
    [
        ("kind: list", "kind: tuple", "`kind` must be one of"),
        # The loop closer must route its `next` arm backward.
        (
            "on: {next: search, done: fix}",
            "on: {next: fix, done: sheet}",
            "must route BACKWARD",
        ),
        # `revise` re-enters THE loop, nothing else.
        ("revise: advance", "revise: choose", "must target the next_item"),
        # The ledger JSON is not a template value.
        ('with: {message: "{item.query}"}', 'with: {message: "{items}"}', "items"),
        # next_item's pick wiring is required.
        (
            '    with: {picked: "{choose.pick}"}\n',
            "",
            "requires `with.picked`",
        ),
        # A dropped loop strands the list input.
        (
            "  - id: advance\n"
            "    type: DECIDE\n"
            "    call: next_item\n"
            '    with: {picked: "{choose.pick}"}\n'
            "    on: {next: search, done: fix}\n",
            "",
            "no next_item",
        ),
    ],
)
def test_ledger_lints(old, new, fragment) -> None:
    pack = _pack()
    assert LEDGERED.count(old) == 1, old

    with pytest.raises(PlaybookError, match=fragment):
        pb.parse_playbook(LEDGERED.replace(old, new), "shop", pack)


def test_two_list_inputs_rejected() -> None:
    pack = _pack()
    text = LEDGERED.replace(
        "mandate:",
        "  extra:\n    description: another list\n    kind: list\nmandate:",
    )

    with pytest.raises(PlaybookError, match="at most one `kind: list`"):
        pb.parse_playbook(text, "shop", pack)


def test_reconcile_without_the_loop_rejected() -> None:
    pack = _pack()
    text = VALID + "  - id: fix\n    type: RECONCILE\n    page: results\n"

    with pytest.raises(PlaybookError, match="RECONCILE needs the next_item"):
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
