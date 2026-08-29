"""Tests for `physiclaw.conductor.playbook` — the PLAYBOOK.yml
grammar, its lints, and pack loading. Every rejection must name the
exact field and rule; the graph/money lints are the safety substance."""

from __future__ import annotations

import pytest
from conductor_fakes import LEDGERED, write_pack

from physiclaw.common import paths
from physiclaw.conductor import playbook as pb
from physiclaw.conductor.playbook import PlaybookError

VALID = """\
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
    with: {message: "{inputs.keyword}"}
    verify: pages.home
  - id: choose
    type: DECIDE
    call: choose_item
    with: {criteria: "cheapest {inputs.keyword}"}
    context: [memory.shopping_prefs, inputs.keyword]
    routes: {pick: to-cart, scroll: choose, none_fit: escalate, escalate: escalate}
  - id: to-cart
    type: LEG
    macro: add-cart
    with: {message: "{choose.pick}"}
    enter: pages.results
    verify: pages.results
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
    assert choose.outcomes == ("pick", "scroll", "none_fit", "escalate")
    assert choose.routes["scroll"] == "choose"  # bounded self-loop
    assert p.nodes[2].irreversible is None
    assert p.nodes[4].gate == "payment" and p.nodes[4].compose == "payment-request"


def test_scan_playbooks_reads_pack_files() -> None:
    write_pack(playbooks={"buy": VALID, "broken": "description: d\nnodes: []"})

    entries = {e.name: e for e in pb.scan_playbooks("demo")}

    assert entries["buy"].spec is not None
    assert entries["broken"].spec is None and entries["broken"].error


# ---------- rejection lints ----------


def _mutate(old: str, new: str) -> str:
    # count==1 pins each entry's blast radius: replace() substitutes every
    # occurrence, so a repeated fragment would mutate more than intended.
    assert VALID.count(old) == 1, old
    return VALID.replace(old, new)


@pytest.mark.parametrize(
    "text, fragment",
    [
        # inner `name:` is gone — the map key IS the name
        (
            _mutate(
                "description: test playbook", "name: buy\ndescription: test playbook"
            ),
            "unknown key",
        ),
        # unknown top key
        (VALID + "bogus: 1\n", "unknown key"),
        # duplicate node id
        (_mutate("id: confirm", "id: open"), "duplicate node id"),
        # unknown node type
        (_mutate("type: CONFIRM", "type: NOPE"), "`type` must be one of"),
        # unroutable target
        (_mutate("routes: {pick: to-cart,", "routes: {pick: nowhere,"), "unknown node"),
        # non-total routes:
        (_mutate("none_fit: escalate, ", ""), "route EVERY out"),
        # unknown macro
        (_mutate("macro: add-cart", "macro: ghost"), "not found in this pack"),
        # unknown page
        (_mutate("verify: pages.home", "verify: pages.mars"), "not declared"),
        # foreign app page
        (_mutate("verify: pages.home", "verify: jd.home"), "reserved namespace"),
        # undeclared placeholder
        (
            _mutate(
                'with: {message: "{inputs.keyword}"}',
                'with: {message: "{inputs.typo}"}',
            ),
            "not declared under `inputs`",
        ),
        # dotted ref to a non-earlier node
        (
            _mutate(
                'with: {message: "{inputs.keyword}"}',
                'with: {message: "{choose.pick}"}',
            ),
            "EARLIER decide node",
        ),
        # dotted ref to unknown payload field
        (_mutate("{choose.pick}", "{choose.nope}"), "no output"),
        # LEG with: key not a macro input
        (
            _mutate(
                'with: {message: "{inputs.keyword}"}',
                'with: {bogus: "x", message: "m"}',
            ),
            "inputs of macro",
        ),
        # choose_item may not author outcomes
        (
            _mutate(
                "call: choose_item", "call: choose_item\n    outcomes: [a, escalate]"
            ),
            "declares its outcomes itself",
        ),
        # unknown irreversible class
        (
            _mutate(
                "enter: pages.results\n    verify: pages.results",
                "enter: pages.results\n    verify: pages.results\n    irreversible: nuclear",
            ),
            "`irreversible` must be one of",
        ),
        # stray brace
        (_mutate('"cheapest {inputs.keyword}"', '"cheapest {Keyword}"'), "stray"),
        # bare ref: `{inputs.name}` is the ONE written form, like page refs
        (
            _mutate(
                'with: {message: "{inputs.keyword}"}', 'with: {message: "{keyword}"}'
            ),
            "every ref is dotted",
        ),
        # a node named `inputs` would shadow the input ref root
        (_mutate("id: open", "id: inputs"), "ref root"),
        # reserved routing target as a node id
        (_mutate("id: confirm", "id: escalate"), "reserved routing target"),
        # context inputs.* typo (with: refs are strict; context must be too)
        (
            _mutate(
                "context: [memory.shopping_prefs, inputs.keyword]",
                "context: [memory.shopping_prefs, inputs.keywrod]",
            ),
            "not declared",
        ),
        # a self-route on anything but the call's re-ask arm can never
        # converge (same screen, same prompt) — parse-time rejection
        (_mutate("none_fit: escalate", "none_fit: choose"), "re-ask arm"),
    ],
)
def test_rejections_name_the_rule(text: str, fragment: str) -> None:
    pack = _pack()

    with pytest.raises(PlaybookError, match=fragment):
        pb.parse_playbook(text, "buy", pack)


def test_decide_call_requires_authored_outcomes_with_escalate() -> None:
    pack = _pack()
    text = _mutate("call: choose_item", "call: decide").replace(
        'with: {criteria: "cheapest {inputs.keyword}"}', 'with: {question: "which?"}'
    )

    with pytest.raises(PlaybookError, match="escalate"):
        # decide with no outcomes at all
        pb.parse_playbook(text, "buy", pack)


def test_money_requires_gate_on_every_path() -> None:
    pack = _pack()
    # Make to-cart a payment node: pay gate sits AFTER it → unguarded.
    text = _mutate(
        "enter: pages.results\n    verify: pages.results",
        "enter: pages.results\n    verify: pages.results\n    irreversible: payment",
    )

    with pytest.raises(PlaybookError, match="HUMAN_GATE"):
        pb.parse_playbook(text, "buy", pack)


def test_money_requires_mandate() -> None:
    pack = _pack()
    text = _mutate(
        "enter: pages.results\n    verify: pages.results",
        "enter: pages.results\n    verify: pages.results\n    irreversible: payment",
    )
    text = text.replace("mandate:\n  max_amount: 100\n", "")

    with pytest.raises(PlaybookError, match="mandate"):
        pb.parse_playbook(text, "buy", pack)


def test_any_irreversible_class_requires_its_gate() -> None:
    # send_message was previously declared but unenforced — a cancel
    # replied to a CONFIRM is never read, so nothing consequential may
    # hide behind one: every irreversible class runs as a gate's
    # fall-through.
    pack = _pack()
    text = _mutate(
        "enter: pages.results\n    verify: pages.results",
        "enter: pages.results\n    verify: pages.results\n    irreversible: send_message",
    )

    with pytest.raises(PlaybookError, match="DIRECTLY follow a HUMAN_GATE"):
        pb.parse_playbook(text, "buy", pack)


def test_send_leg_behind_any_gate_parses() -> None:
    pack = _pack()
    text = (
        VALID
        + """  - id: notify
    type: LEG
    macro: add-cart
    with: {message: "sent"}
    enter: pages.results
    verify: pages.results
    irreversible: send_message
"""
    )

    p = pb.parse_playbook(text, "buy", pack)

    assert p.nodes[-1].irreversible == "send_message"


def test_scaffolded_ios_pack_parses_clean() -> None:
    # The OS-state pack: declarations only — no playbooks, no macros.
    from physiclaw.conductor import pages, scaffold

    root = scaffold.init_pack(pages.IOS_APP)

    assert set(pb.load_pack(pages.IOS_APP).pages) == {"locked"}
    assert pb.scan_playbooks(pages.IOS_APP) == []
    assert not (root / pb.PACK_MACROS_DIRNAME).exists()


def test_scaffolded_pack_parses_clean() -> None:
    from physiclaw.common.text import write_text
    from physiclaw.conductor import scaffold

    root = paths.playbooks_dir() / "newapp"
    (root / pb.PACK_MACROS_DIRNAME / scaffold.EXAMPLE_MACRO).mkdir(parents=True)
    write_text(root / "PLAYBOOK.yml", scaffold.render_pack_stub("newapp"))
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
    text = _mutate("routes: {pick: to-cart,", "routes: {pick: confirm,")

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
    verify: pages.results
  - id: do-pay
    type: LEG
    macro: add-cart
    with: {message: "pay"}
    enter: pages.results
    verify: pages.results
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
    assert loop.call == "next_item" and loop.routes == {"next": "search", "done": "fix"}
    assert type(p.nodes[5]).__name__ == "ReconcileNode" and p.nodes[5].page == "results"
    assert p.nodes[7].revise == "advance"


@pytest.mark.parametrize(
    "old, new, fragment",
    [
        ("kind: list", "kind: tuple", "`kind` must be one of"),
        # The loop closer must route its `next` arm backward.
        (
            "routes: {next: search, done: fix}",
            "routes: {next: fix, done: sheet}",
            "must route BACKWARD",
        ),
        # `revise` re-enters THE loop, nothing else.
        ("revise: advance", "revise: choose", "must target the next_item"),
        # The ledger JSON is not a template value.
        (
            'with: {message: "{item.query}"}',
            'with: {message: "{inputs.items}"}',
            "items",
        ),
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
            "    routes: {next: search, done: fix}\n",
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
    text = VALID + "  - id: fix\n    type: RECONCILE\n    page: pages.results\n"

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
    enter: pages.results
    verify: pages.results
    irreversible: payment
"""
    )

    p = pb.parse_playbook(text, "buy", pack)

    assert p.nodes[-1].irreversible == "payment"


def test_leg_missing_required_macro_input_rejected() -> None:
    pack = _pack()
    text = _mutate(
        'with: {message: "{inputs.keyword}"}\n    verify: pages.home',
        "verify: pages.home",
    )

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
    from conductor_fakes import compose_pack_doc

    root = paths.playbooks_dir() / "demo"
    root.mkdir(parents=True)
    (root / "PLAYBOOK.yml").write_text(
        compose_pack_doc("demo", "Bad Name:\n  anchors: ['x']"), encoding="utf-8"
    )

    with pytest.raises(PlaybookError, match="`pages`"):
        pb.load_pack("demo")


def test_list_apps_finds_packs() -> None:
    write_pack("demo")

    assert pb.list_apps() == ["demo"]


# ---------- mandate ----------


def test_mandate_accepts_input_ref() -> None:
    pack = _pack()
    text = VALID.replace("max_amount: 100", 'max_amount: "{inputs.keyword}"')

    p = pb.parse_playbook(text, "buy", pack)

    assert p.mandate is not None
    assert p.mandate.max_amount == pb.InputRef(name="keyword")


@pytest.mark.parametrize(
    "amount, fragment",
    [
        ("max_amount: -5", "positive"),
        ("max_amount: true", "number or exactly one"),
        ('max_amount: "up to {inputs.keyword} yuan"', "exactly one"),
        ('max_amount: "{inputs.typo}"', "not declared"),
    ],
)
def test_mandate_rejections(amount: str, fragment: str) -> None:
    pack = _pack()

    with pytest.raises(PlaybookError, match=fragment):
        pb.parse_playbook(VALID.replace("max_amount: 100", amount), "buy", pack)


# ---------- money lints (bug-hunt regressions) ----------


def test_non_payment_gate_does_not_satisfy_the_money_dfs() -> None:
    # An address/handoff gate must NOT open the gate-passed flag: a
    # DECIDE arm skipping the PAYMENT gate has to be rejected even when
    # another gate class sits upstream.
    pack = _pack()
    text = """\
description: bypass probe
inputs:
  keyword:
    description: what
mandate:
  max_amount: 100
nodes:
  - id: open
    type: LEG
    macro: open-app
    with: {message: "{inputs.keyword}"}
    verify: pages.home
  - id: addr
    type: HUMAN_GATE
    gate: address
    compose: addr-check
    message: "address ok? reply ok or no"
  - id: route
    type: DECIDE
    call: decide
    with: {question: "ready?"}
    outcomes: [go, ask, escalate]
    routes: {go: pay, ask: paygate, escalate: escalate}
  - id: paygate
    type: HUMAN_GATE
    gate: payment
    compose: pay-confirm
    message: "Total ¥{gate.total}, reply ok or no"
    over_message: "Total ¥{gate.total} over ¥{gate.cap}, reply ok or no"
    return: open-app
  - id: pay
    type: LEG
    macro: add-cart
    with: {message: "pay"}
    enter: pages.results
    verify: pages.home
    irreversible: payment
"""
    with pytest.raises(PlaybookError, match="irreversible leg is entered ONLY"):
        pb.parse_playbook(text, "buy", pack)


def test_routed_in_edge_to_payment_leg_rejected() -> None:
    # Even after a real payment gate, a second payment leg reachable via
    # a DECIDE arm would fire under the FIRST gate's consent.
    pack = _pack()
    text = (
        VALID
        + """  - id: do-pay
    type: LEG
    macro: add-cart
    with: {message: "pay"}
    enter: pages.results
    verify: pages.results
    irreversible: payment
"""
    )
    text = text.replace("none_fit: escalate,", "none_fit: do-pay,")

    with pytest.raises(PlaybookError, match="entered ONLY"):
        pb.parse_playbook(text, "buy", pack)


def test_payment_gate_requires_reentry_into_the_app() -> None:
    # After the ask the phone shows the IM thread; without `return:` on
    # the gate or `enter:` on the leg, money would fire blind.
    pack = _pack()
    text = (
        VALID
        + """  - id: do-pay
    type: LEG
    macro: add-cart
    with: {message: "pay"}
    enter: pages.results
    verify: pages.results
    irreversible: payment
"""
    )
    text = text.replace(
        "    enter: pages.results\n    verify: pages.results\n    irreversible: payment",
        "    verify: pages.results\n    irreversible: payment",
    )

    with pytest.raises(PlaybookError, match="declare `return:`"):
        pb.parse_playbook(text, "buy", pack)


def test_revise_requires_return() -> None:
    from conductor_fakes import LEDGERED

    pack = _pack()
    text = LEDGERED.replace("    return: open-app\n", "")

    with pytest.raises(PlaybookError, match="`revise` needs `return:`"):
        pb.parse_playbook(text, "shop", pack)


def test_scan_of_a_walkless_pack_is_empty() -> None:
    # A pack may be infrastructure-only (channel, ios): no `playbooks:`
    # section means no entries — never an error.
    from physiclaw.conductor.playbook import scan_playbooks

    write_pack(playbooks={})

    assert scan_playbooks("demo") == []


def test_pages_dot_ref_is_the_one_written_form() -> None:
    # Every page ref is `<root>.<page>` over a closed root set — the
    # `pages.` form resolves to the declared page, and a bare name is
    # rejected WITH the fix spelled out (one spelling, no drift).
    pack = _pack()

    spec = pb.parse_playbook(VALID, "buy", pack)
    assert spec.nodes[0].verify == "home"  # pages.home → the declared page

    with pytest.raises(PlaybookError, match="write pages.home"):
        pb.parse_playbook(
            VALID.replace("verify: pages.home", "verify: home"), "buy", pack
        )
