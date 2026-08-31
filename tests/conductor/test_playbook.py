"""Tests for `physiclaw.conductor.playbook` — the route grammar, its
lints, and pack loading. Every rejection must name the exact field and
rule; the graph/money lints are the safety substance."""

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
budget: 100
route:
  - page: home
  - do: open
    macro: open-app
    with: {message: "{inputs.keyword}"}
  - page: home
  - decide: choose
    uses: choose_item
    with: {criteria: "cheapest {inputs.keyword}"}
    context: [memory.shopping_prefs, inputs.keyword]
    routes: {pick: to-cart, scroll: choose, none_fit: escalate, escalate: escalate}
  - do: to-cart
    macro: add-cart
    with: {message: "{choose.pick}"}
  - page: results
  - tell: confirm
    message: "Added to the cart, ordering soon"
  - ask: pay
    approve: payment
    message: "Total ¥{ask.total}, reply ok to pay, or no to cancel"
    over_budget_message: "Total ¥{ask.total}, over budget ¥{ask.cap}, reply ok to pay, or no to cancel"
"""

# A payment move appended as the ask's fall-through — several money
# tests share it.
PAY_TAIL = """\
  - do: do-pay
    macro: add-cart
    with: {message: "pay"}
    irreversible: payment
  - page: results
"""


def _pack(app: str = "demo"):
    write_pack(app)
    return pb.load_pack(app)


def _required_message_pack(app: str = "demo"):
    """The pack with `open-app.message` made REQUIRED again — the shared
    fixture defaults it (a role macro must carry no required inputs), so
    tests exercising the missing-required lints strip the default here."""
    root = write_pack(app)
    mp = root / "macros" / "open-app" / "MACRO.yml"
    mp.write_text(
        mp.read_text(encoding="utf-8").replace("    default: hi\n", ""),
        encoding="utf-8",
    )
    return pb.load_pack(app)


# ---------- happy path ----------


def test_context_is_parsed_onto_the_playbook() -> None:
    text = VALID.replace(
        "description: test playbook\n",
        "description: test playbook\ncontext: [memory.shopping_prefs]\n",
    )

    p = pb.parse_playbook(text, "buy", _pack())

    assert p.context == ("memory.shopping_prefs",)


@pytest.mark.parametrize(
    "entry",
    ["inputs.keyword", "shopping", "memory.", "memory.Bad-Slug"],
)
def test_context_rejects_non_memory_entries(entry: str) -> None:
    text = VALID.replace(
        "description: test playbook\n",
        f'description: test playbook\ncontext: ["{entry}"]\n',
    )

    with pytest.raises(PlaybookError, match="memory"):
        pb.parse_playbook(text, "buy", _pack())


def test_parse_valid_playbook() -> None:
    p = pb.parse_playbook(VALID, "buy", _pack())

    assert p.name == "buy" and p.enabled is False
    assert p.mandate is not None and p.mandate.max_amount == 100.0
    assert p.start == "home"
    kinds = [type(n).__name__ for n in p.nodes]
    assert kinds == ["LegNode", "DecideNode", "LegNode", "ConfirmNode", "HumanGateNode"]
    choose = p.nodes[1]
    assert choose.outcomes == ("pick", "scroll", "none_fit", "escalate")
    assert choose.routes["scroll"] == "choose"  # bounded self-loop
    assert p.nodes[2].irreversible is None
    assert p.nodes[4].gate == "payment"


def test_derived_checks_come_from_the_waypoints() -> None:
    # A do's enter is the nearest preceding page, its verify the page
    # that follows it — no enter/verify keys exist to author.
    p = pb.parse_playbook(VALID, "buy", _pack())

    assert p.nodes[0].enter == "home" and p.nodes[0].verify == "home"
    assert p.nodes[2].enter == "home" and p.nodes[2].verify == "results"


def test_scan_playbooks_reads_pack_files() -> None:
    write_pack(playbooks={"buy": VALID, "broken": "description: d\nroute: []"})

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
        # duplicate move name
        (_mutate("- tell: confirm", "- tell: open"), "duplicate move name"),
        # unknown entry kind
        (_mutate("  - tell: confirm\n", "  - shout: confirm\n"), "exactly one of"),
        # unroutable target
        (
            _mutate("routes: {pick: to-cart,", "routes: {pick: nowhere,"),
            "neither a move nor a page",
        ),
        # non-total routes:
        (_mutate("none_fit: escalate, ", ""), "route EVERY answer"),
        # unknown macro
        (_mutate("macro: add-cart", "macro: ghost"), "not found in this pack"),
        # unknown page
        (_mutate("  - page: results\n", "  - page: mars\n"), "not declared"),
        # foreign app page
        (_mutate("  - page: results\n", "  - page: jd.home\n"), "reserved namespace"),
        # undeclared placeholder
        (
            _mutate(
                'with: {message: "{inputs.keyword}"}',
                'with: {message: "{inputs.typo}"}',
            ),
            "not declared under `inputs`",
        ),
        # dotted ref to a non-earlier move
        (
            _mutate(
                'with: {message: "{inputs.keyword}"}',
                'with: {message: "{choose.pick}"}',
            ),
            "EARLIER decide",
        ),
        # dotted ref to unknown payload field
        (_mutate("{choose.pick}", "{choose.nope}"), "no output"),
        # do with: key not a macro input
        (
            _mutate(
                'with: {message: "{inputs.keyword}"}',
                'with: {bogus: "x", message: "m"}',
            ),
            "inputs of macro",
        ),
        # choose_item may not author answers
        (
            _mutate(
                "uses: choose_item", "uses: choose_item\n    answers: [a, escalate]"
            ),
            "declares its answers itself",
        ),
        # unknown irreversible class
        (
            _mutate(
                'with: {message: "{choose.pick}"}',
                'with: {message: "{choose.pick}"}\n    irreversible: nuclear',
            ),
            "`irreversible` must be one of",
        ),
        # stray brace
        (_mutate('"cheapest {inputs.keyword}"', '"cheapest {Keyword}"'), "stray"),
        # bare ref: `{inputs.name}` is the ONE written form
        (
            _mutate(
                'with: {message: "{inputs.keyword}"}', 'with: {message: "{keyword}"}'
            ),
            "every ref is dotted",
        ),
        # a move named `inputs` would shadow the input ref root
        (_mutate("- do: open", "- do: inputs"), "ref root"),
        # reserved routing target as a move name
        (_mutate("- tell: confirm", "- tell: escalate"), "reserved routing target"),
        # a move sharing a route page's name — one routing namespace
        (_mutate("- tell: confirm", "- tell: results"), "also a page"),
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


# ---------- the route shape ----------


def test_route_must_start_at_a_page() -> None:
    text = _mutate("route:\n  - page: home\n", "route:\n")

    with pytest.raises(PlaybookError, match="START at a page"):
        pb.parse_playbook(text, "buy", _pack())


def test_route_needs_at_least_one_move() -> None:
    text = "description: only a place\nroute:\n  - page: home\n"

    with pytest.raises(PlaybookError, match="needs at least one move"):
        pb.parse_playbook(text, "buy", _pack())


def test_do_must_be_followed_by_its_landing_page() -> None:
    text = _mutate("  - page: results\n", "")

    with pytest.raises(PlaybookError, match="followed by the page"):
        pb.parse_playbook(text, "buy", _pack())


def test_open_belongs_to_the_start_page_only() -> None:
    text = _mutate(
        "  - page: results\n",
        "  - page: results\n    open:\n      steps:\n"
        "        - {name: go, tool: home_screen}\n",
    )

    with pytest.raises(PlaybookError, match="START page only"):
        pb.parse_playbook(text, "buy", _pack())


def test_start_open_body_is_registered() -> None:
    text = _mutate(
        "route:\n  - page: home\n",
        "route:\n  - page: home\n    open:\n      steps:\n"
        "        - {name: go, tool: home_screen}\n",
    )

    p = pb.parse_playbook(text, "buy", _pack())

    assert p.start == "home" and p.open_macro == "buy.home.open"
    assert "buy.home.open" in p.inline_macros


def test_start_open_with_required_input_rejected() -> None:
    # The init ladder dispatches open with no arguments — the
    # undo/return rule, applied to the cold-launch.
    text = _mutate(
        "route:\n  - page: home\n", "route:\n  - page: home\n    open: open-app\n"
    )

    with pytest.raises(PlaybookError, match=r"requires input\(s\) message"):
        pb.parse_playbook(text, "buy", _required_message_pack())


def test_route_target_may_name_a_unique_page() -> None:
    # "this answer lands there": the page target resolves to the move
    # after that waypoint. (to-cart keeps an in-edge via none_fit so the
    # mutation doesn't orphan it.)
    text = _mutate("routes: {pick: to-cart,", "routes: {pick: results,").replace(
        "none_fit: escalate,", "none_fit: to-cart,"
    )

    p = pb.parse_playbook(text, "buy", _pack())

    assert p.nodes[1].routes["pick"] == "confirm"


def test_ambiguous_page_target_rejected() -> None:
    # `home` appears twice on the route — the landing is ambiguous.
    text = _mutate("routes: {pick: to-cart,", "routes: {pick: home,")

    with pytest.raises(PlaybookError, match="more than once"):
        pb.parse_playbook(text, "buy", _pack())


def test_page_target_with_no_following_move_rejected() -> None:
    # `results` is followed only by non-move... make it terminal:
    text = _mutate("routes: {pick: to-cart,", "routes: {pick: last,")
    text = text + '  - page: last\n    anchors: ["End"]\n'

    with pytest.raises(PlaybookError, match="nothing follows"):
        pb.parse_playbook(text, "buy", _pack())


def test_route_declared_page_reaches_the_pack() -> None:
    # A waypoint carrying anchors DECLARES the page — the matcher sees
    # it through every door (load_pack merges route declarations).
    from physiclaw.conductor import pages

    text = _mutate(
        "  - page: results\n",
        '  - page: results\n  - page: cart\n    anchors: ["Cart"]\n',
    )
    write_pack(playbooks={"buy": text})

    pack = pb.load_pack("demo")
    (entry,) = pb.scan_playbooks("demo", pack)

    assert entry.spec is not None, entry.error
    assert "cart" in pack.pages
    assert "cart" in pages.scan_app_decls("demo")


def test_page_declared_twice_rejected() -> None:
    # `home` lives in the appendix already — declaring it again on the
    # route is a conflict, not an override.
    text = _mutate(
        "route:\n  - page: home\n",
        'route:\n  - page: home\n    anchors: ["Files"]\n',
    )
    write_pack(playbooks={"buy": text})

    with pytest.raises(PlaybookError, match="declared twice"):
        pb.load_pack("demo")


def test_waypoints_are_bare_names() -> None:
    # Own-pack pages are written bare — the route IS the pack's context;
    # the old `pages.<name>` spelling is gone.
    with pytest.raises(PlaybookError, match="bare"):
        pb.parse_playbook(
            _mutate("  - page: results\n", "  - page: pages.results\n"),
            "buy",
            _pack(),
        )


# ---------- decide ----------


def test_decide_call_requires_authored_answers_with_escalate() -> None:
    pack = _pack()
    text = _mutate("uses: choose_item", "uses: decide").replace(
        'with: {criteria: "cheapest {inputs.keyword}"}', 'with: {question: "which?"}'
    )

    with pytest.raises(PlaybookError, match="escalate"):
        # decide with no answers at all
        pb.parse_playbook(text, "buy", pack)


# ---------- money lints ----------


def test_money_requires_gate_on_every_path() -> None:
    pack = _pack()
    # Make to-cart a payment move: the pay ask sits AFTER it → unguarded.
    text = _mutate(
        'with: {message: "{choose.pick}"}',
        'with: {message: "{choose.pick}"}\n    irreversible: payment',
    )

    with pytest.raises(PlaybookError, match="DIRECTLY follow an `ask`"):
        pb.parse_playbook(text, "buy", pack)


def test_money_requires_budget() -> None:
    pack = _pack()
    text = _mutate(
        'with: {message: "{choose.pick}"}',
        'with: {message: "{choose.pick}"}\n    irreversible: payment',
    )
    text = text.replace("budget: 100\n", "")

    with pytest.raises(PlaybookError, match="budget"):
        pb.parse_playbook(text, "buy", pack)


def test_any_irreversible_class_requires_its_gate() -> None:
    # send_message was previously declared but unenforced — a cancel
    # replied to a tell is never read, so nothing consequential may
    # hide behind one: every irreversible class runs as an ask's
    # fall-through.
    pack = _pack()
    text = _mutate(
        'with: {message: "{choose.pick}"}',
        'with: {message: "{choose.pick}"}\n    irreversible: send_message',
    )

    with pytest.raises(PlaybookError, match="DIRECTLY follow an `ask`"):
        pb.parse_playbook(text, "buy", pack)


def test_send_move_behind_any_ask_parses() -> None:
    pack = _pack()
    text = (
        VALID
        + """  - do: notify
    macro: add-cart
    with: {message: "sent"}
    irreversible: send_message
  - page: results
"""
    )

    p = pb.parse_playbook(text, "buy", pack)

    assert p.nodes[-1].irreversible == "send_message"


def test_payment_move_must_directly_follow_its_ask() -> None:
    # Reachability is not enough: the conductor reads the sheet AT the
    # ask and fires the move as its fall-through — a move in between
    # desynchronizes consent from the sheet, so the parser rejects it.
    pack = _pack()
    text = (
        VALID
        + """  - do: detour
    macro: add-cart
    with: {message: "x"}
  - page: results
"""
        + PAY_TAIL
    )

    with pytest.raises(PlaybookError, match="DIRECTLY follow"):
        pb.parse_playbook(text, "buy", pack)


def test_payment_move_behind_ask_parses() -> None:
    # The whole point of the ask: a confirmed reply falls through, so
    # the conductor itself executes the payment move under the budget.
    pack = _pack()

    p = pb.parse_playbook(VALID + PAY_TAIL, "buy", pack)

    assert p.nodes[-1].irreversible == "payment"
    # The derived enter (the waypoint before the ask) is what guarantees
    # money fires on a verified app page, never blind off the IM thread.
    assert p.nodes[-1].enter == "results"


def test_non_payment_ask_does_not_satisfy_the_money_dfs() -> None:
    # An address/handoff ask must NOT raise the approved flag: a decide
    # arm skipping the PAYMENT ask has to be rejected even when another
    # ask class sits upstream.
    pack = _pack()
    text = """\
description: bypass probe
inputs:
  keyword:
    description: what
budget: 100
route:
  - page: home
  - do: open
    macro: open-app
    with: {message: "{inputs.keyword}"}
  - page: home
  - ask: addr
    approve: address
    message: "address ok? reply ok or no"
  - decide: route
    uses: decide
    with: {question: "ready?"}
    answers: [go, hold, escalate]
    routes: {go: pay, hold: paygate, escalate: escalate}
  - ask: paygate
    approve: payment
    message: "Total ¥{ask.total}, reply ok or no"
    over_budget_message: "Total ¥{ask.total} over ¥{ask.cap}, reply ok or no"
    return: open-app
  - do: pay
    macro: add-cart
    with: {message: "pay"}
    irreversible: payment
  - page: home
"""
    with pytest.raises(PlaybookError, match="entered ONLY"):
        pb.parse_playbook(text, "buy", pack)


def test_routed_in_edge_to_payment_move_rejected() -> None:
    # Even after a real payment ask, a payment move reachable via a
    # decide arm would fire under the FIRST ask's consent.
    pack = _pack()
    text = (VALID + PAY_TAIL).replace("none_fit: escalate,", "none_fit: do-pay,")

    with pytest.raises(PlaybookError, match="entered ONLY"):
        pb.parse_playbook(text, "buy", pack)


# ---------- ask templates ----------


@pytest.mark.parametrize(
    "old, new, fragment",
    [
        # Messages are REQUIRED — the conductor composes no user-facing
        # prose (only the author knows the user's language).
        ('    message: "Added to the cart, ordering soon"\n', "", "is required"),
        # A payment ask must quote the sheet total…
        (
            '    message: "Total ¥{ask.total}, reply ok to pay, or no to cancel"\n',
            '    message: "reply ok to pay, or no to cancel"\n',
            "must quote the sheet",
        ),
        # …and carry the over-budget variant…
        (
            '    over_budget_message: "Total ¥{ask.total}, over budget ¥{ask.cap}, '
            'reply ok to pay, or no to cancel"\n',
            "",
            "needs `over_budget_message`",
        ),
        # …which must disclose the breached budget too.
        (", over budget ¥{ask.cap}", "", "total AND the cap"),
        # over_budget_message is meaningless outside a payment ask.
        (
            "approve: payment\n"
            '    message: "Total ¥{ask.total}, reply ok to pay, or no to cancel"',
            'approve: handoff\n    message: "reply ok to continue, or no to cancel"',
            "only for `approve: payment`",
        ),
    ],
)
def test_ask_template_lints(old, new, fragment) -> None:
    pack = _pack()

    with pytest.raises(PlaybookError, match=fragment):
        pb.parse_playbook(_mutate(old, new), "buy", pack)


# ---------- the ledger stack ----------


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
        ("type: list", "type: tuple", "`type` must be one of"),
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
            "  - decide: advance\n"
            "    uses: next_item\n"
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
        "budget:",
        "  extra:\n    description: another list\n    type: list\nbudget:",
    )

    with pytest.raises(PlaybookError, match="at most one `type: list`"):
        pb.parse_playbook(text, "shop", pack)


def test_sync_without_the_loop_rejected() -> None:
    pack = _pack()
    text = VALID + "  - sync: fix\n"

    with pytest.raises(PlaybookError, match="`sync` needs the next_item"):
        pb.parse_playbook(text, "buy", pack)


def test_revise_requires_return() -> None:
    pack = _pack()
    text = LEDGERED.replace("    return: open-app\n", "")

    with pytest.raises(PlaybookError, match="`revise` needs `return:`"):
        pb.parse_playbook(text, "shop", pack)


def test_do_missing_required_macro_input_rejected() -> None:
    pack = _required_message_pack()
    text = _mutate(
        '    with: {message: "{inputs.keyword}"}\n',
        "",
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

    with pytest.raises(PlaybookError, match="pages"):
        pb.load_pack("demo")


def test_list_apps_finds_packs() -> None:
    write_pack("demo")

    assert pb.list_apps() == ["demo"]


def test_scan_of_a_walkless_pack_is_empty() -> None:
    # A pack may be infrastructure-only (channel, ios): no `playbooks:`
    # section means no entries — never an error.
    write_pack(playbooks={})

    assert pb.scan_playbooks("demo") == []


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


# ---------- graph lints ----------


def test_cycle_beyond_self_loop_rejected() -> None:
    pack = _pack()
    # Route none_fit back to the EARLIER open move → a wide cycle.
    text = _mutate("none_fit: escalate", "none_fit: open")

    with pytest.raises(PlaybookError, match="cycle"):
        pb.parse_playbook(text, "buy", pack)


def test_unreachable_move_rejected() -> None:
    pack = _pack()
    # A decide routes explicitly (no fall-through), so routing `pick`
    # past to-cart leaves to-cart with no incoming edge at all.
    text = _mutate("routes: {pick: to-cart,", "routes: {pick: confirm,")

    with pytest.raises(PlaybookError, match="unreachable"):
        pb.parse_playbook(text, "buy", pack)


# ---------- budget ----------


def test_budget_accepts_input_ref() -> None:
    pack = _pack()
    text = VALID.replace("budget: 100", 'budget: "{inputs.keyword}"')

    p = pb.parse_playbook(text, "buy", pack)

    assert p.mandate is not None
    assert p.mandate.max_amount == pb.InputRef(name="keyword")


def test_budget_mapping_form_carries_expiry() -> None:
    pack = _pack()
    text = VALID.replace(
        "budget: 100", "budget: {max_amount: 100, expires_minutes: 30}"
    )

    p = pb.parse_playbook(text, "buy", pack)

    assert p.mandate is not None
    assert p.mandate.max_amount == 100.0 and p.mandate.expires_minutes == 30


@pytest.mark.parametrize(
    "amount, fragment",
    [
        ("budget: -5", "positive"),
        ("budget: true", "number or exactly one"),
        ('budget: "up to {inputs.keyword} yuan"', "exactly one"),
        ('budget: "{inputs.typo}"', "not declared"),
    ],
)
def test_budget_rejections(amount: str, fragment: str) -> None:
    pack = _pack()

    with pytest.raises(PlaybookError, match=fragment):
        pb.parse_playbook(VALID.replace("budget: 100", amount), "buy", pack)


# ---------- inline macros (a move's embedded body) ----------


# VALID's `open` move with the body embedded — `with:` still feeds the
# (now inline-declared) `message` input from the playbook's dotted ref.
INLINE_OPEN = """\
    macro:
      inputs:
        message: {description: the text}
      steps:
        - {name: go, tool: home_screen}
"""


def _inline(text: str = VALID) -> str:
    assert text.count("    macro: open-app\n") == 1
    return text.replace("    macro: open-app\n", INLINE_OPEN)


def test_do_macro_may_embed_the_body() -> None:
    p = pb.parse_playbook(_inline(), "buy", _pack())

    node = p.nodes[0]
    assert node.macro == "buy.open"  # synthesized: <playbook>.<move>
    m = p.inline_macros["buy.open"]
    assert m.enabled is True  # the playbook's own `enabled:` is the gate
    assert [s.name for s in m.steps] == ["go"]


def test_do_name_is_the_macro_by_default() -> None:
    # `do: open-app` with no `macro:` key runs the directory macro of
    # that name — no redundant id + macro pair.
    text = _mutate("  - do: open\n    macro: open-app\n", "  - do: open-app\n")

    p = pb.parse_playbook(text, "buy", _pack())

    assert p.nodes[0].id == "open-app" and p.nodes[0].macro == "open-app"


def test_inline_do_with_keys_validate_against_the_body() -> None:
    text = _inline().replace(
        'with: {message: "{inputs.keyword}"}\n  - page: home',
        'with: {wrong: "x"}\n  - page: home',
    )

    with pytest.raises(PlaybookError, match="wrong.*not.*inputs of macro 'buy.open'"):
        pb.parse_playbook(text, "buy", _pack())


def test_inline_do_missing_required_input_is_rejected() -> None:
    text = _inline().replace('    with: {message: "{inputs.keyword}"}\n', "")

    with pytest.raises(PlaybookError, match=r"requires input\(s\) message"):
        pb.parse_playbook(text, "buy", _pack())


def test_inline_body_errors_are_framed_with_the_move() -> None:
    text = _inline().replace("tool: home_screen", "tool: rm_rf")

    with pytest.raises(PlaybookError, match="move 'open': inline `macro`.*`tool`"):
        pb.parse_playbook(text, "buy", _pack())


def test_do_macro_rejects_a_non_string_non_mapping() -> None:
    text = VALID.replace("macro: open-app", "macro: 3")

    with pytest.raises(PlaybookError, match="pack macro name or an inline mapping"):
        pb.parse_playbook(text, "buy", _pack())


def test_disabled_leg_macros_skips_inline_bodies() -> None:
    # An inline macro is not in `pack.macros` — the readiness check must
    # neither KeyError on it nor report it (its gate is the playbook's).
    root = write_pack()
    mp = root / "macros" / "add-cart" / "MACRO.yml"
    mp.write_text(mp.read_text(encoding="utf-8") + "enabled: false\n", encoding="utf-8")
    pack = pb.load_pack("demo")

    spec = pb.parse_playbook(_inline(), "buy", pack)

    assert pb.disabled_leg_macros(spec, pack) == ["add-cart"]


def test_qualified_inline_mints_dispatch_keys() -> None:
    spec = pb.parse_playbook(_inline(), "buy", _pack())

    assert pb.qualified_inline("demo", spec) == {
        "demo/buy.open": spec.inline_macros["buy.open"]
    }


def test_pack_doc_rejects_yaml_aliases() -> None:
    # Inline macros put clause parsing (which materializes per path — the
    # alias-bomb ride) inside the pack file, so the pack door inherits
    # the MACRO.yml document-wide guard.
    root = write_pack()
    (root / "PLAYBOOK.yml").write_text(
        "app: demo\ndescription: d\n"
        "pages: &a {home: {anchors: [Files]}}\nplaybooks: *a\n",
        encoding="utf-8",
    )

    with pytest.raises(PlaybookError, match="aliases"):
        pb.load_pack("demo")


def test_ask_return_may_embed_the_body() -> None:
    text = VALID + (
        "    return:\n      steps:\n        - {name: back-to-app, tool: home_screen}\n"
    )
    pack = _pack()

    spec = pb.parse_playbook(text, "buy", pack)

    gate = spec.nodes[4]
    assert gate.return_macro == "buy.pay.return"  # <playbook>.<move>.<role>
    assert [s.name for s in spec.inline_macros["buy.pay.return"].steps] == [
        "back-to-app"
    ]
    # The readiness check must skip the role body, not KeyError on it.
    assert pb.disabled_leg_macros(spec, pack) == []


def test_do_undo_may_embed_the_body() -> None:
    text = _mutate(
        'with: {message: "{choose.pick}"}',
        'with: {message: "{choose.pick}"}\n'
        "    undo:\n"
        "      steps:\n"
        "        - {name: back, tool: home_screen}",
    )

    spec = pb.parse_playbook(text, "buy", _pack())

    assert spec.nodes[2].compensate == "buy.to-cart.undo"
    assert "buy.to-cart.undo" in spec.inline_macros


def test_inline_role_body_with_required_input_rejected() -> None:
    # undo/return dispatch with no arguments — a required input could
    # only abort at run time (right after a confirmed ask), so the lint
    # runs on the RESOLVED macro.
    text = VALID + (
        "    return:\n"
        "      inputs:\n"
        "        x: {description: d}\n"
        "      steps:\n"
        "        - {name: back, tool: home_screen}\n"
    )

    with pytest.raises(PlaybookError, match=r"requires input\(s\) x"):
        pb.parse_playbook(text, "buy", _pack())


def test_directory_role_macro_with_required_input_rejected() -> None:
    # The same lint, directory spelling: the rule is role-shaped, not
    # embedding-shaped — moving a body out to macros/ must not lose it.
    text = VALID + "    return: open-app\n"

    with pytest.raises(PlaybookError, match=r"requires input\(s\) message"):
        pb.parse_playbook(text, "buy", _required_message_pack())


def test_scrollable_only_waypoint_is_a_declaration_at_both_doors() -> None:
    # `pages.route_decl` is the ONE declaration predicate: a waypoint
    # carrying only `scrollable:` must read as a (re)declaration at the
    # pack door AND the text door — the two once disagreed, so a
    # scrollable-only re-declare slipped the playbook parse while the
    # pack door raised "declared twice".
    text = _mutate(
        "route:\n  - page: home\n",
        "route:\n  - page: home\n    scrollable: true\n",
    )
    write_pack(playbooks={"buy": text})

    with pytest.raises(PlaybookError, match="declared twice"):
        pb.load_pack("demo")


def test_text_door_validates_inplace_declarations_too() -> None:
    # A playbook green at `parse_playbook` must not go red at the pack
    # door: the in-place declaration's CONTENT rides the same page
    # grammar at both (here: a single-char anchor without a region).
    text = _mutate(
        "  - page: results\n",
        '  - page: cart\n    anchors: ["x"]\n',
    )

    with pytest.raises(PlaybookError, match="single-character"):
        pb.parse_playbook(text, "buy", _pack())


def test_disabled_directory_open_is_reported_not_armed() -> None:
    # `open:` may name a directory macro; a disabled one must surface in
    # the readiness check (rehearse-then-enable) — and the walk must not
    # arm it as the reset rung's hand.
    from conductor_fakes import PACK_MACRO

    root = write_pack()
    d = root / "macros" / "go-home"
    d.mkdir(parents=True)
    (d / "MACRO.yml").write_text(
        PACK_MACRO.format(name="go-home") + "enabled: false\n", encoding="utf-8"
    )
    pack = pb.load_pack("demo")
    text = _mutate(
        "route:\n  - page: home\n", "route:\n  - page: home\n    open: go-home\n"
    )

    spec = pb.parse_playbook(text, "buy", pack)

    assert spec.open_macro == "go-home"
    assert pb.disabled_leg_macros(spec, pack) == ["go-home"]
