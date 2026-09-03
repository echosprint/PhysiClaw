"""Tests for `physiclaw.conductor.playbook` and `route` — the route
grammar, its lints, and pack loading. Every rejection must name the
exact field and rule; the money lint is the safety substance."""

from __future__ import annotations

import pytest
from conductor_fakes import write_pack

from physiclaw.common import paths
from physiclaw.conductor import playbook as pb
from physiclaw.conductor.playbook import PlaybookError

VALID = """\
name: buy
description: test playbook
enabled: false
inputs:
  keyword:
    description: what to search
route:
  - page: home
  - do: open
    macro: open-app
    with: {message: "{inputs.keyword}"}
  - page: home
  - agent: choose
    prompt: "Pick the cheapest {inputs.keyword} and land on the results"
    tools: [tap, scroll]
    returns:
      pick: the chosen item's title
    limit: {calls: 4, scrolls: 2}
  - page: results
  - do: to-cart
    macro: add-cart
    with: {message: "{choose.pick}"}
  - page: done
  - tell: confirm
    message: "Added to the cart, ordering soon"
  - ask: pay
    approve: payment
    total: "Total"
    message: "Total ¥{ask.total}, reply ok to pay, or no to cancel"
    yes: ["ok"]
    no: ["no"]
    resume: open-app
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


# A self-contained playbook file: its one page declared in place, its
# one hand embedded — nothing from the manifest or the macros dir.
FLOW_MIN = """\
description: minimal
route:
  - page: home
    anchors: ["Files"]
  - do: open
    macro: {steps: [{name: go, tool: home_screen}]}
  - page: home
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


def test_parse_valid_playbook() -> None:
    p = pb.parse_playbook(VALID, "buy", _pack())

    assert p.name == "buy" and p.enabled is False
    assert p.start == "home"
    kinds = [type(n).__name__ for n in p.nodes]
    assert kinds == ["DoNode", "AgentNode", "DoNode", "TellNode", "AskNode"]
    choose = p.nodes[1]
    assert choose.tools == ("tap", "scroll") and choose.return_fields == ("pick",)
    assert choose.max_calls == 4 and choose.max_scrolls == 2
    assert p.nodes[2].irreversible is None
    assert p.nodes[4].approve == "payment"


def test_derived_checks_come_from_the_waypoints() -> None:
    # A move's enter is the nearest preceding page, its verify the page
    # that follows it — no enter/verify keys exist to author.
    p = pb.parse_playbook(VALID, "buy", _pack())

    assert p.nodes[0].enter == "home" and p.nodes[0].verify == "home"
    assert p.nodes[1].enter == "home" and p.nodes[1].verify == "results"
    assert p.nodes[2].enter == "results" and p.nodes[2].verify == "done"


def test_scan_playbooks_reads_pack_files() -> None:
    write_pack(playbooks={"buy": VALID, "broken": "description: d\nroute: []"})

    entries = {e.name: e for e in pb.scan_playbooks("demo")}

    assert entries["buy"].spec is not None
    assert entries["broken"].spec is None and entries["broken"].error


def test_retired_keys_are_unknown() -> None:
    # The grammar has no decide, sync, context, undo, open, or return:
    # whatever needs judgment is an agent step, whatever needs a human
    # is an ask, and a page's recover is the only recovery.
    pack = _pack()
    for text, fragment in (
        (
            VALID.replace("enabled: false\n", "enabled: false\ncontext: [memory.x]\n"),
            "unknown key",
        ),
        (VALID + "  - sync: fix\n", "exactly one of"),
        (
            VALID.replace('    with: {message: "pay"}', "")  # no-op guard
            + "  - decide: q\n    uses: decide\n",
            "exactly one of",
        ),
        (
            VALID.replace("  - page: done\n", "  - page: done\n    open: open-app\n"),
            "unknown key",
        ),
        (VALID + "    return: open-app\n", "unknown key"),
    ):
        with pytest.raises(PlaybookError, match=fragment):
            pb.parse_playbook(text, "buy", pack)


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
                "description: test playbook", "stem: buy\ndescription: test playbook"
            ),
            "unknown key",
        ),
        # unknown top key
        (VALID + "bogus: 1\n", "unknown key"),
        # duplicate move name
        (_mutate("- tell: confirm", "- tell: open"), "duplicate move name"),
        # unknown entry kind
        (_mutate("  - tell: confirm\n", "  - shout: confirm\n"), "exactly one of"),
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
            "EARLIER agent",
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
        # unknown irreversible class
        (
            _mutate(
                'with: {message: "{choose.pick}"}',
                'with: {message: "{choose.pick}"}\n    irreversible: nuclear',
            ),
            "`irreversible` must be one of",
        ),
        # stray brace
        (_mutate("cheapest {inputs.keyword}", "cheapest {Keyword}"), "stray"),
        # bare ref: `{inputs.name}` is the ONE written form
        (
            _mutate(
                'with: {message: "{inputs.keyword}"}', 'with: {message: "{keyword}"}'
            ),
            "every ref is dotted",
        ),
        # a move named `inputs` would shadow the input ref root
        (_mutate("- do: open", "- do: inputs"), "ref root"),
        # a move sharing a route page's name — one namespace
        (_mutate("- tell: confirm", "- tell: results"), "also a page"),
    ],
)
def test_rejections_name_the_rule(text: str, fragment: str) -> None:
    pack = _pack()

    with pytest.raises(PlaybookError, match=fragment):
        pb.parse_playbook(text, "buy", pack)


# ---------- the route shape ----------


def test_route_must_start_at_a_page() -> None:
    # A screen-touching move before the first page breaks the start
    # contract — only pure-text agents and the `start` move may precede.
    text = _mutate("route:\n  - page: home\n", "route:\n")

    with pytest.raises(PlaybookError, match="precede the first page"):
        pb.parse_playbook(text, "buy", _pack())


def test_route_needs_at_least_one_move() -> None:
    text = "name: buy\ndescription: only a place\nroute:\n  - page: home\n"

    with pytest.raises(PlaybookError, match="needs at least one move"):
        pb.parse_playbook(text, "buy", _pack())


def test_do_must_be_followed_by_its_landing_page() -> None:
    text = _mutate("  - page: done\n", "")

    with pytest.raises(PlaybookError, match="followed by the page"):
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
    # Own-pack pages are written bare — the route IS the pack's context.
    with pytest.raises(PlaybookError, match="bare"):
        pb.parse_playbook(
            _mutate("  - page: results\n", "  - page: pages.results\n"),
            "buy",
            _pack(),
        )


# ---------- money lints ----------


def test_money_requires_an_ask_directly_before() -> None:
    pack = _pack()
    # Make to-cart a payment move: the pay ask sits AFTER it → unguarded.
    text = _mutate(
        'with: {message: "{choose.pick}"}',
        'with: {message: "{choose.pick}"}\n    irreversible: payment',
    )

    with pytest.raises(PlaybookError, match="DIRECTLY follow an `ask`"):
        pb.parse_playbook(text, "buy", pack)


def test_payment_is_the_only_irreversible_class() -> None:
    pack = _pack()
    text = _mutate(
        'with: {message: "{choose.pick}"}',
        'with: {message: "{choose.pick}"}\n    irreversible: send_message',
    )

    with pytest.raises(PlaybookError, match="`irreversible` must be one of payment"):
        pb.parse_playbook(text, "buy", pack)


def test_payment_move_must_directly_follow_its_ask() -> None:
    # Adjacency, not just reachability: the conductor reads the sheet AT
    # the ask and fires the move as its fall-through — a move in between
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
    # the conductor itself executes the payment move under the consent.
    pack = _pack()

    p = pb.parse_playbook(VALID + PAY_TAIL, "buy", pack)

    assert p.nodes[-1].irreversible == "payment"
    # The derived enter (the waypoint before the ask) is what guarantees
    # money fires on a verified app page, never blind off the IM thread.
    assert p.nodes[-1].enter == "done"


def test_non_payment_ask_does_not_approve_payment() -> None:
    # An address/handoff ask must NOT open a payment move: the class the
    # ask approves is declared, and money keys off the declaration.
    pack = _pack()
    text = """\
name: buy
description: bypass probe
inputs:
  keyword:
    description: what
route:
  - page: home
  - do: open
    macro: open-app
    with: {message: "{inputs.keyword}"}
  - page: home
  - ask: addr
    approve: address
    message: "address ok? reply ok or no"
    yes: ["ok"]
    no: ["no"]
  - do: pay
    macro: add-cart
    with: {message: "pay"}
    irreversible: payment
  - page: home
"""
    with pytest.raises(PlaybookError, match="approve: payment"):
        pb.parse_playbook(text, "buy", pack)


def test_payment_agent_episode_takes_the_ask_total() -> None:
    # An `irreversible: payment` agent directly after its ask may quote
    # {ask.total} — the one gate slot a payment episode reads.
    pack = _pack()
    text = (
        VALID
        + """  - agent: checkout
    prompt: "Pay exactly ¥{ask.total}"
    tools: [tap]
    irreversible: payment
    limit: {calls: 3}
  - page: results
"""
    )

    p = pb.parse_playbook(text, "buy", pack)

    assert p.nodes[-1].irreversible == "payment" and p.nodes[-1].enter == "done"


# ---------- ask templates ----------


@pytest.mark.parametrize(
    "old, new, fragment",
    [
        # The reply words are the ask's own — required, non-empty.
        ('    yes: ["ok"]\n', "", "`yes` must be a list"),
        ('    yes: ["ok"]\n', "    yes: []\n", "at least one reply word"),
        ('    no: ["no"]\n', "    no: [3]\n", "must be a string"),
    ],
)
def test_ask_reply_words_are_declared(old, new, fragment) -> None:
    with pytest.raises(PlaybookError, match=fragment):
        pb.parse_playbook(_mutate(old, new), "buy", _pack())


def test_agent_context_is_declared_and_checked() -> None:
    text = _mutate(
        "    limit: {calls: 4, scrolls: 2}\n",
        "    limit: {calls: 4, scrolls: 2}\n    context: [memory.shopping, daylog]\n",
    )

    p = pb.parse_playbook(text, "buy", _pack())
    assert p.nodes[1].context == ("memory.shopping", "daylog")

    with pytest.raises(PlaybookError, match="`context` entry"):
        pb.parse_playbook(
            _mutate(
                "    limit: {calls: 4, scrolls: 2}\n",
                "    limit: {calls: 4, scrolls: 2}\n    context: [pitfalls]\n",
            ),
            "buy",
            _pack(),
        )


def test_tell_may_declare_its_cancel_words() -> None:
    text = _mutate(
        '    message: "Added to the cart, ordering soon"\n',
        '    message: "Added to the cart, ordering soon"\n    no: ["stop"]\n',
    )

    p = pb.parse_playbook(text, "buy", _pack())

    assert p.nodes[3].no == ("stop",)


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
        # {ask.total} exists only inside a payment ask.
        (
            "approve: payment\n"
            '    total: "Total"\n'
            '    message: "Total ¥{ask.total}, reply ok to pay, or no to cancel"',
            'approve: handoff\n    message: "Total ¥{ask.total}, reply ok or no"',
            "references move 'ask'",
        ),
        # A payment ask declares where the total sits…
        ('    total: "Total"\n', "", "declares `total:`"),
        # …and only a payment ask does.
        (
            "approve: payment\n"
            '    total: "Total"\n'
            '    message: "Total ¥{ask.total}, reply ok to pay, or no to cancel"\n',
            'approve: handoff\n    total: "Total"\n    message: "reply ok or no"\n',
            "goes with `approve: payment`",
        ),
        # The ask's patience is bounded by the engine's single sleep.
        (
            '    total: "Total"\n',
            '    total: "Total"\n    wait: {seconds: 600}\n',
            "wait.seconds",
        ),
    ],
)
def test_ask_template_lints(old, new, fragment) -> None:
    pack = _pack()

    with pytest.raises(PlaybookError, match=fragment):
        pb.parse_playbook(_mutate(old, new), "buy", pack)


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
    write_text(root / "PLAYBOOK.yml", scaffold.render_manifest_stub("newapp"))
    write_text(
        root / f"{scaffold.EXAMPLE_PLAYBOOK}.yml",
        scaffold.render_playbook_stub("newapp"),
    )
    write_text(
        root / pb.PACK_MACROS_DIRNAME / scaffold.EXAMPLE_MACRO / "MACRO.yml",
        scaffold.render_example_macro(),
    )

    (entry,) = pb.scan_playbooks("newapp")

    assert entry.name == scaffold.EXAMPLE_PLAYBOOK
    assert entry.error is None, entry.error
    assert entry.spec is not None and entry.spec.enabled is False


# ---------- the manifest is a manifest: shared knowledge, never a route ----------


def test_an_empty_manifest_is_a_pack() -> None:
    root = paths.playbooks_dir() / "bare"
    root.mkdir(parents=True)
    (root / "PLAYBOOK.yml").write_text("", encoding="utf-8")
    (root / "flow.yml").write_text("name: flow\n" + FLOW_MIN, encoding="utf-8")

    pack = pb.load_pack("bare")

    assert pack.app == "bare" and set(pack.pages) == {"home"} and pack.landmarks == {}
    (entry,) = pb.scan_playbooks("bare", pack)
    assert entry.name == "flow" and entry.error is None


def test_manifest_refuses_a_playbooks_section() -> None:
    root = write_pack()
    (root / "PLAYBOOK.yml").write_text(
        "app: demo\nplaybooks:\n  x: {description: d, route: []}\n", encoding="utf-8"
    )

    with pytest.raises(PlaybookError, match="own <name>.yml"):
        pb.load_pack("demo")


def test_a_playbook_file_that_will_not_load_is_an_invalid_entry() -> None:
    root = write_pack(
        pages='results:\n  anchors: ["综合"]\n', playbooks={"flow": FLOW_MIN}
    )
    (root / "broken.yml").write_text("route: [\n", encoding="utf-8")
    (root / "Bad Name.yml").write_text("description: d\n", encoding="utf-8")

    entries = {e.name: e for e in pb.scan_playbooks("demo")}

    assert entries["flow"].spec is not None
    assert "invalid YAML" in (entries["broken"].error or "")
    assert entries["Bad Name"].spec is None and entries["Bad Name"].error


def test_a_page_declared_in_two_files_is_a_pack_error() -> None:
    write_pack(
        pages='results:\n  anchors: ["综合"]\n',
        playbooks={"a": FLOW_MIN, "b": FLOW_MIN},
    )

    with pytest.raises(PlaybookError, match="declared twice"):
        pb.load_pack("demo")


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


def test_disabled_macros_skips_inline_bodies() -> None:
    # An inline macro is not in `pack.macros` — the readiness check must
    # neither KeyError on it nor report it (its gate is the playbook's).
    root = write_pack()
    mp = root / "macros" / "add-cart" / "MACRO.yml"
    mp.write_text(mp.read_text(encoding="utf-8") + "enabled: false\n", encoding="utf-8")
    pack = pb.load_pack("demo")

    spec = pb.parse_playbook(_inline(), "buy", pack)

    assert pb.disabled_macros(spec, pack) == ["add-cart"]


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


def test_ask_resume_may_embed_the_body() -> None:
    text = _mutate(
        "    resume: open-app\n",
        "    resume:\n      steps:\n        - {name: back-to-app, tool: home_screen}\n",
    )
    pack = _pack()

    spec = pb.parse_playbook(text, "buy", pack)

    ask = spec.nodes[4]
    assert ask.resume == "buy.pay.resume"  # <playbook>.<move>.<role>
    assert [s.name for s in spec.inline_macros["buy.pay.resume"].steps] == [
        "back-to-app"
    ]
    # The readiness check must skip the role body, not KeyError on it.
    assert pb.disabled_macros(spec, pack) == []


def test_inline_role_body_with_required_input_rejected() -> None:
    # resume/recover dispatch with no arguments — a required input could
    # only abort at run time (right after a confirmed ask), so the lint
    # runs on the RESOLVED macro.
    text = _mutate(
        "    resume: open-app\n",
        "    resume:\n"
        "      inputs:\n"
        "        x: {description: d}\n"
        "      steps:\n"
        "        - {name: back, tool: home_screen}\n",
    )

    with pytest.raises(PlaybookError, match=r"requires input\(s\) x"):
        pb.parse_playbook(text, "buy", _pack())


def test_directory_role_macro_with_required_input_rejected() -> None:
    # The same lint, directory spelling: the rule is role-shaped, not
    # embedding-shaped — moving a body out to macros/ must not lose it.
    with pytest.raises(PlaybookError, match=r"requires input\(s\) message"):
        pb.parse_playbook(VALID, "buy", _required_message_pack())


def test_scrollable_only_waypoint_is_a_declaration_at_both_doors() -> None:
    # `pages.route_decl` is the ONE declaration predicate: a waypoint
    # carrying only `scrollable:` must read as a (re)declaration at the
    # pack door AND the text door.
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


def test_disabled_recover_macro_is_reported_not_run() -> None:
    # `recover:` may name a directory macro; a disabled one must surface
    # in the readiness check (rehearse-then-enable).
    from conductor_fakes import PACK_MACRO

    root = write_pack()
    d = root / "macros" / "go-home"
    d.mkdir(parents=True)
    (d / "MACRO.yml").write_text(
        PACK_MACRO.format(name="go-home") + "enabled: false\n", encoding="utf-8"
    )
    pack = pb.load_pack("demo")
    text = _mutate(
        "route:\n  - page: home\n",
        "route:\n  - page: home\n    recover: {macro: go-home}\n",
    )

    spec = pb.parse_playbook(text, "buy", pack)

    assert spec.recovers["home"].elsewhere.macro == "go-home"
    assert pb.disabled_macros(spec, pack) == ["go-home"]


# ---------- the boot: `activate`, and the locked reading ----------

CHANNEL_OPEN = """\
name: open
description: open the thread
steps:
  - name: go
    tool: tap
    with: {label: t, bbox: [0.1, 0.1, 0.2, 0.2]}
"""

BOOT = """\
name: boot
description: reach the thread and read it
route:
  - page: thread
    recover:
      locked: {tool: unlock_phone}
      elsewhere: {macro: open}
      limit: 4
  - activate: parse
"""


def _channel_pack():
    from conductor_fakes import write_channel

    write_channel(CHANNEL_OPEN)
    return pb.load_pack("channel")


def test_the_boot_parses_with_its_activate_step_last() -> None:
    from physiclaw.conductor.playbook import ActivateNode

    spec = pb.parse_playbook(BOOT, "boot", _channel_pack())

    node = spec.nodes[-1]
    assert isinstance(node, ActivateNode)
    assert node.enter == "thread" and node.max_scrolls == 2
    assert spec.recovers["thread"].locked is not None
    assert spec.recovers["thread"].limit == 4

    bounded = pb.parse_playbook(
        BOOT.replace(
            "  - activate: parse\n", "  - activate: parse\n    limit: {scrolls: 0}\n"
        ),
        "boot",
        _channel_pack(),
    )
    assert bounded.nodes[-1].max_scrolls == 0


def test_activate_belongs_to_the_channel_boot_only() -> None:
    # An app playbook cannot read the thread for a task — that is the
    # boot's; and even in the channel pack, only the file named boot.
    text = VALID.split("  - tell: confirm")[0] + "  - activate: parse\n"
    with pytest.raises(PlaybookError, match="channel boot's own step"):
        pb.parse_playbook(text, "buy", _pack())
    with pytest.raises(PlaybookError, match="channel boot's own step"):
        pb.parse_playbook(
            BOOT.replace("name: boot", "name: other"), "other", _channel_pack()
        )


@pytest.mark.parametrize(
    "edit, message",
    [
        # activate must read the thread: the page before it is the thread
        (
            lambda t: t.replace(
                "  - activate: parse\n",
                "  - do: nudge\n    macro: open\n  - page: other\n"
                '    anchors: ["Somewhere else"]\n'
                "  - activate: parse\n",
            ),
            "immediately before it",
        ),
        # activate ends the boot
        (
            lambda t: t + "  - do: after\n    macro: open\n  - page: thread\n",
            "must end with an `activate`",
        ),
        # …and only one does
        (
            lambda t: t.replace(
                "  - activate: parse\n",
                "  - activate: first\n  - page: thread\n  - activate: parse\n",
            ),
            "one `activate` only",
        ),
        # a boot must have one
        (
            lambda t: t.replace(
                "  - activate: parse\n",
                "  - do: nudge\n    macro: open\n  - page: thread\n",
            ),
            "must end with an `activate`",
        ),
        # the boot never speaks to the user
        (
            lambda t: t.replace(
                "  - activate: parse\n",
                "  - tell: hi\n    message: hello\n  - activate: parse\n",
            ),
            "messages the user",
        ),
        # limit takes scrolls only
        (
            lambda t: t.replace(
                "  - activate: parse\n", "  - activate: parse\n    limit: {calls: 2}\n"
            ),
            "takes only `scrolls`",
        ),
    ],
)
def test_boot_shape_lints(edit, message) -> None:
    with pytest.raises(PlaybookError, match=message):
        pb.parse_playbook(edit(BOOT), "boot", _channel_pack())


def test_flat_recover_covers_the_locked_reading_too() -> None:
    # `recover: {tool: x}` is one hand for ANY deviation — the lock
    # screen included, exactly as the docstring promises; the keyed form
    # is where an author tells the readings apart.
    text = VALID.replace(
        "  - page: home\n  - do: open",
        "  - page: home\n    recover: {tool: force_quit}\n  - do: open",
        1,
    )
    spec = pb.parse_playbook(text, "buy", _pack())

    r = spec.recovers["home"]
    assert r.locked is r.elsewhere is r.occluded
    assert r.hand_for("locked") is r.locked
