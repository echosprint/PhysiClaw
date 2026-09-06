"""One pack, many playbooks: what the manifest declares once — pages
with their recover hands, landmarks, placeholders — and the hands the
macros directory records, every playbook file sees. The Taobao shape
before the second task arrives, proved on a fixture pack."""

from __future__ import annotations

import pytest
from conductor_fakes import PACK_MACRO, write_local_macro, write_playbook

from physiclaw.common import paths
from physiclaw.common.placeholders import write_placeholder_values
from physiclaw.conductor.drive import activation, build
from physiclaw.conductor.drive import setup as conductor_setup
from physiclaw.conductor.spec import pack as pb
from physiclaw.conductor.spec.model import PlaybookError, RecoverHand
from physiclaw.conductor.spec.pack import qualified_all
from physiclaw.conductor.spec.pages import prints_for_app

MANIFEST = """\
app: shop
description: two tasks in one app
placeholders:
  CONTACT:
    description: the user thread
    example: Someone
landmarks:
  dismiss:
    label: "scrim"
    at: [0.35, 0.16, 0.65, 0.24]
pages:
  home:
    anchors: ["Files"]
    recover: force_quit
  results:
    anchors: ["综合"]
    recover:
      covered: {tap: landmarks.dismiss}
      elsewhere: {macro: launch}
    tries: 2
"""

BUY = """\
name: buy
description: buy something
inputs:
  keyword:
    description: what to search
route:
  - start: app
    macro: launch
  - page: home
  - do: search
    macro: search
    with: {message: "{inputs.keyword}"}
  - page: results
  - agent: pick
    prompt: "pick for <<CONTACT>>"
    tools: [tap]
    give: [landmarks.dismiss]
    returns:
      summary: one line
  - page: results
"""

TRACK = """\
name: track
description: track an order
route:
  - start: app
    macro: launch
  - page: home
  - do: orders
    macro: search
    with: {message: "orders"}
  - page: results
    recover: go_back
  - tell: done
    message: "<<CONTACT>>, your order is on its way"
"""


def _write_pack(app: str, manifest: str, **playbooks: str):
    """A pack the fixtures share: two recorded hands, the manifest, and
    the playbook files given by name."""
    root = paths.playbooks_dir() / app
    (root / "macros").mkdir(parents=True)
    for name in ("launch", "search"):
        (root / "macros" / f"{name}.yml").write_text(
            PACK_MACRO.format(name=name), encoding="utf-8"
        )
    (root / "APP.yml").write_text(manifest, encoding="utf-8")
    for name, text in playbooks.items():
        write_playbook(root, name, text)
    return root


@pytest.fixture()
def shop():
    write_placeholder_values({"CONTACT": "Alice"})
    return _write_pack("shop", MANIFEST, buy=BUY, track=TRACK)


def test_both_playbooks_parse_against_the_one_manifest(shop) -> None:
    pack = pb.load_pack("shop")
    entries = {e.name: e for e in pb.scan_playbooks("shop", pack)}

    assert set(entries) == {"buy", "track"}
    assert all(e.error is None for e in entries.values()), entries
    assert set(pack.pages) == {"home", "results"}
    assert set(pack.landmarks) == {"dismiss"}


def test_manifest_hands_are_inherited_and_a_route_may_override(shop) -> None:
    pack = pb.load_pack("shop")
    buy, track = (
        next(e.spec for e in pb.scan_playbooks("shop", pack) if e.name == n)
        for n in ("buy", "track")
    )
    assert buy is not None and track is not None

    # buy declares nothing: it walks with the manifest's hands as is.
    assert buy.recovers["home"].elsewhere == RecoverHand(tool="force_quit")
    assert buy.recovers["results"].covered == RecoverHand(
        tool="tap", landmark="dismiss"
    )
    assert buy.recovers["results"].elsewhere == RecoverHand(macro="launch")
    assert buy.recovers["results"].tries == 2
    # track overrides results for its own walk and still inherits home.
    assert track.recovers["results"].elsewhere == RecoverHand(tool="go_back")
    assert track.recovers["home"] == buy.recovers["home"]


def test_shared_macros_placeholders_and_landmarks_reach_every_file(shop) -> None:
    pack = pb.load_pack("shop")
    specs = {e.name: e.spec for e in pb.scan_playbooks("shop", pack)}

    # Both routes start with the same recorded hand, by name.
    assert specs["buy"].nodes[0].macro == "launch"
    assert specs["track"].nodes[0].macro == "launch"
    # The one dispatch table a walk of this pack needs: both directory
    # macros, and no inline bodies since neither file embedded one.
    assert set(qualified_all("shop", pack)) == {"shop/launch", "shop/search"}
    # The placeholder filled at load, in a prompt and in a message.
    assert "pick for Alice" in specs["buy"].nodes[2].prompt
    assert specs["track"].nodes[2].message.startswith("Alice,")
    # The landmark granted by name in one file is the manifest's.
    assert specs["buy"].nodes[2].give == ("dismiss",)


def test_the_matcher_sees_the_shared_pages_and_a_walk_builds(shop) -> None:
    pack = pb.load_pack("shop")
    assert {p.decl.name for p in prints_for_app("shop", decls=pack.pages)} == {
        "home",
        "results",
    }
    for name in ("buy", "track"):
        spec, _ = build.load_spec("shop", name, require_live=False)
        program = build.build_program(
            spec,
            pack,
            build.resolve_inputs(spec, {"keyword": "x"} if name == "buy" else {}),
            None,
            dry=True,
        )
        assert set(conductor_setup.walk_registry(program, None)) == {
            "shop/launch",
            "shop/search",
        }
        assert program.spec.recovers["home"].elsewhere == RecoverHand(tool="force_quit")


def test_manifest_hand_must_name_a_recorded_macro_never_a_body(shop) -> None:
    # Resolved by each route's compiler: every playbook of the pack
    # reports the manifest's fault, the pack itself still loads.
    (shop / "APP.yml").write_text(
        MANIFEST.replace(
            "elsewhere: {macro: launch}",
            "elsewhere: {macro: {steps: [home_screen]}}\n",
        ),
        encoding="utf-8",
    )

    entries = pb.scan_playbooks("shop")

    assert all(
        "record the body as macros/<name>.yml" in (e.error or "") for e in entries
    )


def test_manifest_hand_on_an_undeclared_page_is_refused(shop) -> None:
    # A page entry with a hand and no anchors is not a declaration: the
    # pack refuses it as a page, before any route could inherit it.
    (shop / "APP.yml").write_text(
        MANIFEST + "  ghost:\n    recover: go_back\n", encoding="utf-8"
    )

    with pytest.raises(PlaybookError, match="ghost"):
        pb.load_pack("shop")


def test_a_page_declared_in_the_manifest_and_a_file_is_refused(shop) -> None:
    (shop / "track").mkdir(exist_ok=True)
    (shop / "track" / "PLAYBOOK.yml").write_text(
        TRACK.replace(
            "  - page: home\n", '  - page: home\n    anchors: ["Files"]\n', 1
        ),
        encoding="utf-8",
    )

    with pytest.raises(PlaybookError, match="declared twice"):
        pb.load_pack("shop")


def test_activation_menu_is_one_line_per_playbook_and_check_flags_twins(shop) -> None:
    # A second pack offering the same task with the same words: the
    # menu shows two identical lines under different refs, and `check`
    # says so — the fix is a description that names the app.
    from typer.testing import CliRunner

    from physiclaw.cli import app as cli_app

    _write_pack("mall", MANIFEST.replace("app: shop", "app: mall"), buy=BUY)
    entries = {}
    for app in ("shop", "mall"):
        pack = pb.load_pack(app)
        for e in pb.scan_playbooks(app, pack):
            if e.name == "buy":
                entries[f"{app}/buy"] = (e.spec, pack)

    menu = activation.Activation(entries=entries, channel=None)._menu()
    result = CliRunner().invoke(cli_app, ["playbooks", "check"])

    assert menu.splitlines() == [
        "Available playbooks:",
        "- shop/buy: buy something [inputs: keyword (what to search)]",
        "- mall/buy: buy something [inputs: keyword (what to search)]",
    ]
    assert "same description" in result.output and "mall/buy" in result.output


# ---------- a playbook's own recorded hands ----------

LOCAL_BUY = """\
name: buy
description: buy with its own recorded search
route:
  - start: open
    macro: launch
  - page: home
  - do: find
    macro: search-own
  - page: results
"""


def test_a_playbook_macro_file_resolves_by_bare_name_under_the_inline_spelling() -> (
    None
):
    root = _write_pack("shop", MANIFEST, buy=LOCAL_BUY)
    write_local_macro(root, "buy", "search-own")

    pack = pb.load_pack("shop")
    (buy,) = (e.spec for e in pb.scan_playbooks("shop", pack) if e.name == "buy")

    assert buy is not None
    # The route names it bare; it dispatches as the route's own, exactly
    # like an inline body, and joins the pack's dispatch table.
    assert buy.nodes[1].macro == "buy.search-own"
    assert buy.inline_macros["buy.search-own"].name == "buy.search-own"
    assert set(qualified_all("shop", pack)) == {
        "shop/launch",
        "shop/search",
        "shop/buy.search-own",
    }


def test_an_unreferenced_playbook_macro_is_still_the_routes_to_run() -> None:
    root = _write_pack("shop", MANIFEST, buy=LOCAL_BUY)
    write_local_macro(root, "buy", "search-own")
    write_local_macro(root, "buy", "spare")

    pack = pb.load_pack("shop")

    assert "shop/buy.spare" in qualified_all("shop", pack)


def test_a_name_in_both_the_pack_and_the_playbook_is_refused() -> None:
    root = _write_pack("shop", MANIFEST, buy=LOCAL_BUY)
    write_local_macro(root, "buy", "search")  # the pack already records `search`

    entries = {e.name: e for e in pb.scan_playbooks("shop")}

    assert "declared both in buy/macros/ and the pack's macros/" in (
        entries["buy"].error or ""
    )


def test_two_playbooks_may_each_own_a_macro_of_the_same_name() -> None:
    root = _write_pack(
        "shop",
        MANIFEST,
        buy=LOCAL_BUY,
        track=LOCAL_BUY.replace("name: buy", "name: track"),
    )
    write_local_macro(root, "buy", "search-own")
    write_local_macro(root, "track", "search-own")

    pack = pb.load_pack("shop")
    specs = {e.name: e.spec for e in pb.scan_playbooks("shop", pack)}

    assert specs["buy"].nodes[1].macro == "buy.search-own"
    assert specs["track"].nodes[1].macro == "track.search-own"


def test_a_broken_playbook_macro_fails_the_route_that_names_it_with_the_cause() -> None:
    root = _write_pack("shop", MANIFEST, buy=LOCAL_BUY)
    write_local_macro(root, "buy", "search-own", "name: other\n")

    pack = pb.load_pack("shop")
    entries = {e.name: e for e in pb.scan_playbooks("shop", pack)}

    assert pack.local["buy"].macros.errors["search-own"]
    assert "buy/macros/search-own.yml) is invalid" in (entries["buy"].error or "")


def test_an_inline_body_may_not_take_a_recorded_macros_name() -> None:
    inline = LOCAL_BUY.replace(
        "    macro: search-own\n", "    macro: {steps: [home_screen]}\n"
    ).replace("do: find", "do: search-own")
    root = _write_pack("shop", MANIFEST, buy=inline)
    write_local_macro(root, "buy", "search-own")

    entries = {e.name: e for e in pb.scan_playbooks("shop")}

    assert "already holds — rename one" in (entries["buy"].error or "")


# ---------- optional inputs ----------


def test_an_input_with_a_default_is_optional_and_the_menu_says_so() -> None:
    # The boot's parse prompt tells the model to omit inputs the message
    # does not specify; a default makes that safe, and the menu names it
    # so the model knows which omissions the activation will fill.
    optional = LOCAL_BUY.replace(
        "description: buy with its own recorded search\n",
        "description: buy with its own recorded search\n"
        "inputs:\n  keyword:\n    description: what to buy\n"
        '  qty:\n    description: how many\n    default: "1"\n',
    )
    _write_pack("shop", MANIFEST, buy=optional)
    write_local_macro(paths.playbooks_dir() / "shop", "buy", "search-own")
    entries = activation.discover().entries
    spec, pack = entries["shop/buy"]

    menu = activation.Activation(entries=entries, channel=None)._menu()

    assert "qty (how many; optional, default '1')" in menu
    assert "keyword (what to buy)" in menu
    assert build.resolve_inputs(spec, {"keyword": "water"}) == {
        "keyword": "water",
        "qty": "1",
    }
