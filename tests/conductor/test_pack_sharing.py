"""One pack, many playbooks: what the manifest declares once — pages
with their recover hands, landmarks, placeholders — and the hands the
macros directory records, every playbook file sees. The Taobao shape
before the second task arrives, proved on a fixture pack."""

from __future__ import annotations

import pytest
from conductor_fakes import PACK_MACRO

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
    bbox: [0.35, 0.16, 0.65, 0.24]
pages:
  home:
    anchors: ["Files"]
    recover: {tool: force_quit}
  results:
    anchors: ["综合"]
    recover:
      occluded: {tool: tap, with: landmarks.dismiss}
      elsewhere: {macro: launch}
      limit: 2
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
    recover: {tool: go_back}
  - tell: done
    message: "<<CONTACT>>, your order is on its way"
"""


def _write_pack(app: str, manifest: str, **playbooks: str):
    """A pack the fixtures share: two recorded hands, the manifest, and
    the playbook files given by name."""
    root = paths.playbooks_dir() / app
    for name in ("launch", "search"):
        d = root / "macros" / name
        d.mkdir(parents=True)
        (d / "MACRO.yml").write_text(PACK_MACRO.format(name=name), encoding="utf-8")
    (root / "PLAYBOOK.yml").write_text(manifest, encoding="utf-8")
    for name, text in playbooks.items():
        (root / f"{name}.yml").write_text(text, encoding="utf-8")
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
    assert buy.recovers["results"].occluded == RecoverHand(
        tool="tap", landmark="dismiss"
    )
    assert buy.recovers["results"].elsewhere == RecoverHand(macro="launch")
    assert buy.recovers["results"].limit == 2
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
    (shop / "PLAYBOOK.yml").write_text(
        MANIFEST.replace(
            "elsewhere: {macro: launch}",
            "elsewhere: {macro: {steps: [{name: x, tool: home_screen}]}}",
        ),
        encoding="utf-8",
    )

    entries = pb.scan_playbooks("shop")

    assert all("record the body under macros" in (e.error or "") for e in entries)


def test_manifest_hand_on_an_undeclared_page_is_refused(shop) -> None:
    # A page entry with a hand and no anchors is not a declaration: the
    # pack refuses it as a page, before any route could inherit it.
    (shop / "PLAYBOOK.yml").write_text(
        MANIFEST + "  ghost:\n    recover: {tool: go_back}\n", encoding="utf-8"
    )

    with pytest.raises(PlaybookError, match="ghost"):
        pb.load_pack("shop")


def test_a_page_declared_in_the_manifest_and_a_file_is_refused(shop) -> None:
    (shop / "track.yml").write_text(
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
