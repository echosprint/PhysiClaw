"""Contract for the repo's shared template packs (`playbooks/`).

Templates are what other users install; this pins the promises the
README makes: every placeholder is documented in the pack's manifest,
substitution yields a pack the real parsers accept whole, and
everything ships disabled (rehearse-then-enable — a fresh install must
never be able to drive a phone or spend money).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from physiclaw.common.placeholders import find_placeholders
from physiclaw.common.text import read_text

TEMPLATES_ROOT = Path(__file__).resolve().parent.parent / "playbooks"

PACKS = sorted(d.name for d in TEMPLATES_ROOT.iterdir() if d.is_dir())


def _pack_files(app: str) -> list[Path]:
    """What the installer substitutes: every yml, manifest included."""
    return sorted((TEMPLATES_ROOT / app).rglob("*.yml"))


def _manifest(app: str) -> dict:
    from ruamel.yaml import YAML

    return YAML(typ="safe", pure=True).load(
        read_text(TEMPLATES_ROOT / app / "PLAYBOOK.yml")
    )


def test_templates_exist() -> None:
    assert "taobao" in PACKS and "channel" in PACKS


@pytest.mark.parametrize("app", PACKS)
def test_manifest_is_the_canonical_entry(app: str) -> None:
    # The `action.yml` contract: one manifest, `app` matching the
    # folder, a description that says what AND when.
    meta = _manifest(app)

    assert meta.get("app") == app
    assert meta.get("description", "").strip(), "description is the tier-1 index"


@pytest.mark.parametrize("app", PACKS)
def test_every_placeholder_is_documented(app: str) -> None:
    documented = set(_manifest(app).get("placeholders") or {})

    used = {t for f in _pack_files(app) for t in find_placeholders(read_text(f))}

    assert used <= documented, f"undocumented placeholder(s): {used - documented}"


@pytest.mark.parametrize("app", PACKS)
def test_installed_pack_parses_whole_and_ships_disabled(app: str) -> None:
    # Through the REAL installer over the REAL template — values into
    # placeholders.yml, then the real pack machinery: pages, macros,
    # and playbooks must all validate (tokens filled at load), and
    # nothing may be enabled.
    from typer.testing import CliRunner

    from physiclaw.cli import app as cli_app
    from physiclaw.conductor.spec.channel import load_channel
    from physiclaw.conductor.spec.conventions import BOOT_PLAYBOOK, CHANNEL_APP
    from physiclaw.conductor.spec.pack import load_pack, scan_playbooks

    tokens = {t for f in _pack_files(app) for t in find_placeholders(read_text(f))}
    sets = [x for tok in tokens for x in ("--set", f"{tok}=TestUser")]
    result = CliRunner().invoke(
        cli_app, ["playbooks", "install", str(TEMPLATES_ROOT / app), *sets]
    )
    assert result.exit_code == 0, result.output

    pack = load_pack(app)  # raises on an invalid pages.yml
    assert not pack.macro_errors, pack.macro_errors
    assert all(not m.enabled for m in pack.macros.values()), "template macro enabled"
    entries = scan_playbooks(app, pack)
    invalid = [e.name for e in entries if e.spec is None]
    assert not invalid, f"invalid playbook(s): {invalid}"
    # The channel's boot ships enabled on purpose: its gate is the
    # `open` hand it names, which ships disabled — so it is valid, not
    # live, exactly like every other template playbook.
    assert all(
        not e.spec.enabled for e in entries if e.spec and e.name != BOOT_PLAYBOOK
    ), "template playbook enabled"
    if app == CHANNEL_APP:
        ch = load_channel()
        assert ch is not None and ch.boot is None, "template boot is live"
