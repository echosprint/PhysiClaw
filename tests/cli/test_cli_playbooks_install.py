"""Tests for `physiclaw playbooks install` — template packs copied into
the user home VERBATIM (tokens stay in the files), placeholder values
recorded once in `playbooks/placeholders.yml`, disabled state preserved."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from physiclaw.cli import app
from physiclaw.common import paths
from physiclaw.common.placeholders import (
    PLACEHOLDER_VALUES_FILENAME,
    placeholder_values,
)

runner = CliRunner()


def _install(template: Path, *args: str, input: str | None = None):
    return runner.invoke(
        app, ["playbooks", "install", str(template), *args], input=input
    )


PAGES = 'thread:\n  anchors:\n    - {text: "<<CONTACT>>", within: top}\n'
MACRO = 'name: open\ndescription: open the <<CONTACT>> thread\nenabled: false\nsteps:\n  - send_to_clipboard: "<<CONTACT>>"\n'
MANIFEST = (
    "app: channel\ndescription: reach the user thread\n"
    "placeholders:\n  CONTACT:\n    description: the contact\n"
    "    example: SomeOne\n"
    "pages:\n" + "\n".join("  " + line for line in PAGES.splitlines()) + "\n"
)


@pytest.fixture()
def template(tmp_path: Path) -> Path:
    src = tmp_path / "template" / "channel"
    (src / "macros").mkdir(parents=True)
    (src / "macros" / "open.yml").write_text(MACRO, encoding="utf-8")
    (src / "README.md").write_text("# channel\nnotes for people\n", encoding="utf-8")
    (src / "APP.yml").write_text(MANIFEST, encoding="utf-8")
    return src


def test_install_copies_verbatim_and_records_the_value(template: Path) -> None:
    result = _install(template, "--set", "CONTACT=Alice")

    assert result.exit_code == 0, result.output
    assert "reach the user thread" in result.output  # tier-1 description echoed
    dest = paths.playbooks_dir() / "channel"
    # Tokens stay in the installed files — diffable against the template.
    assert "<<CONTACT>>" in (dest / "APP.yml").read_text("utf-8")
    assert (
        (dest / "README.md").read_text("utf-8").startswith("# channel")
    )  # notes travel
    macro = (dest / "macros/open.yml").read_text("utf-8")
    assert "<<CONTACT>>" in macro and "Alice" not in macro
    assert "enabled: false" in macro  # disabled state preserved
    # The value went to the ONE local values file instead.
    assert placeholder_values() == {"CONTACT": "Alice"}
    assert PLACEHOLDER_VALUES_FILENAME in result.output


def test_installed_macro_resolves_from_the_values_file(template: Path) -> None:
    # The end-to-end contract: parsers fill <<TOKEN>>s at load, so the
    # verbatim copy parses to the VALUE.
    from physiclaw.macros import store

    _install(template, "--set", "CONTACT=Alice")

    # Pack-private location — scan the pack's macro root directly.
    entries = store.scan(paths.playbooks_dir() / "channel" / "macros")
    spec = entries[0].spec
    assert spec is not None and spec.steps[0].args == {"text": "Alice"}


def test_install_prompts_for_missing_placeholders(template: Path) -> None:
    result = _install(template, input="Bob\n")

    assert result.exit_code == 0, result.output
    assert "the contact" in result.output  # manifest prose reached the prompt
    assert "SomeOne" in result.output  # and the example
    assert placeholder_values() == {"CONTACT": "Bob"}


def test_install_skips_the_prompt_when_a_value_exists(template: Path) -> None:
    from physiclaw.common.placeholders import write_placeholder_values

    write_placeholder_values({"CONTACT": "Carol"})

    result = _install(template)  # no --set, no stdin — must not prompt

    assert result.exit_code == 0, result.output
    assert placeholder_values() == {"CONTACT": "Carol"}


def test_install_refuses_an_existing_pack_without_force(template: Path) -> None:
    _install(template, "--set", "CONTACT=A")

    result = _install(template, "--set", "CONTACT=B")

    assert result.exit_code != 0
    assert "--force" in result.output


def test_install_force_replaces_and_set_overrides_the_value(template: Path) -> None:
    _install(template, "--set", "CONTACT=A")

    result = _install(template, "--set", "CONTACT=B", "--force")

    assert result.exit_code == 0, result.output
    assert placeholder_values() == {"CONTACT": "B"}


def test_install_rejects_unknown_set_key(template: Path) -> None:
    result = _install(template, "--set", "NOPE=x")

    assert result.exit_code != 0
    assert "NOPE" in result.output


def test_install_rejects_a_placeholder_valued_value(template: Path) -> None:
    result = _install(template, "--set", "CONTACT=<<X>>")

    assert result.exit_code != 0
