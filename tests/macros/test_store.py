"""Tests for `physiclaw.macros.store` — discovery under
``~/.physiclaw/macros/`` and the `## Available Macros` prompt section.

All paths derive from `paths.HOME`, repointed per test by the autouse
`physiclaw_home` fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from physiclaw.common import paths
from physiclaw.common.config import CONFIG
from physiclaw.macros import store as store_mod
from physiclaw.macros.model import MacroError
from physiclaw.macros.scaffold import README_CONTENT
from physiclaw.macros.store import (
    discover_enabled,
    ensure_format_readme,
    init_macro,
    list_dir_names,
    render_section,
    scan,
)

VALID = """
name: {name}
description: Demo macro
enabled: {enabled}

inputs:
  message:
    description: The message text
    example: hello
  greeting:
    description: Opening line
    default: hi

steps:
  - name: home-screen-1
    tool: home_screen
  - name: send-to-clipboard-1
    tool: send_to_clipboard
    with:
      text: "{{message}}"
"""


def _write(name: str, text: str | None = None, *, enabled: bool = True) -> Path:
    d = paths.macros_dir() / name
    d.mkdir(parents=True)
    body = (
        text
        if text is not None
        else VALID.format(name=name, enabled=str(enabled).lower())
    )
    (d / "MACRO.yml").write_text(body, encoding="utf-8")
    return d


# ---------- scan ----------


def test_scan_missing_root_returns_empty() -> None:
    assert scan() == []


def test_scan_valid_macro_returns_spec() -> None:
    _write("demo")

    entries = scan()

    assert len(entries) == 1
    assert entries[0].spec is not None
    assert entries[0].spec.name == "demo"
    assert entries[0].error is None


def test_scan_invalid_yaml_keeps_reason() -> None:
    _write("broken", text="steps: [unclosed")

    entries = scan()

    assert entries[0].spec is None
    assert "invalid YAML" in (entries[0].error or "")


def test_scan_dir_without_macro_yml_keeps_reason() -> None:
    (paths.macros_dir() / "empty").mkdir(parents=True)

    entries = scan()

    assert entries[0].error == "no MACRO.yml"


def test_scan_skips_hidden_and_underscore_dirs() -> None:
    _write("demo")
    (paths.macros_dir() / "_draft").mkdir()
    (paths.macros_dir() / ".git").mkdir()

    entries = scan()

    assert [e.dir_name for e in entries] == ["demo"]


def test_scan_ignores_stats_file_at_root() -> None:
    _write("demo")
    (paths.macros_dir() / "stats.json").write_text("{}", encoding="utf-8")

    entries = scan()

    assert [e.dir_name for e in entries] == ["demo"]


def test_scan_sorted_by_dir_name() -> None:
    _write("zeta")
    _write("alpha")

    entries = scan()

    assert [e.dir_name for e in entries] == ["alpha", "zeta"]


def test_scan_rejects_symlink_escaping_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside-macro"
    outside.mkdir()
    (outside / "MACRO.yml").write_text(
        VALID.format(name="escape", enabled="true"), encoding="utf-8"
    )
    paths.macros_dir().mkdir(parents=True)
    (paths.macros_dir() / "escape").symlink_to(outside)

    entries = scan()

    assert entries[0].spec is None
    assert "outside" in (entries[0].error or "")


# ---------- list_dir_names ----------


def test_list_dir_names_missing_root_returns_empty() -> None:
    assert list_dir_names() == set()


def test_list_dir_names_is_identity_not_validity() -> None:
    # The stats prune set: a dir counts if MACRO.yml exists, valid or not —
    # no read, no parse.
    _write("demo")
    _write("broken", text="steps: [not yaml")
    (paths.macros_dir() / "empty").mkdir()
    (paths.macros_dir() / "_draft").mkdir()

    assert list_dir_names() == {"demo", "broken"}


# ---------- init_macro / ensure_format_readme ----------


def test_init_macro_scaffold_parses_and_is_disabled() -> None:
    path = init_macro("my-macro")

    entries = scan()

    assert path.exists()
    assert entries[0].spec is not None  # the template passes check unedited
    assert entries[0].spec.enabled is False
    assert entries[0].spec.name == "my-macro"


def test_init_macro_rejects_existing_directory() -> None:
    init_macro("my-macro")

    with pytest.raises(MacroError, match="already exists"):
        init_macro("my-macro")


def test_init_macro_rejects_bad_name() -> None:
    with pytest.raises(MacroError, match="lowercase"):
        init_macro("Bad_Name")


def test_init_macro_writes_the_format_readme() -> None:
    init_macro("my-macro")

    assert (paths.macros_dir() / "README.md").read_text(
        encoding="utf-8"
    ) == README_CONTENT


def test_ensure_format_readme_rewrites_stale_content() -> None:
    ensure_format_readme()
    (paths.macros_dir() / "README.md").write_text("old spec", encoding="utf-8")

    ensure_format_readme()

    assert (paths.macros_dir() / "README.md").read_text(
        encoding="utf-8"
    ) == README_CONTENT


# ---------- discover_enabled ----------


def test_discover_enabled_filters_disabled_and_invalid() -> None:
    _write("on", enabled=True)
    _write("off", enabled=False)
    _write("bad", text="steps: [not yaml")

    macros = discover_enabled()

    assert set(macros) == {"on"}


def test_discover_enabled_empty_when_config_gate_off(mocker) -> None:
    # [macros] enabled = false hides every macro from the agent, regardless
    # of the per-file flags — the fleet-wide switch.
    _write("on", enabled=True)
    mocker.patch.object(CONFIG.macros, "enabled", False)

    assert discover_enabled() == {}


def test_scan_ignores_the_config_gate(mocker) -> None:
    # Authoring keeps working while the fleet-wide switch is off — the CLI
    # (list/check/run) reads through scan(), never through the gate.
    _write("on", enabled=True)
    mocker.patch.object(CONFIG.macros, "enabled", False)

    assert scan()[0].spec is not None


# ---------- render_section ----------


def test_render_section_empty_when_no_macros() -> None:
    assert render_section({}) == ""


def test_render_section_lists_name_description_and_step_count() -> None:
    _write("demo")

    section = render_section(discover_enabled())

    assert section.startswith("## Available Macros")
    assert "- **demo** — Demo macro" in section
    # The step list IS the count, and every entry is a valid `start_at`.
    assert "  - steps: home-screen-1, send-to-clipboard-1" in section


def test_render_section_lists_inputs_with_required_and_default() -> None:
    _write("demo")

    section = render_section(discover_enabled())

    assert "- `message` (required): The message text (example: `hello`)" in section
    assert "- `greeting` (default: `hi`): Opening line" in section


def test_render_section_mentions_run_macro_tool() -> None:
    _write("demo")

    section = render_section(discover_enabled())

    assert "run_macro" in section


def test_a_file_the_loader_cannot_survive_is_excluded_not_fatal(mocker) -> None:
    # The loader does not confine itself to YAMLError — deep nesting surfaces
    # as RecursionError, which used to escape scan() -> discover_enabled() ->
    # build_prompt_bundle and STUCK every wake over ONE bad file. The
    # contract is that any unparseable file is excluded whole.
    #
    # The failure is injected rather than provoked with deeply nested YAML:
    # whether real nesting trips the limit depends on stack depth already
    # consumed, so a literal bomb passes alone and fails inside the suite.
    # What needs pinning is the handling, not the trigger.
    _write("demo")
    _write("boom")
    real = store_mod.parse_macro

    def _explode(text: str, dir_name: str):
        if dir_name == "boom":
            raise RecursionError("maximum recursion depth exceeded")
        return real(text, dir_name)

    mocker.patch.object(store_mod, "parse_macro", side_effect=_explode)

    entries = {e.dir_name: e for e in scan()}

    assert entries["boom"].spec is None
    assert entries["boom"].error  # reason kept, so `macros check` can name it
    assert entries["demo"].spec is not None  # a healthy neighbour still loads
    assert "demo" in discover_enabled()


def test_scan_still_lets_keyboard_interrupt_through(mocker) -> None:
    # The broad `except Exception` must not swallow BaseException: a Ctrl-C
    # or a cancelled engine task has to keep propagating.
    _write("demo")
    mocker.patch.object(store_mod, "parse_macro", side_effect=KeyboardInterrupt)

    with pytest.raises(KeyboardInterrupt):
        scan()
