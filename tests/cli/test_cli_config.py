"""Tests for `physiclaw.cli.config` — config CLI subcommands."""

from __future__ import annotations

import importlib

import pytest
from typer.testing import CliRunner

from physiclaw import config as _config

config_mod = importlib.import_module("physiclaw.cli.config")
config_app = config_mod.config_app

runner = CliRunner()


# ---------- path ----------


def test_path_prints_config_file_path(mocker) -> None:
    mocker.patch.object(config_mod._config, "config_path", return_value="/tmp/cfg.toml")

    result = runner.invoke(config_app, ["path"])

    assert result.exit_code == 0
    assert "/tmp/cfg.toml" in result.stdout


# ---------- show ----------


def test_show_dumps_toml_with_redacted_secrets(mocker) -> None:
    cfg = _config.Config()
    # Force any *_api_key on provider to a non-empty so the redaction path
    # runs and emits "<redacted>".
    if hasattr(cfg.provider, "anthropic_api_key"):
        cfg = mocker.patch.object(_config, "load", return_value=cfg).return_value
    mocker.patch.object(config_mod._config, "load", return_value=_config.Config())
    mocker.patch.object(config_mod._config, "config_path", return_value="/tmp/cfg.toml")

    result = runner.invoke(config_app, ["show"])

    assert result.exit_code == 0
    # Header line emitted.
    assert "API key values are masked" in result.stdout


def test_show_exits_1_on_config_error(mocker) -> None:
    mocker.patch.object(
        config_mod._config,
        "load",
        side_effect=_config.ConfigError("bad"),
    )

    result = runner.invoke(config_app, ["show"])

    assert result.exit_code == 1
    assert "bad" in result.stderr or "bad" in result.output


# ---------- get ----------


def test_get_prints_value(mocker) -> None:
    mocker.patch.object(config_mod._config, "load", return_value=_config.Config())
    mocker.patch.object(config_mod._config, "get", return_value=42)

    result = runner.invoke(config_app, ["get", "engine.max_turns"])

    assert result.exit_code == 0
    assert "42" in result.stdout


def test_get_renders_bool_as_lowercase(mocker) -> None:
    mocker.patch.object(config_mod._config, "load", return_value=_config.Config())
    mocker.patch.object(config_mod._config, "get", return_value=True)

    result = runner.invoke(config_app, ["get", "some.flag"])

    assert "true" in result.stdout

    mocker.patch.object(config_mod._config, "get", return_value=False)
    result = runner.invoke(config_app, ["get", "some.flag"])
    assert "false" in result.stdout


def test_get_exit_1_on_load_error(mocker) -> None:
    mocker.patch.object(
        config_mod._config,
        "load",
        side_effect=_config.ConfigError("bad"),
    )

    result = runner.invoke(config_app, ["get", "engine.max_turns"])

    assert result.exit_code == 1


def test_get_exit_1_on_unknown_key(mocker) -> None:
    mocker.patch.object(config_mod._config, "load", return_value=_config.Config())
    mocker.patch.object(
        config_mod._config,
        "get",
        side_effect=_config.ConfigError("unknown key"),
    )

    result = runner.invoke(config_app, ["get", "no.such.thing"])

    assert result.exit_code == 1


# ---------- set ----------


def test_set_calls_set_dotted_and_prints_restart_hint(mocker) -> None:
    spy = mocker.patch.object(config_mod._config, "set_dotted")

    result = runner.invoke(config_app, ["set", "engine.max_turns", "10"])

    assert result.exit_code == 0
    spy.assert_called_once_with("engine.max_turns", "10")
    assert "engine.max_turns updated" in result.stdout
    assert "Restart" in result.stdout


def test_set_exit_1_on_invalid_value(mocker) -> None:
    mocker.patch.object(
        config_mod._config,
        "set_dotted",
        side_effect=_config.ConfigError("not an int"),
    )

    result = runner.invoke(config_app, ["set", "engine.max_turns", "abc"])

    assert result.exit_code == 1


# ---------- unset ----------


def test_unset_when_present(mocker) -> None:
    mocker.patch.object(config_mod._config, "unset_dotted", return_value=True)

    result = runner.invoke(config_app, ["unset", "engine.max_turns"])

    assert result.exit_code == 0
    assert "reverted to default" in result.stdout


def test_unset_when_already_default(mocker) -> None:
    mocker.patch.object(config_mod._config, "unset_dotted", return_value=False)

    result = runner.invoke(config_app, ["unset", "engine.max_turns"])

    assert result.exit_code == 0
    assert "already at default" in result.stdout


def test_unset_exit_1_on_error(mocker) -> None:
    mocker.patch.object(
        config_mod._config,
        "unset_dotted",
        side_effect=_config.ConfigError("bad key"),
    )

    result = runner.invoke(config_app, ["unset", "x.y"])

    assert result.exit_code == 1


# ---------- edit ----------


def test_edit_uses_editor_env(mocker, monkeypatch: pytest.MonkeyPatch) -> None:
    mocker.patch.object(
        config_mod._config, "write_default", return_value="/tmp/cfg.toml"
    )
    mocker.patch.object(config_mod._config, "load", return_value=_config.Config())
    monkeypatch.setenv("EDITOR", "myeditor")
    spy = mocker.patch.object(config_mod.subprocess, "run")

    result = runner.invoke(config_app, ["edit"])

    assert result.exit_code == 0
    spy.assert_called_once_with(["myeditor", "/tmp/cfg.toml"], check=False)
    assert "config OK" in result.stdout


def test_edit_falls_back_to_visual(mocker, monkeypatch: pytest.MonkeyPatch) -> None:
    mocker.patch.object(
        config_mod._config, "write_default", return_value="/tmp/cfg.toml"
    )
    mocker.patch.object(config_mod._config, "load", return_value=_config.Config())
    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.setenv("VISUAL", "subl")
    spy = mocker.patch.object(config_mod.subprocess, "run")

    runner.invoke(config_app, ["edit"])

    assert spy.call_args.args[0] == ["subl", "/tmp/cfg.toml"]


def test_edit_falls_back_to_path_search(
    mocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    mocker.patch.object(
        config_mod._config, "write_default", return_value="/tmp/cfg.toml"
    )
    mocker.patch.object(config_mod._config, "load", return_value=_config.Config())
    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.delenv("VISUAL", raising=False)
    mocker.patch.object(
        config_mod.shutil,
        "which",
        side_effect=lambda c: "/usr/bin/vim" if c == "vim" else None,
    )
    spy = mocker.patch.object(config_mod.subprocess, "run")

    runner.invoke(config_app, ["edit"])

    assert spy.call_args.args[0] == ["vim", "/tmp/cfg.toml"]


def test_edit_exit_1_when_no_editor_found(
    mocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    mocker.patch.object(
        config_mod._config, "write_default", return_value="/tmp/cfg.toml"
    )
    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.delenv("VISUAL", raising=False)
    mocker.patch.object(config_mod.shutil, "which", return_value=None)

    result = runner.invoke(config_app, ["edit"])

    assert result.exit_code == 1


def test_edit_validates_after_save(mocker, monkeypatch: pytest.MonkeyPatch) -> None:
    mocker.patch.object(
        config_mod._config, "write_default", return_value="/tmp/cfg.toml"
    )
    monkeypatch.setenv("EDITOR", "ed")
    mocker.patch.object(config_mod.subprocess, "run")
    mocker.patch.object(
        config_mod._config,
        "load",
        side_effect=_config.ConfigError("post-edit invalid"),
    )

    result = runner.invoke(config_app, ["edit"])

    assert result.exit_code == 1
