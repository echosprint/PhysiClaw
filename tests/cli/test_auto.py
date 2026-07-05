"""Tests for `physiclaw.cli.auto` — the hands-free setup command.

`auto` prompts to bring the phone up, then starts the server with the
internal `auto_calibrate` flag set — the server's own tested branch drives
the rest. We assert the prompt/confirm gate and the delegation here.
"""
from __future__ import annotations

import importlib

import pytest
import typer

auto_mod = importlib.import_module("physiclaw.cli.auto")


def _patch_urls(mocker, primary="http://mac.local:9000", fallback="http://1.2.3.4:9000"):
    return mocker.patch(
        "physiclaw.core.bridge.bridge_base_urls", return_value=(primary, fallback),
    )


def test_auto_prompts_then_invokes_server_with_auto_calibrate(mocker) -> None:
    _patch_urls(mocker)
    server_spy = mocker.patch.object(auto_mod, "server")
    input_spy = mocker.patch("builtins.input", return_value="")

    auto_mod.auto(port=9000, host="127.0.0.1", verbose=True)

    # Operator is asked to confirm the phone is up BEFORE the server starts.
    input_spy.assert_called_once()
    server_spy.assert_called_once_with(
        port=9000, host="127.0.0.1", verbose=True, auto_calibrate=True,
    )


@pytest.mark.parametrize("ans", ["", "y", "Y", "yes", "  ", " y "])
def test_auto_proceeds_on_affirmative_answer(mocker, ans) -> None:
    _patch_urls(mocker)
    server_spy = mocker.patch.object(auto_mod, "server")
    mocker.patch("builtins.input", return_value=ans)

    auto_mod.auto(port=9000, host="127.0.0.1", verbose=True)

    server_spy.assert_called_once()


@pytest.mark.parametrize("ans", ["n", "no", "q", "wait", "later"])
def test_auto_exits_without_starting_server_on_decline(mocker, ans) -> None:
    _patch_urls(mocker)
    server_spy = mocker.patch.object(auto_mod, "server")
    mocker.patch("builtins.input", return_value=ans)

    with pytest.raises(typer.Exit):
        auto_mod.auto(port=9000, host="127.0.0.1", verbose=True)

    server_spy.assert_not_called()


def test_auto_shows_both_mdns_and_ip_urls(mocker, capsys) -> None:
    _patch_urls(mocker, "http://mac.local:9000", "http://1.2.3.4:9000")
    mocker.patch.object(auto_mod, "server")
    mocker.patch("builtins.input", return_value="")

    auto_mod.auto(port=9000, host="127.0.0.1", verbose=False)

    out = capsys.readouterr().out
    assert "http://mac.local:9000/bridge" in out
    assert "http://1.2.3.4:9000/bridge" in out


def test_auto_shows_single_url_when_no_mdns(mocker, capsys) -> None:
    # No mDNS name → primary == fallback → show one line, not a duplicate.
    _patch_urls(mocker, "http://1.2.3.4:9000", "http://1.2.3.4:9000")
    mocker.patch.object(auto_mod, "server")
    mocker.patch("builtins.input", return_value="")

    auto_mod.auto(port=9000, host="127.0.0.1", verbose=False)

    out = capsys.readouterr().out
    assert out.count("http://1.2.3.4:9000/bridge") == 1


def test_auto_proceeds_when_stdin_closed(mocker) -> None:
    # Non-interactive stdin (service/pipe) → the confirm raises EOFError;
    # start the server anyway rather than crashing.
    _patch_urls(mocker)
    server_spy = mocker.patch.object(auto_mod, "server")
    mocker.patch("builtins.input", side_effect=EOFError)

    auto_mod.auto(port=9000, host="127.0.0.1", verbose=True)

    server_spy.assert_called_once_with(
        port=9000, host="127.0.0.1", verbose=True, auto_calibrate=True,
    )
