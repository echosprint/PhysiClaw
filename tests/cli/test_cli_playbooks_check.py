"""Tests for `physiclaw playbooks check` — the pack report: the folder
layout's strays and the prompt files no route reads are named, so an
author never wonders why a file on disk does nothing."""

from __future__ import annotations

from typer.testing import CliRunner

from physiclaw.cli import app

runner = CliRunner()


def _check():
    return runner.invoke(app, ["playbooks", "check"])


def test_check_names_prompt_files_no_route_reads() -> None:
    from conductor_fakes import FLOW, write_pack, write_prompt

    root = write_pack(playbooks={"flow": FLOW})
    write_prompt(root, None, "shared", "unused")
    write_prompt(root, "flow", "own", "unused")

    result = _check()

    assert result.exit_code == 0, result.output
    assert "demo/prompts/shared.md is read by no playbook" in result.output
    assert "demo/flow/prompts/own.md is read by no step of flow" in result.output


def test_check_fails_on_a_broken_prompt_file_and_a_stray_route() -> None:
    from conductor_fakes import FLOW, write_pack, write_prompt

    root = write_pack(playbooks={"flow": FLOW})
    write_prompt(root, "flow", "pick", "")
    (root / "old.yml").write_text("name: old\n" + FLOW, encoding="utf-8")

    result = _check()

    assert result.exit_code == 1
    assert "demo/flow/prompts/pick.md: the prompt file is empty" in result.output
    assert "move it to old/PLAYBOOK.yml" in result.output
