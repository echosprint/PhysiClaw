"""Tests for `physiclaw.agent.claude.plugin`."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest

from physiclaw.agent.claude import plugin
from physiclaw.agent.claude.plugin import (
    _has_context_pollution,
    _link_or_copy,
    prepare_plugin_dir,
)
from physiclaw.agent.engine.skill import Skill


def _make_skill_dir(parent: Path, name: str, *, body: str = "Skill body") -> Path:
    d = parent / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(body)
    return d


def _skill(name: str, dir_: Path) -> Skill:
    return Skill(name=name, description=f"desc {name}", body=f"body {name}", dir=dir_)


# ---------- _has_context_pollution ----------


def test_has_context_pollution_clean_dir(tmp_path: Path) -> None:
    d = _make_skill_dir(tmp_path, "clean")

    assert _has_context_pollution(d) is False


def test_has_context_pollution_detects_claude_md(tmp_path: Path) -> None:
    d = _make_skill_dir(tmp_path, "polluted")
    (d / "CLAUDE.md").write_text("doctrine")

    assert _has_context_pollution(d) is True


def test_has_context_pollution_detects_dot_claude_dir(tmp_path: Path) -> None:
    d = _make_skill_dir(tmp_path, "polluted")
    (d / ".claude").mkdir()

    assert _has_context_pollution(d) is True


def test_has_context_pollution_returns_true_when_both_present(tmp_path: Path) -> None:
    d = _make_skill_dir(tmp_path, "both")
    (d / "CLAUDE.md").write_text("x")
    (d / ".claude").mkdir()

    assert _has_context_pollution(d) is True


def test_has_context_pollution_dot_claude_must_be_dir(tmp_path: Path) -> None:
    d = _make_skill_dir(tmp_path, "file-not-dir")
    (d / ".claude").write_text("not a dir")

    # File named `.claude` (not a directory) should not trigger.
    assert _has_context_pollution(d) is False


# ---------- _link_or_copy ----------


def test_link_or_copy_creates_symlink(tmp_path: Path) -> None:
    src = _make_skill_dir(tmp_path, "src")
    dst = tmp_path / "dst"

    _link_or_copy(src, dst)

    assert dst.is_symlink()
    assert dst.resolve() == src.resolve()


def test_link_or_copy_falls_back_to_copytree_on_oserror(
    tmp_path: Path, mocker
) -> None:
    src = _make_skill_dir(tmp_path, "src")
    dst = tmp_path / "dst"

    # Make symlink_to fail; copytree should be used as fallback.
    real_copytree = shutil.copytree
    spy = mocker.patch.object(plugin.shutil, "copytree", side_effect=real_copytree)
    mocker.patch.object(Path, "symlink_to", side_effect=OSError("no symlinks"))

    _link_or_copy(src, dst)

    assert dst.is_dir()
    assert (dst / "SKILL.md").exists()
    assert spy.called


# ---------- prepare_plugin_dir ----------


@pytest.fixture
def _isolate_claude_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the module's claude-only skills dir to a tmp path."""
    co_dir = tmp_path / "_claude_only"
    co_dir.mkdir()
    monkeypatch.setattr(plugin, "_CLAUDE_ONLY_SKILLS_DIR", co_dir)
    return co_dir


def test_prepare_plugin_dir_writes_plugin_json_with_session_name(
    _isolate_claude_only: Path,
) -> None:
    root = prepare_plugin_dir("sid-abc", skills={})

    try:
        meta = root / ".claude-plugin" / "plugin.json"
        assert meta.exists()
        data = json.loads(meta.read_text())
        assert data["name"] == "physiclaw-agent-sid-abc"
        assert data["skills"] == "./skills"
        assert (root / "skills").is_dir()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_prepare_plugin_dir_links_user_skills(
    tmp_path: Path, _isolate_claude_only: Path,
) -> None:
    user_skills_root = tmp_path / "user_skills"
    sk_dir = _make_skill_dir(user_skills_root, "wechat")
    skills = {"wechat": _skill("wechat", sk_dir)}

    root = prepare_plugin_dir("s1", skills=skills)

    try:
        target = root / "skills" / "wechat"
        assert target.exists()
        assert (target / "SKILL.md").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_prepare_plugin_dir_materializes_flat_builtin_skill(
    tmp_path: Path, _isolate_claude_only: Path,
) -> None:
    # A built-in flat skill has dir = the shared root (no per-skill SKILL.md);
    # the plugin must write skills/<name>/SKILL.md from the snapshot.
    builtin_root = tmp_path / "built-in"
    builtin_root.mkdir()
    sk = Skill(
        name="im", description="messaging", body="im steps",
        dir=builtin_root, flat=True,
    )

    root = prepare_plugin_dir("sflat", skills={"im": sk})

    try:
        md = root / "skills" / "im" / "SKILL.md"
        assert md.exists()
        text = md.read_text()
        assert "name: im" in text
        assert "description: messaging" in text
        assert "im steps" in text
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_prepare_plugin_dir_links_claude_only_skills(
    _isolate_claude_only: Path,
) -> None:
    _make_skill_dir(_isolate_claude_only, "jobs")

    root = prepare_plugin_dir("s2", skills={})

    try:
        assert (root / "skills" / "jobs" / "SKILL.md").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_prepare_plugin_dir_claude_only_wins_on_collision(
    tmp_path: Path, _isolate_claude_only: Path,
) -> None:
    # Claude-only "jobs" with body "claude"
    co_jobs = _make_skill_dir(_isolate_claude_only, "jobs", body="claude version")
    # User skill also named "jobs"
    user_root = tmp_path / "user"
    user_jobs = _make_skill_dir(user_root, "jobs", body="user version")
    skills = {"jobs": _skill("jobs", user_jobs)}

    root = prepare_plugin_dir("s3", skills=skills)

    try:
        linked = root / "skills" / "jobs"
        # Should resolve to claude-only dir, not user dir.
        assert linked.resolve() == co_jobs.resolve()
        assert "claude version" in (linked / "SKILL.md").read_text()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_prepare_plugin_dir_skips_dirs_without_skill_md(
    _isolate_claude_only: Path,
) -> None:
    # A directory but no SKILL.md inside.
    bad = _isolate_claude_only / "no-skill-md"
    bad.mkdir()
    (bad / "README.md").write_text("not a skill")

    root = prepare_plugin_dir("s4", skills={})

    try:
        assert not (root / "skills" / "no-skill-md").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_prepare_plugin_dir_skips_files_in_claude_only_root(
    _isolate_claude_only: Path,
) -> None:
    # A stray file at the top-level skills dir — must be silently skipped.
    (_isolate_claude_only / "README.md").write_text("not a dir")

    root = prepare_plugin_dir("s5", skills={})

    try:
        assert (root / "skills").is_dir()
        # No errors raised.
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_prepare_plugin_dir_skips_polluted_claude_only_skill_and_logs(
    _isolate_claude_only: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    polluted = _make_skill_dir(_isolate_claude_only, "polluted")
    (polluted / "CLAUDE.md").write_text("stray")

    with caplog.at_level(logging.WARNING, logger="physiclaw.agent.claude.plugin"):
        root = prepare_plugin_dir("s6", skills={})

    try:
        assert not (root / "skills" / "polluted").exists()
        assert any(
            "polluted" in r.getMessage() and "skipped" in r.getMessage()
            for r in caplog.records
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_prepare_plugin_dir_skips_polluted_user_skill(
    tmp_path: Path, _isolate_claude_only: Path,
) -> None:
    user_root = tmp_path / "user"
    sk_dir = _make_skill_dir(user_root, "bad")
    (sk_dir / ".claude").mkdir()
    skills = {"bad": _skill("bad", sk_dir)}

    root = prepare_plugin_dir("s7", skills=skills)

    try:
        assert not (root / "skills" / "bad").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_prepare_plugin_dir_calls_discover_when_skills_none(
    _isolate_claude_only: Path, mocker, tmp_path: Path,
) -> None:
    user_root = tmp_path / "user"
    sk_dir = _make_skill_dir(user_root, "auto")
    spy = mocker.patch.object(
        plugin.skill, "discover",
        return_value={"auto": _skill("auto", sk_dir)},
    )

    root = prepare_plugin_dir("s8")

    try:
        assert spy.called
        assert (root / "skills" / "auto").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_prepare_plugin_dir_handles_missing_claude_only_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Point at a path that doesn't exist.
    monkeypatch.setattr(
        plugin, "_CLAUDE_ONLY_SKILLS_DIR", tmp_path / "does-not-exist"
    )

    root = prepare_plugin_dir("s9", skills={})

    try:
        # Empty skills dir but plugin layout still complete.
        assert (root / "skills").is_dir()
        assert (root / ".claude-plugin" / "plugin.json").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_prepare_plugin_dir_logs_info_with_linked_skills(
    tmp_path: Path, _isolate_claude_only: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    _make_skill_dir(_isolate_claude_only, "jobs")
    sk_dir = _make_skill_dir(tmp_path / "user", "wechat")
    skills = {"wechat": _skill("wechat", sk_dir)}

    with caplog.at_level(logging.INFO, logger="physiclaw.agent.claude.plugin"):
        root = prepare_plugin_dir("sX", skills=skills)

    try:
        info = [r for r in caplog.records if r.levelno == logging.INFO]
        assert info, "expected an info log line"
        msg = info[-1].getMessage()
        assert "claude plugin dir" in msg
        assert "jobs" in msg
        assert "wechat" in msg
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_prepare_plugin_dir_session_id_in_dir_prefix(
    _isolate_claude_only: Path,
) -> None:
    root = prepare_plugin_dir("session42", skills={})

    try:
        assert "physiclaw-plugin-session42-" in root.name
    finally:
        shutil.rmtree(root, ignore_errors=True)
