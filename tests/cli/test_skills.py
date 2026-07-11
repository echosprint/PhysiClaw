"""Tests for `physiclaw.cli.skills` — install/list/uninstall CLI."""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

skills_mod = importlib.import_module("physiclaw.cli.skills")
skills_app = skills_mod.skills_app

runner = CliRunner()


@pytest.fixture
def fake_home(tmp_path: Path, mocker) -> Path:
    """Redirect skill install root to tmp_path."""
    skills_root = tmp_path / "skills"
    mocker.patch.object(skills_mod.paths, "skills_dir", return_value=skills_root)
    mocker.patch.object(skills_mod.paths, "HOME", tmp_path / "home")
    return skills_root


def _git_completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(
        args=["git"], returncode=returncode, stdout=stdout, stderr=stderr
    )


# ---------- _normalize_source ----------


@pytest.mark.parametrize(
    "inp,expected",
    [
        ("https://github.com/x/y.git", "https://github.com/x/y.git"),
        ("ssh://git@host/x.git", "ssh://git@host/x.git"),
        ("git@github.com:x/y.git", "git@github.com:x/y.git"),
    ],
)
def test_normalize_source_passes_url_forms_through(inp: str, expected: str) -> None:
    assert skills_mod._normalize_source(inp) == expected


def test_normalize_source_expands_user_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    out = skills_mod._normalize_source("~/repos")

    assert out == str(tmp_path / "repos")


def test_normalize_source_passes_absolute_path() -> None:
    assert skills_mod._normalize_source("/tmp/x") == "/tmp/x"


def test_normalize_source_handles_relative_paths() -> None:
    # `./repo` is recognized as a path and Path() collapses leading `./`.
    assert skills_mod._normalize_source("./repo") == "repo"


def test_normalize_source_expands_owner_repo_shorthand() -> None:
    assert skills_mod._normalize_source("foo/bar") == "https://github.com/foo/bar.git"


def test_normalize_source_owner_repo_keeps_existing_dot_git() -> None:
    assert (
        skills_mod._normalize_source("foo/bar.git") == "https://github.com/foo/bar.git"
    )


def test_normalize_source_empty_returns_empty() -> None:
    assert skills_mod._normalize_source("   ") == ""


def test_normalize_source_unrecognized_passes_through() -> None:
    # Not URL, not path, not owner/repo — pass through unchanged.
    assert skills_mod._normalize_source("weird thing") == "weird thing"


# ---------- _validate_name ----------


def test_validate_name_accepts_safe_names() -> None:
    skills_mod._validate_name("jd")
    skills_mod._validate_name("foo-bar")
    skills_mod._validate_name("foo_bar")
    skills_mod._validate_name("foo.v2")


@pytest.mark.parametrize("bad", ["../escape", "/abs", ".dotfile", "has space", ""])
def test_validate_name_rejects_unsafe(bad: str) -> None:
    import typer

    with pytest.raises(typer.Exit):
        skills_mod._validate_name(bad)


# ---------- read_provenance ----------


def test_read_provenance_returns_none_when_missing(tmp_path: Path) -> None:
    assert skills_mod.read_provenance(tmp_path) is None


def test_read_provenance_returns_dict_when_valid(tmp_path: Path) -> None:
    (tmp_path / skills_mod.PROVENANCE_FILE).write_text(
        json.dumps({"source": "x", "sha": "abc"}),
    )

    out = skills_mod.read_provenance(tmp_path)

    assert out == {"source": "x", "sha": "abc"}


def test_read_provenance_returns_none_on_corrupt_json(tmp_path: Path) -> None:
    (tmp_path / skills_mod.PROVENANCE_FILE).write_text("not json")

    assert skills_mod.read_provenance(tmp_path) is None


# ---------- installed_skill_dirs ----------


def test_installed_skill_dirs_empty_when_root_missing(
    tmp_path: Path,
    mocker,
) -> None:
    mocker.patch.object(
        skills_mod.paths,
        "skills_dir",
        return_value=tmp_path / "missing",
    )

    assert skills_mod.installed_skill_dirs() == []


def test_installed_skill_dirs_lists_dirs_skipping_dotfiles(fake_home: Path) -> None:
    fake_home.mkdir(parents=True)
    (fake_home / "alpha").mkdir()
    (fake_home / "beta").mkdir()
    (fake_home / ".hidden").mkdir()
    (fake_home / "not-a-dir.txt").write_text("x")

    out = skills_mod.installed_skill_dirs()

    assert [d.name for d in out] == ["alpha", "beta"]


# ---------- _resolve_source ----------


def test_resolve_source_from_flag_wins(mocker) -> None:
    out = skills_mod._resolve_source("foo/bar")

    assert out == "https://github.com/foo/bar.git"


def test_resolve_source_falls_back_to_config(mocker) -> None:
    cfg = MagicMock()
    cfg.skills.default_source = "alpha/beta"
    mocker.patch.object(skills_mod, "_load_config", return_value=cfg)

    out = skills_mod._resolve_source(None)

    assert out == "https://github.com/alpha/beta.git"


def test_resolve_source_exits_1_when_unconfigured(mocker) -> None:
    cfg = MagicMock()
    cfg.skills.default_source = ""
    mocker.patch.object(skills_mod, "_load_config", return_value=cfg)
    import typer

    with pytest.raises(typer.Exit):
        skills_mod._resolve_source(None)


def test_resolve_source_rejects_path_inside_physiclaw_home(
    tmp_path: Path,
    mocker,
) -> None:
    mocker.patch.object(skills_mod.paths, "HOME", tmp_path)
    inside = tmp_path / "skills-source"
    inside.mkdir()
    import typer

    with pytest.raises(typer.Exit):
        skills_mod._resolve_source(str(inside))


def test_resolve_source_url_skips_home_check(mocker) -> None:
    # URLs bypass the home-check.
    out = skills_mod._resolve_source("https://github.com/x/y.git")

    assert out == "https://github.com/x/y.git"


# ---------- _git ----------


def test_git_runs_command(mocker) -> None:
    spy = mocker.patch.object(
        skills_mod.subprocess,
        "run",
        return_value=_git_completed(stdout="ok"),
    )

    out = skills_mod._git("rev-parse", "HEAD")

    assert out.stdout == "ok"
    assert spy.call_args.args[0] == ["git", "rev-parse", "HEAD"]


def test_git_exits_1_when_not_found(mocker) -> None:
    mocker.patch.object(
        skills_mod.subprocess,
        "run",
        side_effect=FileNotFoundError,
    )
    import typer

    with pytest.raises(typer.Exit):
        skills_mod._git("status")


def test_git_exits_1_on_nonzero(mocker) -> None:
    mocker.patch.object(
        skills_mod.subprocess,
        "run",
        return_value=_git_completed(stderr="bad ref", returncode=1),
    )
    import typer

    with pytest.raises(typer.Exit):
        skills_mod._git("clone", "x")


# ---------- _read_skill_name ----------


def test_read_skill_name_no_frontmatter_uses_fallback(tmp_path: Path) -> None:
    md = tmp_path / "SKILL.md"
    md.write_text("# Title\n\nbody")

    assert skills_mod._read_skill_name(md, "dirname") == "dirname"


def test_read_skill_name_extracts_from_frontmatter(tmp_path: Path) -> None:
    md = tmp_path / "SKILL.md"
    md.write_text("---\nname: actual\nother: x\n---\n\nbody")

    assert skills_mod._read_skill_name(md, "dirname") == "actual"


def test_read_skill_name_strips_quotes(tmp_path: Path) -> None:
    md = tmp_path / "SKILL.md"
    md.write_text('---\nname: "with-quotes"\n---\n')

    assert skills_mod._read_skill_name(md, "fallback") == "with-quotes"


def test_read_skill_name_unclosed_frontmatter(tmp_path: Path) -> None:
    md = tmp_path / "SKILL.md"
    md.write_text("---\nname: x")  # no closing ---

    assert skills_mod._read_skill_name(md, "fb") == "fb"


# ---------- list ----------


def test_list_empty_skills_dir(fake_home: Path) -> None:
    result = runner.invoke(skills_app, ["list"])

    assert result.exit_code == 0
    assert "no skills installed" in result.output


def test_list_with_provenance_and_local(fake_home: Path) -> None:
    fake_home.mkdir(parents=True)
    a = fake_home / "alpha"
    a.mkdir()
    (a / skills_mod.PROVENANCE_FILE).write_text(
        json.dumps({"source": "x/y", "ref": "main", "sha": "abcdef0"})
    )
    (fake_home / "beta").mkdir()  # no provenance

    result = runner.invoke(skills_app, ["list"])

    assert result.exit_code == 0
    assert "alpha" in result.output
    assert "x/y" in result.output
    assert "beta" in result.output
    assert "no .installed-from" in result.output


def test_list_uses_sha_when_no_ref(fake_home: Path) -> None:
    fake_home.mkdir(parents=True)
    a = fake_home / "alpha"
    a.mkdir()
    (a / skills_mod.PROVENANCE_FILE).write_text(
        json.dumps({"source": "x/y", "sha": "abcdef01234"})
    )

    result = runner.invoke(skills_app, ["list"])

    assert "abcdef0" in result.output


# ---------- uninstall ----------


def test_uninstall_missing_skill(fake_home: Path) -> None:
    result = runner.invoke(skills_app, ["uninstall", "ghost"])

    assert result.exit_code == 1
    assert "is not installed" in result.output


def test_uninstall_user_authored_without_force_refuses(fake_home: Path) -> None:
    fake_home.mkdir(parents=True)
    (fake_home / "myskill").mkdir()

    result = runner.invoke(skills_app, ["uninstall", "myskill"])

    assert result.exit_code == 1
    assert "user-authored" in result.output
    assert (fake_home / "myskill").exists()


def test_uninstall_user_authored_with_force_removes(fake_home: Path) -> None:
    fake_home.mkdir(parents=True)
    (fake_home / "myskill").mkdir()

    result = runner.invoke(skills_app, ["uninstall", "myskill", "--force"])

    assert result.exit_code == 0
    assert not (fake_home / "myskill").exists()
    assert "no provenance marker" in result.output


def test_uninstall_provenanced_skill_removes(fake_home: Path) -> None:
    fake_home.mkdir(parents=True)
    skill = fake_home / "alpha"
    skill.mkdir()
    (skill / skills_mod.PROVENANCE_FILE).write_text(json.dumps({"source": "x"}))

    result = runner.invoke(skills_app, ["uninstall", "alpha"])

    assert result.exit_code == 0
    assert not skill.exists()


def test_uninstall_validates_name(fake_home: Path) -> None:
    result = runner.invoke(skills_app, ["uninstall", "../escape"])

    assert result.exit_code == 1


# ---------- install ----------


def _populate_clone(
    clone_dir: Path,
    *,
    name: str = "myskill",
    declared_name: str | None = None,
    skill_md: bool = True,
) -> None:
    """Build a fake cloned repo at `clone_dir` with skills/<name>/."""
    sk = clone_dir / "skills" / name
    sk.mkdir(parents=True)
    if skill_md:
        if declared_name is not None:
            (sk / "SKILL.md").write_text(f"---\nname: {declared_name}\n---\n")
        else:
            (sk / "SKILL.md").write_text("# Title\nbody\n")


def _stub_clone(mocker, populate):
    """Stub `_clone` to populate the clone dir + return a dummy sha."""

    def _fake_clone(source, ref, dest):
        populate(dest)
        return "abcdef0123456789"

    mocker.patch.object(skills_mod, "_clone", side_effect=_fake_clone)


def test_install_happy_path(fake_home: Path, mocker) -> None:
    _stub_clone(mocker, lambda d: _populate_clone(d, name="myskill"))

    result = runner.invoke(
        skills_app,
        ["install", "myskill", "--from", "owner/repo"],
    )

    assert result.exit_code == 0
    assert (fake_home / "myskill" / "SKILL.md").exists()
    prov = json.loads((fake_home / "myskill" / skills_mod.PROVENANCE_FILE).read_text())
    assert prov["source"] == "https://github.com/owner/repo.git"
    assert prov["sha"].startswith("abcdef")


def test_install_invalid_name_exit_1(fake_home: Path) -> None:
    result = runner.invoke(skills_app, ["install", "../bad", "--from", "o/r"])

    assert result.exit_code == 1


def test_install_existing_provenanced_without_force(
    fake_home: Path,
    mocker,
) -> None:
    fake_home.mkdir(parents=True)
    target = fake_home / "myskill"
    target.mkdir()
    (target / skills_mod.PROVENANCE_FILE).write_text(
        json.dumps({"source": "x/y", "ref": "main"})
    )

    result = runner.invoke(skills_app, ["install", "myskill", "--from", "o/r"])

    assert result.exit_code == 1
    assert "already installed" in result.output


def test_install_existing_user_authored_without_force(
    fake_home: Path,
    mocker,
) -> None:
    fake_home.mkdir(parents=True)
    target = fake_home / "myskill"
    target.mkdir()

    result = runner.invoke(skills_app, ["install", "myskill", "--from", "o/r"])

    assert result.exit_code == 1
    assert "user-authored" in result.output


def test_install_no_skill_in_source(fake_home: Path, mocker) -> None:
    _stub_clone(mocker, lambda d: None)

    result = runner.invoke(
        skills_app,
        ["install", "missing", "--from", "owner/repo"],
    )

    assert result.exit_code == 1
    assert "does not contain skills/missing" in result.output


def test_install_missing_skill_md(fake_home: Path, mocker) -> None:
    _stub_clone(
        mocker,
        lambda d: _populate_clone(d, name="myskill", skill_md=False),
    )

    result = runner.invoke(
        skills_app,
        ["install", "myskill", "--from", "owner/repo"],
    )

    assert result.exit_code == 1
    assert "missing SKILL.md" in result.output


def test_install_declared_name_mismatch(fake_home: Path, mocker) -> None:
    _stub_clone(
        mocker,
        lambda d: _populate_clone(d, name="myskill", declared_name="other"),
    )

    result = runner.invoke(
        skills_app,
        ["install", "myskill", "--from", "owner/repo"],
    )

    assert result.exit_code == 1
    assert "name='other'" in result.output or 'name="other"' in result.output


def test_install_with_ref_flag(fake_home: Path, mocker) -> None:
    refs_seen = []

    def _fake_clone(source, ref, dest):
        refs_seen.append(ref)
        _populate_clone(dest, name="myskill")
        return "abcdef0"

    mocker.patch.object(skills_mod, "_clone", side_effect=_fake_clone)

    result = runner.invoke(
        skills_app,
        ["install", "myskill", "--from", "owner/repo", "--ref", "v1.0"],
    )

    assert result.exit_code == 0
    assert refs_seen == ["v1.0"]


def test_install_force_overwrites_existing(fake_home: Path, mocker) -> None:
    fake_home.mkdir(parents=True)
    target = fake_home / "myskill"
    target.mkdir()
    (target / "OLD").write_text("old")
    (target / skills_mod.PROVENANCE_FILE).write_text(
        json.dumps({"source": "old", "ref": "main"})
    )

    _stub_clone(mocker, lambda d: _populate_clone(d, name="myskill"))

    result = runner.invoke(
        skills_app,
        ["install", "myskill", "--from", "owner/repo", "--force"],
    )

    assert result.exit_code == 0
    assert not (target / "OLD").exists()
    assert (target / "SKILL.md").exists()


# ---------- _clone ----------


def test_clone_invokes_git_with_depth_and_branch(mocker, tmp_path: Path) -> None:
    spy = mocker.patch.object(
        skills_mod,
        "_git",
        side_effect=[
            _git_completed(),  # clone
            _git_completed(stdout="abc123\n"),  # rev-parse
        ],
    )

    sha = skills_mod._clone("o/r", "v1", tmp_path / "dest")

    assert sha == "abc123"
    clone_call = spy.call_args_list[0]
    assert "--depth=1" in clone_call.args
    assert "--branch" in clone_call.args


def test_clone_no_branch_when_ref_omitted(mocker, tmp_path: Path) -> None:
    spy = mocker.patch.object(
        skills_mod,
        "_git",
        side_effect=[
            _git_completed(),
            _git_completed(stdout="def456\n"),
        ],
    )

    skills_mod._clone("o/r", None, tmp_path / "dest")

    assert "--branch" not in spy.call_args_list[0].args
