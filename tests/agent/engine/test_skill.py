"""Tests for `physiclaw.agent.engine.skill` — discovery + dispatch.

Discovery scans exactly two roots: the built-in package root
(`BUILTIN_SKILLS_DIR`, flat `.md`) and the user home root
(`HOME_SKILLS_DIR` = paths.skills_dir(), folder `SKILL.md`). The autouse
`_skill_roots` fixture re-points both at per-test tmp dirs. The cwd is
NOT a discovery root; the fixture still chdirs to an empty tmp for
hygiene.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from physiclaw.agent.engine import skill as skill_mod
from physiclaw.agent.engine.skill import (
    Skill,
    _load_reference,
    _split_frontmatter,
    discover,
    discover_builtin_skills,
    discover_user_skills,
    dispatch,
    render_builtin,
    render_section,
)


@pytest.fixture(autouse=True)
def _skill_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Per-test home + cwd skill roots.

    Also re-points the built-in root at an empty tmp dir so the real
    shipped flat-.md skills don't leak into discovery assertions.
    """
    home = tmp_path / "home_skills"
    cwd = tmp_path / "cwd_root"
    cwd.mkdir()
    builtin = tmp_path / "builtin_skills"
    builtin.mkdir()
    monkeypatch.setattr(skill_mod, "BUILTIN_SKILLS_DIR", builtin)
    monkeypatch.setattr(skill_mod, "HOME_SKILLS_DIR", home)
    monkeypatch.chdir(cwd)
    # CWD_SKILLS_DIR is `Path("skills")` — relative to cwd.
    return home, cwd


def _write_skill(
    root: Path, name: str, *, description: str = "do something",
    body: str = "Workflow steps here.",
    extra_files: dict[str, str] | None = None,
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}\n"
    )
    if extra_files:
        for relpath, content in extra_files.items():
            target = skill_dir / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
    return skill_dir


# ---------- discover ----------


def test_discover_empty_when_no_roots_exist() -> None:
    assert discover() == {}


def test_discover_ignores_cwd_skills_dir(_skill_roots) -> None:
    # The cwd is not a discovery root — a ./skills there must be ignored.
    home, cwd = _skill_roots
    _write_skill(cwd / "skills", "foo")

    assert discover() == {}


def test_discover_finds_skill_in_home_skills_root(_skill_roots) -> None:
    home, _ = _skill_roots
    _write_skill(home, "bar")

    out = discover()

    assert "bar" in out


def test_discover_skips_directories_starting_with_underscore(
    _skill_roots,
) -> None:
    home, _ = _skill_roots
    _write_skill(home, "_internal")

    assert discover() == {}


def test_discover_skips_skills_without_skill_md(_skill_roots) -> None:
    home, _ = _skill_roots
    skill_dir = home / "no-md"
    skill_dir.mkdir(parents=True)
    (skill_dir / "README.md").write_text("not a skill")

    assert discover() == {}


def test_discover_skips_non_directory_entries(_skill_roots) -> None:
    home, _ = _skill_roots
    home.mkdir()
    (home / "stray.txt").write_text("not a dir")

    assert discover() == {}


def test_discover_uses_directory_name_when_frontmatter_lacks_name(
    _skill_roots,
) -> None:
    home, _ = _skill_roots
    skill_dir = home / "no-name"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\ndescription: only desc\n---\nbody\n"
    )

    out = discover()

    assert "no-name" in out


def test_discover_rejects_symlink_resolving_outside_root(
    tmp_path: Path, _skill_roots,
) -> None:
    home, _ = _skill_roots
    home.mkdir()
    # Create a real skill outside the home root, then symlink into it.
    outside = tmp_path / "outside"
    _write_skill(outside, "evil")
    symlink_path = home / "evil-link"
    symlink_path.symlink_to(outside / "evil")

    out = discover()

    assert "evil" not in out


def test_discover_strips_frontmatter_whitespace_from_description_and_body(
    _skill_roots,
) -> None:
    home, _ = _skill_roots
    skill_dir = home / "trimmed"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: trimmed\ndescription: padded around\n---\n\n  body\n"
    )

    out = discover()

    assert out["trimmed"].description == "padded around"
    assert out["trimmed"].body == "body"


# ---------- built-in flat-.md discovery ----------


def _write_flat(name: str, *, description: str = "flat desc", body: str = "flat body") -> None:
    """Write a built-in-style flat `<name>.md` into the (monkeypatched) root."""
    root = skill_mod.BUILTIN_SKILLS_DIR
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}\n"
    )


def test_discover_finds_builtin_flat_md_skill() -> None:
    _write_flat("im", description="messaging", body="im workflow")

    out = discover()

    assert out["im"].description == "messaging"
    assert out["im"].body == "im workflow"
    assert out["im"].flat is True


def test_load_reference_rejects_flat_skill() -> None:
    sk = Skill(name="im", description="d", body="b", dir=Path("/tmp"), flat=True)

    with pytest.raises(ValueError, match=r"built-in and self-contained"):
        _load_reference(sk, "spec.md")


def test_discover_flat_uses_filename_stem_when_name_missing() -> None:
    (skill_mod.BUILTIN_SKILLS_DIR / "open-app.md").write_text(
        "---\ndescription: launch apps\n---\nbody\n"
    )

    assert "open-app" in discover()


def test_discover_flat_skips_underscore_prefixed_files() -> None:
    (skill_mod.BUILTIN_SKILLS_DIR / "_draft.md").write_text(
        "---\nname: _draft\ndescription: wip\n---\nbody\n"
    )

    assert discover() == {}


def test_discover_home_skill_overrides_builtin(_skill_roots) -> None:
    # Built-in is the lowest-precedence root; a home skill of the same name wins.
    home, _ = _skill_roots
    _write_flat("im", description="built-in", body="built-in-body")
    _write_skill(home, "im", description="user", body="user-body")

    out = discover()

    assert out["im"].description == "user"
    assert out["im"].body == "user-body"


# ---------- discover_builtin_skills / discover_user_skills ----------


def test_discover_builtin_only_scans_builtin_root(_skill_roots) -> None:
    home, _ = _skill_roots
    _write_flat("im", description="messaging")
    _write_skill(home, "taobao")

    out = discover_builtin_skills()

    assert set(out) == {"im"}
    assert out["im"].flat is True


def test_discover_user_only_scans_home_root(_skill_roots) -> None:
    home, _ = _skill_roots
    _write_flat("im")
    _write_skill(home, "taobao")

    out = discover_user_skills()

    assert set(out) == {"taobao"}
    assert out["taobao"].flat is False


# ---------- render_builtin ----------


def test_render_builtin_empty_when_no_skills() -> None:
    assert render_builtin({}) == ""


def test_render_builtin_inlines_full_bodies() -> None:
    skills = {
        "im": Skill(name="im", description="d", body="IM WORKFLOW", dir=Path("/"), flat=True),
    }

    out = render_builtin(skills)

    assert out.startswith("## Built-in Skills")
    assert "### im" in out
    assert "IM WORKFLOW" in out  # full body, not just metadata


# ---------- Skill dataclass ----------


def test_skill_dataclass_holds_required_fields() -> None:
    s = Skill(name="x", description="y", body="z", dir=Path("/tmp"))

    assert s.name == "x"
    assert s.description == "y"
    assert s.body == "z"
    assert s.dir == Path("/tmp")


# ---------- dispatch ----------


def test_dispatch_returns_body_for_known_skill(_skill_roots) -> None:
    home, _ = _skill_roots
    _write_skill(home, "foo", body="step 1\nstep 2")

    out = dispatch(discover(), {"name": "foo"})

    assert out == "step 1\nstep 2"


def test_dispatch_raises_on_missing_name_arg() -> None:
    with pytest.raises(
        ValueError, match=r"^Skill call requires a 'name' argument$"
    ):
        dispatch({}, {})


def test_dispatch_raises_on_empty_name_arg() -> None:
    with pytest.raises(ValueError, match=r"^Skill call requires a 'name'"):
        dispatch({}, {"name": "   "})


def test_dispatch_raises_on_unknown_skill_name(_skill_roots) -> None:
    home, _ = _skill_roots
    _write_skill(home, "alpha")
    _write_skill(home, "beta")

    with pytest.raises(
        ValueError,
        match=r"^unknown skill 'mystery'\. Available: alpha, beta$",
    ):
        dispatch(discover(), {"name": "mystery"})


def test_dispatch_lists_none_when_no_skills_available() -> None:
    with pytest.raises(
        ValueError, match=r"^unknown skill 'x'\. Available: \(none\)$"
    ):
        dispatch({}, {"name": "x"})


def test_dispatch_loads_reference_when_reference_arg_present(
    _skill_roots,
) -> None:
    home, _ = _skill_roots
    _write_skill(
        home, "foo",
        extra_files={"references/spec.md": "spec contents here"},
    )

    out = dispatch(discover(), {"name": "foo", "reference": "spec.md"})

    assert out == "spec contents here"


# ---------- _load_reference ----------


def test_load_reference_returns_file_contents(_skill_roots) -> None:
    home, _ = _skill_roots
    _write_skill(
        home, "foo",
        extra_files={"references/note.md": "hi"},
    )
    skills = discover()

    assert _load_reference(skills["foo"], "note.md") == "hi"


def test_load_reference_raises_on_path_traversal(_skill_roots, tmp_path: Path) -> None:
    home, _ = _skill_roots
    _write_skill(home, "foo")
    # Place an "escape" file outside the references dir.
    (home / "secret.md").write_text("don't read me")
    skills = discover()

    with pytest.raises(
        ValueError,
        match=r"^reference path '\.\./secret\.md' escapes skill directory$",
    ):
        _load_reference(skills["foo"], "../secret.md")


def test_load_reference_raises_when_target_missing(_skill_roots) -> None:
    home, _ = _skill_roots
    _write_skill(home, "foo")
    skills = discover()

    with pytest.raises(
        FileNotFoundError,
        match=r"^reference 'gone\.md' not found in skill 'foo'$",
    ):
        _load_reference(skills["foo"], "gone.md")


# ---------- _split_frontmatter ----------


def test_split_frontmatter_returns_empty_when_no_leading_dashes() -> None:
    fm, body = _split_frontmatter("hello world\n")

    assert fm == {}
    assert body == "hello world\n"


def test_split_frontmatter_returns_empty_when_end_marker_missing() -> None:
    fm, body = _split_frontmatter("---\nname: foo\nno end here\n")

    assert fm == {}


def test_split_frontmatter_parses_scalar_keys() -> None:
    fm, body = _split_frontmatter(
        "---\nname: foo\ndescription: do thing\n---\nbody text\n"
    )

    assert fm == {"name": "foo", "description": "do thing"}
    assert body == "body text\n"


def test_split_frontmatter_skips_lines_without_colon() -> None:
    fm, _ = _split_frontmatter(
        "---\nname: foo\nrandom-no-colon\ndescription: x\n---\nbody\n"
    )

    assert fm == {"name": "foo", "description": "x"}


def test_split_frontmatter_handles_value_containing_colons() -> None:
    fm, _ = _split_frontmatter(
        "---\nurl: https://example.com:443\n---\nbody\n"
    )

    assert fm == {"url": "https://example.com:443"}


def test_split_frontmatter_strips_leading_newlines_from_body() -> None:
    fm, body = _split_frontmatter("---\nname: foo\n---\n\n\nbody\n")

    assert body == "body\n"


# ---------- render_section ----------


def test_render_section_empty_when_no_skills() -> None:
    assert render_section({}) == ""


def test_render_section_lists_each_skill_as_bullet() -> None:
    skills = {
        "foo": Skill(name="foo", description="do foo", body="SECRET STEPS", dir=Path("/")),
        "bar": Skill(name="bar", description="do bar", body="", dir=Path("/")),
    }

    out = render_section(skills)

    assert out.startswith("## Available skills\n")
    assert "- **foo** — do foo" in out
    assert "- **bar** — do bar" in out
    assert "SECRET STEPS" not in out  # index only — body loads on demand


def test_render_section_uses_no_description_placeholder_when_empty() -> None:
    skills = {
        "foo": Skill(name="foo", description="", body="", dir=Path("/")),
    }

    assert "- **foo** — (no description)" in render_section(skills)
