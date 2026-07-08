"""Skill discovery and tiered dispatch.

A skill comes in one of two shapes — a built-in flat file, or a folder (the
shape both official and user skills share):

  Built-in (flat file):  ``<pkg>/agent/skills/<name>.md``
      frontmatter (name, description) + body. Ships in the wheel, so every
      install has the core skills out of the box — no repo, no download.
      References (tier 3) aren't supported here; built-in skills are
      self-contained.

  Official (folder):  ``~/.physiclaw/official/skills/<name>/SKILL.md`` — the
      curated pack synced by ``physiclaw skills sync official``. Same folder
      shape as user skills; the sync owns the tree wholesale.

  User (folder):  ``~/.physiclaw/skills/<name>/SKILL.md`` (+ optional
      ``references/``, ``assets/``). The user's own skills.

Three roots are scanned:

  1. built-in ``<pkg>/agent/skills/*.md`` — the shipped baseline, resolved
     from the installed ``physiclaw`` package (never the cwd). **Always on:**
     built-ins are inlined into SYSTEM every session and never shadowed.
  2. ``~/.physiclaw/official/skills/<name>/SKILL.md`` (official).
  3. ``~/.physiclaw/skills/<name>/SKILL.md`` (home) — the ONLY place a
     user's custom skills come from, so a ``uv tool install``-ed server
     finds them without a repo cwd.

Official and user are the on-demand set (``## Available skills``) and do NOT
shadow each other: on a name clash both are kept, disambiguated as
``<name>-official`` / ``<name>-user`` (see ``_merge_disambiguated``); an
unclashed name passes through unchanged. Built-ins are separate and always
on. The cwd is deliberately not a root — discovery must not depend on where
the server was launched.

Loading differs by kind — the two render into SEPARATE prompt sections, so
they never merge or override each other:
  * Built-in — the full body is always inlined by `render_builtin` under
    `## Built-in Skills`. No Skill() call; the model always has the steps.
  * User — only name+description go in, via `render_section` under
    `## Available skills`. The description is the ONLY relevance signal, so it
    must name triggers. The model then loads the body on demand:
      - `Skill(name=...)` returns the SKILL.md body as a tool_result.
      - `Skill(name=..., reference=<path>)` loads `<skill>/references/<path>`
        as text, resolved under that skill's dir only (built-in skills, being
        self-contained, have no references).

Discovery is realpath-scoped within each root: a symlink that escapes
its root is rejected so third-party skill installs can't path-traverse.
"""
import logging
from dataclasses import dataclass, replace
from pathlib import Path

from physiclaw import paths
from physiclaw.text import read_text

log = logging.getLogger(__name__)

BUILTIN_SKILLS_DIR = paths.builtin_skills_dir()
OFFICIAL_SKILLS_DIR = paths.official_skills_dir()
HOME_SKILLS_DIR = paths.skills_dir()


@dataclass
class Skill:
    """Snapshot of a skill at discover() time. `body` and `dir` are
    captured on session start; edits to SKILL.md mid-session are not
    reflected (deliberate: protects the SYSTEM prompt cache prefix)."""
    name: str
    description: str
    body: str
    dir: Path   # realpath; used to resolve references/ safely
    flat: bool = False  # built-in flat-.md skill: body is authoritative, no references/
    source: str = "user"  # origin root: "built-in" | "official" | "user" — groups the index


def discover_builtin_skills() -> dict[str, Skill]:
    """Built-in flat-`.md` skills shipped in the package. Their full body is
    always inlined into the SYSTEM prompt, so they need no Skill() call."""
    out: dict[str, Skill] = {}
    _scan_flat_root(BUILTIN_SKILLS_DIR, out)
    return out


def discover_user_skills() -> dict[str, Skill]:
    """The on-demand folder skills for ``## Available skills``: official (synced
    to ~/.physiclaw/official/skills) + user (~/.physiclaw/skills). Only their
    name+description go in the SYSTEM prompt; the body loads on demand via the
    Skill() tool. Built-in skills are separate (always-on, inlined) and never
    appear here. Official and user do NOT shadow each other — a name clash keeps
    both, disambiguated ``<name>-official`` / ``<name>-user``."""
    official: dict[str, Skill] = {}
    _scan_root(OFFICIAL_SKILLS_DIR, official, source="official")
    user: dict[str, Skill] = {}
    _scan_root(HOME_SKILLS_DIR, user, source="user")
    return _merge_disambiguated(official, user)


def _merge_disambiguated(
    official: dict[str, Skill], user: dict[str, Skill]
) -> dict[str, Skill]:
    """Combine the official + user on-demand skills WITHOUT shadowing. On a name
    clash both are kept, renamed ``<name>-official`` / ``<name>-user`` — on the
    dict key AND ``Skill.name``, so the index bullet (uses ``.name``), the
    Skill-tool name list + dispatch (use the key) all agree. Unclashed names
    pass through unchanged. Official is emitted first so it sorts ahead of its
    user counterpart in the tool list."""
    clashes = official.keys() & user.keys()
    out: dict[str, Skill] = {}
    for suffix, group in (("official", official), ("user", user)):
        for name, sk in group.items():
            key = f"{name}-{suffix}" if name in clashes else name
            out[key] = replace(sk, name=key)
    return out


def discover() -> dict[str, Skill]:
    """Merged view — built-in (always on), then the disambiguated official+user
    on-demand set layered on top. Used by the Claude-provider plugin dir, which
    materializes every skill as a Claude Code skill; the engine keeps built-in
    (inlined) and on-demand (Skill()) separate instead. One bad skill skips
    silently rather than taking down the session."""
    out = discover_builtin_skills()
    out.update(discover_user_skills())
    return out


def _scan_flat_root(root: Path, out: dict[str, Skill]) -> None:
    """Scan a built-in root of flat ``<name>.md`` files. The frontmatter
    `name` wins; otherwise the filename stem is the skill name. `dir` is
    the root itself — built-in skills have no `references/`, so any tier-3
    load resolves under (and fails inside) this shared dir, which is fine.
    """
    if not root.exists():
        return
    for md in sorted(root.glob("*.md")):
        if md.name.startswith("_") or not md.is_file():
            continue
        fm, body = _split_frontmatter(read_text(md))
        name = fm.get("name") or md.stem
        out[name] = Skill(
            name=name,
            description=(fm.get("description") or "").strip(),
            body=body.strip(),
            dir=root.resolve(),
            flat=True,
            source="built-in",
        )


def _scan_root(root: Path, out: dict[str, Skill], *, source: str = "user") -> None:
    if not root.exists():
        return
    root_real = root.resolve()
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        real = d.resolve()
        if not real.is_relative_to(root_real):
            log.warning(
                "skill %s resolves outside %s; skipping (path traversal guard)",
                d.name, root,
            )
            continue
        md = real / "SKILL.md"
        if not md.exists():
            continue
        fm, body = _split_frontmatter(read_text(md))
        name = fm.get("name") or d.name
        out[name] = Skill(
            name=name,
            description=(fm.get("description") or "").strip(),
            body=body.strip(),
            dir=real,
            source=source,
        )


def dispatch(skills: dict[str, Skill], args: dict) -> str:
    """Route a Skill() invocation by exact name. Raises ValueError /
    FileNotFoundError on bad input so the engine's _dispatch marks the
    tool_result as is_error=True (principle 5)."""
    name = (args.get("name") or "").strip()
    if not name:
        raise ValueError("Skill call requires a 'name' argument")
    skill = skills.get(name)
    if skill is None:
        available = ", ".join(sorted(skills.keys())) or "(none)"
        raise ValueError(f"unknown skill {name!r}. Available: {available}")

    ref = args.get("reference")
    if ref:
        return _load_reference(skill, ref)
    return skill.body


def _load_reference(skill: Skill, ref_path: str) -> str:
    if skill.flat:
        raise ValueError(
            f"skill {skill.name!r} is built-in and self-contained — it has no "
            "references"
        )
    # skill.dir is already a realpath from discover(); no need to resolve it
    # again. Only the target needs resolving to collapse any `..` segments
    # in `ref_path` before the scope check.
    root = skill.dir / "references"
    target = (root / ref_path).resolve()
    if not target.is_relative_to(root):
        raise ValueError(
            f"reference path {ref_path!r} escapes skill directory"
        )
    if not target.exists():
        raise FileNotFoundError(
            f"reference {ref_path!r} not found in skill {skill.name!r}"
        )
    return read_text(target)


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse `---\\nkey: value\\n...\\n---\\n<body>`. Simple scalar-only
    format; missing or malformed frontmatter returns ({}, text)."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end < 0:
        return {}, text
    fm_text = text[4:end]
    body = text[end + 4:].lstrip("\n")
    fm: dict[str, str] = {}
    for line in fm_text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip()
    return fm, body


def _skill_bullets(skills: list[Skill]) -> list[str]:
    """One `- **name** — description` index line per skill."""
    return [
        f"- **{s.name}** — {s.description or '(no description)'}"
        for s in skills
    ]


# Subsection heading per `source`, in the order they appear under
# `## Available skills` (dict insertion order). A subsection is emitted only if
# that source has any skills.
_SECTION_LABELS = {
    "built-in": "built-in skills",
    "official": "official skills",
    "user": "user skills",
}


def render_section(skills: dict[str, Skill]) -> str:
    """`## Available skills`, split into `### official skills` / `### user skills`
    subsections (and `### built-in skills` on the Claude path, which merges all
    three) — one `- **name** — description` line per skill. Empty string if none.
    A pure index; the body loads on demand via the Skill tool. Built-in skills in
    the engine path use `render_builtin` instead (full bodies, not an index)."""
    if not skills:
        return ""
    grouped: dict[str, list[Skill]] = {}
    for s in skills.values():
        grouped.setdefault(s.source, []).append(s)
    # Known sources in fixed order first, then any unknown source (stable) so a
    # future origin still renders rather than vanishing.
    ordered = [s for s in _SECTION_LABELS if s in grouped]
    ordered += [s for s in grouped if s not in _SECTION_LABELS]
    blocks = [
        f"### {_SECTION_LABELS.get(src, f'{src} skills')}\n\n"
        + "\n".join(_skill_bullets(grouped[src]))
        for src in ordered
    ]
    return "## Available skills\n\n" + "\n\n".join(blocks)


def render_builtin(builtin: dict[str, Skill]) -> str:
    """The built-in skills' FULL bodies, inlined under `## Built-in Skills`. Always
    present in the SYSTEM prompt, so the model never calls Skill() for these
    — the steps and bboxes are already in front of it. Own section, so a
    user skill can never shadow or collide with a built-in workflow."""
    if not builtin:
        return ""
    lines = [
        "## Built-in Skills",
        "",
        "Always available — no loading. Before acting, check for a match: if "
        "one fits, follow its steps in order; else proceed normally. Boxes "
        "written as `{{…}}` name learned elements — resolve them from SYSTEM "
        "§ Screen layout (code templates already carry the substituted "
        "coordinates, labeled by comments).",
        "",
    ]
    for s in builtin.values():
        lines.append(f"### {s.name}")
        lines.append("")
        if s.body:
            lines.append(s.body)
            lines.append("")
    return "\n".join(lines).rstrip()


