"""``physiclaw skills`` — install / list / uninstall community skills.

Convention: a skill source is a git repo with a top-level ``skills/``
directory; ``skills install jd`` looks for ``skills/jd/SKILL.md`` in the
source and copies the whole ``skills/jd/`` tree to
``~/.physiclaw/skills/jd/``.

Source resolution order:
    1. ``--from <url>`` flag (per-call override)
    2. ``CONFIG.skills.default_source`` (set via ``physiclaw config set``)
    3. error with instructions — no implicit fallback

Installed dirs carry a ``.installed-from`` JSON file so ``list`` can
show provenance and ``uninstall`` can distinguish CLI-managed skills
from user-authored ones (the latter need ``--force`` to delete).
"""

import datetime as dt
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from physiclaw import paths
from physiclaw.cli._format import exit_error, ok, warn
from physiclaw.config import load as _load_config
from physiclaw.text import read_text, write_text

skills_app = typer.Typer(
    help="Install, list, and remove skills from a git-repo source.",
    epilog="Example: physiclaw skills install wechat --from physiclaw/PhysiClaw",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    add_completion=False,
)


PROVENANCE_FILE = ".installed-from"

# Skill names share disk paths with the user's skills dir, so we refuse
# anything that could traverse or shadow. Matches SKILL.md frontmatter
# `name:` handling in agent/engine/skill.py.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


def installed_skill_dirs() -> list[Path]:
    """Sorted list of installed skill directories under
    ``~/.physiclaw/skills/``. Dot-prefixed entries (including
    in-progress ``.<name>.installing`` staging dirs) are skipped.
    Public so ``doctor`` and future ``status`` commands can enumerate
    installed skills without duplicating the walk."""
    home = paths.skills_dir()
    if not home.exists():
        return []
    return sorted(
        d for d in home.iterdir() if d.is_dir() and not d.name.startswith(".")
    )


def read_provenance(skill_dir: Path) -> dict | None:
    """Parse the ``.installed-from`` JSON marker, or None if absent OR
    corrupt. Callers that need to distinguish those cases can check
    ``(skill_dir / PROVENANCE_FILE).exists()`` separately — a rare
    enough corner that folding them here is the cleaner default."""
    p = skill_dir / PROVENANCE_FILE
    if not p.exists():
        return None
    try:
        return json.loads(read_text(p))
    except (OSError, ValueError):
        return None


def _normalize_source(source: str) -> str:
    """Accept several source forms:

    - URLs (``https://…``, ``ssh://…``, ``git@host:…``) — passed through.
    - Local paths (``/…``, ``./…``, ``../…``, ``~/…``) — passed through
      with ``~`` expanded. Explicit prefixes bypass the ``owner/repo``
      rewrite; ``./foo`` would otherwise look like the strict shorthand
      form because ``.`` is a valid char in the shorthand's first segment.
    - ``owner/repo`` shorthand — expanded to
      ``https://github.com/owner/repo.git``. A ``.git`` suffix is kept;
      otherwise added.
    """
    s = source.strip()
    if not s:
        return s
    if "://" in s or s.startswith("git@"):
        return s
    if s.startswith(("/", "./", "../", "~")):
        return str(Path(s).expanduser())
    if re.fullmatch(r"[A-Za-z0-9._\-]+/[A-Za-z0-9._\-]+(\.git)?", s):
        if not s.endswith(".git"):
            s += ".git"
        return f"https://github.com/{s}"
    return s


def _resolve_source(from_flag: str | None) -> str:
    """``--from`` wins; config default is the fallback; else error.
    Rejects sources that resolve inside ``paths.HOME`` — skills install
    from an external repo, never from PhysiClaw's own state dir."""
    if from_flag:
        source = _normalize_source(from_flag)
    else:
        cfg = _load_config()
        if not cfg.skills.default_source:
            exit_error(
                "no skill source configured.\n\n"
                "Options:\n"
                "  • Install from a specific repo:\n"
                "      physiclaw skills install <name> --from owner/repo\n"
                "  • Set a permanent default:\n"
                "      physiclaw config set skills.default_source owner/repo\n"
                "  • Repo convention: top-level skills/<name>/SKILL.md."
            )
        source = _normalize_source(cfg.skills.default_source)
    _reject_physiclaw_home_source(source)
    return source


def _reject_physiclaw_home_source(source: str) -> None:
    """Refuse ``--from`` / ``default_source`` that resolves inside
    ``paths.HOME``. A self-copy would create weird provenance and
    briefly copy agent state (memory/, logs, calibration) through
    $TMPDIR. URLs and SSH forms skip this check — only local paths
    can land inside HOME."""
    if "://" in source or source.startswith("git@"):
        return
    try:
        src_path = Path(source).resolve()
    except (OSError, ValueError):
        return
    try:
        src_path.relative_to(paths.HOME.resolve())
    except ValueError:
        return
    exit_error(
        f"--from {source!r} resolves inside {paths.HOME} "
        "(PhysiClaw's own home).\n"
        "Skills install from an external git repo. Pick a path outside "
        "the PhysiClaw home, or use a remote URL."
    )


def _validate_name(name: str) -> None:
    if not _NAME_RE.fullmatch(name):
        exit_error(
            f"invalid skill name {name!r}. Allowed: "
            "letters, digits, '.', '_', '-'."
        )


def _git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run ``git`` with captured output. Non-zero exit surfaces as Exit(1)
    with stderr echoed — no stack trace for expected failures (bad ref,
    network down, auth prompt). Git-not-installed also short-circuits
    here rather than dumping a FileNotFoundError traceback."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        exit_error(
            "`git` not found on PATH. Install it and retry:\n"
            "  macOS: xcode-select --install  (or `brew install git`)\n"
            "  linux: apt install git  (or your distro's package manager)"
        )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        exit_error(f"git {args[0]} failed: {err}")
    return result


def _clone(source: str, ref: str | None, dest: Path) -> str:
    """Shallow-clone ``source`` into ``dest``. If ``ref`` is given, check
    out that branch or tag. Returns the HEAD sha of the cloned tree."""
    clone_args = ["clone", "--depth=1"]
    if ref:
        clone_args += ["--branch", ref]
    clone_args += [source, str(dest)]
    _git(*clone_args)
    sha = _git("rev-parse", "HEAD", cwd=dest).stdout.strip()
    return sha


def _read_skill_name(skill_md: Path, fallback: str) -> str:
    """Pull ``name:`` from SKILL.md frontmatter; fall back to the directory
    name if the frontmatter omits it."""
    text = read_text(skill_md)
    if not text.startswith("---"):
        return fallback
    _, _, rest = text[3:].partition("---")
    if not rest:
        return fallback
    header = text[3:].split("---", 1)[0]
    for line in header.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            if k.strip() == "name":
                return v.strip().strip('"').strip("'") or fallback
    return fallback


def _write_provenance(dest: Path, source: str, ref: str | None, sha: str) -> None:
    payload = {
        "source": source,
        "ref": ref or "",
        "sha": sha,
        "installed_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    write_text(dest / PROVENANCE_FILE, json.dumps(payload, indent=2) + "\n")


def _atomic_replace(src: Path, dst: Path) -> None:
    """Move ``src`` into ``dst``, replacing any existing ``dst`` directory.
    Caller must have already obtained user consent for overwrite (via
    ``--force``). ``src`` and ``dst`` MUST be on the same filesystem."""
    if dst.exists():
        shutil.rmtree(dst)
    # os.replace, not Path.rename: the cross-platform atomic move — matches
    # this function's name and stays consistent with the official-skills sync
    # swap. (dst is already removed above, so the move targets an absent path.)
    os.replace(src, dst)


@skills_app.command(
    "install",
    no_args_is_help=True,
    epilog="Example: physiclaw skills install wechat --from physiclaw/PhysiClaw",
)
def _install(
    name: Annotated[
        str, typer.Argument(help="Skill name (matches a skills/<name>/ dir in the source repo)."),
    ],
    from_: Annotated[
        str | None,
        typer.Option(
            "--from", help="Git repo to install from. Overrides skills.default_source.",
        ),
    ] = None,
    ref: Annotated[
        str | None,
        typer.Option(
            "--ref", help="Branch or tag to check out. Defaults to the repo's default branch.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force", help="Overwrite an existing install of this skill.",
        ),
    ] = False,
) -> None:
    """Install a skill by copying ``skills/<name>/`` from the source repo
    into ``~/.physiclaw/skills/<name>/``."""
    _validate_name(name)
    source = _resolve_source(from_)

    home = paths.skills_dir()
    target = home / name

    if target.exists() and not force:
        prov = read_provenance(target)
        if prov is None:
            exit_error(
                f"~/.physiclaw/skills/{name}/ exists and was not installed "
                f"by this CLI (no {PROVENANCE_FILE!s} marker). Refusing to touch "
                f"user-authored skills. Pass --force to overwrite anyway."
            )
        exit_error(
            f"{name} is already installed (from {prov.get('source')!s} "
            f"@ {prov.get('ref') or prov.get('sha', '')[:7]}). "
            f"Pass --force to reinstall."
        )

    home.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="physiclaw-skills-") as tmp:
        clone_dir = Path(tmp) / "repo"
        typer.echo(f"  cloning {source}" + (f" @ {ref}" if ref else "") + " …")
        sha = _clone(source, ref, clone_dir)

        src_skill = clone_dir / "skills" / name
        if not src_skill.is_dir():
            exit_error(
                f"{source} does not contain skills/{name}/ "
                f"(convention: top-level skills/<name>/SKILL.md)."
            )

        skill_md = src_skill / "SKILL.md"
        if not skill_md.is_file():
            exit_error(
                f"skills/{name}/ is missing SKILL.md — not a valid skill."
            )

        declared_name = _read_skill_name(skill_md, fallback=name)
        if declared_name != name:
            # Hard reject: runtime discovery (agent/engine/skill.py) keys by
            # SKILL.md frontmatter, not by dir name. Installing `foo` when
            # the source declares `bar` leaves `Skill(name="foo")` broken at
            # runtime and `uninstall` orphaned from the discovered name.
            # Source must be reconciled — the installer can't fix it here.
            alt_dir_exists = (clone_dir / "skills" / declared_name).is_dir()
            lines = [
                f"source skills/{name}/SKILL.md declares "
                f"name={declared_name!r} — inconsistent with its directory "
                f"name. Runtime would key this skill as {declared_name!r}, "
                f"so `Skill(name={name!r})` would fail after install.",
                "",
                "Fix the source repo (one of):",
                f"  • rename skills/{name}/ → skills/{declared_name}/",
                f"  • or edit skills/{name}/SKILL.md frontmatter: name: {name}",
            ]
            if alt_dir_exists:
                lines += [
                    "",
                    f"Or, if you meant the other one: "
                    f"`physiclaw skills install {declared_name}` — that dir "
                    f"also exists in the source.",
                ]
            exit_error("\n".join(lines))

        # Copy to a sibling of target, then atomic-rename — leaves the old
        # install intact on any copy failure.
        staging = home / f".{name}.installing"
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(src_skill, staging, symlinks=False)
        _write_provenance(staging, source, ref, sha)

        _atomic_replace(staging, target)

    typer.echo(ok(f"installed {name} (from {source}" + (f" @ {ref}" if ref else "") + f", sha {sha[:7]})"))
    typer.echo(f"  → {target}")


sync_app = typer.Typer(
    help="Sync curated skill packs published by PhysiClaw.",
    epilog="Example: physiclaw skills sync official",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    add_completion=False,
)


@sync_app.command(
    "official",
    epilog="Example: physiclaw skills sync official --dry-run",
)
def _sync_official(
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Only check whether a sync would run; download nothing.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Re-download and re-extract even if already up to date.",
        ),
    ] = False,
) -> None:
    """Pull the official skill pack from the site into
    ``~/.physiclaw/official/``. Idempotent — a matching commit is a no-op."""
    # Imported lazily so the heavy stdlib bits (zipfile/hashlib) load only
    # when this command actually runs, keeping `physiclaw skills` startup lean.
    from physiclaw.cli import sync_official_skills

    sync_official_skills.sync(force=force, dry_run=dry_run)


skills_app.add_typer(sync_app, name="sync")


@skills_app.command(
    "list",
    epilog="Example: physiclaw skills list",
)
def _list() -> None:
    """List skills installed in ``~/.physiclaw/skills/``."""
    entries = installed_skill_dirs()
    if not entries:
        typer.echo(f"(no skills installed in {paths.skills_dir()})")
        return
    for d in entries:
        prov = read_provenance(d)
        if prov is None:
            typer.echo(f"  {d.name}  (local — no {PROVENANCE_FILE})")
        else:
            ref_or_sha = prov.get("ref") or prov.get("sha", "")[:7] or "?"
            typer.echo(f"  {d.name}  ← {prov.get('source', '?')} @ {ref_or_sha}")


@skills_app.command(
    "uninstall",
    no_args_is_help=True,
    epilog="Example: physiclaw skills uninstall wechat",
)
def _uninstall(
    name: Annotated[
        str, typer.Argument(help="Skill name to remove."),
    ],
    force: Annotated[
        bool,
        typer.Option(
            "--force", help="Remove even if the skill has no provenance marker (user-authored).",
        ),
    ] = False,
) -> None:
    """Remove a skill from ``~/.physiclaw/skills/``. Refuses to delete
    user-authored skills (no ``.installed-from`` marker) unless ``--force``
    is set."""
    _validate_name(name)
    target = paths.skills_dir() / name
    if not target.exists():
        exit_error(f"{name} is not installed (no {target}).")

    prov = read_provenance(target)
    if prov is None and not force:
        exit_error(
            f"{name} has no {PROVENANCE_FILE} marker — looks user-authored. "
            f"Pass --force to remove it anyway, or delete manually:\n"
            f"    rm -rf {target}"
        )

    shutil.rmtree(target)
    typer.echo(ok(f"removed {name}"))
    if prov is None:
        typer.echo(warn("no provenance marker — removed user-authored skill."))


__all__ = ["skills_app"]
