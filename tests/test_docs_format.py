"""Format guards for the Starlight docs under ``docs/``.

The pages are rendered by Astro Starlight in a separate repo
(``physiclaw/docs-site``): on every push touching ``docs/``, a GitHub
Action mirrors this directory there and the site builds. That means
frontmatter mistakes surface only at deploy time, in another repo, or
never: Starlight's Zod schema *strips* unknown keys, so a misspelled
``descripton:`` deploys fine and silently ships a page with no search
snippet. These tests fail the same mistakes in the PR that makes them.

The contract they enforce comes from two places:

- Starlight's frontmatter reference
  (https://starlight.astro.build/reference/frontmatter/): ``title`` is
  required on every page.
- This repo's authoring rules (``docs/README.md``): frontmatter is just
  ``title`` + ``description`` as flat one-line scalars, pages carry no
  ESM ``import``/``export`` (the renderer injects component imports),
  every page ships as an EN/ZH pair (``foo.mdx`` + ``foo.zh.mdx``),
  and the sidebar order lives in ``docs.json``.

The frontmatter parser below is deliberately a strict subset of YAML:
flat ``key: value`` lines with plain or quoted scalar values. The docs
are parsed at deploy time by a different YAML dialect (js-yaml, 1.2)
than Python tooling would use (PyYAML, 1.1), and the dialects disagree
exactly on the ambiguous cases — so the guard accepts only scalars
that read as the same string under any dialect and rejects the rest
with a quote-it message: leading indicator characters, ``: `` or
`` #`` inside the value (the classic "colon in the description" deploy
breaker), and values YAML type-coerces away from string (``no``,
``1.20``, ``22:30``, dates). Extend the parser if the authoring
contract ever grows past flat scalars.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

# docs/README.md: "Frontmatter is just `title` + `description`." Starlight
# accepts more, but that wider schema lives in docs-site next to the Zod
# definition that owns it; here, any other key is either a typo or a
# renegotiation of the authoring contract.
CONTRACT_FIELDS = ("title", "description")

_KEY_LINE = re.compile(r"^([A-Za-z][A-Za-z0-9]*): (\S.*)$")

# Characters YAML reserves at the start of a plain scalar: they switch the
# node's type (flow collection, alias, tag, block scalar, …) instead of
# reading as text.
_YAML_INDICATORS = set("-?,[]{}#&*!|>%@`")

# Plain scalars YAML silently coerces to a non-string: booleans (incl. the
# YAML-1.1 "Norway problem" set), null, numbers (1.20 → 1.2, 0123 → octal),
# sexagesimals (22:30 → 1350) and timestamps. Starlight's Zod schema then
# rejects the page (title/description must be strings) — at deploy time,
# in the docs-site repo. Quoting removes the ambiguity.
_YAML_COERCED = re.compile(
    r"""(?xi)^(
        true|false|yes|no|on|off|null|~
        | [-+]?(\.inf|\.nan)
        | [-+]?\d[\d_]*(\.[\d_]*)?([eE][-+]?\d+)?
        | 0[xo][0-9a-f_]+
        | \d+(:\d+)+(\.\d*)?
        | \d{4}-\d{1,2}-\d{1,2}([Tt ].*)?
    )$""",
)


def _pages() -> list[Path]:
    """Every rendered page: ``.md``/``.mdx`` under docs/, minus the README."""
    return sorted(
        p
        for p in DOCS.rglob("*")
        if p.suffix in {".md", ".mdx"} and p.name != "README.md"
    )


def _quoted_errors(rel: Path, key: str, value: str) -> list[str]:
    quote, inner = value[0], value[1:-1]
    if len(value) < 2 or not value.endswith(quote):
        return [f"{rel}: {key}: unterminated {quote}-quoted value"]
    if quote == '"' and re.search(r'(?<!\\)"', inner):
        return [f'{rel}: {key}: unescaped " inside a double-quoted value (use \\")']
    if quote == "'" and inner.replace("''", "").count("'"):
        return [f"{rel}: {key}: unescaped ' inside a single-quoted value (double it)"]
    return []


def _scalar_errors(rel: Path, key: str, value: str) -> list[str]:
    if value[0] in "\"'":
        return _quoted_errors(rel, key, value)
    bad = []
    if value[0] in _YAML_INDICATORS:
        bad.append(
            f"{rel}: {key}: value starts with YAML indicator {value[0]!r}; quote it"
        )
    if ": " in value or value.endswith(":"):
        bad.append(
            f"{rel}: {key}: unquoted ': ' breaks the YAML parse"
            " ('Failed to parse Markdown frontmatter' at deploy); quote the value"
        )
    if " #" in value:
        bad.append(f"{rel}: {key}: ' #' starts a YAML comment mid-value; quote it")
    if _YAML_COERCED.match(value):
        bad.append(
            f"{rel}: {key}: YAML coerces {value!r} to a non-string"
            " (bool/number/date), which fails Starlight's schema; quote it"
        )
    return bad


def _parse(path: Path) -> tuple[dict[str, str], list[str]]:
    """(frontmatter fields, errors) for one page."""
    rel = path.relative_to(REPO)
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}, [f"{rel}: page must open with a '---' frontmatter fence"]
    head, sep, _ = text[4:].partition("\n---\n")
    if not sep:
        return {}, [f"{rel}: frontmatter fence is never closed"]
    fields: dict[str, str] = {}
    errors: list[str] = []
    for line in head.splitlines():
        if not line.strip():
            continue
        match = _KEY_LINE.match(line)
        if not match:
            errors.append(
                f"{rel}: frontmatter line {line!r} is not a flat 'key: value'"
                " scalar (see the parser note in this module's docstring)"
            )
            continue
        key, value = match.group(1), match.group(2).rstrip()
        if key in fields:
            errors.append(f"{rel}: duplicate frontmatter key {key!r}")
        fields[key] = value
        errors += _scalar_errors(rel, key, value)
    return fields, errors


def test_frontmatter_is_valid() -> None:
    """Every page parses and carries exactly the contract keys: ``title``
    (Starlight-required) and ``description`` (this repo's rule)."""
    bad: list[str] = []
    for page in _pages():
        rel = page.relative_to(REPO)
        fields, errors = _parse(page)
        bad += errors
        if not fields:
            continue
        for key in CONTRACT_FIELDS:
            if key not in fields:
                bad.append(f"{rel}: missing required frontmatter key {key!r}")
        for key in fields:
            if key not in CONTRACT_FIELDS:
                bad.append(
                    f"{rel}: {key!r} is outside the docs frontmatter contract"
                    " (docs/README.md: just title + description; Starlight's"
                    " Zod schema strips unknown keys silently)"
                )
    assert not bad, "docs frontmatter violations:\n" + "\n".join(bad)


def test_every_page_ships_en_and_zh() -> None:
    """Pages come in pairs: the English source and its ``.zh`` translation
    must both exist, declaring the same frontmatter keys (values are
    translated, shape isn't)."""
    bad: list[str] = []
    orphans = {p for p in _pages() if ".zh" in p.suffixes}
    for page in _pages():
        if ".zh" in page.suffixes:
            continue
        rel = page.relative_to(REPO)
        zh = page.with_suffix(f".zh{page.suffix}")
        if zh not in orphans:
            bad.append(f"{rel}: missing its {zh.name} translation")
            continue
        orphans.remove(zh)
        en_fields, _ = _parse(page)
        zh_fields, _ = _parse(zh)
        if set(zh_fields) != set(en_fields):
            diff = sorted(set(zh_fields) ^ set(en_fields))
            bad.append(f"{rel}: frontmatter keys differ from {zh.name}: {diff}")
    for zh in sorted(orphans):
        bad.append(f"{zh.relative_to(REPO)}: translation without an English source")
    assert not bad, "EN/ZH pairing violations:\n" + "\n".join(bad)


def test_pages_carry_no_esm() -> None:
    """docs/README.md: no ``import`` statements — the renderer injects
    component imports at build time, so an ESM line here would collide.
    Fenced code blocks are exempt (prose about imports is fine)."""
    bad: list[str] = []
    for page in _pages():
        rel = page.relative_to(REPO)
        fences = 0  # '---' lines seen; the body starts after the second
        in_code = False
        for i, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1):
            if fences < 2:
                if line == "---":
                    fences += 1
            elif line.lstrip().startswith("```"):
                in_code = not in_code
            elif not in_code and re.match(r"^(import|export)\s", line):
                bad.append(f"{rel}:{i}: ESM statement in page body: {line!r}")
    assert not bad, "ESM in docs pages:\n" + "\n".join(bad)


def test_sidebar_lists_every_page_exactly_once() -> None:
    """docs.json slugs resolve to English pages, no duplicates across
    sections (docs.schema.json's ``uniqueItems`` only covers one section),
    and every page is reachable from the sidebar. Deliberately stricter
    than the renderer, which only *warns* on unlisted pages; ``index`` is
    the landing page and can't be a sidebar item."""
    sidebar = json.loads((DOCS / "docs.json").read_text(encoding="utf-8"))["sidebar"]
    slugs = [slug for section in sidebar for slug in section["items"]]
    english = {
        p.relative_to(DOCS).with_suffix("").as_posix()
        for p in _pages()
        if ".zh" not in p.suffixes
    }

    dupes = sorted(slug for slug, n in Counter(slugs).items() if n > 1)
    assert not dupes, f"duplicate sidebar slugs: {dupes}"

    missing = sorted(set(slugs) - english)
    assert not missing, f"sidebar slugs with no English page: {missing}"

    unlisted = sorted(english - set(slugs) - {"index"})
    assert not unlisted, f"pages missing from docs.json sidebar: {unlisted}"
