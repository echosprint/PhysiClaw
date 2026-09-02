"""What an agent step may load beside its prompt — declared, never
implied. An `agent`'s `context:` lists the sources; nothing else of the
agent's memory travels to the model:

  - ``memory``          the whole memory.md
  - ``memory.<slug>``   ONLY its `## <slug>` section — a slug matches a
                        heading as a whole whitespace-separated token
                        (`shopping` never bleeds into `## shopping_blacklist`;
                        bilingual headings work as `## shopping_prefs 购物偏好`),
                        and no match means no text (fail closed)
  - ``daylog``          the recent daily-log window — the same one the
                        engine preloads into the model's wake context

One loader per root in `_LOADERS`; the parser's `check_entry` reads
legality off the same table, so a new source is one row. The conductor
may never import the engine (architecture rule), so memory.md is read
off the shared path constant here.
"""

from collections.abc import Callable

from physiclaw.common import daylog, paths
from physiclaw.common.config import CONFIG
from physiclaw.common.text import read_text
from physiclaw.conductor._spec import INPUT_NAME_RE

MEMORY = "memory"
DAYLOG = "daylog"


def check_entry(entry: object) -> str | None:
    """None when `entry` is a legal `context:` item, else the rule it
    breaks."""
    if not isinstance(entry, str):
        return "must be a string"
    root, sep, slug = entry.partition(".")
    if root not in _LOADERS or (
        sep and not (root == MEMORY and INPUT_NAME_RE.match(slug))
    ):
        return f"must be `{MEMORY}`, `{MEMORY}.<slug>`, or `{DAYLOG}`"
    return None


def load(entries: tuple[str, ...]) -> str:
    """The declared sources, read now and joined — "" when nothing is
    declared or nothing matches. Entries group by root, so memory.md is
    read once however many of its sections are named."""
    by_root: dict[str, list[str]] = {}
    for entry in entries:
        root, _, slug = entry.partition(".")
        by_root.setdefault(root, []).append(slug)
    parts = [_LOADERS[root](slugs) for root, slugs in by_root.items()]
    return "\n\n".join(p for p in parts if p)


def _load_memory(slugs: list[str]) -> str:
    """The whole file when named bare (an empty slug), else ONLY the
    named sections."""
    f = paths.memory_file()
    text = read_text(f).strip() if f.exists() else ""
    if not text:
        return ""
    if "" in slugs:
        return f"memory.md:\n{text}"
    return match_sections(split_sections(text), slugs)


def _load_daylog(slugs: list[str]) -> str:
    recent = daylog.load_recent_entries(CONFIG.memory.bootstrap_log_entries)
    return f"Recent daily-log entries (newest first):\n{recent}" if recent else ""


_LOADERS: dict[str, Callable[[list[str]], str]] = {
    MEMORY: _load_memory,
    DAYLOG: _load_daylog,
}


# One parsed section: (heading tokens, whole section text).
Sections = list[tuple[frozenset[str], str]]


def split_sections(text: str) -> Sections:
    """The file carved at its `## ` headings — parsed once, matched many
    times."""
    sections: Sections = []
    tokens: frozenset[str] | None = None
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if tokens is not None:
                sections.append((tokens, "\n".join(body).strip()))
            tokens = frozenset(line[3:].casefold().split())
            body = [line]
        elif tokens is not None:
            body.append(line)
    if tokens is not None:
        sections.append((tokens, "\n".join(body).strip()))
    return sections


def match_sections(sections: Sections, slugs: list[str]) -> str:
    wanted = {slug.casefold() for slug in slugs}
    return "\n".join(body for tokens, body in sections if wanted & tokens)
