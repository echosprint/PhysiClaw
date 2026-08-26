"""Memory slices for micro-call context — NOT `engine.memory`.

The conductor may import only `engine.dto` (architecture rule), so
memory.md is read off the shared path constant here. The contract is
least-privilege and FAIL CLOSED: a DECIDE's `context: [memory.<slug>]`
receives ONLY the matching `## <slug>` section — micro-calls may run on
a different vendor's cheap tier, and no match means NO memory context
(a degraded pick escalates safely; a privacy boundary must never
silently widen to the whole file). A slug matches a heading only as a
whole whitespace-separated token — `shopping` never bleeds into
`## shopping_blacklist`; bilingual headings work as
`## shopping_prefs 购物偏好`.
"""

from physiclaw.common import paths
from physiclaw.common.text import read_text

# One parsed section: (heading tokens, whole section text).
Sections = list[tuple[frozenset[str], str]]


def read_sections() -> Sections:
    """memory.md, read and carved in one step — what a walk caches."""
    f = paths.memory_file()
    return split_sections(read_text(f).strip() if f.exists() else "")


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
