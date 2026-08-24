"""Session-listing corpus — recorded screens for offline matching work.

`wire.jsonl` is the one artifact that retains full listings (trace events
truncate them); the wire-shape walking itself lives beside the writer
(`trace.rawlog.iter_request_texts`) — this module owns only "is this text
a screen" and the corpus file format. A corpus file is JSONL, one screen
per line::

    {"label": "taobao.results" | "other" | "?", "listing": "<text>"}

`extract` prefills labels with "?" for the human to edit; `partition`
consumes the labeled file — `<app>.<page>` lines are that page's genuine
observations, everything else labeled non-"?" is a hard negative.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from physiclaw.agent.trace.rawlog import iter_request_texts
from physiclaw.common import paths
from physiclaw.common.listing import LISTING_HEADER, Screen, parse_row
from physiclaw.common.text import read_text, write_text

UNLABELED = "?"


@dataclass(frozen=True)
class CorpusItem:
    label: str
    listing: str


def session_listings(sid: str) -> list[str]:
    """Every distinct listing text in one recorded session, in first-seen
    order."""
    p = paths.engine_sessions_dir() / sid / "wire.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"no wire.jsonl for session {sid!r} ({p})")
    seen: dict[str, None] = {}
    for role, text in iter_request_texts(p):
        # The SYSTEM prompt quotes the listing header verbatim in
        # doctrine, so system messages must not extract as screens.
        if role == "system" or text in seen:
            continue
        if LISTING_HEADER in text and _has_rows(text):
            seen[text] = None
    return list(seen)


def _has_rows(text: str) -> bool:
    """At least one line parses as a real element row — a header quoted in
    prose (or a superseded labels-only stub) is not a screen."""
    return any(parse_row(line) is not None for line in text.splitlines())


def partition(
    items: list[CorpusItem], app: str, page_names: set[str]
) -> tuple[dict[str, list[Screen]], list[Screen]]:
    """Split a labeled corpus for `capture_app`: `<app>.<page>` lines are
    that page's genuine observations; any other non-'?' label is a hard
    negative; '?' lines are ignored."""
    prefix = f"{app}."
    by_page: dict[str, list[Screen]] = {}
    negatives: list[Screen] = []
    for it in items:
        if it.label == UNLABELED:
            continue
        page = it.label[len(prefix) :] if it.label.startswith(prefix) else None
        screen = Screen.read(it.listing)
        if page in page_names:
            by_page.setdefault(page, []).append(screen)
        else:
            negatives.append(screen)
    return by_page, negatives


def write_corpus(path: Path, items: list[CorpusItem]) -> None:
    lines = [
        json.dumps({"label": it.label, "listing": it.listing}, ensure_ascii=False)
        for it in items
    ]
    write_text(path, "\n".join(lines) + "\n")


def read_corpus(path: Path) -> list[CorpusItem]:
    out: list[CorpusItem] = []
    for n, line in enumerate(read_text(path).splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            out.append(CorpusItem(label=str(rec["label"]), listing=str(rec["listing"])))
        except (ValueError, KeyError) as e:
            raise ValueError(f"{path}:{n}: bad corpus line ({e})") from e
    return out
