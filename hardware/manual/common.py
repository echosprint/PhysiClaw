"""Helpers shared by both document builders — localization, content
loading, table row-span math, step timing, and the document chrome
conventions. Keeps ``build_sourcing_guide`` from reaching into
``build_manual``'s internals.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

CONTENT_DIR = Path(__file__).resolve().parent / "content"
VENDOR_FILE = Path(__file__).resolve().parent / "sourcing_vendors.json"
MANUAL_VERSION_FILE = Path(__file__).resolve().parent / "MANUAL_VERSION"

# The <html lang> attribute value per language.
HTML_LANG = {"en": "en", "zh": "zh-Hans"}

URL_MARK = "PhysiClaw.ai"  # masthead mark, never localized.


def loc(value: Any, lang: str) -> str:
    """Resolve a localized value.

    Localized text is ``{"en": ..., "zh": ...}``; this returns the requested
    language, falling back to English when the translation is empty. Plain
    strings (specs, ids, URLs) pass straight through. Localized strings are
    trusted HTML and are emitted raw — the source embeds inline ``<span>``/``<a>``.
    """
    if isinstance(value, dict):
        return value.get(lang) or value.get("en", "")
    return value


def manual_version() -> str:
    """The manual's cover version stamp; MANUAL_VERSION is the single source of truth."""
    return MANUAL_VERSION_FILE.read_text(encoding="utf-8").strip()


def load_pages() -> list[dict]:
    """Load every content/*.json (sorted by filename = page order) into pages."""
    pages: list[dict] = []
    for path in sorted(CONTENT_DIR.glob("*.json")):
        pages.extend(json.loads(path.read_text(encoding="utf-8")))
    return pages


def _rowspans(rows: list[dict], key: Callable[[dict], Any]) -> list[int]:
    """For consecutive rows sharing ``key``, return the group size at each
    group's first row and 0 at the rest — i.e. the rowspan to emit (or skip)."""
    spans = [0] * len(rows)
    i = 0
    while i < len(rows):
        j = i + 1
        while j < len(rows) and key(rows[j]) == key(rows[i]):
            j += 1
        spans[i] = j - i
        i = j
    return spans


_STEP_LABEL_W = 22  # pad labels to a column so every duration lines up


@contextlib.contextmanager
def _step(label: str) -> Iterator[None]:
    """Print a build step (padded with dot leaders to a fixed column) and, once
    it finishes, how long it took — so a slow phase shows what's running and the
    times read down a clean right-aligned column."""
    print(f"  {label:.<{_STEP_LABEL_W}} ", end="", flush=True)
    t0 = time.monotonic()
    try:
        yield
    except BaseException:
        print("FAILED", flush=True)
        raise
    print(f"{time.monotonic() - t0:>5.1f}s")
