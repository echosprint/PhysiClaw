"""Tests for `physiclaw.conductor.spec.context` — what an agent step loads
beside its prompt, exactly as declared and nothing more."""

from __future__ import annotations

import pytest

from physiclaw.common import daylog, paths
from physiclaw.common.text import write_text
from physiclaw.conductor.spec import context

MEMORY = "## shopping_prefs 购物偏好\n- 只买伊利\n\n## shopping_blacklist\n- 三无\n"


def _write_memory(text: str = MEMORY) -> None:
    f = paths.memory_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    write_text(f, text)


@pytest.mark.parametrize(
    "entry, ok",
    [
        ("memory", True),
        ("memory.shopping_prefs", True),
        ("daylog", True),
        ("pitfalls", False),
        ("memory.", False),
        ("memory.Bad-Slug", False),
        (3, False),
    ],
)
def test_check_entry_names_the_three_sources(entry, ok: bool) -> None:
    assert (context.check_entry(entry) is None) is ok


def test_memory_slice_is_token_matched_and_fail_closed() -> None:
    # Least-privilege: only the declared section travels — token-exact
    # heading match (no substring bleed), and NO match means NO text.
    _write_memory()

    sliced = context.load(("memory.shopping_prefs",))
    assert "只买伊利" in sliced and "三无" not in sliced
    # `shopping` is a substring of both headings but a token of neither.
    assert context.load(("memory.shopping",)) == ""
    assert context.load(("memory.nothing",)) == ""


def test_whole_memory_and_daylog_load_when_declared() -> None:
    _write_memory()
    daylog.append_log("[11:02] demo: bought milk ¥45")

    loaded = context.load(("memory", "daylog"))

    assert "只买伊利" in loaded and "三无" in loaded
    assert "bought milk ¥45" in loaded


def test_nothing_declared_loads_nothing() -> None:
    _write_memory()
    daylog.append_log("[11:02] demo: bought milk ¥45")

    assert context.load(()) == ""
