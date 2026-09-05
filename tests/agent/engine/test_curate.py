"""Tests for `physiclaw.agent.engine.curate` — the post-session LLM pass that
consolidates the learned-pitfalls list.

The autouse `physiclaw_home` fixture isolates `paths.pitfalls_dir()`.
"""

from __future__ import annotations

import pytest

from physiclaw.agent.engine import curate, pitfalls
from physiclaw.contract.dto import AssistantMessage, FinishReason


class _FakeProvider:
    """Returns `replies` in order from chat(); records the histories it saw."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[list] = []

    async def chat(self, history, tools, **kw):
        self.calls.append(history)
        return AssistantMessage(
            content=self._replies.pop(0), tool_calls=[], finish_reason=FinishReason.STOP
        )


class _BoomProvider:
    async def chat(self, history, tools, **kw):
        raise RuntimeError("provider down")


# ---------- _parse ----------


def test_parse_plain_fenced_and_prose_wrapped_array() -> None:
    assert curate._parse('["a", "b"]') == ["a", "b"]
    assert curate._parse('```json\n["a"]\n```') == ["a"]
    assert curate._parse('Here you go:\n["a", "b"]\nDone.') == ["a", "b"]


def test_parse_drops_non_strings_and_blanks() -> None:
    assert curate._parse('["a", 1, "", "  ", "b"]') == ["a", "b"]


def test_parse_garbage_returns_none() -> None:
    assert curate._parse("not json") is None  # no brackets
    assert curate._parse('{"a": 1}') is None  # not an array
    assert curate._parse("") is None


# ---------- curate ----------


@pytest.mark.asyncio
async def test_curate_consolidates_list_through_replace() -> None:
    pitfalls.add(["京东: Ai搜索 opens AI chat"])
    pitfalls.add(["京东: the Ai搜索 toggle opens AI chat"])  # near-dup
    provider = _FakeProvider(['["京东: Ai搜索 opens AI chat → use right-side 搜索"]'])

    assert await curate.curate(provider) is True
    assert pitfalls.read() == ["京东: Ai搜索 opens AI chat → use right-side 搜索"]
    # A curate snapshot lands in history.
    hist = (pitfalls.paths.pitfalls_dir() / "history.jsonl").read_text(encoding="utf-8")
    assert '"op": "curate"' in hist


@pytest.mark.asyncio
async def test_curate_empty_list_is_noop() -> None:
    provider = _FakeProvider([])  # never called
    assert await curate.curate(provider) is False
    assert provider.calls == []


@pytest.mark.asyncio
async def test_curate_fail_open_on_provider_error() -> None:
    pitfalls.add(["京东: a trap"])
    assert await curate.curate(_BoomProvider()) is False
    assert pitfalls.read() == ["京东: a trap"]  # untouched


@pytest.mark.asyncio
async def test_curate_skips_unparseable_without_clobbering() -> None:
    pitfalls.add(["keep me"])
    assert await curate.curate(_FakeProvider(["garbage not json"])) is False
    assert pitfalls.read() == ["keep me"]
