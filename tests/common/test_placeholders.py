"""Tests for `physiclaw.common.placeholders` — the `<<TOKEN>>` grammar
and the local `playbooks/placeholders.yml` values file it fills from."""

from __future__ import annotations

import pytest

from physiclaw.common.placeholders import (
    PLACEHOLDER_VALUES_FILENAME,
    find_placeholders,
    placeholder_values,
    resolve_placeholders,
    write_placeholder_values,
)


def test_find_placeholders_dedupes_in_order() -> None:
    found = find_placeholders("hi <<CONTACT>> and <<CITY>> and <<CONTACT>>")

    assert found == ["CONTACT", "CITY"]


def test_find_placeholders_ignores_input_braces_and_lowercase() -> None:
    # `{input}` refs and lowercase pseudo-tokens are NOT placeholders —
    # the sigil is uppercase-only so the grammars can't collide.
    assert find_placeholders("tap {keyword} <<x>> <<不>>") == []


def test_resolve_placeholders_fills_from_the_local_values_file() -> None:
    # The load-time contract: pack files keep <<TOKEN>>s on disk; the
    # values live ONLY in playbooks/placeholders.yml.
    write_placeholder_values({"CONTACT": "Alice"})

    out = resolve_placeholders('anchors: ["<<CONTACT>>"]', ValueError)

    assert out == 'anchors: ["Alice"]'


def test_resolve_placeholders_rejects_a_token_with_no_value() -> None:
    with pytest.raises(ValueError, match="placeholders.yml"):
        resolve_placeholders("x: <<UNSET>>", ValueError)


def test_resolve_placeholders_wraps_a_broken_values_file() -> None:
    # A malformed values file must fail loudly in the caller's error
    # class — silence would resurface as a baffling "unpopulated
    # placeholder" at every pack parse.
    from physiclaw.common import paths
    from physiclaw.common.text import write_text

    class Boom(Exception):
        pass

    path = paths.playbooks_dir() / PLACEHOLDER_VALUES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text(path, "- just\n- a list\n")

    with pytest.raises(Boom, match="mapping"):
        resolve_placeholders("plain text", Boom)


def test_write_placeholder_values_round_trips() -> None:
    write_placeholder_values({"CONTACT": "小乔", "CITY": "Hangzhou"})

    assert placeholder_values() == {"CONTACT": "小乔", "CITY": "Hangzhou"}
