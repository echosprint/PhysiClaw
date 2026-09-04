"""Tests for `physiclaw.conductor.drive.corpus` — wire.jsonl listing
extraction and the labeled-corpus round trip."""

from __future__ import annotations

import json

import pytest

from physiclaw.common import paths
from physiclaw.common.listing import LISTING_HEADER
from physiclaw.conductor.drive import corpus

LISTING_A = f'{LISTING_HEADER}\n0 [text] "综合" [0.100,0.100,0.200,0.120] 0.90'
LISTING_B = f'{LISTING_HEADER}\n0 [text] "结算" [0.500,0.900,0.600,0.940] 0.90'


def _write_wire(sid: str, lines: list[dict]) -> None:
    d = paths.engine_sessions_dir() / sid
    d.mkdir(parents=True)
    (d / "wire.jsonl").write_text(
        "\n".join(json.dumps(rec, ensure_ascii=False) for rec in lines) + "\n",
        encoding="utf-8",
    )


def test_session_listings_extracts_deduped_in_order() -> None:
    _write_wire(
        "20260101-000000-aaaaaa",
        [
            {"kind": "session_start", "tools": []},
            {
                "kind": "request",
                "turn": 0,
                "messages": [
                    # System quotes the header in doctrine — must not extract.
                    {"role": "system", "content": f"doctrine: {LISTING_HEADER}"},
                    # Header without any parseable row (prose/stub) — skipped.
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": LISTING_HEADER}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": LISTING_A},
                            {"type": "image", "ref": "images/x.jpg"},
                        ],
                    },
                ],
            },
            {
                "kind": "request",
                "turn": 1,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": LISTING_A}]},
                    # Anthropic wire shape: the listing rides inside a
                    # tool_result block's own content list.
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": [
                                    {"type": "text", "text": LISTING_B},
                                    {"type": "ref", "ref": "images/y.jpg"},
                                ],
                            }
                        ],
                    },
                ],
            },
            {"kind": "response", "turn": 1, "raw": {}},
        ],
    )

    out = corpus.session_listings("20260101-000000-aaaaaa")

    assert out == [LISTING_A, LISTING_B]


def test_session_listings_missing_session_raises() -> None:
    with pytest.raises(FileNotFoundError):
        corpus.session_listings("nope")


def test_corpus_round_trip(tmp_path) -> None:
    items = [
        corpus.CorpusItem(label="taobao.results", listing=LISTING_A),
        corpus.CorpusItem(label="other", listing=LISTING_B),
    ]
    p = tmp_path / "corpus.jsonl"

    corpus.write_corpus(p, items)

    assert corpus.read_corpus(p) == items


def test_read_corpus_names_bad_line(tmp_path) -> None:
    p = tmp_path / "corpus.jsonl"
    p.write_text('{"label": "a"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="bad corpus line"):
        corpus.read_corpus(p)


def test_partition_splits_genuine_and_negatives() -> None:
    items = [
        corpus.CorpusItem(label="taobao.results", listing=LISTING_A),
        corpus.CorpusItem(label="taobao.unknown-page", listing=LISTING_B),
        corpus.CorpusItem(label="other", listing=LISTING_B),
        corpus.CorpusItem(label=corpus.UNLABELED, listing=LISTING_B),
    ]

    by_page, negatives = corpus.partition(items, "taobao", {"results"})

    assert set(by_page) == {"results"} and len(by_page["results"]) == 1
    # a label that names no declared page counts as a hard negative;
    # '?' lines are ignored entirely.
    assert len(negatives) == 2
