"""Tests for `physiclaw.contract.wire` — the reader half of the codec:
request messages as leaf blocks, texts, and persisted image refs."""

from __future__ import annotations

import json

from physiclaw.contract import wire


def _wire(tmp_path, records):
    p = tmp_path / "wire.jsonl"
    p.write_text(
        "\n".join(json.dumps(r) for r in records) + "\nnot json\n", encoding="utf-8"
    )
    return p


def test_messages_flatten_tool_results_and_skip_non_requests(tmp_path) -> None:
    p = _wire(
        tmp_path,
        [
            {"kind": "session_start"},
            {
                "kind": "request",
                "messages": [
                    {"role": "system", "content": "doctrine"},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "ref",
                                            "ref": "images/a.jpg",
                                        },
                                    },
                                    {"type": "text", "text": "listing"},
                                ],
                            },
                            "stray string",
                        ],
                    },
                ],
            },
            "a string record",
        ],
    )

    out = list(wire.iter_request_messages(p))

    assert out == [
        ("system", [{"type": "text", "text": "doctrine"}]),
        (
            "user",
            [
                {"type": "image", "source": {"type": "ref", "ref": "images/a.jpg"}},
                {"type": "text", "text": "listing"},
            ],
        ),
    ]
    assert list(wire.iter_request_texts(p)) == [
        ("system", "doctrine"),
        ("user", "listing"),
    ]


def test_image_ref_reads_both_scrubbed_shapes_only() -> None:
    assert (
        wire.image_ref({"type": "image_url", "image_url": {"url": "images/a.jpg"}})
        == "images/a.jpg"
    )
    assert (
        wire.image_ref(
            {"type": "image", "source": {"type": "ref", "ref": "images/b.jpg"}}
        )
        == "images/b.jpg"
    )
    # Unscrubbed data, a byte-count stub, and text are not refs.
    assert (
        wire.image_ref(
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,xx"}}
        )
        is None
    )
    assert (
        wire.image_ref({"type": "image", "source": {"type": "base64", "byte_count": 3}})
        is None
    )
    assert wire.image_ref({"type": "text", "text": "images/a.jpg"}) is None


def test_scrub_and_image_ref_round_trip() -> None:
    persisted = lambda mime, data: "images/x.jpg"  # noqa: E731
    for block in (
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aGk="}},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": "aGk="},
        },
    ):
        assert wire.image_ref(wire.scrub_block(block, persisted)) == "images/x.jpg"
