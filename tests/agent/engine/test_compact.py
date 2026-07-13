"""Tests for `physiclaw.agent.engine.compact` — turn collapse + stub.

Covers:
  - placeholder constructors
  - collapse_old_turns: bootstrap state, threshold gating, summary
    accumulation across collapses, memory/skill artifact harvesting
  - drop_stale_screens: idempotency, latest preserved, gesture views
  - scale_image_bytes: small image passthrough, oversized scaled, decode
    failure fallback
  - small helpers: _content_to_text, _has_image, _stub_body,
    _format_artifact_text, _carry_items, _render_slot
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from physiclaw.agent.engine import compact
from physiclaw.agent.engine.compact import (
    MEMORY_HEADER,
    MEMORY_INITIAL,
    SKILLS_HEADER,
    SKILLS_INITIAL,
    SUMMARY_HEADER,
    SUMMARY_INITIAL,
    _carry_items,
    _content_to_text,
    _format_artifact_text,
    _has_image,
    _render_slot,
    _stub_body,
    collapse_old_turns,
    collapse_pending,
    drop_stale_screens,
    inject_checkpoint_tail,
    new_memory_placeholder,
    new_skills_placeholder,
    new_summary_placeholder,
    scale_image_bytes,
)
from physiclaw.agent.engine.dto import (
    AssistantMessage,
    FinishReason,
    ImageBlock,
    Message,
    SystemMessage,
    TextBlock,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

# ---------- placeholder constructors ----------


def test_new_summary_placeholder_initial() -> None:
    p = new_summary_placeholder()
    assert isinstance(p, UserMessage)
    assert p.content == SUMMARY_INITIAL


def test_new_skills_placeholder_initial() -> None:
    assert new_skills_placeholder().content == SKILLS_INITIAL


def test_new_memory_placeholder_falls_back_when_no_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physiclaw.agent.engine import memory

    monkeypatch.setattr(memory, "load_recent_entries", lambda n: "")

    assert new_memory_placeholder().content == MEMORY_INITIAL


def test_new_memory_placeholder_pre_populates_when_logs_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physiclaw.agent.engine import memory

    monkeypatch.setattr(
        memory,
        "load_recent_entries",
        lambda n: "[2026-04-28 09:00] hi",
    )
    monkeypatch.setattr(memory, "BOOTSTRAP_LOG_ENTRIES", 3)

    out = new_memory_placeholder().content

    assert isinstance(out, str)
    assert MEMORY_HEADER in out
    assert "read_logs" in out
    assert "[2026-04-28 09:00] hi" in out


# ---------- _carry_items / _render_slot ----------


def test_carry_items_returns_empty_for_non_string_input() -> None:
    assert _carry_items([], SUMMARY_HEADER, sep="\n") == []


def test_carry_items_returns_empty_when_header_missing() -> None:
    assert _carry_items("no header here", SUMMARY_HEADER, sep="\n") == []


def test_carry_items_returns_empty_for_initial_state() -> None:
    assert _carry_items(SUMMARY_INITIAL, SUMMARY_HEADER, sep="\n") == []


def test_carry_items_extracts_items_separated_by_newline() -> None:
    body = f"{SUMMARY_HEADER}\n- one\n- two\n- three"

    assert _carry_items(body, SUMMARY_HEADER, sep="\n") == ["- one", "- two", "- three"]


def test_carry_items_handles_double_newline_separator() -> None:
    body = f"{MEMORY_HEADER}\nfirst entry\n\nsecond entry"

    assert _carry_items(body, MEMORY_HEADER, sep="\n\n") == [
        "first entry",
        "second entry",
    ]


def test_render_slot_with_items() -> None:
    out = _render_slot(SUMMARY_HEADER, ["- a", "- b"], sep="\n")

    assert out == f"{SUMMARY_HEADER}\n- a\n- b"


def test_render_slot_empty_items_uses_none_yet() -> None:
    out = _render_slot(MEMORY_HEADER, [], sep="\n\n")

    assert out == f"{MEMORY_HEADER}\n(none yet)"


# ---------- _format_artifact_text ----------


def test_format_artifact_text_renders_args_as_sorted_json() -> None:
    out = _format_artifact_text("read_logs", {"entries": 5}, "log line")

    assert out == 'read_logs({"entries": 5}) →\nlog line'


def test_format_artifact_text_handles_unicode_args() -> None:
    out = _format_artifact_text("Skill", {"name": "微信"}, "body")

    assert "微信" in out


# ---------- _content_to_text / _has_image ----------


def test_content_to_text_string_passthrough() -> None:
    assert _content_to_text("hi") == "hi"


def test_content_to_text_first_text_block_in_multipart() -> None:
    out = _content_to_text(
        [ImageBlock(media_type="image/jpeg", data_b64="aGk="), TextBlock(text="cap")]
    )

    assert out == "cap"


def test_content_to_text_returns_empty_when_no_text_block_in_list() -> None:
    out = _content_to_text([ImageBlock(media_type="image/jpeg", data_b64="aGk=")])

    assert out == ""


def test_has_image_false_for_string() -> None:
    assert _has_image("text") is False


def test_has_image_true_when_image_block_present() -> None:
    assert (
        _has_image(
            [TextBlock(text="x"), ImageBlock(media_type="image/jpeg", data_b64="a")]
        )
        is True
    )


def test_has_image_false_when_only_text_blocks() -> None:
    assert _has_image([TextBlock(text="x")]) is False


# ---------- _stub_body ----------


def test_stub_body_keeps_text_rows_drops_icon_rows() -> None:
    text = (
        'id [kind] "label" [left,top,right,bottom] conf\n'
        '1 [icon] "" [0.1,0.1,0.2,0.2] 0.95\n'
        '2 [text] "Send" [0.5,0.8,0.6,0.9] 0.99\n'
        '3 [icon] "" [0.7,0.1,0.8,0.2] 0.90'
    )

    out = _stub_body(text)

    # Labels only: header, icon rows, id/tag/bbox/conf all gone — just "Send".
    assert out == "Send"
    assert "[text]" not in out  # kind tag dropped
    assert "[icon]" not in out
    assert "0.99" not in out  # confidence dropped
    assert "0.5,0.8" not in out  # bbox dropped


def test_stub_body_keeps_action_text_and_verdict() -> None:
    # A gesture view's text: action line first, then the listing.
    text = (
        "Tapped at bbox [0.9, 0.5, 0.98, 0.56] | screen: no visible change "
        "— read the attached view\n"
        'id [kind] "label" [left,top,right,bottom] conf\n'
        '1 [icon] "" [0.1,0.1,0.2,0.2] 0.95\n'
        '2 [text] "Add to Cart" [0.5,0.8,0.6,0.9] 0.99'
    )

    out = _stub_body(text)

    assert "screen: no visible change" in out  # retry history stays legible
    assert "Add to Cart" in out  # label survives...
    assert '"Add to Cart"' not in out  # ...but not the quoted-row form
    assert "[icon]" not in out
    assert "0.99" not in out  # confidence dropped
    assert "0.5,0.8" not in out  # row bbox dropped


def test_stub_body_drops_header_when_no_text_rows_survive() -> None:
    text = (
        'id [kind] "label" [left,top,right,bottom] conf\n'
        '1 [icon] "" [0.1,0.1,0.2,0.2] 0.95'
    )

    assert _stub_body(text) == ""


def test_stub_body_keeps_sequence_step_lines() -> None:
    # `1 tap ok — …` starts with a digit but is not an element row.
    text = (
        "1 tap ok — Tapped at bbox [0.1, 0.9, 0.7, 0.95]\n"
        "2 send_to_clipboard ok — Copied 'hi' to phone clipboard"
    )

    out = _stub_body(text)

    assert "1 tap ok" in out and "2 send_to_clipboard ok" in out


def test_stub_body_empty_for_empty_text() -> None:
    assert _stub_body("") == ""


# ---------- scale_image_bytes ----------


def _encode_jpg(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", arr)
    assert ok
    return buf.tobytes()


def test_scale_image_bytes_passthrough_when_within_max_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compact, "MAX_IMAGE_EDGE", 1000)
    img = np.full((300, 200, 3), 128, dtype=np.uint8)
    raw = _encode_jpg(img)

    out_bytes, mime = scale_image_bytes(raw)

    assert mime == "image/jpeg"
    decoded = cv2.imdecode(np.frombuffer(out_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (300, 200, 3)


def test_scale_image_bytes_scales_when_over_max_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compact, "MAX_IMAGE_EDGE", 100)
    img = np.full((300, 600, 3), 128, dtype=np.uint8)
    raw = _encode_jpg(img)

    out_bytes, mime = scale_image_bytes(raw)

    assert mime == "image/jpeg"
    decoded = cv2.imdecode(np.frombuffer(out_bytes, np.uint8), cv2.IMREAD_COLOR)
    # Long edge 600 → 100; aspect 2 preserved.
    assert max(decoded.shape[:2]) == 100


def test_scale_image_bytes_returns_input_on_decode_failure() -> None:
    raw = b"definitely not an image"

    out_bytes, mime = scale_image_bytes(raw)

    assert out_bytes == raw
    assert mime == "application/octet-stream"


# ---------- drop_stale_screens ----------


def _peek_pair(
    listing: str = 'id [kind] "label" [left,top,right,bottom] conf',
) -> list[Message]:
    """Build [asst-with-peek-call, tool_result-with-image]."""
    return [
        AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(id="t1", name="note", arguments={"summary": "x"}),
                ToolCall(id="t2", name="peek", arguments={}),
            ],
            finish_reason=FinishReason.TOOL_CALLS,
        ),
        ToolResultMessage(tool_call_id="t1", content="noted: x"),
        ToolResultMessage(
            tool_call_id="t2",
            content=[
                TextBlock(text=listing),
                ImageBlock(media_type="image/jpeg", data_b64="aGk="),
            ],
        ),
    ]


def test_drop_stale_screens_no_op_when_one_or_zero_obs_turns() -> None:
    msgs = [SystemMessage(content="x"), UserMessage(content="hi")]
    snapshot = list(msgs)

    drop_stale_screens(msgs)

    assert msgs == snapshot


def test_drop_stale_screens_stubs_earlier_obs_keeps_latest() -> None:
    listing = (
        'id [kind] "label" [left,top,right,bottom] conf\n'
        '1 [text] "Send" [0.5,0.8,0.6,0.9] 0.99\n'
    )
    msgs: list[Message] = [
        SystemMessage(content="sys"),
        *_peek_pair(listing),
        *_peek_pair(),
    ]

    drop_stale_screens(msgs)

    # Earlier peek's tool_result (index 3) is now superseded.
    earlier = msgs[3]
    assert isinstance(earlier, ToolResultMessage)
    assert earlier.is_superseded is True
    assert isinstance(earlier.content, str)
    assert "(superseded peek)" in earlier.content
    # Later peek's tool_result still has the image.
    latest = msgs[6]
    assert isinstance(latest, ToolResultMessage)
    assert latest.is_superseded is False
    assert _has_image(latest.content)


def _gesture_pair(tcid: str = "g1") -> list[Message]:
    """Build [asst-with-tap-call, tool_result-with-fused-view] — the
    gesture result text (verdict + hint) and listing share one TextBlock,
    as `mcp_blocks_to_content_blocks` produces."""
    text = (
        "Tapped at bbox [0.9, 0.5, 0.98, 0.56] | screen: changed — read the view\n"
        'id [kind] "label" [left,top,right,bottom] conf\n'
        '1 [text] "Add to Cart" [0.5,0.8,0.6,0.9] 0.99\n'
        '2 [icon] "" [0.1,0.1,0.2,0.2] 0.95'
    )
    return [
        AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(id=f"{tcid}-n", name="note", arguments={"summary": "x"}),
                ToolCall(
                    id=tcid, name="tap", arguments={"bbox": [0.9, 0.5, 0.98, 0.56]}
                ),
            ],
            finish_reason=FinishReason.TOOL_CALLS,
        ),
        ToolResultMessage(tool_call_id=f"{tcid}-n", content="noted: x"),
        ToolResultMessage(
            tool_call_id=tcid,
            content=[
                TextBlock(text=text),
                ImageBlock(media_type="image/jpeg", data_b64="aGk="),
            ],
        ),
    ]


def test_drop_stale_screens_stubs_gesture_view_keeps_latest_peek() -> None:
    # Gesture views participate in latest-screen-wins: an older tap view
    # is stubbed (action text + verdict + text rows survive) while the
    # newest view — here a peek — keeps its image.
    msgs: list[Message] = [
        SystemMessage(content="sys"),
        *_gesture_pair(),
        *_peek_pair(),
    ]

    drop_stale_screens(msgs)

    stale = msgs[3]
    assert isinstance(stale, ToolResultMessage)
    assert stale.is_superseded is True
    assert isinstance(stale.content, str)
    assert stale.content.startswith("(superseded tap)")
    assert "labels only, in order" in stale.content  # reminder on the marker
    assert "screen: changed" in stale.content
    assert "Add to Cart" in stale.content  # label survives (unquoted)
    assert '"Add to Cart"' not in stale.content  # quoted-row form gone
    assert "[icon]" not in stale.content
    latest = msgs[6]
    assert isinstance(latest, ToolResultMessage)
    assert _has_image(latest.content)


def test_drop_stale_screens_gesture_after_gesture() -> None:
    msgs: list[Message] = [
        SystemMessage(content="sys"),
        *_gesture_pair("g1"),
        *_gesture_pair("g2"),
    ]

    drop_stale_screens(msgs)

    assert isinstance(msgs[3].content, str)  # older stubbed
    assert _has_image(msgs[6].content)  # newest keeps its view


def test_drop_stale_screens_idempotent_on_second_pass() -> None:
    msgs: list[Message] = [SystemMessage(content=""), *_peek_pair(), *_peek_pair()]

    drop_stale_screens(msgs)
    snapshot = [(m.__class__, getattr(m, "content", None)) for m in msgs]
    drop_stale_screens(msgs)

    assert [(m.__class__, getattr(m, "content", None)) for m in msgs] == snapshot


# ---------- collapse_old_turns ----------


def test_collapse_old_turns_warns_when_slots_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    msgs: list[Message] = [SystemMessage(content="s"), UserMessage(content="u")]

    with caplog.at_level(logging.WARNING, logger="physiclaw.agent.engine.compact"):
        collapse_old_turns(msgs, first_at=10, interval=10, keep=5)

    assert any(
        "missing summary/memory/skill slots" in r.getMessage() for r in caplog.records
    )


def _scaffold_with_slots() -> list[Message]:
    return [
        SystemMessage(content="sys"),
        UserMessage(content="trigger"),
        new_summary_placeholder(),
        UserMessage(content=MEMORY_INITIAL),
        new_skills_placeholder(),
    ]


def test_collapse_no_op_when_below_first_at_threshold() -> None:
    msgs = _scaffold_with_slots()
    snapshot = list(msgs)

    collapse_old_turns(msgs, first_at=10, interval=5, keep=3)

    assert msgs == snapshot


def _note_turn(summary: str) -> list[Message]:
    """Synthetic [asst with `note(summary=...)`, tool_result] pair."""
    return [
        AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(
                    id=f"t-{summary}", name="note", arguments={"summary": summary}
                ),
            ],
            finish_reason=FinishReason.TOOL_CALLS,
        ),
        ToolResultMessage(tool_call_id=f"t-{summary}", content=f"noted: {summary}"),
    ]


def test_collapse_first_collapse_harvests_note_summaries_into_slot() -> None:
    msgs: list[Message] = _scaffold_with_slots()
    # Add 4 turns — first_at=3, keep=1 → 3 turns harvested, 1 kept.
    for s in ("step-a", "step-b", "step-c", "step-d"):
        msgs.extend(_note_turn(s))

    collapse_old_turns(msgs, first_at=3, interval=10, keep=1)

    # Slot at index 2 now contains the summaries.
    summary = msgs[2]
    assert isinstance(summary, UserMessage)
    assert isinstance(summary.content, str)
    assert SUMMARY_HEADER in summary.content
    assert "- step-a" in summary.content
    assert "- step-b" in summary.content
    assert "- step-c" in summary.content
    # Most recent step kept intact, NOT in the summary.
    assert "- step-d" not in summary.content


def test_collapse_no_op_when_no_salvageable_content() -> None:
    """Many turns but none have note/memory/skill calls → no-op."""
    msgs = _scaffold_with_slots()
    for i in range(5):
        msgs.append(
            AssistantMessage(
                content="",
                tool_calls=[ToolCall(id=f"t{i}", name="tap", arguments={})],
                finish_reason=FinishReason.TOOL_CALLS,
            )
        )
        msgs.append(ToolResultMessage(tool_call_id=f"t{i}", content="ok"))

    snapshot = [(m.__class__, getattr(m, "content", None)) for m in msgs]

    collapse_old_turns(msgs, first_at=3, interval=10, keep=1)

    assert [(m.__class__, getattr(m, "content", None)) for m in msgs] == snapshot


def test_collapse_harvests_memory_tool_results() -> None:
    msgs = _scaffold_with_slots()
    for i in range(3):
        msgs.append(
            AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(id=f"r{i}", name="read_memory", arguments={"key": f"k{i}"})
                ],
                finish_reason=FinishReason.TOOL_CALLS,
            )
        )
        msgs.append(ToolResultMessage(tool_call_id=f"r{i}", content=f"value-{i}"))
    msgs.extend(_note_turn("step-keep"))  # latest kept turn

    collapse_old_turns(msgs, first_at=3, interval=10, keep=1)

    memory_slot = msgs[3]
    assert isinstance(memory_slot, UserMessage)
    text = memory_slot.content
    assert isinstance(text, str)
    assert MEMORY_HEADER in text
    for i in range(3):
        assert f"value-{i}" in text


def test_collapse_harvests_skill_tool_results() -> None:
    msgs = _scaffold_with_slots()
    msgs.append(
        AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="s1", name="Skill", arguments={"name": "wechat"})],
            finish_reason=FinishReason.TOOL_CALLS,
        )
    )
    msgs.append(ToolResultMessage(tool_call_id="s1", content="WeChat workflow body"))
    for s in ("a", "b", "c"):
        msgs.extend(_note_turn(s))

    collapse_old_turns(msgs, first_at=3, interval=10, keep=1)

    skill_slot = msgs[4]
    assert isinstance(skill_slot, UserMessage)
    text = skill_slot.content
    assert isinstance(text, str)
    assert SKILLS_HEADER in text
    assert "WeChat workflow body" in text


def test_collapse_subsequent_collapse_uses_keep_plus_interval_threshold() -> None:
    msgs = _scaffold_with_slots()
    # Pre-set the summary slot to indicate first collapse already happened.
    msgs[2] = UserMessage(content=f"{SUMMARY_HEADER}\n- pre-existing")
    for s in [f"s{i}" for i in range(8)]:
        msgs.extend(_note_turn(s))

    # keep+interval = 1+5 = 6 turns needed before subsequent fires.
    collapse_old_turns(msgs, first_at=100, interval=5, keep=1)

    summary = msgs[2]
    assert isinstance(summary, UserMessage)
    text = summary.content
    assert isinstance(text, str)
    assert "- pre-existing" in text  # carried forward
    # Older steps harvested; latest kept.
    assert "- s0" in text


def test_collapse_skips_when_artifact_result_is_error() -> None:
    msgs = _scaffold_with_slots()
    msgs.append(
        AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="s1", name="Skill", arguments={"name": "x"})],
            finish_reason=FinishReason.TOOL_CALLS,
        )
    )
    msgs.append(
        ToolResultMessage(
            tool_call_id="s1",
            content="oops",
            is_error=True,
        )
    )
    for s in ("a", "b", "c"):
        msgs.extend(_note_turn(s))

    collapse_old_turns(msgs, first_at=3, interval=10, keep=1)

    skill_slot = msgs[4]
    assert isinstance(skill_slot, UserMessage)
    # Error skipped — no skill artifact in slot.
    assert "oops" not in str(skill_slot.content)


# ---------- collapse_pending / checkpoint tail ----------


def test_collapse_pending_false_when_slots_missing() -> None:
    msgs: list[Message] = [SystemMessage(content="s"), UserMessage(content="u")]
    assert collapse_pending(msgs, first_at=1, interval=1, keep=1) is False


def test_collapse_pending_predicts_first_collapse_exactly() -> None:
    # first_at=3: with 2 complete turns the UPCOMING turn is the 3rd —
    # its completion triggers the collapse, so pending is True; one
    # turn earlier it isn't.
    msgs = _scaffold_with_slots()
    msgs.extend(_note_turn("a"))
    assert collapse_pending(msgs, first_at=3, interval=10, keep=1) is False
    msgs.extend(_note_turn("b"))
    assert collapse_pending(msgs, first_at=3, interval=10, keep=1) is True


def test_collapse_pending_agrees_with_collapse_trigger() -> None:
    # The prediction and the collapse fire on the same turn: pending
    # True → one more complete turn → collapse_old_turns folds.
    msgs = _scaffold_with_slots()
    for s in ("a", "b"):
        msgs.extend(_note_turn(s))
    assert collapse_pending(msgs, first_at=3, interval=10, keep=1) is True
    msgs.extend(_note_turn("c"))
    collapse_old_turns(msgs, first_at=3, interval=10, keep=1)
    assert SUMMARY_HEADER in msgs[2].content
    assert "- a" in msgs[2].content


def test_collapse_pending_switches_threshold_after_first_collapse() -> None:
    # After the first collapse the summary slot is rewritten and the
    # trigger switches to keep+interval — pending must switch with it.
    msgs = _scaffold_with_slots()
    for s in ("a", "b", "c", "d"):
        msgs.extend(_note_turn(s))
    collapse_old_turns(msgs, first_at=3, interval=10, keep=1)
    # 1 turn kept; threshold now keep+interval=11 → far from pending.
    assert collapse_pending(msgs, first_at=3, interval=10, keep=1) is False
    for i in range(9):
        msgs.extend(_note_turn(f"t{i}"))
    # 10 complete turns; the upcoming one is the 11th → pending.
    assert collapse_pending(msgs, first_at=3, interval=10, keep=1) is True


def test_inject_checkpoint_tail_appends_notice_last() -> None:
    msgs = _scaffold_with_slots()

    out = inject_checkpoint_tail(msgs, keep=7)

    assert out is not msgs and len(out) == len(msgs) + 1
    tail = out[-1]
    assert isinstance(tail, UserMessage)
    assert "compresses after this turn" in tail.content
    assert "newest 7" in tail.content
    assert "scratchpad" in tail.content


# ---------- robustness: _stub_body × the real formatter ----------


def test_stub_body_against_real_format_elements_output() -> None:
    """The format coupling test: `_LISTING_HEADER` / `_ROW_RE` mirror
    `core.vision.util.format_elements` — run the REAL formatter's output
    through `_stub_body` so a formatter change breaks here, not in prod."""
    from physiclaw.core.vision.util import format_elements

    listing = format_elements(
        [
            {
                "id": 0,
                "kind": "icon",
                "label": "",
                "bbox": [0.1, 0.1, 0.2, 0.2],
                "conf": 0.95,
            },
            {
                "id": 1,
                "kind": "text",
                "label": "加入购物车",
                "bbox": [0.5, 0.8, 0.6, 0.9],
                "conf": 0.99,
            },
            {
                "id": 2,
                "kind": "text",
                "label": 'He said "hi" [ok]',
                "bbox": [0.1, 0.3, 0.4, 0.35],
                "conf": 0.80,
            },
            {
                "id": 3,
                "kind": "icon",
                "label": "",
                "bbox": [0.7, 0.1, 0.8, 0.2],
                "conf": 0.90,
            },
        ]
    )
    text = "Tapped at bbox [0.5, 0.8, 0.6, 0.9] | screen: changed — hint\n" + listing

    out = _stub_body(text)

    assert "screen: changed" in out  # action line survives
    assert "加入购物车" in out  # CJK label survives
    assert 'He said "hi" [ok]' in out  # label w/ quotes+brackets peeled off whole
    assert "[icon]" not in out  # icon rows dropped
    assert "[text]" not in out  # kind tag dropped
    assert compact._LISTING_HEADER not in out  # header dropped
    assert "0.99" not in out  # bbox + confidence dropped
    # Labels emitted in listing order, one per line.
    assert '加入购物车\nHe said "hi" [ok]' in out


def test_stub_body_real_formatter_icon_only_listing_drops_header() -> None:
    from physiclaw.core.vision.util import format_elements

    listing = format_elements(
        [
            {
                "id": 0,
                "kind": "icon",
                "label": "",
                "bbox": [0.1, 0.1, 0.2, 0.2],
                "conf": 0.95,
            },
        ]
    )

    assert _stub_body(listing) == ""


# ---------- robustness: _stub_body properties (hypothesis) ----------


# Lines that are NOT element rows and NOT the listing header — action
# results, hints, warnings, sequence step lines, arbitrary text. The
# blacklist covers every `str.splitlines` boundary, not just `\n` —
# hypothesis found that e.g. `\x1e` inside a "line" gets re-split (a
# pre-existing, benign normalization shared with the old filter).
_LINE_BREAKS = "\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"
_non_row_line = st.text(
    alphabet=st.characters(
        blacklist_categories=("Cs",),
        blacklist_characters=_LINE_BREAKS,
    ),
    max_size=80,
).filter(lambda s: not compact._ROW_RE.match(s) and s != compact._LISTING_HEADER)


@given(st.lists(_non_row_line, max_size=12))
def test_stub_body_preserves_all_non_listing_lines(lines) -> None:
    text = "\n".join(lines)

    out = _stub_body(text)

    # Every non-listing line survives verbatim (modulo outer strip).
    assert out == text.strip()


def _row(id_: int, kind: str, label: str) -> str:
    return f'{id_} [{kind}] "{label}" [0.100,0.200,0.300,0.400] 0.90'


@given(
    kinds=st.lists(st.sampled_from(["icon", "text"]), min_size=1, max_size=10),
    label=st.text(
        # No line-breaks (they'd split a row) and no `"` (label delimiter),
        # so each row round-trips to exactly its label.
        alphabet=st.characters(
            blacklist_categories=("Cs",),
            blacklist_characters=_LINE_BREAKS + '"',
        ),
        min_size=1,
        max_size=20,
        # No boundary whitespace: `_stub_body` strips the joined output, which
        # would trim the first/last label's edges.
    ).filter(lambda s: s == s.strip()),
)
def test_stub_body_keeps_exactly_the_text_rows(kinds, label) -> None:
    # Icon rows carry an empty label in real formatter output.
    rows = [_row(i, k, label if k == "text" else "") for i, k in enumerate(kinds)]
    text = compact._LISTING_HEADER + "\n" + "\n".join(rows)

    out = _stub_body(text)

    # One line per text row, each exactly the label, icon rows dropped,
    # order preserved.
    assert out.splitlines() == [label] * kinds.count("text")


@given(
    label=st.text(
        alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters='"'),
        min_size=1,
        max_size=20,
    ).filter(lambda s: s == s.strip()),
)
def test_stub_body_emitted_label_never_contains_a_newline(label) -> None:
    """No emitted label carries a newline. `_stub_body` matches rows against
    individual `splitlines()` lines, so a label field with an embedded
    line break splits the row and can't round-trip as one label — it falls
    through to the preamble instead. Locks the invariant against a future
    switch to `re.DOTALL` / away from `splitlines()`."""
    out = _stub_body(compact._LISTING_HEADER + "\n" + _row(0, "text", label))
    if any(brk in label for brk in _LINE_BREAKS):
        assert label not in out.splitlines()  # split apart, not a clean label
    else:
        assert out == label  # newline-free label round-trips exactly


@given(st.text(max_size=400))
def test_stub_body_idempotent_and_total(text) -> None:
    # Never raises on arbitrary input, and stubbing a stub is stable.
    once = _stub_body(text)
    assert _stub_body(once) == once


# ---------- robustness: drop_stale_screens under an engine-like loop ----------


def _view_result(tcid: str, text: str) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=tcid,
        content=[
            TextBlock(text=text),
            ImageBlock(media_type="image/jpeg", data_b64="aGk="),
        ],
    )


def test_drop_stale_screens_engine_loop_invariant() -> None:
    """Simulate _loop: append a view-bearing turn, run drop_stale_screens
    after every turn (as the engine does). Invariant at every point:
    exactly ONE image in history; every earlier view is a superseded
    string stub; no stub ever regains an image."""
    msgs: list[Message] = [SystemMessage(content="sys")]
    tools = ["tap", "peek", "swipe", "sequence", "screenshot", "go_back"]
    for turn in range(30):
        tool = tools[turn % len(tools)]
        tcid = f"t{turn}"
        msgs.append(
            AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(id=f"{tcid}-n", name="note", arguments={"summary": "x"}),
                    ToolCall(id=tcid, name=tool, arguments={}),
                ],
                finish_reason=FinishReason.TOOL_CALLS,
            )
        )
        msgs.append(ToolResultMessage(tool_call_id=f"{tcid}-n", content="noted: x"))
        msgs.append(
            _view_result(
                tcid,
                f"{tool} result | screen: changed\n"
                f"{compact._LISTING_HEADER}\n"
                f'1 [text] "Send" [0.5,0.8,0.6,0.9] 0.99',
            )
        )

        drop_stale_screens(msgs)

        imaged = [
            m
            for m in msgs
            if isinstance(m, ToolResultMessage) and _has_image(m.content)
        ]
        assert len(imaged) == 1, f"turn {turn}: {len(imaged)} images in history"
        stubs = [
            m for m in msgs if isinstance(m, ToolResultMessage) and m.is_superseded
        ]
        assert len(stubs) == turn  # every prior view stubbed, none skipped
        for s in stubs:
            assert isinstance(s.content, str)
            assert s.content.startswith("(superseded ")
            assert "screen: changed" in s.content  # verdict survives stubbing
            assert "Send" in s.content  # label survives stubbing
            assert '"Send"' not in s.content  # but not the quoted-row form


def test_drop_stale_screens_batch_stubs_all_but_last() -> None:
    """One call over MANY accumulated views must stub every earlier one —
    not just the first. (Mutation testing caught that incremental-call
    tests never exercise 3+ unstubbed views at once: `[:-1]` vs `[:1]`
    are indistinguishable with only two.)"""
    msgs: list[Message] = [SystemMessage(content="sys")]
    for turn in range(5):
        tcid = f"t{turn}"
        msgs.append(
            AssistantMessage(
                content="",
                tool_calls=[ToolCall(id=tcid, name="tap", arguments={})],
                finish_reason=FinishReason.TOOL_CALLS,
            )
        )
        msgs.append(_view_result(tcid, f"Tapped {turn} | screen: changed"))

    drop_stale_screens(msgs)  # single batch call, 5 views pending

    results = [m for m in msgs if isinstance(m, ToolResultMessage)]
    assert [m.is_superseded for m in results] == [True, True, True, True, False]
    assert sum(1 for m in results if _has_image(m.content)) == 1
    assert _has_image(results[-1].content)


def test_drop_stale_screens_preserves_is_error_flag() -> None:
    msgs: list[Message] = [
        SystemMessage(content="s"),
        AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="e1", name="tap", arguments={})],
            finish_reason=FinishReason.TOOL_CALLS,
        ),
        ToolResultMessage(
            tool_call_id="e1",
            content=[
                TextBlock(text="boom"),
                ImageBlock(media_type="image/jpeg", data_b64="aGk="),
            ],
            is_error=True,
        ),
        *_peek_pair(),
    ]

    drop_stale_screens(msgs)

    stub = msgs[2]
    assert isinstance(stub, ToolResultMessage)
    assert stub.is_error is True
    assert stub.is_superseded is True


def test_drop_stale_screens_orphan_tool_call_id_falls_back_to_view() -> None:
    # A result whose id matches no assistant tool_call (shouldn't happen,
    # but must not crash) stubs with the generic header.
    msgs: list[Message] = [
        SystemMessage(content="s"),
        ToolResultMessage(
            tool_call_id="ghost",
            content=[
                TextBlock(text="x"),
                ImageBlock(media_type="image/jpeg", data_b64="aGk="),
            ],
        ),
        *_peek_pair(),
    ]

    drop_stale_screens(msgs)

    stub = msgs[1]
    assert isinstance(stub, ToolResultMessage)
    assert stub.content.startswith("(superseded view)")


def test_drop_stale_screens_composes_with_collapse_old_turns() -> None:
    # The engine runs drop_stale_screens then collapse_old_turns each
    # turn — verify the pair leaves slots intact and exactly one image.
    msgs = _scaffold_with_slots()
    for turn in range(12):
        tcid = f"t{turn}"
        msgs.append(
            AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"{tcid}-n", name="note", arguments={"summary": f"s{turn}"}
                    ),
                    ToolCall(id=tcid, name="tap", arguments={}),
                ],
                finish_reason=FinishReason.TOOL_CALLS,
            )
        )
        msgs.append(ToolResultMessage(tool_call_id=f"{tcid}-n", content="noted"))
        msgs.append(_view_result(tcid, "Tapped | screen: changed"))
        drop_stale_screens(msgs)
        collapse_old_turns(msgs, first_at=6, interval=4, keep=2)

    imaged = [
        m for m in msgs if isinstance(m, ToolResultMessage) and _has_image(m.content)
    ]
    assert len(imaged) == 1
    summary = msgs[2]
    assert isinstance(summary, UserMessage)
    assert SUMMARY_HEADER in str(summary.content)
    assert "- s0" in str(summary.content)  # folded turns' notes harvested


def test_scale_image_bytes_jpeg_within_cap_passes_through_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The server already sized this view to the shared knob — a re-encode
    # would only stack a second generation of JPEG loss onto screen text.
    monkeypatch.setattr(compact, "MAX_IMAGE_EDGE", 1000)
    raw = _encode_jpg(np.full((300, 200, 3), 128, dtype=np.uint8))

    out_bytes, mime = scale_image_bytes(raw)

    assert out_bytes is raw
    assert mime == "image/jpeg"


def test_scale_image_bytes_png_within_cap_reencoded_to_jpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compact, "MAX_IMAGE_EDGE", 1000)
    ok, buf = cv2.imencode(".png", np.full((300, 200, 3), 128, dtype=np.uint8))
    assert ok
    raw = buf.tobytes()

    out_bytes, mime = scale_image_bytes(raw)

    assert mime == "image/jpeg"
    assert out_bytes.startswith(b"\xff\xd8\xff")
