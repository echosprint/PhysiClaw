"""Tests for `physiclaw.conductor.spec.reply` — the gate's deterministic
reading: whole-message matching against the ask's own words, and
new-incoming-bubble detection."""

from __future__ import annotations

import pytest
from conductor_fakes import make_screen

from physiclaw.conductor.spec import reply

YES = frozenset(map(reply.normalize, ["好的", "嗯", "ok", "go ahead", "confirm"]))
NO = frozenset(map(reply.normalize, ["不用", "不要", "算了", "no thanks", "cancel"]))


def _new(rows, baseline, own, **kw):
    return reply.new_incoming(rows, baseline, own, **kw)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("好的", "confirm"),
        ("好的！", "confirm"),  # trailing punctuation stripped
        ("OK", "confirm"),  # casefold
        ("ｏｋ", "confirm"),  # full-width folds via NFKC
        ("go ahead", "confirm"),  # inner whitespace removed on both sides
        ("不用", "deny"),
        ("算了。", "deny"),
        ("No thanks", "deny"),
        ("等等", None),  # a hold is neither — undeclared
        ("好的，但是买两盒", None),  # qualifier → whole-message rule defers
        ("what's the price?", None),
    ],
)
def test_classify_whole_message_only(text: str, expected: str | None) -> None:
    assert reply.classify(text, YES, NO) == expected


def test_classify_all_deny_wins_and_partial_defers() -> None:
    assert reply.classify_all(["好的", "不要"], YES, NO) == "deny"
    assert reply.classify_all(["好的", "嗯"], YES, NO) == "confirm"
    # One unclassifiable message alongside a confirm defers — the model reads.
    assert reply.classify_all(["好的", "顺便查下天气"], YES, NO) is None
    assert reply.classify_all([], YES, NO) is None


def test_only_the_declared_words_count() -> None:
    # The conductor holds no word list of its own: an undeclared "yes"
    # spelling is unclassified, a declared one classifies.
    assert reply.classify("sure", YES, NO) is None
    assert reply.classify("sure", frozenset({"sure"}), NO) == "confirm"


def test_new_incoming_filters_side_baseline_and_own_ask() -> None:
    ask = "「买牛奶」已到付款页，合计 ¥45。回复 好的 确认支付，或 不用 取消。"
    screen = make_screen(
        ("MyChat", 0.5, 0.05),  # header (baseline)
        (ask[:20], 0.75, 0.3),  # our ask bubble, wrapped line (right side)
        ("好的", 0.25, 0.5),  # the NEW user reply (left side)
        ("已发送", 0.75, 0.6),  # our own new bubble — right side, excluded
    )

    new = _new(screen.rows, {"MyChat"}, ask)

    assert new == ["好的"]


def test_quoted_reply_words_are_never_swallowed_as_own_lines() -> None:
    # The ask quotes "confirm"/"cancel" (≥ _OWN_FRAGMENT_MIN chars); a
    # verbatim reply must still register — a swallowed yes reads as
    # silence and the gate suspend-loops past explicit consent.
    ask = 'Total ¥45. Reply "confirm" to pay, or "cancel" to stop.'
    screen = make_screen(
        ("MyChat", 0.5, 0.05),
        (ask[:20], 0.75, 0.3),
        ("confirm", 0.25, 0.5),
    )

    assert _new(screen.rows, {"MyChat"}, ask) == ["confirm"]


def test_bubbles_above_the_visible_ask_never_count() -> None:
    # A keyboard hides bubbles without touching the thread's anchors, so
    # the baseline can miss old history; when it dismisses, a stale "ok"
    # from another conversation resurfaces as "not in baseline". A real
    # reply is chronologically after the ask — below it on screen.
    ask = 'Total ¥45. Reply "ok" to pay, or "no" to cancel.'
    screen = make_screen(
        ("MyChat", 0.5, 0.05),
        ("ok", 0.25, 0.2),  # STALE — an old confirm above the ask
        (ask[:20], 0.75, 0.5),  # our ask bubble
        ("ok", 0.25, 0.8),  # the real reply, below the ask
    )

    new = _new(screen.rows, {"MyChat"}, ask)

    assert len(new) == 1  # only the reply below the ask

    # The re-ask deny sweep runs with the filter OFF — it must see what
    # arrived above a just-sent ask.
    swept = _new(screen.rows, {"MyChat"}, ask, after_ask=False)
    assert len(swept) == 2


def test_centered_timestamp_rows_are_not_incoming() -> None:
    # System rows (timestamps) sit centered ~0.5 — outside the incoming
    # band, or every fresh timestamp after a suspension burns an LLM check.
    screen = make_screen(
        ("昨天 14:32", 0.5, 0.2),
        ("好的", 0.25, 0.4),
    )

    assert _new(screen.rows, set(), "ask text") == ["好的"]


def test_reply_repeating_a_visible_word_below_the_ask_counts() -> None:
    # The user's earlier "好的" (to a previous ask) is still on screen and
    # in the baseline; their new "好的" below this ask is a reply all the
    # same — position, not the label set, decides below a visible ask.
    ask = "现在下单吗？回复 好的 或 不用"
    screen = make_screen(
        ("MyChat", 0.5, 0.05),
        ("好的", 0.25, 0.2),  # the old reply, above
        (ask, 0.75, 0.4),
        ("好的", 0.25, 0.6),  # the new reply, below
    )

    assert _new(screen.rows, {"MyChat", "好的"}, ask) == ["好的"]


def test_user_echo_of_the_ask_counts() -> None:
    # "好的 确认支付" is a substring of the ask; it is still the user's
    # bubble (left side, below the ask), never one of our own lines.
    ask = "合计 ¥45。回复 好的 确认支付，或 不用 取消。"
    screen = make_screen(
        ("MyChat", 0.5, 0.05), (ask, 0.75, 0.3), ("好的 确认支付", 0.25, 0.5)
    )

    assert _new(screen.rows, {"MyChat"}, ask) == ["好的 确认支付"]


def test_sweep_skips_the_just_sent_asks_own_band() -> None:
    # A short wrapped tail of OUR new ask can OCR left of center; the
    # deny sweep reads above the ask and must not read the ask itself.
    ask = "现在下单吗？回复 好的 或 不用"
    screen = make_screen(
        ("MyChat", 0.5, 0.05),
        ("cancel", 0.25, 0.2),  # sent while the walk was in the app
        (ask[:8], 0.75, 0.5),
        ("不用", 0.3, 0.52),  # the ask's own last line, left-aligned
    )

    assert _new(screen.rows, {"MyChat"}, ask, after_ask=False) == ["cancel"]
