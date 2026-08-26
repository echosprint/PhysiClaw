"""Tests for `physiclaw.agent.conductor.reply` — the gate's deterministic
tiers: whole-message word matching and new-incoming-bubble detection."""

from __future__ import annotations

import pytest
from conductor_fakes import make_screen

from physiclaw.agent.conductor import reply


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
        ("等等", None),  # a hold is neither — LLM tier
        ("好的，但是买两盒", None),  # qualifier → whole-message rule defers
        ("what's the price?", None),
    ],
)
def test_classify_whole_message_only(text: str, expected: str | None) -> None:
    assert reply.classify(text) == expected


def test_classify_all_deny_wins_and_partial_defers() -> None:
    assert reply.classify_all(["好的", "不要"]) == "deny"
    assert reply.classify_all(["好的", "嗯"]) == "confirm"
    # One unclassifiable message alongside a confirm defers to the LLM.
    assert reply.classify_all(["好的", "顺便查下天气"]) is None
    assert reply.classify_all([]) is None


def test_new_incoming_filters_side_baseline_and_own_ask() -> None:
    ask = "「买牛奶」已到付款页，合计 ¥45。回复 好的 确认支付，或 不用 取消。"
    screen = make_screen(
        ("MyChat", 0.5, 0.05),  # header (baseline)
        (ask[:20], 0.75, 0.3),  # our ask bubble, wrapped line (right side)
        ("好的", 0.25, 0.5),  # the NEW user reply (left side)
        ("已发送", 0.75, 0.6),  # our own new bubble — right side, excluded
    )

    new = reply.new_incoming(screen.rows, {"MyChat"}, ask)

    assert new == ["好的"]
