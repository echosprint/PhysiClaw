"""Tests for `physiclaw.conductor.money` — the declared total and the
fire-time predicates, pure arithmetic over a screen."""

from __future__ import annotations

from conductor_fakes import make_screen

from physiclaw.conductor import money


def test_declared_total_reads_the_label_row_itself() -> None:
    screen = make_screen(("原价 ¥79", 0.5, 0.4), ("合计: ¥59.9", 0.5, 0.9))

    assert money.declared_total(screen, ("合计",)) == 59.9


def test_declared_total_joins_a_split_row_within_radius() -> None:
    # OCR split the footer: "合计" and its amount sit side by side.
    screen = make_screen(("合计", 0.2, 0.9), ("¥59.9", 0.3, 0.9), ("¥79", 0.5, 0.2))

    assert money.declared_total(screen, ("合计",)) == 59.9


def test_declared_total_ignores_a_far_amount() -> None:
    screen = make_screen(("合计", 0.2, 0.9), ("¥79", 0.8, 0.2))

    assert money.declared_total(screen, ("合计",)) is None


def test_declared_total_needs_the_label() -> None:
    screen = make_screen(("¥45", 0.5, 0.5))

    assert money.declared_total(screen, ("合计", "总计")) is None
    assert (
        money.declared_total(make_screen(("总计 ¥45", 0.5, 0.5)), ("合计", "总计"))
        == 45
    )


def test_fire_block_requires_consent_then_the_same_sheet() -> None:
    sheet = make_screen(("合计 ¥45", 0.5, 0.5))

    assert "without a confirmed total" in (
        money.fire_block(consented=None, screen=sheet) or ""
    )
    assert money.fire_block(consented=45.0, screen=sheet) is None
    changed = make_screen(("合计 ¥60", 0.5, 0.5))
    assert "sheet changed" in (money.fire_block(consented=45.0, screen=changed) or "")
    above = make_screen(("合计 ¥45", 0.5, 0.5), ("¥99", 0.5, 0.7))
    assert "exceed" in (money.fire_block(consented=45.0, screen=above) or "")
