"""Tests for `physiclaw.conductor.walk.money` — the declared total and the
fire-time predicates, pure arithmetic over a screen."""

from __future__ import annotations

from conductor_fakes import make_screen

from physiclaw.conductor.walk import money


def test_declared_total_reads_the_label_row_itself() -> None:
    screen = make_screen(("原价 ¥79", 0.5, 0.4), ("合计: ¥59.9", 0.5, 0.9))

    assert money.declared_total(screen, ("合计",)) == 59.9


def test_declared_total_joins_a_split_row_on_the_same_line() -> None:
    # OCR split the footer: "合计" and its amount sit side by side.
    screen = make_screen(("合计", 0.2, 0.9), ("¥59.9", 0.3, 0.9), ("¥79", 0.5, 0.2))

    assert money.declared_total(screen, ("合计",)) == 59.9


def test_declared_total_ignores_a_far_amount() -> None:
    screen = make_screen(("合计", 0.2, 0.9), ("¥79", 0.8, 0.2))

    assert money.declared_total(screen, ("合计",)) is None


def test_declared_total_never_reads_a_row_above_the_bare_label() -> None:
    # The rig's Taobao order sheet, 2026-09-03: the pay button reads the
    # exact label 免密支付 with no price on its line, and a 顺手买 add-on
    # (¥15.80) sits a hand above it — closer than the 实付 row at the
    # top. The quoted total is the 实付 row's, never the add-on's.
    sheet = make_screen(
        ("实付￥24.75 优惠前￥45", 0.55, 0.245),
        ("五常大米5kg真空锁鲜 每千克￥4.95", 0.42, 0.42),
        ("顺手买·天猫维达集团旗舰店", 0.27, 0.655),
        ("￥15.80￥49.90", 0.44, 0.77),
        ("免密支付", 0.50, 0.935),
    )

    assert money.declared_total(sheet, ("实付", "免密支付", "优惠后")) == 24.75


def test_amounts_read_at_most_two_decimals() -> None:
    # The rig's sheet, 2026-09-03: OCR ran "实付￥24.75" into the next
    # word ("1优惠前…"), and the ask quoted ¥24.751. A price has two
    # decimals; the glued digit is not part of it.
    screen = make_screen(
        ("实付￥24.751优惠前￥45", 0.5, 0.25), ("共减￥12.501开88VIP", 0.5, 0.3)
    )

    assert money.amounts(screen) == [24.75, 45.0, 12.5]
    assert money.declared_total(screen, ("实付",)) == 24.75


def test_declared_total_takes_the_nearest_amount_on_the_line() -> None:
    # OCR split the label from its amount, and the struck-through
    # original price shares the line further right: the amount beside
    # the label is the nearest one, not the first in listing order.
    sheet = make_screen(
        ("优惠前￥62.5", 0.62, 0.27),
        ("实付", 0.20, 0.27),
        ("￥50", 0.33, 0.27),
    )

    assert money.declared_total(sheet, ("实付",)) == 50


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
        money.fire_block(consented=None, seen=(), screen=sheet) or ""
    )
    assert money.fire_block(consented=45.0, seen=(45.0,), screen=sheet) is None
    changed = make_screen(("合计 ¥60", 0.5, 0.5))
    assert "sheet changed" in (
        money.fire_block(consented=45.0, seen=(45.0,), screen=changed) or ""
    )
    above = make_screen(("合计 ¥45", 0.5, 0.5), ("¥99", 0.5, 0.7))
    assert "appeared after the ask" in (
        money.fire_block(consented=45.0, seen=(45.0,), screen=above) or ""
    )


def test_fire_block_allows_an_amount_the_user_already_saw() -> None:
    # A struck-through original price above the total was on the sheet
    # the user consented to — it is not a change. Only an amount above
    # the total that APPEARED since the ask blocks.
    sheet = make_screen(("原价 ¥79", 0.5, 0.4), ("合计 ¥59.9", 0.5, 0.9))
    seen = tuple(money.amounts(sheet))
    assert money.declared_total(sheet, ("合计",)) == 59.9

    assert money.fire_block(consented=59.9, seen=seen, screen=sheet) is None
    grew = make_screen(
        ("原价 ¥79", 0.5, 0.4), ("合计 ¥59.9", 0.5, 0.9), ("¥120", 0.5, 0.7)
    )
    assert "120.0" in (money.fire_block(consented=59.9, seen=seen, screen=grew) or "")


def test_amounts_read_thousands_separators() -> None:
    sheet = make_screen(("合计 ¥1,234.56", 0.5, 0.9), ("¥12", 0.5, 0.3))

    assert money.declared_total(sheet, ("合计",)) == 1234.56
    assert money.amounts(sheet) == [1234.56, 12.0]


def test_declared_total_prefers_the_exact_label_and_the_footer() -> None:
    # "商品合计" is a subtotal wearing the total's characters; the payable
    # total is the bare label, and among bare labels the lowest row.
    sheet = make_screen(
        ("商品合计 ¥79", 0.5, 0.3), ("合计: ¥59.9", 0.5, 0.9), ("合计 ¥10", 0.5, 0.2)
    )

    assert money.declared_total(sheet, ("合计",)) == 59.9
