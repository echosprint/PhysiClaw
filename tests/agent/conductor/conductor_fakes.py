"""Shared builders for the conductor test files (the `engine_fakes`
pattern: sibling module, imported bare thanks to pytest's rootdir path)."""

from __future__ import annotations

from physiclaw.common.listing import Element, Screen, format_elements

# One bbox convention for every fake row: ±0.05 × ±0.02 around the center.
BOX_W, BOX_H = 0.05, 0.02


def make_screen(*rows: tuple) -> Screen:
    """Rows are (label, cx, cy) or (label, cx, cy, conf)."""
    els = []
    for i, row in enumerate(rows):
        label, cx, cy = row[0], row[1], row[2]
        conf = row[3] if len(row) > 3 else 0.9
        els.append(
            Element(
                id=i,
                kind="text",
                label=label,
                bbox=(cx - BOX_W, cy - BOX_H, cx + BOX_W, cy + BOX_H),
                conf=conf,
            )
        )
    return Screen.read(format_elements(els))


# One canonical demo pack for the pack-consuming test files (playbook,
# program): two declared pages, two enabled macros.

PAGES = """\
home:
  anchors: ["Files"]
results:
  anchors: ["综合"]
"""

PACK_MACRO = """\
name: {name}
description: test leg
inputs:
  message:
    description: text to use
steps:
  - name: go
    tool: tap
    with: {{bbox: [0.1, 0.1, 0.2, 0.2]}}
"""


def write_pack(
    app: str = "demo",
    *,
    macros: tuple[str, ...] = ("open-app", "add-cart"),
    playbooks: dict[str, str] | None = None,
):
    """Write a pack under the (fixture-scoped) playbooks dir; returns its
    root."""
    from physiclaw.common import paths

    root = paths.playbooks_dir() / app
    (root / "macros").mkdir(parents=True, exist_ok=True)
    (root / "pages.yml").write_text(PAGES, encoding="utf-8")
    for m in macros:
        d = root / "macros" / m
        d.mkdir(parents=True, exist_ok=True)
        (d / "MACRO.yml").write_text(PACK_MACRO.format(name=m), encoding="utf-8")
    for name, text in (playbooks or {}).items():
        (root / f"{name}.yml").write_text(text, encoding="utf-8")
    return root


# The canonical ledger playbook — the full P8 stack (list input,
# next_item loop, RECONCILE, payment gate with revise) — shared by the
# parser and walk test files so grammar changes are edited ONCE.
LEDGERED = """\
name: shop
description: shop the list
inputs:
  items:
    description: the buying list
    kind: list
mandate:
  max_amount: 100
nodes:
  - id: open
    type: LEG
    macro: open-app
    with: {message: "cart"}
    verify: home
  - id: search
    type: LEG
    macro: open-app
    with: {message: "{item.query}"}
    verify: results
  - id: choose
    type: DECIDE
    call: choose_item
    with: {criteria: "cheapest {item.query}"}
    on: {pick: add, scroll: choose, none_fit: escalate, escalate: escalate}
  - id: add
    type: LEG
    macro: add-cart
    with: {message: "add"}
    verify: results
  - id: advance
    type: DECIDE
    call: next_item
    with: {picked: "{choose.pick}"}
    on: {next: search, done: fix}
  - id: fix
    type: RECONCILE
    page: results
  - id: sheet
    type: LEG
    macro: open-app
    with: {message: "sheet"}
    verify: results
  - id: gate
    type: HUMAN_GATE
    gate: payment
    compose: payment-request
    message: "List ready, total ¥{gate.total}. Reply ok to pay, or no to cancel."
    over_message: "Total ¥{gate.total}, over the ¥{gate.cap} budget. Reply ok to pay, or no to cancel."
    return: open-app
    revise: advance
  - id: pay
    type: LEG
    macro: add-cart
    with: {message: "pay"}
    enter: results
    verify: home
    irreversible: payment
"""
