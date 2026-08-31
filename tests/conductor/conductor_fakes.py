"""Shared builders for the conductor test files (the `engine_fakes`
pattern: sibling module, imported bare thanks to pytest's rootdir path)."""

from __future__ import annotations

from textwrap import indent

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


# The three screens every driver test needs, and the loop contract for
# feeding a synthesized turn's result back. One home: the conductor's own
# `turns.py` centralized this on the src side, so the tests must not
# re-spell "the result is keyed to tool_calls[1]" per file.
ELSEWHERE = make_screen(("Nothing known", 0.5, 0.5)).text


def thread_screen(*bubbles: tuple) -> str:
    """The user-channel thread, anchored on the demo contact name."""
    return make_screen(("MyChat", 0.5, 0.05), *bubbles).text


def history() -> list:
    from physiclaw.contract.dto import SystemMessage, UserMessage

    return [SystemMessage(content="sys"), UserMessage(content="wake")]


def feed(history: list, turn, text: str = "", *, error: bool = False) -> None:
    """Append the synthesized turn plus its ACTION's tool result — the
    loop's contract (one result per call, in the very next messages)."""
    from physiclaw.contract.dto import ToolResultMessage

    history.append(turn)
    history.append(
        ToolResultMessage(
            tool_call_id=turn.tool_calls[1].id, content=text, is_error=error
        )
    )


def finish(driver, history: list, step) -> str:
    """The terminal contract: a handover/completion/quit mints ONE final
    synthesized [note, peek] brief turn; feed its peek result and the
    driver is permanently quiet. Returns the brief's note summary so
    tests can assert on the report itself."""
    assert step is not None, "expected the terminal brief turn, got quiet"
    assert step.synthesized and step.tool_names() == ["note", "peek"]
    feed(history, step, ELSEWHERE)
    assert driver.advance(history) is None
    return step.tool_calls[0].arguments["summary"]


# One canonical demo pack for the pack-consuming test files (playbook,
# program): two declared pages, two enabled macros.

PAGES = """\
home:
  anchors: ["Files"]
results:
  anchors: ["综合"]
done:
  anchors: ["AllDone"]
"""

PACK_MACRO = """\
name: {name}
description: test leg
inputs:
  message:
    description: text to use
    default: hi
steps:
  - name: go
    tool: tap
    with: {{label: t, bbox: [0.1, 0.1, 0.2, 0.2]}}
"""


def compose_pack_doc(
    app: str, pages: str, playbooks: dict[str, str] | None = None
) -> str:
    """One unified PLAYBOOK.yml from the fixtures' historical pieces —
    pages text + full playbook docs, indented under their map keys (the
    keys ARE the names; entries carry no inner `name:`)."""
    doc = f"app: {app}\ndescription: test pack\npages:\n{indent(pages, '  ')}\n"
    if playbooks:
        doc += "playbooks:\n"
        for name, text in playbooks.items():
            doc += f"  {name}:\n{indent(text, '    ')}\n"
    return doc


def write_pack(
    app: str = "demo",
    *,
    macros: tuple[str, ...] = ("open-app", "add-cart"),
    playbooks: dict[str, str] | None = None,
    pages: str = PAGES,
):
    """Write a pack under the (fixture-scoped) playbooks dir; returns its
    root."""
    from physiclaw.common import paths

    root = paths.playbooks_dir() / app
    (root / "macros").mkdir(parents=True, exist_ok=True)
    (root / "PLAYBOOK.yml").write_text(
        compose_pack_doc(app, pages, playbooks), encoding="utf-8"
    )
    for m in macros:
        d = root / "macros" / m
        d.mkdir(parents=True, exist_ok=True)
        (d / "MACRO.yml").write_text(PACK_MACRO.format(name=m), encoding="utf-8")
    return root


# The canonical ledger playbook — the full P8 stack (list input,
# next_item loop, sync, payment ask with revise) — shared by the
# parser and walk test files so grammar changes are edited ONCE. Route
# form: `goto` hops from the start page to `results` so the loop head
# (`search`) ENTERS at the loop's own page — the backward `next:` arm
# re-runs it from `results` on every later iteration, and a derived
# enter of `home` would rescue instead of searching.
LEDGERED = """\
description: shop the list
inputs:
  items:
    description: the buying list
    type: list
budget: 100
route:
  - page: home
  - do: goto
    macro: open-app
    with: {message: "go"}
  - page: results
  - do: search
    macro: open-app
    with: {message: "{item.query}"}
  - page: results
  - decide: choose
    uses: choose_item
    with: {criteria: "cheapest {item.query}"}
    routes: {pick: add, scroll: choose, none_fit: escalate, escalate: escalate}
  - do: add
    macro: add-cart
    with: {message: "add"}
  - page: results
  - decide: advance
    uses: next_item
    with: {picked: "{choose.pick}"}
    routes: {next: search, done: fix}
  - sync: fix
  - do: sheet
    macro: open-app
    with: {message: "sheet"}
  - page: results
  - ask: gate
    approve: payment
    message: "List ready, total ¥{ask.total}. Reply ok to pay, or no to cancel."
    over_budget_message: "Total ¥{ask.total}, over the ¥{ask.cap} budget. Reply ok to pay, or no to cancel."
    return: open-app
    revise: advance
  - do: pay
    macro: add-cart
    with: {message: "pay"}
    irreversible: payment
  - page: home
"""
