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
