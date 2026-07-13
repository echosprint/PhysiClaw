"""Press-geometry helpers shared by the engine's screen-facing guards.

`stuck.py` imports `screen_layout.py`, so these lived as private copies
in both — this leaf breaks the cycle. The two tolerances are deliberately
separate names even though the values coincide today: MATCH_TOLERANCE
decides "same target" for loop counting, LINT_MARGIN decides "inside
this learned box" for the layout lint — they may drift apart
legitimately.
"""

# Two press centers within this L∞ distance are the same target —
# covers the coordinate jitter of re-transcribed bboxes without
# swallowing a genuinely different neighbor element.
MATCH_TOLERANCE = 0.02

# Slack around a learned keyboard box before a press counts as inside it.
LINT_MARGIN = 0.02


def center_of(bbox) -> tuple[float, float] | None:
    """Center of a [left, top, right, bottom] bbox; None on garbage."""
    try:
        left, top, right, bottom = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    return (left + right) / 2, (top + bottom) / 2


def near(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """True if two press centers are the same target (L∞ ≤ MATCH_TOLERANCE)."""
    return abs(a[0] - b[0]) <= MATCH_TOLERANCE and abs(a[1] - b[1]) <= MATCH_TOLERANCE


def inside(center: tuple[float, float], bbox: list) -> bool:
    """True if center lies within bbox expanded by LINT_MARGIN."""
    left, top, right, bottom = bbox
    return (
        left - LINT_MARGIN <= center[0] <= right + LINT_MARGIN
        and top - LINT_MARGIN <= center[1] <= bottom + LINT_MARGIN
    )
