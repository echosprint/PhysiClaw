"""Bbox validation + geometry primitives — the shared contract between
core and agent.

Both layers police the same `[left, top, right, bottom]` 0-1 screen
box: the engine validator rejects malformed LLM tool arguments before
dispatch, and the orchestrator re-checks at gesture time as
defense-in-depth before any GRBL move. One implementation here means
the agent sees the same diagnostic regardless of which layer catches
the violation.

The primitives (`center_of` / `near` / `inside`) are tolerance-neutral:
callers own their slack and pass it in — policy knobs stay with the
policy, only the math lives here.

Dependency-free on purpose: raises plain ValueError; each layer wraps
or re-exports into its own error surface (`agent.engine.validator`
converts to ValidationError; `core.vision.util` re-exports as-is).
"""

# The canonical 4-tuple spelling of the same box — element rows
# (`common.listing.Element`) and macro region clauses share it.
Bbox = tuple[float, float, float, float]


def validate_bbox(bbox: list[float]) -> list[float]:
    """Raise ValueError if bbox is malformed; return `bbox` unchanged.

    Checks run in a fixed order — shape, element type, range, ordering —
    and the messages are pinned by tests on both sides of the MCP
    boundary; reword only with a coordinated change.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"bbox: must be [left, top, right, bottom]; got {bbox!r}")
    if not all(isinstance(v, (int, float)) for v in bbox):
        raise ValueError(f"bbox: each coord must be a number; got {bbox!r}")
    left, top, right, bottom = bbox
    if any(v < 0 or v > 1 for v in bbox):
        raise ValueError(
            f"bbox: each coord must be in [0, 1]; got [{left}, {top}, {right}, {bottom}]"
        )
    if left >= right or top >= bottom:
        raise ValueError(
            f"bbox: left < right, top < bottom; got [{left}, {top}, {right}, {bottom}]"
        )
    return bbox


def center_of(bbox) -> tuple[float, float] | None:
    """Center of a [left, top, right, bottom] bbox; None on garbage."""
    try:
        left, top, right, bottom = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    return (left + right) / 2, (top + bottom) / 2


def near(a: tuple[float, float], b: tuple[float, float], *, tolerance: float) -> bool:
    """True if two points are within `tolerance` of each other (L∞)."""
    return abs(a[0] - b[0]) <= tolerance and abs(a[1] - b[1]) <= tolerance


def inside(center: tuple[float, float], bbox: list, *, margin: float) -> bool:
    """True if `center` lies within `bbox` expanded by `margin` per side
    (0.0 = exact containment). Like `near`'s tolerance, the margin is
    required: it is the caller's policy, never a hidden default."""
    left, top, right, bottom = bbox
    return (
        left - margin <= center[0] <= right + margin
        and top - margin <= center[1] <= bottom + margin
    )
