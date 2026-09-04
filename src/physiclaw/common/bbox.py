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

# The named bands a check's `within:` may spell instead of a box — the
# one vocabulary the macro grammar (`within: top`) and the pack grammar
# (an anchor pinned to a band) share. Here because macros never import
# the conductor, and the two must not drift.
BANDS: dict[str, Bbox] = {
    "top": (0.0, 0.0, 1.0, 0.25),
    "bottom": (0.0, 0.75, 1.0, 1.0),
    "left": (0.0, 0.0, 0.3, 1.0),
    "right": (0.7, 0.0, 1.0, 1.0),
}


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


def parse_box(value) -> Bbox:
    """A declared `[left, top, right, bottom]` → the canonical tuple.
    `validate_bbox` plus the one check a YAML author can trip that a
    tool call cannot: a bool is not a coordinate. Raises ValueError."""
    if isinstance(value, (list, tuple)) and any(isinstance(v, bool) for v in value):
        raise ValueError(f"bbox: each coord must be a number; got {value!r}")
    left, top, right, bottom = (float(v) for v in validate_bbox(value))
    return (left, top, right, bottom)


def parse_within(value) -> Bbox:
    """A check's `within:` — a band name (`BANDS`) or a box. The ONE
    reader of "where to look", shared by the macro grammar's checks and
    the pack grammar's anchors. Raises ValueError."""
    if isinstance(value, str):
        if value not in BANDS:
            raise ValueError(
                f"must be one of {', '.join(BANDS)} or a [left, top, right, bottom] box"
            )
        return BANDS[value]
    return parse_box(value)


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
