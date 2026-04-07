"""Bounding box helpers — classification by aspect ratio."""

# Agent-proposed boxes use this color (deep orange — not in user palette)
AGENT_COLOR = "#ff6d00"

# Aspect ratio threshold for classifying boxes as column/row
LINE_ASPECT_RATIO = 10


def classify_bbox(bbox: list[float]) -> tuple[str, list[float]]:
    """Classify a bbox by aspect ratio and strip the irrelevant axis.

    Args:
        bbox: [left, top, right, bottom] as 0-1 decimals

    Returns:
        (type, coords) where:
        - "box"    → [left, top, right, bottom]
        - "column" → [left, right]   (X interval, tall thin)
        - "row"    → [top, bottom]   (Y interval, wide thin)
    """
    w = abs(bbox[2] - bbox[0])
    h = abs(bbox[3] - bbox[1])
    if h > 0 and w > 0:
        if h / w > LINE_ASPECT_RATIO:
            return "column", [bbox[0], bbox[2]]
        if w / h > LINE_ASPECT_RATIO:
            return "row", [bbox[1], bbox[3]]
    return "box", bbox
