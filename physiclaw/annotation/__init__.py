"""Annotation system — browser-based UI element labeling.

Two workflows:
  1. User draws boxes manually in the browser
  2. Agent proposes boxes → user reviews/edits/confirms → agent acts
"""

from physiclaw.annotation.bbox import (
    AGENT_COLOR,
    LINE_ASPECT_RATIO,
    classify_bbox,
)
from physiclaw.annotation.state import AnnotationState
from physiclaw.annotation.handler import (
    serve_annotate_page,
    freeze_snapshot,
    get_frozen_snapshot,
    handle_annotations,
    handle_confirm,
)

__all__ = [
    "AGENT_COLOR",
    "LINE_ASPECT_RATIO",
    "classify_bbox",
    "AnnotationState",
    "serve_annotate_page",
    "freeze_snapshot",
    "get_frozen_snapshot",
    "handle_annotations",
    "handle_confirm",
]
