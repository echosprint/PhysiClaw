"""Computer vision modules for PhysiClaw.

All image processing in the project lives here:
- detection: color_segment, icon_detect, ocr, screen_match,
  grid_detect, keyboard, detect (multi-tool orchestration)
- rendering: render (draw_bbox, draw_grid_overlay, watermark_index,
  process_annotations, encode_jpeg)

Pure functions: frame in → results or annotated frame out. Zero hardware
dependencies. Independently testable.
"""

from physiclaw.vision.color_segment import detect_color_blocks, ColorBlob
from physiclaw.vision.icon_detect import IconDetector, Element
from physiclaw.vision.ocr import OCRReader, TextResult
from physiclaw.vision.screen_match import (
    match_screen,
    match_best,
    frames_differ,
    detect_dark_overlay,
    MatchResult,
)
from physiclaw.vision.keyboard import detect_key_boxes, label_keyboard
from physiclaw.vision.render import (
    draw_bbox,
    draw_grid_overlay,
    watermark_index,
    process_annotations,
    encode_jpeg,
)

__all__ = [
    # detection
    "detect_color_blocks",
    "ColorBlob",
    "IconDetector",
    "Element",
    "OCRReader",
    "TextResult",
    "match_screen",
    "match_best",
    "frames_differ",
    "detect_dark_overlay",
    "MatchResult",
    "detect_key_boxes",
    "label_keyboard",
    # rendering
    "draw_bbox",
    "draw_grid_overlay",
    "watermark_index",
    "process_annotations",
    "encode_jpeg",
]
