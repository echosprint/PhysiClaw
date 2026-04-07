"""Computer vision modules for PhysiClaw.

All modules in this package are pure CV: frame in → results out.
Zero hardware dependencies. Independently testable.
"""

from physiclaw.vision.phone_detect import PhoneDetector
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

__all__ = [
    "PhoneDetector",
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
]
