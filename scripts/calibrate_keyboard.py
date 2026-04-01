"""
Keyboard calibration script — detect keys and generate UI preset template.

Usage:
    uv run python scripts/calibrate_keyboard.py                          # all images in data/image/keyboard/
    uv run python scripts/calibrate_keyboard.py data/image/keyboard/foo.png  # single image

Outputs:
    data/image/keyboard/bbox/bbox_<name>.png — screenshot with numbered key boxes
    .claude/ui-presets/system-keyboard.md     — preset template with positions filled in

QWERTY letters (rows 1-3), digits, shift, and delete are auto-labeled.
Bottom row and numeric symbol keys marked ??? need to be filled in from the bounding box images.
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from physiclaw.keyboard import (
    detect_key_boxes, draw_detected_keys,
    label_keyboard, generate_preset,
)

logging.basicConfig(level=logging.DEBUG, format="%(message)s")

parser = argparse.ArgumentParser(description="Detect keyboard keys and generate preset")
parser.add_argument("images", nargs="*",
                    help="Image paths (default: all in data/image/keyboard/)")
parser.add_argument("--output", default="data/image/keyboard/bbox",
                    help="Bounding box image output directory (default: data/image/keyboard/bbox/)")
args = parser.parse_args()

# Collect images
if args.images:
    image_paths = [Path(p) for p in args.images]
else:
    img_dir = Path("data/image/keyboard")
    if not img_dir.exists():
        print(f"Error: {img_dir} does not exist")
        sys.exit(1)
    image_paths = sorted(img_dir.glob("*.*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in (".png", ".jpg", ".jpeg")]

if not image_paths:
    print("No images found")
    sys.exit(1)

out_dir = Path(args.output)
out_dir.mkdir(parents=True, exist_ok=True)

pages = {}
bbox_images = {}

for img_path in image_paths:
    print(f"\n{'='*60}")
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"Error: cannot read {img_path}")
        continue

    h, w = frame.shape[:2]
    print(f"Image: {img_path.name} ({w}x{h})")

    # Detect and label
    rows = label_keyboard(frame)
    if rows is None:
        print("No keys detected")
        continue

    # Count keys and determine page type
    row_counts = [len(row) for row in rows]
    total = sum(row_counts)
    is_numeric = len(row_counts) >= 2 and row_counts[0] == 10 and row_counts[1] == 10
    page_name = "Numeric Keyboard" if is_numeric else "Alpha Keyboard"
    print(f"Page: {page_name} ({total} keys, rows: {row_counts})")

    # Save bounding box image
    boxes, bg = detect_key_boxes(frame)
    bbox_img = draw_detected_keys(frame, boxes, bg)
    bbox_path = out_dir / f"bbox_{img_path.stem}.png"
    cv2.imwrite(str(bbox_path), bbox_img)
    print(f"Bounding box image: {bbox_path}")

    # Keep first detection per page type (skip duplicates)
    if page_name not in pages:
        pages[page_name] = rows
        bbox_images[page_name] = str(bbox_path)
    else:
        print(f"  (skipped — {page_name} already captured)")

    # Print summary
    for i, row in enumerate(rows):
        labels = [k["element"] for k in row]
        print(f"  Row {i+1}: {' '.join(labels)}")

# Generate preset
if pages:
    preset = generate_preset(pages, bbox_images)
    preset_dir = Path(".claude/ui-presets")
    preset_dir.mkdir(parents=True, exist_ok=True)
    preset_path = preset_dir / "system-keyboard.md"
    preset_path.write_text(preset)
    # Save a reference copy for verifying positions after ??? are filled
    ref_path = out_dir / "system-keyboard.ref.md"
    ref_path.write_text(preset)
    print(f"\n{'='*60}")
    print(f"Preset written to {preset_path}")
    print(f"Reference copy saved to {ref_path}")
    print(f"Keys marked ??? need to be filled in from the bounding box images.")
    print()
    print(preset)
