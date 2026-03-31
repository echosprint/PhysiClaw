"""
Keyboard calibration script — detect key bounding boxes from screenshots.

Usage:
    uv run python scripts/calibrate_keyboard.py                          # all images in data/image/keyboard/
    uv run python scripts/calibrate_keyboard.py data/image/keyboard/foo.png  # single image

Outputs per image:
    data/keyboard/debug_<name>.png   — screenshot with numbered key boxes
    data/keyboard/boxes_<name>.txt   — box coordinates as text

Show debug images to Claude to label each numbered box and write
.claude/ui-presets/system-keyboard.md for the AI agent to use.
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from physiclaw.keyboard import detect_key_boxes, draw_detected_keys, boxes_to_text

logging.basicConfig(level=logging.DEBUG, format="%(message)s")

parser = argparse.ArgumentParser(description="Detect keyboard key bounding boxes")
parser.add_argument("images", nargs="*",
                    help="Image paths (default: all in data/image/keyboard/)")
parser.add_argument("--output", default="data/keyboard",
                    help="Output directory (default: data/keyboard/)")
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

for img_path in image_paths:
    print(f"\n{'='*60}")
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"Error: cannot read {img_path}")
        continue

    h, w = frame.shape[:2]
    print(f"Image: {img_path.name} ({w}x{h})")

    boxes = detect_key_boxes(frame)
    if not boxes:
        print("No keys detected")
        continue

    # Save debug image
    debug = draw_detected_keys(frame, boxes)
    debug_path = out_dir / f"debug_{img_path.stem}.png"
    cv2.imwrite(str(debug_path), debug)
    print(f"Debug image: {debug_path}")

    # Save box coordinates
    text = boxes_to_text(boxes)
    txt_path = out_dir / f"boxes_{img_path.stem}.txt"
    txt_path.write_text(text)
    print(f"Box listing: {txt_path}")
    print(text)

print(f"\n{'='*60}")
print(f"Done. {len(image_paths)} images processed → {out_dir}/")
