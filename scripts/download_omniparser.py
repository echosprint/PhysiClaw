"""Download OmniParser V2 icon detection model and convert to ONNX.

Downloads model.pt from microsoft/OmniParser-v2.0, converts to ONNX
using ultralytics, then removes the .pt file. The result is a standalone
.onnx file that can be loaded with OpenCV DNN (no torch at runtime).

Requires (install temporarily for conversion):
    pip install ultralytics

Usage:
    uv run python scripts/download_omniparser.py
"""

import logging
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "data" / "model" / "omniparser_icon_detect"
PT_PATH = MODEL_DIR / "model.pt"
ONNX_PATH = MODEL_DIR / "model.onnx"

PT_URL = "https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_detect/model.pt"


def download_pt():
    """Download the V2 icon detect model.pt from HuggingFace."""
    if PT_PATH.exists():
        log.info(f"Already exists: {PT_PATH}")
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading model.pt ({PT_URL}) ...")
    urllib.request.urlretrieve(PT_URL, PT_PATH)
    size_mb = PT_PATH.stat().st_size / (1024 * 1024)
    log.info(f"  Saved to {PT_PATH} ({size_mb:.1f} MB)")


def convert_to_onnx():
    """Convert model.pt to model.onnx using ultralytics."""
    if ONNX_PATH.exists():
        log.info(f"Already exists: {ONNX_PATH}")
        return
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError(
            "ultralytics is required for conversion.\n"
            "  pip install ultralytics\n"
            "You only need it once — uninstall after conversion."
        )
    log.info("Converting to ONNX ...")
    model = YOLO(str(PT_PATH))
    model.export(format="onnx", imgsz=1280)
    # ultralytics saves the onnx next to the .pt file
    exported = PT_PATH.with_suffix(".onnx")
    if exported != ONNX_PATH:
        exported.rename(ONNX_PATH)
    size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)
    log.info(f"  Saved to {ONNX_PATH} ({size_mb:.1f} MB)")


def cleanup_pt():
    """Remove model.pt after successful conversion."""
    if PT_PATH.exists() and ONNX_PATH.exists():
        PT_PATH.unlink()
        log.info(f"Removed {PT_PATH}")


def main():
    download_pt()
    convert_to_onnx()
    cleanup_pt()
    log.info("Done.")


if __name__ == "__main__":
    main()
