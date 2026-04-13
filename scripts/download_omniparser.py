"""Download OmniParser V2 icon detection model and convert to ONNX.

Requires `convert` dependency group: uv sync --group convert
Usage: uv run python scripts/download_omniparser.py
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


def main():
    if ONNX_PATH.exists():
        log.info(f"Already exists: {ONNX_PATH}")
        return

    # Download
    if not PT_PATH.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        log.info(f"Downloading model.pt ...")
        urllib.request.urlretrieve(PT_URL, PT_PATH)
        log.info(f"  {PT_PATH.stat().st_size / 1024 / 1024:.1f} MB")

    # Convert
    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ImportError:
        raise RuntimeError("Run `uv sync --group convert` first")
    log.info("Converting to ONNX ...")
    YOLO(str(PT_PATH)).export(format="onnx", imgsz=1280)
    exported = PT_PATH.with_suffix(".onnx")
    if exported != ONNX_PATH:
        exported.rename(ONNX_PATH)
    log.info(f"  {ONNX_PATH.stat().st_size / 1024 / 1024:.1f} MB")

    # Cleanup .pt
    PT_PATH.unlink(missing_ok=True)
    log.info("Done. Run `uv sync` to remove conversion deps.")


if __name__ == "__main__":
    main()
