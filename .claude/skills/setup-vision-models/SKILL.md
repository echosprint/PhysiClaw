---
name: setup-vision-models
description: Download and prepare OmniParser icon detection model for the screenshot() tool. One-time setup — installs temporary deps, converts model, then cleans up.
allowed-tools: Bash, Read
---

# Setup Vision Models

One-time setup for icon detection and OCR in `screenshot()`. RapidOCR and onnxruntime are already in project dependencies — this only sets up the OmniParser ONNX model.

## Step 1: Check

```bash
ls data/model/omniparser_icon_detect/model.onnx 2>/dev/null && echo "OK" || echo "MISSING"
```

If OK, tell the user it's already set up and stop.

## Step 2: Install, convert, clean up

```bash
uv sync --group convert
uv run python scripts/download_omniparser.py
uv sync
```

## Step 3: Verify

```bash
uv run python -c "
from physiclaw.vision.icon_detect import IconDetector
from physiclaw.vision.ocr import OCRReader
IconDetector(); OCRReader()
print('OK')
"
```

Tell the user setup is complete.
