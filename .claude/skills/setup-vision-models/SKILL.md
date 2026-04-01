---
name: setup-vision-models
description: Download and prepare OmniParser icon detection model and RapidOCR for the detect_elements() tool. One-time setup — installs temporary deps, converts models, then cleans up.
allowed-tools: Bash, Read, Write
---

# Setup Vision Models

One-time setup for the `detect_elements()` MCP tool. Downloads the OmniParser V2 icon detection model and installs RapidOCR. No permanent dependency changes to pyproject.toml.

## Step 1: Check current state

```bash
ls -la data/model/omniparser_icon_detect/model.onnx 2>/dev/null && echo "OmniParser ONNX: OK" || echo "OmniParser ONNX: MISSING"
uv run --group vision python -c "from rapidocr import RapidOCR; print('RapidOCR: OK')" 2>/dev/null || echo "RapidOCR: MISSING"
```

If both are OK, tell the user everything is already set up and stop.

## Step 2: OmniParser icon detection model

The model needs to be downloaded as a .pt file and converted to ONNX. This requires `ultralytics` temporarily.

```bash
# Install temporary conversion deps (NOT in pyproject.toml)
uv pip install ultralytics onnx 'onnxslim>=0.1.71' onnxruntime

# Download and convert
uv run python scripts/download_omniparser.py
```

Verify the ONNX file exists:

```bash
ls -lh data/model/omniparser_icon_detect/model.onnx
```

Then remove the temporary conversion deps (ultralytics pulls in torch which is huge):

```bash
uv sync --group vision
```

This removes all packages not declared in pyproject.toml (ultralytics, torch, etc.) while keeping `onnxruntime` and `rapidocr` (declared in the `vision` dependency group). The ONNX model file stays on disk.

## Step 3: RapidOCR

RapidOCR uses PaddleOCR models on ONNX Runtime. It auto-downloads its own small models on first use. It's declared in the `vision` dependency group in pyproject.toml.

```bash
uv sync --group vision
```

Verify:

```bash
uv run python -c "
from rapidocr import RapidOCR
ocr = RapidOCR()
print('RapidOCR: OK')
"
```

## Step 4: End-to-end test

Run a quick test on an existing screenshot to confirm both detectors work:

```bash
uv run python -c "
import cv2
from pathlib import Path

# Find a screenshot to test with
imgs = sorted(Path('data/snapshot').glob('*.jpg'))
imgs = [p for p in imgs if '_grid' not in p.name and '_bbox' not in p.name and '_icons' not in p.name and '_ocr' not in p.name]
if not imgs:
    print('No test images in data/snapshot/ — skipping end-to-end test')
    exit(0)

img = cv2.imread(str(imgs[0]))
print(f'Testing with {imgs[0].name} ({img.shape[1]}x{img.shape[0]})')

from physiclaw.icon_detect import IconDetector
detector = IconDetector()
elements = detector.detect(img, confidence=0.2)
print(f'Icon detection: {len(elements)} elements found')

from physiclaw.ocr import OCRReader
reader = OCRReader()
texts = reader.read(img)
print(f'OCR: {len(texts)} text regions found')

print('All good.')
"
```

Tell the user the setup is complete. The `detect_elements()` tool is now available in the MCP server.
