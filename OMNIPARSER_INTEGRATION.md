# OmniParser V2 Integration Plan

## Why

The current bbox_target/confirm_bbox flow asks the AI agent (Claude) to visually judge whether a colored rectangle covers a UI element in a camera photo. This fails reliably because:

1. Claude is a language model, not a vision model. It cannot judge pixel-level bounding box alignment from a photo. It guesses based on spatial reasoning ("backspace is on the right, so I'll pick the rightmost rectangle") rather than actually seeing what's inside each rectangle.

2. Prompt engineering has a ceiling. We've tried: verification framing ("name what's inside"), rejection rules ("partial = miss"), self-check questions ("am I choosing because it covers or because it's closest"), concrete failure examples. The agent still confirms bad bounding boxes because the underlying task — judging visual overlap — is not something a language model can do well.

3. The real fix is to stop asking Claude to judge rectangles. Instead, give it a named list of detected elements with coordinates, and let it pick by name.

OmniParser V2 is Microsoft's open-source screen parsing tool. It combines a fine-tuned YOLOv8 Nano model for detecting interactable UI elements (buttons, keys, icons, text fields), PaddleOCR for reading text labels, and a Florence2 model for describing icons. It takes a screenshot as input and returns a structured list of every UI element with its label and bounding box coordinates. V2 specifically improved detection of small elements like keyboard keys, which is exactly our weak spot.

## Architecture change

Current flow (agent judges rectangles):
```
park() → screenshot() → agent estimates percentages → bbox_target() draws rectangles
→ agent tries to judge if rectangles cover target → confirm_bbox() → tap()
```

New flow (agent picks from named list):
```
park() → screenshot() → OmniParser parses the screen → agent receives:
  [{"label": "h", "bbox": [50, 84, 56, 89]},
   {"label": "backspace ⌫", "bbox": [88, 84, 96, 89]},
   {"label": "Send", "bbox": [80, 72, 95, 76]}, ...]
→ agent says tap("backspace") → system looks up coordinates → tap()
```

The agent never sees rectangles. It picks from names. The forced-choice and partial-coverage problems disappear entirely.

## Implementation

1. Install OmniParser V2 as a Python dependency. Models are ~300MB total (YOLOv8 Nano + Florence2 + PaddleOCR). Can run on CPU, GPU preferred for speed.

2. Add a `parse_screen()` method to the PhysiClaw core that: crops and rectifies the phone screen from the camera frame (using existing homography), runs OmniParser on the rectified image, maps the detected bounding box coordinates back to screen percentages.

3. Add a new MCP tool `detect_elements()` that calls `parse_screen()` and returns the element list as structured text. The agent calls this instead of guessing percentages for bbox_target.

4. Add a new MCP tool `tap_element(label)` that looks up the element by label from the last `detect_elements()` call, computes the center point, and taps. This replaces the bbox_target → confirm_bbox → tap sequence for most cases.

5. Keep bbox_target/confirm_bbox as a fallback for when OmniParser misses an element or the agent needs to tap an unlabeled region. But the primary flow should be detect_elements → tap_element.

## Model hosting

Run OmniParser in-process on the PhysiClaw server. The YOLOv8 Nano model is small enough for CPU inference with acceptable latency (~200-500ms per frame). If a GPU is available, latency drops to ~50ms. No external API calls needed — everything runs locally.
