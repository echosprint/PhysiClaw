"""Computer vision modules for PhysiClaw.

All image processing in the project lives here; import the submodule
you need (no package-level re-exports — several submodules pull heavy
deps like the OCR/ONNX stacks that shouldn't load as a side effect):

- primitives: preprocess (grayscale/HSV/blur/resize/crop stages),
  colors (HSV masks + calibrated range tables), blobs (color-blob
  centroids)
- detection: icon_detect, ocr, screen_match, grid_detect, keyboard,
  quality, change, watchdog
- rendering: render (watermark_index, annotate_elements)
- util: encode_jpeg, encode_view_jpeg (LLM-view size cap), decode_image,
  bbox validation, one-off diagnostics

Pure functions: frame in → results or annotated frame out. Zero hardware
dependencies. Independently testable.
"""
