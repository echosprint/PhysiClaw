"""Image ingress cap for LLM-bound payloads.

Lives in `common` because both sides of the provider/engine boundary
need the cap (`provider.wire` scales MCP images on ingress; the server
sizes its views to the same `compact.max_image_edge_px` knob) — when it
lived in `agent.engine.compact`, the provider package had to import the
engine's behavior module for one pure function.
"""

import logging

from physiclaw.common import config

log = logging.getLogger(__name__)

_JPEG_MAGIC = b"\xff\xd8\xff"  # JPEG SOI marker


def scale_image_bytes(raw: bytes) -> tuple[bytes, str]:
    """Decode, scale long-edge to `compact.max_image_edge_px` if larger,
    re-encode JPEG.

    A JPEG already within the cap (the steady state — the server sizes
    views to the same knob) passes through byte-identical rather than
    stacking a second generation of JPEG loss onto on-screen text.

    Returns (bytes, mime_type). On decode/encode failure returns the input
    unchanged with a generic mime so the caller still has something to
    forward — context bloat is preferable to a dropped screenshot.
    """
    # Lazy cv2/numpy: common stays import-light. Config reads go through
    # the module — `set_dotted`/`unset_dotted` rebind `config.CONFIG`.
    import cv2
    import numpy as np

    max_edge = config.CONFIG.compact.max_image_edge_px
    quality = config.CONFIG.compact.jpeg_quality
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        log.warning("scale_image_bytes: decode failed on %d bytes", len(raw))
        return raw, "application/octet-stream"
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_edge and raw.startswith(_JPEG_MAGIC):
        return raw, "image/jpeg"
    if long_edge > max_edge:
        scale = max_edge / long_edge
        new_size = (int(round(w * scale)), int(round(h * scale)))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        log.warning("scale_image_bytes: encode failed")
        return raw, "application/octet-stream"
    return buf.tobytes(), "image/jpeg"
