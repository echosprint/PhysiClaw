"""Central orchestrator for PhysiClaw.

The PhysiClaw class owns hardware lifecycle (arm, camera, calibration)
and bbox workflow state. Image-output helpers (drawing, encoding,
watermarking) live in physiclaw.core.vision.render.
"""

from physiclaw.core.orchestration.orchestrator import GestureResult, PhysiClaw

__all__ = ["GestureResult", "PhysiClaw"]
