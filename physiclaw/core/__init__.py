"""Central orchestrator for PhysiClaw.

The PhysiClaw class owns hardware lifecycle (arm, camera, calibration)
and bbox workflow state. Image-output helpers (drawing, encoding,
watermarking) live in physiclaw.vision.render.
"""

from physiclaw.core.orchestrator import PhysiClaw

__all__ = ["PhysiClaw"]
