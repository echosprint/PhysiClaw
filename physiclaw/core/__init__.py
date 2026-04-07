"""Central orchestrator for PhysiClaw.

The PhysiClaw class owns hardware lifecycle (arm, camera, calibration)
and bbox workflow state. Pure rendering helpers live in core.rendering.
"""

from physiclaw.core.orchestrator import PhysiClaw

__all__ = ["PhysiClaw"]
