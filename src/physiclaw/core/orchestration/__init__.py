"""Orchestration layer for PhysiClaw.

Three parts, one lock: :class:`HardwareRig` owns devices, calibration,
lifecycle, and the busy lock; :class:`Perception` owns detectors and
frame acquisition; :class:`PhysiClaw` is the facade composing both and
exposing the agent-facing tool operations. Image-output helpers
(drawing, encoding, watermarking) live in physiclaw.core.vision.render.
"""

from physiclaw.core.orchestration.observation import GestureResult
from physiclaw.core.orchestration.orchestrator import PhysiClaw
from physiclaw.core.orchestration.perception import Perception
from physiclaw.core.orchestration.rig import HardwareRig

__all__ = ["GestureResult", "HardwareRig", "Perception", "PhysiClaw"]
