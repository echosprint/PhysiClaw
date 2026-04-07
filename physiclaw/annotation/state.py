"""AnnotationState — shared state for the annotation UI and agent loop."""

import threading
import uuid

from physiclaw.annotation.bbox import AGENT_COLOR


class AnnotationState:
    """Shared state for the web annotation UI and agent propose-confirm loop."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frozen_frame = None          # BGR numpy array
        self.snapshot_id: str = ""        # timestamp of frozen snapshot

        # Agent → UI staging area
        self.agent_proposals: list[dict] = []

        # Confirmation flow
        self._confirmed = threading.Event()
        self.confirmed_annotations: list[dict] = []  # final boxes (typed, 0-1 coords)

    def freeze(self, frame, snapshot_id: str = ""):
        with self.lock:
            self.frozen_frame = frame.copy()
            self.snapshot_id = snapshot_id
            self.agent_proposals = []
            self._confirmed.clear()
            self.confirmed_annotations = []

    def get_frozen_frame(self):
        with self.lock:
            return self.frozen_frame

    def clear(self):
        with self.lock:
            self.frozen_frame = None
            self.snapshot_id = ""
            self.agent_proposals = []
            self._confirmed.clear()
            self.confirmed_annotations = []

    # ─── Agent proposals ──────────────────────────────────────

    def push_agent_proposals(self, proposals: list[dict]):
        """Stage agent-proposed boxes for the UI to pick up.

        Each proposal: {"bbox": [l,t,r,b], "label": "..."}
        Coordinates are in 0-1 phone screen decimals.
        Replaces any previous proposals (agent sends fresh set each time).
        """
        enriched = []
        for p in proposals:
            enriched.append({
                "id": str(uuid.uuid4())[:8],
                "bbox": p["bbox"],
                "label": p.get("label", ""),
                "source": "agent",
                "color": AGENT_COLOR,
            })
        with self.lock:
            self.agent_proposals = enriched
            self._confirmed.clear()
            self.confirmed_annotations = []

    def pop_agent_proposals(self) -> list[dict]:
        """Return and clear pending agent proposals. Called by UI polling."""
        with self.lock:
            proposals = self.agent_proposals
            self.agent_proposals = []
            return proposals

    # ─── Confirmation flow ────────────────────────────────────

    def confirm(self, annotations: list[dict]):
        """Called when user clicks Confirm in the UI.

        annotations: final boxes with 0-1 phone coords, labels, sources.
        """
        with self.lock:
            self.confirmed_annotations = annotations
            self._confirmed.set()

    def wait_confirmed(self, timeout: float = 120.0) -> list[dict] | None:
        """Block until user confirms or timeout. Returns confirmed boxes or None."""
        if self._confirmed.wait(timeout=timeout):
            with self.lock:
                return list(self.confirmed_annotations)
        return None

    def clear_confirmation(self):
        """Reset confirmation state for next cycle."""
        with self.lock:
            self._confirmed.clear()
            self.confirmed_annotations = []
