"""PTFE (Teflon) tube — a hollow tube routed along a tangent-constrained spine.

A purchased part (flexible PTFE tubing), modeled as an annular cross-section
swept along a smooth ``Spline`` between two endpoints with a prescribed tangent
at each end. Used to route a tube/conduit through a bend whose ends must leave
and arrive along given directions.

Spec defaults to generic 4 mm OD / 2 mm ID PTFE bowden tube; the routing
(endpoints + end tangents) and the spec are per-instance parameters so the same
class serves any run. ``tangent_scalars`` tune the Bézier-handle length (curve
fullness) without changing the end directions — lower them if ``sweep()``
self-intersects on a tight bend.

Canonical frame: as given by ``point1``/``point2`` (caller's coordinates); the
tube is open at both ends. A ``start`` joint sits at point1 oriented along the
start tangent for assembly routing.

Run from the repo root:

    uv run --group cad python -m hardware.parts.standard.teflon
"""

from build123d import *

from hardware.parts.base import BaseStandardPart

# ── Defaults ──────────────────────────────────────────────────────────────────
default_od = 4 * MM  # outer diameter (generic PTFE bowden)
default_id = 2 * MM  # inner diameter; wall = (od - id) / 2 = 1 mm

# Default routing: a smooth bend leaving vertical (+Z) and arriving horizontal (+X).
default_point1 = (0, 0, 0)
default_direction1 = (0, 0, 1)
default_point2 = (30 * MM, 0, 20 * MM)
default_direction2 = (1, 0, 0)

# Apex offset (fraction of the chord) inserted when the end tangents oppose, so
# the U / inverted-U sweep stays clean. Larger = taller bulge (needs a larger
# tangent_scalar to sweep without kinking).
WAYPOINT_FRAC = 0.3


class Teflon(BaseStandardPart):
    """Hollow PTFE tube swept along a Spline with end-tangent constraints."""

    def __init__(
        self,
        point1=default_point1,
        direction1=default_direction1,
        point2=default_point2,
        direction2=default_direction2,
        od: float = default_od,
        id: float = default_id,
        tangent_scalars=(1.0, 1.0),
        qty: int = 1,
    ):
        super().__init__(qty=qty)
        self.point1 = point1
        self.direction1 = direction1
        self.point2 = point2
        self.direction2 = direction2
        self.od = od
        self.id = id
        self.tangent_scalars = tangent_scalars

    def _spline(self):
        """The routing spine as a standalone edge (drives length + sweep).

        When the end tangents oppose (dot < 0) a single spline doubles back and
        the sweep self-intersects, so thread it through an apex derived from the
        endpoints and tangents — a U / inverted-U bulge along (d1 - d2)."""
        p1, p2 = Vector(*self.point1), Vector(*self.point2)
        d1 = Vector(*self.direction1).normalized()
        d2 = Vector(*self.direction2).normalized()
        pts = [self.point1, self.point2]
        if d1.dot(d2) < 0:
            apex = (p1 + p2) / 2 + (d1 - d2).normalized() * (
                WAYPOINT_FRAC * (p2 - p1).length
            )
            pts = [self.point1, tuple(apex), self.point2]
        return Spline(
            *pts,
            tangents=(self.direction1, self.direction2),
            tangent_scalars=self.tangent_scalars,
        )

    @property
    def cut_length(self) -> float:
        """Tube length along the spine (the purchase dimension), cached."""
        if (cached := getattr(self, "_cut_length", None)) is None:
            cached = self._cut_length = self._spline().length
        return cached

    def name_suffix(self) -> str:
        return f"_OD{self.od:g}xID{self.id:g}_{round(self.cut_length)}mm_x{self.qty}"

    def bom_key(self):
        # Purchased by spec + cut length (1 mm granularity); routing shape is
        # irrelevant to the line item, only the resulting length.
        return ("Teflon", self.od, self.id, round(self.cut_length))

    def bom_display(self) -> str:
        return f"PTFE tube Ø{self.od:g}/{self.id:g} ({round(self.cut_length)} mm)"

    def geom_key(self):
        # Geometry depends on the full routing, not just the spec (the apex is
        # derived from these, so it needn't be keyed separately).
        return (
            "Teflon",
            self.point1,
            self.direction1,
            self.point2,
            self.direction2,
            self.od,
            self.id,
            self.tangent_scalars,
        )

    def _build(self):
        spine = self._spline()
        # Annular profile, square to the path start. Rotationally symmetric →
        # no frenet/twist needed. (`spine` is a plain edge, so `@`/`%` work.)
        start_plane = Plane(origin=spine @ 0, z_dir=spine % 0)
        with BuildPart() as tube:
            with BuildSketch(start_plane):
                Circle(radius=self.od / 2)
                Circle(radius=self.id / 2, mode=Mode.SUBTRACT)
            sweep(path=spine)

        part = tube.part
        part.label = "Teflon"

        # Routing reference: tube entry at point1, local +Z along the start
        # tangent (a Location from the start-plane, not raw Euler angles).
        RigidJoint("start", to_part=part, joint_location=Location(start_plane))
        return part


if __name__ == "__main__":
    Teflon().export()
