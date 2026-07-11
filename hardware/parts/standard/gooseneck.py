"""Gooseneck arm — a flexible neck routed along a tangent-constrained spine,
with a 1/4"-20 threaded fitting at each end.

Like ``teflon`` it sweeps a circular section along a smooth ``Spline`` between
two endpoints with a prescribed tangent at each end — but the neck is SOLID
(no bore), and instead of open ends it carries:

  * a male 1/4"-20 stud on the start end (the "screw"), and
  * a female 1/4"-20 socket on the far end (the "screw-in").

Both ends are the standard 1/4"-20 photo/CNC mount thread; threads are shown
as annular grooves (representation, not cuttable threads). The routing
(endpoints + end tangents) and the neck diameter are per-instance parameters;
``tangent_scalars`` tune curve fullness (lower if the sweep self-intersects).

Canonical frame: caller's coordinates. ``stud`` joint sits at the male tip
(+Z outward), ``socket`` joint at the female mouth (+Z outward).

Run from the repo root:

    uv run --group cad python -m hardware.parts.standard.gooseneck
"""

from build123d import *

from hardware.parts.base import BaseStandardPart

# ── Neck ──────────────────────────────────────────────────────────────────────
default_od = 10 * MM  # solid neck outer diameter

# Default routing: a smooth bend leaving vertical (+Z), arriving horizontal (+X).
default_point1 = (0, 0, 0)
default_direction1 = (0, 0, 1)
default_point2 = (30 * MM, 0, 20 * MM)
default_direction2 = (1, 0, 0)

# ── 1/4"-20 UNC thread (photo / CNC mount standard) ───────────────────────────
thread_major = 6.35 * MM  # 1/4"
stud_len = 10 * MM  # male protrusion
socket_bore = 5.2 * MM  # female tapped-hole minor diameter
socket_depth = 11 * MM  # female thread depth
thread_pitch = 1.27 * MM  # 20 TPI
groove_depth = 0.35 * MM  # radial groove (thread valley) depth
groove_w = 0.4 * MM  # groove width along the axis

# End ferrules (metal collars that carry the threads).
male_collar_len = 6 * MM
female_collar_len = socket_depth + 2 * MM  # deep enough to hold the bore
end_chamfer = 0.6 * MM

COL_NECK = Color(0.20, 0.20, 0.22)  # coated flexible neck
COL_METAL = Color(0.75, 0.75, 0.78)  # threaded end fittings


def _stud_grooves(p, z0: float, length: float, major_r: float):
    """Carve evenly spaced annular grooves over [z0, z0+length] on an external
    stud of radius ``major_r`` — slice off a disc, add the core back (the core
    is concentric with the stud, so it stays attached). External threads only;
    on an internal bore the re-added core would float free."""
    for i in range(int(length / thread_pitch)):
        z = z0 + (i + 0.5) * thread_pitch
        with Locations((0, 0, z)):
            Cylinder(major_r + 0.5 * MM, groove_w, mode=Mode.SUBTRACT)
            Cylinder(major_r - groove_depth, groove_w)


def _male_end(od: float) -> Part:
    """Collar + male 1/4"-20 stud, canonical +Z (z=0 on the neck end face).
    The collar matches the neck diameter ``od``."""
    with BuildPart() as p:
        Cylinder(od / 2, male_collar_len, align=(Align.CENTER, Align.CENTER, Align.MIN))
        with Locations((0, 0, male_collar_len)):
            Cylinder(
                thread_major / 2,
                stud_len,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
            )
        _stud_grooves(p, male_collar_len, stud_len, thread_major / 2)
        # Chamfer the stud tip (highest circular edge).
        tip = p.edges().filter_by(GeomType.CIRCLE).group_by(Axis.Z)[-1]
        chamfer(tip, end_chamfer)
    return p.part


def _female_end(od: float) -> Part:
    """Collar with a female 1/4"-20 socket bored in from the outer face.
    The collar matches the neck diameter ``od``."""
    with BuildPart() as p:
        Cylinder(
            od / 2, female_collar_len, align=(Align.CENTER, Align.CENTER, Align.MIN)
        )
        with Locations((0, 0, female_collar_len)):
            Cylinder(
                socket_bore / 2,
                socket_depth,
                align=(Align.CENTER, Align.CENTER, Align.MAX),
                mode=Mode.SUBTRACT,
            )
        # Chamfer the outer-face edges (collar rim + socket mouth).
        mouth = p.edges().filter_by(GeomType.CIRCLE).group_by(Axis.Z)[-1]
        chamfer(mouth, end_chamfer)
    return p.part


class Gooseneck(BaseStandardPart):
    """Solid flexible neck swept along a Spline, 1/4"-20 male + female ends."""

    def __init__(
        self,
        point1=default_point1,
        direction1=default_direction1,
        point2=default_point2,
        direction2=default_direction2,
        od: float = default_od,
        tangent_scalars=(1.0, 1.0),
        qty: int = 1,
    ):
        super().__init__(qty=qty)
        self.point1 = point1
        self.direction1 = direction1
        self.point2 = point2
        self.direction2 = direction2
        self.od = od
        self.tangent_scalars = tangent_scalars

    def _spline(self):
        """The routing spine as a standalone edge (drives length + sweep)."""
        return Spline(
            self.point1,
            self.point2,
            tangents=(self.direction1, self.direction2),
            tangent_scalars=self.tangent_scalars,
        )

    @property
    def neck_length(self) -> float:
        """Flexible-neck length along the spine (purchase dimension), cached."""
        if (cached := getattr(self, "_neck_length", None)) is None:
            cached = self._neck_length = self._spline().length
        return cached

    def name_suffix(self) -> str:
        return f"_OD{self.od:g}_{round(self.neck_length)}mm_x{self.qty}"

    def bom_key(self):
        # Purchased by neck diameter + length; the thread spec is fixed (1/4-20).
        return ("Gooseneck", self.od, round(self.neck_length))

    def bom_display(self) -> str:
        return f"Gooseneck Ø{self.od:g} × {round(self.neck_length)} mm, 1/4-20 M/F"

    def geom_key(self):
        return (
            "Gooseneck",
            self.point1,
            self.direction1,
            self.point2,
            self.direction2,
            self.od,
            self.tangent_scalars,
        )

    def _build(self):
        spine = self._spline()

        # Solid neck: sweep a full circle (no bore) along the spine.
        start_plane = Plane(origin=spine @ 0, z_dir=spine % 0)
        with BuildPart() as neck:
            with BuildSketch(start_plane):
                Circle(radius=self.od / 2)
            sweep(path=spine)
        neck_part = neck.part
        neck_part.color = COL_NECK
        neck_part.label = "neck"

        # End fittings, each placed with local +Z pointing outward along the
        # tangent: the male stud leaves point1 (so -direction1), the female
        # socket faces out of point2 (so +direction2).
        male_plane = Plane(origin=self.point1, z_dir=tuple(-c for c in self.direction1))
        female_plane = Plane(origin=self.point2, z_dir=self.direction2)

        male = _male_end(self.od).moved(Location(male_plane))
        male.color = COL_METAL
        male.label = "male_stud"

        female = _female_end(self.od).moved(Location(female_plane))
        female.color = COL_METAL
        female.label = "female_socket"

        body = Compound(label="Gooseneck", children=[neck_part, male, female])

        # Mounting references: male stud tip and female socket mouth.
        RigidJoint(
            "stud",
            to_part=body,
            joint_location=Location(male_plane.offset(male_collar_len + stud_len)),
        )
        RigidJoint(
            "socket",
            to_part=body,
            joint_location=Location(female_plane.offset(female_collar_len)),
        )
        return body


if __name__ == "__main__":
    Gooseneck().export()
