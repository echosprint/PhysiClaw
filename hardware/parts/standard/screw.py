import math

from build123d import *

from hardware.parts.base import BaseStandardPart

# ── Dimension tables (mm) ─────────────────────────────────────────────────────
# d = nominal Ø, P = coarse pitch.
COMMON = {
    "M2": {"d": 2.0, "P": 0.40},
    "M2.5": {"d": 2.5, "P": 0.45},
    "M3": {"d": 3.0, "P": 0.50},
    "M4": {"d": 4.0, "P": 0.70},
    "M5": {"d": 5.0, "P": 0.80},
    "M6": {"d": 6.0, "P": 1.00},
    "M8": {"d": 8.0, "P": 1.25},
    # 1/4"-20 UNC: d = 1/4", P = 25.4/20 mm. Mates the Gooseneck female socket.
    "1/4-20": {"d": 6.35, "P": 1.27},
}

# dk = max head Ø, k = head height, s = hex socket across-flats (Allen key).
# SHCS — ISO 4762 / DIN 912 (head height ≈ d).
SHCS_DIMS = {
    "M2": {"dk": 3.8, "k": 2.0, "s": 1.5},
    "M2.5": {"dk": 4.5, "k": 2.5, "s": 2.0},
    "M3": {"dk": 5.5, "k": 3.0, "s": 2.5},
    "M4": {"dk": 7.0, "k": 4.0, "s": 3.0},
    "M5": {"dk": 8.5, "k": 5.0, "s": 4.0},
    "M6": {"dk": 10.0, "k": 6.0, "s": 5.0},
    "M8": {"dk": 13.0, "k": 8.0, "s": 6.0},
    # 1/4"-20 UNC SHCS (ASME B18.3): dk = 3/8", k = 1/4", s = 3/16" hex.
    "1/4-20": {"dk": 9.53, "k": 6.35, "s": 4.76},
}

# FHCS — ISO 10642 / DIN 7991 (90° countersunk).
FHCS_DIMS = {
    "M3": {"dk": 6.0, "k": 1.86, "s": 2.0},
    "M4": {"dk": 8.0, "k": 2.48, "s": 2.5},
    "M5": {"dk": 10.0, "k": 3.10, "s": 3.0},
    "M6": {"dk": 12.0, "k": 3.72, "s": 4.0},
    "M8": {"dk": 16.0, "k": 4.96, "s": 5.0},
}

# BHCS — ISO 7380-1 / DIN 7380. Button heads have head height ≈ 0.5 d and the
# shallowest socket of the three (lowest torque ceiling). Not a drop-in for
# SHCS in clamp-critical joints (CoreXY corners, NEMA17 mounts).
BHCS_DIMS = {
    "M3": {"dk": 5.7, "k": 1.65, "s": 2.0},
    "M4": {"dk": 7.6, "k": 2.20, "s": 2.5},
    "M5": {"dk": 9.5, "k": 2.75, "s": 3.0},
    "M6": {"dk": 10.5, "k": 3.30, "s": 4.0},
    "M8": {"dk": 14.0, "k": 4.40, "s": 5.0},
}

# Shoulder screw (a.k.a. stripper bolt) — socket head, smooth load-bearing
# shoulder, then a shorter threaded section. Keyed by THREAD size; the
# shoulder diameter is larger than the thread. `length` on the Screw class
# is the shoulder length (the smooth rod).
SHOULDER_DIMS = {
    "M4": {
        "dk": 9.0,  # head diameter
        "k": 3.5,  # head height
        "s": 3.0,  # hex socket across-flats
        "shoulder_d": 5.0,  # smooth shoulder diameter
        "thread_len": 7.2,  # threaded section axial length
    },
}

TABLES = {
    "SHCS": SHCS_DIMS,
    "FHCS": FHCS_DIMS,
    "BHCS": BHCS_DIMS,
    "SHOULDER": SHOULDER_DIMS,
}

# Hex socket depth as a fraction of head height k.
SOCKET_DEPTH_FRAC = {"SHCS": 0.6, "FHCS": 0.6, "BHCS": 0.5, "SHOULDER": 0.6}

# Small vertical step at the rim (radius = dk/2) of the BHCS dome and the
# FHCS cone — makes each head a little thicker and avoids a knife edge
# where the curved / tapered surface would otherwise meet its neighbor
# tangentially.
head_skirt = 0.2 * MM

# Chamfer on the outer rim of a flat-topped head (SHCS / SHOULDER).
head_top_chamfer = 0.2 * MM

# BHCS hex socket sits in a small flat-bottomed counterbore so it isn't
# slicing through the dome's curved surface. The pocket is bigger than the
# hex by `pocket_rim_margin` (giving a visible flat ring around the hex);
# its depth is the minimum that keeps the floor inside the dome material
# plus `pocket_depth_margin` of slack.
pocket_rim_margin = 0.2 * MM
pocket_depth_margin = 0.05 * MM


# ── Half-profile builders (called inside BuildLine on Plane.XZ; r ≥ 0) ────────
def _shcs_profile(d, length, dim):
    # Cylindrical shank + cylindrical cap head.
    dk, k = dim["dk"], dim["k"]
    Polyline(
        (0, -length),
        (d / 2, -length),
        (d / 2, 0),
        (dk / 2, 0),
        (dk / 2, k),
        (0, k),
        close=True,
    )


def _fhcs_profile(d, length, dim):
    # FHCS length is OVERALL; head occupies z ∈ [0, k + head_skirt] and the
    # shank fills the rest. Profile bottom-up: shank → cone (z = 0 → k) →
    # rim cylinder of height head_skirt at radius dk/2 → flat top.
    dk, k = dim["dk"], dim["k"]
    top_z = k + head_skirt
    shank = length - top_z
    Polyline(
        (0, -shank),
        (d / 2, -shank),
        (d / 2, 0),
        (dk / 2, k),  # cone
        (dk / 2, top_z),  # rim cylinder
        (0, top_z),  # flat top
        close=True,
    )


def _bhcs_profile(d, length, dim):
    # Dome = spherical cap of height k spanning radial 0 → dk/2. With the
    # rim skirt added, the dome itself sits in z ∈ [head_skirt, apex_z];
    # geometry is unchanged, just shifted up by head_skirt. Sphere center
    # is on the Z axis at z = apex_z − R, where (dk/2)² + (R − k)² = R²
    # ⇒ R = ((dk/2)² + k²) / (2k).
    # RadiusArc sign: positive R bulges to the LEFT of the start→end vector
    # (which here points up-and-inward) — i.e. the arc would dip toward
    # the axis, giving a concave dish. Use −R so the arc bulges outward
    # (up-and-away from the axis) into a convex dome.
    dk, k = dim["dk"], dim["k"]
    rim_z = head_skirt
    apex_z = k + head_skirt
    R = ((dk / 2) ** 2 + k**2) / (2 * k)
    Polyline(
        (0, -length),
        (d / 2, -length),
        (d / 2, 0),
        (dk / 2, 0),
        (dk / 2, rim_z),  # rim cylinder before the dome
    )
    try:
        RadiusArc((dk / 2, rim_z), (0, apex_z), -R)
    except Exception:
        # Fallback: 3-point spline through the mid-arc point on the dome
        # (computed from the sphere center on the Z axis).
        cz = apex_z - R
        am = (math.atan2(rim_z - cz, dk / 2) + math.pi / 2) / 2
        mx, mz = R * math.cos(am), cz + R * math.sin(am)
        Spline((dk / 2, rim_z), (mx, mz), (0, apex_z))
    Line((0, apex_z), (0, -length))


def _shoulder_profile(d, length, dim):
    # Thread (Ø d, len thread_len) → smooth shoulder (Ø shoulder_d, len `length`)
    # → cylindrical head (Ø dk × k). z=0 sits at the underhead seating plane
    # (top of shoulder); thread occupies z ∈ [−length−thread_len, −length].
    dk, k = dim["dk"], dim["k"]
    sd, tl = dim["shoulder_d"], dim["thread_len"]
    Polyline(
        (0, -length - tl),
        (d / 2, -length - tl),
        (d / 2, -length),
        (sd / 2, -length),
        (sd / 2, 0),
        (dk / 2, 0),
        (dk / 2, k),
        (0, k),
        close=True,
    )


_PROFILE_FNS = {
    "SHCS": _shcs_profile,
    "FHCS": _fhcs_profile,
    "BHCS": _bhcs_profile,
    "SHOULDER": _shoulder_profile,
}


# ── Geometry ──────────────────────────────────────────────────────────────────
class Screw(BaseStandardPart):
    """Metric socket screw (SHCS / FHCS / BHCS / SHOULDER).

    Z = 0 sits at the underhead seating plane; head at +Z, shank at −Z so the
    part drops into an assembly hole naturally. `length` semantics depend on
    the type:
      • SHCS / BHCS — underhead length (head excluded).
      • FHCS       — overall length (head included).
      • SHOULDER   — shoulder length (smooth rod between head and thread)."""

    def __init__(
        self,
        screw_type: str,
        size: str,
        length: float,
        qty: int = 1,
    ):
        super().__init__(qty=qty)
        if screw_type not in TABLES:
            raise ValueError(
                f"Unknown screw_type {screw_type!r}; expected one of {sorted(TABLES)}"
            )
        if size not in TABLES[screw_type]:
            raise ValueError(
                f"{screw_type} has no entry for size {size!r}; "
                f"available: {sorted(TABLES[screw_type])}"
            )
        k = TABLES[screw_type][size]["k"]
        if screw_type == "FHCS" and length <= k + head_skirt:
            raise ValueError(
                f"FHCS length ({length} mm) must exceed head height "
                f"({k + head_skirt} mm = k + head_skirt)"
            )
        self.screw_type = screw_type
        self.size = size
        self.length = length

    def name_suffix(self) -> str:
        # Sanitize the size for the filename ("1/4-20" → "1-4-20").
        safe_size = self.size.replace("/", "-")
        return f"_{self.screw_type}_{safe_size}x{self.length:g}_x{self.qty}"

    def bom_key(self):
        if self.screw_type == "SHOULDER":
            shoulder_d = TABLES[self.screw_type][self.size]["shoulder_d"]
            return ("Screw", self.screw_type, shoulder_d, self.size, self.length)
        return ("Screw", self.screw_type, self.size, self.length)

    def bom_display(self):
        if self.screw_type == "SHOULDER":
            shoulder_d = TABLES[self.screw_type][self.size]["shoulder_d"]
            return f"Shoulder Screw {shoulder_d:g}x{self.size}x{self.length:g}"
        return None

    def _build(self):
        d = COMMON[self.size]["d"]
        dim = TABLES[self.screw_type][self.size]
        k, s, dk = dim["k"], dim["s"], dim["dk"]

        with BuildPart() as p:
            # Half-profile in Plane.XZ, revolved around the Z axis.
            with BuildSketch(Plane.XZ):
                with BuildLine():
                    _PROFILE_FNS[self.screw_type](d, self.length, dim)
                make_face()
            revolve(axis=Axis.Z)

            # Z of the head top face. BHCS / FHCS heads are lifted by
            # head_skirt; SHCS / SHOULDER stay at z = k.
            head_top = k + head_skirt if self.screw_type in ("BHCS", "FHCS") else k
            sock_depth = SOCKET_DEPTH_FRAC[self.screw_type] * k
            hex_circumradius = s / (2 * math.cos(math.radians(30)))

            # BHCS counterbore: small flat-bottomed pocket above the hex so
            # the socket sits in a clean step rather than slicing through
            # the dome's curved surface. The pocket is *additive* — the hex
            # below still gets the full sock_depth.
            hex_top_z = head_top
            if self.screw_type == "BHCS":
                R = ((dk / 2) ** 2 + k**2) / (2 * k)
                pocket_r = hex_circumradius + pocket_rim_margin
                pocket_depth = R - math.sqrt(R**2 - pocket_r**2) + pocket_depth_margin
                with BuildSketch(Plane.XY.offset(head_top)):
                    Circle(pocket_r)
                extrude(amount=-pocket_depth, mode=Mode.SUBTRACT)
                hex_top_z = head_top - pocket_depth

            # Hex drive socket — sketched on Plane.XY (above either the head
            # top or the BHCS pocket floor) and extruded downward.
            with BuildSketch(Plane.XY.offset(hex_top_z)):
                RegularPolygon(radius=hex_circumradius, side_count=6)
            extrude(amount=-sock_depth, mode=Mode.SUBTRACT)

            # Break the sharp outer rim on flat-topped heads.
            if self.screw_type in ("SHCS", "SHOULDER"):
                top_rim = [
                    e
                    for e in p.edges().filter_by(GeomType.CIRCLE)
                    if abs(e.radius - dk / 2) < 0.01 and abs(e.center().Z - k) < 0.01
                ]
                chamfer(top_rim, length=head_top_chamfer)

            # Mating joint at the underhead seating plane. The +90°
            # X rotation flips the joint frame so that when paired with
            # an LinearJoint whose default slider frame is identity
            # (e.g. an extrusion counterbore axis), the screw's local
            # +Z (head direction) ends up along the host's outward
            # normal — head outside, shank into the part.
            RigidJoint(
                "head",
                joint_location=Location((0, 0, 0), (90, 0, 0)),
            )

        p.part.label = f"Screw_{self.screw_type}_{self.size}x{self.length:g}"
        return p.part


if __name__ == "__main__":
    Screw("SHCS", "M3", 12 * MM).export()
    Screw("FHCS", "M3", 8 * MM).export()
    Screw("BHCS", "M3", 8 * MM).export()
    Screw("SHOULDER", "M4", 20 * MM).export()
    Screw("SHCS", "1/4-20", 16 * MM).export()
