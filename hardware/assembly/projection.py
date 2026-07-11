import math
import re
from dataclasses import dataclass

from build123d import Axis, Shape, Vector


@dataclass(frozen=True)
class Camera:
    """Camera direction (degrees) for project_to_viewport.

    Conventions:
      * Z is up in world coords.
      * azimuth: 0 = front (-Y), 90 = right (+X), -90 = left, 180 = back.
      * elevation: 0 = horizontal, 90 = top-down.
      * roll: roll around the view axis. Positive = object rotates
        clockwise on paper. 0 = world +Z projected as up.
    """

    azimuth: float
    elevation: float
    roll: float = 0.0

    def __mul__(self, distance: float) -> Vector:
        """Camera origin scaled to `distance` from world origin."""
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        horizontal = math.cos(el)
        return Vector(
            horizontal * math.sin(az) * distance,
            -horizontal * math.cos(az) * distance,
            math.sin(el) * distance,
        )

    __rmul__ = __mul__

    @property
    def up(self) -> Vector:
        """Up vector for project_to_viewport, rolled by ``self.roll``."""
        z = Vector(0, 0, 1)
        view_dir = (self * -1.0).normalized()
        base = (z - view_dir * view_dir.dot(z)).normalized()
        # Negative angle so positive roll rotates the object clockwise on
        # paper. Skip the rotation at 0 for float stability (Rodrigues with
        # θ=0 is mathematically identity but introduces cross-product noise).
        if self.roll == 0:
            return base
        theta = -math.radians(self.roll)
        # Rodrigues simplifies — base ⊥ view_dir means the parallel term is 0.
        return base * math.cos(theta) + view_dir.cross(base) * math.sin(theta)

    @classmethod
    def from_freecad_view(cls, s: str) -> "Camera":
        """Parse a FreeCAD/Coin3D Inventor camera string into a ``Camera``.

        Reads only the ``orientation`` field (axis-angle in radians).
        The returned Camera describes the camera's gaze direction in
        the convention defined on this dataclass: Z up, azimuth 0 =
        looking toward +Y (camera at -Y), azimuth +90 = looking toward
        -X (camera at +X), elevation +90 = top-down, roll positive =
        clockwise on paper.
        """
        m = re.search(
            r"orientation\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)",
            s,
        )
        if not m:
            raise ValueError("no 'orientation' field in camera string")
        ax, ay, az_, angle_rad = (float(v) for v in m.groups())

        # Rotate the camera's canonical right / up / forward vectors
        # (camera-space +X / +Y / -Z) through the axis-angle orientation.
        # Axis normalises the direction internally; a zero-length axis
        # raises ValueError, surfaced to the caller.
        axis = Axis((0, 0, 0), (ax, ay, az_))
        deg = math.degrees(angle_rad)
        right = Vector(1, 0, 0).rotate(axis, deg)
        up_cam = Vector(0, 1, 0).rotate(axis, deg)
        fwd = Vector(0, 0, -1).rotate(axis, deg)

        elevation = math.degrees(math.asin(max(-1.0, min(1.0, -fwd.Z))))
        if abs(fwd.Z) > 0.99999:
            # Gimbal lock: looking straight up or down. Azimuth degenerates;
            # fold it into roll using the camera-up direction in world XY.
            azimuth = 0.0
            roll = math.degrees(math.atan2(up_cam.X, up_cam.Y))
        else:
            azimuth = math.degrees(math.atan2(-fwd.X, fwd.Y))
            roll = math.degrees(math.atan2(right.Z, up_cam.Z))

        azimuth = (azimuth + 180.0) % 360.0 - 180.0
        return cls(azimuth=azimuth, elevation=elevation, roll=roll)


FRONT = Camera(0, 0)
ISO = Camera(45, 35.2644)

# Procedure-view presets. Naming encodes azimuth / elevation / non-zero
# roll so the angle is readable at the import site. `MAIN_FRAME_VIEW` is
# the one exception — kept semantic because the angle is calibrated to
# the X-rail bbox.
FRONT_LEFT_HIGH = Camera(-30, 25)
FRONT_LEFT_LOW = Camera(-30, -20)
BACK_RIGHT_LOW_R90 = Camera(120, -20, 90)
FRONT_LEFT_LOW_R70 = Camera(-45, -20, 70)
FRONT_HIGH_L10 = Camera(15, 45, -10)
MAIN_FRAME_VIEW = Camera(-15, -15, -3)


def camera_view(
    shape: Shape,
    camera: Camera,
    *,
    distance_factor: float = 4,
) -> tuple[Vector, Vector, Vector]:
    """Camera position + up + look_at for projecting `shape` from `camera`.

    Camera position = ``shape.bounding_box().center() + camera * (reach * distance_factor)``
    where ``reach`` is the part's farthest extent from world origin.
    ``distance_factor=4`` leaves noticeable perspective foreshortening;
    raise to push toward orthographic.

    Returned ``look_at`` is the bbox center; pass it (along with pos / up)
    to ``project_to_viewport`` when projecting subsets that must align —
    otherwise project_to_viewport defaults look_at to each subset's own
    center, which warps the projection direction per subset.
    """
    bbox = shape.bounding_box()
    reach = max(
        abs(v)
        for v in (
            bbox.min.X,
            bbox.max.X,
            bbox.min.Y,
            bbox.max.Y,
            bbox.min.Z,
            bbox.max.Z,
        )
    )
    look_at = bbox.center()
    return look_at + camera * (reach * distance_factor), camera.up, look_at


if __name__ == "__main__":
    # Convert a FreeCAD camera view string (from View → Camera settings)
    # into a ready-to-paste ``Camera(az, el, roll)`` literal.
    #
    # To grab the view string from FreeCAD, frame the desired angle in the
    # 3D viewport, then run in the View → Panels → Python console:
    #
    #     import FreeCADGui as Gui
    #     view = Gui.ActiveDocument.ActiveView
    #     view.getCamera()
    #
    # The printed Inventor block (``#Inventor V2.1 ascii ... PerspectiveCamera
    # { ... }``) is what gets fed to this module.
    #
    # The string must be passed as a single shell argument or piped on
    # stdin — a bare quoted string at the start of a pipeline is run as
    # a command by the shell, not redirected as input.
    #
    #     uv run --group cad python -m hardware.assembly.projection '<paste>'
    #     uv run --group cad python -m hardware.assembly.projection "$(pbpaste)"
    #     pbpaste | uv run --group cad python -m hardware.assembly.projection
    #     echo  '<paste>' | uv run --group cad python -m hardware.assembly.projection
    #
    import sys

    s = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    try:
        cam = Camera.from_freecad_view(s)
    except Exception as err:
        print(f"error: {err}", file=sys.stderr)
        print(
            "\nTo grab the camera string in FreeCAD, frame the angle in the\n"
            "3D viewport, then run in View → Panels → Python console:\n"
            "\n"
            "    import FreeCADGui as Gui\n"
            "    view = Gui.ActiveDocument.ActiveView\n"
            "    view.getCamera()\n"
            "\n"
            "Pass the printed Inventor block to this module:\n"
            "\n"
            '    uv run --group cad python -m hardware.assembly.projection "$(pbpaste)"\n',
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Camera({cam.azimuth:.2f}, {cam.elevation:.2f}, {cam.roll:.2f})")
