"""ISO 273 metric screw clearance hole diameters.

Three fit classes:
  CLOSE  — tight; better alignment, harder assembly.
  NORMAL — general purpose; default for most mounting.
  LOOSE  — extra clearance; tolerates misalignment.

For FDM-printed parts, real holes print 0.1-0.3 mm undersized. Bump up
by ~0.2 mm above NORMAL, or use LOOSE, when the target is FDM rather
than machined. Test-print a tolerance gauge if unsure.
"""

from build123d import MM

#                              CLOSE     NORMAL    LOOSE
M3_CLOSE, M3_NORMAL, M3_LOOSE = 3.2 * MM, 3.4 * MM, 3.6 * MM
M4_CLOSE, M4_NORMAL, M4_LOOSE = 4.3 * MM, 4.5 * MM, 4.8 * MM
M5_CLOSE, M5_NORMAL, M5_LOOSE = 5.3 * MM, 5.5 * MM, 5.8 * MM


# ── Countersunk flat-head screw heads (ISO 10642 / DIN 7991, 90° included) ────
# Max head Ø = top diameter of the conical recess so an FHCS head seats flush;
# CSK_ANGLE is the included angle (metric flat heads are 90°, not the 82° of
# inch screws). Cut a CounterSinkHole with counter_sink_radius = head/2 and
# counter_sink_angle = CSK_ANGLE.
M3_CSK_HEAD, M4_CSK_HEAD, M5_CSK_HEAD = 6.0 * MM, 8.0 * MM, 10.0 * MM
CSK_ANGLE = 90


# ── Square nut nominal dimensions (DIN 557) ───────────────────────────────────
# W = width across flats (side of the square)
# T = thickness (axial dimension)
# For 3D-printed nut sockets, add ~0.2 mm clearance on each dimension so the
# nut slides in.

#                    W         T
M3_NUT_W, M3_NUT_T = 5.7 * MM, 2.7 * MM
M4_NUT_W, M4_NUT_T = 7.3 * MM, 3.3 * MM
M5_NUT_W, M5_NUT_T = 8.3 * MM, 4.0 * MM


# ── 0.1" pin-header / driver-carrier interface ────────────────────────────────
# Standard 0.1-inch header pitch, and the row-to-row spacing of a 2×8
# stepper-driver carrier (StepStick / Pololu form factor = 5 × 2.54 mm). Shared
# by the controller board's driver sockets and the driver module so they mate.
HDR_PITCH = 2.54 * MM
DRIVER_ROW_PITCH = 12.7 * MM
