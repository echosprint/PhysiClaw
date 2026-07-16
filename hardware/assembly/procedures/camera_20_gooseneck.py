"""Camera gooseneck — camera_10_bracket with a flexible gooseneck mounted on
its 1/4-20 screw.

Takes the corner-bracket sub-assembly from camera_10 (bracket + frame-side
BHCS/T-nuts + the 1/4-20 SHCS and hex nut) and threads a Gooseneck onto that
screw by its FEMALE 1/4-20 socket: the socket mouth opens toward the stud and
captures the thread exposed past the hex nut, then the neck arcs up to a free
male 1/4-20 stud (where a camera mounts in a later step).

Variants (the camera_10 bracket is always embedded assembled; only the
gooseneck moves):
  * exploded — the gooseneck slides off the stud along −X (GOOSE_GAP) so the
    female socket reads as separate.
  * assembled — the female socket seated on the stud against the hex nut.

Parts (adds to camera_10_bracket):
  * 1 x Gooseneck (Ø10 neck, 1/4-20 male + female ends)

Run from the repo root:

    uv run --group cad python -m hardware step camera_20_gooseneck
"""

from build123d import Axis, Compound

from hardware.assembly.base import ASSEMBLED_CAM0, EXPLODED_CAM0, BaseAssembly
from hardware.assembly.procedures.camera_10_bracket import Camera10Bracket
from hardware.assembly.projection import Camera
from hardware.parts.standard.gooseneck import (
    Gooseneck,
    female_collar_len,
    male_collar_len,
)
from hardware.parts.standard.nut import SPECS as NUT_SPECS

NUT_THICK = NUT_SPECS["hex"]["1/4-20"]["thickness"]  # mm — hex nut height
# Gooseneck arm span. The names follow camera_20's own (bracket-local) frame —
# RISE along bracket +Z, REACH along bracket +X — but camera_40's mount rotates
# the bracket 90°, so INSTALLED the bracket-Z rise reads as the HORIZONTAL reach
# out from the extrusion face and the bracket-X reach as the VERTICAL drop. The
# values are set for the installed result: 150 mm vertical × 85 mm horizontal.
GOOSE_REACH = 100  # along bracket +Z → installed HORIZONTAL (out from the face)
GOOSE_RISE = 150  # along bracket +X → installed VERTICAL (drop toward the bed)
GOOSE_GAP = 16  # mm — exploded: gooseneck slid off the stud along −X


class Camera20Gooseneck(BaseAssembly):
    compound_label = "camera_20_gooseneck"
    camera = [Camera(-14.11, -7.34, 88.02), Camera(-10.74, 22.98, 88.54)]
    views = [EXPLODED_CAM0, ASSEMBLED_CAM0]

    def _build(self) -> Compound:
        # Embed the camera_10 bracket sub-assembly in its assembled form (only
        # this step's new part — the gooseneck — explodes); it exposes the
        # camera-screw anchors (cam_y / cam_z / cam_offset).
        bracket = Camera10Bracket(exploded=False)
        base = bracket.build()  # bracket + frame fasteners + 1/4-20 screw + hex nut
        cam_y, cam_z, cam_offset = bracket.cam_y, bracket.cam_z, bracket.cam_offset

        # The 1/4-20 screw protrudes on the −X face (head on +X). Thread the
        # female socket onto the exposed stud just past the hex nut: the mouth
        # opens +X (toward the stud) so the screw enters it; the neck then arcs
        # up to a free male stud.
        nut_back_x = -cam_offset - NUT_THICK
        mouth_x = nut_back_x - (GOOSE_GAP if self.exploded else 0)

        female_pt = (mouth_x - female_collar_len, cam_y, cam_z)
        male_pt = (female_pt[0] - GOOSE_RISE, cam_y, cam_z + GOOSE_REACH)
        goose = Gooseneck(
            point1=male_pt,
            direction1=(0, 0, -1),  # male stud points +Z (up)
            point2=female_pt,
            direction2=(1, 0, 0),  # female mouth opens +X onto the stud
        ).build()

        # Rotate the whole gooseneck 180° about the screw axis (world X
        # through the socket): the female end stays seated on the stud while
        # the neck swings to the opposite side (arm now descends instead of
        # rising).
        goose = goose.rotate(Axis((female_pt[0], cam_y, cam_z), (1, 0, 0)), 180)

        # Expose the male-stud mount seat for downstream steps (camera_30
        # clips a Camera here). Pre-rotation the stud base (male-collar face)
        # sits male_collar_len above male_pt with the stud pointing +Z; the
        # 180° flip about X sends it to z = cam_z − GOOSE_RISE − male_collar_len
        # with the stud pointing −Z.
        self.stud_base = (male_pt[0], cam_y, cam_z - GOOSE_REACH - male_collar_len)
        self.stud_axis = (0.0, 0.0, -1.0)

        return Compound(label=self.compound_label, children=[base, goose])
