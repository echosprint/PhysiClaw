"""Camera mount (corner bracket) — an L-shaped CornerBracket fastened to a
frame extrusion through its horizontal face, carrying a 1/4-20 camera screw
on its vertical face.

  * Horizontal face (frame side, one centered hole): 1 x BHCS M5 x 10
    dropped from the top, catching a hammer M5 T-nut in the extrusion slot
    below (the extrusion isn't modelled here).
  * Vertical face (camera side, one centered hole): a SHCS 1/4-20 (the
    photographic-tripod thread) with a 1/4-20 hex nut on the far face. The
    screw axis is horizontal (along X); the nut clamps the 4 mm plate from
    the back.

Variants (same convention as idler_11_lu / frame_30_bracket_tnut):
  * exploded — horizontal stack opens along +Z (T-nuts on the floor,
    bracket lifted by BRACKET_GAP, BHCS shank tips floating SCREW_GAP above
    the deck); the camera screw and nut slide apart along their own ±X axis.
  * assembled — the bracket deck sits on the T-nut bosses, BHCS dropped
    through, and the camera screw + nut closed onto the vertical face.

Parts:
  * 1 x CornerBracket (26 mm bend, two 30 mm faces, 4 mm thick)
  * 1 x BHCS M5 x 10 + 1 x TNut "hammer" M5  (horizontal / frame side)
  * 1 x SHCS 1/4-20 x 16 + 1 x Nut "hex" 1/4-20  (vertical / camera side)

Run from the repo root:

    uv run --group cad python -m hardware.assembly.procedures.camera_10_bracket
"""

from build123d import Compound, Location, Plane

from hardware.assembly.base import BaseAssembly
from hardware.assembly.projection import Camera
from hardware.parts.standard.bracket import (
    CornerBracket,
    corner_face_depth as face_depth,
    corner_thickness as plate_thick,
)
from hardware.parts.standard.nut import Nut
from hardware.parts.standard.screw import Screw
from hardware.parts.standard.t_nut import (
    HAMMER_TOTAL_HEIGHT,
    LENGTHS as TNUT_LENGTHS,
    TNut,
)

BHCS_LENGTH = 10     # mm — BHCS M5 underhead length (frame side)
CAM_LENGTH  = 16     # mm — SHCS 1/4-20 underhead length (camera side)
BRACKET_GAP = 20     # mm — exploded: gap between T-nut tops and the deck bottom
SCREW_GAP   = 8      # mm — exploded: gap between deck top and BHCS shank tips
CAM_GAP     = 14     # mm — exploded: camera nut slid off along −X (gooseneck side)
CAM_SCREW_GAP = 32   # mm — exploded: SHCS pulled further off the +X install face
HAMMER_LOOSE_ENGAGE = 3  # mm — assembled: BHCS tip threads this far into the
                         #      hanging T-nut (no extrusion here yet to clamp it)


class Camera10Bracket(BaseAssembly):
    compound_label: str = "camera_10_bracket"
    camera = [Camera(-14.11, -7.34, 88.02), Camera(-10.74, 22.98, 88.54)]

    def _build(self) -> Compound:
        return Compound(label=self.compound_label, children=self._parts())

    def _parts(self) -> list:
        """Build and place every part, returning them as a flat list.
        Also stashes the camera-screw anchors (``cam_y`` / ``cam_z`` /
        ``cam_offset``) on ``self`` — downstream steps (camera_20) build
        this assembly and read those anchors to mount onto the 1/4-20
        stud."""
        bracket = CornerBracket().build()
        bhcs    = Screw("BHCS", "M5", BHCS_LENGTH).build()
        tnut    = TNut("hammer", "M5").build()
        cam_screw = Screw("SHCS", "1/4-20", CAM_LENGTH).build()
        cam_nut   = Nut("hex", "1/4-20").build()

        hx          = face_depth / 2                 # horizontal-hole X (centered in depth)
        hammer_half = TNUT_LENGTHS["hammer"] / 2     # centres the T-nut bore under the hole

        # Vertical placement: assembled is the base pose, exploding just adds
        # gaps. ``bracket_z`` lifts the deck (horizontal face, local z
        # 0..plate_thick) to its mounted height — one HAMMER_TOTAL_HEIGHT above
        # the (notional) slot face, where the boss would seat it once dropped
        # into a 2020 slot at camera_40. BHCS underheads seat on the deck top;
        # ``cam_offset`` slides the camera screw / nut apart along ±X.
        bracket_z  = HAMMER_TOTAL_HEIGHT + (BRACKET_GAP if self.exploded else 0)
        screw_z    = bracket_z + plate_thick + (SCREW_GAP + BHCS_LENGTH if self.exploded else 0)
        cam_offset = CAM_GAP if self.exploded else 0          # nut / gooseneck side (−X)
        cam_screw_offset = CAM_SCREW_GAP if self.exploded else 0   # SHCS side (+X)

        bracket.move(Location((0, 0, bracket_z)))
        bhcs.move(Location((hx, 0, screw_z)))

        # ── Frame side: hammer T-nut on the BHCS shank ────────────────────────
        # T-nut bore vertical, boss up: local +Y → world +Z, length runs along
        # world −Y (z_dir = −Y), so the +hammer_half Y offset centres the bore
        # under the hole. ``tnut_z`` is the world Z of the T-nut origin (local
        # y = 0); its boss top sits HAMMER_TOTAL_HEIGHT above that.
        #   * exploded  — on the deck floor (z = 0), screw floating above it.
        #   * assembled — no extrusion here to clamp against, so the nut is NOT
        #     pulled flush to the bracket; it hangs LOOSE on the BHCS shank tip,
        #     threaded on HAMMER_LOOSE_ENGAGE mm with a gap below the deck — the
        #     pre-mount state. (Once mounted at camera_40 it ends up in the slot,
        #     hidden inside the extrusion — the loose pose there isn't visible.)
        if self.exploded:
            tnut_z = 0
        else:
            screw_tip_z = screw_z - BHCS_LENGTH
            tnut_z = screw_tip_z + HAMMER_LOOSE_ENGAGE - HAMMER_TOTAL_HEIGHT
        tnut.move(Location(Plane(
            origin=(hx, hammer_half, tnut_z),
            x_dir=(1, 0, 0),
            z_dir=(0, -1, 0),
        )))

        # ── Camera side: 1/4-20 SHCS + hex nut through the centered hole ──────
        # The vertical-face hole sits at world (x 0..plate_thick, y = 0,
        # z = face_depth/2). The screw axis is along X with its head on the
        # +X face and the hex nut clamping the −X face.
        cam_z = bracket_z + face_depth / 2
        cam_screw.move(Location(Plane(
            origin=(plate_thick + cam_screw_offset, 0, cam_z),
            x_dir=(0, 1, 0),
            z_dir=(1, 0, 0),          # head along +X, shank into the plate (−X)
        )))
        cam_nut.move(Location(Plane(
            origin=(-cam_offset, 0, cam_z),
            x_dir=(0, 1, 0),
            z_dir=(-1, 0, 0),         # bore coaxial with the screw, on the −X face
        )))

        # Camera-screw anchors for downstream steps: the stud axis is world
        # −X through (cam_y, cam_z) with its head on the +X face; cam_offset
        # is the per-variant X explode shift (0 when assembled).
        self.cam_y = 0
        self.cam_z = cam_z
        self.cam_offset = cam_offset

        return [bracket, bhcs, tnut, cam_screw, cam_nut]


if __name__ == "__main__":
    for exploded in (True, False):
        asm = Camera10Bracket(exploded=exploded)
        asm.export()
        asm.render()
