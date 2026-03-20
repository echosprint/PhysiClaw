"""Generate a 3D-printable stylus holder with cap, handle, cone, and clamp.

Dependencies: trimesh, manifold3d (boolean engine for trimesh)

To fit a different stylus tip, change TIP_OD, TIP_ID, and TIP_H below:
  TIP_OD — outer diameter of the stylus tip (the cap grips around this)
  TIP_ID — inner diameter of the stylus tip (the inner annulus fits inside this)
  TIP_H  — tip height (for reference)
"""

import math
import os
from functools import reduce

try:
    import trimesh
except ImportError:
    raise SystemExit(
        "Missing dependencies. Install with:\n  uv sync --group stylus"
    )

SEGMENTS = 64  # mesh resolution (number of circular sections)

TIP_OD = 5.3  # stylus tip outer diameter (mm)
TIP_ID = 3  # stylus tip inner diameter (mm)
TIP_H = 6.2  # stylus tip height (mm)

CAP_H = 3  # cap height (mm)
WALL = 0.8  # wall thickness (mm)
EPS = 0.01  # overlap epsilon for boolean operations (mm)

HANDLE_R = 0.75  # handle radius (mm)
HANDLE_L = 30  # handle length (mm)

CONE_L = 30  # visible cone length (mm)
CONE_R = 3  # cone large end radius (mm)

L_TUBE_R_MIN = 3.5  # vertical tube inner radius (mm)
L_TUBE_R_MAX = 4.5  # vertical tube outer radius (mm)
L_TUBE_H = 100  # vertical tube height (mm)

CLAMP_W = 2  # clamp tab width (mm)
CLAMP_T = 4  # clamp tab thickness (mm)
CLAMP_H = 80  # clamp tab height (mm)

PI_HALF = math.pi / 2


def create_cap():
    outer = trimesh.creation.annulus(
        r_min=TIP_OD / 2.0, r_max=(TIP_OD / 2.0 + WALL), height=CAP_H, sections=SEGMENTS
    )

    inner = trimesh.creation.annulus(
        r_min=0.7, r_max=TIP_ID / 2.0, height=CAP_H + 0.5, sections=SEGMENTS
    )
    inner.apply_translation([0, 0, -0.25])

    # torus at the bottom of inner annulus to thicken the end
    inner_mid_r = (0.7 + TIP_ID / 2.0) / 2.0
    tube_r = (TIP_ID / 2.0 - 0.5) / 2.0
    torus = trimesh.creation.torus(major_radius=inner_mid_r, minor_radius=tube_r)
    inner_bottom_z = -0.05 - (CAP_H + 0.5) / 2.0
    torus.apply_translation([0, 0, inner_bottom_z])

    top = trimesh.creation.annulus(
        r_min=(0.7 + EPS),
        r_max=(TIP_OD / 2.0 + WALL - EPS),
        height=0.5,
        sections=SEGMENTS,
    )
    top.apply_translation([0, 0, CAP_H / 2 + 0.25 - EPS])

    print(f"cap fit: inner_d={TIP_ID:.1f} outer_d={TIP_OD:.1f} cap_h={CAP_H:.1f}mm")
    return [outer, inner, torus, top]


def create_handle():
    top_ring_z_max = CAP_H / 2 + 0.5 - EPS
    handle_z = top_ring_z_max - HANDLE_R

    # main handle cylinder
    handle = trimesh.creation.cylinder(
        radius=HANDLE_R, height=HANDLE_L, sections=SEGMENTS
    )
    handle.apply_transform(trimesh.transformations.rotation_matrix(PI_HALF, [0, 1, 0]))
    handle_x = HANDLE_L / 2 + TIP_OD / 2.0 + WALL / 2
    handle.apply_translation([handle_x, 0, handle_z])

    # support handle
    support_handle = trimesh.creation.cylinder(
        radius=0.5, height=HANDLE_L + 15, sections=SEGMENTS
    )
    support_handle.apply_transform(
        trimesh.transformations.rotation_matrix(PI_HALF, [0, 1, 0])
    )
    support_handle.apply_translation([handle_x + 7.51, 0, handle_z - HANDLE_R])

    # support strut
    support_h = handle_z + HANDLE_R + CAP_H / 2
    support = trimesh.creation.cylinder(radius=0.5, height=2, sections=SEGMENTS)
    support_x = TIP_OD / 2.0 + WALL / 2 + 0.3
    support.apply_translation([support_x, 0, handle_z - support_h / 2 + HANDLE_R])

    return [handle, support_handle, support]


def create_cone():
    top_ring_z_max = CAP_H / 2 + 0.5 - EPS
    handle_end_x = HANDLE_L + TIP_OD / 2.0 + WALL / 2

    cone_full_h = CONE_L * CONE_R / (CONE_R - HANDLE_R)
    cone_overlap = cone_full_h - CONE_L

    cone = trimesh.creation.cone(radius=CONE_R, height=cone_full_h, sections=SEGMENTS)
    # rotate so apex points toward -x (into the handle)
    cone.apply_transform(trimesh.transformations.rotation_matrix(-PI_HALF, [0, 1, 0]))
    cone.apply_translation(
        [
            handle_end_x - cone_overlap + cone_full_h / 2 + 20,
            0,
            top_ring_z_max - HANDLE_R,
        ]
    )

    # cut below z = -1
    cone_center_x = handle_end_x - cone_overlap + cone_full_h / 2 + 20
    cut_box = trimesh.creation.box(
        extents=[cone_full_h + 2, CONE_R * 2 + 2, CONE_R * 2 + 2]
    )
    cut_box.apply_translation([cone_center_x, 0, -1 - (CONE_R + 1)])
    cone = cone.difference(cut_box)

    return [cone]


def create_clamp():
    handle_end_x = HANDLE_L + TIP_OD / 2.0 + WALL / 2
    cone_full_h = CONE_L * CONE_R / (CONE_R - HANDLE_R)
    cone_overlap = cone_full_h - CONE_L
    cone_end_x = handle_end_x - cone_overlap + cone_full_h / 2 + 23
    tube_z = L_TUBE_H / 2 - 1 + EPS

    # vertical tube
    l_tube = trimesh.creation.annulus(
        r_min=L_TUBE_R_MIN, r_max=L_TUBE_R_MAX, height=L_TUBE_H, sections=SEGMENTS
    )
    l_tube.apply_translation([cone_end_x, 0, tube_z])

    # clamp tabs on both sides
    clamp_pos = trimesh.creation.box(extents=[CLAMP_T, CLAMP_W, CLAMP_H])
    clamp_pos.apply_translation([cone_end_x, L_TUBE_R_MIN + CLAMP_W / 2 - EPS, tube_z])

    clamp_neg = trimesh.creation.box(extents=[CLAMP_T, CLAMP_W, CLAMP_H])
    clamp_neg.apply_translation([cone_end_x, -L_TUBE_R_MIN - CLAMP_W / 2 + EPS, tube_z])

    return [l_tube, clamp_pos, clamp_neg]


def main():
    cap = create_cap()
    handle = create_handle()
    cone = create_cone()
    clamp = create_clamp()

    all_parts = cap + handle + cone + clamp
    names = [
        "cap_outer",
        "cap_inner",
        "cap_torus",
        "cap_top",
        "handle",
        "support_handle",
        "support",
        "cone",
        "clamp_tube",
        "clamp_pos",
        "clamp_neg",
    ]
    for name, mesh in zip(names, all_parts):
        b = mesh.bounds
        e = mesh.extents
        print(
            f"  {name:16s}  {e[0]:5.1f} x {e[1]:5.1f} x {e[2]:5.1f}  bbox x[{b[0][0]:7.2f},{b[1][0]:7.2f}] y[{b[0][1]:7.2f},{b[1][1]:7.2f}] z[{b[0][2]:7.2f},{b[1][2]:7.2f}]"
        )

    result = reduce(lambda a, b: a.union(b), all_parts)

    project_root = os.path.dirname(os.path.dirname(__file__))
    out_dir = os.path.join(project_root, "data", "stylus")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "stylus.stl")
    result.export(out_path)
    print("done:", len(result.faces), "faces →", out_path)
    print("watertight:", result.is_watertight)
    e = result.extents
    print(f"size: {e[0]:.1f} x {e[1]:.1f} x {e[2]:.1f} mm")


if __name__ == "__main__":
    main()
