import math
from functools import reduce

import trimesh

SEGMENTS = 64  # mesh resolution (number of circular sections)

TIP_OD = 5.3  # stylus tip outer diameter (mm)
TIP_ID = 3  # stylus tip inner diameter (mm)
TIP_H = 6.2  # stylus tip height (mm)

CAP_H = 3  # cap height (mm)
WALL = 0.2  # wall thickness (mm)
EPS = 0.01  # overlap epsilon for boolean operations (mm)

HANDLE_R = 0.75  # handle radius (mm)
HANDLE_L = 30  # handle length (mm)

CONE_L = 30  # visible cone length (mm)
CONE_R = 3  # cone large end radius (mm)

L_TUBE_R_MIN = 3.5  # vertical tube inner radius (mm)
L_TUBE_R_MAX = 4.5  # vertical tube outer radius (mm)
L_TUBE_H = 100  # vertical tube height (mm)

CLAMP_W = 1  # clamp tab width (mm)
CLAMP_T = 4  # clamp tab thickness (mm)
CLAMP_H = 80  # clamp tab height (mm)

PI_HALF = math.pi / 2


def create_cap():
    outer = trimesh.creation.annulus(
        r_min=TIP_OD / 2.0, r_max=(TIP_OD / 2.0 + WALL), height=CAP_H, sections=SEGMENTS
    )

    inner = trimesh.creation.annulus(
        r_min=(TIP_ID / 2.0 - 2 * WALL), r_max=TIP_ID / 2.0, height=CAP_H + 0.5, sections=SEGMENTS
    )
    inner.apply_translation([0, 0, -0.25])

    top = trimesh.creation.annulus(
        r_min=(TIP_ID / 2.0 - 2 * WALL + EPS), r_max=(TIP_OD / 2.0 + WALL - EPS),
        height=0.5, sections=SEGMENTS
    )
    top.apply_translation([0, 0, CAP_H / 2 + 0.25 - EPS])

    return [outer, inner, top]


def create_handle():
    top_ring_z_max = CAP_H / 2 + 0.5 - EPS
    handle_z = top_ring_z_max - HANDLE_R

    # main handle cylinder
    handle = trimesh.creation.cylinder(radius=HANDLE_R, height=HANDLE_L, sections=SEGMENTS)
    handle.apply_transform(trimesh.transformations.rotation_matrix(PI_HALF, [0, 1, 0]))
    handle_x = HANDLE_L / 2 + TIP_OD / 2.0 + WALL / 2
    handle.apply_translation([handle_x, 0, handle_z])

    # support strut
    support_h = handle_z + HANDLE_R + CAP_H / 2
    support = trimesh.creation.cylinder(radius=0.5, height=2, sections=SEGMENTS)
    support_x = TIP_OD / 2.0 + WALL / 2 + 0.3
    support.apply_translation([support_x, 0, handle_z - support_h / 2 + HANDLE_R])

    return [handle, support]


def create_cone():
    top_ring_z_max = CAP_H / 2 + 0.5 - EPS
    handle_end_x = HANDLE_L + TIP_OD / 2.0 + WALL / 2

    cone_full_h = CONE_L * CONE_R / (CONE_R - HANDLE_R)
    cone_overlap = cone_full_h - CONE_L

    cone = trimesh.creation.cone(radius=CONE_R, height=cone_full_h, sections=SEGMENTS)
    # rotate so apex points toward -x (into the handle)
    cone.apply_transform(trimesh.transformations.rotation_matrix(-PI_HALF, [0, 1, 0]))
    cone.apply_translation([handle_end_x - cone_overlap + cone_full_h / 2 + 20, 0, top_ring_z_max - HANDLE_R])

    # cut below z = -1
    cut_box = trimesh.creation.box(extents=[200, 200, 200])
    cut_box.apply_translation([0, 0, -1 - 100])
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


def print_bounds(parts):
    for name, mesh in parts.items():
        b = mesh.bounds
        c = mesh.centroid
        print(f"{name:16s}  bbox x[{b[0][0]:7.2f},{b[1][0]:7.2f}] y[{b[0][1]:7.2f},{b[1][1]:7.2f}] z[{b[0][2]:7.2f},{b[1][2]:7.2f}]  center({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})")


def main():
    cap = create_cap()
    handle = create_handle()
    cone = create_cone()
    clamp = create_clamp()

    all_parts = cap + handle + cone + clamp

    parts = {
        "outer_annulus": cap[0],
        "inner_annulus": cap[1],
        "top_ring": cap[2],
        "handle": handle[0],
        "support": handle[1],
        "cone": cone[0],
        "l_tube": clamp[0],
        "clamp_pos": clamp[1],
        "clamp_neg": clamp[2],
    }
    print_bounds(parts)

    result = reduce(lambda a, b: a.union(b), all_parts)
    import os
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "stylus")
    os.makedirs(out_dir, exist_ok=True)
    result.export(os.path.join(out_dir, "stylus.stl"))
    print("done:", len(result.faces), "faces")


if __name__ == "__main__":
    main()
