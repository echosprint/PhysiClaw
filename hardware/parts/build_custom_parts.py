"""Build the 3D-print package for the custom (SLS/MJF) parts.

Exports every custom part as a STEP file and bundles them with English and
Chinese parts manifests into a single zip:

    hardware/output/print_3d/physiclaw_custom_parts.zip

One STEP per unique design; the print quantity is baked into the ``_xN``
filename suffix and listed in the manifests. Run from the repo root::

    uv run --group cad python -m hardware.parts.build_custom_parts
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

from build123d import export_step

from hardware.parts.custom.belt_clamp import BeltClamp
from hardware.parts.custom.idler_mount_front import IdlerMountFront
from hardware.parts.custom.idler_mount_motor import IdlerMountMotor
from hardware.parts.custom.pcb_holder import PcbHolder
from hardware.parts.custom.phone_bed import PhoneBed
from hardware.parts.custom.solenoid_mount import SolenoidMount
from hardware.parts.custom.tube_holder import TubeHolder
from hardware.parts.custom.xy_joint_left import XyJointLeft
from hardware.parts.custom.xy_joint_right import XyJointRight
from hardware.scheme import OUTPUT_DIR

OUT_DIR = OUTPUT_DIR / "print_3d"
ZIP_PATH = OUT_DIR / "physiclaw_custom_parts.zip"

# (class, qty to print, English name, 中文名). Quantities match the consolidated
# BOM's "Custom parts" line (11 pieces across 9 designs). The STEP filename comes
# from the part itself (BasePart.output_path()), so it isn't restated here.
CUSTOM_PARTS: list[tuple[type, int, str, str]] = [
    (BeltClamp, 1, "Belt clamp", "同步带夹具"),
    (IdlerMountFront, 2, "Idler mount (front)", "惰轮支座（前）"),
    (IdlerMountMotor, 2, "Idler mount (motor)", "惰轮支座（电机侧）"),
    (PcbHolder, 1, "Control board holder", "控制板支架"),
    (PhoneBed, 1, "Phone bed", "手机托架"),
    (SolenoidMount, 1, "Solenoid mount", "电磁铁支架"),
    (TubeHolder, 1, "Tube holder", "导管支架"),
    (XyJointLeft, 1, "XY joint (left)", "XY 连接件（左）"),
    (XyJointRight, 1, "XY joint (right)", "XY 连接件（右）"),
]


# Per-language manifest strings. Process and material are separate lines.
_MANIFEST = {
    "en": {
        "title": "PhysiClaw — Custom 3D-printed parts",
        "process": "Process:  Selective Laser Sintering (SLS) or Multi Jet Fusion (MJF)",
        "material": "Material: PA12 nylon (or equivalent)",
        "color": "Color:    Black",
        "units": "Units:    millimeters (mm)",
        "intro": "Print one STEP per row, in the quantity shown.",
        "qty_note": "The _xN suffix in each filename is how many to print (e.g. _x2 = print 2).",
        "cols": ("STEP file", "Qty", "Part"),
        "total": "Total",
        "unit": "pieces",
    },
    "zh": {
        "title": "PhysiClaw — 定制 3D 打印件",
        "process": "打印工艺：选择性激光烧结（SLS）或 多射流熔融（MJF）",
        "material": "材料：PA12 尼龙（或同等材料）",
        "color": "颜色：黑色",
        "units": "尺寸单位：毫米（mm）",
        "intro": "每个 STEP 按所列数量打印。",
        "qty_note": "文件名中的 _xN 表示打印数量（例如 _x2 表示打印 2 个）。",
        "cols": ("STEP 文件", "数量", "零件"),
        "total": "合计",
        "unit": "件",
    },
}


def _manifest(rows: list[tuple[str, int, str, str]], lang: str) -> str:
    """Single-language parts manifest (file · qty · name). The name is the last
    column so a CJK name (double-width) never throws off the file/qty columns."""
    m = _MANIFEST[lang]
    c_file, c_qty, c_name = m["cols"]
    fw = max(len(c_file), *(len(f) for f, *_ in rows))
    head = f"{c_file:<{fw}}  {c_qty:>3}  {c_name}"
    out = [
        m["title"],
        "",
        m["process"],
        m["material"],
        m["color"],
        m["units"],
        "",
        m["intro"],
        m["qty_note"],
        "",
        head,
        "-" * len(head),
    ]
    total = 0
    for fname, qty, en, zh in rows:
        out.append(f"{fname:<{fw}}  {qty:>3}  {en if lang == 'en' else zh}")
        total += qty
    out += ["-" * len(head), f"{m['total']:<{fw}}  {total:>3}  {m['unit']}"]
    return "\n".join(out) + "\n"


def build() -> Path:
    """Export each custom part to STEP and write the zip. Returns the zip path."""
    # Start clean so a renamed zip or an old format (e.g. a previous STL build)
    # never lingers in the package directory.
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, int, str, str]] = []
    with (
        tempfile.TemporaryDirectory() as tmp,
        zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf,
    ):
        for cls, qty, en, zh in CUSTOM_PARTS:
            part = cls(qty=qty)
            shape = part.build()
            # output_path().name is "<stem>_x<qty>.step" (from BasePart); just
            # prefix the project name for the print package.
            fname = f"physiclaw_{part.output_path().name}"
            step = Path(tmp) / fname
            export_step(shape, str(step))
            zf.write(step, arcname=fname)
            rows.append((fname, qty, en, zh))
            print(f"  + {fname}  ×{qty}  {en} / {zh}")
        zf.writestr("3d_printed_parts_guide.txt", _manifest(rows, "en"))
        zf.writestr("3d打印零件说明.txt", _manifest(rows, "zh"))
    total = sum(qty for _, qty, _, _ in rows)
    print(f"\nwrote {ZIP_PATH}  ({len(rows)} designs, {total} pieces)")
    return ZIP_PATH


if __name__ == "__main__":
    build()
