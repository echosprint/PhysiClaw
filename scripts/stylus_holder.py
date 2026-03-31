"""
Custom Stylus — generates a PDF fabrication drawing.

Usage:
    uv run --group drawing python scripts/stylus_holder.py

Output:
    data/stylus/bracket_drawing.pdf — A4 dimensioned drawing for fabrication

- Steel rod φ2.5mm, 75mm long (horizontal)
- Left end: threaded M3 section (φ3mm, 5mm long, going UP) — no head
- Right end: inserts into knurled tube (OD 9mm, ID 7mm, 100mm, open both ends)
"""

from reportlab.lib.pagesizes import A4  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
from reportlab.lib.units import mm  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
from reportlab.pdfgen import canvas  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
from reportlab.lib.colors import black, white, HexColor  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
from reportlab.pdfbase import pdfmetrics  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
from reportlab.pdfbase.ttfonts import TTFont  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
import math
import os


_FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fonts")
_FONT_PATH = os.path.join(_FONT_DIR, "NotoSansSC-Regular.ttf")
_FONT_VAR_URL = "https://github.com/google/fonts/raw/main/ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf"


HAS_CJK = False

def _try_load_cjk_font():
    """Try to download and register Noto Sans SC Regular. Sets HAS_CJK flag."""
    global HAS_CJK
    try:
        if not os.path.exists(_FONT_PATH):
            os.makedirs(_FONT_DIR, exist_ok=True)
            var_path = os.path.join(_FONT_DIR, "NotoSansSC-Variable.ttf")
            if not os.path.exists(var_path):
                print("Downloading Noto Sans SC font...")
                import urllib.request
                urllib.request.urlretrieve(_FONT_VAR_URL, var_path)
            # Instantiate variable font at Regular weight (400)
            from fontTools import ttLib as ftLib  # pyright: ignore[reportMissingImports]
            from fontTools.varLib import instancer  # pyright: ignore[reportMissingImports]
            vf = ftLib.TTFont(var_path)
            static = instancer.instantiateVariableFont(
                vf, {"wght": 400}, updateFontNames=True)
            static.save(_FONT_PATH)
            static.close()
            vf.close()
            os.remove(var_path)
            print(f"Saved Regular weight to {_FONT_PATH}")
        pdfmetrics.registerFont(TTFont('CJK', _FONT_PATH))
        HAS_CJK = True
    except Exception as e:
        print(f"CJK font unavailable ({e}), Chinese text will be omitted")

_try_load_cjk_font()

_project_root = os.path.dirname(os.path.dirname(__file__))
_out_dir = os.path.join(_project_root, "data", "stylus")
os.makedirs(_out_dir, exist_ok=True)
OUTPUT = os.path.join(_out_dir, "bracket_drawing.pdf")
c = canvas.Canvas(OUTPUT, pagesize=A4)
W, H = A4

# Page margins (printable area for most printers)
MARGIN = 15 * mm

THIN  = 0.3 * mm
MED   = 0.5 * mm
THICK = 0.8 * mm
ARROW = 2.5 * mm
DIM_FONT = 9

GRAY = HexColor("#BBBBBB")
BLUE = HexColor("#0055AA")
HATCH_COL = HexColor("#AAAAAA")

SCALE = 3.0
def s(v): return v * SCALE

# ── Helpers ──
def draw_arrow(x, y, angle, length=ARROW):
    rad = math.radians(angle)
    dx = length * math.cos(rad)
    dy = length * math.sin(rad)
    hw = length * 0.3
    px = -hw * math.sin(rad); py = hw * math.cos(rad)
    p = c.beginPath()
    p.moveTo(x, y)
    p.lineTo(x - dx + px, y - dy + py)
    p.lineTo(x - dx - px, y - dy - py)
    p.close()
    c.drawPath(p, fill=1, stroke=0)

def dim_h(x1, x2, y, text, side='below', offset=12*mm):
    c.setLineWidth(THIN); c.setStrokeColor(BLUE); c.setFillColor(BLUE)
    dy = y - offset if side == 'below' else y + offset
    ext = dy - 1.5*mm if side == 'below' else dy + 1.5*mm
    c.setDash([], 0)
    c.line(x1, y, x1, ext); c.line(x2, y, x2, ext)
    c.line(x1, dy, x2, dy)
    draw_arrow(x1, dy, 0); draw_arrow(x2, dy, 180)
    c.setFont("Helvetica", DIM_FONT)
    tw = c.stringWidth(text, "Helvetica", DIM_FONT)
    mx = (x1 + x2) / 2
    c.setFillColor(white); c.rect(mx-tw/2-2, dy-4, tw+4, 8, fill=1, stroke=0)
    c.setFillColor(BLUE); c.drawCentredString(mx, dy-2.5, text)
    c.setStrokeColor(black); c.setFillColor(black)

def dim_v(y1, y2, x, text, side='right', offset=12*mm):
    c.setLineWidth(THIN); c.setStrokeColor(BLUE); c.setFillColor(BLUE)
    dx = x + offset if side == 'right' else x - offset
    ext = dx + 1.5*mm if side == 'right' else dx - 1.5*mm
    c.setDash([], 0)
    c.line(x, y1, ext, y1); c.line(x, y2, ext, y2)
    c.line(dx, y1, dx, y2)
    draw_arrow(dx, y1, 270); draw_arrow(dx, y2, 90)
    c.saveState()
    c.setFont("Helvetica", DIM_FONT)
    tw = c.stringWidth(text, "Helvetica", DIM_FONT)
    my = (y1 + y2) / 2
    c.translate(dx, my); c.rotate(90)
    c.setFillColor(white); c.rect(-tw/2-2, -4, tw+4, 8, fill=1, stroke=0)
    c.setFillColor(BLUE); c.drawCentredString(0, -2.5, text)
    c.restoreState()
    c.setStrokeColor(black); c.setFillColor(black)

def center_h(x1, x2, y):
    c.setLineWidth(THIN); c.setStrokeColor(GRAY)
    c.setDash([6,2,1,2], 0); c.line(x1, y, x2, y)
    c.setDash([], 0); c.setStrokeColor(black)

def center_v(y1, y2, x):
    c.setLineWidth(THIN); c.setStrokeColor(GRAY)
    c.setDash([6,2,1,2], 0); c.line(x, y1, x, y2)
    c.setDash([], 0); c.setStrokeColor(black)

def dashed(x1, y1, x2, y2):
    c.setLineWidth(MED); c.setStrokeColor(black)
    c.setDash([4,2], 0); c.line(x1, y1, x2, y2)
    c.setDash([], 0)

def hatch_rect(x1, y1, x2, y2, sp=1.5*mm):
    c.saveState()
    c.setLineWidth(0.2*mm); c.setStrokeColor(HATCH_COL)
    p = c.beginPath(); p.rect(x1, y1, x2-x1, y2-y1)
    c.clipPath(p, stroke=0)
    h = y2 - y1
    n = int(((x2-x1) + h) / sp) + 4
    for i in range(-n, n):
        lx = x1 + i * sp
        c.line(lx, y1, lx + h, y2)
    c.restoreState()


# ═══════════════════════════════════════
#  Part dimensions (mm)
# ═══════════════════════════════════════
ROD_DIA   = 2.5
ROD_LEN   = 75
TUBE_OD   = 9
TUBE_ID   = 7
TUBE_WALL = (TUBE_OD - TUBE_ID) / 2  # = 1mm
TUBE_LEN  = 100
SCREW_DIA = 3      # M3 threaded section
SCREW_LEN = 5

# ═══════════════════════════════════════
#  Layout
# ═══════════════════════════════════════
OX_LEFT = 100
TUBE_TOP_Y = H - 200   # tube top edge
HOLE_MARGIN = 2        # rod hole is 2mm below tube top

rod_left  = OX_LEFT
rod_right = OX_LEFT + s(ROD_LEN)
rod_top   = TUBE_TOP_Y - s(HOLE_MARGIN)    # 2mm below tube top
rod_bot   = rod_top - s(ROD_DIA)            # rod hangs down from there
OY        = (rod_top + rod_bot) / 2         # rod center line

# Rod right end = tube outer right wall (rod goes all the way through)
tube_cx  = rod_right - s(TUBE_OD/2)     # tube centered so rod exits at far wall
tube_top = TUBE_TOP_Y
tube_bot = TUBE_TOP_Y - s(TUBE_LEN)
tube_ol  = tube_cx - s(TUBE_OD/2)
tube_or  = tube_cx + s(TUBE_OD/2)
tube_il  = tube_cx - s(TUBE_ID/2)
tube_ir  = tube_cx + s(TUBE_ID/2)

screw_l   = rod_left                   # screw left aligns with rod left
screw_r   = rod_left + s(SCREW_DIA)
screw_cx  = (screw_l + screw_r) / 2
screw_bot = rod_top                    # screw starts at rod top edge
screw_top = rod_top + s(SCREW_LEN)

# ═══════════════════════════════════════
#  Title
# ═══════════════════════════════════════
c.setFont("Helvetica-Bold", 16)
c.drawCentredString(W/2, H - MARGIN - 18, "PhysiClaw Stylus Rod")
if HAS_CJK:
    c.setFont("CJK", 12)
    c.drawCentredString(W/2, H - MARGIN - 36, "触控笔杆设计图")
c.setFont("Helvetica", 10)
c.drawCentredString(W/2, H - MARGIN - 52, "Front View  |  Dimensions in mm  |  Stainless Steel")
if HAS_CJK:
    c.setFont("CJK", 9)
    c.drawCentredString(W/2, H - MARGIN - 66, "正视图  |  单位 mm  |  不锈钢")

c.setFont("Helvetica", 9); c.setFillColor(HexColor("#444444"))
c.drawString(MARGIN + 4, H - MARGIN - 84, "M3 external thread φ 3 × 5  —  Rod φ 2.5 × 75  —  Knurled Tube φ 9 / φ 7 × 100")
if HAS_CJK:
    # Chinese line: mix CJK for Chinese chars, Helvetica for φ and numbers
    y_cn = H - MARGIN - 98
    x = MARGIN + 4
    c.setFont("CJK", 9)
    c.drawString(x, y_cn, "M3 外螺纹"); x += c.stringWidth("M3 外螺纹", "CJK", 9)
    c.setFont("Helvetica", 9)
    c.drawString(x, y_cn, " φ 3 × 5  —  "); x += c.stringWidth(" φ 3 × 5  —  ", "Helvetica", 9)
    c.setFont("CJK", 9)
    c.drawString(x, y_cn, "钢棒"); x += c.stringWidth("钢棒", "CJK", 9)
    c.setFont("Helvetica", 9)
    c.drawString(x, y_cn, " φ 2.5 × 75  —  "); x += c.stringWidth(" φ 2.5 × 75  —  ", "Helvetica", 9)
    c.setFont("CJK", 9)
    c.drawString(x, y_cn, "滚花管材"); x += c.stringWidth("滚花管材", "CJK", 9)
    c.setFont("Helvetica", 9)
    c.drawString(x, y_cn, " φ 9 / φ 7 × 100")
c.setFillColor(black)


# ═══════════════════════════════════════
#  DRAW: Rod (horizontal) — penetrates through tube
# ═══════════════════════════════════════
c.setLineWidth(THICK)

# Segment 1: rod_left to tube outer left (visible)
c.line(rod_left, rod_top, tube_ol, rod_top)
c.line(rod_left, rod_bot, tube_ol, rod_bot)

# Segment 2: through left wall (hidden — inside tube wall)
dashed(tube_ol, rod_top, tube_il, rod_top)
dashed(tube_ol, rod_bot, tube_il, rod_bot)

# Segment 3: across bore (visible)
c.setLineWidth(THICK)
c.line(tube_il, rod_top, tube_ir, rod_top)
c.line(tube_il, rod_bot, tube_ir, rod_bot)

# Segment 4: through right wall (hidden — inside tube wall)
dashed(tube_ir, rod_top, tube_or, rod_top)
dashed(tube_ir, rod_bot, tube_or, rod_bot)

# Rod right end at tube_or (flush with outer wall)
c.setLineWidth(THICK)
c.line(tube_or, rod_bot, tube_or, rod_top)   # close off right end

# Center line
center_h(rod_left - s(8), tube_or + s(8), OY)


# ═══════════════════════════════════════
#  DRAW: M3 threaded rod (left, going UP)
#  Just a φ3mm rod sticking up — no head
# ═══════════════════════════════════════
c.setLineWidth(THICK)

# Shank walls
c.line(screw_l, screw_bot, screw_l, screw_top)
c.line(screw_r, screw_bot, screw_r, screw_top)

# Flat top end
c.line(screw_l, screw_top, screw_r, screw_top)

# Transition from rod (φ2.5) to screw (φ3)
# Left sides are aligned, screw extends 0.5mm further right
# Left wall: continuous from rod_bot up through screw_top
c.line(rod_left, rod_bot, rod_left, screw_bot)   # rod left wall up to junction
# Right shoulder where screw is wider than rod
c.line(rod_left + s(ROD_DIA), screw_bot, screw_r, screw_bot)

# Thread lines (fine horizontal)
c.setLineWidth(0.15*mm); c.setStrokeColor(GRAY)
pitch = 0.5
for i in range(1, int(SCREW_LEN / pitch)):
    ty = screw_bot + i * s(pitch)
    if ty < screw_top - s(0.2):
        c.line(screw_l + 0.5, ty, screw_r - 0.5, ty)
c.setStrokeColor(black)

# Center line of screw
center_v(rod_bot - s(3), screw_top + s(5), screw_cx)


# ═══════════════════════════════════════
#  DRAW: Knurled Tube (right, going DOWN, open-ended)
#  Rod penetrates both walls 2mm below top
# ═══════════════════════════════════════
c.setLineWidth(THICK)

# Outer walls
c.line(tube_ol, tube_top, tube_ol, tube_bot)
c.line(tube_or, tube_top, tube_or, tube_bot)

# Inner walls (hidden) — split around rod penetration
# Above rod
dashed(tube_il, tube_top, tube_il, rod_top)
dashed(tube_ir, tube_top, tube_ir, rod_top)
# Below rod
dashed(tube_il, rod_bot, tube_il, tube_bot)
dashed(tube_ir, rod_bot, tube_ir, tube_bot)

# Bottom — OPEN ended
c.setLineWidth(THICK)
c.line(tube_ol, tube_bot, tube_il, tube_bot)
c.line(tube_ir, tube_bot, tube_or, tube_bot)

# Top — OPEN ended
c.line(tube_ol, tube_top, tube_il, tube_top)
c.line(tube_ir, tube_top, tube_or, tube_top)

# Hatching on tube wall at top
hatch_rect(tube_ol, tube_top - s(2.5), tube_il, tube_top + s(0.3), sp=1.2*mm)
hatch_rect(tube_ir, tube_top - s(2.5), tube_or, tube_top + s(0.3), sp=1.2*mm)

# Hatching on tube wall at bottom
hatch_rect(tube_ol, tube_bot - s(0.3), tube_il, tube_bot + s(2.5), sp=1.2*mm)
hatch_rect(tube_ir, tube_bot - s(0.3), tube_or, tube_bot + s(2.5), sp=1.2*mm)

# Center line of tube
center_v(tube_bot - s(7), tube_top + s(3), tube_cx)


# ═══════════════════════════════════════
#  DIMENSIONS
# ═══════════════════════════════════════

# Rod: 75mm
dim_h(rod_left, rod_right, rod_bot, "75", side='below', offset=10*mm)

# Rod: φ2.5
dim_v(rod_bot, rod_top, rod_left, "φ 2.5", side='left', offset=18*mm)

# Screw: 5mm
dim_v(screw_bot, screw_top, screw_r, "5", side='right', offset=25*mm)

# Screw: φ3 (M3)
dim_h(screw_l, screw_r, screw_top, "φ 3 (M3)", side='above', offset=8*mm)

# Knurled Tube: 100mm
dim_v(tube_bot, tube_top, tube_or, "100", side='right', offset=18*mm)

# Knurled Tube OD: φ9
# Knurled Tube OD: φ 9 — leader line to right (span too narrow for inline dim)
c.setLineWidth(THIN); c.setStrokeColor(BLUE); c.setFillColor(BLUE)
od_y = tube_bot - s(15)
c.setDash([], 0)
c.line(tube_ol, tube_bot, tube_ol, od_y - 1.5*mm)
c.line(tube_or, tube_bot, tube_or, od_y - 1.5*mm)
c.line(tube_ol, od_y, tube_or, od_y)
draw_arrow(tube_ol, od_y, 0)
draw_arrow(tube_or, od_y, 180)
# Leader out to the right with label
ldr_od_x = tube_or + 15*mm
c.line(tube_or, od_y, ldr_od_x, od_y)
c.setFont("Helvetica", DIM_FONT)
c.drawString(ldr_od_x + 2, od_y - 3, "φ 9 (OD)")
c.setStrokeColor(black); c.setFillColor(black)

# Knurled Tube ID: φ 7 — leader line to left
c.setLineWidth(THIN); c.setStrokeColor(BLUE); c.setFillColor(BLUE)
id_y = tube_top - s(35)
c.setDash([], 0)
# Leader from inner wall going left
ldr_end = tube_ol - 15*mm
c.line(tube_il, id_y, ldr_end, id_y)
draw_arrow(tube_il, id_y, 0, ARROW)
c.setFont("Helvetica", DIM_FONT)
c.drawString(ldr_end - c.stringWidth("φ 7 (ID)", "Helvetica", DIM_FONT) - 2, id_y - 3, "φ 7 (ID)")
c.setStrokeColor(black); c.setFillColor(black)

# Hole margin 2mm — already shown in left side view


# ═══════════════════════════════════════
#  LEFT SIDE VIEW of tube
#  Looking from left → tube appears as rectangle (OD × OD)
#  Rod hole φ2.5 drilled through, 2mm from top, centered
# ═══════════════════════════════════════
SV_SCALE = 5.0   # larger scale for clarity
def sv(v): return v * SV_SCALE

SVX = 110   # center X of side view
SVY = 110    # center Y of side view (below front view)

# Labels added after M3 drawing below

# Knurled Tube end face = circle seen from side = rectangle OD × OD
# Actually tube cross-section from left is a circle, but user wants rectangle
# This is the tube WALL face: a square showing OD
sv_left   = SVX - sv(TUBE_OD/2)
sv_right  = SVX + sv(TUBE_OD/2)
sv_bot    = SVY - sv(TUBE_OD/2)
sv_top    = SVY + sv(TUBE_OD/2)

# Outer rectangle — top and sides solid, bottom zigzag (tube continues)
c.setLineWidth(THICK)
# Top line: split around M3 — solid outside, dashed behind screw
m3_sv_left = SVX - sv(SCREW_DIA/2)
m3_sv_right = SVX + sv(SCREW_DIA/2)
c.line(sv_left, sv_top, m3_sv_left, sv_top)      # left of M3
c.line(m3_sv_right, sv_top, sv_right, sv_top)     # right of M3
# Dashed behind M3
c.setLineWidth(MED)
c.setDash([4,2], 0)
c.line(m3_sv_left, sv_top, m3_sv_right, sv_top)
c.setDash([], 0)
c.setLineWidth(THICK)
c.line(sv_left, sv_top, sv_left, sv_bot)      # left
c.line(sv_right, sv_top, sv_right, sv_bot)    # right

# Bottom: zigzag break line (thin)
c.setLineWidth(THIN)
zz_segs = 8
zz_amp = 1.5*mm
zz_step = (sv_right - sv_left) / zz_segs
p = c.beginPath()
p.moveTo(sv_left, sv_bot)
for i in range(zz_segs):
    mid_x = sv_left + zz_step * (i + 0.5)
    end_x = sv_left + zz_step * (i + 1)
    direction = 1 if i % 2 == 0 else -1
    p.lineTo(mid_x, sv_bot + direction * zz_amp)
    p.lineTo(end_x, sv_bot)
c.drawPath(p, fill=0, stroke=1)

# Inner bore — hidden, dashed, same height as outer, narrower width
bore_left  = SVX - sv(TUBE_ID/2)
bore_right = SVX + sv(TUBE_ID/2)
c.setLineWidth(MED)
c.setDash([4,2], 0)
c.line(bore_left, sv_bot, bore_left, sv_top)    # left bore wall, full height
c.line(bore_right, sv_bot, bore_right, sv_top)   # right bore wall, full height
c.setDash([], 0)

# Rod hole: φ2.5, centered horizontally, 2mm from top
hole_cx = SVX
hole_cy = sv_top - sv(HOLE_MARGIN) - sv(ROD_DIA/2)
hole_r  = sv(ROD_DIA/2)

c.setLineWidth(THICK)
c.circle(hole_cx, hole_cy, hole_r, fill=0, stroke=1)

# M3 thread: looking from left, appears as rectangle 3mm wide × 5mm tall
# Sits directly above the rod hole, aligned center
m3_left = SVX - sv(SCREW_DIA/2)
m3_right = SVX + sv(SCREW_DIA/2)
m3_bot = hole_cy + hole_r    # starts at top of rod hole
m3_top = m3_bot + sv(SCREW_LEN)

c.setLineWidth(THICK)
c.rect(m3_left, m3_bot, sv(SCREW_DIA), sv(SCREW_LEN), fill=0, stroke=1)

# Thread lines on M3 (same as front view)
c.setLineWidth(0.15*mm); c.setStrokeColor(GRAY)
pitch = 0.5
for i in range(1, int(SCREW_LEN / pitch)):
    ty = m3_bot + i * sv(pitch)
    if ty < m3_top - sv(0.2):
        c.line(m3_left + 0.5, ty, m3_right - 0.5, ty)
c.setStrokeColor(black)

# M3 label
c.setFont("Helvetica", DIM_FONT); c.setFillColor(BLUE)
c.drawString(m3_right + 3, m3_top + 2, "M3")
c.setFillColor(black)

# M3 height dimension: 5mm
dim_v(m3_bot, m3_top, m3_left, "5", side='left', offset=10*mm)

# Side view labels — positioned above M3
c.setFont("Helvetica-Bold", 11)
c.drawCentredString(SVX, m3_top + 22, "Left Side View")
if HAS_CJK:
    c.setFont("CJK", 9)
    c.drawCentredString(SVX, m3_top + 10, "左视图")

# Center lines — full span of rectangle, proving hole is centered
c.setLineWidth(THIN); c.setStrokeColor(GRAY)
c.setDash([6,2,1,2], 0)
# Vertical center line: full height including M3
c.line(SVX, sv_bot - 3, SVX, m3_top + 3)
# Horizontal center line through hole: full width of rectangle
c.line(sv_left - 3, hole_cy, sv_right + 3, hole_cy)
c.setDash([], 0); c.setStrokeColor(black)

# Dimensions on side view
# Margin: 2mm from top to hole top
c.setLineWidth(THIN); c.setStrokeColor(BLUE); c.setFillColor(BLUE)
c.setDash([], 0)
mg_x = sv_right + 6*mm
c.line(sv_right, sv_top, mg_x + 1.5*mm, sv_top)
c.line(sv_right, hole_cy + hole_r, mg_x + 1.5*mm, hole_cy + hole_r)
# Vertical dim for 2mm margin: top edge to hole top edge
hole_top_y = hole_cy + hole_r
c.line(mg_x, sv_top, mg_x, hole_top_y)
draw_arrow(mg_x, sv_top, 270)
draw_arrow(mg_x, hole_top_y, 90)
c.setFont("Helvetica", DIM_FONT)
c.saveState()
mid_mg = (sv_top + hole_top_y) / 2
c.translate(mg_x, mid_mg); c.rotate(90)
tw = c.stringWidth("2", "Helvetica", DIM_FONT)
c.setFillColor(white); c.rect(-tw/2-2, -4, tw+4, 8, fill=1, stroke=0)
c.setFillColor(BLUE); c.drawCentredString(0, -2.5, "2")
c.restoreState()
c.setStrokeColor(black); c.setFillColor(black)



# ═══════════════════════════════════════
#  PAGE BORDER
# ═══════════════════════════════════════
c.setLineWidth(THIN)
c.setStrokeColor(black)
c.rect(MARGIN, MARGIN, W - 2 * MARGIN, H - 2 * MARGIN, fill=0, stroke=1)

# ═══════════════════════════════════════
#  TITLE BLOCK
# ═══════════════════════════════════════
tb_w = 240; tb_h = 85
tb_x = W - MARGIN - tb_w - 2; tb_y = MARGIN + 2
lc = 52  # label column width
c.setLineWidth(MED)
c.rect(tb_x, tb_y, tb_w, tb_h, fill=0, stroke=1)
c.line(tb_x, tb_y + 68, tb_x + tb_w, tb_y + 68)
c.line(tb_x, tb_y + 51, tb_x + tb_w, tb_y + 51)
c.line(tb_x, tb_y + 34, tb_x + tb_w, tb_y + 34)
c.line(tb_x, tb_y + 17, tb_x + tb_w, tb_y + 17)
c.line(tb_x + lc, tb_y, tb_x + lc, tb_y + tb_h)

# Row labels
lx = tb_x + 3
c.setFont("Helvetica", 7)
c.drawString(lx, tb_y + 73, "Part")
if HAS_CJK:
    c.setFont("CJK", 7)
    c.drawString(lx + 22, tb_y + 73, "零件")

c.setFont("Helvetica", 7)
c.drawString(lx, tb_y + 56, "Tip")
if HAS_CJK:
    c.setFont("CJK", 7)
    c.drawString(lx + 16, tb_y + 56, "笔头")

c.setFont("Helvetica", 7)
c.drawString(lx, tb_y + 39, "Material")
if HAS_CJK:
    c.setFont("CJK", 7)
    c.drawString(lx + 34, tb_y + 39, "材料")

c.setFont("Helvetica", 7)
c.drawString(lx, tb_y + 22, "Note")
if HAS_CJK:
    c.setFont("CJK", 7)
    c.drawString(lx + 22, tb_y + 22, "说明")

c.setFont("Helvetica", 7)
c.drawString(lx, tb_y + 5, "Date")
if HAS_CJK:
    c.setFont("CJK", 7)
    c.drawString(lx + 22, tb_y + 5, "日期")

# Row values
vx = tb_x + lc + 4
c.setFont("Helvetica-Bold", 8)
c.drawString(vx, tb_y + 73, "Stylus Rod")
if HAS_CJK:
    c.setFont("CJK", 7)
    c.drawString(vx + 52, tb_y + 73, "触控笔杆")

c.setFont("Helvetica", 8)
c.drawString(vx, tb_y + 56, "M3 threaded stylus fabric tip (7mm)")
if HAS_CJK:
    c.setFont("CJK", 8)
    c.drawString(vx + 128, tb_y + 56, "M3内牙7.0布头")

c.setFont("Helvetica", 8)
c.drawString(vx, tb_y + 39, "Stainless Steel")
if HAS_CJK:
    c.setFont("CJK", 8)
    c.drawString(vx + 72, tb_y + 39, "不锈钢")

c.setFont("Helvetica", 8)
c.drawString(vx, tb_y + 22, "Knurled for grip")
if HAS_CJK:
    c.setFont("CJK", 8)
    c.drawString(vx + 72, tb_y + 22, "管材滚花便于夹持")

c.setFont("Helvetica", 8)
from datetime import date
c.drawString(vx, tb_y + 5, date.today().isoformat())

c.save()
print(f"PDF saved to {OUTPUT}")