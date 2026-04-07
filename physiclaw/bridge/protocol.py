"""LAN bridge page protocol — constants describing what the phone page renders.

These are page-level protocol constants: where to draw the AssistiveTouch
positioning circle, and the layout of the color nonce barcode used to
verify screenshot uploads.

Both the bridge (which tells the page what to display) and the hardware
screenshot pipeline (which verifies the result) read from this module.
This file is the single source of truth so neither side drifts.
"""

# AssistiveTouch button position (CSS viewport pixels, iPhone left edge snap)
AT_CSS_X = 38       # 10pt edge margin + 28pt button radius
AT_CSS_Y = 200      # hardcoded vertical position
AT_RADIUS = 28      # matches AT button (56pt diameter)

# Color nonce barcode — page renders NONCE_COUNT squares of NONCE_SQUARE_SIZE
# CSS pixels each, starting at (NONCE_CSS_X, NONCE_CSS_Y).
NONCE_CSS_X = 180
NONCE_CSS_Y = 300
NONCE_COUNT = 20
NONCE_SQUARE_SIZE = 15  # CSS pixels per square
