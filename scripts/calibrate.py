"""
Run PhysiClaw stylus calibration.

Open /pen-calib on the phone, position the stylus just above the center
orange circle, then run:

    uv run python scripts/calibrate.py

Note: On macOS, OpenCV won't trigger the camera permission dialog.
If the camera returns blank frames, run `imagesnap` once first to
grant camera access to your terminal app, then re-run this script.
"""

from physiclaw import PhysiClaw

PhysiClaw().calibrate()
