# PhysiClaw

**What if AI could use your phone — exactly the way you do?**

PhysiClaw gives AI agents a pair of eyes (camera) and a finger (robotic arm) to physically operate any phone. It looks at the screen, decides what to do, and taps — just like you would.

Order food delivery. Check your email. Shop for groceries. Book a hotel. Any app, any phone, iOS or Android.

No OAuth tokens. No ADB cables. No APIs. No app to install. No developer setup.
Just unlock your phone, put it on the desk, and let the agent work.

## Overview

PhysiClaw is an MCP Server that exposes hardware operation tools (screenshot, move, tap, park, side screenshot) for any MCP client (Claude Desktop, OpenClaw, Claude Code) to call. It only handles hardware operations internally — no AI decisions. All AI decisions are made by the upstream MCP client (Claude).

### Design Philosophy

- The phone is a black box: no cables, no apps installed, no debug mode
- PhysiClaw is hands and eyes, not a brain: it only takes photos and moves motors; Claude sees and decides
- Works on both iOS and Android: pure physical touch, no software layer involved

### Key References

- **[See-Control](https://arxiv.org/abs/2512.08629)** (Zhao et al., 2025): *See-Control: A Multimodal Agent Framework for Smartphone Interaction with a Robotic Arm*. Proposes the Embodied Smartphone Operation task — using multimodal AI to generate robotic arm control commands for purely physical phone operation without ADB. Academically validates the feasibility of MLLM + robotic arm approach. Includes a 155-task benchmark.
- **TapsterBot**: Open-source Delta robot designed specifically for phone touchscreen testing
- **GRBL**: Open-source CNC motion control firmware, industry standard protocol for pen plotters

## System Architecture

```
┌─────────────────────────────────────────────┐
│  MCP Client (Claude Desktop / OpenClaw)      │
│  Role: View photos, understand screen,       │
│        plan tasks, call PhysiClaw MCP tools   │
└─────────────────┬───────────────────────────┘
                  │ MCP Protocol (stdio or SSE)
                  ▼
┌─────────────────────────────────────────────┐
│  PhysiClaw MCP Server (Python)               │
│  Tools: screenshot_top / screenshot_side     │
│         move / tap / park                    │
│  Modules: hand.py / eyes.py / config.py      │
└──────┬──────────────────┬───────────────────┘
       │ USB Camera        │ USB Serial (GRBL G-code)
       ▼                  ▼
  Two Cameras         Pen Plotter Machine
  (top-down + side)   (XY gantry + servo + stylus)
                          │ Physical Touch
                          ▼
                     Phone (black box)
```

### Layer Responsibilities

```
┌─────────────────────────────────────────┐
│  AI (Brain)                              │
│  · Understand screen content             │
│  · Decide next action                    │
│  · Read coordinates from grid / output   │
│    direction and distance                │
│  · Verify operation results              │
├─────────────────────────────────────────┤
│  OpenCV (Paintbrush, optional)           │
│  · Overlay coordinate grid on photos     │
│  · Track stylus tip marker               │
├─────────────────────────────────────────┤
│  Calibration Formula (Translator)        │
│  · Screen coordinates → GRBL coordinates │
│  · One multiply-add, zero latency        │
├─────────────────────────────────────────┤
│  GRBL (Muscle)                           │
│  · Receive G-code                        │
│  · Drive stepper motors and servo        │
└─────────────────────────────────────────┘
```

## Hardware

### Bill of Materials

| Purpose | Item | Qty | Est. Price |
|---------|------|-----|------------|
| Mechanical platform | Paixi Kuaichaobao pen plotter P25 | 1 | Purchased |
| Top-down camera | USB industrial camera 2MP fixed-focus low-distortion UVC | 1 | ~$14 |
| Side camera | USB industrial camera (same as above) | 1 | ~$14 |
| Anti-glare | CPL polarizing filter matching lens diameter | 2 | ~$3 |
| Camera mount | Gooseneck desk clamp, metal, 50cm | 2 | ~$4 |
| Stylus | Capacitive stylus, conductive fiber tip 8-10mm | 2 | ~$1.5 |
| Tip marker | Red silicone ring, 8mm inner diameter | Several | ~$0.5 |
| Grounding | Copper foil conductive tape, 6mm wide | 1 roll | ~$0.7 |
| Grounding cable | Alligator clip DuPont wire | 2 | ~$0.5 |
| Tip buffer | Compression spring, 0.3mm wire, 10mm length | Several | ~$0.5 |
| Phone mount | Anti-slip pad | 1 | ~$0.5 |
| Phone alignment | L-shaped blocks | 2 | ~$0.7 |
| USB Hub | USB 2.0 Hub | 1 | ~$4 |
| **Total (excluding plotter and computer)** | | | **~$44** |

Backup (if plotter serial port doesn't work): Raspberry Pi 5 kit + A4988 driver boards ×2

### Pen Plotter Machine

- Manufactured by Paixi Technology, gantry-style XY structure, stepper motor driven
- Host software is "Kuixiang Engraving" (Chengdu Kuixiang Tech), based on standard GRBL protocol
- Serial chip CH340, baud rate 115200
- Servo controls pen up/down via M3 Sxx (S0 = pen up, S12 = pen down, exact values tuned after arrival)
- First thing after arrival: connect USB to computer, send `$$` to test GRBL communication

### Stylus Modification

- Replace the gel pen in the plotter's pen holder with a capacitive stylus
- Wrap copper foil conductive tape on the stylus body, run alligator clip wire to GRBL board GND
- Attach a red silicone ring 1cm above the tip (visual marker for the side camera)
- Add a compression spring between stylus and holder as buffer to prevent excessive pen-down pressure
- **Tilted mounting (30-45°) optional:** prevents stylus body from blocking the target in top-down camera view

### Camera Setup

- Top-down camera: gooseneck extends from behind the desk, pointing down at phone screen, 20-25cm distance
- Side camera: shoots from the side at 4-5° low angle, to see stylus tip position relative to the screen
- Both cameras fitted with CPL polarizing filters to eliminate screen reflections
- Camera-to-phone relative position must stay fixed once set
- **Resolution:** 1080p recommended
- **Focus:** Manual focus, slightly defocused to suppress moiré patterns
- **Exposure:** Set manually to integer multiples of screen refresh period (60Hz → 16.7ms or 33.3ms)

### Phone Mounting

- Phone placed face-up flat on the plotter platform
- Anti-slip pad + L-shaped blocks for positioning, ensuring consistent placement
- Screen brightness set to maximum (reduces PWM dimming flicker)

## Communication Protocol

### GRBL G-code (Serial → Pen Plotter)

All commands used in this project:

```
G90                    # Absolute coordinate mode
G91                    # Relative coordinate mode (primary mode)
G0 Xxx Yyy Fxxx        # Rapid positioning
G1 Xxx Yyy Fxxx        # Linear move at constant speed (for screen swiping)
G4 Pxxx                # Dwell (seconds)
M3 Sxx                 # Servo pen down (S12 = down, S0 = up)
M5                     # Servo off
$$                     # Query all parameters
?                      # Query real-time status
```

Communication rule: append `\n` to each command, wait for GRBL to reply `ok` before sending the next. `?` queries position, `!` pauses, `~` resumes.

### Key GRBL Parameters

| Parameter | Meaning | Typical Value |
|-----------|---------|---------------|
| `$100` / `$101` | Steps per mm (X/Y) | 80 |
| `$110` / `$111` | Max speed mm/min (X/Y) | 5000 |
| `$120` / `$121` | Acceleration mm/sec² (X/Y) | 200 |
| `$22` | Enable Homing | 1 |

### MCP Protocol (MCP Client → PhysiClaw)

Tools communicate via stdio or SSE transporting JSON messages. MCP is a standard inter-process communication protocol, language-agnostic.

## Calibration System (PenCal)

### Phone Position Detection

Take one photo with screen on, one with screen off. Frame differencing finds the illuminated region = screen boundary. Output: pixel positions of screen corners in camera frame.

### Calibration Grid Photo (CalGrid)

1. Open PenCal webpage's CalGrid mode on the phone (fullscreen)
2. The webpage draws a labeled coordinate grid with absolute screen pixel coordinates
3. Camera captures and saves as the coordinate layer photo
4. Webpage must be fullscreen (Fullscreen API) to avoid browser UI taking up space

### Touch Probe Calibration (TapProbe)

1. Switch phone to PenCal's TapProbe mode
2. Webpage listens for touch events, reports absolute screen coordinates via `screenX` / `screenY`
3. Machine moves to 5 GRBL coordinates (4 corners + center), pen touches screen at each
4. Record 5 data pairs: `(GRBL_X, GRBL_Y) ↔ (Screen_X, Screen_Y)`

**Affine transform formula (system core):**

```
GRBL_X = a × Screen_X + b × Screen_Y + c
GRBL_Y = d × Screen_X + e × Screen_Y + f
```

- 6 parameters, 5 points yield 10 equations, least squares solution
- Automatically accounts for: coordinate offset, axis flipping, phone rotation, stylus tilt offset
- Use the 5th point for verification, error should be < 0.5mm

### Calibration Verification

After calibration, tap a new screen position and check the deviation between actual touch point and expected position. Pass if deviation < 1mm.

### Online Drift Compensation

During normal operation, each touch event yields a new data pair. Accumulated data enables sliding average correction to compensate for long-term drift.

## Tech Stack

### Language

Python 3.11+

### Dependencies

```
pip install pyserial opencv-python mcp
```

- pyserial: Serial port G-code for stepper motors and servo
- opencv-python: USB camera capture
- mcp: MCP Server framework

No Anthropic SDK needed (Claude runs on the MCP client side, not inside PhysiClaw).

### Platform Compatibility

Mac / Windows / Linux (Raspberry Pi) all supported. The only platform difference is serial device names (Mac: `/dev/tty.usbserial-xxx`, Windows: `COM3`, Linux: `/dev/ttyUSB0`).

## Code Structure

```
physiclaw/
├── physiclaw_server.py   # MCP Server entry point, exposes tools
├── hand.py               # Serial G-code control for motors and servo
├── eyes.py               # Two USB cameras for photo capture
├── config.py             # Serial port, servo angles, distance mapping constants
├── brain.py              # AI API wrapper (standalone mode)
└── main.py               # Standalone operation main loop
calibration/
├── pencal.html           # PenCal calibration webpage (CalGrid + TapProbe)
├── phone_detect.py       # Phone position detection (on/off screen differencing)
├── grid_capture.py       # Calibration grid photo capture
├── tap_probe.py          # 5-point touch probe calibration + formula fitting
└── verify.py             # Calibration verification
data/
├── grid_layer.jpg        # Calibration grid photo (coordinate layer)
└── calibration.json      # Calibration parameters (affine transform coefficients a-f)
```

## Operation Modes

### Mode 1: Visual Servoing (MCP Server Mode)

The LLM does not output coordinates — only direction and distance level. Each step is verified by photo.

**Directions:** up / down / left / right

**Distance Levels:**

| Level | Meaning | Physical Displacement |
|-------|---------|----------------------|
| far | Most of the screen | 20mm |
| medium | A few icons | 8mm |
| near | One icon | 3mm |
| tiny | Almost there | 1mm |

**Full Operation Cycle:**

```
1. park()              → Retract pen out of frame
2. screenshot_top()    → Take clean screenshot, Claude sees screen content
3. Claude decides      → Outputs move command
4. move(dir, dist)     → Pen moves toward target
5. screenshot_side()   → Photo with red tip marker visible
6. Claude judges       → Aligned → tap; not aligned → continue move to adjust
7. tap()               → Pen down for 80ms
8. park()              → Retract pen out of frame
9. screenshot_top()    → Photo to verify operation result
10. Repeat until task complete
```

### Mode 2: Direct Coordinate (Post-Calibration Mode)

AI reads coordinates from grid-overlaid photo, converts to GRBL coordinates via calibration formula, one-shot positioning.

**Available AI Actions:**

| Action | Description | Parameters |
|--------|-------------|------------|
| `tap` | Single tap | x, y |
| `long_press` | Long press | x, y, duration_ms |
| `swipe` | Swipe | x1, y1, x2, y2 |
| `double_tap` | Double tap | x, y |
| `type_text` | Type text | text |
| `wait` | Wait | seconds |
| `task_complete` | Task complete | — |
| `task_failed` | Task failed | reason |

**Operation Flow:**

```
1. Capture photo + overlay coordinate grid
2. AI views image, reads coordinates
3. Calibration formula: Screen coords → GRBL coords
4. Execute action (no visual servoing needed)
5. Capture photo to verify result
6. Repeat until task complete
```

### Gesture Implementation

**Single Tap:** G0 to target → pen down → hold 50-100ms → pen up

**Long Press:** G0 to target → pen down → hold 800ms → pen up

**Swipe:** G0 to start → pen down → G4 P0.03 → G1 to end F3000 → G4 P0.03 → pen up

**Double Tap:** pen down 50ms → pen up → wait 100ms → pen down 50ms → pen up (interval < 300ms)

### Why Two Cameras

- Top-down camera: views screen content; captures clean screenshots when pen is retracted
- Side camera: views stylus tip position relative to screen; from the side you can see both where the tip is on screen and how high it is above the surface

## MCP Client Configuration

### Claude Desktop (claude_desktop_config.json)

```json
{
  "mcpServers": {
    "physiclaw": {
      "command": "python",
      "args": ["path/to/physiclaw_server.py"]
    }
  }
}
```

### Remote Deployment (Raspberry Pi)

Change physiclaw_server.py to SSE transport:
```python
mcp.run(transport="sse", host="0.0.0.0", port=8080)
```

Client configuration:
```json
{
  "mcpServers": {
    "physiclaw": {
      "type": "url",
      "url": "http://raspberrypi.local:8080/sse"
    }
  }
}
```

## Image Processing Notes

| Issue | Solution |
|-------|----------|
| Moiré patterns | Slightly defocus camera, or adjust shooting distance/angle |
| Refresh rate flicker | Set exposure to integer multiple of refresh period (60Hz → 16.7ms), max brightness |
| Reflections | Avoid direct light sources, add light shield or CPL polarizing filter |
| Color deviation | Minimal impact on AI; OpenCV stylus tracking needs HSV threshold tuning |

## Deployment & Debugging Steps

### Phase 1: GRBL Communication Verification

1. Pen plotter arrives, connect USB to computer
2. Open serial terminal, baud rate 115200
3. Send `$$`, check if GRBL parameters are returned
4. Send `G91` then `G0 X50 F5000`, check if motor moves
5. Send `M3 S12` and `M3 S0`, check if servo raises/lowers pen

If serial doesn't work: open the machine to check the main board chip, consider Raspberry Pi + A4988 alternative.

### Phase 2: Touch Verification

1. Install capacitive stylus + grounding wire
2. Mount phone on platform
3. Manually send G-code to move pen to a screen position
4. Send servo pen-down command, observe if phone registers touch
5. Test at different positions, confirm full-screen touch coverage

### Phase 3: Vision Verification

1. Install two cameras + polarizing filters
2. Test camera capture with Python
3. Confirm image clarity, no moiré, no reflections
4. Verify top-down camera captures full phone screen
5. Verify side camera can see the red stylus tip marker

### Phase 4: Calibration

1. PenCal webpage fullscreen displaying CalGrid
2. Camera captures and saves coordinate layer
3. TapProbe 5-point calibration, fit affine transform formula
4. Verify accuracy < 1mm

### Phase 5: Integration Testing

1. Run physiclaw_server.py
2. Configure MCP Server in Claude Desktop
3. Simple task: "Open Settings"
4. Medium task: "Open WeChat, go to Moments"
5. Complex task: "Open Meituan, order food delivery"

## Use Cases

| Scenario | Can Complete Independently? |
|----------|---------------------------|
| Food delivery ordering | Yes, up to payment |
| Reorder / order history | Yes, up to payment |
| Ride-hailing (saved addresses) | Yes, up to payment |
| Utility bill payment | Yes, up to payment |
| Check delivery / weather / stocks | Fully capable |
| Set alarm | Yes |
| App daily check-in | Fully capable |
| Search & shop (requires typing) | Difficult |

Payment steps require human intervention (password / fingerprint / Face ID). PhysiClaw handles all the preparation, then prompts you to take over at payment.

## Future Directions

- **Bluetooth keyboard input:** Raspberry Pi emulates BLE HID keyboard, bypassing on-screen keyboard key-by-key tapping
- **Keyboard coordinate pre-calibration:** Calibrate virtual keyboard key positions on first use, skip visual servoing when typing
- **Local vision model:** Run lightweight VLM (e.g., Qwen2-VL-2B) on Raspberry Pi to reduce latency and cost
- **OpenCV-assisted alignment:** cv2.inRange detects red stylus tip marker, fine-tuning done with pure CV

## Phone Screen Size Reference

| Model | Resolution (px) | Physical Size (mm) |
|-------|-----------------|---------------------|
| iPhone 15 | 1179 × 2556 | 71.6 × 147.5 |
| iPhone 15 Pro | 1179 × 2556 | 70.6 × 146.6 |
| Redmi K70 | 1220 × 2712 | 73.2 × 161.3 |
| Generic 6.1" | 1080 × 2400 | 68 × 144 |
| Generic 6.7" | 1080 × 2400 | 74 × 162 |

## Brand

- Project name: PhysiClaw (Physical + Claw)
- Domain: physiclaw.io
