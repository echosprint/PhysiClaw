# PhysiClaw

**What if AI could use your phone — exactly the way you do?**

PhysiClaw gives AI agents a pair of eyes (camera) and a finger (robotic arm) to physically operate any phone. It looks at the screen, decides what to do, and taps — just like you would.

Order food delivery. Check your email. Shop for groceries. Book a hotel. Any app, any phone, iOS or Android.

No OAuth tokens. No ADB cables. No APIs. No app to install. No developer setup.
Just unlock your phone, put it on the desk, and let the agent work.

The tradeoff? PhysiClaw needs hardware: an embedded system running GRBL/grblHAL firmware to control a gantry (X/Y) and stylus (Z), plus two USB cameras. A small desk-sized rig that gives your AI agent physical presence.

## How It Works

```text
 Top Camera ──→ AI Agent ──→ 3-Axis Arm ──→ Side Camera ──→ Aligned?
 (read screen)  (decide)     (move stylus)   (check pos)     │
      ▲                                                  Yes │ No
      │                                                   │  │
      │         Touch Phone ◄─────────────────────────────┘  │
      │              │                                       │
      └──────────────┘ (next action)         adjust & retry ◄┘
```

Two cameras, two perspectives — just like you sitting at a desk:

- **Top camera** looks straight down at the screen, reads what's on it
- **Side camera** looks at the phone from ~45°, the way you'd glance at your desk — it sees where the stylus tip is relative to the screen, helping the AI confirm the stylus is at the right spot before tapping or swiping
- **Stylus** moves on X/Y axes to reach any point on the screen, and pulls up/down (Z axis) to touch or release

The loop is simple: **look → think → move → confirm → touch → repeat**.

### Why PhysiClaw

Today's AI agents can control your computer — but they hit walls everywhere:

- Want to order food? Need a delivery API + OAuth.
- Want to check your bank? Blocked by data walls.
- Want to book a ride? Another service integration.
- Every new skill/service = new OAuth, new API, new setup. Tedious, fragile, limited.

PhysiClaw takes a different approach: **let AI agent physically use your phone.** A camera sees the screen. A robotic finger taps it. No OAuth to apply for. No API to integrate. No app can detect or block it — because to the phone, it's just a finger.

One setup. Every app. Just put an unlocked phone on the desk.

## System Architecture

```text
┌───────────────────────────────────────┐
│           AI Agent (Brain)            │
│  Claude Desktop / OpenClaw / etc.     │
│  Sees screen → decides → calls tools  │
└──────────────────┬────────────────────┘
                   │ MCP Protocol
                   ▼
┌───────────────────────────────────────┐
│     PhysiClaw MCP Server (Python)     │
│                                       │
│  Tools:                               │
│   · screenshot_top   (top camera)     │
│   · screenshot_side  (side camera)    │
│   · move             (X/Y plane)      │
│   · tap / swipe      (Z down + move)  │
│   · park             (retract)        │
└──────────┬────────────────┬───────────┘
           │                │
     USB Cameras      USB Serial (GRBL)
           │                │
           ▼                ▼
    ┌────────────┐   ┌───────────────┐
    │ Top Camera │   │ GRBL Board    │
    │ (screen)   │   │ (embedded)    │
    ├────────────┤   │ X/Y gantry    │
    │ Side Camera│   │ Z stylus      │
    │ (stylus)   │   └──────┬────────┘
    └────────────┘          │ touch
                            ▼
                   ┌─────────────────┐
                   │  Phone          │
                   │  (unlocked)     │
                   └─────────────────┘
```

## Hardware

### Bill of Materials

| Component | Item | Qty | Est. Price |
| --------- | ---- | --- | ---------- |
| **GRBL Arm** | [Paixi Kuaichaobao pen plotter P25](https://e.tb.cn/h.ifgckUqg9Zmph9n?tk=cxpFUxr6Z5C) (X/Y gantry + Z servo) | 1 | ~$80 |
| **Top Camera** | UGREEN 1080P USB camera, fixed focus | 1 | ~$14 |
| **Side Camera** | UGREEN 1080P USB camera (same) | 1 | ~$14 |
| **Stylus** | Capacitive stylus, conductive fiber tip 8-10mm | 1 | ~$1.5 |
| Camera mount | Gooseneck desk clamp, metal, 50cm | 2 | ~$4 |
| Phone mount | Anti-slip pad + L-shaped blocks | 1 set | ~$1.2 |
| USB Hub | USB 3.0 Hub (extend Mac USB ports) | 1 | ~$13 |
| **Total (excluding computer)** | | | **~$127** |

### Camera Setup

- **Top camera:** straight above the screen center, ~25cm distance, reads screen content
- **Side camera:** ~45° angle, like a human sitting at the desk looking at the phone — easily checks where the stylus tip is relative to the screen

### Phone Mounting

- Phone placed face-up flat on the plotter platform
- Anti-slip pad + L-shaped blocks for positioning, ensuring consistent placement

## Communication Protocol (PhysiClaw ↔ GRBL Arm)

### GRBL G-code (USB → GRBL Arm)

All commands used in this project:

```gcode
G91                    # Relative coordinate mode (default)
G0 Xxx Yyy Fxxx        # Rapid move on X/Y plane (position stylus)
G1 Xxx Yyy Fxxx        # Linear move at constant speed (swipe gesture)
M3 S12                 # Stylus down (touch screen)
M3 S0                  # Stylus up (release screen)
M5                     # Servo off
G90                    # Absolute coordinate mode (for park)
G0 X0 Y0 F5000         # Return to home position
$$                     # Query all GRBL parameters
?                      # Query real-time position
```

Protocol: USB serial (CH340, 115200 baud). Send one line at a time, wait for `ok` before next.

### Key GRBL Parameters

| Parameter | Meaning | Typical Value |
| --------- | ------- | ------------- |
| `$100` / `$101` | Steps per mm (X/Y) | 80 |
| `$110` / `$111` | Max speed mm/min (X/Y) | 5000 |
| `$120` / `$121` | Acceleration mm/sec² (X/Y) | 200 |
| `$22` | Enable Homing | 1 |

### MCP Protocol (MCP Client → PhysiClaw)

Tools communicate via stdio or SSE transporting JSON messages. MCP is a standard inter-process communication protocol, language-agnostic.

## Tech Stack

### Language

Python 3.11+

### Dependencies

```bash
pip install pyserial opencv-python mcp
```

- pyserial: Serial port G-code for stepper motors and servo
- opencv-python: USB camera capture
- mcp: MCP Server framework

No Anthropic SDK needed (Claude runs on the MCP client side, not inside PhysiClaw).

### Platform Compatibility

Mac / Windows / Linux (Raspberry Pi) all supported. The only platform difference is serial device names (Mac: `/dev/tty.usbserial-xxx`, Windows: `COM3`, Linux: `/dev/ttyUSB0`).

## Code Structure

```text
physiclaw/
├── physiclaw_server.py   # MCP Server entry point, exposes tools
├── hand.py               # Serial G-code control for motors and servo
├── eyes.py               # Two USB cameras for photo capture
└── config.py             # Serial port, servo angles, distance mapping constants
```

## Operation

The AI agent does not output coordinates — only direction and distance level. Each step is verified by photo.

**Directions:** up / down / left / right / up-left / up-right / down-left / down-right

**Distance Levels:**

| Level | Think of it as... | Physical Displacement |
| ----- | ----------------- | --------------------- |
| large | half the screen away | 20mm |
| medium | a few icons away | 8mm |
| small | one icon away | 3mm |
| nudge | almost there, fine-tune | 1mm |

**Full Operation Cycle:**

```text
1. park()              → Retract stylus out of frame
2. screenshot_top()    → Clean screenshot, AI sees screen content
3. AI decides          → e.g. "move down-right, large"
4. move(dir, dist)     → Stylus moves toward target
5. screenshot_side()   → AI checks stylus position
6. Aligned?
   → No:  back to step 3 (AI re-evaluates and adjusts)
   → Yes: tap()  → Stylus touches screen
7. park()              → Retract stylus out of frame
8. screenshot_top()    → Verify result, continue next action
```

### Gesture Implementation

**Single Tap:** G0 to target → pen down → hold 50-100ms → pen up

**Long Press:** G0 to target → pen down → hold 800ms → pen up

**Swipe:** G0 to start → pen down → G4 P0.03 → G1 to end F3000 → G4 P0.03 → pen up

**Double Tap:** pen down 50ms → pen up → wait 100ms → pen down 50ms → pen up (interval < 300ms)

### Why Two Cameras

- Top-down camera: views screen content; captures clean screenshots when pen is retracted
- Side camera: views stylus tip position relative to screen; from the side you can see both where the tip is on screen and how high it is above the surface

## Use Cases

| Scenario | Status |
| -------- | ------ |
| Order food delivery (Meituan, Uber Eats) | Yes (enable password-free or give agent the password for full autonomy) |
| Hail a ride (Didi, Uber) | Yes (same as above) |
| Browse and shop (Taobao, Amazon) | Yes (same as above) |
| Check weather / news / stocks | Fully capable |
| Read and reply to messages (WeChat, WhatsApp) | Yes |
| Scroll social media (TikTok, Instagram) | Yes |
| App daily check-in / collect rewards | Fully capable |
| Set alarm / timer / reminder | Yes |
| Take a screenshot and send it | Yes |

## Security Warning

PhysiClaw has **full physical control** of your phone — it can see and tap anything on screen. Even without your passwords, it could open your password manager, read saved credentials, receive OTP codes to reset passwords, or access any app that's already logged in. If a malicious actor compromises your agent, they have the same access.

**Treat it like handing your unlocked phone to a stranger.**

- **Use a dedicated backup phone** — never your primary device
- **Separate phone number** — not linked to your main accounts
- **Fresh accounts** — don't log into your real accounts on it
- **Different passwords** — never reuse credentials from your primary phone
- **Limited funds** — only load a small amount of money, enough for the task
- **No password manager** — don't install one; only store what the agent needs

## License

MIT
