# PhysiClaw System Architecture Plan

## Design Philosophy

1. **Don't mimic humans.** AI should search, not scroll. AI should paste, not type. Every interaction path should use the most efficient method available to a machine, not replicate how a finger and eye would do it. Humans scroll app pages to browse and discover — AI agents should search the web first to gather information, decide what to get, then enter the app with a clear target and execute in the fewest possible actions. Scrolling through app pages to gather information is the least efficient path for an agent.

2. **Three channels, each doing what it's best at.** Physical touch for precise tapping. LAN bridge for text/data transfer. Camera for visual confirmation.

3. **Zero phone intrusion.** No USB debug mode, no installed apps, no Bluetooth pairing, no special permissions. The phone remains completely stock.

4. **One skill, one task.** Each skill handles one type of daily errand — ordering food delivery, buying groceries, topping up phone credit, booking train tickets, ordering milk tea, making a hospital appointment, checking and replying to messages in WeChat or WhatsApp. Create the skill once, run it automatically for a long time.

5. **Talk like a person, not an API.** The agent has a phone. The user has a phone. They communicate through messages, just like two people would. Need a verification code? Ask the user. Need a password? Ask the user. Job done? Tell the user. Something went wrong? Explain what happened. No OAuth, no API tokens, no technical handshakes — just conversation between two phones.

---

## System Architecture

```
                    ┌─────────────────────────────┐
                    │   Agent (Mac/PC/Raspberry Pi) │
                    │                               │
                    │   Claude ←→ MCP Server         │
                    │       │         │              │
                    │   Skill Engine  CV Pipeline     │
                    │       │         │              │
                    │   Web Server (Bridge)           │
                    └───┬──────┬──────┬──────────────┘
                        │      │      │
            ┌───────────┘      │      └───────────┐
            ▼                  ▼                  ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ Physical      │  │ LAN Bridge   │  │ Camera       │
    │ Channel       │  │ Channel      │  │ Channel      │
    │               │  │              │  │              │
    │ GRBL → Stylus │  │ WebSocket    │  │ USB Camera   │
    │ • tap         │  │ • text→clip  │  │ • screen     │
    │ • swipe       │  │ • device info│  │   state      │
    │ • press combo │  │ • touch coord│  │ • fallback   │
    │               │  │   (calibrate)│  │   sensing    │
    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
           │                 │                 │
           └────────► Phone ◄─────────────────┘
                    (stock, unmodified)
```

### Channel Responsibilities

| Channel | Direction | Skill Building | Skill Runtime |
|---------|-----------|---------------|---------------|
| Physical (GRBL+stylus) | Agent→Phone | Tap AssistiveTouch for screenshots, tap calibration points | Tap, swipe, long-press, paste |
| Bridge (WebSocket) | Agent↔Phone | Receive screenshots, touch coords, device info, clipboard write | Clipboard write only |
| Camera | Phone→Agent | Not primary (screenshots are better) | **Primary sensing** — compare against reference screenshots |

### Bridge Detail

**Prerequisite:** Phone and agent computer must be on the same WiFi / LAN.

**Server discovery:** Agent detects its own LAN IP at startup, generates a QR code containing the full URL (e.g., `http://192.168.1.100:8048/message`), and displays it on the computer screen. User opens phone camera → points at QR code → iOS auto-detects → tap notification → browser opens the page. No manual IP typing.

The agent serves two web pages. Both use WebSocket for real-time communication.

**Page 1: `/calibrate` — Calibration page**

Used only during calibration. Displays visual targets (orange circle, UP/RIGHT markers, 15-dot grid, validation dots). Reports touch coordinates on every tap. Also auto-sends device info on connect (screen size, safe area, pixel ratio, iOS version).

**Page 2: `/message` — Message page**

Used during skill building and runtime. Does one thing: display text, large and centered. Tap anywhere to copy that text to clipboard.

```
Agent → Phone:
  Send text: {type:"text", content:"美团外卖"} → page shows it
  
Phone → Agent:
  Stylus taps page → text enters clipboard → {type:"clipboard_ready"}
  Screenshot uploads via Shortcuts HTTP POST (skill building only)
```

**Opening Apps via Spotlight:**
```
Agent sends "美团外卖" → page displays it → stylus taps page → clipboard ready
→ stylus pulls down (Spotlight) → long press → paste → tap first result → App opens
```

**Auto-reconnect:** WebSocket drops when user switches to another app. Page reconnects when brought back to foreground.

### Two Modes: Build vs Runtime

**Skill Building Mode (AssistiveTouch ON):**

AssistiveTouch enabled, configured: single tap = run Shortcut (`Take Screenshot` → `HTTP POST to agent`). Agent taps AssistiveTouch to capture pixel-perfect screenshots. CV runs on clean images with near-perfect accuracy. All reference data extracted here.

```
User: "I'm on the menu page"
Agent: → taps AssistiveTouch → receives clean screenshot → analyzes
       "I see 4 food cards with '+' buttons. Correct?"
```

User only navigates and confirms. Agent handles all capture and analysis.

**Skill Runtime Mode (AssistiveTouch OFF):**

AssistiveTouch disabled — no floating button on screen, no screenshots, nothing for apps to detect. Camera is the only sensing channel. But camera doesn't need to "understand" the screen from scratch — it compares against the reference screenshots captured during building.

```
Camera captures frame → feature match against reference screenshot →
confirm correct screen → use pre-stored coordinates → stylus taps
```

The reference screenshots make camera-based sensing viable: the agent already knows what the page looks like, where every element is, and what text is on screen. Camera just needs to confirm "yes, this matches" or "no, something changed."

---

## Calibration

Seven steps. User places phone on the surface and does one manual action to start.

**Preparation:** User manually positions stylus over phone screen center.
**Step 0:** Find the phone's Z height (screen surface).
**Step 1:** Check phone-arm alignment.
**Steps 2–3:** Ensure camera sees the phone correctly.
**Mapping A (Step 4): GRBL ↔ Screen** — so the stylus can tap the right screen coordinate.
**Mapping B (Step 5): Camera pixels ↔ Screen pixels** — so the agent can convert camera observations to screen coordinates.
**Step 6:** End-to-end validation.

### Preparation: Phone Placement + Device Info + Stylus Positioning

Three things before calibration steps begin.

**A. Phone placement.**

Place phone face-up on the platform (roughly A4-sized). Align phone's long edge with platform's long edge, short edge with short edge. Pad under the phone to level the camera bump so it doesn't wobble.

**B. Device info collection (automatic, instant).**

User opens the bridge webpage in Safari or Chrome. The page auto-collects and sends to agent via WebSocket:

```
Screen:  1170 × 2532 px (physical), 390 × 844 pt (CSS), 3x pixel ratio
Safe area: top=59pt (Dynamic Island), bottom=34pt (Home Indicator)
Usable area: 390 × 751 pt (between safe areas)
Device: iPhone 14 Pro, iOS 17.4
Browser: Safari/Chrome, WebSocket ✓, Clipboard Write ✓
```

Agent now knows the exact screen dimensions. This determines:
- Step 1: how far apart the two alignment taps should be
- Step 2: how far to expand from center to cover the full screen
- Step 5: where to place the 15 calibration dots (within safe area)
- Skill system: all coordinate conversions between CSS points and physical pixels

**C. Manual stylus positioning (~10 seconds).**

Calibration page displays a large orange circle at screen center as a visual target. Agent: "Move the stylus tip above the orange circle, about 3mm above the screen surface."



### Step 0: Z-Axis Surface Detection

Calibration page in foreground, listening for touch events. Stylus is above the phone center (from Preparation step). Descends in small steps.

```
Z = 0 (init position)

loop:
    Z += 0.25mm           # descend one small step (Z+ is downward)
    wait 100ms            # hold still, check for touch
    if touch detected:
        Z_surface = Z     # screen surface found
        Z = 0             # return to init position
        done ✓
    if total descent > 5mm:
        Z = 0             # return to init position
        error: "No screen detected. Check phone placement."
```

Stationary detection at each step — more reliable than continuous descent. Stylus is still when the touchscreen senses it. 0.25mm resolution, 100ms per step, worst case ~6 seconds.

After detection, the working tap height is defined:

```
Z_tap = Z_surface + 0.5mm  (spring compresses, ensures contact)
```

All taps in Steps 1, 2, and 6: arm moves to target (x,y) at Z=0, descends to Z_tap, holds 50ms, returns to Z=0.

**Safety:** Spring-loaded stylus absorbs overtravel. Step motor cannot exert enough force to damage the screen. The 5mm descent limit prevents wasting time if the phone isn't there.

### Step 1: Arm–Phone Alignment Check (2 taps)

Calibration page in foreground, listening for touch events via WebSocket.

Arm moves along its X axis only, taps two points on the screen.

```
Tap A at GRBL (x1, y_fixed) → phone reports (sx1, sy1)
Tap B at GRBL (x2, y_fixed) → phone reports (sx2, sy2)

tilt_ratio = |sy1 - sy2| / |sx1 - sx2|

If tilt_ratio < 0.02 (~1°):
  → Phone X axis parallel to Arm X axis ✓

If tilt_ratio > 0.02:
  → Phone is tilted by atan(tilt_ratio)
  → Agent: "Phone is tilted ~2°. Please adjust."
  → Repeat until aligned
```

### Step 2: Camera Physical Rotation Check

Camera outputs a rectangular image (e.g., 1920×1080). The image's long dimension (1920) should correspond to the phone's long dimension. This maximizes screen coverage in the frame.

```
Camera captures one frame.
Detect the phone region in the frame (bright rectangle on dark background,
or use the blue markers from the calibration page).

Phone region in image:
  appears taller than wide → phone long axis = image long axis ✓
  appears wider than tall  → camera is rotated 90° relative to phone
    → Agent: "Please rotate the camera 90°."
    → Recheck
```

Camera is adjusted, not the phone — phone is already aligned with the arm.

### Step 3: Image Software Rotation

Calibration page displays two blue markers: **"UP"** near screen top, **"RIGHT"** near screen right. Camera captures one frame.

The camera might be mounted in any of 4 orientations even after Step 3 (upside down, mirrored). Agent finds UP and RIGHT in the image and determines which software rotation (0°, 90°, 180°, 270°) to apply so that UP appears at the top of the final processed image.

```
Find UP and RIGHT marker positions in raw camera image.
Determine rotation needed:
  UP above RIGHT     → 0° rotation
  UP to left of RIGHT  → 90° CW rotation
  UP below RIGHT     → 180° rotation
  UP to right of RIGHT → 270° CW rotation

Store this rotation. Apply it to every camera frame before processing.
```

**This is a software setting, not a physical adjustment.** No user action needed. From now on, every camera image is auto-rotated so that phone UP = image top.

### Step 4: GRBL ↔ Screen Mapping (10+ distributed taps)

Phone aligned with arm. Starting from the center position (known from Step 0's successful touch), arm expands outward. Agent uses screen dimensions from Preparation step to calculate how far to expand in each direction.

```
Screen is 390 × 844 pt (from device info).
Center touch from Step 0 landed at screen coordinate (sx0, sy0).
Expansion distances calculated from screen size:
  half_width  ≈ 390/2 × 0.8 = 156pt outward left/right
  half_height ≈ 844/2 × 0.8 = 338pt outward up/down
  (0.8 factor keeps points within screen, avoiding bezel misses)

Sampling pattern (from center outward):
  - Center (already known from Step 0)
  - 4 cardinal directions (up/down/left/right)
  - 4 diagonal corners
  - 1-2 additional points for redundancy

Each tap: move to (x,y) at Z=0 → descend to Z_tap → hold 50ms → return to Z=0
Phone reports (sx, sy) for each.

All 10 taps must land on screen successfully. If any tap misses, retry at adjusted position.
```

Compute affine transform from all pairs (RANSAC to reject outliers). Verify: 3 additional taps at new positions, predicted vs actual error < 2px.

**Why distributed, not random:** Random sampling may cluster, leaving screen edges poorly constrained. The affine transform is most accurate when calibration points span the full working area.

**Result: Mapping A is done.** Agent can now convert any screen coordinate to a GRBL position and tap it.

### Step 5: Camera Pixel ↔ Screen Pixel Mapping (15 points)

Calibration page displays 15 clearly visible dots at known screen coordinates. Dot positions are a 5×3 grid calculated from device info, placed within the usable screen area (between safe area insets). Camera captures one frame (auto-rotated per Step 3).

```
For each dot:
  Screen coordinate: known (bridge webpage placed it there)
  Camera pixel coordinate: detected by vision (color segmentation → blob center)

15 pairs → compute homography matrix (camera pixels → screen pixels).
Verify with a few extra dots.
```

**Result: Mapping B is done.** Agent can now see any point in the camera image and convert it to a screen coordinate. Combined with Mapping A, it can then convert to GRBL and tap it.

### Two Mappings, Complete Chain

```
Camera pixel → (Mapping B) → Screen coordinate → (Mapping A) → GRBL position → Tap
```

Or directly from a skill's annotated screenshot:
```
Screenshot coordinate = Screen coordinate → (Mapping A) → GRBL position → Tap
```

### Step 6: Full-Chain Validation

Tests the entire pipeline end-to-end: camera sees target → Mapping B → screen coordinate → Mapping A → GRBL → physical tap → phone confirms.

```
Repeat 3 times:
  1. Calibration page shows an orange dot at a random screen coordinate
  2. Camera captures frame → detect orange dot (trivial: high-saturation blob)
  3. Mapping B: orange dot camera pixel → screen coordinate
  4. Mapping A: screen coordinate → GRBL position
  5. Arm moves there → taps
  6. Phone reports touch coordinate via WebSocket
  7. Compare: touch coordinate vs orange dot's actual screen coordinate

All 3 within 5px → calibration passed ✓
Any miss → identify which mapping has error, re-run that step
```

One tap validates everything — both mappings, their consistency, and the physical tap accuracy. No ambiguous hovering or stylus detection needed.

### Post-Calibration: Set Screen Center as GRBL Origin

After all calibration steps pass, agent uses Mapping A to compute the GRBL position corresponding to the phone's screen center (screen_width/2, screen_height/2). Arm moves there and GRBL sets this as its work origin (G92 X0 Y0).

From this point on, all GRBL coordinates are relative to the screen center. This makes coordinates intuitive (0,0 = center, negative = left/up, positive = right/down) and consistent across different phone placements.

### Total Calibration Time: ~80 seconds

| Step | Time | User action |
|------|------|-------------|
| Prep. Position stylus over screen | ~10s | **Move stylus to roughly screen center** |
| 0. Z-axis surface detection | ~2–6s | None |
| 1. Arm-phone alignment | ~3s | Maybe adjust phone once |
| 2. Camera rotation check | ~3s | Maybe rotate camera once |
| 3. Image software rotation | ~2s | None |
| 4. GRBL↔Screen mapping | ~15s | None |
| 5. Camera↔Screen mapping | ~15s | None |
| 6. Full-chain validation | ~5s | None |
| Set origin at screen center | ~2s | None |

---

## Skill System

### What a Skill Is

A recorded, parameterized app automation flow. A folder containing:

```
skills/meituan_order/
├── SKILL.md               # Everything: flow, parameters, constants, per-screen annotations
└── screens/
    ├── 01_home.png
    ├── 02_search.png
    ├── 03_specs.png
    └── 04_cart.png
```

### Bootstrap: The Screenshot Skill

The screenshot skill is the foundation everything else depends on. The meta-skill needs screenshots to analyze pages. But the screenshot skill itself can't be built using screenshots — it's the bootstrap.

**What it provides:** Agent taps AssistiveTouch → iOS takes screenshot → Shortcut POSTs it to LAN server → agent receives clean PNG.

**Setup is manual, guided by agent via chat:**

```
Agent: "We need to set up screenshot capability first. I'll walk you through."

Step 1: Enable AssistiveTouch
  Agent: "Open Settings → Accessibility → Touch → AssistiveTouch → turn it on."
  (Agent can send "Settings" to bridge → Spotlight opens Settings.
   But navigating inside Settings still needs user or physical taps.)

Step 2: Create the Shortcut
  Agent: "Open the Shortcuts app. Create a new shortcut with two actions:
         1. Take Screenshot
         2. Get Contents of URL → POST to http://192.168.1.100:8048/upload"
  (Agent provides the exact URL. User builds the shortcut manually.)

Step 3: Link AssistiveTouch to Shortcut
  Agent: "Go to Settings → Accessibility → Touch → AssistiveTouch →
         Custom Actions → Single Tap → select the shortcut you just made."

Step 4: Position the button
  Agent: "Drag the AssistiveTouch button to the right edge, vertically centered."
  
Step 5: Verify
  Agent taps at right-center edge position.
  → If screenshot arrives → done ✓, record this position
  → If not → Agent: "Didn't get it. Please move the button slightly and I'll try again."
  → Repeat until screenshot arrives
```

**This takes ~5 minutes with agent guidance.** Done once, never again (unless the user moves AssistiveTouch or changes the Shortcut). After this, the agent can take screenshots autonomously — which unlocks the meta-skill.

### Bootstrap: Open App Skill

A built-in skill that opens any app via Spotlight. No user setup needed — it uses the bridge and calibration that are already working.

```
1. Stylus swipes up from bottom edge → return to home screen
   (Ensures clean state regardless of which app was in foreground)
2. Agent sends app name to /message page → e.g. "美团外卖"
3. Stylus taps the message page → text copied to clipboard
4. Stylus swipes down from screen top → Spotlight opens
5. Stylus taps search field → long press → tap "Paste"
6. Stylus taps the first search result → App opens
```

Step 1 (go home) guarantees a known starting state. Without it, Spotlight might behave differently if an app is in foreground, a modal is open, or the notification center is showing.

This is a fixed sequence — the same physical actions every time, only the app name changes. No screenshots, no CV, no skill building needed.

Every other skill starts with this: go home → open-app → then do the task-specific steps.

### Meta-Skill: The Skill Builder's Analysis Toolkit

The meta-skill is what the agent uses to create other skills. It must extract enough information from each screenshot that non-technical users only need to navigate and confirm — never draw boxes, specify coordinates, or understand technical concepts.

Three analysis tools run on every screenshot, each finding different things:

**Tool 1: Color Block Segmentation (HSV saturation pipeline)**

Finds all colored UI elements by exploiting the fact that functional apps use high-saturation colors only for actionable elements against white/gray backgrounds.

```
Input:  clean screenshot
Process: HSV convert → S-channel Otsu → morphology → connected components
Output: list of color blobs with bbox, color name, H_std, is_solid flag

Finds: CTA buttons, "+" buttons, selected tabs, promotional tags,
       price text (red), cart icons, brand-colored elements
Misses: gray buttons, unselected tabs, text-only links, input fields
```

Additionally classifies each blob as "content image" (H_std > 25, e.g., food photos) vs "solid UI element" (H_std < 12, e.g., buttons). Content images become list structure anchors.

**Tool 2: Icon Detection (OmniParser / YOLOv8)**

Detects UI elements by visual pattern — icons, buttons, text fields, checkboxes, toggles — regardless of color. Catches the gray/white elements that color segmentation misses.

```
Input:  clean screenshot
Output: list of UI elements with bbox, type label, confidence

Finds: gray buttons, unselected tabs, input fields, navigation icons,
       close buttons, back arrows — anything that looks like a UI control
Note:   On clean screenshots (vs camera captures), detection accuracy 
        is much higher than PhysiClaw's current camera-based usage
```

**Tool 3: OCR (RapidOCR)**

Reads all text on screen. On clean screenshots, Chinese text accuracy is near 100%.

```
Input:  clean screenshot (or cropped region)
Output: list of text strings with bbox, confidence

Finds: button labels ("去结算", "加入购物车"), item names, prices,
       category names, tab labels, navigation titles, spec options
```

**How the three tools combine on one screenshot:**

```
All bounding boxes use normalized 0-1 values: [left, top, right, bottom] relative to screen.

Screenshot captured
    │
    ├─ Color segmentation → colored elements + content images
    │   "4 orange circles [0.88, 0.12, 0.95, 0.16] (add buttons)"
    │   "5 square photos [0.02, 0.10, 0.22, 0.18] (food images, list structure)"
    │   "1 yellow rectangle [0.05, 0.90, 0.95, 0.96] (CTA button)"
    │   "3 red small blobs [0.15, 0.14, 0.25, 0.16] (price text)"
    │
    ├─ Icon detection → all UI elements including colorless ones
    │   "search icon [0.90, 0.02, 0.97, 0.05]"
    │   "back arrow [0.01, 0.03, 0.07, 0.06]"
    │   "gray minus buttons next to orange plus buttons"
    │
    └─ OCR → all text with positions
        "去结算" [0.70, 0.91, 0.85, 0.95]
        "宫保鸡丁" [0.12, 0.10, 0.40, 0.13]
        "¥18.8" [0.12, 0.14, 0.25, 0.16]
        "热销 | 主食 | 小吃 | 饮品" [0.02, 0.07, 0.50, 0.09]

Merge results:
    → Orange circle [0.88, 0.12, 0.95, 0.16] + OCR finds no text inside → "add_button"
    → Yellow rect [0.05, 0.90, 0.95, 0.96] + OCR reads "去结算" → "CTA: checkout"
    → Square photos vertically aligned → "list structure, card_height≈0.08"
    → Text "热销" at left side + color shows highlight → "selected category"
```

The agent presents merged results in plain language: *"This is a food menu. Left side has categories. Right side has 4 items with photos, names, prices, and '+' buttons. There's a checkout bar at the bottom."* User says "yes" or corrects.

### Skill Build Phase (one-time, ~15 min per skill)

Agent drives the entire process. User only confirms or corrects. **Critically, the agent must think about future executions, not just record the current walkthrough.** The skill must work for any item, not just the one used during building.

1. **User describes the task** — "帮我建一个美团外卖的技能"
2. **Agent opens the app** — open-app skill runs automatically ("美团外卖")
3. **For each screen, agent analyzes with generality in mind:**
   - Agent taps AssistiveTouch → screenshot → all three CV tools run
   - Agent classifies every element into two categories:

   **Fixed elements** (same every time, save exact bbox):
   - Navigation icons: search icon, back arrow, home tab
   - CTA buttons: "去结算", "加入购物车"
   - Tab bar items, sidebar categories
   - These get saved as fixed coordinates in SKILL.md

   **Dynamic elements** (change with content, save rules instead):
   - List items: food cards, restaurant cards — different every time
   - For these, agent extracts **layout constants**: image column x-position, add-button x-position, card height, scrollable region boundaries
   - Search results: first result position is roughly fixed, but content varies
   - Spec options (大杯/中杯): text varies, need OCR at runtime to find the right one

   - Agent proposes: *"Search icon is fixed, I'll save its position. The food list scrolls — I'll save the layout pattern (image at x=0.02, add button at x=0.88, card height≈0.08) so I can find any item at runtime."*
   - User confirms or corrects

4. **Agent identifies parameters:**
   - What changes between executions? → dish name, quantity, specs (size, sugar level)
   - What's the search strategy? → paste dish name into search bar, tap first result
   - *"这个技能需要参数：菜名、数量。下次执行时我会搜索菜名，在结果里找到它。"*

5. **Agent plans runtime CV strategy for each screen:**
   - How will camera (not screenshot) identify this screen? → save distinctive fixed regions as fingerprints (nav bar color, tab bar layout, page title text)
   - Which elements need camera detection at runtime? → only dynamic elements (finding the right list item)
   - Which elements can use saved coordinates directly? → all fixed elements

6. **Agent saves skill** — SKILL.md contains: flow steps, fixed element positions, layout constants, parameters, runtime CV strategy per screen

**The skill is a template, not a recording.** It says "search for {dish_name}, find it in the list using layout pattern, tap its add button" — not "tap at position (0.88, 0.35) where 宫保鸡丁 was".

### Skill Runtime Phase

Two distinct phases: gather information first, then execute with minimum actions.

**Phase A: Pre-Execution (before touching the app)**

Agent understands what the user wants and gathers all needed information using its own tools — web search, knowledge, reasoning. The goal is to enter the app with a clear, specific target.

```
User: "帮我点一份辣的外卖"

Agent thinks:
  → Web search: "美团外卖 附近 辣菜 评价好 推荐"
  → Web search: "宫保鸡丁 哪家好吃 评分"
  → Decides: 宫保鸡丁 from 川湘人家, large portion
  → Confirms with user: "我打算在川湘人家点一份宫保鸡丁大份，可以吗？"
  → User: "好"
  → Now agent knows exactly: app=美团外卖, search="川湘人家 宫保鸡丁", specs=大份
```

**Phase B: Execution (inside the app, minimum actions)**

Every step uses search-first strategy. **Never scroll to browse. Never tap through categories to find items.** Scrollable items are hard to locate precisely with camera and even harder to tap accurately — the position depends on scroll state, card heights, and content loading.

```
1. Open app (open-app skill)
2. For each skill step:
   → Match screen via camera (feature match against reference screenshot)
   → Fixed elements: tap directly using saved coordinates
   → Search elements: bridge sends text → tap /message → clipboard → paste into search box
   → Confirm screen transition via camera

Search is the primary interaction:
   → Find restaurant: paste restaurant name into search → tap first result
   → Find dish: paste dish name into search → tap first result
   → Select specs: OCR reads options on screen, tap the matching one (fixed layout, not scrollable)
```

**Scroll is a last resort, and only for information gathering:**

If web search didn't find the menu (e.g., merchant updated app menu but web isn't updated yet), agent may scroll through the in-app menu to read items. But even then — once the agent knows what item it wants, it goes back to the search box and types the item name. **Never tap a scrollable list item directly.** Always use search to navigate to it.

```
Fallback flow:
  Agent can't find "川湘人家" menu online
  → Opens the restaurant page in app
  → Scrolls through menu (camera + OCR reads item names)
  → Finds "水煮鱼" looks good
  → Goes BACK to search box → pastes "水煮鱼" → taps first result
  → Now the item is at a known, stable position → taps "加入购物车"
```

### Example: Message Skill (Agent ↔ User Communication)

This is design philosophy #5 in action — the agent has a phone, the user has a phone, they talk through messages like two people.

**The agent's phone has WeChat/WhatsApp. The user sends commands to the agent via messages. The agent reads, executes, and replies.**

```
User's phone (WeChat) → Agent's phone (WeChat)

User sends: "帮我点一份宫保鸡丁"
Agent's phone receives the message

Agent: 
  → Open WeChat (open-app skill)
  → Screenshot → OCR reads the latest message from user
  → Understands: user wants to order 宫保鸡丁
  → Replies: "收到，正在帮你下单美团外卖"
     (bridge paste reply → tap input → send)
  → Switches to food delivery skill → executes
  → Comes back to WeChat → replies: "已下单，预计30分钟送达，订单号xxx"

User sends: "帮我充100话费"
Agent:
  → Reads message → understands → replies: "好的，正在充值"
  → Runs phone credit skill
  → Replies: "充值成功，到账100元"

User sends: "验证码发到你手机了，是多少？"
Agent:
  → Reads SMS or notification → replies: "验证码是 384729"
```

**This is the agent's main loop:**
```
while True:
    check WeChat for new message from user
    if new message:
        read → understand → reply "收到"
        execute the task (may involve other skills)
        reply with result
    sleep(interval)
```

No API, no OAuth, no webhook. Just two phones messaging each other.

**Building the message skill (via meta-skill, same as any other skill):**

WeChat's UI is not a system-level fixed layout. The search box, chat list, input field, and send button all need to be located during skill building.

```
Agent opens WeChat → screenshot → CV analysis:

Screen 1: Chat list
  → Search box [0.10, 0.03, 0.90, 0.07]     (fixed, save bbox)
  → Chat list area [0.0, 0.10, 1.0, 0.90]    (dynamic, save layout pattern)
  → Each chat card: avatar x-position, name x-position, badge position

Screen 2: Conversation with user
  → Message bubble area [0.0, 0.08, 1.0, 0.90]  (dynamic, latest = bottom-most)
  → Input field [0.05, 0.92, 0.80, 0.97]         (fixed, save bbox)
  → Send button [0.82, 0.92, 0.98, 0.97]         (fixed, save bbox)
  → How to identify latest unread: OCR the bottom bubbles, read newest first

All saved to skills/wechat_message/SKILL.md
```

**Runtime uses the same search-first principle:** search the user's contact name in WeChat search box (paste, don't scroll the chat list), open conversation, read latest messages via OCR, compose reply, paste into input field, tap send.

---

## CV Pipeline

### Role in New Architecture

Three analysis tools serve different purposes at different times:

**During skill building (on clean screenshots):**

| Tool | What It Finds | Accuracy |
|------|--------------|----------|
| Color segmentation | Colored buttons, icons, tags, content images, list structure | Near-perfect on clean screenshots |
| Icon detection (YOLO) | All UI elements including gray/colorless ones | High on clean screenshots |
| OCR (RapidOCR) | All text: labels, names, prices, categories | Near 100% on clean screenshots |

All three run on every screenshot. Results are merged to produce a complete page understanding. This is the heavy analysis — runs once per screen during skill creation.

**During skill execution (on camera frames):**

| Task | Method | Notes |
|------|--------|-------|
| Screen state matching | ORB feature match against reference screenshot | Noisy input, but comparing against known reference |
| State confirmation | Frame differencing / histogram comparison | Only needs coarse "changed or not" answer |
| List card y-detection | 1D saturation scan in known x-column | High-S food images visible even through noise |
| Unexpected popup detection | V-channel brightness check (dark overlay) | Robust even on noisy frames |

Runtime CV is lightweight — the heavy understanding was done during building.

### Implementation (lightweight)

```python
# Only what's actually needed:

def segment_saturation(image_bgr):
    """HSV S-channel Otsu threshold. Core of all detection."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, mask = cv2.threshold(hsv[:,:,1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask, hsv

def classify_blobs(mask, hsv):
    """Connected components + H_std split into images vs buttons."""
    # morphology close+open, connectedComponentsWithStats
    # per blob: H_std > 25 → image, H_std < 12 → solid button
    ...

def match_screen(camera_frame, reference_screenshot):
    """ORB feature matching → homography → confidence score."""
    ...

def find_card_y_positions(camera_frame, layout_constants):
    """1D scan: in the known image-column, find rows with high S mean."""
    ...
```

Total: ~500 lines of Python. OpenCV + NumPy only.

---

## Tech Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Agent brain | Claude via MCP | Drives conversation, makes decisions, guides skill creation |
| MCP server | Python + FastMCP | Exposes tools: skill_create, skill_capture, tap, swipe, etc. |
| Bridge server | Python (aiohttp or FastAPI) | Serves webpage, handles WebSocket, receives screenshot uploads |
| Bridge client | Single HTML file | Display text + tap to copy + touch reporting + device info. ~80 lines |
| Motion control | GRBL firmware | Existing PhysiClaw hardware |
| Camera | OpenCV VideoCapture | USB camera, 1080p |
| **Skill build analysis tools:** | | |
| — Color segmentation | OpenCV + NumPy | HSV Otsu + connected components + H_std classification. ~500 lines |
| — Icon detection | OmniParser V2 / YOLOv8 | Existing PhysiClaw dependency. Detects all UI elements including gray ones |
| — OCR | RapidOCR | Existing PhysiClaw dependency. Text reading on screenshots |
| **Runtime CV** | OpenCV + NumPy | ORB matching, frame differencing, 1D y-scan. Lightweight |
| Skill storage | Markdown + PNG files | Human-readable, no database needed |

---

## Pros

- **Near-zero runtime CV** for fixed-element interactions. Annotated coordinates + calibration matrix = direct tap.
- **Perfect text input speed.** Bridge clipboard paste vs 20 seconds of physical keyboard tapping.
- **Sub-pixel calibration precision** via bridge touch feedback, surpassing camera-based methods.
- **15 minutes to add a new app.** No code, no training data, no model fine-tuning.
- **Surgical repair on app updates.** Re-screenshot one changed screen, not rebuild entire pipeline.
- **Zero phone modification.** Stock phone, browser only. No USB debug, no sideloaded apps.
- **Tiny footprint.** ~500 lines CV code, ~200 lines bridge, JSON skill files. No GPU, no model weights.
- **Secure by design.** Physical camera can't be hijacked remotely. Bridge only writes clipboard (agent→phone). No background data extraction from phone.

## Cons

- **Not general-purpose.** Every new workflow needs a skill built. Can't handle "just explore this app for me" without a pre-built skill.
- **Screenshots only during skill building.** AssistiveTouch is ON and screenshots are taken only during the one-time skill setup (~10 per skill). During daily runtime execution, AssistiveTouch is OFF, no screenshots are taken, nothing for apps to detect.
- **App A/B testing can shift layouts.** Same app, different users may see slightly different UI. Skill may need per-user adjustment.
- **Runtime OCR on camera frames is less accurate than on screenshots.** Mitigated by search-first strategy (text is pasted via bridge, not OCR'd) and fuzzy matching. OCR is only needed when scrolling through list items as a fallback — the rare case where search isn't available.
- **Bridge requires the browser tab to remain connected.** If iOS aggressively suspends the background tab, WebSocket may drop. Requires reconnection logic. AssistiveTouch + Shortcuts pathway is unaffected.
- **Single-phone assumption.** Calibration and skills are tied to one phone model/resolution. Switching phones requires recalibration and possibly skill coordinate scaling.
- **Initial setup friction.** User must: open browser to bridge URL, configure AssistiveTouch, create Shortcuts automation. ~10 minutes one-time setup with agent guidance. Calibration is ~80 seconds — user places phone, positions stylus roughly over screen center, possibly adjusts phone or camera once if prompted. Everything else is automatic.

---

## Implementation Order

### Phase 1: Bridge + Calibration (foundation)

Build the bridge web server and client page. Implement WebSocket communication. Auto-collect device info on page load. Implement calibration: user positions stylus over screen center; (0) Z-axis step-and-check surface detection at 0.25mm increments; (1) two X-axis taps for arm-phone alignment; (2) camera checks phone long axis vs image long axis; (3) UP/RIGHT markers determine software rotation; (4) distributed taps expanding from center to edges/corners for GRBL↔Screen affine transform; (5) 15 displayed dots for Camera↔Screen homography; (6) full-chain validation with orange dot tap test.

**Deliverable:** Agent can precisely tap any screen coordinate (Mapping A) and convert any camera pixel to a screen coordinate (Mapping B). ~80 seconds, one manual action (position stylus).

### Phase 2: Screenshot Skill (bootstrap)

Guide user through AssistiveTouch + Shortcut setup via chat conversation. User follows step-by-step instructions: enable AssistiveTouch, create Shortcut (Take Screenshot → HTTP POST), link single-tap to shortcut. Agent detects AssistiveTouch button position from the first user-triggered screenshot. Agent verifies by tapping the button itself and confirming a screenshot arrives.

**Deliverable:** Agent can take a screenshot at any time by tapping AssistiveTouch. This unlocks all subsequent phases.

### Phase 3: Clipboard Text Input

Implement clipboard write via bridge. Implement physical paste sequence (tap field → long press → tap "Paste"). Test with search boxes across target apps.

**Deliverable:** Agent can input arbitrary text into any app's search field in ~2 seconds.

### Phase 4: Skill Builder (meta-skill)

Implement the meta-skill: the tool suite that creates other skills. Three analysis tools run on each screenshot: color segmentation (find colored elements + list structure), icon detection (find all UI elements including gray ones), OCR (read all text). Results are merged and presented in plain language. Agent guides non-technical users through conversation — user only navigates and answers simple questions.

**Deliverable:** A non-technical user can build a skill by saying "帮我建一个美团外卖的技能" and following the agent's plain-language guidance. No coordinates, no technical terms, no manual annotation.

### Phase 5: Screen Matching + Skill Execution

Implement ORB feature matching for screen state identification. Implement skill executor: load skill → match screen → execute action → verify transition. Camera-based state confirmation.

**Deliverable:** Agent can execute a fixed-button skill end-to-end autonomously.

### Phase 6: List Handling via Search

Implement search-first list interaction: clipboard paste search term → tap first result. Implement fallback OCR-based card matching for apps without good search. 1D y-scan for card position detection using layout constants.

**Deliverable:** Agent can complete a full food ordering flow — search for item, select specs, checkout.

### Phase 7: Robustness

Unexpected popup dismissal (detect dark overlay → find close button). Transition timeout recovery (retry action, back button). WiFi reconnection for bridge. Multi-frame stability check before acting. Human-like timing (random delays, slight coordinate jitter).

**Deliverable:** Reliable unattended operation.
