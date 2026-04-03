# QWERTY Mobile Keyboard Key Locator — Implementation Plan

## Goal

Given a mobile phone screenshot with the keyboard visible, output pixel coordinates for every key.
Support iOS system keyboard and Android Gboard, both light and dark themes.
Pure Python, only dependencies: opencv-python-headless and numpy.

---

## Core Insight

The QWERTY keyboard layout is fully deterministic. The arrangement of 26 letters, number of keys per row (10-9-7), and row offsets (0, 0.5, 1.5 key-widths) are hardcoded constants in every OS.
The only unknown is `kb_top` (the y-coordinate of the keyboard's top edge).
Once kb_top + screen_w + kb_bottom are known, every key coordinate is pure arithmetic.

---

## Architecture: Two Phases

### Phase 1: Calibration (runs once on first use)

```
Screenshot → auto-detect kb_top/kb_bottom → grid formula computes all key coords → save JSON
```

### Phase 2: Runtime (every keystroke)

```
Load JSON → table lookup → return (x, y)
```

---

## Module 1: Keyboard Region Detection

### Input
- BGR numpy array (phone screenshot)

### Output
- kb_top: int (keyboard top y-coordinate)
- kb_bottom: int (keyboard bottom y-coordinate)

### 5 Detection Strategies

Each strategy independently returns a kb_top candidate or None. Final result: remove outliers, take median.

#### Strategy 1: Row Variance Jump
- Principle: keyboard rows have very low pixel variance (uniform color); app content has high variance
- Steps:
  1. Convert to grayscale
  2. Compute per-row pixel variance → array of length h
  3. Smooth with length-5 mean kernel
  4. Take median variance in y ∈ [70%h, 85%h] as "typical keyboard variance"
  5. Threshold = max(typical_variance × 3, 100)
  6. Scan upward from 70%h, find where variance jumps above threshold → kb_top

#### Strategy 2: Grayscale Mean Gradient Jump
- Principle: keyboard top edge has a sharp brightness step
- Steps:
  1. Compute per-row grayscale mean
  2. Compute absolute first-order difference
  3. In range [h/3, 85%h], dynamic threshold = median + 3×std
  4. Find positions exceeding threshold
  5. Among candidates, pick the one closest to 65%h (typical keyboard top position)

#### Strategy 3: HSV Saturation Jump
- Principle: keyboard background has near-zero saturation regardless of dark/light theme; app content usually has color
- Steps:
  1. Convert to HSV
  2. Compute per-row ratio of pixels with S < 30
  3. Smooth with length-7 mean kernel
  4. Scan upward from 80%h, find where ratio drops from > 0.6 to < 0.4

#### Strategy 4: Horizontal Edge Density
- Principle: keyboard interior has dense horizontal edges (key borders); keyboard top has a strong horizontal divider
- Steps:
  1. Canny edge detection (50, 150)
  2. Morphological close with (30, 1) rectangular kernel to connect horizontal edges
  3. Compute per-row edge pixel ratio
  4. In [h/3, 85%h], use sliding window to find density jump from < 0.05 to > 0.1

#### Strategy 5: Color Histogram Entropy
- Principle: keyboard region colors cluster in few values (grays), low histogram entropy; content region has rich colors, high entropy
- Steps:
  1. Convert to grayscale
  2. Compute 32-bin histogram entropy for every 10-row block
  3. Scan upward, find where entropy jumps from < 3.0 to > 3.5

#### Ensemble Fusion
1. Collect all non-None candidate values from 5 strategies
2. Discard values outside [h/3, 85%h]
3. Compute median
4. Discard outliers more than 5%h from median
5. Take median of remaining values as final kb_top
6. If all strategies fail, fallback to 62%h (typical keyboard position on most phones)

#### kb_bottom Detection
- Starting from 92%h, scan downward for 4+ consecutive rows with std < 8 (navigation bar / home indicator)
- Start of that region = kb_bottom
- If not found, kb_bottom = h

---

## Module 2: Keyboard Internal Structure Parameters

### Input
- kb_top, kb_bottom, screen_w

### Output
- Y-boundaries for each internal zone, key dimensions

### Standard QWERTY Keyboard Structure (Portrait)

```
kb_top
├── Suggestion bar         height ratio ~13%
├── Letter row 1 (QWERTYUIOP)    ─┐
├── Letter row 2 (ASDFGHJKL)      ├ Letter zone ratio ~52%
├── Letter row 3 (⇧ZXCVBNM⌫)    ─┘
├── Bottom bar (123/space/return)  ratio ~35% (includes iOS safe area)
kb_bottom
```

### Parameter Calculation

```
kb_h = kb_bottom - kb_top
suggest_h = kb_h × 0.13          # suggestion bar height
letter_h = kb_h × 0.52           # total height of 3 letter rows
bottom_h = kb_h × 0.35           # bottom function bar height

letter_top = kb_top + suggest_h
row_h = letter_h / 3             # single letter row height
key_w = screen_w / 10            # standard key width (row 1 has 10 keys, full width)
```

---

## Module 3: Key Coordinate Calculation

### Layout Definitions

Support multiple keyboard modes. Each mode is defined as a list of rows, each row containing a list of keys and a left offset.

#### Mode 1: English Alphabet Keyboard

```
Row 1: Q W E R T Y U I O P        10 keys, offset 0,   each 1.0× key_w
Row 2: A S D F G H J K L          9 keys,  offset 0.5, each 1.0× key_w
Row 3: ⇧ Z X C V B N M ⌫         9 keys,  offset 0,   ⇧=1.5× letters=1.0× ⌫=1.5×
Row 4: 123 🌐 [space] . ↵         5 keys,  offset 0,   123=1.5× 🌐=1.0× space=5.0× .=1.0× ↵=1.5×
```

Uppercase and lowercase share the same coordinates; only the output character differs.

#### Mode 2: Numeric/Symbol Keyboard (iOS Style)

```
Row 1: 1 2 3 4 5 6 7 8 9 0        10 keys, offset 0
Row 2: - / : ; ( ) $ & @ "        10 keys, offset 0
Row 3: #+= . , ? ! ' ⌫            8 keys,  offset 0,   #+= and ⌫ are 1.5×
Row 4: ABC 🌐 [space] . ↵         5 keys,  same as alpha row 4
```

#### Mode 3: Numeric/Symbol Keyboard (Android Gboard Style)

```
Row 1: 1 2 3 4 5 6 7 8 9 0        10 keys, offset 0
Row 2: @ # $ _ & - + ( ) /        10 keys, offset 0
Row 3: =\< * " ' : ; ! ? ⌫       9 keys,  offset 0,   =\< and ⌫ are 1.5×
Row 4: ABC , [space] . ↵          5 keys,  offset 0,   ABC=1.5× ,=1.0× space=5.0× .=1.0× ↵=1.5×
```

### Coordinate Formula

For each key in a row:

```
x_cursor starts at left_offset × key_w

For each key (name, action, width_units):
  key_pixel_width = width_units × key_w
  cx = x_cursor + key_pixel_width / 2
  cy = letter_top + row_index × row_h + row_h / 2    (for letter rows)
  x1 = x_cursor
  y1 = cy - row_h / 2
  x2 = x_cursor + key_pixel_width
  y2 = cy + row_h / 2
  x_cursor += key_pixel_width
```

Bottom function bar cy:
```
bottom_cy = letter_top + letter_h + bottom_h / 2
```

---

## Module 4: Keyboard State Machine

### States

```
ALPHA_LOWER  — English lowercase (default)
ALPHA_UPPER  — English uppercase (single shift tap)
ALPHA_CAPS   — Caps lock (double shift tap)
NUM_SYMBOL   — Numeric/symbol page
```

### State Transitions

```
ALPHA_LOWER --tap SHIFT-->    ALPHA_UPPER --type one letter--> ALPHA_LOWER
ALPHA_LOWER --double SHIFT--> ALPHA_CAPS  --tap SHIFT-->       ALPHA_LOWER
ALPHA_LOWER --tap 123-->      NUM_SYMBOL  --tap ABC-->         ALPHA_LOWER
ALPHA_UPPER --tap SHIFT-->    ALPHA_LOWER
```

### Smart Typing Interface

Given a string, auto-generate the full tap sequence including mode switches:

```
type_string("Hello 123") should produce:
  1. Current LOWER → need uppercase H → tap SHIFT → state becomes UPPER
  2. Tap H
  3. State auto-reverts to LOWER
  4. Tap e, l, l, o
  5. Need space → tap SPACE
  6. Need digits → tap MODE_NUM → state becomes NUM_SYMBOL
  7. Tap 1, 2, 3
```

Algorithm logic:
```
For each character ch in input string:

  if ch is lowercase letter:
    Ensure in ALPHA_LOWER or ALPHA_CAPS state
    If in NUM_SYMBOL → tap MODE_ALPHA first
    If in ALPHA_UPPER → just type (will auto-revert to LOWER)
    Look up ch.upper() in alpha keymap, output tap
    If was in ALPHA_UPPER → state reverts to ALPHA_LOWER

  if ch is uppercase letter:
    Ensure in ALPHA_UPPER or ALPHA_CAPS state
    If in NUM_SYMBOL → tap MODE_ALPHA first
    If in ALPHA_LOWER → tap SHIFT first
    Look up ch in alpha keymap, output tap
    If was in ALPHA_UPPER (not CAPS) → state reverts to ALPHA_LOWER

  if ch is digit:
    Ensure in NUM_SYMBOL state
    If not → tap MODE_NUM first
    Look up ch in numeric keymap, output tap

  if ch is space:
    Space key exists in all modes at same position, tap directly

  if ch is common punctuation (. ,):
    . and , exist on the alpha keyboard bottom row, no mode switch needed
    Other punctuation requires switching to NUM_SYMBOL
```

---

## Module 5: Configuration Persistence

### JSON Format

```json
{
  "platform": "ios",
  "screen": [1170, 2532],
  "kb_region": [1680, 2532],
  "modes": {
    "alpha": {
      "q": {"cx": 58.5, "cy": 1755, "bbox": [0, 1722, 117, 1788]},
      "w": {"cx": 175.5, "cy": 1755, "bbox": [117, 1722, 234, 1788]},
      "SHIFT": {"cx": 87.75, "cy": 1887, "bbox": [0, 1854, 175.5, 1920]},
      "DELETE": {"cx": 1082.25, "cy": 1887, "bbox": [994.5, 1854, 1170, 1920]},
      "MODE_NUM": {"cx": 87.75, "cy": 1960, "bbox": [0, 1920, 175.5, 2000]},
      "SPACE": {"cx": 585, "cy": 1960, "bbox": [292.5, 1920, 877.5, 2000]},
      "ENTER": {"cx": 1082.25, "cy": 1960, "bbox": [994.5, 1920, 1170, 2000]}
    },
    "num": {
      "1": {"cx": 58.5, "cy": 1755, "bbox": [0, 1722, 117, 1788]},
      "2": {"cx": 175.5, "cy": 1755, "bbox": [117, 1722, 234, 1788]},
      "MODE_ALPHA": {"cx": 87.75, "cy": 1960, "bbox": [0, 1920, 175.5, 2000]}
    }
  }
}
```

### Load Logic
- If config file exists AND screen size matches → load directly
- Otherwise → run calibration flow

---

## Module 6: Debug Visualization

Draw detection results for verification:
- Red horizontal line at kb_top
- Blue horizontal lines at row boundaries
- Green rectangles for each key bbox
- Red text inside each box showing key name
- Orange boxes for low-confidence keys
- Save as xxx_debug.png

---

## Module 7: CLI Interface

```bash
# Calibrate: detect keyboard + save config
python kb.py calibrate screenshot.png --platform ios

# Tap: query coordinates for a single key
python kb.py tap H
python kb.py tap SPACE
python kb.py tap SHIFT

# Type: output full tap sequence for a string
python kb.py type "Hello World 123"

# Debug: output visualization image
python kb.py debug screenshot.png
```

---

## Data Structures Summary

```
KeyBox:
  name: str               # display name ("Q", "⇧", "123")
  action: str             # action identifier ("q", "SHIFT", "MODE_NUM")
  cx, cy: float           # center coordinates
  x1, y1, x2, y2: float  # bounding box
  width_units: float      # how many standard key widths this key spans

KBMode: Enum
  ALPHA_LOWER, ALPHA_UPPER, ALPHA_CAPS, NUM_SYMBOL

KeyboardController:
  platform: str
  mode: KBMode
  mode_keys: dict[KBMode, dict[str, KeyBox]]

  calibrate(screenshot) → self
  tap(char) → (x, y)
  type_string(text) → list[(action, x, y)]
  get_switch_sequence(from_mode, to_mode) → list[(action, x, y)]
```

---

## Implementation Constraints

1. No deep learning models
2. No OCR
3. No network access
4. Only dependencies: opencv-python-headless and numpy
5. Zero image processing at runtime after calibration
6. Single file implementation, no more than 500 lines
7. All coordinates in pixels, no dp/pt logical units
8. Support both iOS and Android numeric keyboard layouts

---

## Testing Checklist

- Validate with 1080×2400 (common Android FHD+) and 1170×2532 (iPhone 14) resolutions
- Verify all 26 letter keys + shift + delete + space + enter + 123 coordinates on alpha keyboard
- Verify 0-9 + ABC coordinates on numeric keyboard
- Verify type_string output for mixed case + digits + spaces + punctuation
- Verify shift single-tap (temporary uppercase) and double-tap (caps lock) state transitions
- Verify mode switching: alpha → num → alpha round trip
- Verify deep/light theme detection works on both iOS and Android screenshots