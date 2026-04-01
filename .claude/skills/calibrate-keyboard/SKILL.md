---
name: calibrate-keyboard
description: Detect keyboard keys from phone screenshots and generate UI preset for typing. Use when setting up a new phone or the keyboard layout has changed.
allowed-tools: Bash, Read, Edit, Write
---

# Keyboard Calibration

Guide the user through calibrating the on-screen keyboard for PhysiClaw.

## Step 1: Collect screenshots

Ask the user to take two phone screenshots with the keyboard visible:

1. **Alpha keyboard** (default QWERTY layout)
2. **Numeric keyboard** (tap the 123 key first)

Save both as PNG/JPG in `data/image/keyboard/`. Remove any old images from that directory first.

## Step 2: Check images and run detection

Run this check first:

```bash
uv run python -c "
import cv2
from pathlib import Path
imgs = sorted(Path('data/image/keyboard').glob('*.*'))
imgs = [p for p in imgs if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]
print(f'{len(imgs)} images found')
sizes = set()
for p in imgs:
    img = cv2.imread(str(p))
    if img is not None:
        sizes.add((img.shape[1], img.shape[0]))
        print(f'  {p.name}: {img.shape[1]}x{img.shape[0]}')
if len(sizes) == 1:
    print('All same size')
elif len(sizes) > 1:
    print(f'ERROR: different sizes: {sizes}')
if len(imgs) < 2:
    print('ERROR: need at least 2 images (alpha + numeric)')
"
```

Verify:

1. At least 2 images (one alpha, one numeric)
2. All images have the same width and height (same phone, same orientation)
3. The keyboard background is clean and uniform (no custom themes, no wallpaper keyboards)
4. The keyboard is the system default (Gboard, iOS keyboard, etc.) -- not a third-party keyboard
5. Original images, no resize, no editing

If any check fails, ask the user to fix and re-screenshot.

Then run detection:

```bash
uv run python scripts/calibrate_keyboard.py
```

This detects key bounding boxes and generates:

- Bounding box images in `data/image/keyboard/bbox/`
- A preset template at `.claude/ui-presets/system-keyboard.md` with positions filled in

Check the output: it should report 4 rows per keyboard, ~33 keys for alpha, ~35 for numeric.
If detection fails or key counts are wrong, ask the user to retake screenshots (original image, no resize, no editing).

## Step 3: Fill ??? labels

Keys marked ??? need to be identified. After filling, tell the user:
"You can open `.claude/ui-presets/system-keyboard.md` to check my editing."

For each keyboard page:

1. Tell the user: "Please open `{bbox image path}` to verify my labels." (the path is listed under each page heading)
2. Read the bounding box image yourself to identify keys
3. Fill ??? entries one page at a time
4. Always refer to keys by bbox index: "bbox 11: @, bbox 12: #"
5. After filling each page, list your guesses and ask the user to confirm
6. Punctuation/symbols are hard to distinguish -- always ask the user to verify these

### Symbol reference for punctuation

When identifying symbol keys, select from this list.
Note: Chinese Pinyin keyboards have no straight quotes (' "). Only curly quotes and enumeration comma.
If a symbol looks like ' or `, it is most likely 、 (enumeration comma).

| Symbol | Name | Type |
| -- | -------------- | ---- |
| @ | at sign | English |
| # | hash / pound | English |
| $ | dollar sign | English |
| _ | underscore | English |
| & | ampersand | English |
| * | asterisk | English |
| - | hyphen | English |
| + | plus | English |
| / | slash | English |
| , | comma | English |
| . | period | English |
| : | colon | English |
| ; | semicolon | English |
| ! | exclamation mark | English |
| ? | question mark | English |
| ( | left parenthesis | English |
| ) | right parenthesis | English |
| （ | left parenthesis (Chinese) | Chinese |
| ） | right parenthesis (Chinese) | Chinese |
| 、 | enumeration comma | Chinese |
| " | left double quote (Chinese) | Chinese |
| " | right double quote (Chinese) | Chinese |
| ： | colon (Chinese) | Chinese |
| ； | semicolon (Chinese) | Chinese |
| ！ | exclamation mark (Chinese) | Chinese |
| ？ | question mark (Chinese) | Chinese |
| ， | comma (Chinese) | Chinese |
| 。 | period (Chinese) | Chinese |
| ¥ | yen / RMB sign | Currency |

## Step 4: Verify positions unchanged

After filling all ???, verify that no Position values were accidentally modified by comparing
the Position columns of the filled file against the reference copy:

```bash
diff <(grep -oP '\[[\d., ]+\]' .claude/ui-presets/system-keyboard.md) \
     <(grep -oP '\[[\d., ]+\]' data/image/keyboard/bbox/system-keyboard.ref.md)
```

If no output, all positions match. If there are differences, the Position column was accidentally
edited — restore those rows from the reference file.

Then ask the user to review the final `.claude/ui-presets/system-keyboard.md`.

Done. The AI agent can now use the keyboard preset for typing.
