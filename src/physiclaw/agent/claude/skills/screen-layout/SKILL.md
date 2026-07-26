---
name: screen-layout
description: First-run setup — learn the bboxes of your three key input boxes (Spotlight search, chat input keyboard-down, chat input keyboard-up) plus the keyboard keys and Paste buttons. Run once when SYSTEM shows the first-run notice, before opening apps by search or sending messages. Screenshot each page, read the box coordinates off the returned elements, save them with the screen_layout.py CLI.
---

# Learn the screen layout

Opening apps (Spotlight) and messaging the user (IM) both need exact input-box positions. First run only, capture the three pages below.

Read coordinates from `screenshot` (pixel-perfect), never camera `peek` — but DO `peek` to confirm what a tap did (keyboard up, chat open). Copy each element's `bbox` **verbatim** — the exact `[left, top, right, bottom]` (0–1, 3 decimals) the screenshot returned; never eyeball, re-estimate, or re-round. The keyboard must be the system default (no third-party keyboard).

## How to save

Boxes are written through a single CLI — never edit `layout.json` by hand:

```bash
uv run python "$SKILL_DIR/screen_layout.py" record \
  --page <spotlight | chat-no-keyboard | chat-keyboard> \
  [--app <chat app>] \
  --box <field>=<l,t,r,b> [--box <field>=<l,t,r,b> ...]
```

- One `--box` per element you read; pass every box for a page in one call.
- `--app` = the chat app you opened (`wechat`, `whatsapp`, `telegram`, `signal`, …) — chat pages only; it labels the chat boxes. Omit for Spotlight.
- Each box is sanity-checked and saved to `~/.physiclaw/screen-layout/`; the command prints the layout so far and how many boxes remain. A box rejected as out-of-region means you picked a neighbouring element — re-read that box off the screenshot and run again (re-recording a field overwrites it).
- `$SKILL_DIR` is provided by Claude Code; it resolves to this skill's directory.
- Check progress any time with `uv run python "$SKILL_DIR/screen_layout.py" status`.

## Page 1 — Spotlight (keyboard up)

Fields: `spotlight_input`, `space`, `backspace`, `return`, `spotlight_paste`.

1. `home_screen` (skip if already on home — it opens the App Switcher there) → `swipe` down from mid-screen → Spotlight opens.
2. `tap` the search field; `peek` — keyboard up.
3. `screenshot`; read the search field (`spotlight_input`) and the keyboard's `space`, `backspace`, and `return` keys.
4. `spotlight_paste`: put any text on the clipboard, **long-press** the search field, `screenshot`, read the **Paste** button.
5. Save them all in one call, e.g.:

```bash
uv run python "$SKILL_DIR/screen_layout.py" record --page spotlight \
  --box spotlight_input=0.020,0.582,0.958,0.660 \
  --box space=... --box backspace=... --box return=... --box spotlight_paste=...
```

## Page 2 — Chat input, keyboard DOWN

Field: `chat_input_kb_hidden`. Pass `--app`.

1. `home_screen`, then open the dock's IM app; if the dock has none, open one via `im` / `open-app` using the Spotlight boxes you just read. Enter any real thread, keyboard hidden.
2. `screenshot`; read the **chat input bar**.
3. `... record --page chat-no-keyboard --app wechat --box chat_input_kb_hidden=…`

## Page 3 — Chat input, keyboard UP

Fields: `chat_input_kb_visible`, `send`, `chat_paste`. Same `--app` as Page 2.

1. `tap` the chat input box; `peek` — keyboard fully up (all rows, no predictive-bar collapse).
2. `screenshot`; read the **input bar** and **Send** — either a bottom-right **keyboard key** (WeChat-style) or an **input-bar button** on the right (WhatsApp-style). Send hidden while the input is empty → tap any key so it appears (step 3 clears it).
3. `chat_paste`: **clear the input** (backspace until empty — a non-empty box shifts Paste in the popover), put text on the clipboard → **long-press** the input box → `screenshot` → read **Paste**.
4. Save all three with the same `--app`.

## Done

When every box is in, the CLI reports the layout complete. From the next wake the SYSTEM prompt carries the full layout under `## Screen layout`. Proceed with the original task, or emit `>> IDLE` if learning the layout was the only reason you woke.

**Note:** keyboard geometry is device-global — learned once from Spotlight, it holds in every app. Input/Send boxes are per-context but stable.
