---
name: screen-layout
description: First-run setup — learn the bboxes of your three key input boxes (Spotlight search, chat input keyboard-down, chat input keyboard-up) plus the keyboard keys and Paste buttons. Run once when SYSTEM shows the first-run notice, before opening apps by search or sending messages. Screenshot each page, read the box coordinates off the returned elements, report via report_screen_layout.
---

# Learn the screen layout

Opening apps (Spotlight) and messaging the user (IM) both need exact input-box positions. First run only, capture the three pages below.

Read coordinates from `screenshot` (pixel-perfect), never camera `peek` — but DO `peek` to confirm what a tap did (keyboard up, chat open). Copy each element's `bbox` **verbatim** into `report_screen_layout` — the exact `[left, top, right, bottom]` (0–1, 3 decimals) the screenshot returned; never eyeball, re-estimate, or re-round. The keyboard must be the system default (no third-party keyboard).

The tool sanity-checks each box and saves it to `~/.physiclaw/screen-layout/`; each call confirms the save and how many boxes remain (the first-run notice lists which). A box rejected as out-of-region means you picked a neighbouring element — re-read and call again.

## How to report

One box per call:

```text
report_screen_layout(
  page="<spotlight | chat-no-keyboard | chat-keyboard>",
  field="<one field name for that page>",
  bbox=[left, top, right, bottom],   # verbatim from the element
  app="<chat app>",                  # chat pages only; omit for spotlight
)
```

`app` = whatever chat app you opened (`wechat`, `whatsapp`, `telegram`, `signal`, …) — it just labels the chat boxes.

## Page 1 — Spotlight (keyboard up)

Fields: `spotlight_input`, `space`, `backspace`, `return`, `spotlight_paste`.

1. `home_screen` → `swipe` down from mid-screen → Spotlight opens.
2. `tap` the search field; `peek` — keyboard up.
3. `screenshot`; read the search field (`spotlight_input`) and the keyboard's `space`, `backspace`, and `return` keys.
4. `spotlight_paste`: put any text on the clipboard, **long-press** the search field, `screenshot`, read the **Paste** button.
5. Report each, e.g.:

```text
report_screen_layout(page="spotlight", field="spotlight_input", bbox=[0.020, 0.582, 0.958, 0.660])
```

## Page 2 — Chat input, keyboard DOWN

Field: `chat_input_kb_hidden`. Pass `app`.

1. Open the chat app (see `im` / `open-app` — the layout isn't loaded yet, so use the Spotlight boxes you just read), enter any real thread, keyboard hidden.
2. `screenshot`; read the **chat input bar**.
3. `report_screen_layout(page="chat-no-keyboard", field="chat_input_kb_hidden", bbox=[…], app="wechat")`

## Page 3 — Chat input, keyboard UP

Fields: `chat_input_kb_visible`, `send`, `chat_paste`. Same `app` as Page 2.

1. `tap` the chat input box; `peek` — keyboard fully up (all rows, no predictive-bar collapse).
2. `screenshot`; read the **input bar** and **Send** — either a bottom-right **keyboard key** (WeChat-style) or an **input-bar button** on the right (WhatsApp-style). Send hidden while the input is empty → tap any key so it appears (step 3 clears it).
3. `chat_paste`: **clear the input** (backspace until empty — a non-empty box shifts Paste in the popover), put text on the clipboard → **long-press** the input box → `screenshot` → read **Paste**.
4. Report each with the same `app`.

## Done

When every box is in, the session restarts with the layout loaded. Proceed with the original task, or `end_session(IDLE)` if learning the layout was the only reason you woke.

**Note:** keyboard geometry is device-global — learned once from Spotlight, it holds in every app. Input/Send boxes are per-context but stable.