---
name: wechat
description: Operate WeChat (微信) — open from dock, find a contact in Chats, send a message via paste→Send.
---

# WeChat (微信)

The owner's primary IM.

## Tool choice

High-frequency app. Use `scan()` (~1s, text-only) to check chat status.

## Flow — read messages

1. If already in the right chat, skip to step 5.
2. From Home, tap the WeChat dock icon.
3. If not on the **Chats** tab, tap it.
4. Tap the target 1:1 contact's row.
5. Read the new messages.

## Flow — voice message

Use the convert-to-text option, then `scan()` the transcript that renders under the bubble.

Reply with your reading + planned action and wait for OK before executing. ASR mishears names, amounts, addresses — never act on a voice instruction unconfirmed.

## Flow — send a message

Two states: **keyboard hidden** (input bar at bottom) and **keyboard visible** (shifted up, send key on keyboard).

1. Confirm right contact in the chat header.
2. If keyboard is hidden, tap the input box (keyboard-hidden bbox) — the keyboard becomes visible.
3. If the input has stale text, tap **backspace** until empty.
4. `send_to_clipboard(text)`, then long-press the input box (keyboard-visible bbox).
5. Tap **Paste** in the popup.
6. Tap the keyboard's **send** key — **not** the (+) on the input bar.
7. Hide the keyboard (see Hide keyboard below).
8. Confirm the bubble appeared in the chat.
9. Tap the back arrow `<` (top-left) to return to the Chats list — leaves WeChat in its main state for the next wake.

## Fixed elements

Bboxes `[left, top, right, bottom]` in 0-1 screen coords. Rescan if a row looks off — banners shift the layout.

**WeChat dock icon (Home Screen):** `[0.294, 0.891, 0.474, 0.967]`

### Bottom nav

| Tab               | Bbox                            |
| ----------------- | ------------------------------- |
| Chats (微信)      | `[0.070, 0.945, 0.190, 0.965]`  |
| Contacts (通讯录) | `[0.324, 0.947, 0.437, 0.962]`  |
| Discover (发现)   | `[0.578, 0.947, 0.684, 0.963]`  |
| Me (我)           | `[0.840, 0.945, 0.940, 0.965]`  |

### Chats list rows

Each row is **0.08 tall**. Derive the Nth row from the 1st by adding `0.08 × (N-1)` to the y values.

| Row | Bbox                            |
| --- | ------------------------------- |
| 1st | `[0.160, 0.180, 0.940, 0.260]`  |
| 2nd | `[0.160, 0.260, 0.940, 0.340]`  |

### Chat page

| Element         | When                | Bbox                            |
| --------------- | ------------------- | ------------------------------- |
| Back arrow `<`  | always              | `[0.016, 0.066, 0.082, 0.102]`  |
| Contact name    | always              | `[0.400, 0.060, 0.600, 0.100]`  |
| Text input area | keyboard hidden     | `[0.100, 0.910, 0.700, 0.960]`  |
| Text input area | keyboard visible    | `[0.100, 0.575, 0.700, 0.625]`  |
| Paste button    | long-pressing input | `[0.050, 0.530, 0.220, 0.570]`  |
| send (keyboard) | keyboard visible    | `[0.752, 0.864, 0.992, 0.917]`  |
| backspace `⌫`   | keyboard visible    | `[0.867, 0.804, 0.994, 0.857]`  |

### Hide keyboard

A small upward swipe in the empty chat scroll area dismisses the keyboard without scrolling the just-sent bubble off-screen.

`swipe(bbox=[0.300, 0.300, 0.700, 0.500], direction="up", size="s")`
