---
name: im
description: Use for any instant-messaging task — reading the user's chat, sending a reply, or finding a contact by name — in WeChat / 微信 or WhatsApp. Covers "check messages", "reply to <name>", "tell <name> ...". This is the primary channel you talk to the user through. NOT for SMS / iMessage, phone calls, or FaceTime.
---

# Instant Messaging

**Which app:** § User's if named → the dock's IM app → WeChat (微信) for Chinese users, WhatsApp for English.

Most IM apps share the same chat-page shape: bubble list, bottom input bar, Send. Input/Send/Paste bboxes come from **SYSTEM § Screen layout** — valid only for the IM app it names; in any other app, `peek` and ground live (Spotlight + keyboard keys are system-wide). First-run notice showing → complete the `screen-layout` skill before sending; reading works without it.

## Read

1. **Tap its dock icon** — bottom dock, no labels, so go by the icon, don't search. `open-app` only if it's not in the dock.
2. Chats tab (WeChat: **微信**; WhatsApp: **Chats**).
3. **Tap the contact's row** — the thread, not the list preview (truncated, hides earlier messages).
4. Read at the bottom; top bubble doesn't connect to your last reply → `swipe` down to the first unread.

**Voice messages:** convert-to-text, `peek` the transcript. **Never act unconfirmed** — ASR mishears names, amounts, addresses. Reply with your reading + planned action; wait for OK.

## Send

1. Confirm the contact in the chat header.
2. Keyboard hidden → `tap` the chat input field (`{{input-hidden}}`); it shifts up above the keyboard (`{{input-visible}}`).
3. Stale text in the input → `tap` backspace (`{{backspace}}`) until empty.
4. `send_to_clipboard(text)` → `long_press` the input (`{{input-visible}}`) → `tap` Paste (`{{paste-button}}`).
5. `tap` Send (`{{send}}`) — NOT (+) / voice / camera. (WeChat: keyboard Send key; WhatsApp: round arrow replacing the mic.)
6. Hide the keyboard: `tap` an empty chat area, or `swipe(bbox=[0.300, 0.300, 0.700, 0.500], direction="up", size="s")` — its attached view should show your sent bubble.
7. `go_back` to the chats list — clean state for the next wake.

### Fast path — REQUIRED here

On the right 1:1 chat with clean input, steps 2 + 4–5 are ONE `sequence` call (CONVENTION § Sequence bundling); the attached view shows the sent bubble. **Pick by the current keyboard state — run verbatim, never mix.**

Keyboard **hidden** — the tap raises it; the input moves up to `{{input-visible}}`:

```python
sequence(actions=[
    {"tool_name": "tap",               "arg": {{input-hidden}}},
    {"tool_name": "send_to_clipboard", "arg": "<your text>"},
    {"tool_name": "long_press",        "arg": {{input-visible}}},
    {"tool_name": "tap",               "arg": {{paste-button}}},
    {"tool_name": "tap",               "arg": {{send}}},
])
```

Keyboard **already up** — skip the tap:

```python
sequence(actions=[
    {"tool_name": "send_to_clipboard", "arg": "<your text>"},
    {"tool_name": "long_press",        "arg": {{input-visible}}},
    {"tool_name": "tap",               "arg": {{paste-button}}},
    {"tool_name": "tap",               "arg": {{send}}},
])
```

With the keyboard up, `{{input-hidden}}` is the keyboard itself — never press it.

Then steps 6–7.

### Recovery

Layout shifts (banners, unread badges) → re-ground from the current view; anchors are the header name + learned boxes. Batch failed → single steps, never the same bundle again.

### WeChat: accidental quote

A mis-aimed long-press can quote a message — quoted text under the input shifts every box: layout bboxes invalid, no `sequence`, ground live. Don't chase the tiny ✕; paste and send, the next input is quote-free.
