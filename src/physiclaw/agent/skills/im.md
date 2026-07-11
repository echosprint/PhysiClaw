---
name: im
description: Use for any instant-messaging task — reading the user's chat, sending a reply, or finding a contact by name — in WeChat / 微信 or WhatsApp. Covers "check messages", "reply to <name>", "tell <name> ...". This is the primary channel you talk to the user through. NOT for SMS / iMessage, phone calls, or FaceTime.
---

# Instant Messaging

**Which app:** § User's if named → the dock's IM app → WeChat (微信) for Chinese users, WhatsApp for English.

Most IM apps share the same chat-page shape: bubble list, bottom input bar, Send. Input/Send/Paste bboxes come from **SYSTEM § Screen layout** — valid only for the IM app it names; in any other app, `peek` and ground live (Spotlight + keyboard keys are system-wide). First-run notice showing → complete the `screen-layout` skill before sending; reading works without it.

## Read

1. **Tap its dock icon** — the dock has no labels, so go by the icon; `open-app` only if it's not there.
2. Chats tab (WeChat: **微信**; WhatsApp: **Chats**).
3. **Tap the contact's row** — the thread, not the list preview (truncated, hides earlier messages).
4. Read at the bottom; top bubble doesn't connect to your last reply → `swipe` down to the first unread.

**Voice messages:** convert-to-text, `peek` the transcript. **Never act unconfirmed** — ASR mishears names, amounts, addresses. Reply with your reading + planned action; wait for OK.

## Send

1. Confirm the contact in the chat header.
2. Stale text in the input → raise the keyboard if hidden (`tap` the input, `{{input-hidden}}`), then `tap` backspace (`{{backspace}}`) until empty.
3. Run the fast path below — **REQUIRED on the right 1:1 chat with clean input** (CONVENTION § Sequence bundling); its attached view shows the sent bubble. Send is `{{send}}` — NOT (+) / voice / camera (WeChat: keyboard Send key; WhatsApp: round arrow replacing the mic).
4. Hide the keyboard: `tap` an empty chat area, or `swipe(bbox=[0.300, 0.300, 0.700, 0.500], direction="up", size="s")`.
5. `go_back` to the chats list — clean state for the next wake.

### Fast path — pick by keyboard state, run verbatim, never mix

Keyboard **hidden** — the tap raises it; the input moves up from `{{input-hidden}}` to `{{input-visible}}`:

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

### Recovery

Layout shifts (banners, unread badges) or another IM app → re-ground live from the current view; anchors are the header name + learned boxes. Same actions, one gesture per turn: tap input → clipboard → long-press input → Paste → Send. Batch failed → single steps, never the same bundle again.

### WeChat: accidental quote

A mis-aimed long-press can quote a message — quoted text under the input shifts every box: layout bboxes invalid, no `sequence`, ground live. Don't chase the tiny ✕; paste and send — the next input is quote-free.
