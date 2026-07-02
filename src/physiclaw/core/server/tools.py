"""All MCP tools — the agent's surface for controlling the phone.

Mental model: **See → Act**. Take a photo, pick a bbox, do something there.

Each docstring is consumed twice: the first sentence becomes the inline
`## Tooling` card bullet (via AST parse in `src/physiclaw/agent/engine/mcp_inventory.py`),
and the full docstring becomes the JSONSchema `description` sent over the
native `tools=` API. So the first sentence must carry "what + when" on
its own; param semantics and return shape go in the body below.

All tools are async — blocking hardware I/O runs in a thread pool via
asyncio.to_thread() so the event loop stays free for HTTP routes.

Source order here is wire order. FastMCP registers tools in decorator-
call order (module top-to-bottom), and the MCP `list_tools()` RPC
replays them in that order. The engine concatenates MCP tools + local
tools in `tool_schemas`, so the order in this file directly determines
where each tool sits in the provider's `tools[]` array. LLMs show mild
position bias, so keep the most-reached-for tools (`peek`, `tap`) near
the top and don't shuffle without reason.
"""

import asyncio
from typing import Literal

from mcp.server.fastmcp import FastMCP, Image

from physiclaw.core import PhysiClaw
from physiclaw.core.logger import logged, save_tool_call
from physiclaw.core.server.types import Bbox, ClipboardText


# ─── Trailing hints ───────────────────────────────────────────
#
# Each tool ends with a short hint that nudges the agent toward the
# correct next step. These are part of the prompt contract — silent
# edits change agent behavior with no signal. Tests pin every hint by
# constant, so any rewording surfaces as a failed test.
#
HINT_PEEK_AFTER_TAP = "— `peek` to verify and plan the next move"
HINT_PEEK_AFTER_DOUBLE_TAP = "— `peek` to verify the zoom / selection landed"
HINT_PEEK_AFTER_LONG_PRESS = (
    "— `peek` next: you'll need a fresh bbox from the popover to tap "
    "(Paste / Copy / etc.)"
)
HINT_PEEK_AFTER_SWIPE = "— `peek` to verify the page scrolled and plan the next move"
HINT_PEEK_AFTER_HOME = (
    "— `peek` to plan your next tap on the home-screen icons. If you see the "
    "App Switcher (you were already home), tap an empty area "
    "(bbox=[0.4, 0.9, 0.6, 0.95]) to reach home — don't call `home_screen` again"
)
HINT_PEEK_AFTER_BACK = "— `peek` to verify navigation landed and plan the next move"
HINT_AFTER_FORCE_QUIT = "— now on home screen; reopen the app fresh"
HINT_PEEK_AFTER_UNLOCK = (
    "— `peek` to confirm you're on the home screen and plan the next tap"
)
HINT_AFTER_CLIPBOARD = (
    "— next: `long_press` the target field, then `tap` the Paste "
    "button that appears"
)
HINT_PEEK_AFTER_SEQUENCE = "— `peek` to verify the final state and plan the next move"


def register(mcp: FastMCP, physiclaw: PhysiClaw):
    """Register every MCP tool on the given FastMCP instance."""

    # ─── See ─────────────────────────────────────────────────

    @mcp.tool()
    @logged
    async def peek() -> list:
        """Default view tool — overhead camera (~4s, non-mutating).

        Call BEFORE any tap/swipe to ground the target's bbox; AFTER to
        verify the screen changed. Identical listing across two peeks =
        the screen didn't change.

        Returns: [Image, listing] — JPEG with icon bboxes drawn, plus
        plain-text listing, one row per element:
            id [kind] "label" [left,top,right,bottom] conf
        """
        jpeg, listing = await asyncio.to_thread(physiclaw.peek)
        save_tool_call("peek", listing, jpeg)
        return [Image(data=jpeg, format="jpeg"), listing]

    @mcp.tool()
    @logged
    async def screenshot() -> list:
        """Phone's pixel-perfect capture (~12s, MUTATING) — escalate when peek misses the target.

        Triggers the iOS screenshot gesture; apps observe and react
        (share sheet, similar-items panel, watermarked frames). **Always
        `peek` after `screenshot` before tapping** — screen state may
        have changed.

        Use only when:
          - The target is too small for the camera (`peek` omits it).
          - Camera glare or motion blur makes peek unreadable.
          - You need to read fine print the camera can't resolve.

        Returns: [Image, listing] — same shape as `peek`, but the JPEG
        is the phone's pixel-perfect capture.
        """
        jpeg, listing = await asyncio.to_thread(physiclaw.screenshot)
        save_tool_call("screenshot", listing, jpeg)
        return [Image(data=jpeg, format="jpeg"), listing]

    # ─── Act ─────────────────────────────────────────────────

    @mcp.tool()
    @logged
    async def tap(bbox: Bbox) -> str:
        """Tap once at the bbox center — buttons, links, list items, dismissing dialogs.

        After this, `peek` to verify and ground the next target.
        Identical listing to before = the tap missed (retry once) or the
        bbox was wrong (pick a different element from the new peek).
        """
        result = await asyncio.to_thread(physiclaw.tap, bbox)
        return f"{result} {HINT_PEEK_AFTER_TAP}"

    @mcp.tool()
    @logged
    async def double_tap(bbox: Bbox) -> str:
        """Two quick taps (~150ms apart) at the bbox center — for zoom or word-select.

        Use for zooming maps / photos / web pages, or selecting a word
        in editable text. For buttons, use `tap`.
        """
        result = await asyncio.to_thread(physiclaw.double_tap, bbox)
        return f"{result} {HINT_PEEK_AFTER_DOUBLE_TAP}"

    @mcp.tool()
    @logged
    async def long_press(bbox: Bbox) -> str:
        """Press and hold ~1.2s at the bbox center — context menus, edit mode, paste popover.

        Use for opening context menus, entering icon-rearrange edit
        mode, or triggering the paste popover after `send_to_clipboard`.
        Always `peek` next — you need a fresh bbox from the just-
        revealed popover to tap (Paste / Copy / etc.).
        """
        result = await asyncio.to_thread(physiclaw.long_press, bbox)
        return f"{result} {HINT_PEEK_AFTER_LONG_PRESS}"

    # ─── Swipe ───────────────────────────────────────────────

    @mcp.tool()
    @logged
    async def swipe(
        bbox: Bbox,
        direction: Literal["up", "down", "left", "right"],
        size: Literal["s", "m", "l", "xl", "xxl"] = "m",
        speed: Literal["slow", "medium", "fast"] = "medium",
    ) -> str:
        """Slide the stylus across `bbox` in `direction` (1–8cm) — for scrolling, paging, dismissing cards, opening Control / Notification Center.

        After this, `peek` to verify the page scrolled.

        **Direction is the STYLUS motion, not the page motion:**
          - swipe UP   → page scrolls DOWN (reveals content below)
          - swipe DOWN → page scrolls UP (opens Notification Center)
          - swipe LEFT → page scrolls RIGHT (next photo / next page)

        Args:
            bbox: gesture origin (0–1 decimals).
            direction: stylus motion — `up` / `down` / `left` / `right`.
            size: stroke length (default `m`):
                `s`   ≈ 1cm — small nudge
                `m`   ≈ 2cm — most scrolls
                `l`   ≈ 4cm — long scroll
                `xl`  ≈ 6cm — page-sized scroll
                `xxl` ≈ 8cm — full-screen (Control / Notification Center)
            speed: stylus velocity (default `medium`). Faster strokes
                coast further on iOS scroll lists.
        """
        result = await asyncio.to_thread(physiclaw.swipe, bbox, direction, size, speed)
        return f"{result} {HINT_PEEK_AFTER_SWIPE}"

    # ─── Navigate ────────────────────────────────────────────

    @mcp.tool()
    @logged
    async def home_screen() -> str:
        """Return to the iPhone home screen — exit any app to a known launch pad.

        Issues the iPhone swipe-up-from-bottom gesture. Use to start a
        fresh task or recover from getting lost in app navigation. If
        ALREADY on the home screen, this opens the App Switcher instead.
        """
        result = await asyncio.to_thread(physiclaw.home_screen)
        return f"{result} {HINT_PEEK_AFTER_HOME}"

    @mcp.tool()
    @logged
    async def go_back() -> str:
        """Pop back one screen via the iPhone left-edge swipe gesture.

        Works in apps with a navigation stack (most do). Same screen
        after `peek` = either the gesture didn't register (retry once)
        or this screen has no back action (modals, root tabs, lock
        screen) — try `home_screen` and re-enter, or tap an in-screen
        `<` / Back button.

        Trap — full-screen image viewers (product images, Messages /
        WeChat photos): the viewer reclaims left/right swipes for image
        navigation, so edge-swipe won't pop. Close via the in-viewer
        `X` / `Done` button instead.
        """
        result = await asyncio.to_thread(physiclaw.go_back)
        return f"{result} {HINT_PEEK_AFTER_BACK}"

    @mcp.tool()
    @logged
    async def force_quit() -> str:
        """Force-quit the current app via the iOS app-switcher gesture (~7s) — hard reset when you're stuck on the wrong app page and can't find the entry you need from here.

        Lands on home screen. Reopen the app fresh from there.

        Use after `go_back` hasn't gotten you to the right entry point —
        when popups won't dismiss, the back stack loops, or the wrong
        page keeps returning.
        """
        result = await asyncio.to_thread(physiclaw.force_quit)
        return f"{result} {HINT_AFTER_FORCE_QUIT}"

    @mcp.tool()
    @logged
    async def unlock_phone() -> str:
        """Unlock the phone with passcode `111111` (~12s).

        Wakes the screen, swipes up, waits for Face ID to fail, OCRs
        the keypad, taps each digit. Hardcoded to `111111` — a
        throwaway tool-phone code so a real password never leaks via
        git or logs.

        On failure, don't retry — every other tool needs the phone
        unlocked. Close out:
          1. `append_log("[HH:MM] tried unlock_phone — failed")`
          2. `end_session("STUCK", "phone unlock failed — user needs
             passcode 111111 or auto-lock disabled")`

        User fix: set passcode to `111111`, or disable auto-lock
        (Settings → Display & Brightness → Auto-Lock → Never;
        trade-off: faster display wear).
        """
        result = await asyncio.to_thread(physiclaw.unlock_phone)
        return f"{result} {HINT_PEEK_AFTER_UNLOCK}"

    # ─── Text ────────────────────────────────────────────────

    @mcp.tool()
    @logged
    async def send_to_clipboard(text: ClipboardText) -> str:
        """Copy `text` to the phone's clipboard — paste is far faster than typing.

        Standard flow:
          1. `send_to_clipboard(text)`
          2. `long_press(field_bbox)` — opens the paste popover.
          3. `tap` the `Paste` (or `粘贴`) button that appears.

        Fall back to the keyboard only if the field rejects paste
        (passcode fields, some search bars).
        """
        result = await asyncio.to_thread(physiclaw.send_to_clipboard, text)
        return f"{result} {HINT_AFTER_CLIPBOARD}"

    # ─── Sequence ────────────────────────────────────────────

    @mcp.tool()
    @logged
    async def sequence(
        step1: dict,
        step2: dict | None = None,
        step3: dict | None = None,
        step4: dict | None = None,
        step5: dict | None = None,
    ) -> str:
        """Run up to 5 actions in one call — saves turns on deterministic flows.

        Use when intermediate observations would add nothing (opening
        an app via tap → tap → tap, or paste + send an IM message).
        Stops at the first failure; earlier steps are NOT rolled back —
        `peek` after a failure to see where it landed before retrying.

        DON'T use `sequence` when:
          - You'd want to confirm the screen changed between steps
            (dialogs that may or may not appear).
          - The target's bbox is uncertain — verify with `peek` first.

        Each step is a dict with two fields:
            tool_name: `tap` / `double_tap` / `long_press` / `swipe` /
                       `send_to_clipboard`.
            arg:       that tool's argument — Bbox for tap/double_tap/
                       long_press; `{bbox, direction, size?, speed?}`
                       for swipe; string for send_to_clipboard.

        Args:
            step1: first action (required).
            step2-5: additional actions (optional, run in order).
        """
        steps = [s for s in (step1, step2, step3, step4, step5) if s is not None]
        result = await asyncio.to_thread(physiclaw.sequence, steps)
        return f"{result}\n{HINT_PEEK_AFTER_SEQUENCE}"
