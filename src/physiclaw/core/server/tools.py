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

Layering: each tool body is ONE core call + ONE reply step (a
`_*_reply` adapter, or a trailing hint on text-only tools) — no other
logic. The core stays transport-agnostic (it returns
bytes/str/GestureResult, never MCP types); this module is the MCP
boundary, so wire shapes, the `Image` type, and the hint text they
carry live here and nowhere deeper.
"""

import asyncio
from typing import Literal

from mcp.server.fastmcp import FastMCP, Image

from physiclaw.common.dumps import save_tool_call
from physiclaw.common.logger import logged
from physiclaw.core import PhysiClaw
from physiclaw.core.orchestration import GestureResult
from physiclaw.core.server.types import Bbox, ClipboardText, SequenceActions

# ─── Trailing hints ───────────────────────────────────────────
#
# Each tool ends with a short hint that nudges the agent toward the
# correct next step. These are part of the prompt contract — silent
# edits change agent behavior with no signal. Tests pin every hint by
# constant, so any rewording surfaces as a failed test.
#
# Gesture tools carry their own fused post-gesture view, so their hints
# say "read the attached view" — a hint that said "`peek` to verify"
# would cause the exact wasted turn the fused view removes.
#
HINT_VIEW_AFTER_TAP = (
    "— read the attached view: did the value/state you targeted change?"
)
HINT_VIEW_AFTER_DOUBLE_TAP = "— read the attached view: did the zoom / selection land?"
HINT_VIEW_AFTER_LONG_PRESS = (
    "— the attached view shows the revealed popover / menu; tap your "
    "target from its listing"
)
HINT_VIEW_AFTER_SWIPE = "— the attached view shows the page after the swipe"
HINT_VIEW_AFTER_HOME = (
    "— the attached view shows where you landed. App Switcher visible "
    "(you were already home)? Tap an empty area "
    "(bbox=[0.4, 0.9, 0.6, 0.95]) — don't call `home_screen` again"
)
HINT_VIEW_AFTER_BACK = "— check the attached view: did navigation pop?"
HINT_VIEW_AFTER_FORCE_QUIT = (
    "— check the attached view: home screen means done, reopen the app "
    "fresh; anything else means the gesture missed — recover from there"
)
HINT_VIEW_AFTER_UNLOCK = "— check the attached view: home screen means unlocked"
HINT_AFTER_CLIPBOARD = (
    "— now paste it. Target field's Paste box learned (SYSTEM § Screen "
    "layout) or already grounded? Bundle `long_press` + `tap` Paste (+ the "
    "next grounded taps) in ONE `sequence`. Otherwise: `long_press` the "
    "field, then tap Paste from its attached view"
)
HINT_VIEW_AFTER_SEQUENCE = "— the attached view shows the state after the final step"

# Appended instead of a view when the camera/detector hiccupped.
NO_VIEW_FALLBACK = "(auto-view unavailable — `peek` to see the screen)"


def _view_reply(name: str, jpeg: bytes, listing: str) -> list:
    """`[Image, listing]` — the wire shape every screen view ships in —
    and dump the pair to the tool-call log."""
    save_tool_call(name, listing, jpeg)
    return [Image(data=jpeg, format="jpeg"), listing]


def _gesture_reply(name: str, res: GestureResult, hint: str) -> str | list:
    """GestureResult → MCP reply: `[result, image, listing]` when the
    fused view is present (result = action text with verdict + hint,
    first per prompting guidance), else a text-only fallback that
    routes the agent to `peek`."""
    text = f"{res.text} {hint}"
    if res.jpeg is None or res.listing is None:
        return f"{text} {NO_VIEW_FALLBACK}"
    return [text, *_view_reply(name, res.jpeg, res.listing)]


def register(mcp: FastMCP, physiclaw: PhysiClaw) -> None:
    """Register every MCP tool on the given FastMCP instance."""

    # ─── See ─────────────────────────────────────────────────

    @mcp.tool(structured_output=False)
    @logged
    async def peek() -> list:
        """Fresh look at the screen (~4s, camera, non-mutating) — orient when you have no current view.

        Gestures attach their own post-action view, so you rarely need
        `peek` right after acting. Use it: for the wake's first look,
        after `wait`, when a gesture's auto-view failed or caught a
        loading state, and after `screenshot` (side-effect check).

        Returns `[image, listing]` — JPEG with icon bboxes drawn, plus
        plain-text listing, one row per element:
            id [kind] "label" [left,top,right,bottom] conf
        """
        jpeg, listing = await asyncio.to_thread(physiclaw.peek)
        return _view_reply("peek", jpeg, listing)

    @mcp.tool(structured_output=False)
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

        Returns `[image, listing]` — same shape as `peek`, but the JPEG
        is the phone's pixel-perfect capture.
        """
        jpeg, listing = await asyncio.to_thread(physiclaw.screenshot)
        return _view_reply("screenshot", jpeg, listing)

    # ─── Act ─────────────────────────────────────────────────

    @mcp.tool(structured_output=False)
    @logged
    async def tap(bbox: Bbox) -> list | str:
        """Tap once at the bbox center — buttons, links, list items, dismissing dialogs.

        Returns `[result, image, listing]` — the fresh post-tap view is
        ATTACHED (~2s after the tap), so you normally don't need a
        separate `peek`. The result also carries a camera verdict:
        `screen: changed` or `screen: no visible change`. On no visible
        change, READ the target value in the attached view first — a
        real change can be too small for the camera. Value unmoved = the
        tap missed (retry ONCE) or the app refused with a toast you can
        never see — stock limit, cap, disabled control (PHYSICLAW §
        Unchanged screen). Unmoved after the retry = refusal; stop
        tapping it.
        """
        res = await asyncio.to_thread(physiclaw.tap, bbox)
        return _gesture_reply("tap", res, HINT_VIEW_AFTER_TAP)

    @mcp.tool(structured_output=False)
    @logged
    async def double_tap(bbox: Bbox) -> list | str:
        """Two quick taps (~150ms apart) at the bbox center — for zoom or word-select.

        Use for zooming maps / photos / web pages, or selecting a word
        in editable text. For buttons, use `tap`. Returns
        `[result, image, listing]` — the post-gesture view is attached.
        """
        res = await asyncio.to_thread(physiclaw.double_tap, bbox)
        return _gesture_reply("double_tap", res, HINT_VIEW_AFTER_DOUBLE_TAP)

    @mcp.tool(structured_output=False)
    @logged
    async def long_press(bbox: Bbox) -> list | str:
        """Press and hold ~1.2s at the bbox center — context menus, edit mode, paste popover.

        Use for opening context menus, entering icon-rearrange edit
        mode, or triggering the paste popover after `send_to_clipboard`.
        Returns `[result, image, listing]` — the attached view shows the
        just-revealed popover; tap Paste / Copy from its listing
        directly, no extra `peek` needed.
        """
        res = await asyncio.to_thread(physiclaw.long_press, bbox)
        return _gesture_reply("long_press", res, HINT_VIEW_AFTER_LONG_PRESS)

    # ─── Swipe ───────────────────────────────────────────────

    @mcp.tool(structured_output=False)
    @logged
    async def swipe(
        bbox: Bbox,
        direction: Literal["up", "down", "left", "right"],
        size: Literal["s", "m", "l", "xl", "xxl"] = "m",
        speed: Literal["slow", "medium", "fast"] = "medium",
    ) -> list | str:
        """Slide the stylus across `bbox` in `direction` (1–8cm) — for scrolling, paging, dismissing cards, opening Control / Notification Center.

        Returns `[result, image, listing]` — the attached view shows the
        page after the swipe; unchanged = end of the list.

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
        res = await asyncio.to_thread(physiclaw.swipe, bbox, direction, size, speed)
        return _gesture_reply("swipe", res, HINT_VIEW_AFTER_SWIPE)

    # ─── Navigate ────────────────────────────────────────────

    @mcp.tool(structured_output=False)
    @logged
    async def home_screen() -> list | str:
        """Return to the iPhone home screen — exit any app to a known launch pad.

        Issues the iPhone swipe-up-from-bottom gesture. Use to start a
        fresh task or recover from getting lost in app navigation. If
        ALREADY on the home screen, this opens the App Switcher instead
        — the attached view shows which one you got.
        """
        res = await asyncio.to_thread(physiclaw.home_screen)
        return _gesture_reply("home_screen", res, HINT_VIEW_AFTER_HOME)

    @mcp.tool(structured_output=False)
    @logged
    async def go_back() -> list | str:
        """Pop back one screen via the iPhone left-edge swipe gesture.

        Works in apps with a navigation stack (most do). Attached view
        unchanged = either the gesture didn't register (retry once)
        or this screen has no back action (modals, root tabs, lock
        screen) — try `home_screen` and re-enter, or tap an in-screen
        `<` / Back button.

        Trap — full-screen image viewers (product images, Messages /
        WeChat photos): the viewer reclaims left/right swipes for image
        navigation, so edge-swipe won't pop. Close via the in-viewer
        `X` / `Done` button instead.
        """
        res = await asyncio.to_thread(physiclaw.go_back)
        return _gesture_reply("go_back", res, HINT_VIEW_AFTER_BACK)

    @mcp.tool(structured_output=False)
    @logged
    async def force_quit() -> list | str:
        """Force-quit the current app via the iOS app-switcher gesture (~8s) — hard reset when you're stuck on the wrong app page and can't find the entry you need from here.

        Normally lands on the home screen — CHECK the attached view
        before reopening the app: if it shows the app switcher or the
        same app page, the gesture missed; recover from what you see.

        Use after `go_back` hasn't gotten you to the right entry point —
        when popups won't dismiss, the back stack loops, or the wrong
        page keeps returning.
        """
        res = await asyncio.to_thread(physiclaw.force_quit)
        return _gesture_reply("force_quit", res, HINT_VIEW_AFTER_FORCE_QUIT)

    @mcp.tool(structured_output=False)
    @logged
    async def unlock_phone() -> list | str:
        """Unlock the phone with passcode `111111` (~20-40s).

        Wakes the screen, swipes up, waits out the Face-ID attempt, OCRs
        the keypad, and taps each digit.
        Hardcoded to `111111` — a throwaway tool-phone code so a real
        password never leaks via git or logs.

        The passcode keypad only stays up a few seconds — a slow
        detection loses the race and the phone falls back to the lock
        screen. Confirm from the attached view: if it still shows the
        lock screen (or the keypad was never found), retry `unlock_phone`
        once or twice — a retry starts with the screen awake, so it is
        faster and usually lands. If retries keep failing on a phone that
        is clearly locked, close out:
          1. `append_log("[HH:MM] tried unlock_phone — failed")`
          2. `end_session("STUCK", "phone unlock failed — user needs
             passcode 111111 or auto-lock disabled")`

        User fix: set passcode to `111111`, or disable auto-lock
        (Settings → Display & Brightness → Auto-Lock → Never;
        trade-off: faster display wear).
        """
        res = await asyncio.to_thread(physiclaw.unlock_phone)
        return _gesture_reply("unlock_phone", res, HINT_VIEW_AFTER_UNLOCK)

    # ─── Text ────────────────────────────────────────────────

    @mcp.tool(structured_output=False)
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

    @mcp.tool(structured_output=False)
    @logged
    async def sequence(actions: SequenceActions) -> list | str:
        """Run 2-5 gestures in one call — REQUIRED when every target box is already grounded and no decision is needed between steps.

        Issuing such a run as single-gesture turns is a planning error,
        not caution: each extra turn costs ~10s and context. Grounded =
        from SYSTEM § Screen layout, the current listing, or a
        scratchpad copy (CONVENTION § Sequence bundling).

        Runs in order; stops at the first failure; earlier steps are NOT
        rolled back. The result lists per-step ok/FAIL, a whole-batch
        `screen:` verdict, and ATTACHES the post-batch view — that view
        is your verification (and, after a mid-batch FAIL, shows where
        it landed).

        DON'T bundle:
          - A step whose target is born mid-batch (a popover from a
            long_press) — unless that Paste box is learned in SYSTEM §
            Screen layout.
          - Payment / order-confirm taps — always their own turn, off a
            fresh view. (Sending a confirmed IM to your user IS
            bundleable.)
          - Anything you'd want to see the screen between.

        Each action is a dict with two fields:
            tool_name: `tap` / `double_tap` / `long_press` / `swipe` /
                       `send_to_clipboard`.
            arg:       that tool's argument — Bbox for tap/double_tap/
                       long_press; `{bbox, direction, size?, speed?}`
                       for swipe; string for send_to_clipboard.
        """
        res = await asyncio.to_thread(physiclaw.sequence, actions)
        return _gesture_reply("sequence", res, f"\n{HINT_VIEW_AFTER_SEQUENCE}")
