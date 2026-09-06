# taobao

Buys one item on 淘宝 (Taobao) by keyword: search, pick by criteria,
open the buy sheet, ask the user to approve the exact total, pay with
one tap, report. The boot offers `taobao/buy` for a purchase request
in the user's thread.

## Device

Recorded on an iPhone with the system in English and Taobao in
Chinese. The pack's pages (`home`, `results`, `buysheet`, `paid`) are
declared by anchors only; run `physiclaw playbooks pages calibrate
taobao` on your phone so one OCR miss does not read a page unknown.

## Traps

- `force_quit` kills the foreground app, so `launch` raises Taobao
  before quitting it; a resumed app lands mid-state and fails verify.
- Two-character anchors (综合, 销量) match feed text unless pinned
  `within: top`.
- The detail footer's third icon is 收藏, not the cart; the route never
  touches the cart, it buys off the 领券购买 / 立即购买 sheet.
- Promo popups (天降红包, coupons) cover a page right after it opens;
  `landmarks.close` is their ✕, never their buttons.
- The buy sheet shows `￥…起` until a spec is chosen, and an option row
  tap sometimes needs a second tap.
