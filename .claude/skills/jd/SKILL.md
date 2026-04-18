---
name: jd
description: Shop groceries via 京东七鲜 (JD 7Fresh) on the 京东 app. Handles the add-to-cart-returns-to-item quirk and the screenshot share-popup trap.
---

# JD (京东) — Grocery shopping

Use **京东七鲜** (JD 7Fresh) for groceries. Other JD categories need explicit owner ask.

## Tool choice

Prefer `scan()` and `peek()` — cheap, no app-side reactions. Reach for `screenshot()` only when the target is icon-only (no text label). **`screenshot()` on a commodity detail page triggers the share overlay** — see Gotcha.

## Flow

1. `/open-app 京东` (or `JD`). Make sure you're on the **首页** tab (top nav), then tap into 京东七鲜.
2. Tap the search box. If it has stale text, tap **backspace** at `[0.867, 0.804, 0.994, 0.857]` (iOS keyboard, Z-row far right) until empty. Type/paste the item, open its shop page.
3. Tap 加入购物车. The app returns to the item page — **don't tap again**; the item is in the cart.
4. Tap the cart icon, review line items, tap 去结算.
5. Send the owner: item, qty, price, address, fees, ETA. Wait for explicit OK.
6. Tap 提交订单 / 立即支付.

## Gotcha — screenshot triggers share popup

JD intercepts the iOS screenshot gesture and overlays a 分享截屏 menu (朋友圈 / QQ / 微信好友 / 保存图片 / 搜问题). The screenshot still captured the real page — recover without re-shooting:

1. `scan()` — baseline.
2. `screenshot()` — capture bboxes.
3. `scan()` — if it differs (share-sheet text + dim area), the popup is covering the page.
4. Dismiss the popup (see below).
5. `scan()` — should match the baseline.
6. **Use the original screenshot's bboxes** for the next tap. Saves ~12s vs retaking.

## Dismiss popup / share sheet / bottom sheet

Tap the **dimmed area** above the popup. Safe target: `[0.05, 0.40, 0.15, 0.60]` (center ≈ x=0.10, y=0.50 — left edge, vertical middle). The mirror bbox on the right edge `[0.85, 0.40, 0.95, 0.60]` works too.
