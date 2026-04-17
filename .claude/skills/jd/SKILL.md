---
name: jd
description: Shop groceries via 京东七鲜 (JD 7Fresh) on the 京东 app. Handles the add-to-cart-returns-to-item quirk and the screenshot share-popup trap.
---

# JD (京东) — Grocery shopping

Use **京东七鲜** (JD 7Fresh) for groceries. Other JD categories need explicit owner ask.

## Flow

1. `/open-app 京东` (or `JD`), tap into 京东七鲜.
2. Search the item, open its shop page.
3. Tap 加入购物车. The app returns to the item page — **don't tap again**; the item is already in the cart.
4. Tap the cart icon on the page, review line items, tap 去结算.
5. Send the owner: item, qty, price, address, fees, ETA. Wait for explicit OK (CLAUDE.md → Rules → Confirm before payment).
6. Tap 提交订单 / 立即支付.

## Gotcha — screenshot triggers share popup

Taking a `screenshot()` of a shop item page may overlay a 分享截屏 (Share screenshot) menu with targets 朋友圈 / QQ / 微信好友 / 保存图片 / 搜问题. If `scan()` after a screenshot doesn't confirm the item UI, a popup is covering it. Dismiss by tapping the dimmed top/left/right edge — see `memory/memory.md` → UI patterns.
