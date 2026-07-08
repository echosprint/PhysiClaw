---
name: jd
description: PLACEHOLDER — Use when shopping on the JD (京东) app: searching for a product, comparing listings, adding to cart, and checking out. Load before acting in 京东. NOT yet implemented — the flow below is a stub to be filled in from real screens.
---

# 京东 (JD) — placeholder

> **Status: placeholder.** This skill is a stub so the `./skills` sync and
> `physiclaw skills install` pipeline have something to carry. The flow,
> element tables, and reference screenshots below are TODO — capture them
> from the real app before relying on this.

## Parameters

| Name | Description | Example |
|------|-------------|---------|
| item_name | What to buy | 无线鼠标 |

## Flow

### Screen 1: Home (jd_home)

Fingerprint: TODO — one-line visual description for screen matching.
Reference: screens/01_home.png (TODO)

**Fixed elements** (same position every visit — tap directly):

| Element | Position | Action |
|---------|----------|--------|
| Search bar | TODO `[l,t,r,b]` | → Search |

**Action:** TODO — tap the search bar to reach the Search screen.

### Screen 2: Search results (jd_search)

TODO — search-first: paste `item_name`, submit, then pick a listing.

### Screen 3: Product / Cart / Checkout

TODO — add to cart, review item + price + address + fees, then
**confirm with the user before paying**.

## Notes

- Paste over typing: `send_to_clipboard(item_name)` → `long_press` the
  search field → tap **粘贴 / Paste**.
- Read prices/specs exactly as shown; never round or guess.
- Confirm before payment.
