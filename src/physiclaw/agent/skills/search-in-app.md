---
name: search-in-app
description: Use when typing a query into a search box / search field inside any app — JD, Taobao, Meituan, App Store, Spotlight, Settings, etc. Covers finding the box → focus → clear stale text → paste → submit. NOT for sending IM messages (use the `im` skill).
---

# Search in App

## The search box

A rounded bar: magnifier icon left, gray placeholder text in the middle, camera / mic / Search button right. **The placeholder text's row IS the field — target its bbox**; the edge icons are decoys (camera = photo-search, right-edge button = submit).

- **Home-page bars are buttons** (most shopping apps): the first tap opens the real search page — re-ground the input there (the placeholder word repeats as a chip below; take the topmost row).
- **The placeholder is a live, rotating ad query**: submitting an empty field searches it — never tap search/return until YOUR text is in the box.

## Steps

1. `send_to_clipboard(text)`.
2. `tap` the search box. Keyboard up = focused.
3. **Stale text?** Gray text is the placeholder — it vanishes on paste. Dark text + `✕` = a stale query: `tap` the `✕`, else backspace (`{{backspace}}`) 10–20× (`sequence`s of 5; over-tap freely) — paste does NOT replace existing text.
4. `long_press` the box — **its OWN turn, never in a `sequence`.** Long-press, then **LOOK**: ground Paste from the attached view.
5. One `sequence`: `tap` Paste (now that you see it) + `tap` return/search (`{{return}}`).
6. Verify in the attached view: results rendered AND the query header shows YOUR text — the placeholder word instead = the paste missed; `go_back`, redo from step 4.

## Pitfalls

- **Auto-suggest** — return submits your typed string; a tapped suggestion searches the SUGGESTION.
- **AI-search toggle** (`AI搜索` / sparkle icon beside the bar) — NOT the field; tapping enters AI-answer mode. `go_back`, target the plain field.
- **Keyboard didn't rise** — the field may already be focused; tap once more, or tap elsewhere then re-tap.
