---
name: build-skill
description: Build an app automation skill by analyzing each screen with all 3 CV tools (color + icons + OCR). Guides the user through the app while extracting fixed/dynamic elements, parameters, and flow steps. Creates a SKILL.md in data/app-skills/.
allowed-tools: Bash, Read, Write, Edit, mcp__physiclaw__park, mcp__physiclaw__screenshot, mcp__physiclaw__detect_elements, mcp__physiclaw__grid_overlay, mcp__physiclaw__bbox_target, mcp__physiclaw__confirm_bbox, mcp__physiclaw__tap, mcp__physiclaw__swipe, mcp__physiclaw__long_press, mcp__physiclaw__propose_bboxes, mcp__physiclaw__wait_for_confirmation, mcp__physiclaw__phone_screenshot, mcp__physiclaw__bridge_status, mcp__physiclaw__bridge_send_text, mcp__physiclaw__bridge_tap
---

# Build App Skill (Meta-Skill)

Create an app automation skill by walking through the app with the user, analyzing each screen, and recording the flow.

**Argument:** A description of what the skill should do (e.g., "帮我建一个美团外卖的技能", "Build a WeChat messaging skill").

## Philosophy

**Think about future executions, not just the current walkthrough.** The skill must work for any item, not just the one used during building. Save fixed element positions. For dynamic content, save layout rules and search strategies.

**Don't mimic humans.** AI should search, not scroll. When the skill needs to find a dynamic item at runtime, use clipboard paste into the app's search box — never scroll through lists.

## Step 1: Understand the task

Ask the user:
- What app? What task?
- What varies between executions? (item name, quantity, specs)
- What's the starting point? (home screen, specific page)

Record the answers. These become the skill's parameters.

## Step 2: Open the app

Use `/open-app {app_name}` or ask the user to navigate to the app's starting screen.

## Step 3: For each screen — analyze

The core loop. For each screen the user navigates to:

### 3a. Capture and analyze

1. `park()` + `screenshot()` to see the current screen
2. `detect_elements()` — runs all 3 CV tools (color segmentation + icon detection + OCR)
3. If `phone_screenshot()` is available (AssistiveTouch set up), take a clean screenshot too and save it as the reference image

### 3b. Classify every element

Present the analysis results to the user in plain language. For each detected element, classify it:

**Fixed elements** (same every visit — save exact bbox):
- Navigation icons (search, back, home, menu)
- CTA buttons ("去结算", "加入购物车", "发送")
- Tab bar items, sidebar categories
- Text input fields
- Toggle switches, checkboxes

**Dynamic elements** (change with content — save rules):
- List items (food cards, chat threads, product cards)
- Search results (content varies)
- Spec options (大杯/中杯 — text varies, need OCR at runtime)

For each fixed element, propose its bbox using `propose_bboxes()` and get the user to confirm. Save confirmed positions.

For dynamic elements, extract layout constants:
- Image column x-position
- Add-button column x-position  
- Card height estimate
- Scrollable region boundaries

### 3c. Confirm with user

Present your classification:
> "This is the restaurant menu page. I found:
> - Fixed: search icon (top-right), back button (top-left), cart button (bottom-right), category tabs (left sidebar)
> - Dynamic: food item list (scrollable, image at x≈0.02, price at x≈0.12, add button at x≈0.88, card height≈0.08)
> 
> The search icon lets us find items by name. That's the path I'll use at runtime."

User confirms or corrects.

### 3d. Plan navigation

Ask: "What do you tap next?" The user taps or tells you. Record the action and the screen transition.

## Step 4: Identify parameters

After walking through all screens, identify:
- What changes between executions? → parameters
- What's the search strategy for each dynamic element?
- What's the expected screen flow? (which screen follows which)

Present to the user:
> "This skill needs parameters: dish name, quantity. At runtime, I'll search for the dish name, select it, set quantity, and checkout."

## Step 5: Plan runtime CV strategy

For each screen, plan how the agent will identify it at runtime:
- **Fingerprint:** distinctive fixed regions (nav bar layout, page title, tab state)
- **Camera matching:** which reference screenshot to compare against
- **Fixed elements:** use saved coordinates directly
- **Dynamic elements:** search-first (clipboard paste) or OCR scan

## Step 6: Save the skill

Create the skill directory and files:

```bash
mkdir -p data/app-skills/{app_slug}/screens
```

Write `data/app-skills/{app_slug}/SKILL.md` with:
- App name and description
- Parameter table
- For each screen:
  - Fingerprint (one-line visual description)
  - Reference screenshot path
  - Fixed element table (Element, Position, Action)
  - Dynamic element rules
  - Navigation action (what to tap, which screen follows)

Save any reference screenshots to `data/app-skills/{app_slug}/screens/`.

## Step 7: Verify

Read back the skill file and confirm with the user that it captures the full workflow correctly.

## Example output

```
data/app-skills/meituan/
├── SKILL.md
└── screens/
    ├── 01_home.png
    ├── 02_restaurant.png
    ├── 03_menu.png
    └── 04_cart.png
```

## Notes

- The user only navigates and confirms. They never draw boxes, specify coordinates, or understand technical concepts.
- Use `propose_bboxes()` for all fixed element position capture — don't guess coordinates.
- If `phone_screenshot()` is available, prefer it over camera screenshots for reference images — much cleaner for future screen matching.
- Always prefer search over scrolling. If the app has a search box, that's the path to use at runtime.
- One skill per task. "Order food on Meituan" and "Check Meituan order status" are two separate skills.
