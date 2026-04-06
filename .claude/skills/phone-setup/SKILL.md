---
name: phone-setup
description: Guide user step-by-step to set up their iPhone for PhysiClaw — enable AssistiveTouch, create iOS Shortcuts for screenshot upload and clipboard sync.
allowed-tools: Bash, Read
---

# iPhone Setup for PhysiClaw

Walk the user through setting up their iPhone with two iOS Shortcuts (screenshot upload + clipboard sync) and AssistiveTouch. No hardware or calibration needed — just the phone and the running server.

## Step 1: Get the server URL

```bash
uv run python -c "from physiclaw.bridge import get_lan_ip; print(f'http://{get_lan_ip()}:8048')"
```

Save this IP for later. Tell the user:

> Make sure your iPhone is on the **same WiFi** as this computer.

## Step 2: Create the iOS Shortcut

Tell the user (replace `SERVER_IP:8048` with the actual value from Step 1):

> Open the **Shortcuts** app on your iPhone:
>
> 1. Tap **+** to create a new shortcut
> 2. Add action: search **"Get Latest Screenshots"** — add it
> 3. Add action: search **"Get Contents of URL"** — add it, then configure:
>    - **URL**: `http://SERVER_IP:8048/api/bridge/screenshot`
>    - Tap **Show More**
>    - **Method**: **POST**
>    - **Request Body**: **File**
>    - **File**: tap and select the **Screenshots** variable from the previous step
> 4. Rename the shortcut to **"PhysiClaw Screenshot"**
> 5. Tap **Done**

Wait for user confirmation.

## Step 3: Create the Clipboard Shortcut

Tell the user (replace `SERVER_IP:8048` with the actual value from Step 1):

> Create a second shortcut in the **Shortcuts** app:
>
> 1. Tap **+** to create a new shortcut
> 2. Add action: search **"Get Contents of URL"** — add it, then configure:
>    - **URL**: `http://SERVER_IP:8048/api/bridge/clipboard`
>    - Method stays **GET** (default)
> 3. Add action: search **"Copy to Clipboard"** — add it
>    - Input: select the **Contents of URL** variable from the previous step
> 4. Rename the shortcut to **"PhysiClaw Clipboard"**
> 5. Tap **Done**

This shortcut lets the server send text directly to the phone's clipboard — no screen tap needed.

Wait for user confirmation.

## Step 4: Enable AssistiveTouch (skip if already configured)

Tell the user:

> On your iPhone:
>
> 1. Open **Settings**
> 2. Go to **Accessibility** → **Touch** → **AssistiveTouch**
> 3. Turn **AssistiveTouch ON** — a floating circle button appears on screen
> 4. Under **Custom Actions**, set:
>    - **Single-Tap** → **Screenshot**
>    - **Double-Tap** → **Shortcut** → select **"PhysiClaw Screenshot"**
>    - **Long Press** → **Shortcut** → select **"PhysiClaw Clipboard"**

Wait for user confirmation.

## Step 5: Test the upload

Tell the user:

> 1. Tap AssistiveTouch **once** to take a screenshot
> 2. Tap AssistiveTouch **twice** to upload the latest screenshot to PhysiClaw
>
> I'll check if it arrived.

Wait a moment, then check:

```bash
ls -lt data/phone/screenshot/ 2>/dev/null | head -5
```

If files exist, show the latest:

```bash
uv run python -c "
from pathlib import Path
files = sorted(Path('data/phone/screenshot').glob('*'), key=lambda f: f.stat().st_mtime, reverse=True)
if files:
    f = files[0]
    print(f'Screenshot received: {f.name} ({f.stat().st_size:,} bytes)')
    print(f'Path: {f}')
else:
    print('No screenshots found in data/phone/screenshot/')
"
```

Open the image to show the user:

```bash
open data/phone/screenshot/LATEST_FILENAME
```

If no screenshot arrived, troubleshoot:

- Open Shortcuts → tap "PhysiClaw Screenshot" → tap ▶ to run manually. Any errors?
- Check the URL matches `http://SERVER_IP:8048/api/bridge/screenshot`
- Try opening `http://SERVER_IP:8048/bridge` in Safari — if it loads, the network works
- Make sure there is at least one screenshot in the Photos app

Retry until a screenshot arrives and is displayed.

## Done

Tell the user:

> iPhone setup complete! AssistiveTouch actions:
>
> - **Single tap** → takes a screenshot (saved to Photos)
> - **Double tap** → uploads the latest screenshot to the server
> - **Long press** → fetches text from server to clipboard (for pasting into apps)
