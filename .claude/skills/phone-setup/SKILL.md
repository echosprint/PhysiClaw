---
name: phone-setup
description: Guide user step-by-step to set up their iPhone for PhysiClaw — enable AssistiveTouch, create three iOS Shortcuts (take screenshot, upload latest, clipboard sync).
allowed-tools: Bash, Read
---

# iPhone Setup for PhysiClaw

Set up three iOS Shortcuts (take screenshot · upload latest · clipboard sync) and bind them to AssistiveTouch taps. The server must be running.

## Step 1: Stable `<name>.local` hostname

Shortcuts that embed the LAN IP break when DHCP changes it. `<name>.local` survives IP changes on the same Wi-Fi.

Skip if the current name already ends with `-xxx` (3 lowercase letters) and `ping` succeeds:

```bash
CUR=$(scutil --get LocalHostName) && echo "$CUR" && ping -c 1 -W 1000 "${CUR}.local"
```

Otherwise rename (strips either our old `-xxx` suffix or macOS's numeric collision suffix like `-2`/`-3`; sets `LocalHostName` only — `ComputerName`/`HostName` don't drive mDNS):

```bash
BASE=$(scutil --get LocalHostName | sed -E 's/-([a-z]{3}|[0-9]+)$//') && \
NAME="${BASE}-$(LC_ALL=C tr -dc 'a-z' </dev/urandom | head -c 3)" && \
sudo scutil --set LocalHostName "$NAME" && \
dscacheutil -flushcache && \
echo "Set to: $NAME"
```

Shortcut URLs use the lowercase form (`macair-qqd.local`).

## Step 2: Server URLs

```bash
uv run python -c "
from physiclaw.bridge import bridge_base_urls
p, f = bridge_base_urls(8048)
if p != f: print(f'Recommended: {p}')
print(f'Fallback (IP): {f}')
"
```

Use `Recommended` in the Shortcuts. Fallback to the IP only if the network blocks mDNS.

## Step 3: "PhysiClaw Tap" Shortcut (take screenshot)

Tell the user:

> Shortcuts app → **+**:
>
> 1. Add **"Take Screenshot"**
> 2. Rename to **"PhysiClaw Tap"** → Done

Wait for confirmation.

## Step 4: "PhysiClaw Screenshot" Shortcut (upload latest)

Tell the user (replace `HOST` with the URL from Step 2):

> Shortcuts app → **+**:
>
> 1. Add **"Get Latest Screenshots"**
> 2. Add **"Get Contents of URL"**:
>    - URL: `http://HOST/api/bridge/screenshot`
>    - Show More → Method **POST**, Request Body **File**, File → **Screenshots** variable
> 3. Rename to **"PhysiClaw Screenshot"** → Done

Wait for confirmation.

## Step 5: "PhysiClaw Clipboard" Shortcut (sync clipboard)

Tell the user (same `HOST`):

> Shortcuts app → **+**:
>
> 1. Add **"Get Contents of URL"** → URL `http://HOST/api/bridge/clipboard` (GET)
> 2. Add **"Copy to Clipboard"** → input = **Contents of URL**
> 3. Rename to **"PhysiClaw Clipboard"** → Done

Wait for confirmation.

## Step 6: AssistiveTouch (skip if configured)

Tell the user:

> Settings → Accessibility → Touch → **AssistiveTouch ON**. Custom Actions:
>
> - **Single-Tap** → Shortcut → PhysiClaw Tap
> - **Double-Tap** → Shortcut → PhysiClaw Screenshot
> - **Long Press** → Shortcut → PhysiClaw Clipboard

Wait for confirmation.

## Step 7: Test

Tell the user:

> Tap AssistiveTouch **once** (take screenshot), then **twice** (upload latest).

Record the baseline, then poll for a new file:

```bash
baseline=$(ls -t data/phone/screenshot/ 2>/dev/null | head -1)
for i in $(seq 1 25); do
  newest=$(ls -t data/phone/screenshot/ 2>/dev/null | head -1)
  [ -n "$newest" ] && [ "$newest" != "$baseline" ] && { echo "✓ $newest"; open "data/phone/screenshot/$newest"; exit 0; }
  sleep 1
done
echo "✗ no upload in 25s"
```

If nothing arrives:

- Run the Shortcut manually (Shortcuts app → ▶) — check for errors.
- From the Mac, `ping "$(scutil --get LocalHostName).local"` — if it fails from the phone's network, mDNS is blocked; swap both Shortcut URLs to the IP fallback from Step 2.
- Ensure at least one screenshot exists in Photos.

## Done

> AssistiveTouch: single tap = take screenshot, double tap = upload latest, long press = clipboard fetch. If a Shortcut breaks after a network change, rerun this setup to get the new URL and paste it into each Shortcut's "Get Contents of URL" step.
