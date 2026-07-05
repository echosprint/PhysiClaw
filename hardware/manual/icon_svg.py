"""Inline SVG assets for the manual.

These are small, hand-tuned decorations that never change per page. They live
here as plain string constants so ``manual.py`` stays focused on layout:

- ``COVER_STRIPES_SVG`` / ``BACK_CORNER_SVG`` — the corner wedges on the front
  and back covers, emitted inline.
- ``GITHUB_OCTICON_SVG`` — the GitHub mark on the introduction page, emitted inline.
- ``CRAB_SVG`` — the PhysiClaw logo. Referenced as an external ``crab.svg``
  (cover mark, intro Docs logo, back-cover footmark); ``manual.py`` writes this
  string into the output directory at build time.

All are byte-identical to the originals in the hand-written manual.html / crab.svg.
"""

# Cover: a single solid triangle anchored in the bottom-left corner.
COVER_STRIPES_SVG = """\
<svg viewBox="0 0 400 500" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMinYMax meet">
        <path d="M 0,500 L 0,150 L 300,500 Z" fill="#d4511a"/>
      </svg>"""

# Intro: the GitHub octicon mark.
GITHUB_OCTICON_SVG = (
    '<svg viewBox="0 0 98 96" aria-hidden="true"><path fill-rule="evenodd" '
    'clip-rule="evenodd" d="M48.854 0C21.839 0 0 22 0 49.217c0 21.756 13.993 '
    "40.172 33.405 46.69 2.427.49 3.316-1.059 3.316-2.362 0-1.141-.08-5.052-.08-"
    "9.127-13.59 2.934-16.42-5.867-16.42-5.867-2.184-5.704-5.42-7.17-5.42-7.17-"
    "4.448-3.015.324-3.015.324-3.015 4.934.326 7.523 5.052 7.523 5.052 4.367 "
    "7.496 11.404 5.378 14.235 4.074.404-3.178 1.699-5.378 3.074-6.6-10.839-"
    "1.141-22.243-5.378-22.243-24.283 0-5.378 1.94-9.778 5.014-13.2-.486-1.222-"
    "2.184-6.275.486-13.038 0 0 4.125-1.304 13.426 5.052a46.97 46.97 0 0 1 "
    "12.214-1.63c4.125 0 8.33.571 12.213 1.63 9.302-6.356 13.427-5.052 13.427-"
    "5.052 2.67 6.763.97 11.816.485 13.038 3.155 3.422 5.015 7.822 5.015 13.2 0 "
    "18.905-11.404 23.06-22.324 24.283 1.78 1.548 3.316 4.481 3.316 9.126 0 6.6-"
    ".08 11.897-.08 13.526 0 1.304.89 2.853 3.316 2.364 19.412-6.52 33.405-"
    '24.935 33.405-46.691C97.707 22 75.788 0 48.854 0z" fill="currentColor"/></svg>'
)

# Back cover: corner wedge mirroring the front cover motif.
BACK_CORNER_SVG = """\
<svg viewBox="0 0 400 500" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMaxYMin meet">
        <path d="M 400,0 L 400,350 L 100,0 Z" fill="#d4511a"/>
      </svg>"""

# Info icon — the small circled-"i" that leads the flex callout notes
# (a <svg> sibling of the text block, inside a display:flex .note).
INFO_ICON_SVG = """\
<svg viewBox="0 0 24 24" style="flex:none; width:9mm; height:9mm; margin-top:1mm;" xmlns="http://www.w3.org/2000/svg">
        <circle cx="12" cy="12" r="11" fill="none" stroke="#4a8bc7" stroke-width="2"/>
        <line x1="12" y1="6" x2="12" y2="14" stroke="#4a8bc7" stroke-width="2.4" stroke-linecap="round"/>
        <circle cx="12" cy="18" r="1.4" fill="#4a8bc7"/>
      </svg>"""

# PhysiClaw logo — written out as crab.svg and referenced as an external image.
CRAB_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 280" fill="none">
  <defs>
    <radialGradient id="bg" cx="50%" cy="50%" r="48%">
      <stop offset="0%" stop-color="#d4511a" stop-opacity="0.04"/>
      <stop offset="100%" stop-color="#d4511a" stop-opacity="0"/>
    </radialGradient>
  </defs>

  <circle cx="180" cy="115" r="110" fill="url(#bg)"/>

  <!-- ======= CLAWS — behind body, smooth pincers ======= -->

  <!-- Left claw — below body -->
  <g>
    <path d="M68 182 C50 156, 26 120, 14 102 C10 96, 18 92, 26 98 C36 106, 46 128, 46 150 C46 160, 42 168, 36 172" fill="#d4511a"/>
    <path d="M68 192 C44 198, 10 190, -2 166 C-8 152, 0 140, 14 142 C28 144, 38 156, 40 170 C42 176, 46 180, 52 182" fill="#d4511a"/>
  </g>

  <!-- Right claw — below body, mirrored -->
  <g>
    <path d="M292 182 C310 156, 334 120, 346 102 C350 96, 342 92, 334 98 C324 106, 314 128, 314 150 C314 160, 318 168, 324 172" fill="#d4511a"/>
    <path d="M292 192 C316 198, 350 190, 362 166 C368 152, 360 140, 346 142 C332 144, 322 156, 320 170 C318 176, 314 180, 308 182" fill="#d4511a"/>
  </g>

  <!-- ======= BODY — wide dome, covers the claw arms ======= -->
  <path
    d="M180 55
       C248 55, 300 85, 300 125
       C300 160, 252 182, 180 182
       C108 182, 60 160, 60 125
       C60 85, 112 55, 180 55Z"
    fill="#d4511a"
  />

  <!-- Shell subtle shading -->
  <path
    d="M180 62
       C242 62, 290 88, 290 125
       C290 155, 248 176, 180 176
       C112 176, 70 155, 70 125
       C70 88, 118 62, 180 62Z"
    fill="none" stroke="#e07a45" stroke-width="1.5" opacity="0.2"
  />
  <line x1="180" y1="60" x2="180" y2="178" stroke="#b8440f" stroke-width="1" opacity="0.1"/>

  <!-- ======= ROBOT EARS ======= -->
  <rect x="128" y="24" width="12" height="34" rx="4" fill="#d4511a"/>
  <rect x="220" y="24" width="12" height="34" rx="4" fill="#d4511a"/>

  <!-- ======= EYES ======= -->
  <circle cx="148" cy="105" r="14" fill="#1a1a1e"/>
  <circle cx="212" cy="105" r="14" fill="#1a1a1e"/>
  <circle cx="150" cy="103" r="5.5" fill="#34d399"/>
  <circle cx="214" cy="103" r="5.5" fill="#34d399"/>
  <circle cx="153" cy="100" r="2" fill="white" opacity="0.25"/>
  <circle cx="217" cy="100" r="2" fill="white" opacity="0.25"/>


</svg>
"""

# Wiring section figure — splice the 12V and GND leads, staggering the two
# joints 10 mm apart. White page background + the manual palette (accent 12V
# wire, ink GND wire, mono dimension).
WIRE_SPLICE_SVG = """\
<svg width="760" height="340" viewBox="0 0 760 340" xmlns="http://www.w3.org/2000/svg" font-family="'Inter', 'Helvetica Neue', 'Segoe UI', Arial, sans-serif">
  <!-- Background matches the manual page background (styles.css page-bg) -->
  <rect x="0" y="0" width="760" height="340" fill="#ffffff"/>

  <!-- ===== 12V WIRE (accent) ===== -->
  <line x1="120" y1="120" x2="680" y2="120" stroke="#d4511a" stroke-width="9" stroke-linecap="round"/>
  <!-- splice: soldered core (dashed) under a heat-shrink sleeve -->
  <line x1="272" y1="120" x2="328" y2="120" stroke="#6a6a6a" stroke-width="4" stroke-linecap="round" stroke-dasharray="4 4"/>
  <rect x="266" y="109" width="68" height="22" rx="7" fill="#f3d9cb" stroke="#d4511a" stroke-width="1.5"/>

  <!-- ===== GND WIRE (ink) ===== -->
  <line x1="120" y1="210" x2="680" y2="210" stroke="#1f1f1f" stroke-width="9" stroke-linecap="round"/>
  <line x1="442" y1="210" x2="498" y2="210" stroke="#6a6a6a" stroke-width="4" stroke-linecap="round" stroke-dasharray="4 4"/>
  <rect x="436" y="199" width="68" height="22" rx="7" fill="#e2e0db" stroke="#1f1f1f" stroke-width="1.5"/>

  <!-- ===== wire labels ===== -->
  <text x="104" y="125" font-size="15" font-weight="600" fill="#d4511a" text-anchor="end">12 V</text>
  <text x="104" y="215" font-size="15" font-weight="600" fill="#1f1f1f" text-anchor="end">GND</text>

  <!-- ===== stagger dimension: 10 mm between the two splices ===== -->
  <line x1="300" y1="138" x2="300" y2="184" stroke="#b8b8b8" stroke-width="1" stroke-dasharray="3 3"/>
  <line x1="470" y1="138" x2="470" y2="184" stroke="#b8b8b8" stroke-width="1" stroke-dasharray="3 3"/>
  <line x1="300" y1="165" x2="470" y2="165" stroke="#6a6a6a" stroke-width="1.5"/>
  <polygon points="300,165 310,160.5 310,169.5" fill="#6a6a6a"/>
  <polygon points="470,165 460,160.5 460,169.5" fill="#6a6a6a"/>
  <rect x="351" y="154" width="68" height="22" rx="6" fill="#ffffff"/>
  <text x="385" y="170" font-size="13" fill="#d4511a" text-anchor="middle" font-family="'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace">10 mm</text>
</svg>
"""
