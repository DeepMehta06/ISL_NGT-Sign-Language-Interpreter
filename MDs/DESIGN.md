---
name: SignBridge
version: 1.0.0
aesthetic: Clay Organic + Airbnb Human Warmth
mode: light-only
colors:
  primary: "#5B8DEF"
  primary-soft: "#EEF3FD"
  secondary: "#F97B5C"
  secondary-soft: "#FEF0EC"
  accent-teal: "#3ECFB2"
  accent-teal-soft: "#E8FAF7"
  accent-violet: "#A78BFA"
  accent-violet-soft: "#F3F0FF"
  surface-base: "#FAFAF8"
  surface-card: "#FFFFFF"
  surface-elevated: "#F5F3F0"
  surface-clay: "#F0EDE8"
  text-primary: "#1A1A2E"
  text-secondary: "#5C5C7A"
  text-muted: "#9696AA"
  text-on-accent: "#FFFFFF"
  border-subtle: "#EBEBEB"
  border-default: "#E0DDD8"
  border-strong: "#C8C5C0"
  success: "#34C98A"
  warning: "#FFBA3B"
  error: "#FF5757"
  isl-primary: "#3ECFB2"
  isl-soft: "#E8FAF7"
  ngt-primary: "#A78BFA"
  ngt-soft: "#F3F0FF"
typography:
  display:
    fontFamily: "Plus Jakarta Sans"
    fontSize: "3.5rem"
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: "-0.03em"
  h1:
    fontFamily: "Plus Jakarta Sans"
    fontSize: "2.5rem"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "-0.02em"
  h2:
    fontFamily: "Plus Jakarta Sans"
    fontSize: "1.75rem"
    fontWeight: 600
    lineHeight: 1.3
  h3:
    fontFamily: "Plus Jakarta Sans"
    fontSize: "1.25rem"
    fontWeight: 600
    lineHeight: 1.4
  body-lg:
    fontFamily: "Inter"
    fontSize: "1.0625rem"
    fontWeight: 400
    lineHeight: 1.7
  body-md:
    fontFamily: "Inter"
    fontSize: "0.9375rem"
    fontWeight: 400
    lineHeight: 1.6
  body-sm:
    fontFamily: "Inter"
    fontSize: "0.8125rem"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Inter"
    fontSize: "0.75rem"
    fontWeight: 500
    lineHeight: 1.4
    letterSpacing: "0.04em"
    textTransform: "uppercase"
  mono:
    fontFamily: "JetBrains Mono"
    fontSize: "0.875rem"
rounded:
  xs: "6px"
  sm: "10px"
  md: "16px"
  lg: "24px"
  xl: "32px"
  2xl: "40px"
  full: "9999px"
spacing:
  1: "4px"
  2: "8px"
  3: "12px"
  4: "16px"
  5: "20px"
  6: "24px"
  8: "32px"
  10: "40px"
  12: "48px"
  16: "64px"
  20: "80px"
  24: "96px"
shadows:
  clay-sm: "0 2px 8px rgba(0,0,0,0.06), 0 0 0 1px rgba(0,0,0,0.04)"
  clay-md: "0 4px 20px rgba(0,0,0,0.08), 0 0 0 1px rgba(0,0,0,0.04)"
  clay-lg: "0 8px 40px rgba(0,0,0,0.10), 0 0 0 1px rgba(0,0,0,0.04)"
  clay-float: "0 20px 60px rgba(0,0,0,0.12), 0 0 0 1px rgba(0,0,0,0.06)"
  glow-teal: "0 0 24px rgba(62,207,178,0.25)"
  glow-violet: "0 0 24px rgba(167,139,250,0.25)"
  glow-blue: "0 0 24px rgba(91,141,239,0.25)"
components:
  button-primary:
    background: "#5B8DEF"
    text: "#FFFFFF"
    radius: "full"
    padding: "12px 24px"
    fontWeight: 600
    shadow: "0 4px 14px rgba(91,141,239,0.35)"
    hover-transform: "translateY(-1px)"
    hover-shadow: "0 6px 20px rgba(91,141,239,0.45)"
  button-secondary:
    background: "#FFFFFF"
    border: "1.5px solid #E0DDD8"
    text: "#1A1A2E"
    radius: "full"
    padding: "12px 24px"
    fontWeight: 600
    hover-background: "#F5F3F0"
  card:
    background: "#FFFFFF"
    radius: "24px"
    shadow: "0 4px 20px rgba(0,0,0,0.08), 0 0 0 1px rgba(0,0,0,0.04)"
    padding: "28px"
  badge-isl:
    background: "#E8FAF7"
    text: "#0B9E84"
    radius: "full"
    padding: "4px 12px"
    fontWeight: 600
    fontSize: "0.75rem"
  badge-ngt:
    background: "#F3F0FF"
    text: "#7C5CE0"
    radius: "full"
    padding: "4px 12px"
    fontWeight: 600
    fontSize: "0.75rem"
---
## 1. Visual Theme & Atmosphere

SignBridge is a real-time sign language interpreter for ISL (Indian Sign Language) and NGT (Dutch Sign Language). The visual language must communicate trust, warmth, and technical capability simultaneously.

**Primary aesthetic:** Clay Organic — soft, tactile, three-dimensional elements that feel like they exist in physical space. Cards look sculpted. Buttons feel pressable. 3D illustrated hands and icons float above the surface. Nothing feels flat or generic.

**Secondary influence:** Airbnb Human Warmth — photography-driven thinking applied to iconography. Every visual element should feel like it was made by a human for a human. Rounded forms, warm neutrals, genuine personality.

**Mood:** Approachable expertise. Not clinical. Not cold. Not a research tool. A product that a deaf student in Mumbai or Rotterdam would actually want to use every day. Light, airy, optimistic — with serious ML running underneath.

**Light mode only.** The webcam feed and skeleton overlay read best on warm light backgrounds. Dark mode is explicitly out of scope for v1.

---

## 2. Color Palette & Roles

The palette is built on warm off-white surfaces with three functional accent colors and two language-specific identity colors.

### Surface system

- **`--surface-base: #FAFAF8`** — The page background. Warm limestone, never pure white. Every screen starts here.
- **`--surface-card: #FFFFFF`** — Card surfaces. Pure white creates just enough contrast against the base.
- **`--surface-elevated: #F5F3F0`** — Hover states, secondary panels, slightly pressed elements.
- **`--surface-clay: #F0EDE8`** — The "clay" surface — deepest background used for sectioning and depth.

### Primary accent — Blue

- **`--primary: #5B8DEF`** — The main action color. CTAs, active states, primary buttons, links. A friendly, confident blue — not corporate, not cold.
- **`--primary-soft: #EEF3FD`** — Blue-tinted backgrounds for highlighted sections, selected states.

### Secondary accent — Coral

- **`--secondary: #F97B5C`** — Energy, warmth, celebration. Used sparingly: onboarding illustrations, success animations, landing page hero accents. Borrowed from Airbnb's warmth.
- **`--secondary-soft: #FEF0EC`** — Coral-tinted soft backgrounds.

### Language identity colors

- **`--isl-primary: #3ECFB2`** — Teal. SignBridge ISL mode. All ISL-specific UI elements, the skeleton overlay for ISL, the ISL language badge, active state when ISL is selected.
- **`--ngt-primary: #A78BFA`** — Violet. SignBridge NGT mode. All NGT-specific UI elements, skeleton overlay for NGT, NGT badge, active state when NGT is selected.

### Text

- **`--text-primary: #1A1A2E`** — Near-black with a blue undertone. All body copy and headings.
- **`--text-secondary: #5C5C7A`** — Secondary info, descriptions, metadata.
- **`--text-muted: #9696AA`** — Placeholders, disabled states, timestamps.

**Contrast rule:** Every text-on-background combination must pass WCAG AA (4.5:1 for body, 3:1 for large text). Never place `--text-secondary` on `--surface-clay` without checking contrast first.

---

## 3. Typography Rules

Two Google Fonts only. Import both at project start.

```html
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
```

**Plus Jakarta Sans** — all headings and display text. Has personality without being decorative. Feels modern and warm simultaneously. Use weights 600–800 for headings. Never use 400 weight for a heading.

**Inter** — all body text, labels, UI copy. The most legible UI font at small sizes. Use 400 for body, 500 for labels and captions, 600 for emphasized UI text.

**JetBrains Mono** — confidence scores, keypoint data, technical readouts only. Never use for general UI text.

### Type scale — apply these exactly, never invent intermediate sizes

| Role    | Font              | Size      | Weight | Use for                                 |
| ------- | ----------------- | --------- | ------ | --------------------------------------- |
| Display | Plus Jakarta Sans | 3.5rem    | 700    | Landing page hero, major section titles |
| H1      | Plus Jakarta Sans | 2.5rem    | 700    | Page titles                             |
| H2      | Plus Jakarta Sans | 1.75rem   | 600    | Section headers                         |
| H3      | Plus Jakarta Sans | 1.25rem   | 600    | Card titles, subsections                |
| Body LG | Inter             | 1.0625rem | 400    | Lead paragraphs, important descriptions |
| Body MD | Inter             | 0.9375rem | 400    | Standard body copy                      |
| Body SM | Inter             | 0.8125rem | 400    | Captions, helper text                   |
| Label   | Inter             | 0.75rem   | 500    | ALL CAPS labels, category tags          |
| Mono    | JetBrains Mono    | 0.875rem  | 400    | Confidence scores, data readouts        |

**Letter spacing:** `-0.03em` on Display, `-0.02em` on H1, `0` on all body text, `+0.04em` on Label (uppercase labels always need tracking).

---

## 4. Component Styling

### 3D Clay Cards

The signature visual element of SignBridge. Every card must feel like it has physical weight.

```css
.card-clay {
  background: #FFFFFF;
  border-radius: 24px;
  box-shadow: 
    0 4px 20px rgba(0,0,0,0.08),
    0 0 0 1px rgba(0,0,0,0.04);
  padding: 28px;
  transition: transform 150ms ease, box-shadow 150ms ease;
}
.card-clay:hover {
  transform: translateY(-2px);
  box-shadow: 
    0 8px 40px rgba(0,0,0,0.10),
    0 0 0 1px rgba(0,0,0,0.04);
}
```

### Buttons

All buttons are pill-shaped (`border-radius: 9999px`). No exceptions.

**Primary button:** Blue fill, white text, subtle blue glow shadow. On hover: lift 1px, deepen shadow.
**Secondary button:** White fill, `1.5px solid #E0DDD8` border, dark text. On hover: `#F5F3F0` background.
**Danger button:** `#FF5757` fill, white text. Only for destructive actions.
**Ghost button:** Transparent fill, no border, colored text. For tertiary actions only.

All buttons: `font-weight: 600`, `padding: 12px 24px` (standard), `padding: 10px 20px` (small). Active state: `scale(0.98)`.

### Language toggle (ISL / NGT)

Pill-shaped container with two options. Active option slides with a smooth 200ms transition.

- ISL active: `--isl-primary` background, white text, teal glow shadow
- NGT active: `--ngt-primary` background, white text, violet glow shadow
- Inactive option: transparent, `--text-secondary` text

### Skeleton overlay

Drawn on `<canvas>` over the webcam feed.

- Hand landmarks: 5px radius circles, `--isl-primary` or `--ngt-primary` depending on active language
- Connections between landmarks: 2px lines, 60% opacity of the same color
- Body pose landmarks: 3px radius, `--primary` color
- Everything anti-aliased. Never pixelated.

### Confidence meter

Full-width bar, 6px height, `border-radius: 9999px`.

- Track: `--surface-clay`
- Fill: gradient from `--primary` to `--isl-primary` (or `--ngt-primary`)
- Animates smoothly with `transition: width 100ms ease`

### Word chips

Small pill-shaped tags showing recognised words.

- Background: `--surface-elevated`
- Border: `1px solid --border-default`
- Text: `--text-secondary`, `0.8125rem`, `Inter 500`
- Border radius: `9999px`
- Padding: `4px 12px`
- Animate in: `fade + translateY(4px)` over `120ms`

### Input fields

- Background: `--surface-card`
- Border: `1.5px solid --border-default`
- Border radius: `12px`
- Focus border: `--primary`, with `box-shadow: 0 0 0 3px rgba(91,141,239,0.15)`
- Padding: `12px 16px`
- Font: Inter 400, `0.9375rem`
- Placeholder: `--text-muted`

### 3D Icons — MANDATORY

Every icon in the UI must be 3D illustrated, not flat line icons. Use Lucide React ONLY for functional utility icons inside inputs and buttons (search, close, chevron). For all feature icons, section icons, and landing page icons — use 3D illustrated icons from Spline, or implement them as CSS + SVG 3D approximations with multiple shadow layers.

3D icon style: soft clay material, single light source from top-left, subtle ambient occlusion, rounded forms. Same color family as the section's accent color.

---

## 5. Layout Principles

**Grid:** 12-column, `24px` gutters, `80px` max-width container padding on desktop.
**Max content width:** `1280px`
**Breakpoints:** mobile `< 768px`, tablet `768–1199px`, desktop `≥ 1200px`

**Spacing rhythm:** All spacing must be multiples of `4px`. Use the spacing scale — never arbitrary values like `13px` or `22px`.

**Whitespace philosophy:** Generous. Section vertical padding is minimum `80px`. Card internal padding minimum `28px`. Never crowd elements — breathing room is a feature.

**App layout (interpreter screen):**

- Camera panel: 7/12 columns, minimum `520px` width
- Right panel: 5/12 columns
- Panels are separated by `24px` gap
- Minimum total width: `1024px` — no mobile layout in v1

**Landing page layout:**

- Full-width hero with 3D illustration
- Bento-grid feature section (alternating large + small cards)
- Social proof / demo section
- Simple footer

**Bento grid rule:** Feature cards come in two sizes only: `large` (spans 2 columns) and `small` (spans 1 column). Alternate the pattern. Never put two large cards side by side.

---

## 6. Depth & Elevation

SignBridge uses a 4-level elevation system. Higher elevation = more shadow + slight scale increase.

| Level         | Use                        | Shadow                                                       |
| ------------- | -------------------------- | ------------------------------------------------------------ |
| 0 — Flat     | Page background, dividers  | None                                                         |
| 1 — Raised   | Standard cards, panels     | `0 2px 8px rgba(0,0,0,0.06), 0 0 0 1px rgba(0,0,0,0.04)`   |
| 2 — Floating | Hover cards, dropdowns     | `0 8px 40px rgba(0,0,0,0.10), 0 0 0 1px rgba(0,0,0,0.04)`  |
| 3 — Overlay  | Modals, tooltips, popovers | `0 20px 60px rgba(0,0,0,0.12), 0 0 0 1px rgba(0,0,0,0.06)` |

**Glow shadows for accent elements:**

- Teal elements (ISL): `0 0 24px rgba(62,207,178,0.25)`
- Violet elements (NGT): `0 0 24px rgba(167,139,250,0.25)`
- Blue elements (primary): `0 0 24px rgba(91,141,239,0.25)`

**The clay effect:** All major cards use a combination of the shadow above PLUS `0 0 0 1px rgba(0,0,0,0.04)` — this 1px ring creates the subtle border that makes elements feel sculpted rather than floating.

---

## 7. Do's and Don'ts

### ALWAYS do

- Use Plus Jakarta Sans for every heading, no exceptions
- Make every major feature icon 3D illustrated, not flat
- Apply `border-radius: 9999px` to all buttons
- Add hover lift (`transform: translateY(-2px)`) to all interactive cards
- Use the language identity color (teal for ISL, violet for NGT) consistently everywhere a language context is shown
- Add `transition` to every interactive element — minimum `150ms ease`
- Use warm off-white `#FAFAF8` as the page background, never pure white
- Add glow shadows to primary CTA buttons
- Use the full clay shadow system (multi-layer shadows) on cards
- Import Plus Jakarta Sans and Inter from Google Fonts every time

### NEVER do

- Use flat line icons (Lucide/Feather/etc.) for feature/section icons — 3D only
- Use pure black `#000000` or pure white `#FFFFFF` as page background
- Use `border-radius` below `10px` on cards — everything has generous rounding
- Place any text below `12px` font size
- Use more than 2 typefaces (Plus Jakarta Sans + Inter + JetBrains Mono for data)
- Add dark mode styles — light only in v1
- Use generic gradient backgrounds (blue-to-purple, etc.) — use solid warm surfaces instead
- Use font-weight below 400 or above 700
- Use colors outside the defined palette — no improvisation
- Use box shadows with a spread above 60px — keeps the clay effect subtle, not muddy
- Use centered text for body paragraphs — left-align all body copy
- Skip transition animations on interactive elements

---

## 8. Responsive Behavior

**Desktop first.** The interpreter screen is desktop-only (camera + panels require 1024px minimum).

**Landing page responsive rules:**

- Display type scales from `3.5rem` (desktop) → `2.5rem` (tablet) → `1.75rem` (mobile)
- Bento grid collapses: 2-col on tablet, 1-col on mobile
- Card padding: `28px` (desktop) → `20px` (tablet/mobile)
- Buttons: full-width on mobile (`width: 100%`)
- Navigation: hamburger menu on mobile

**Touch targets:** Minimum `44×44px` for all interactive elements on mobile.

**3D illustrations:** Scale down proportionally but never disappear on mobile. They are core to the identity.

---

## 9. Agent Prompt Guide

Use these prompts when asking Claude or any agent to build UI for SignBridge:

**Starting any new screen:**

> "Read DESIGN.md before writing any code. Use Plus Jakarta Sans for all headings, Inter for body text. All cards must use the clay shadow system. All feature icons must be 3D illustrated. Page background is `#FAFAF8`, never pure white."

**Building the landing page:**

> "Build a landing page for SignBridge following DESIGN.md exactly. Include: full-width hero with a 3D illustrated hand signing, a bento-grid features section, a live demo section showing the interpreter UI. Use coral (`#F97B5C`) accents in the hero, blue (`#5B8DEF`) for all CTAs, and warm off-white surfaces throughout."

**Building the interpreter screen:**

> "Build the interpreter screen following DESIGN.md. 7-column camera panel on the left with skeleton overlay canvas, 5-column sentence panel on the right. Language toggle (ISL teal / NGT violet) at the top center. Confidence meter below camera. Word chips animate in as signs are recognised."

**When output looks generic:**

> "This looks generic. Apply the DESIGN.md clay aesthetic: add multi-layer box shadows to all cards, change all icons to 3D illustrated style, increase border-radius to 24px on cards, switch page background to `#FAFAF8`, add hover lift animations to all interactive cards."

**For 3D icons specifically:**

> "All icons must be 3D illustrated in clay style — soft material, top-left light source, rounded forms. Use the section's accent color as the base. Never use flat line icons from Lucide or similar libraries for feature representation."
