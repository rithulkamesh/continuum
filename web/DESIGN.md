# Continuum design system

Applies to the landing page (`web/index.html`) and the API docs (Sphinx +
Doxygen, themed via `docs/api/python/_static/continuum.css` and
`docs/continuum-doxygen.css`). Single-file landing page, deploys as-is
(`cp web/index.html site/index.html`).

## 1. Visual theme

A dark engineering instrument. Warm near-black ground, near-white ink, one
rust accent used the way a red pen is used: sparingly, on the thing that
matters. The 12-column grid is drawn on the page as faint vertical rules, so
the structure is visible rather than implied. One slow, low-contrast glow
behind the hero makes the page read as alive rather than static; it breathes on
an 11-second cycle and holds still under `prefers-reduced-motion`.

Prose carries the page. Headlines are set in a text serif (Newsreader) because
they are sentences meant to be read, not slogans. Monospace is reserved for
code and measured numbers. The writing follows Zinsser: short sentences, plain
words, concrete claims, active voice. No em dashes. Every number is real and
sourced from `docs/benchmarks.md`.

**Key characteristics**
- Warm near-black `#100f0d`, not pure black. Ink `#ece9e0`, not white.
- One accent: rust `#e8623c`. No gradients on type, no glow on components.
- Visible 12-col grid as hairline rules; one breathing glow behind the hero.
- Newsreader (serif) for statements, Hanken Grotesk for UI, IBM Plex Mono for code and data.
- Hairline rules between every section. Generous vertical space. Sharp corners.
- One inverted section (Start) in warm paper, for rhythm.
- Motion is opacity and transform only, and honors `prefers-reduced-motion`.

## 2. Color

| Role | Hex | Usage |
|---|---|---|
| Background | `#100f0d` | Page background |
| Background 2 | `#17150f` | Alternating band, hover fills |
| Surface | `#1b1a16` | Code blocks, raised surfaces |
| Ink | `#ece9e0` | Primary text, headlines. ~15:1 on background |
| Ink 2 | `#b6b2a6` | Body copy. ~9:1 on background |
| Ink 3 | `#8a8579` | Captions, labels. ~5:1 on background |
| Rule | `#2a2823` | Hairlines, grid lines |
| Rule 2 | `#3b382f` | Stronger dividers, borders |
| Accent | `#e8623c` | Links, key numeral, marks. ~4.6:1 on background |
| Accent dim | `#c14a2a` | Accent hover / pressed |
| Focus | `#6f9bff` | Keyboard focus ring only |

**Inverted Start section** rebinds the tokens to a warm-paper set: bg `#f4f1ea`,
ink `#1b1a16`, ink-2 `#4b4a42`, ink-3 `#6b6a60`, rule `#d8d4c7`, accent
`#b23a1e` (6.5:1 on paper), surface `#fbfaf6`. All pairs pass WCAG AA.

## 3. Typography

- **Serif (statements):** Newsreader, 400 / 500, plus 500 italic. Google Fonts.
- **Sans (UI, body):** Hanken Grotesk, 400 / 500 / 600. Google Fonts.
- **Mono (code, data):** IBM Plex Mono, 400 / 500. Google Fonts.

| Role | Font | Size | Weight | Line height | Tracking |
|---|---|---|---|---|---|
| Hero | Newsreader | clamp(2.4rem, 1.6rem + 3.6vw, 4.2rem) | 500 | 1.06 | -0.02em |
| Section statement | Newsreader | clamp(1.8rem, 1.3rem + 2vw, 2.9rem) | 500 | 1.1 | -0.015em |
| Big numeral | Newsreader | clamp(3rem, 1.5rem + 7vw, 6rem) | 500 | 1 | -0.03em |
| Lede | Hanken Grotesk | clamp(1.05rem, 1rem + 0.4vw, 1.2rem) | 400 | 1.55 | 0 |
| Body | Hanken Grotesk | 1rem | 400 | 1.62 | 0 |
| Kicker / overline | IBM Plex Mono | 0.78rem | 500 | 1.4 | 0.14em, uppercase |
| Data numeral | IBM Plex Mono | context | 500 | 1 | tabular-nums |
| Button / tab | Hanken Grotesk | 0.9rem | 500-600 | 1 | 0.01em |
| Caption | Hanken Grotesk | 0.85rem | 400 | 1.5 | 0 |

Docs pages use the same three families: Newsreader for `h1`-`h6`, Hanken
Grotesk for body, IBM Plex Mono for code.

## 4. Logo

`web/logo.svg`. A 32x32 mark: a bold **C** drawn as one ~262 degree arc
(radius 9, stroke 4.6, round caps) opening to the right, with a small solid
rust dot (`r 2.5`) sitting in the mouth. The C is the continuum and the cycle
of reuse; the dot is the marker on the run you can return to.

- Two-tone: the arc takes `currentColor`, the dot is always `#e8623c`.
- Wordmark and footer: arc in `--ink` (white on the dark ground), dot rust.
  "Continuum" in Newsreader 500, 0.55rem gap, 24px mark in the nav.
- Favicon: mono `#e8623c` (arc and dot), inlined as a `data:` URI, so it holds
  up on a light or dark browser tab.
- `web/logo.svg` carries `style="color:#ece9e0"` so `<img>` embeds render a
  light C on dark; inlined, the page's `color` wins.
- Docs: wired as `html_logo` (Sphinx) and `PROJECT_LOGO` (Doxygen), copied
  from `web/logo.svg` at build time.

## 5. Components

- **Primary action** is a copyable command: mono text, hairline border, `Copy`
  button inside (min-height 40px). Focus ring `#6f9bff`.
- **Text link:** ink text, rust arrow glyph (`\2197` for external), translates
  2px on hover, still under reduced motion.
- **Pill switcher (tabs):** WAI-ARIA tabs pattern. Buttons in a row over a
  hairline; the selected one gets ink text and a 2px accent bottom border.
  Roving `tabindex`, Arrow / Home / End keys move selection and focus, panels
  toggle `hidden`. Used in section 4 for the three durable-execution examples.
- **Section:** top hairline, `padding-block: clamp(4.5rem, 10vw, 8rem)`,
  content in a 12-col grid, statements in columns 1 to 8.
- **Code block:** `<figure>` with mono `<figcaption>`, Surface background,
  hairline border. Comments `#8a8579`, strings `#9bbb7a`, keywords `#7ea2dd`.
- **Tier row:** grid `name / description / cost`, hairline between rows, rust
  bar on the left edge, hover fill Background 2.
- **Nav:** fixed, transparent until scroll, then Background at 82% with a
  bottom hairline and blur. Section anchors hidden below 768px; the Python and
  C++ doc links plus GitHub stay visible.

## 6. Motion

| Element | Animation | Notes |
|---|---|---|
| Hero glow | `breathe` 11s + 17s layers: opacity 0.72 to 1, scale 1 to 1.06 | Gated on `prefers-reduced-motion: no-preference` |
| Page vignette | `glow` 16s: opacity 0.6 to 0.95 | Same gate |
| Section content | `reveal`: opacity + `translateY(14px)` once on scroll-in | Disabled under reduced motion |
| Benchmark bars | `transform: scaleX()` once on scroll-in | Transform only |

No infinite spinners, no autoplaying media, no `transition: all`. Every
animated property is `opacity` or `transform`.

## 7. Do's and don'ts

**Do**
- Keep the grid rules under 50% opacity and the breathing glow low-contrast.
- Use the rust accent on at most one prominent element per viewport.
- Set every headline in Newsreader, every label in mono, everything else in Hanken Grotesk.
- Write section statements as full sentences in sentence case.
- Use `tabular-nums` on every numeral that sits next to another numeral.
- Use "to" for ranges. Use commas, periods, colons, parentheses for breaks.

**Don't**
- No em dashes anywhere.
- No monospace for body copy or headlines.
- No gradients on type, no glow or drop shadow on components.
- No rounded corners beyond 4px.
- No purple, no teal-and-coral, no stock hero gradient, no centered hero.
- No filler adjectives: "powerful", "seamless", "blazing", "revolutionary".

### The AI slop test
Would a skeptical infra engineer believe a person wrote this, both the design
and the copy? If it reads like a template with the nouns swapped, redo it.

## 8. Responsive

| Name | Width | Columns | Gutter |
|---|---|---|---|
| Mobile | 375px | 1 | 1.25rem |
| Tablet | 768px | 12 | 1.5rem |
| Desktop | 1024px | 12 | 2rem |
| Large | 1280px+ | 12 | 3rem |

- Grid rules hidden below 1024px (too dense to read).
- Tier rows and result rows drop the numeral above the text on mobile.
- Tabs wrap; each pill stays a 44px touch target.
- `touch-action: manipulation` on all controls; 44px minimum hit target.

## 9. Agent prompt guide

Quick reference:
- Background `#100f0d`, Ink `#ece9e0`, Ink 2 `#b6b2a6`, Ink 3 `#8a8579`, Accent `#e8623c`, Rule `#2a2823`.
- Serif: Newsreader 500. Sans: Hanken Grotesk. Mono: IBM Plex Mono.
- Sections: top hairline, `padding-block: clamp(4.5rem, 10vw, 8rem)`, content in a 12-col grid, statements in columns 1 to 8.
- Headline prompt: "Newsreader 500, clamp(1.8rem, 1.3rem + 2vw, 2.9rem), line-height 1.1, tracking -0.015em, color `#ece9e0`, text-wrap balance, sentence case, no em dashes."
- Logo prompt: "Monoline 32x32: an open ring with `stroke-dasharray` gap on the right, plus a forward chevron in the gap. `currentColor`, stroke-width 3, round caps."
