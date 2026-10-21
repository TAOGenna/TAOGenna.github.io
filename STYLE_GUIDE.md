# Style guide

How the site looks and how to add to it without breaking the look. Everything below is read off the code;
if the code and this file disagree, fix whichever is wrong.

**The idea.** An editorial case study, not a card grid: quiet cool paper, ink, a serif display face, mono
captions, and tiles that are each a *different kind of object* (a photo, a printed page, a data card, a colour
field, a texture) held together by one palette and one set of details. Reference: blunarova.com/hermeneutic
("Perception-to-Pattern": soft blurred shapes that resolve into sharp structure).

Scope: the homepage and `/projects` use this system. Blog, bookshelf, misc and `/work` keep the older
parchment/gold look on purpose (the retone is scoped, see §2).

---

## 1. Tokens

### Colour

| Role | Value | Used for |
|---|---|---|
| ink | `#0b0f14` | dark tiles, title tile, text on light |
| on-ink text | `#e9edf0` | text on dark tiles |
| paper | `#f1efea` | light tiles (the printed-page surface) |
| glacier | `#dfe9ee` → `#c9dae2` (text `#10202a`) | pale blue data cards |
| slate | `#7e8ea1` | mid-tone tile |
| steel | `#8fa9b8` | hairlines, link underlines, orbs |
| lavender | `#cfd2f5` | the one accent: highlights, hub, data on dark |
| mint | `#d8ece4` | secondary accent, colour ramps |

Soil/raster ramp: `#eef3f0 #d8ece4 #b9d6d3 #8fb3bd #5f8199 #2f4a60`.
**No amber / gold / terracotta** in this system (the old site's gold is remapped, §2). Don't add a new hue
per tile; pull from the table.

Page variables (set in `index.astro`, scoped to `html:has(.ha)`): light `--background 48 22% 95.5%`,
`--foreground 212 28% 9%`, `--muted-foreground 210 11% 38%`, `--border 210 12% 80%`, `--gold` → steel
`205 22% 55%`, `--bronze` slate-ink `212 16% 32%`; dark `--background 212 30% 7%`, `--gold` → lavender
`235 55% 80%`. Use the variables (`text-foreground`, `text-muted-foreground`, `hsl(var(--border))`) in page
chrome; use the hex values only inside tile scenes, where the tone is fixed.

### Type

- **Display / headings / prose on the homepage: Newsreader Variable** (`font-newsreader`). Weight **300** for
  big display and section headings, 350–450 for prose/titles. Tight tracking (`-0.01` to `-0.02em`),
  tight leading (1.05–1.18) for display.
- **Mono** (`font-mono` / `ui-monospace…`) for everything that's a *label*: kickers, name chips, "fig N"
  captions, dates, axis ticks. Size `0.55–0.62rem`, UPPERCASE, tracking `0.06–0.1em`.
- Two voices only: serif for the thing, mono for the label on the thing. No bold sans.
- Cormorant / EB Garamond / Lora belong to the old pages and the blog; don't introduce them to new homepage
  or tile work.

### Shape and surface

- Tiles `rounded-xl`, `ring-1 ring-inset` (`black/10` on light, `white/10` on ink). Page gap `gap-3`.
- Hairlines: `0.5px` strokes at 12–20% opacity; thin beats heavy.
- Grain on every tile (see `BentoTile.astro`): `multiply` 7% on light, `screen` 5% on ink.
- Glass (hover panel, wheel): `backdrop-blur-xl backdrop-saturate-150`, 1px lighter top border, soft
  shadow. Smoked (`#0b0f14/60`) on light tiles, slate (`#1b2330/55`) on ink tiles, white text.

---

## 2. Page structure (homepage)

- Wrapper `<div class='ha …'>` activates the retone via `html:has(.ha)`: it remaps `--gold/--bronze/
  --background…`, which recolours the profile ring, mode toggle, header/footer hovers and shared dividers.
  New pages that should look like the homepage add the same `.ha` wrapper and the same `<style is:global>`
  block (or move it to `app.css` if the retone becomes site-wide).
- **Section** = left column `md:w-1/5` with a mono kicker (`Who`, `Notes`, `Papers`, `Elsewhere`; a word, never a
  number: numbered kickers read as counts) over a Newsreader 1.65rem/300 heading; right column `md:w-2/3`.
- **Dividers**: `.ha-div`, a fading hairline with one 5px ink dot. Not the ornamental `GeometricDivider`
  (it survives only in personal mode, retoned).
- **Links** in prose: `.illuminated-link`, a steel underline that turns ink on hover. Publication titles
  underline-draw on hover.
- **Hero**: profile flip photo, name in Newsreader 300 (`.ha-name`), social icons in muted foreground.
- **"More Work" wheel**: `<RotatingBadge glass …>`: frosted lens over three drifting orbs (lavender, mint,
  steel). The gold wheel (no `glass`) stays for the personal collage and `/work`.
- **Writing list** (`components/blog/WritingList.astro`): image left, then title (Newsreader 1.2rem),
  two-line excerpt (muted), date below in mono `DD MON YYYY`. **No** tags, read time or timeline. The
  whole row is the link. Thumbnail = a bento scene authored at 271×160 and scaled (`--k` 0.44, 0.31 on
  phones). A post with no scene gets a quiet plate with its initial; to give one a real thumbnail, add a
  scene and key it by slug in the `fig` map.

---

## 3. Tiles

**Frame.** `BentoTile.astro` owns the frame; a scene never draws its own border, chip or hover panel.
Props: `project` (from `data/projects.ts`), `tone` (`ink | paper | glacier | slate`), `label` (chip text),
`reveal` (hover panel, default on; turn off only when the tile already says everything, like Yape),
`blurb`/`tags` (override the panel copy), `logo` (a special case: show a logo instead of the mono name
chip; only YaVendio does, with its real white wordmark `public/projects/yavendio-wordmark-white.png`).

**Grid.** `auto-rows-[10rem]`, `gap-3`, 2 columns on phones (`grid-flow-dense`), 4 from `lg`. Spans:
1×1 default, `col-span-2`, `row-span-2`; a scene must survive 1×1 up to 2×2 and stay readable at phone
width. **The grid is deliberately uneven**: big tiles are staggered (Majorana 2×2 on the left, YaVendio 2×2
below it, Distill a wide band), and a row can end in a full-width banner (`col-span-2 lg:col-span-4`, LSeg).
Never tidy it into a regular lattice; the section's character is that it isn't homogeneous. A tile that has
to work at very different widths (YaVendio: 2×2 on desktop, full width on phones) sizes itself from its own
width with a CSS container query (`container-type: inline-size`, `cqw` units, `@container (min-width: …)`)
instead of viewport breakpoints. Layout lives in `BentoGrid.astro` (homepage + top of `/projects`) and `MoreGrid.astro` (archive);
**source order is grid order**, and the diagram in each file's header comment is the map. Update it when
you move things.

**Scene.** A component in `src/components/bento/<Name>.astro`, filling the slot (`absolute inset-0`).
Author vector scenes in a **271×160** viewBox (`preserveAspectRatio='xMidYMid slice'`), so one scene also
works as a Writing thumbnail. Leave the top-left free for the name chip and the bottom for the hover
panel. The first comment in the file says: which format it is, where its data comes from, what is
illustrative.

**Tone.** Pick the tone from the scene's content, then look at the neighbours and break up runs of the same
one (glacier and slate are the cool breaks between paper tiles). Dark tiles are rare and earn their place
(title tile, YaVendio, Olive, Quantum).

---

## 4. Formats: the non-uniformity rule

"Uniformity is boring." Each tile picks a **format**, and neighbours must differ. Current catalogue:

| Format | Examples | Notes |
|---|---|---|
| Real photo | Pong, LSeg input, Style transfer | the original, uncropped story |
| Printed page / book spread | Majorana (paper + blackboard), Inversion (paper + replayed simulation), Thesis | slow pan; chip top-left and plate top-right stay clear of the text |
| Object on a surface | YaVendio phone, MLIP embossed card | CSS-3D or shadow, a real prop |
| Pale data card on a plate | Yape, Soil | frosted card, mono labels |
| Colour field / strips | Photonics | flat strips, thin ink lines over them |
| Digit-contour texture | Quantum, PPCA, Soil background | digits only on contour lines; **max ~2 per block** |
| Quiet type | Olive, Non-Markov | near-empty, one idea |
| Real product video | Distill film, MAHO | recorded, cropped, colour poster |
| Reel (several works, one tile) | Competitive programming (vrating → heatmap → theme → predictor → library), Majorana, Inversion | one card per body of work, not one per sub-tool; ≤ ~6 s per scene, ticker bottom-left, tool plate bottom-right, pauses on hover |

Don't ship a wall of thin-line plots on flat backgrounds, and don't use matplotlib frames on cream (the
old look). If a project only has a plot, re-draw it in the palette or pick a different format for it.

---

## 5. Motion

- **One quiet loop per tile**: ~8–12 s cycle, ease-out, every element holds its end state, then all fade
  and restart together (see `Olive.astro`). No bounce, no flashing, nothing that fights the neighbours.
- **Busy scenes stay still until hovered** (Distill film, Yape tour, Connect 4 replay): the idle state is a
  deliberate still frame, not frame 0, and leaving the tile puts it back (`utils/hover.ts: onTileHover`).
  Quiet one-pass loops (Majorana, Olive, …) may run on their own.
- Hover motion lives on the frame (panel slide-up), the wheel (spin up, orbs speed up) and Writing rows
  (2px lift). Scenes don't need to react to hover.
- **Astro does not scope `@keyframes`**: prefix every name per tile (`yv-`, `st-`, `ls-`, `soil-`, `qc-`,
  `ol-`, `ph-`, `th-`, `ion-`, `inv-`, `iv-`, `ml-`, `mj-`, `dst-`, `cpc-`, `vr-`, `hm-`, `ct-`, `pd-`, `lb-`, …) or tiles overwrite each other.
- Honour `prefers-reduced-motion: reduce` (stop or freeze the loop). Every animated vector scene does;
  the video and photo tiles are exempt.
- Video: `autoplay loop muted playsinline preload='metadata'`, a coloured `poster` (not a white-fade
  frame), `aria-label` describing the content.

---

## 6. Content and honesty

- **Real data wherever it exists**: from the project's own repo/paper/output, extracted by a script in
  `scripts/tile-data/` (JSON → `src/data/tiles/`, images → `public/projects/tiles/`). Don't hand-draw a
  fake result when the real one can be computed.
- **Headline numbers come from his own record** (CV, LinkedIn, paper, repo), never invented. A number
  must mean something to a stranger (+130% correct payment predictions, $55M closed sales, 600K
  conversations/month, 10K conversations/month) — a bare "18M users" does not.
- **Example data from someone else's account** is allowed only when it comes from the project's own README/docs, is
  labelled in the file header, and never shows a handle (vrating, cf-heatmap and cf-predictor use the
  competitive-programming repo's example screenshots; swap in his own handles when he gives them).
- **One card per body of work.** A repo with several tools (competitive-programming) is a single reel tile, not
  a tile per tool.
- **Illustrative parts are labelled in the file header.** Currently illustrative: Distill's dependency
  graph (the real generated `depends_on` lists every earlier module, a Distill bug, so never draw it),
  the YaVendio chat, the Yape scores (provider names are real).
- Show **results, not job titles** (YaVendio shows what shipped, not a role card).
- Hover copy: one or two sentences of what it is + result, then mono tags. No marketing adjectives.
- **Copy must sound like a person, not a model.** Plain first-person sentences, concrete numbers, the words he
  would actually say ("Tools I wrote for Codeforces and AtCoder: …"). Avoid: clever italic taglines, "X, before Y"
  slogans, em-dash asides, lists of three adjectives, "the library behind them", "built along the way", stacked
  middots used as a sentence. A label says what the thing is ("Rating change"), a caption says what to look at.

---

## 7. Adding a project tile

1. Add the entry to `src/data/projects.ts` (heading, href, subheading, tags).
2. Pick a **format** the neighbours aren't (§4) and a **tone** (§3).
3. If it needs real data: write `scripts/tile-data/<name>.py` (run with a venv that has `pillow numpy
   scipy`; header docstring says how), commit the output JSON/images, not the venv.
4. Write `src/components/bento/<Name>.astro`: header comment (format · data · what's illustrative),
   271×160 vector or an `<img loading='lazy'>`/`<video>`, prefixed keyframes, reduced-motion guard,
   `aria-hidden='true'` on decoration or `role='img'` + `aria-label` on informative SVG, descriptive `alt`
   on images.
5. Place it in `BentoGrid.astro` or `MoreGrid.astro` with `tone`, `label` and a span; update the layout
   diagram comment.
6. **Restart `astro dev`**: Tailwind misses classes in a brand-new component file otherwise.
7. Check with Playwright at **1440** and **390** px: hover panel legible on that tone, no horizontal
   overflow, loop restarts cleanly, nothing clipped under the chip. (The homepage has a known ~6px phone
   overflow from the header mode toggle; don't make it worse.)
8. Real product film: record the real UI in real time with Playwright, restyle it with injected CSS in the
   palette, trim/ramp/crossfade with ffmpeg (`scripts/film/` has the working recipe; needs `PW_DIR`,
   the product running, and a Chromium via `CHROME`). Export a wide cut sized to the tile.

---

## 8. If you generate thumbnails with an image model

Nothing on the site is AI-generated today: tiles are code or real captures. If you do generate one, keep a
fixed **style spine** and change only the **format slot**; the spine is what keeps variety consistent.

```
STYLE SPINE (identical for every image):
Editorial case-study visual, cool and quiet. Palette limited to ink black #0b0f14, warm off-white paper
#f1efea, glacier blue #dfe9ee, steel #8fa9b8 and one lavender accent #cfd2f5. Soft film grain, hairline
details, generous negative space, no gradients-as-decoration, no neon, no amber, no text except a tiny
monospace caption. Slightly out-of-focus shapes resolving into crisp structure. 271:160 aspect.

FORMAT SLOT (change per image; pick one not used by the neighbouring tiles):
- "a photograph of <the real object>, shallow depth of field, on a matte surface"
- "a printed book spread lying flat, paper texture, soft shadow, one figure and a caption"
- "a pale frosted-glass data card floating on a dark slate plate, mono labels"
- "flat colour-field strips in two palette colours, nothing else"
- "a sparse field of digits that only appear along contour lines, on <ink|glacier>"
- "near-empty type tile: one large light serif word, one hairline arrow"

SUBJECT SLOT: <one noun phrase: what the project is>
```

Then: crop to the tile's aspect, put the file in `public/projects/tiles/`, and check it on its tone with the
chip and the hover panel over it. Prefer an `<img>` scene that matches §3 (`alt`, `loading='lazy'`).

---

## 9. Don'ts

- No gold/amber/terracotta in homepage or tile work; no new accent hue per project.
- No second chip, border, or hover panel inside a scene.
- No uniform plot walls, no matplotlib-on-cream, no emoji, no stock gradients.
- No invented metrics; no unlabeled illustrative data; no job title on a project card.
- No global `@keyframes` names without a prefix.
- No timeline, tag chips or read-time in the Writing list.
- Don't leave scratch/mockup pages in `src/pages/` (they ship); keep experiments outside the repo or delete
  them before pushing.
