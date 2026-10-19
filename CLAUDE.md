Personal site, built with [Astro](https://astro.build/) + Tailwind. Based on the
[astro-theme-resume](https://github.com/ThariqS/astro-theme-resume) theme (thariq.io),
adapted with a Professional/Personal mode toggle.

## Local development

```bash
bun install
bunx astro dev        # http://localhost:4321/
```

The homepage has a mode toggle driven by a `?mode=` query param
(`?mode=professional` or `?mode=personal`). Default is `personal`.

## Build & deploy

```bash
bunx astro build      # static output -> dist/
```

Deployed to **GitHub Pages via GitHub Actions** (`.github/workflows/deploy.yml`).
Pushing to `main` triggers a build + deploy. Repo Settings → Pages → Source must be
set to **GitHub Actions**. Output is static (`output: 'static'`, site
`https://taogenna.github.io`).

## Mode toggle (Professional / Personal)

- Personal mode = dark theme; Professional mode = light theme.
- Elements tagged `data-mode="professional"` or `data-mode="personal"` are shown/hidden
  by a script in `src/pages/index.astro`. `ThemeProvider.astro` maps mode → light/dark.
- `ModeToggle.astro` (header) and `FlippableProfile.astro` (the photo) both drive/read
  the mode via `?mode=` and dispatch `mode-change` / `theme-change` events.

## Project structure

```
src/
├── pages/
│   ├── index.astro       # home (mode toggle, About, projects, writing, misc/reading links)
│   ├── bookshelf.astro   # reading list (data from src/data/books.ts)
│   ├── work.astro        # "Adventures" gallery (data from src/data/work.ts)
│   ├── misc.astro        # coursework, online courses, links
│   └── blog/             # blog routes (renders src/content/post)
├── components/
│   ├── bento/            # BentoGrid/MoreGrid + one scene per project (see 'Project tile')
│   ├── Collage.astro + RotatingBadge.astro  # personal "Adventures" collage (hidden for now)
│   ├── BookShelf.astro, Section.astro, GeometricDivider.astro, ...
├── content/post/         # blog posts (.md) — frontmatter schema in src/content/config.ts
├── data/                 # books.ts, work.ts
├── assets/               # images optimized by Astro <Image> (referenced from components)
└── site.config.ts        # site meta + expressive-code (syntax theme) options

public/                   # served as-is (not optimized)
├── courses/              # Distill-generated courses (static HTML)
├── topcoder/ + data/ + styles/ + fonts/ + m/   # ported legacy pages (TopCoder archive, Harada 9x9)
├── work/                 # collage/gallery images
├── images/books/         # book covers
└── projects/             # project demo videos (.mp4) used by the bento tiles

legacy-hugo/              # the previous Hugo site, archived (not built/served)
```

## Adding content

- **Blog post**: create `src/content/post/<slug>.md` with frontmatter
  (`title` ≤60 chars, `description` 50–160 chars, `publishDate`, `tags`). Math via
  `$...$` / `$$...$$` (remark-math + rehype-katex). Co-locate images and use `![](./img.png)`.
  Post pages (`layouts/BlogPost.astro`) use a **hand-rolled type engine** ported from
  rajan.sh — a `.blog-prose` CSS block (NOT `@tailwindcss/typography`) with a deliberately
  flat scale: **Lora** body `1.05rem/1.85`, plain-foreground normal-weight headings only
  modestly larger than body (content `h1`≈1.7rem so it never outsizes the Hero title).
  Header is centered (`Hero.astro`, non-spiritual branch: Lora title + description subtitle
  + author·date). TOC is a **left** sticky sidebar with scroll-spy (`TableOfContents.astro`).
  renders as **Tufte-style margin sidenotes** (float into the right margin ≥1280px, inline
  below otherwise) via `src/utils/rehypeSidenotes.ts` — no bottom footnotes section. Set
  `citation: true` (optional `citationAuthors: "A, B"`) to append a citation block
  (plaintext + BibTeX) built from title/date/URL by `components/blog/Citation.astro`.
- **Design / style**: read [`STYLE_GUIDE.md`](STYLE_GUIDE.md) before touching the homepage, /projects, a tile or
  a Writing thumbnail (palette, type, tile frame, format variety, motion, honesty rules, prompt template).
- **Project tile** (homepage + /projects): each project is a bespoke scene in `src/components/bento/<Name>.astro`
  wrapped by `BentoTile.astro` (surface tone, name chip, frosted-glass hover panel from `data/projects.ts`). The
  layout lives in `BentoGrid.astro` (homepage + top of /projects) and `MoreGrid.astro` (/projects archive); source
  order is the grid order. Rule: **mix formats** (photo / book spread / object print / data card / colour field /
  digit-texture / quiet type / real product video) — no wall of line plots. Real data comes from
  `scripts/tile-data/*.py` (JSON in `src/data/tiles/`, images in `public/projects/tiles/`); the Distill film is
  recorded by `scripts/film/` (see its header). Astro doesn't scope `@keyframes`: prefix names per tile. Restart
  `astro dev` after adding a new component file (Tailwind misses it otherwise).
- **Book**: add to `src/data/books.ts` (drop the cover into `public/images/books/`).
- **Work / Adventures item**: add to `src/data/work.ts` (drop image into `public/work/`).
  The collage block is currently commented out in `index.astro`.
- **Misc / coursework**: edit the arrays at the top of `src/pages/misc.astro`.

## Notes

- Code highlighting: Expressive Code with `light-plus` / `dark-plus`, gated on the `.dark`
  class (config in `src/site.config.ts`). `useDarkModeMediaQuery` is off so code theme
  follows the site toggle, not the OS.
- `/courses/` and `/topcoder/` are directory pages; links use explicit `index.html`
  so they resolve in the Astro dev server (GitHub Pages resolves directories natively).
