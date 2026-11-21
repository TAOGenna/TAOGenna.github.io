Personal site, built with [Astro](https://astro.build/) + Tailwind. Based on the
[astro-theme-resume](https://github.com/ThariqS/astro-theme-resume) theme (thariq.io),
adapted with a Professional/Personal mode toggle.

## Local development

```bash
bun install
bunx astro dev        # http://localhost:4321/
```

The homepage has a mode toggle driven by a `?mode=` query param
(`?mode=professional` or `?mode=personal`). Default is `professional`.

## Build & deploy

```bash
bunx astro build      # static output -> dist/
```

Deployed to **GitHub Pages via GitHub Actions** (`.github/workflows/deploy.yml`).
Pushing to `main` triggers a build + deploy. Repo Settings → Pages → Source must be
set to **GitHub Actions**. Output is static (`output: 'static'`, site
`https://taogenna.github.io`).

## Mode toggle (Professional / Personal)

- Professional mode = light theme; Personal mode = dark theme.
- Elements tagged `data-mode="professional"` or `data-mode="personal"` are shown/hidden
  by a script in `src/pages/index.astro`. `ThemeProvider.astro` maps mode → light/dark.
- `ModeToggle.astro` (header) and `FlippableProfile.astro` (the photo) both drive/read
  the mode via `?mode=` and dispatch `mode-change` / `theme-change` events.

## Project structure

```
src/
├── pages/
│   ├── index.astro         # home (mode toggle, About, projects, writing, misc/reading links)
│   ├── projects.astro      # /projects: BentoGrid + MoreGrid
│   ├── projects/[slug].astro  # project write-ups (renders src/content/project)
│   ├── blog/               # blog routes (renders src/content/post)
│   ├── bookshelf.astro     # reading list (data from src/data/books.ts)
│   ├── work.astro          # "Adventures" gallery (data from src/data/work.ts)
│   ├── misc.astro          # coursework, online courses, links
│   └── tools/, tags/
├── layouts/                # BaseLayout, BlogPost, ProjectPage
├── components/
│   ├── bento/              # BentoTile, BentoGrid/MoreGrid + one scene per project (see 'Project tile')
│   ├── prose/              # Poem, Film, Lede: blocks usable in any .mdx post or write-up
│   ├── Collage.astro + RotatingBadge.astro  # "Adventures" collage (personal mode)
│   ├── BookShelf.astro, Section.astro, GeometricDivider.astro, ...
├── content/
│   ├── post/               # blog posts (.md / .mdx)
│   └── project/            # project write-ups (.mdx); both schemas in src/content/config.ts
├── data/                   # projects.ts, books.ts, work.ts, tiles/ (generated tile data)
├── styles/prose.css        # shared reading engine (.blog-prose)
├── utils/destination.ts    # where a project/link goes (see 'Project tile')
├── assets/                 # images optimized by Astro <Image> (referenced from components)
└── site.config.ts          # site meta + expressive-code (syntax theme) options

public/                     # served as-is (not optimized)
├── courses/                # Distill-generated courses (static HTML)
├── Files/                  # PDFs (bachelor thesis)
├── topcoder/ + data/ + styles/ + fonts/ + m/   # ported legacy pages (TopCoder archive, Harada 9x9)
├── work/                   # collage/gallery images
├── images/books/           # book covers
└── projects/               # project videos/stills used by tiles and write-ups

legacy-hugo/                # the previous Hugo site, archived (not built/served)
```

## Adding content

- **Blog post**: create `src/content/post/<slug>.md` (or `.mdx` to use the prose blocks) with
  frontmatter (`title` ≤60 chars, `description` 50–160 chars, `publishDate`, `tags`). `draft: true`
  shows only in `astro dev`. Math via `$...$` / `$$...$$` (remark-math + rehype-katex). Co-locate
  images and use `![](./img.png)`. Post pages (`layouts/BlogPost.astro`) use a **hand-rolled type
  engine** ported from rajan.sh: `.blog-prose` in `src/styles/prose.css` (NOT
  `@tailwindcss/typography`, shared with project write-ups) with a deliberately flat scale: **Lora**
  body `1.05rem/1.85`, plain-foreground normal-weight headings only modestly larger than body
  (content `h1`≈1.7rem so it never outsizes the Hero title). Header is centered (`Hero.astro`,
  non-spiritual branch: Lora title + description subtitle + author·date). TOC is a **left** sticky
  sidebar with scroll-spy (`TableOfContents.astro`). Footnotes render as **Tufte-style margin
  sidenotes** (float into the right margin ≥1280px, inline below otherwise) via
  `src/utils/rehypeSidenotes.ts`; no bottom footnotes section. Set `citation: true` (optional
  `citationAuthors: "A, B"`) to append a citation block (plaintext + BibTeX) built from
  title/date/URL by `components/blog/Citation.astro`.
- **Prose blocks** (any `.mdx` post or write-up, no import needed): `<Poem title author translator
  source>` (a poem on a sheet of paper; each paragraph is a stanza, line breaks kept), `<Film src
  poster caption>` (click-to-play video), `<Lede>` (highlighted opening paragraph). The draft post
  `src/content/post/prose-blocks.mdx` previews them at `/blog/prose-blocks/` in dev. A new block goes
  in `components/prose/` and into the `components={{...}}` map of both `pages/blog/[slug].astro`
  and `pages/projects/[slug].astro`.
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
- **Where a tile goes** is decided only by `src/utils/destination.ts` (`destinationOf`), in this order:
  1. `slug` set and `src/content/project/<slug>.mdx` exists → the write-up at `/projects/<slug>/`
     (hover panel says "Read the write-up →");
  2. `href` starting with `/` → a page or file on this site (blog post, `/courses/`, a PDF), same tab.
     **The build fails** if it is not a blog post, write-up, file in `public/` or page in `src/pages/`;
  3. any other `href` → another site: new tab, hover panel shows the host with ↗. A full
     `https://taogenna.github.io/...` URL is a page from another repo (e.g. `/projections/`): same tab, unchecked;
  4. neither → the tile is a `<div>`, not a link.
  Never hand-roll an `<a href>` for a project; call `destinationOf` (tiles, write-up "More projects") or
  `resolveHref` (any other link that should be checked, e.g. a write-up's link row).
- **Project write-up**: add `slug: '<slug>'` to the project in `data/projects.ts`, then create
  `src/content/project/<slug>.mdx` with `year`, optional `hero` (`video` + `poster`, or `image`, plus
  `caption`) and optional `links` (`[{ label, href }]`, the row under the hero). Title, subtitle and tags
  come from `data/projects.ts`, so don't repeat them. The build fails if a write-up has no matching project.
  Layout: `layouts/ProjectPage.astro` (after thijs.net/projects). Write only facts from the project's own
  README/paper/repo (STYLE_GUIDE honesty rules); leave motivation for Kenyi to write.
- **Book**: add to `src/data/books.ts` (drop the cover into `public/images/books/`).
- **Work / Adventures item**: add to `src/data/work.ts` (drop image into `public/work/`).
- **Misc / coursework**: edit the arrays at the top of `src/pages/misc.astro`.

## Notes

- Code highlighting: Expressive Code with `light-plus` / `dark-plus`, gated on the `.dark`
  class (config in `src/site.config.ts`). `useDarkModeMediaQuery` is off so code theme
  follows the site toggle, not the OS.
- `/courses/` and `/topcoder/` are directory pages; links use explicit `index.html`
  so they resolve in the Astro dev server (GitHub Pages resolves directories natively).
