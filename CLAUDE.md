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

- Personal mode = light theme; Professional mode = dark theme.
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
│   ├── ProjectCard.astro # project tile (image OR videoPath, tags, optional demo pill)
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
└── projects/             # project demo videos (.mp4) used by ProjectCard videoPath

legacy-hugo/              # the previous Hugo site, archived (not built/served)
```

## Adding content

- **Blog post**: create `src/content/post/<slug>.md` with frontmatter
  (`title` ≤60 chars, `description` 50–160 chars, `publishDate`, `tags`). Math via
  `$...$` / `$$...$$` (remark-math + rehype-katex). Co-locate images and use `![](./img.png)`.
- **Project card**: add a `<ProjectCard>` in `index.astro`. Use `imagePath='/src/assets/x'`
  for a still, or `videoPath='/projects/x.mp4'` (served from `public/`) for an autoplay clip.
  Optional `tags={[{text,color}]}` and `demo='/path'` (gold "See output" pill).
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
