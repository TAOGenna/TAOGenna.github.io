// Records the Distill launch film: one real-time Playwright capture per scene, from the
// real Distill UI + the real generated course, restyled with injected CSS (our palette).
// Usage:  PW_DIR=<dir containing node_modules/playwright-core> node record.mjs <outDir> [scene ...]
//   needs: Distill UI on :8420 (uv run python -m distill --no-open), astro dev on :4321,
//          a Chromium (CHROME env var or the Playwright cache path below).
import { createRequire } from 'node:module'
import fs from 'node:fs'
import path from 'node:path'
import { W, H, BASE_CSS, captionCss, captionJs, camJs, sleep, PAPER, INK } from './common.mjs'

const { chromium } = createRequire(path.join(process.env.PW_DIR, 'x.js'))('playwright-core')
const CHROME =
	process.env.CHROME ||
	`${process.env.HOME}/Library/Caches/ms-playwright/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing`
const OUT = path.resolve(process.argv[2])
const want = process.argv.slice(3)
const WIDE = H < 400
const FORM = 'http://127.0.0.1:8420/'
const COURSE = 'http://localhost:4321/courses/differentiable_equilibrium_averages_from_zwanzig_t/index.html'

// Restyle the Distill form to the site's editorial look (variables + a few overrides).
const FORM_CSS = `
:root{--fg:${INK};--fg-dim:#46505b;--fg-muted:#7b8591;--fg-faint:#9aa3ad;--bg:${PAPER};--bg-panel:#e8e5de;
 --border:#cdc9c0;--border-light:#dbd7ce;--border-faint:#e4e0d8;--input-border:${INK};--btn-bg:${INK};--btn-fg:${PAPER};
 --track-bg:#dcd8cf;--edge-color:#cfcbc2;--edge-done:#8fa9b8;--accent-ok:${INK};--accent-ok-bg:#e8e5de;--accent-ok-border:#cdc9c0;--code-bg:rgba(0,0,0,.06)}
html{border-top:0!important}
body{background:${PAPER}}
h1{font-family:NewsR,serif!important;font-weight:300!important;font-size:64px!important;letter-spacing:-.02em;line-height:1!important}
.tagline{font-family:NewsR,serif;font-size:19px!important;font-style:italic;color:#46505b!important}
.github-badge,.github-badge-link,#theme-toggle,#setup-panel,.presets-bar,#add-ref,#refs,.field-hint,#courses-section{display:none!important}
.btn{border-radius:4px;font-size:16px!important;padding:14px!important}
#url-input{font-size:21px!important}#level-input{font-size:19px!important}
label,.label-row label{font-size:13px!important}select{font-size:15px!important}
.container{max-width:760px!important}
/* generating screen: grayscale iso icons, lavender once generated */
.flow-icon svg{filter:grayscale(1) contrast(1.05)}
.flow-node.active .flow-icon svg,.flow-node.readable .flow-icon svg,.flow-node.generated .flow-icon svg{filter:hue-rotate(95deg) saturate(.45)}
.flow-label{font-family:NewsR,serif!important;font-size:17px!important}
.flow-sub{font-size:12px!important}
`

const READER_CSS = `
:root{--r-bg:${PAPER};--r-fg:${INK};--r-fg-dim:#46505b;--r-fg-muted:#7b8591;--r-border:#cdc9c0;--r-accent:${INK};
 --r-code-bg:${INK};--r-code-border:${INK};--r-header-bg:rgba(241,239,234,.92);
 --c-keyword:#cfd2f5;--c-string:#b9d3dc;--c-comment:#7f8a96;--c-number:#d8ece4;--c-function:#cfd2f5;--c-builtin:#8fa9b8;--c-class:#cfd2f5}
html{font-size:16px!important}
body{font-family:NewsR,Georgia,serif!important;padding-left:0!important;max-width:none!important}
header,#sidebar,.sidebar,.pager{display:none!important}
h1,h2,h3,h4{font-family:NewsR,serif!important;font-weight:300!important;letter-spacing:-.01em}
#article{max-width:760px;margin:0 auto!important;padding:70px 0 140px!important}
#article a{color:${INK}!important;text-decoration:underline;text-decoration-color:#8fa9b8}
pre,code,.exercise-header{font-family:ui-monospace,SFMono-Regular,Menlo,monospace!important}
.hljs,pre code{color:#e9edf0!important}
.hljs-attr,.hljs-variable,.hljs-template-variable,.hljs-params{color:#e9edf0!important}
.exercise-card{border-color:#cdc9c0!important}
`

async function withPage(name, fn) {
	const dir = path.join(OUT, name)
	fs.rmSync(dir, { recursive: true, force: true })
	const ctx = await browser.newContext({ viewport: { width: W, height: H }, recordVideo: { dir, size: { width: W, height: H } } })
	const page = await ctx.newPage()
	const t0 = Date.now()
	const mark = {}
	await fn(page, mark, t0)
	await ctx.close()
	const file = fs.readdirSync(dir).find((f) => f.endsWith('.webm'))
	fs.renameSync(path.join(dir, file), path.join(OUT, `${name}.webm`))
	fs.rmSync(dir, { recursive: true, force: true })
	fs.writeFileSync(path.join(OUT, `${name}.json`), JSON.stringify(mark))
	console.log(name, 'start', mark.start, 'end', mark.end)
}
const now = (t0) => (Date.now() - t0) / 1000

const scenes = {
	// 01 — fill in the real form: URL, background, press generate
	async s1(page, mark, t0) {
		await page.goto(FORM, { waitUntil: 'networkidle' })
		await page.addStyleTag({ content: BASE_CSS + captionCss + FORM_CSS })
		await page.evaluate(camJs)
		await page.evaluate(() => document.addEventListener('submit', (e) => { e.preventDefault(); e.stopImmediatePropagation() }, true)) // never start a real job
		await page.evaluate(captionJs(1, 'Paste a source'))
		await page.evaluate(() => document.fonts.ready)
		await page.evaluate(() => window.cam('h1', 1.12, 0, 0, 170))
		await sleep(1300)
		mark.start = now(t0)
		await sleep(250)
		await page.evaluate(() => window.cam('#url-input', 1.55, 650, 0, -60))
		await sleep(700)
		await page.locator('#url-input').pressSequentially('https://github.com/cusp-ai-oss/kups', { delay: 20 })
		await sleep(250)
		await page.evaluate(() => window.cam('#level-input', 1.55, 600, 0, -60))
		await sleep(650)
		await page.locator('#level-input').pressSequentially('physics + PyTorch, new to molecular simulation', { delay: 17 })
		await sleep(250)
		await page.evaluate(() => window.cam('.field-row', 1.4, 600, 0, -150))
		await sleep(750)
		await page.evaluate(() => { const b = document.querySelector('#go'); b.style.transition = 'transform .12s'; b.style.transform = 'scale(.97)'; b.textContent = 'generating…' })
		await sleep(500)
		mark.end = now(t0)
		await sleep(200)
	},

	// 02 — the real generating screen (DAG renderer + progress bar), fed a correct sparse curriculum
	async s2(page, mark, t0) {
		await page.goto(FORM, { waitUntil: 'networkidle' })
		await page.addStyleTag({ content: BASE_CSS + captionCss + FORM_CSS + '#form,.tagline{display:none!important}' })
		await page.evaluate(camJs)
		await page.evaluate(captionJs(2, 'Blueprint · generate · review'))
		await page.evaluate(() => {
			// roomier layers so node labels don't collide (stock spacing is 140px)
			;(0, eval)(renderCurriculumFlow.toString().replace('var layerSpacing = 140;', 'var layerSpacing = 190;'))
			progressEl.classList.add('active')
			document.querySelector('#log-section')?.classList.add('collapsed')
			genStart = Date.now()
			updateProgressBar('preprocess', '')
			window.__emit = (ev) => {
				if (ev.type === 'phase') { updateProgressBar(ev.phase, ''); appendPhase(ev.phase) }
				else if (ev.type === 'curriculum') {
					totalModules = ev.data.modules.length
					flowCourseName = 'differentiable_equilibrium_averages'
					const st = window.setTimeout
					window.setTimeout = (f, ms, ...a) => st(f, ms / 1.8, ...a) // time-lapse the DAG draw-in
					renderCurriculumFlow(ev.data)
					setTimeout(() => { window.setTimeout = st }, 600)
				} else if (ev.type === 'module_start') { activeModuleSet.add(ev.i); markFlowNodeActive(ev.i) }
				else if (ev.type === 'lesson_ready') { activeModuleSet.delete(ev.i); lessonsCompletedSet.add(ev.i); markFlowNodeReadable(ev.i, 'x'); updateLessonProgress() }
				else if (ev.type === 'module_complete') { activeModuleSet.delete(ev.i); completedModuleSet.add(ev.i); updateModuleProgress(); activateFlowNode(ev.i); updateEdgeStates() }
			}
		})
		// Sparse, realistic prerequisites (the generated course's own depends_on is cumulative — a Distill bug)
		const data = {
			title: 'Differentiable Equilibrium Averages: From Zwanzig to Sampler-Agnostic Gradients for MOF Design',
			modules: [
				{ index: 0, title: 'Score-Function Identity', exercise_count: 4, depends_on: [] },
				{ index: 1, title: 'GCMC Sampler + Estimator', exercise_count: 4, depends_on: [0] },
				{ index: 2, title: 'Working Capacity ΔN', exercise_count: 1, depends_on: [1] },
				{ index: 3, title: 'Structural Parameters', exercise_count: 4, depends_on: [1] },
				{ index: 4, title: 'Sampler-Agnostic Design', exercise_count: 0, depends_on: [2, 3] }
			]
		}
		await page.evaluate(() => window.cam('#phase-bar', 1.55, 0, 0, 120))
		await sleep(700)
		mark.start = now(t0)
		const ev = (e) => page.evaluate((x) => window.__emit(x), e)
		const cam = (sel, sc, ms, dy = 0) => page.evaluate(([s, c, m, d]) => window.cam(typeof s === 'number' ? flowNodes[s] : s, c, m, 0, d), [sel, sc, ms, dy])
		await sleep(250)
		await ev({ type: 'phase', phase: 'analyze' })
		await sleep(400)
		await ev({ type: 'phase', phase: 'design' })
		await sleep(450)
		await ev({ type: 'curriculum', data })
		await sleep(150)
		await cam('.course-flow', 1.5, 700, 330)
		await sleep(500)
		await ev({ type: 'phase', phase: 'generate_lessons' })
		await ev({ type: 'module_start', i: 0 }); await cam(0, 1.6, 700, -40)
		await sleep(650)
		await ev({ type: 'lesson_ready', i: 0 }); await ev({ type: 'module_start', i: 1 }); await cam(1, 1.6, 700, -40)
		await sleep(650)
		await ev({ type: 'lesson_ready', i: 1 }); await ev({ type: 'module_start', i: 2 }); await ev({ type: 'module_start', i: 3 })
		await cam(2, 1.45, 700, -40)
		await sleep(650)
		await ev({ type: 'lesson_ready', i: 2 }); await ev({ type: 'lesson_ready', i: 3 }); await ev({ type: 'module_start', i: 4 })
		await cam(4, 1.6, 700, -40)
		await sleep(550)
		await ev({ type: 'phase', phase: 'generate_exercises' })
		for (const i of [0, 1]) await ev({ type: 'module_complete', i })
		await sleep(280)
		for (const i of [2, 3]) await ev({ type: 'module_complete', i })
		await ev({ type: 'lesson_ready', i: 4 })
		await sleep(280)
		await ev({ type: 'module_complete', i: 4 })
		await page.evaluate(() => window.cam('.course-flow', 0.68 * Math.min(1, innerHeight / 720 * 1.45), 900, 0, 0))
		await sleep(450)
		await ev({ type: 'phase', phase: 'review' })
		await sleep(650)
		mark.end = now(t0)
		await sleep(200)
	},

	// 03 — the lesson: a real generated module, typeset math
	async s3(page, mark, t0) {
		await page.goto(COURSE, { waitUntil: 'networkidle' })
		await page.addStyleTag({ content: BASE_CSS + captionCss + READER_CSS })
		await page.waitForSelector('#article h1')
		await page.evaluate(camJs)
		await page.evaluate(captionJs(3, 'A lesson, not a summary'))
		await page.evaluate(() => window.cam('#article h1', 1.5, 0, 0, 130))
		await sleep(700)
		mark.start = now(t0)
		await sleep(900)
		await page.evaluate(() => {
			const h = [...document.querySelectorAll('#article h2')].find((x) => /Zwanzig Identity in Full/.test(x.textContent))
			let n = h.nextElementSibling
			while (n && !n.querySelector('.katex-display') && !n.classList.contains('katex-display')) n = n.nextElementSibling
			window.__eq = n.querySelector('.katex-display') || n
			window.cam(h, 1.5, 1500, 0, 90)
		})
		await sleep(1700)
		await page.evaluate(() => window.cam(window.__eq, 1.65, 1200, 0, -10))
		await sleep(1500)
		mark.end = now(t0)
		await sleep(200)
	},

	// 04 — exercises you can run: open a scaffold with TODOs
	async s4(page, mark, t0) {
		await page.goto(COURSE, { waitUntil: 'networkidle' })
		await page.addStyleTag({ content: BASE_CSS + captionCss + READER_CSS })
		await page.waitForSelector('#article h1')
		await page.evaluate(() => loadModule(1))
		await page.waitForSelector('.exercises-section')
		await page.evaluate(camJs)
		await page.evaluate(captionJs(4, 'Exercises that run'))
		await page.evaluate(() => window.cam('.exercises-section > h2', 1.45, 0, 0, 120))
		await sleep(700)
		mark.start = now(t0)
		await sleep(700)
		await page.evaluate(() => {
			const hd = [...document.querySelectorAll('.exercise-header')].find((x) => /ex03_online_covariance/.test(x.textContent))
			window.__hd = hd
			window.cam(hd, 1.5, 700, 0, 20)
		})
		await sleep(850)
		await page.evaluate(() => window.__hd.click())
		await sleep(500)
		await page.evaluate(() => {
			const body = window.__hd.nextElementSibling
			const todo = [...body.querySelectorAll('.hljs-comment')].find((x) => /TODO/.test(x.textContent)) || body.querySelector('pre')
			window.cam(todo, 1.9, 1300, 0, -30)
		})
		await sleep(1800)
		mark.end = now(t0)
		await sleep(200)
	},

	// 05 — the payoff: the lesson's own verification gate
	async s5(page, mark, t0) {
		const html = `<!doctype html><meta charset=utf-8><style>${BASE_CSS}${captionCss}
html,body{margin:0;height:100%;background:${INK};color:${PAPER};overflow:hidden}
.wrap{position:absolute;inset:0 0 ${WIDE ? 0 : 110}px 0;display:grid;grid-template-columns:1.3fr 1fr;align-items:center;gap:36px;padding:0 ${WIDE ? 64 : 84}px}
h1{font:300 ${WIDE ? 70 : 100}px/0.98 NewsR,serif;letter-spacing:-.03em;margin:0;white-space:nowrap;opacity:0;animation:up .8s .15s forwards cubic-bezier(.2,.7,.2,1)}
.card{background:#141a22;border:1px solid #243040;border-radius:12px;padding:${WIDE ? 18 : 34}px ${WIDE ? 28 : 38}px;opacity:0;animation:in .4s .9s forwards}
.card small{display:block;font:500 ${WIDE ? 14 : 17}px var(--mono);letter-spacing:.1em;text-transform:uppercase;color:#7f8a96;margin-bottom:${WIDE ? 10 : 22}px}
pre{font:500 ${WIDE ? 22 : 29}px/${WIDE ? 1.5 : 1.65} var(--mono);margin:0;color:#e9edf0}
pre span{display:block;opacity:0;animation:in .01s forwards}
.pass{color:#cfd2f5}
.meta{margin-top:${WIDE ? 14 : 26}px;font:500 18px var(--mono);letter-spacing:.09em;text-transform:uppercase;color:#8fa9b8;opacity:0;animation:in .5s 3.0s forwards}
#film-cap{background:${PAPER};color:${INK}}#film-ticks b{background:${PAPER}}
@keyframes up{from{opacity:0;transform:translateY(26px)}to{opacity:1;transform:none}}@keyframes in{to{opacity:1}}
</style>
<div class=wrap><div><h1>Paste a URL.<br>Get a course.</h1><div class=meta>5 modules · 13 exercises</div></div>
<div class=card><small>module 3 · exercise 4 · verification</small>
<pre><span style="animation-delay:1.3s">d⟨N⟩/da</span><span style="animation-delay:1.6s">  estimator = −0.031</span><span style="animation-delay:1.9s">  FD        = −0.033</span><span style="animation-delay:2.2s">  rel_err   = 0.06   sign ✓</span><span class=pass style="animation-delay:2.7s">Gate: PASS</span></pre></div></div>`
		await page.route('http://film.local/end', (r) => r.fulfill({ contentType: 'text/html', body: html }))
		await page.goto('http://film.local/end')
		await page.evaluate(() => document.fonts.ready)
		await page.evaluate(captionJs(5, 'Output you can verify'))
		mark.start = now(t0) + 0.1
		await sleep(4300)
		mark.end = now(t0)
	}
}

const browser = await chromium.launch({ executablePath: CHROME, headless: true })
fs.mkdirSync(OUT, { recursive: true })
for (const [name, fn] of Object.entries(scenes)) if (!want.length || want.includes(name)) await withPage(name, fn)
await browser.close()
