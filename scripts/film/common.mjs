// Shared helpers for the Distill launch-film recorder (see record.mjs).
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

export const W = 1280
export const H = Number(process.env.FILM_H || 720) // 720 = 16:9, 356 = the wide 2x1 tile band (≈3.6:1)
const f = H < 400 ? 0.78 : 1 // caption scale
const here = path.dirname(fileURLToPath(import.meta.url))
const fontFile = path.resolve(here, '../../node_modules/@fontsource-variable/newsreader/files/newsreader-latin-wght-normal.woff2')
const fontItalic = path.resolve(here, '../../node_modules/@fontsource-variable/newsreader/files/newsreader-latin-wght-italic.woff2')
const b64 = (f) => fs.readFileSync(f).toString('base64')

export const PAPER = '#f1efea'
export const INK = '#0b0f14'

// Site palette + Newsreader (inlined so it works on any origin) + mono captions.
export const BASE_CSS = `
@font-face{font-family:'NewsR';src:url(data:font/woff2;base64,${b64(fontFile)}) format('woff2');font-weight:200 800;font-style:normal}
@font-face{font-family:'NewsR';src:url(data:font/woff2;base64,${b64(fontItalic)}) format('woff2');font-weight:200 800;font-style:italic}
:root{--paper:${PAPER};--ink:${INK};--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
*{scrollbar-width:none}*::-webkit-scrollbar{display:none}
`

// Bottom-left caption plate + progress ticks, identical on every scene.
export const captionCss = `
#film-cap{position:fixed;left:${44 * f}px;bottom:${40 * f}px;z-index:99999;display:flex;gap:${16 * f}px;align-items:center;
  font:500 ${21 * f}px/1 var(--mono);letter-spacing:.09em;text-transform:uppercase;background:${INK};color:${PAPER};
  padding:${14 * f}px ${20 * f}px;border-radius:7px;opacity:0;transform:translateY(14px);transition:opacity .45s,transform .45s cubic-bezier(.2,.7,.2,1)}
#film-cap.on{opacity:1;transform:none}
#film-cap i{font-style:normal;opacity:.5}
#film-ticks{position:fixed;right:${44 * f}px;bottom:${56 * f}px;z-index:99999;display:flex;gap:7px}
#film-ticks b{width:${30 * f}px;height:4px;border-radius:2px;background:${INK};opacity:.18}
#film-ticks b.on{opacity:1}
`
export const captionJs = (n, label, total = 5) => `
(() => {
  const c = document.createElement('div'); c.id = 'film-cap'
  c.innerHTML = '<i>0${n}</i><span>${label}</span>'
  const t = document.createElement('div'); t.id = 'film-ticks'
  t.innerHTML = Array.from({length: ${total}}, (_, i) => '<b' + (i < ${n} ? ' class="on"' : '') + '></b>').join('')
  document.body.append(c, t)
  setTimeout(() => c.classList.add('on'), 350)
})()
`

// Camera: wraps the page in #cam and lets us glide/zoom onto any element (vector, so text stays crisp).
export const camJs = `
(() => {
  const body = document.body
  const w = document.createElement('div'); w.id = 'cam'
  w.style.cssText = 'transform-origin:0 0;will-change:transform;width:100%'
  while (body.firstChild) w.appendChild(body.firstChild)
  body.appendChild(w)
  document.documentElement.style.overflow = 'hidden'
  const mat = () => { const m = new DOMMatrixReadOnly(getComputedStyle(w).transform); return { s: m.a || 1, x: m.e, y: m.f } }
  window.cam = (sel, scale, ms = 1000, dx = 0, dy = 0) => {
    const el = typeof sel === 'string' ? document.querySelector(sel) : sel
    const r = el.getBoundingClientRect(), m = mat()
    const cx = (r.left + r.width / 2 - m.x) / m.s, cy = (r.top + r.height / 2 - m.y) / m.s
    w.style.transition = 'transform ' + ms + 'ms cubic-bezier(.62,0,.22,1)'
    w.style.transform = 'translate(' + (innerWidth / 2 - cx * scale + dx) + 'px,' + (innerHeight / 2 - cy * scale + dy * innerHeight / 720) + 'px) scale(' + scale + ')'
  }
  window.camReset = (ms = 800) => { w.style.transition = 'transform ' + ms + 'ms cubic-bezier(.62,0,.22,1)'; w.style.transform = 'none' }
})()
`

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
