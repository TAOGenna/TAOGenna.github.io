// Geometry + painting for the Inversion tile: the repo's real quasi-parabolic inversion of one
// Jicamarca ionogram, step by step (see scripts/tile-data/inversion.py).
import data from '@/data/tiles/inversion.json'

type Step = { label: string; fp: number[]; rh: number[]; rf: number[]; rv: number[] }
export const steps = data.steps as Step[]
export const measured = data.measured as { f: number[]; h: number[] }
export const sounding = data.sounding as string // e.g. 'Sat May 11 11:28:04 2024' (UT)
export const LAST = steps.length - 1

type Box = { x0: number; x1: number; y0: number; y1: number }
export type Cfg = { W: number; H: number; L: Box; R: Box; wide: boolean }
// two layouts of the plots strip on the pale card: wide (desktop) and narrow (phones)
export const CFG: Record<'wide' | 'narrow', Cfg> = {
	wide: { W: 480, H: 150, wide: true, L: { x0: 22, x1: 200, y0: 18, y1: 116 }, R: { x0: 280, x1: 458, y0: 18, y1: 116 } },
	narrow: { W: 300, H: 120, wide: false, L: { x0: 6, x1: 136, y0: 6, y1: 112 }, R: { x0: 164, x1: 294, y0: 6, y1: 112 } }
}
// axis ranges: ionogram (MHz, km virtual) and profile (MHz, km true height)
export const F: [number, number] = [1, 9.5]
export const VH: [number, number] = [80, 720]
export const FP: [number, number] = [0, 9.5]
export const RH: [number, number] = [60, 400]

export const sx = (b: Box, v: number, [a, c]: [number, number]) => b.x0 + ((v - a) / (c - a)) * (b.x1 - b.x0)
export const sy = (b: Box, v: number, [a, c]: [number, number]) => b.y1 - ((v - a) / (c - a)) * (b.y1 - b.y0)
const f1 = (n: number) => n.toFixed(1)

export function view(cfg: Cfg, k: number) {
	const s = steps[k]
	const dots = s.rf.map((f, i) => `M${f1(sx(cfg.L, f, F))} ${f1(sy(cfg.L, s.rv[i], VH))}h0`).join('')
	const pts = s.fp.map((f, i) => [sx(cfg.R, f, FP), sy(cfg.R, s.rh[i], RH)])
	const prof = pts.map(([x, y], i) => `${i ? 'L' : 'M'}${f1(x)} ${f1(y)}`).join('')
	const tip = pts[pts.length - 1]
	const last = s.fp.length - 1
	return {
		dots,
		prof,
		tip,
		label: s.label.replace('  ·  ', ' · ').toUpperCase(),
		num: s.fp[last].toFixed(2),
		at: s.rh[last] > RH[0] ? `AT ${Math.round(s.rh[last])} KM` : '—', // E-layer candidates start below the axis
		frac: (k + 1) / steps.length,
		step: `STEP ${k + 1} / ${steps.length}`
	}
}

// svg-specific geometry (differs per layout)
export function paint(svg: SVGSVGElement, cfg: Cfg, k: number) {
	const v = view(cfg, k)
	const q = (s: string) => svg.querySelector(s)!
	q('[data-dots]').setAttribute('d', v.dots)
	q('[data-prof]').setAttribute('d', v.prof)
	q('[data-tip]').setAttribute('transform', `translate(${f1(v.tip[0])} ${f1(v.tip[1])})`)
}

// shared readout (HTML around the plots: numeral, labels, progress)
export function paintText(root: HTMLElement, k: number) {
	const v = view(CFG.wide, k)
	const set = (sel: string, t: string) => root.querySelectorAll(sel).forEach((el) => (el.textContent = t))
	set('[data-num]', v.num)
	set('[data-at]', v.at)
	set('[data-label]', v.label)
	set('[data-step]', v.step)
	root.querySelectorAll<HTMLElement>('[data-bar]').forEach((el) => (el.style.width = `${(v.frac * 100).toFixed(1)}%`))
}
