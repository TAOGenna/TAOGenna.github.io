// Replay of the repo's MCTS self-play (see scripts/tile-data/connect4.py).
import data from '@/data/tiles/connect4.json'

export type Move = { r: number; c: number; p: number; visits: number[] }
export const moves = data.moves as Move[]
export const win = data.win as [number, number][]
export const START = 17

export const CX = (c: number) => 20 + c * 18.5
export const CY = (r: number) => 140 - r * 18.5
export const BAR_X = (c: number) => 163 + c * 13.7
const BAR_BASE = 128,
	BAR_MAX = 58
const INK = '#0b0f14',
	STEEL = '#8fa9b8'

// k = moves shown; showWin lights the winning four in amber.
export function paint(svg: SVGSVGElement, k: number, showWin = false) {
	const grid = new Map<string, number>()
	moves.slice(0, k).forEach((m) => grid.set(`${m.r},${m.c}`, m.p))
	svg.querySelectorAll<SVGCircleElement>('[data-cell]').forEach((el) => {
		const p = grid.get(el.dataset.cell!)
		el.setAttribute('fill', p === 1 ? INK : p === -1 ? STEEL : 'none')
		el.setAttribute('stroke-opacity', p ? '0' : '0.35')
	})
	const last = k > 0 ? moves[k - 1] : null
	const ring = svg.querySelector('[data-last]')!
	ring.setAttribute('opacity', last ? '1' : '0')
	if (last) {
		ring.setAttribute('cx', String(CX(last.c)))
		ring.setAttribute('cy', String(CY(last.r)))
	}
	const max = last ? Math.max(...last.visits, 1) : 1
	svg.querySelectorAll<SVGRectElement>('[data-bar]').forEach((bar) => {
		const c = Number(bar.dataset.bar)
		const h = last ? (last.visits[c] / max) * BAR_MAX : 0
		bar.setAttribute('y', String(BAR_BASE - h))
		bar.setAttribute('height', String(h))
		bar.setAttribute('fill', last && last.c === c ? INK : STEEL)
	})
	svg.querySelector('[data-move]')!.textContent = `move ${k}`
	const w = svg.querySelector('[data-win]')!
	w.setAttribute('opacity', showWin ? '1' : '0')
}
