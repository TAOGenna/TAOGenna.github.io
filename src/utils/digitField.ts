// Drifting topographic field (the "ASCII number structures" from the reference):
// digits sit only on the contour lines of a few drifting soft blobs; a second
// layer of blocks (blurred in CSS) is the glow underneath.
export const COLS = 50
export const ROWS = 15
const CW = 5.4, // px per cell at 9px mono
	CH = 10.5
const LEVELS = 6
const blobs = [
	{ x: 60, y: 60, s: 38, sp: 0.5, ph: 0 },
	{ x: 150, y: 90, s: 46, sp: 0.37, ph: 2 },
	{ x: 225, y: 50, s: 34, sp: 0.61, ph: 4 },
	{ x: 105, y: 130, s: 30, sp: 0.43, ph: 1 }
]

function f(px: number, py: number, t: number) {
	let v = 0
	for (const b of blobs) {
		const bx = b.x + 26 * Math.sin(t * b.sp + b.ph)
		const by = b.y + 18 * Math.cos(t * b.sp * 0.8 + b.ph)
		v += Math.exp(-((px - bx) ** 2 + (py - by) ** 2) / (2 * b.s * b.s))
	}
	return v
}

export function field(t: number) {
	const lv: number[][] = []
	const val: number[][] = []
	for (let r = 0; r < ROWS; r++) {
		lv.push([])
		val.push([])
		for (let c = 0; c < COLS; c++) {
			const v = f((c + 0.5) * CW, (r + 0.5) * CH, t)
			val[r].push(v)
			lv[r].push(Math.min(Math.floor(v * LEVELS), LEVELS))
		}
	}
	// fixed per-cell hash -> sparse, non-flickering dotted contours + a few strays
	const h = (c: number, r: number) => ((c * 73856093) ^ (r * 19349663)) % 1000 / 1000
	let digits = ''
	let glow = ''
	for (let r = 0; r < ROWS; r++) {
		for (let c = 0; c < COLS; c++) {
			const L = lv[r][c]
			const edge = L >= 1 && ((c < COLS - 1 && lv[r][c + 1] !== L) || (r < ROWS - 1 && lv[r + 1][c] !== L))
			const show = edge ? Math.abs(h(c, r)) < 0.62 : L >= 1 && Math.abs(h(c, r)) < 0.025
			digits += show ? String(Math.min(L + 3, 9)) : ' '
			glow += val[r][c] > 0.5 ? '█' : ' '
		}
		digits += '\n'
		glow += '\n'
	}
	return { digits, glow }
}
