// PPCA as a texture. Grid cells show the density of the first n samples of
//   x ~ N(0, I_q),  t | x ~ N(Wx + μ, σ²I)
// (a kernel-density estimate of the real draws), blended from the latent prior's round blob.
// Digits sit only on contour lines (the technique of utils/digitField.ts); the samples are a fixed seed.
export const COLS = 50
export const ROWS = 15
const CW = 5.4 // px per cell at 9px mono
const CH = 10.5
const LEVELS = 6
export const N = 64
const MU = [135, 78]
const S = 50 // |W|: pixels per unit of latent x
const SIG = 7 // σ in pixels
const A = (-28 * Math.PI) / 180
const PRIOR = 22 // the latent blob's std in pixels
const H = 8 // kernel bandwidth

const rng = (a: number) => () => {
	a |= 0
	a = (a + 0x6d2b79f5) | 0
	let t = Math.imul(a ^ (a >>> 15), 1 | a)
	t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
	return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}
const rand = rng(23)
const gauss = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand())

const pts = Array.from({ length: N }, () => {
	const x = Math.max(-2.4, Math.min(2.4, gauss()))
	return [MU[0] + S * x * Math.cos(A) + SIG * gauss(), MU[1] + S * x * Math.sin(A) + SIG * gauss()]
})

const cx = (c: number) => (c + 0.5) * CW
const cy = (r: number) => (r + 0.5) * CH
const kde = (px: number, py: number, n: number) => {
	let s = 0
	for (let k = 0; k < n; k++) s += Math.exp(-((px - pts[k][0]) ** 2 + (py - pts[k][1]) ** 2) / (2 * H * H))
	return s / n
}
let NORM = 1e-6 // final density's peak, so n = N reads as 1.0
for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) NORM = Math.max(NORM, kde(cx(c), cy(r), N))

const hash = (c: number, r: number) => Math.abs(((c * 73856093) ^ (r * 19349663)) % 1000) / 1000

export function field(n: number) {
	const w = n / N
	let digits = ''
	let glow = ''
	let wide = ''
	for (let r = 0; r < ROWS; r++) {
		const lv: number[] = []
		const val: number[] = []
		for (let c = 0; c < COLS; c++) {
			const prior = Math.exp(-((cx(c) - MU[0]) ** 2 + (cy(r) - MU[1]) ** 2) / (2 * PRIOR * PRIOR))
			const v = (1 - w) * prior + (n ? (w * kde(cx(c), cy(r), n)) / NORM : 0)
			val.push(v)
			lv.push(Math.min(Math.floor(v * LEVELS), LEVELS))
		}
		// edge test needs the row below: recompute cheaply from the same formula
		for (let c = 0; c < COLS; c++) {
			const L = lv[c]
			const below = (() => {
				if (r >= ROWS - 1) return L
				const prior = Math.exp(-((cx(c) - MU[0]) ** 2 + (cy(r + 1) - MU[1]) ** 2) / (2 * PRIOR * PRIOR))
				return Math.min(Math.floor(((1 - w) * prior + (n ? (w * kde(cx(c), cy(r + 1), n)) / NORM : 0)) * LEVELS), LEVELS)
			})()
			const edge = L >= 1 && ((c < COLS - 1 && lv[c + 1] !== L) || below !== L)
			const show = edge ? hash(c, r) < 0.66 : L >= 1 && hash(c, r) < 0.03
			digits += show ? String(Math.min(L + 3, 9)) : ' '
			glow += val[c] > 0.2 ? '█' : ' '
			wide += val[c] > 0.06 ? '█' : ' '
		}
		digits += '\n'
		glow += '\n'
		wide += '\n'
	}
	return { digits, glow, wide }
}
