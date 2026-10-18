// Marching squares -> one SVG path of hairline segments (the topographic
// contour language from spatiali.se). Build-time only.
const CASES: number[][][] = [[], [[3, 2]], [[2, 1]], [[3, 1]], [[0, 1]], [[3, 0], [2, 1]], [[0, 2]], [[3, 0]], [[3, 0]], [[0, 2]], [[3, 2], [0, 1]], [[0, 1]], [[3, 1]], [[2, 1]], [[3, 2]], []]

export function contourPath(w: number, h: number, cell: number, levels: number[], fn: (x: number, y: number) => number) {
	const nx = Math.ceil(w / cell) + 1
	const ny = Math.ceil(h / cell) + 1
	const v = Array.from({ length: ny }, (_, j) => Array.from({ length: nx }, (_, i) => fn(i * cell, j * cell)))
	const f = (n: number) => n.toFixed(1)
	let d = ''
	for (const L of levels)
		for (let j = 0; j < ny - 1; j++)
			for (let i = 0; i < nx - 1; i++) {
				const [tl, tr, br, bl] = [v[j][i], v[j][i + 1], v[j + 1][i + 1], v[j + 1][i]]
				const idx = (tl > L ? 8 : 0) | (tr > L ? 4 : 0) | (br > L ? 2 : 0) | (bl > L ? 1 : 0)
				const x = i * cell
				const y = j * cell
				const t = (a: number, b: number) => (L - a) / (b - a)
				// edge crossings: 0 top, 1 right, 2 bottom, 3 left
				const pt = [
					[x + cell * t(tl, tr), y],
					[x + cell, y + cell * t(tr, br)],
					[x + cell * t(bl, br), y + cell],
					[x, y + cell * t(tl, bl)]
				]
				for (const [a, b] of CASES[idx]) d += `M${f(pt[a][0])} ${f(pt[a][1])}L${f(pt[b][0])} ${f(pt[b][1])}`
			}
	return d
}
