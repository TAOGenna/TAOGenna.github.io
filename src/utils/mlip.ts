// A tiny Behler–Parrinello environment: one central atom, neighbors with a cutoff function
// f_c(R) = ½(cos(πR/R_c) + 1), and radial symmetry functions
//   G_m = Σ_j exp(-η (R_j - R_s,m)²) f_c(R_j).
// Neighbors drift and the whole cluster rotates; the G values (computed live) don't care about rotation.
export const RC = 50 // cutoff radius in px
export const CX = 62
export const CY = 84
export const NG = 8
export const RS = Array.from({ length: NG }, (_, m) => 0.12 + m * 0.115) // in units of R_c
const ETA = 38
export const N = 10
const base = Array.from({ length: N }, (_, j) => ({
	phi: (j * 2.399963) % (2 * Math.PI), // golden angle
	r0: 0.34 + 0.95 * ((j * 0.618) % 1)
}))
export const fc = (r: number) => (r < 1 ? 0.5 * (Math.cos(Math.PI * r) + 1) : 0)

export function env(t: number) {
	const spin = 0.32 * t
	const atoms = base.map((b, j) => {
		const r = b.r0 + 0.09 * Math.sin(t * (0.6 + 0.11 * j) + j * 1.7)
		const phi = b.phi + 0.1 * Math.sin(t * (0.4 + 0.07 * j) + j * 2.3) + spin
		return { x: CX + RC * r * Math.cos(phi), y: CY + RC * r * Math.sin(phi), r, f: fc(r) }
	})
	const G = RS.map((rs) => atoms.reduce((s, a) => s + Math.exp(-ETA * (a.r - rs) ** 2) * a.f, 0))
	return { atoms, G }
}
