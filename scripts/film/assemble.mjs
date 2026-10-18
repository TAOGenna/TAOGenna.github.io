// Cuts the recorded scenes into the launch film: trims each take, ramps the slow opening
// beats, and joins them with short transitions. Usage: node assemble.mjs <recDir> <out.mp4>
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

const [REC, OUT] = process.argv.slice(2).map((p) => path.resolve(p))
const XF = 0.4 // transition length (s)
const OH = process.env.FILM_H || 720 // output height (width is always 1280)
const scenes = [
	{ n: 's1', ramps: [] },
	{ n: 's2', ramps: [[0, 1.45, 2.2]] }, // static progress bar → time-lapse
	{ n: 's3', ramps: [[0, 0.9, 1.7]] },
	{ n: 's4', ramps: [[0, 1.3, 1.9]] },
	{ n: 's5', end: 3.9 }
]
const trans = ['smoothleft', 'fade', 'smoothleft', 'fadeblack']
const ff = (...a) => execFileSync('ffmpeg', ['-v', 'error', '-y', ...a], { stdio: 'inherit' })
const dur = (f) => parseFloat(execFileSync('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', f]).toString())

const clips = scenes.map((s) => {
	const m = JSON.parse(fs.readFileSync(path.join(REC, `${s.n}.json`)))
	const t0 = m.start, t1 = s.end ? Math.min(m.end, m.start + s.end) : m.end
	const len = t1 - t0
	// split [0,len] at the ramp edges; ramped slices get setpts scaling
	const cuts = [0]
	for (const [a, b] of s.ramps ?? []) cuts.push(a, b)
	cuts.push(len)
	const segs = []
	for (let i = 0; i < cuts.length - 1; i++) {
		const [a, b] = [cuts[i], cuts[i + 1]]
		if (b <= a) continue
		const r = (s.ramps ?? []).find(([x, y]) => x === a && y === b)
		segs.push({ a, b, speed: r ? r[2] : 1 })
	}
	const parts = segs.map((g, i) => `[0:v]trim=${t0 + g.a}:${t0 + g.b},setpts=(PTS-STARTPTS)/${g.speed}[v${i}]`)
	const fg = `${parts.join(';')};${segs.map((_, i) => `[v${i}]`).join('')}concat=n=${segs.length}:v=1,fps=30,scale=1280:${OH},format=yuv420p[o]`
	const out = path.join(REC, `${s.n}.clip.mp4`)
	ff('-i', path.join(REC, `${s.n}.webm`), '-filter_complex', fg, '-map', '[o]', '-c:v', 'libx264', '-crf', '12', '-preset', 'veryfast', out)
	return { file: out, d: dur(out) }
})

// xfade chain
let filt = '', last = '[0:v]', acc = clips[0].d
clips.slice(1).forEach((c, i) => {
	const off = (acc - XF).toFixed(3)
	filt += `${last}[${i + 1}:v]xfade=transition=${trans[i]}:duration=${XF}:offset=${off}[x${i}];`
	last = `[x${i}]`
	acc = acc - XF + c.d
})
filt += `${last}fade=t=out:st=${(acc - 0.35).toFixed(3)}:d=0.35[out]`
ff(...clips.flatMap((c) => ['-i', c.file]), '-filter_complex', filt, '-map', '[out]',
	'-c:v', 'libx264', '-preset', 'slow', '-crf', '26', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', OUT)
console.log('clips', clips.map((c) => c.d.toFixed(2)).join(' + '), '→ total', acc.toFixed(2), 's', (fs.statSync(OUT).size / 1024).toFixed(0), 'KB')
