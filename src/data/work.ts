export interface WorkItem {
	title: string
	image: string // path under /public, e.g. /work/bariloche_lake.jpg
	href?: string // optional external/detail link (e.g. a video)
	video?: boolean // show a play badge; `href` should point to the video
}

// Personal "Adventures" — art, places, and music. Add pieces here; drop the
// image file into /public/work/ with a matching name. For a video, set
// `video: true` and point `href` at it (use a thumbnail as the image).
export const work: WorkItem[] = [
	{ title: 'Ink drawing', image: '/work/art_deer.jpg' },
	{
		title: 'Playing guitar',
		image: '/work/guitar_video.jpg',
		href: 'https://www.youtube.com/watch?v=GmebmPBdBQQ',
		video: true
	},
	{ title: 'Nahuel Huapi, Bariloche', image: '/work/bariloche_lake.jpg' },
	{
		title: 'Playing piano',
		image: '/work/piano_video.jpg',
		href: 'https://www.youtube.com/watch?v=YjEOuWM_Qeo',
		video: true
	},
	{ title: 'Instituto Balseiro in winter', image: '/work/balseiro_snow.webp' }
]
