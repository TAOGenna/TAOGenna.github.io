export interface WorkItem {
	title: string
	image: string // path under /public, e.g. /work/hamparte.png
	href?: string // optional external/detail link
}

// Add pieces here; drop the image file into /public/work/ with a matching name.
export const work: WorkItem[] = [
	{ title: 'Hamparte', image: '/work/hamparte.png' },
	{ title: 'Connect 4', image: '/work/connect4_page.gif' },
	{ title: 'Neural Style Transfer', image: '/work/neural_style_transition.gif' },
	{ title: 'Majorana zero modes', image: '/work/hartree_fock_majorana.png' },
	{ title: 'IonogramNET', image: '/work/ionogram_segmentation.png' },
	{ title: 'LSeg', image: '/work/lseg_project.jpg' },
	{ title: 'Comma compression', image: '/work/comma_comp.jpg' }
]
