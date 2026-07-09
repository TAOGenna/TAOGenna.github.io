export interface ProjectTag {
	text: string
	color?: string
	outline?: boolean
}

export interface Project {
	heading: string
	href: string
	subheading: string
	altText: string
	imagePath?: string
	videoPath?: string
	tags?: ProjectTag[]
	demo?: string
	demoLabel?: string
}

// Order matters: the homepage shows the first `HOMEPAGE_COUNT`, the full list
// lives at /projects. Newest / most representative work first.
export const HOMEPAGE_COUNT = 6

export const projects: Project[] = [
	{
		heading: 'YaVendio',
		href: 'https://yavendio.com',
		subheading: 'Conversational AI agents that sell and support customers over WhatsApp.',
		imagePath: '/src/assets/yavendio.png',
		altText: 'YaVendio'
	},
	{
		heading: 'Majorana zero modes',
		href: 'https://arxiv.org/abs/2309.10888',
		subheading: 'Self-consistent Hartree-Fock for a quantum-dot–superconducting-nanowire hybrid.',
		videoPath: '/projects/majorana_sweep.mp4',
		altText: 'Majorana zero modes',
		tags: [
			{ text: 'Phys. Rev. B', color: '#7f1d2e' },
			{ text: 'arXiv', color: '#7f1d2e' }
		]
	},
	{
		heading: 'Olive',
		href: 'https://github.com/YaVendio/olive',
		subheading: 'A decorator that exposes Python functions as remote tools for AI agents.',
		imagePath: '/src/assets/olive_architecture.png',
		altText: 'Olive',
		tags: [{ text: 'OSS contributor', color: '#0f766e' }]
	},
	{
		heading: 'Inversion of ionograms',
		href: 'https://arxiv.org/abs/2411.09215',
		subheading: 'Inversion algorithm for plasma frequency profiles.',
		videoPath: '/projects/inversion_progress.mp4',
		altText: 'Ionogram inversion',
		tags: [{ text: 'arXiv', color: '#7f1d2e' }]
	},
	{
		heading: 'Language-driven Segmentation',
		href: 'https://github.com/TAOGenna/pytorch-language-driven-semantic-segmentation',
		subheading:
			'From-scratch DPT encoder + CLIP text encoder in a shared multimodal latent space (LSeg, ICML 2022).',
		videoPath: '/projects/lseg_progress.mp4',
		altText: 'LSeg',
		tags: [{ text: 'Paper reproduction', color: '#4338ca' }]
	},
	{
		heading: 'MCTS for Connect 4',
		href: 'https://github.com/TAOGenna/Connect4-MonteCarlo',
		subheading: 'A playable game with an AI opponent powered by Monte Carlo Tree Search and UCB.',
		videoPath: '/projects/connect4_progress.mp4',
		altText: 'Connect 4',
		tags: [{ text: 'Prototype', color: '#8a6d2f' }]
	},
	{
		heading: 'Distill',
		href: 'https://github.com/TAOGenna/Distill',
		subheading:
			'AI that turns any URL into a full progressive course with lessons, exercises, and solutions.',
		videoPath: '/projects/distill_small_demo.mp4',
		altText: 'Distill demo',
		demo: '/courses/index.html',
		tags: [{ text: 'Prototype', color: '#8a6d2f' }]
	},
	{
		heading: 'Neural Style Transfer',
		href: 'https://github.com/TAOGenna/pytorch-neural-style-transfer',
		subheading: 'Artistic texture extraction via Gram matrices and VGG-19 features (Gatys et al.).',
		videoPath: '/projects/style_transfer_progress.mp4',
		altText: 'Neural style transfer',
		tags: [{ text: 'Paper reproduction', color: '#4338ca' }]
	},
	{
		heading: 'IonogramNET',
		href: 'https://github.com/TAOGenna/ionospheric-echo-detection-with-convolutional-neural-networks',
		subheading:
			'Encoder-decoder CNN for extracting echo traces from Jicamarca Observatory ionograms.',
		imagePath: '/src/assets/ionogram_segmentation.png',
		altText: 'Raw ionogram beside the U-Net predicted echo trace',
		tags: [{ text: 'Paper reproduction', color: '#4338ca' }]
	},
	{
		heading: 'MAHO',
		href: 'https://maho.kenyi-rtp.workers.dev/',
		subheading:
			'Compiles your project markdown into a living garden, where each project is an island that blooms when you give it focused time and one honest line, and wilts when you don’t.',
		videoPath: '/projects/maho_hero.mp4',
		altText: 'MAHO garden of projects',
		tags: [{ text: 'Prototype', color: '#8a6d2f' }]
	},
	{
		heading: 'Holographic Entanglement Entropy',
		href: '/Files/bachelor_thesis.pdf',
		subheading:
			'Bachelor’s thesis on holographic entanglement entropy and the Ryu-Takayanagi prescription in AdS/CFT (PUCP, 2021).',
		imagePath: '/src/assets/bachelor_thesis.png',
		altText: 'Ryu-Takayanagi surface in AdS/CFT',
		tags: [{ text: 'Thesis', outline: true }]
	}
]
