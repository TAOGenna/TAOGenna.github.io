/**
 * rehypeCaptions — mark a paragraph whose sole meaningful content is a single
 * `<em>` (a caption or a standalone italic line) with `class="blog-caption"`.
 *
 * This replaces the fragile CSS selector `p > em:only-child`, which also matches
 * inline emphasis mid-sentence: `:only-child` ignores text-node siblings, so
 * `<p>text <em>x</em> text</p>` wrongly qualifies. Checking in hast (where we can
 * see the text nodes) lets us center genuine captions without touching inline
 * `*emphasis*`.
 */

type HastNode = {
	type: string
	tagName?: string
	value?: string
	properties?: Record<string, unknown>
	children?: HastNode[]
}

const isBlankText = (n: HastNode) => n.type === 'text' && !n.value?.trim()

export function rehypeCaptions() {
	return (tree: HastNode) => {
		const walk = (node: HastNode) => {
			if (!node.children) return
			if (node.type === 'element' && node.tagName === 'p') {
				const meaningful = node.children.filter((c) => !isBlankText(c))
				if (meaningful.length === 1 && meaningful[0].type === 'element' && meaningful[0].tagName === 'em') {
					const em = meaningful[0]
					em.properties = em.properties ?? {}
					const cls = em.properties.className
					em.properties.className = Array.isArray(cls) ? [...cls, 'blog-caption'] : ['blog-caption']
				}
			}
			for (const child of node.children) walk(child)
		}
		walk(tree)
	}
}
