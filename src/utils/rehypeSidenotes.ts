/**
 * rehypeSidenotes — turn GFM footnotes into Tufte-style margin sidenotes.
 *
 * Authors keep writing standard markdown footnotes:
 *
 *     Some claim.[^1]
 *
 *     [^1]: The supporting aside, which can contain $math$ and [links](…).
 *
 * This plugin runs AFTER rehype-katex (so math inside a note is already
 * rendered) and rewrites the tree so that each footnote definition is moved
 * inline right after its reference, wrapped in `<span class="sidenote">`, and
 * the bottom `<section data-footnotes>` block is removed. CSS in BlogPost.astro
 * floats the span into the right margin on wide screens and shows it as an
 * indented block on narrow screens.
 *
 * No unist/hast helper deps — a plain recursive walk keeps this self-contained.
 */

type HastNode = {
	type: string
	tagName?: string
	value?: string
	properties?: Record<string, unknown>
	children?: HastNode[]
}

const BLOCK_TAGS = new Set(['p', 'li', 'blockquote', 'td', 'th', 'figcaption'])

function addClass(node: HastNode, className: string) {
	node.properties = node.properties ?? {}
	const existing = node.properties.className
	if (Array.isArray(existing)) existing.push(className)
	else if (typeof existing === 'string') node.properties.className = [existing, className]
	else node.properties.className = [className]
}

export function rehypeSidenotes() {
	return (tree: HastNode) => {
		const children = tree.children ?? []

		// 1. Locate the footnotes section (`<section data-footnotes>`).
		const sectionIndex = children.findIndex(
			(n) => n.type === 'element' && n.properties != null && 'dataFootnotes' in n.properties
		)
		if (sectionIndex === -1) return
		const section = children[sectionIndex]

		// 2. Collect each definition's inline content, keyed by its id
		//    (e.g. "user-content-fn-1"), stripping the ↩ back-reference link.
		const defs = new Map<string, HastNode[]>()
		const ol = section.children?.find((n) => n.type === 'element' && n.tagName === 'ol')
		for (const li of ol?.children ?? []) {
			if (li.type !== 'element' || li.tagName !== 'li') continue
			const id = li.properties?.id as string | undefined
			if (!id) continue
			const content: HastNode[] = []
			for (const child of li.children ?? []) {
				if (child.type === 'element' && child.tagName === 'p') {
					for (const c of child.children ?? []) {
						if (c.type === 'element' && c.properties && 'dataFootnoteBackref' in c.properties)
							continue
						content.push(c)
					}
				} else if (child.type !== 'text' || child.value?.trim()) {
					content.push(child)
				}
			}
			// Drop a trailing whitespace-only text node left by the removed backref.
			while (content.length && content[content.length - 1].type === 'text' && !content[content.length - 1].value?.trim())
				content.pop()
			defs.set(id, content)
		}

		// 3. Walk the tree; for each footnote reference, inject the sidenote span
		//    and tag its nearest block ancestor as a positioning container.
		let counter = 0
		const walk = (node: HastNode, ancestors: HastNode[]) => {
			const kids = node.children
			if (!kids) return
			for (let i = 0; i < kids.length; i++) {
				const child = kids[i]
				const ref =
					child.type === 'element' && child.tagName === 'sup'
						? child.children?.find(
								(c) =>
									c.type === 'element' &&
									c.tagName === 'a' &&
									c.properties != null &&
									'dataFootnoteRef' in c.properties
							)
						: undefined
				if (child.type === 'element' && child.tagName === 'sup' && ref) {
					const href = (ref.properties?.href as string | undefined) ?? ''
					const refId = href.replace(/^#/, '')
					counter += 1
					const num = counter

					addClass(child, 'sidenote-marker')
					ref.properties = ref.properties ?? {}
					ref.properties.className = ['sidenote-ref']
					// The bottom footnotes section is removed below, so repoint the
					// link at the inline note and drop the now-dangling aria ref.
					ref.properties.href = `#sidenote-${num}`
					delete ref.properties['ariaDescribedBy']
					delete ref.properties['aria-describedby']

					const noteSpan: HastNode = {
						type: 'element',
						tagName: 'span',
						properties: { className: ['sidenote'], id: `sidenote-${num}` },
						children: [
							{
								type: 'element',
								tagName: 'span',
								properties: { className: ['sidenote-number'] },
								children: [{ type: 'text', value: String(num) }]
							},
							{ type: 'text', value: ' ' },
							...(defs.get(refId) ?? [])
						]
					}
					kids.splice(i + 1, 0, noteSpan)
					i += 1 // skip the span we just inserted

					// Mark the nearest block-level container as relative-positioned.
					const container =
						node.type === 'element' && BLOCK_TAGS.has(node.tagName ?? '')
							? node
							: [...ancestors].reverse().find((a) => a.type === 'element' && BLOCK_TAGS.has(a.tagName ?? ''))
					if (container) addClass(container, 'sidenote-container')
				} else {
					walk(child, [...ancestors, node])
				}
			}
		}
		walk(tree, [])

		// 4. Remove the now-empty footnotes section from the bottom of the page.
		children.splice(sectionIndex, 1)
	}
}
