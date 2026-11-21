import { defineCollection, z } from 'astro:content'

function removeDupsAndLowerCase(array: string[]) {
	if (!array.length) return array
	const lowercaseItems = array.map((str) => str.toLowerCase())
	const distinctItems = new Set(lowercaseItems)
	return Array.from(distinctItems)
}

const post = defineCollection({
	type: 'content',
	schema: ({ image }) =>
		z.object({
			title: z.string().max(60),
			description: z.string().min(50).max(160),
			publishDate: z
				.string()
				.or(z.date())
				.transform((val) => new Date(val)),
			updatedDate: z
				.string()
				.optional()
				.transform((str) => (str ? new Date(str) : undefined)),
			coverImage: z
				.object({
					src: image(),
					alt: z.string()
				})
				.optional(),
			draft: z.boolean().default(false),
			pinned: z.boolean().default(false),
			tags: z.array(z.string()).default([]).transform(removeDupsAndLowerCase),
			ogImage: z.string().optional(),
			hackerNewsId: z.string().optional(),
			// Show an academic-style citation block (plaintext + BibTeX) at the
			// end of the post. `citationAuthors` overrides the default author name.
			citation: z.boolean().default(false),
			citationAuthors: z.string().optional()
		})
})

// Project write-ups, served at /projects/<slug>/. The file name must match a `slug` in
// data/projects.ts: title, subtitle and tags come from there, so they are written once.
const project = defineCollection({
	type: 'content',
	schema: z.object({
		year: z.number().int(),
		hero: z
			.object({
				video: z.string().optional(),
				image: z.string().optional(),
				poster: z.string().optional(),
				caption: z.string().optional()
			})
			.optional(),
		// The link row under the hero. Internal paths are checked at build (utils/destination).
		links: z.array(z.object({ label: z.string(), href: z.string() })).default([])
	})
})

export const collections = { post, project }
