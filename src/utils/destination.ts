// Where a project (or any link on a project page) goes. The one place that decides it, so tiles,
// project pages and "More projects" agree, and a link to a page that does not exist fails the build
// instead of 404ing for a visitor.
import fs from 'node:fs'
import path from 'node:path'
import { getCollection } from 'astro:content'
import type { Project } from '@/data/projects'

export type Destination =
	| { kind: 'page'; href: string } // a write-up at /projects/<slug>/
	| { kind: 'site'; href: string } // a page or file on this site (blog post, /courses/, a PDF)
	| { kind: 'external'; href: string } // another site: open in a new tab, mark with ↗
	| { kind: 'none' }

export async function destinationOf(p: Pick<Project, 'heading' | 'slug' | 'href'>): Promise<Destination> {
	if (p.slug && (await projectSlugs()).has(p.slug)) return { kind: 'page', href: `/projects/${p.slug}/` }
	if (p.href) return resolveHref(p.href, p.heading)
	return { kind: 'none' }
}

// Classify a raw href. `from` names the caller in the build error.
export async function resolveHref(href: string, from: string): Promise<Destination> {
	if (/^[a-z]+:/i.test(href)) {
		// Our own domain written in full is a page from another repo (e.g. /projections/): same tab, can't be checked.
		const own = new URL(href).host === new URL(import.meta.env.SITE).host
		return { kind: own ? 'site' : 'external', href }
	}
	if (!href.startsWith('/')) throw new Error(`${from}: href "${href}" must start with "/" or be a full URL`)
	if (!(await existsOnSite(href))) throw new Error(`${from}: links to ${href}, which is not a page or file on this site`)
	return { kind: href.startsWith('/projects/') ? 'page' : 'site', href }
}

const ROOT = process.cwd()
let slugs: Promise<Set<string>> | undefined
const projectSlugs = () => (slugs ??= getCollection('project').then((es) => new Set(es.map((e) => e.slug))))

async function existsOnSite(href: string) {
	const p = decodeURI(href.split(/[?#]/)[0])
	const blog = p.match(/^\/blog\/([^/]+)\/?$/)
	if (blog) return (await getCollection('post')).some((e) => e.slug === blog[1])
	const proj = p.match(/^\/projects\/([^/]+)\/?$/)
	if (proj) return (await projectSlugs()).has(proj[1])
	const bare = p.replace(/\/$/, '')
	return [
		path.join(ROOT, 'public', p),
		path.join(ROOT, 'public', p, 'index.html'),
		path.join(ROOT, 'src/pages', `${bare || '/index'}.astro`),
		path.join(ROOT, 'src/pages', bare, 'index.astro')
	].some((f) => fs.existsSync(f) && fs.statSync(f).isFile())
}
