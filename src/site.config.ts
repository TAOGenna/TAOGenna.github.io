import type { SiteConfig } from '@/types'
import type { AstroExpressiveCodeOptions } from 'astro-expressive-code'

export const siteConfig: SiteConfig = {
	// Used as both a meta property (src/components/BaseHead.astro L:31 + L:49) & the generated satori png (src/pages/og-image/[slug].png.ts)
	author: 'Kenyi Takagui-Perez',
	// Meta property used to construct the meta title property, found in src/components/BaseHead.astro L:11
	title: 'Kenyi Takagui-Perez',
	// Meta property used as the default description meta property
	description: 'Applied AI engineer & physicist — agents, ML, and quantum matter',
	// HTML lang property, found in src/layouts/Base.astro L:18
	lang: 'en-GB',
	// Meta property, found in src/components/BaseHead.astro L:42
	ogLocale: 'en_GB',
	// Date.prototype.toLocaleDateString() parameters, found in src/utils/date.ts.
	date: {
		locale: 'en-GB',
		options: {
			day: 'numeric',
			month: 'short',
			year: 'numeric'
		}
	}
}

export const menuLinks: Array<{ title: string; path: string }> = [
	{
		title: 'Home',
		path: '/'
	},
	{
		title: 'Blog',
		path: '/blog/'
	}
]

// https://expressive-code.com/reference/configuration/
export const expressiveCodeOptions: AstroExpressiveCodeOptions = {
	// Light theme is the default (clean VS Code "Light+" look — blue keywords,
	// teal numerics, dark-red strings on white); the dark theme is gated on the
	// `.dark` class that ThemeProvider toggles on <html> for professional mode.
	themes: ['light-plus', 'dark-plus'],
	themeCssSelector(theme, { styleVariants }) {
		// Base (first) theme is the default — no selector needed.
		if (theme === styleVariants[0]?.theme) return false
		// All other themes apply only under the `.dark` class.
		return '.dark'
	},
	useThemedScrollbars: false,
	// Don't auto-switch to the dark theme based on the OS color scheme — code
	// theme must follow the site's own light/dark (.dark class) toggle only.
	useDarkModeMediaQuery: false,
	styleOverrides: {
		frames: {
			frameBoxShadowCssValue: 'none'
		},
		// Thin neutral hairline like the source article, not a warm/heavy frame.
		borderColor: 'hsl(0 0% 88%)',
		borderWidth: '1px',
		borderRadius: '0.5rem',
		uiLineHeight: 'inherit',
		codeFontSize: '0.9rem',
		codeLineHeight: '1.65',
		codePaddingInline: '1.25rem',
		codePaddingBlock: '1rem',
		codeFontFamily:
			'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;'
	}
}
