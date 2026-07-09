import { defineConfig } from 'astro/config'
import mdx from '@astrojs/mdx'
import tailwind from '@astrojs/tailwind'
import sitemap from '@astrojs/sitemap'
import { remarkReadingTime } from './src/utils/remarkReadingTime.ts'
import remarkUnwrapImages from 'remark-unwrap-images'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeExternalLinks from 'rehype-external-links'
import { rehypeSidenotes } from './src/utils/rehypeSidenotes.ts'
import { rehypeCaptions } from './src/utils/rehypeCaptions.ts'
import expressiveCode from 'astro-expressive-code'
import { expressiveCodeOptions } from './src/site.config'
import icon from 'astro-icon'

import react from '@astrojs/react';

import { rehypeHeadingIds } from '@astrojs/markdown-remark';

// https://astro.build/config
export default defineConfig({
    site: 'https://taogenna.github.io',
    integrations: [
        // Must precede mdx so code blocks in .mdx are handled by Expressive Code.
        expressiveCode(expressiveCodeOptions),
        mdx({
            components: {
                a: './src/components/Link.astro'
            }
        }),
        tailwind({
            applyBaseStyles: false
        }),
        sitemap(),
        icon(),
        react()
    ],
    markdown: {
        remarkPlugins: [remarkUnwrapImages, remarkReadingTime, remarkMath],
        rehypePlugins: [
            rehypeHeadingIds,
            rehypeKatex,
            // Must run after rehypeKatex so math inside footnotes is rendered
            // before the note is relocated into the margin.
            rehypeSidenotes,
            rehypeCaptions,
        ],
        remarkRehype: {
            footnoteLabelProperties: {
                className: ['']
            }
        },
        components: {
            a: './src/components/Link.astro'
        }
    },
    prefetch: true,
    output: 'static'
})