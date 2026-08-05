import { useEffect, useState } from 'react'

import type { SegmentKey } from './spec'

/**
 * Which theme the page is showing, as a boolean the chart components can act on.
 *
 * The interactive charts used to read their colours from `var(--seg-N)`, which happens to work:
 * the theme block Python injects into each static SVG is scoped to `svg.chart`, and the
 * interactive SVGs carry that class too, so they match the same selector. Verified in the browser
 * — light resolved `#2a78d6`, dark `#3987e5`, and the marks changed with them.
 *
 * "Happens to work" is the problem. The interactive chart's colours depended on a stylesheet
 * injected into a *different* element by the other renderer, so removing the static drawing for
 * charts that have a spec — a reasonable future change — would silently drop every interactive
 * chart back to light-mode hexes on a dark page, with nothing failing.
 *
 * The palette still has exactly one home: `charts.py` measures it, and both steps travel in the
 * spec. This only decides which of the two the component asks for.
 */
export function useDarkMode(): boolean {
  const [dark, setDark] = useState(() => prefersDark())

  useEffect(() => {
    const media = window.matchMedia?.('(prefers-color-scheme: dark)')
    const update = () => setDark(prefersDark())
    media?.addEventListener?.('change', update)
    // The reader's own toggle stamps data-theme on the root element, which no media query sees.
    const watcher = new MutationObserver(update)
    watcher.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    update()
    return () => {
      media?.removeEventListener?.('change', update)
      watcher.disconnect()
    }
  }, [])

  return dark
}

function prefersDark(): boolean {
  const stamped = document.documentElement.getAttribute('data-theme')
  // An explicit choice wins over the operating system, in both directions.
  if (stamped === 'dark') return true
  if (stamped === 'light') return false
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false
}

/** A segment's colour for the theme now on screen. Both steps come from the spec. */
export function segColour(key: SegmentKey | undefined, dark: boolean): string {
  if (!key) return 'currentColor'
  return dark ? key.colour_dark : key.colour
}
