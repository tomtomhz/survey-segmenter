import { useEffect } from 'react'

/**
 * Open every collapsed panel before printing, and put them back afterwards.
 *
 * "Save PDF" is how a result leaves this app and reaches people who will never open it — and it
 * was silently dropping the full statistical report, the group names and the column picker,
 * because all three are collapsed `<details>` and a browser does not paint the contents of one.
 * The print stylesheet cannot fix that: `.card[open] .rep { display: block }` only applies to
 * panels that were already open, so the rule was quietly doing nothing for the ones that matter.
 *
 * Opening them in `beforeprint` is the only approach that does not depend on how a particular
 * engine collapses `<details>` — the panels are genuinely open when the snapshot is taken.
 */
export function usePrintExpansion() {
  useEffect(() => {
    // Only panels this opened get closed again, so anything the reader had expanded stays that way.
    let opened: HTMLDetailsElement[] = []

    const expand = () => {
      opened = [...document.querySelectorAll<HTMLDetailsElement>('details:not([open])')]
      for (const panel of opened) panel.open = true
    }
    const restore = () => {
      for (const panel of opened) panel.open = false
      opened = []
    }

    window.addEventListener('beforeprint', expand)
    window.addEventListener('afterprint', restore)
    // Safari fires neither; it drives printing through the print media query instead.
    const printing = window.matchMedia?.('print')
    const onMedia = (event: MediaQueryListEvent) => (event.matches ? expand() : restore())
    printing?.addEventListener?.('change', onMedia)

    return () => {
      window.removeEventListener('beforeprint', expand)
      window.removeEventListener('afterprint', restore)
      printing?.removeEventListener?.('change', onMedia)
    }
  }, [])
}
