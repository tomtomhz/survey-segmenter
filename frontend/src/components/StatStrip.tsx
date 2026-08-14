import { useEffect, useRef, useState } from 'react'
import type { Analysis } from '../api/types'
import { CONFIDENCE_ADVICE, CONFIDENCE_DETAIL, titleCase } from '../lib/labels'

/** The answer, legible at a glance: how many groups, how many people, how much to trust it. */
export function StatStrip({ result }: { result: Analysis }) {
  const confidence = result.confidence ?? 'unknown'
  const tiles = useRef<HTMLDivElement | null>(null)
  const [scrolledPast, setScrolledPast] = useState(false)

  // A full statistical report is long, and the three facts a reader keeps checking against —
  // how many groups, out of how many people, and whether to believe it — scrolled away at the
  // top with no way back except scrolling up. Watching the tiles rather than the scroll position
  // means this needs no measurement and cannot drift out of step with the layout.
  useEffect(() => {
    const node = tiles.current
    if (!node || typeof IntersectionObserver === 'undefined') return
    const observer = new IntersectionObserver(
      ([entry]) => setScrolledPast(!entry.isIntersecting && entry.boundingClientRect.top < 0),
      { threshold: 0 },
    )
    observer.observe(node)
    return () => observer.disconnect()
  }, [])

  return (
    <div>
      <div className="eyebrow">Result</div>
      <div className="stats" style={{ marginTop: 8 }} ref={tiles}>
        <div className="stat">
          <div className="eyebrow">Groups found</div>
          <div className="v">{result.k}</div>
        </div>
        <div className="stat">
          <div className="eyebrow">People</div>
          <div className="v">{result.n_people}</div>
        </div>
        <div className="stat">
          <div className="eyebrow">Confidence</div>
          <div style={{ marginTop: 7 }}>
            <span className={`pill ${confidence}`}>{titleCase(confidence)}</span>
          </div>
          <div style={{ fontSize: '.78rem', color: 'var(--muted)', marginTop: 6 }}>
            {CONFIDENCE_ADVICE[confidence]}
          </div>
        </div>
      </div>

      {/* The same three facts, one line, pinned once the tiles have gone by. `aria-hidden` because
          it is a restatement of what is already in the document — a screen reader announcing the
          result twice would be worse than not having it. */}
      {scrolledPast && (
        <div className="statbar" aria-hidden="true">
          <b>{result.k}</b> groups <span className="sep">·</span> <b>{result.n_people}</b> people{' '}
          <span className="sep">·</span> <span className={`pill ${confidence}`}>
            {titleCase(confidence)}
          </span>
        </div>
      )}

      {/* Below the row rather than inside the confidence tile. The tiles are equal-height flex
          children, so three lines of caveat in one of them stretched all three, left "Groups
          found" as a tall empty box, and squeezed the text into a third of the width.

          Still on the card and not in the report, because "Trust these groups" is the sentence
          people act on, and the evidence for it belongs within reach of that sentence. */}
      <details className="whatmeans">
        <summary>What does &ldquo;{titleCase(confidence)}&rdquo; actually mean?</summary>
        <p>{CONFIDENCE_DETAIL[confidence]}</p>
      </details>
    </div>
  )
}
