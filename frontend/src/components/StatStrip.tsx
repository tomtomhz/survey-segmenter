import type { Analysis } from '../api/types'
import { CONFIDENCE_ADVICE, CONFIDENCE_DETAIL, titleCase } from '../lib/labels'

/** The answer, legible at a glance: how many groups, how many people, how much to trust it. */
export function StatStrip({ result }: { result: Analysis }) {
  const confidence = result.confidence ?? 'unknown'
  return (
    <div>
      <div className="eyebrow">Result</div>
      <div className="stats" style={{ marginTop: 8 }}>
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

      {/* Below the row rather than inside the confidence tile. The tiles are equal-height flex
          children, so three lines of caveat in one of them stretched all three and left "Groups
          found" as a tall empty box — and squeezed the text into a third of the width, where it
          wrapped every four words. Out here it gets a readable measure and the tiles stay the
          compact scannable row they are meant to be.

          Still on the card and not in the report, because "Trust these groups" is the sentence
          people act on, and the evidence for it belongs within reach of that sentence. */}
      <details className="whatmeans">
        <summary>What does &ldquo;{titleCase(confidence)}&rdquo; actually mean?</summary>
        <p>{CONFIDENCE_DETAIL[confidence]}</p>
      </details>
    </div>
  )
}
