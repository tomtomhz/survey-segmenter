import type { Analysis } from '../api/types'
import { CONFIDENCE_ADVICE, titleCase } from '../lib/labels'

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
    </div>
  )
}
