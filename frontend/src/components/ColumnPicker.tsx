import { useState } from 'react'
import type { Analysis } from '../api/types'
import { COLUMN_ROLE } from '../lib/labels'

/**
 * The detector's guess about which questions to group people on is a starting point, not a
 * verdict. This is how someone overrules it — most often because a question they care about was
 * set aside as a background trait.
 */
export function ColumnPicker({
  result,
  busy,
  error,
  onRegroup,
}: {
  result: Analysis
  busy: boolean
  error: string | null
  onRegroup: (items: string[]) => void
}) {
  const columns = Object.keys(result.columns ?? {})
  const [picked, setPicked] = useState<Set<string>>(
    () => new Set(columns.filter((column) => result.columns[column] === 'used')),
  )
  const [tooFew, setTooFew] = useState(false)

  if (columns.length === 0) return null

  function toggle(column: string) {
    setPicked((current) => {
      const next = new Set(current)
      if (next.has(column)) next.delete(column)
      else next.add(column)
      return next
    })
    setTooFew(false)
  }

  return (
    <details className="card" style={{ marginTop: 12 }}>
      <summary>
        <span className="chev">▸</span>Group people on different questions
      </summary>
      <div className="rep">
        <p style={{ color: 'var(--muted)' }}>
          I chose the ticked questions. Tick or untick to group people on something else — useful
          if a question you care about was set aside as a background trait.
        </p>
        {columns.map((column) => (
          <label
            key={column}
            style={{ display: 'flex', gap: 8, alignItems: 'center', padding: '3px 0', fontSize: '.9rem' }}
          >
            <input type="checkbox" checked={picked.has(column)} onChange={() => toggle(column)} />
            <span>{column}</span>
            <span className="eyebrow" style={{ marginLeft: 'auto' }}>
              {COLUMN_ROLE[result.columns[column]] ?? result.columns[column]}
            </span>
          </label>
        ))}
        <div className="chips" style={{ marginTop: 10 }}>
          <button
            type="button"
            className="chip"
            disabled={busy}
            onClick={() => {
              if (picked.size < 2) {
                setTooFew(true)
                return
              }
              onRegroup([...picked])
            }}
          >
            Re-group with these questions
          </button>
        </div>
        <div style={{ marginTop: 8 }}>
          {tooFew && <span style={{ color: 'var(--accent)' }}>Pick at least two questions.</span>}
          {error && <span className="err-text">{error}</span>}
        </div>
      </div>
    </details>
  )
}
