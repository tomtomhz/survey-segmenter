import { useState } from 'react'
import { api } from '../api/client'
import { isFailure, type Analysis, type ErrorKind, type NamesResult, type Result } from '../api/types'

/**
 * The automatic labels are built from question codes. This is where they become names a team
 * will actually recognise — and those names flow into the downloads, so they are the thing that
 * ends up in the CRM.
 */
export function NamePanel({
  result,
  busy,
  setBusy,
  onNeedsKey,
}: {
  result: Analysis
  busy: boolean
  setBusy: (busy: boolean) => void
  onNeedsKey: () => void
}) {
  const [names, setNames] = useState<string[]>(() =>
    Array.from({ length: result.k }, (_, i) => result.names?.[i] ?? ''),
  )
  const [state, setState] = useState<
    { kind: 'idle' } | { kind: 'working' } | { kind: 'saved' }
    | { kind: 'error'; message: string; errorKind?: ErrorKind }
  >({ kind: 'idle' })

  if (!result.k) return null

  async function submit(request: Promise<Result<NamesResult>>) {
    setBusy(true)
    setState({ kind: 'working' })
    const reply = await request
    setBusy(false)
    if (isFailure(reply)) {
      setState({ kind: 'error', message: reply.error, errorKind: reply.kind })
      return
    }
    setNames((current) => current.map((name, i) => reply.names[i] || name))
    setState({ kind: 'saved' })
  }

  function save() {
    const trimmed = names.map((name) => name.trim())
    if (trimmed.some((name) => !name)) {
      setState({ kind: 'error', message: 'Give every group a name, or use Suggest.' })
      return
    }
    void submit(api.saveNames(result.session_id, trimmed))
  }

  return (
    <details className="card" style={{ marginTop: 12 }}>
      <summary>
        <span className="chev">▸</span>Name the groups
      </summary>
      <div className="rep">
        <p style={{ color: 'var(--muted)' }}>
          The automatic labels are built from question codes. Give the groups names your team will
          recognise — they go into the downloads.
        </p>
        {names.map((name, i) => (
          <label
            key={i}
            style={{ display: 'flex', gap: 8, alignItems: 'center', margin: '5px 0', fontSize: '.9rem' }}
          >
            <span style={{ minWidth: 74, color: 'var(--muted)' }}>Group {i}</span>
            <input
              className="field"
              style={{ flex: 1 }}
              placeholder="e.g. Privacy-First Lurkers"
              value={name}
              onChange={(event) => {
                const next = event.target.value
                setNames((current) => current.map((old, j) => (j === i ? next : old)))
              }}
            />
          </label>
        ))}
        <div className="chips" style={{ marginTop: 10 }}>
          <button
            type="button"
            className="chip"
            disabled={busy}
            onClick={() => void submit(api.suggestNames(result.session_id))}
          >
            Suggest names with Claude
          </button>
          <button type="button" className="chip" disabled={busy} onClick={save}>
            Save these names
          </button>
        </div>
        <div style={{ marginTop: 8 }}>
          {state.kind === 'working' && <span className="think">Working…</span>}
          {state.kind === 'error' && (
            <span className="err-text">
              {state.message}{' '}
              {(state.errorKind === 'nokey' || state.errorKind === 'nosdk') && (
                <button type="button" className="link" onClick={onNeedsKey}>
                  Open Settings
                </button>
              )}
            </span>
          )}
          {state.kind === 'saved' && (
            <>
              <b>Saved.</b> The names are now in the downloads.
              <div className="chips" style={{ margin: '8px 0 0' }}>
                <a
                  className="chip"
                  href={api.downloadUrl(result.session_id, 'group_names.csv')}
                  download
                >
                  Group names (CSV)
                </a>
                <a
                  className="chip"
                  href={api.downloadUrl(result.session_id, 'segment_assignments.csv')}
                  download
                >
                  Who is in which group (CSV)
                </a>
              </div>
            </>
          )}
        </div>
      </div>
    </details>
  )
}
