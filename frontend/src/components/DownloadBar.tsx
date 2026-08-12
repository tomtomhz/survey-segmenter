import { useRef, useState } from 'react'
import { api } from '../api/client'
import { isFailure, type Analysis, type ScoreResult } from '../api/types'
import { DOWNLOAD_LABEL } from '../lib/labels'

/**
 * The files that let someone act on the groups, plus the typing tool: upload people who were
 * not in the study and each one is placed into one of the segments found.
 */
export function DownloadBar({
  result,
  downloads,
  busy,
  setBusy,
}: {
  result: Analysis
  /** Passed in rather than read off `result`: naming the groups adds a file to it. */
  downloads: string[]
  busy: boolean
  setBusy: (busy: boolean) => void
}) {
  const [scored, setScored] = useState<ScoreResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [scoring, setScoring] = useState(false)
  const picker = useRef<HTMLInputElement>(null)

  if (downloads.length === 0) return null

  async function score(file: File) {
    setScoring(true)
    setError(null)
    setScored(null)
    setBusy(true)
    const reply = await api.score(result.session_id, file)
    setBusy(false)
    setScoring(false)
    if (isFailure(reply)) setError(reply.error)
    else setScored(reply)
  }

  return (
    <div className="note" style={{ marginTop: 12 }}>
      <b>Take it away.</b> Use these to act on the groups — load them into your CRM, an ad
      audience, or a mail tool.
      <div className="chips" style={{ margin: '10px 0 0' }}>
        {downloads.map((file) => (
          <a
            key={file}
            className="chip"
            href={api.downloadUrl(result.session_id, file)}
            download
          >
            {DOWNLOAD_LABEL[file] ?? file}
          </a>
        ))}
      </div>
      <div className="sep">
        <b>Score new people.</b> Upload a file of people who were not in this study and I will put
        each of them into one of these groups.
        <div className="chips" style={{ margin: '10px 0 0' }}>
          <button
            type="button"
            className="chip"
            disabled={busy}
            onClick={() => picker.current?.click()}
          >
            Choose a file of new people…
          </button>
          <input
            ref={picker}
            type="file"
            accept=".csv,.xlsx,.xls"
            hidden
            onChange={(event) => {
              const file = event.target.files?.[0]
              event.target.value = ''
              if (file) void score(file)
            }}
          />
        </div>
        <div style={{ marginTop: 8 }}>
          {scoring && <span className="think">Scoring…</span>}
          {error && <span className="err-text">{error}</span>}
          {scored && <ScoreSummary result={result} scored={scored} />}
        </div>
      </div>
    </div>
  )
}

export function ScoreSummary({ result, scored }: { result: Analysis; scored: ScoreResult }) {
  const breakdown = Object.entries(scored.breakdown)
    .map(([group, count]) => `Group ${group}: ${count}`)
    .join(' · ')
  return (
    <>
      <b>{scored.n} people scored.</b> {breakdown}
      <br />
      {/* The floor is shown beside the figure because the figure means nothing without it: this
          scale runs from 1/k, not from 0. Measured over thirty-six holdout studies, two-group
          runs averaged 0.59 and three-group runs 0.44 — which reads as a large difference and is
          not one, since both sit about 17% of the way up their own range. */}
      <span style={{ color: 'var(--muted)' }}>
        Average confidence {scored.mean_confidence}
        {scored.confidence_floor != null && (
          <>
            {' '}— on this study&rsquo;s scale, {scored.confidence_floor} means no better than
            guessing and 1.0 means sitting exactly on a group&rsquo;s centre
          </>
        )}
        .
      </span>
      {!!scored.off_scale && (
        <div className="note warn" style={{ margin: '8px 0 0' }}>
          <b>{scored.off_scale} of {scored.n} answered something outside your original scale</b> —
          usually a &ldquo;no answer&rdquo; code such as 99. They have still been scored, but a value
          the study never saw pulls someone towards whichever group is extreme on that question, so
          their group is unreliable. The column{' '}
          <code>answers_off_the_original_scale</code> in the download says who they are.
        </div>
      )}
      <div className="chips" style={{ margin: '8px 0 0' }}>
        <a className="chip" href={api.downloadUrl(result.session_id, scored.file)} download>
          Download the scored list (CSV)
        </a>
      </div>
    </>
  )
}
