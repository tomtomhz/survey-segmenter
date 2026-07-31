import { ChartsCard } from './ChartsCard'
import { ColumnPicker } from './ColumnPicker'
import { DownloadBar } from './DownloadBar'
import { NamePanel } from './NamePanel'
import { StatStrip } from './StatStrip'
import type { Analysis } from '../api/types'

/**
 * Everything a finished analysis shows, in one place.
 *
 * The three routes to a result — a fresh run, a re-group, a reopened project — used to paint it
 * from three call sites and drifted apart once already. There is now one component and no way
 * for them to differ.
 */
export function ResultCard({
  result,
  busy,
  setBusy,
  regroupError,
  onRegroup,
  onNeedsKey,
}: {
  result: Analysis
  busy: boolean
  setBusy: (busy: boolean) => void
  regroupError: string | null
  onRegroup: (items: string[]) => void
  onNeedsKey: () => void
}) {
  return (
    <>
      <StatStrip result={result} />
      <ChartsCard charts={result.charts ?? []} />
      <details className="card">
        <summary>
          <span className="chev">▸</span>
          {result.title} — full statistical report
        </summary>
        {/* Server-rendered and server-escaped; group names the user typed are escaped there. */}
        <div className="rep" dangerouslySetInnerHTML={{ __html: result.report_html }} />
      </details>
      <DownloadBar result={result} busy={busy} setBusy={setBusy} />
      <NamePanel result={result} busy={busy} setBusy={setBusy} onNeedsKey={onNeedsKey} />
      <ColumnPicker result={result} busy={busy} error={regroupError} onRegroup={onRegroup} />
    </>
  )
}
