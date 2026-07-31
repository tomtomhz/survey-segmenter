import { useState } from 'react'
import type { Chart } from '../api/types'
import { CHART_TAB } from '../lib/labels'

/**
 * The point of showing the charts is that a reader can disagree with the write-up: a wrong
 * conclusion — mine, Claude's, or the statistics' — is visible in the segment map in a second.
 * Tabbed rather than stacked so the evidence is one glance, not two metres of scrolling.
 *
 * The old implementation toggled `.hide` classes across the DOM by hand and had to re-attach a
 * delegated listener every time a card was re-rendered. Here the selected tab is just state.
 *
 * Panes stay mounted and are hidden with a class rather than unmounted, because the print
 * stylesheet un-hides all of them: a PDF someone circulates must carry every chart, not
 * whichever one happened to be in front.
 */
export function ChartsCard({ charts }: { charts: Chart[] }) {
  const [active, setActive] = useState(0)
  if (charts.length === 0) return null

  return (
    <details className="card charts" open>
      <summary>
        <span className="chev">▸</span>
        See the data yourself — check the groups with your own eyes
      </summary>
      <div className="cbody">
        <div className="ctabs" role="tablist">
          {charts.map((chart, i) => (
            <button
              key={chart.id}
              type="button"
              role="tab"
              aria-selected={i === active}
              className={`ctab${i === active ? ' on' : ''}`}
              onClick={() => setActive(i)}
            >
              {CHART_TAB[chart.id] ?? `Chart ${i + 1}`}
            </button>
          ))}
        </div>
        {charts.map((chart, i) => (
          <div key={chart.id} className={`cpane${i === active ? '' : ' hide'}`} role="tabpanel">
            <div className="ctitle">{chart.title}</div>
            {/* Both are built by the Python chart engine from aggregate numbers and never
                contain anything the user typed. */}
            <div className="cwrap" dangerouslySetInnerHTML={{ __html: chart.svg }} />
            <p className="ccap" dangerouslySetInnerHTML={{ __html: chart.caption }} />
          </div>
        ))}
      </div>
    </details>
  )
}
