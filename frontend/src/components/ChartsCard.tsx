import { useId, useRef, useState } from 'react'
import type { Chart } from '../api/types'
import { CHART_TAB } from '../lib/labels'

/**
 * The point of showing the charts is that a reader can disagree with the write-up: a wrong
 * conclusion — mine, Claude's, or the statistics' — is visible in the segment map in a second.
 * Tabbed rather than stacked so the evidence is one glance, not two metres of scrolling.
 *
 * Panes stay mounted and are hidden with a class rather than unmounted, because the print
 * stylesheet un-hides all of them: a PDF someone circulates must carry every chart, not
 * whichever one happened to be in front.
 */
export function ChartsCard({ charts }: { charts: Chart[] }) {
  const [active, setActive] = useState(0)
  const tabs = useRef<HTMLDivElement>(null)
  const base = useId()
  if (charts.length === 0) return null

  /**
   * Arrow keys move between tabs, Home and End jump to the ends. This is the behaviour a tablist
   * promises the moment it claims `role="tablist"`, and without it the only way through six
   * charts by keyboard was Tab — which walks out of the group entirely rather than around it.
   */
  function onKeyDown(event: React.KeyboardEvent) {
    const last = charts.length - 1
    const moves: Record<string, number> = {
      ArrowRight: active === last ? 0 : active + 1,
      ArrowLeft: active === 0 ? last : active - 1,
      Home: 0,
      End: last,
    }
    const next = moves[event.key]
    if (next === undefined) return
    event.preventDefault()
    setActive(next)
    tabs.current?.querySelectorAll<HTMLButtonElement>('.ctab')[next]?.focus()
  }

  return (
    <details className="card charts" open>
      <summary>
        <span className="chev">▸</span>
        See the data yourself — check the groups with your own eyes
      </summary>
      <div className="cbody">
        <div className="ctabs" role="tablist" aria-label="Charts" ref={tabs} onKeyDown={onKeyDown}>
          {charts.map((chart, i) => (
            <button
              key={chart.id}
              type="button"
              role="tab"
              id={`${base}-tab-${chart.id}`}
              aria-controls={`${base}-panel-${chart.id}`}
              aria-selected={i === active}
              // Roving tabindex: the group is one stop, and arrows move within it.
              tabIndex={i === active ? 0 : -1}
              className={`ctab${i === active ? ' on' : ''}`}
              onClick={() => setActive(i)}
            >
              {CHART_TAB[chart.id] ?? `Chart ${i + 1}`}
            </button>
          ))}
        </div>
        {charts.map((chart, i) => (
          <div
            key={chart.id}
            id={`${base}-panel-${chart.id}`}
            role="tabpanel"
            aria-labelledby={`${base}-tab-${chart.id}`}
            className={`cpane${i === active ? '' : ' hide'}`}
          >
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
