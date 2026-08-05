import { useId, useMemo, useState } from 'react'
import { scaleLinear } from 'd3-scale'

import type { FitSpec } from './spec'
import { segColour, useDarkMode } from './theme'

/**
 * How well each group's people fit it, one distribution per segment.
 *
 * The static version shows the shapes; what it cannot tell you is a number you can act on. Pointing
 * at a band reports *how many people in this group scored between here and here*, which is exactly
 * what somebody filtering a list before spending money on it needs — the low bands are the people
 * who sit between two groups and could belong to either.
 *
 * Bands are per-segment bars rather than a single hit area per row, because the question is about a
 * range of fit, not about the group as a whole. Each is focusable, so the keyboard reaches the same
 * numbers as the pointer.
 */
export function FitRidges({ spec, title }: { spec: FitSpec; title: string }) {
  const [active, setActive] = useState<{ row: number; bin: number } | null>(null)
  const titleId = useId()
  const dark = useDarkMode()

  const width = 720
  const rowHeight = 74
  const pad = { top: 14, right: 18, bottom: 44, left: 108 }
  const height = pad.top + pad.bottom + spec.rows.length * rowHeight

  const { x, byIndex, tallest } = useMemo(() => {
    const map: Record<number, FitSpec['segments'][number]> = {}
    for (const key of spec.segments) map[key.index] = key
    return {
      x: scaleLinear()
        .domain([spec.edges[0], spec.edges[spec.edges.length - 1]])
        .range([pad.left, width - pad.right]),
      byIndex: map,
      // Each row is scaled to its own tallest band, so a 40-person group is as readable as a
      // 900-person one. The counts are reported in words, which is where absolute size belongs.
      tallest: spec.rows.map((r) => Math.max(...r.counts, 1)),
    }
  }, [spec, pad.left, pad.right])

  const reading = (() => {
    if (!active) {
      return `${spec.rows.length} groups. Point at a band, or tab into one, to read how many `
        + 'people it holds.'
    }
    const row = spec.rows[active.row]
    const key = byIndex[row.segment]
    const people = row.counts[active.bin]
    const from = spec.edges[active.bin]
    const to = spec.edges[active.bin + 1]
    return `${key?.label ?? `Group ${row.segment}`}: ${people.toLocaleString()} `
      + `${people === 1 ? 'person fits' : 'people fit'} between ${from.toFixed(2)} and `
      + `${to.toFixed(2)}${to < spec.overall_median ? ' — below the typical respondent' : ''}`
  })()

  return (
    <div className="iridge">
      <svg className="chart" viewBox={`0 0 ${width} ${height}`}
           preserveAspectRatio="xMidYMid meet" role="img" aria-labelledby={titleId}>
        <title id={titleId}>
          {`${title}. One distribution per group, showing how well its people fit it.`}
        </title>

        {spec.rows.map((row, r) => {
          const key = byIndex[row.segment]
          const base = pad.top + (r + 1) * rowHeight - 16
          return (
            <g key={row.segment}>
              <text x={pad.left - 12} y={base - 18} textAnchor="end" fontSize={12}
                    fill="currentColor">
                {key?.label ?? `Group ${row.segment}`}
              </text>
              <line x1={pad.left} x2={width - pad.right} y1={base} y2={base}
                    stroke="currentColor" strokeWidth={0.8} opacity={0.28} />
              {row.counts.map((count, b) => {
                const left = x(spec.edges[b])
                const right = x(spec.edges[b + 1])
                const tall = (count / tallest[r]) * (rowHeight - 30)
                const on = active?.row === r && active?.bin === b
                return (
                  <rect
                    key={b}
                    x={left}
                    y={base - tall}
                    width={Math.max(right - left - 1, 1)}
                    height={Math.max(tall, count > 0 ? 1 : 0)}
                    fill={segColour(key, dark)}
                    opacity={on ? 1 : 0.82}
                    tabIndex={0}
                    role="button"
                    aria-label={`${key?.label ?? ''}, ${count} people between `
                      + `${spec.edges[b].toFixed(2)} and ${spec.edges[b + 1].toFixed(2)}`}
                    onPointerOver={() => setActive({ row: r, bin: b })}
                    onFocus={() => setActive({ row: r, bin: b })}
                    onPointerLeave={() => setActive(null)}
                    onBlur={() => setActive(null)}
                  />
                )
              })}
              {/* The group's own median, which is the single number that ranks the rows. */}
              <line x1={x(row.median)} x2={x(row.median)} y1={base - (rowHeight - 30)} y2={base}
                    stroke="currentColor" strokeWidth={1.1} />
            </g>
          )
        })}

        {/* The whole sample, so each row can be read against the study rather than in isolation. */}
        <line x1={x(spec.overall_median)} x2={x(spec.overall_median)} y1={pad.top}
              y2={height - pad.bottom} stroke="currentColor" strokeWidth={1} opacity={0.35} />
        <text x={x(spec.overall_median)} y={pad.top - 2} textAnchor="middle" fontSize={11}
              fill="currentColor" opacity={0.75}>
          {`whole sample ${spec.overall_median.toFixed(2)}`}
        </text>
        <text x={(pad.left + width - pad.right) / 2} y={height - 12} textAnchor="middle"
              fontSize={12} fill="currentColor">
          how well each person fits their group →
        </text>
      </svg>
      <div className="imap-read" role="status" aria-live="polite">{reading}</div>
    </div>
  )
}
