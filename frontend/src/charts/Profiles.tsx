import { useId, useMemo, useState } from 'react'
import { scaleLinear } from 'd3-scale'

import type { ProfilesSpec } from './spec'
import { markerPath } from './marks'
import { segColour, useDarkMode } from './theme'

/**
 * What separates the groups, one row per question.
 *
 * A Cleveland dot plot: each row is a question, each mark a group, joined by a rule showing how far
 * apart they are. The reading it supports — how far apart are these groups on this question — is a
 * distance along a shared axis, which people judge accurately, unlike the area of a radar polygon.
 *
 * Interactive, it answers the question the static chart raises and cannot settle: two dots that
 * nearly touch, is that a real difference or not? Pointing at one reports the group, its value, and
 * the spread across the whole row, so "these two are within 0.1 of each other on a five-point
 * scale" becomes something you can read rather than estimate.
 */
export function Profiles({ spec, title }: { spec: ProfilesSpec; title: string }) {
  const [active, setActive] = useState<{ row: number; item: number } | null>(null)
  const titleId = useId()
  const dark = useDarkMode()

  const width = 720
  const rowHeight = 42
  const pad = { top: 16, right: 26, bottom: 46, left: 210 }
  const height = pad.top + pad.bottom + spec.items.length * rowHeight

  const { x, spreads } = useMemo(() => {
    const flat = spec.values.flat().filter((v) => Number.isFinite(v))
    const lo = Math.min(...flat)
    const hi = Math.max(...flat)
    const gap = Math.max(0.25, (hi - lo) * 0.12)
    return {
      // Never anchored at zero. On a 1-5 scale a zero baseline spends a fifth of the axis on a
      // region no respondent can occupy, which is why this is a dot plot and not bars.
      x: scaleLinear().domain([lo - gap, hi + gap]).range([pad.left, width - pad.right]),
      spreads: spec.items.map((_, i) => {
        const column = spec.values.map((row) => row[i]).filter((v) => Number.isFinite(v))
        return column.length ? Math.max(...column) - Math.min(...column) : 0
      }),
    }
  }, [spec, pad.left, pad.right])

  const reading = (() => {
    if (!active) {
      return `${spec.segments.length} groups across ${spec.items.length} questions. `
        + 'Point at a mark, or tab to one, to read it.'
    }
    const key = spec.segments[active.row]
    const value = spec.values[active.row]?.[active.item]
    const spread = spreads[active.item]
    return `${key?.label ?? ''} — ${spec.items[active.item]}: ${spec.measure} `
      + `${value?.toFixed(2) ?? '—'}. The groups span ${spread.toFixed(2)} on this question`
      + `${spread < 0.25 ? ', which is barely any difference at all' : ''}`
  })()

  return (
    <div className="iprofiles">
      <svg className="chart" viewBox={`0 0 ${width} ${height}`}
           preserveAspectRatio="xMidYMid meet" role="img" aria-labelledby={titleId}>
        <title id={titleId}>
          {`${title}. Each group's ${spec.measure} on ${spec.items.length} questions.`}
        </title>

        {spec.items.map((item, i) => {
          const y = pad.top + i * rowHeight + rowHeight / 2
          const column = spec.values.map((row) => row[i]).filter((v) => Number.isFinite(v))
          if (!column.length) return null
          return (
            <g key={item}>
              <text x={pad.left - 12} y={y} textAnchor="end" dominantBaseline="central"
                    fontSize={12} fill="currentColor">
                {item.length > 30 ? `${item.slice(0, 29)}…` : item}
              </text>
              {/* The rule IS the answer: its length is how far apart the groups are here. */}
              <line x1={x(Math.min(...column))} x2={x(Math.max(...column))} y1={y} y2={y}
                    stroke="currentColor" strokeWidth={2.5} opacity={0.22} strokeLinecap="round" />
              {spec.values.map((row, r) => {
                const value = row[i]
                if (!Number.isFinite(value)) return null
                const key = spec.segments[r]
                const on = active?.row === r && active?.item === i
                return (
                  <path
                    key={r}
                    className="iprofiles-mark"
                    d={markerPath(key?.marker ?? 'o', on ? 190 : 130)}
                    transform={`translate(${x(value)} ${y})`}
                    fill={segColour(key, dark)}
                    stroke="var(--card)"
                    strokeWidth={1.4}
                    tabIndex={0}
                    role="button"
                    aria-label={`${key?.label ?? ''}, ${item}, ${value.toFixed(2)}`}
                    onPointerOver={() => setActive({ row: r, item: i })}
                    onFocus={() => setActive({ row: r, item: i })}
                    onPointerLeave={() => setActive(null)}
                    onBlur={() => setActive(null)}
                  />
                )
              })}
            </g>
          )
        })}

        <text x={(pad.left + width - pad.right) / 2} y={height - 14} textAnchor="middle"
              fontSize={12} fill="currentColor">
          {spec.measure === 'share' ? 'share of the group' : 'average answer'}
        </text>
      </svg>

      <ul className="imap-key">
        {spec.segments.map((key) => (
          <li key={key.index}>
            <svg viewBox="-9 -9 18 18" aria-hidden="true" focusable="false">
              <path d={markerPath(key.marker, 90)} fill={segColour(key, dark)} />
            </svg>
            {key.label}
          </li>
        ))}
      </ul>
      <div className="imap-read" role="status" aria-live="polite">{reading}</div>
    </div>
  )
}
