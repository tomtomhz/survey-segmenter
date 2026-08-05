import { useId, useMemo, useState } from 'react'
import { scaleLinear, scalePoint } from 'd3-scale'

import type { KChoiceSpec } from './spec'

/**
 * Every quality measure at every number of groups tried.
 *
 * The static chart asks the reader to trace three lines back to an axis to learn what a given
 * number of groups actually scored. Here, pointing at a number of groups reports every measure at
 * once — which is the comparison the chart exists for: not "what is prediction strength", but
 * "what does the evidence look like at four groups rather than three".
 *
 * The hit areas are full-height columns, one per candidate, rather than the line points. A reader
 * asking about four groups is asking about the column, and a column is a target you can hit.
 */
export function KChoice({ spec, title }: { spec: KChoiceSpec; title: string }) {
  const [active, setActive] = useState<number | null>(null)
  const titleId = useId()

  const width = 720
  const height = 360
  const pad = { top: 20, right: 20, bottom: 66, left: 44 }

  const { x, y } = useMemo(() => ({
    x: scalePoint<number>().domain(spec.ks).range([pad.left, width - pad.right]).padding(0.5),
    y: scaleLinear().domain([0, 1]).range([height - pad.bottom, pad.top]),
  }), [spec.ks, pad.left, pad.right, pad.top, pad.bottom])

  const reading = active === null
    ? `Tried ${spec.ks.length} different numbers of groups. Point at one to see how it scored.`
    : [`${spec.ks[active]} groups`
        + (spec.ks[active] === spec.chosen ? ' (chosen)' : ''),
      ...spec.series.map((s) => {
        const v = s.values[active]
        return `${s.label} ${v === null ? '—' : v.toFixed(2)}`
      })].join(' · ')

  return (
    <div className="ikchoice">
      <svg className="chart" viewBox={`0 0 ${width} ${height}`}
           preserveAspectRatio="xMidYMid meet" role="img" aria-labelledby={titleId}>
        <title id={titleId}>
          {`${title}. ${spec.series.length} quality measures at each of `}
          {`${spec.ks.length} candidate numbers of groups.`}
        </title>

        {/* The reproduces line, drawn rather than left for the reader to hold in their head. */}
        <line x1={pad.left} x2={width - pad.right} y1={y(spec.cutoff)} y2={y(spec.cutoff)}
              stroke="currentColor" strokeWidth={1} opacity={0.45} strokeDasharray="3 3" />
        <text x={width - pad.right} y={y(spec.cutoff) - 5} textAnchor="end" fontSize={11}
              fill="currentColor" opacity={0.75}>
          {`${spec.cutoff.toFixed(2)} — reproduces`}
        </text>

        {spec.series.map((s) => {
          const points = s.values
            .map((v, i) => (v === null ? null : `${x(spec.ks[i])},${y(v)}`))
            .filter(Boolean)
            .join(' ')
          return (
            <g key={s.key}>
              <polyline
                points={points}
                fill="none"
                stroke={s.lead ? 'var(--chart-lead, #46785C)' : 'currentColor'}
                strokeWidth={s.lead ? 2.4 : 1.6}
                opacity={s.lead ? 1 : 0.45}
              />
              {s.values.map((v, i) => (v === null ? null : (
                <circle key={i} cx={x(spec.ks[i])} cy={y(v)} r={s.lead ? 3.4 : 2.6}
                        fill={s.lead ? 'var(--chart-lead, #46785C)' : 'currentColor'}
                        opacity={s.lead ? 1 : 0.45} />
              )))}
            </g>
          )
        })}

        {/* One column per candidate: the reader is asking about a number of groups, not about a
            point on one line, and a column is a target that can actually be hit. */}
        {spec.ks.map((k, i) => {
          const step = (width - pad.left - pad.right) / Math.max(spec.ks.length, 1)
          return (
            <rect
              key={k}
              x={(x(k) ?? 0) - step / 2}
              y={pad.top}
              width={step}
              height={height - pad.top - pad.bottom}
              fill={active === i ? 'currentColor' : 'transparent'}
              opacity={active === i ? 0.06 : 0}
              tabIndex={0}
              role="button"
              aria-label={`${k} groups`}
              onPointerOver={() => setActive(i)}
              onFocus={() => setActive(i)}
              onPointerLeave={() => setActive(null)}
              onBlur={() => setActive(null)}
            />
          )
        })}

        {spec.ks.map((k) => (
          <text key={k} x={x(k)} y={height - pad.bottom + 18} textAnchor="middle" fontSize={12}
                fill="currentColor" fontWeight={k === spec.chosen ? 700 : 400}>
            {k}
          </text>
        ))}
        <text x={(pad.left + width - pad.right) / 2} y={height - pad.bottom + 40}
              textAnchor="middle" fontSize={12} fill="currentColor">
          number of groups
        </text>
        <text x={x(spec.chosen)} y={height - pad.bottom + 58} textAnchor="middle" fontSize={11}
              fontWeight={700} fill="currentColor">
          chosen
        </text>
      </svg>

      <ul className="imap-key">
        {spec.series.map((s) => (
          <li key={s.key}>
            <svg viewBox="-9 -9 18 18" aria-hidden="true" focusable="false">
              <circle r={5} fill={s.lead ? 'var(--chart-lead, #46785C)' : 'currentColor'}
                      opacity={s.lead ? 1 : 0.45} />
            </svg>
            {s.label}{s.lead ? ' — decides' : ''}
          </li>
        ))}
      </ul>
      <div className="imap-read" role="status" aria-live="polite">{reading}</div>
    </div>
  )
}
