import { useId, useMemo, useState } from 'react'
import { scaleLinear } from 'd3-scale'

import type { GorgeSpec } from './spec'

/**
 * The whole sample's fit, in one distribution.
 *
 * Leisch's gorge plot, and the tool's central question answered in one picture: two humps with a
 * dip between them means the segments genuinely separate; a single hump piled up near the middle
 * means everybody is stranded between centres and there is nothing there.
 *
 * Drawn in ink rather than a segment colour, because it describes everyone. Pointing at a band
 * reports how many respondents it holds and what share of the study that is — turning "the shape
 * looks a bit bimodal" into a number.
 */
export function Gorge({ spec, title }: { spec: GorgeSpec; title: string }) {
  const [active, setActive] = useState<number | null>(null)
  const titleId = useId()

  const width = 720
  const height = 300
  const pad = { top: 24, right: 20, bottom: 54, left: 48 }

  const { x, y, tallest } = useMemo(() => {
    const tall = Math.max(...spec.counts, 1)
    return {
      x: scaleLinear()
        .domain([spec.edges[0], spec.edges[spec.edges.length - 1]])
        .range([pad.left, width - pad.right]),
      y: scaleLinear().domain([0, tall]).range([height - pad.bottom, pad.top]),
      tallest: tall,
    }
  }, [spec, pad.left, pad.right, pad.top, pad.bottom])

  const reading = active === null
    ? `${spec.people.toLocaleString()} respondents. Point at a band, or tab to one, to read it.`
    : (() => {
      const people = spec.counts[active]
      const share = spec.people ? people / spec.people : 0
      return `${people.toLocaleString()} ${people === 1 ? 'respondent' : 'respondents'} `
        + `(${(share * 100).toFixed(1)}%) score between ${spec.edges[active].toFixed(2)} and `
        + `${spec.edges[active + 1].toFixed(2)}`
    })()

  return (
    <div className="igorge">
      <svg className="chart" viewBox={`0 0 ${width} ${height}`}
           preserveAspectRatio="xMidYMid meet" role="img" aria-labelledby={titleId}>
        <title id={titleId}>
          {`${title}. How ${spec.people.toLocaleString()} respondents are spread between sitting `}
          {'firmly in one segment and being stranded between two.'}
        </title>

        {spec.counts.map((count, i) => {
          const left = x(spec.edges[i])
          const right = x(spec.edges[i + 1])
          const top = y(count)
          return (
            <rect
              key={i}
              x={left}
              y={top}
              width={Math.max(right - left - 0.5, 0.5)}
              height={Math.max(height - pad.bottom - top, count > 0 ? 1 : 0)}
              fill="currentColor"
              opacity={active === i ? 0.78 : 0.5}
              tabIndex={0}
              role="button"
              aria-label={`${count} respondents between ${spec.edges[i].toFixed(2)} and `
                + `${spec.edges[i + 1].toFixed(2)}`}
              onPointerOver={() => setActive(i)}
              onFocus={() => setActive(i)}
              onPointerLeave={() => setActive(null)}
              onBlur={() => setActive(null)}
            />
          )
        })}

        <line x1={x(spec.median)} x2={x(spec.median)} y1={pad.top} y2={height - pad.bottom}
              stroke="currentColor" strokeWidth={1.2} strokeDasharray="4 3" opacity={0.7} />
        <text x={x(spec.median)} y={pad.top - 8} textAnchor="middle" fontSize={11}
              fill="currentColor" opacity={0.8}>
          {`typical respondent ${spec.median.toFixed(2)}`}
        </text>

        <text x={pad.left} y={height - 16} fontSize={12} fill="currentColor">
          ← sits firmly in one segment
        </text>
        <text x={width - pad.right} y={height - 16} textAnchor="end" fontSize={12}
              fill="currentColor">
          stranded between two →
        </text>
        <text x={14} y={height / 2} fontSize={12} fill="currentColor" textAnchor="middle"
              transform={`rotate(-90 14 ${height / 2})`}>
          respondents
        </text>
        <text x={pad.left - 8} y={y(tallest) + 4} textAnchor="end" fontSize={11}
              fill="currentColor" opacity={0.7}>
          {tallest}
        </text>
      </svg>
      <div className="imap-read" role="status" aria-live="polite">{reading}</div>
    </div>
  )
}
