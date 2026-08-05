import { useId, useMemo, useRef, useState } from 'react'
import { Delaunay } from 'd3-delaunay'
import { scaleLinear } from 'd3-scale'

import type { SegmentMapSpec } from './spec'
import { markerPath } from './marks'

/**
 * The segment map, drawn in the browser so it can be interrogated.
 *
 * The static matplotlib version answers "do these groups separate". This one answers the question
 * a reader asks next and could not ask before: *what is that dot?* Hovering a mark reports which
 * group it belongs to, how many people share that exact set of answers, and where it sits.
 *
 * Three things here are deliberate rather than incidental:
 *
 * **Nearest-point hit-testing, not per-mark mouse handlers.** Dots are as small as a few pixels
 * and vary in size, so requiring a direct hit means the reader hunts for a target. A Delaunay
 * triangulation of the points answers "which mark is nearest the cursor" in log time, so the whole
 * plot is live and the nearest dot lights up. Attaching a listener to each of several thousand
 * marks would also be several thousand listeners.
 *
 * **Keyboard reaches everything the mouse does.** Arrow keys walk the marks in order, so the chart
 * is usable without a pointer and readable by a screen reader, which a hover-only chart is not.
 *
 * **The numbers are also a table.** Colour and position are not available to everyone, and a chart
 * that cannot be read any other way is a chart some people cannot read at all. The table carries
 * the same rows the marks do.
 */
export function SegmentMap({ spec, title }: { spec: SegmentMapSpec; title: string }) {
  const [active, setActive] = useState<number | null>(null)
  const [showTable, setShowTable] = useState(false)
  const svgRef = useRef<SVGSVGElement | null>(null)
  const titleId = useId()

  // A viewBox rather than pixel sizes: the card is fluid and the chart should scale with it, the
  // same way the static SVG does.
  const width = 720
  const height = 400
  const pad = { top: 18, right: 18, bottom: 52, left: 34 }

  const { xs, ys, sizes, byIndex, xScale, yScale, delaunay, regions } = useMemo(() => {
    const { x, y, people } = spec.points
    const xDomain = extent(x)
    const yDomain = extent(y)
    const xScale = scaleLinear().domain(xDomain).range([pad.left, width - pad.right]).nice()
    const yScale = scaleLinear().domain(yDomain).range([height - pad.bottom, pad.top]).nice()
    const busiest = Math.max(spec.busiest_spot, 1)
    // Area proportional to how many people share the spot, with a floor so a single respondent
    // stays visible beside a stack of hundreds. Matches the static chart's encoding.
    const areaFor = (n: number) => 16 + (460 - 16) * (busiest > 1 ? (n - 1) / (busiest - 1) : 0)
    const px = x.map((v) => xScale(v))
    const py = y.map((v) => yScale(v))
    const byIndex: Record<number, SegmentMapSpec['segments'][number]> = {}
    for (const key of spec.segments) byIndex[key.index] = key
    return {
      xs: px,
      ys: py,
      sizes: people.map(areaFor),
      byIndex,
      // The scales themselves, so anything else that needs to place a value — the centroids —
      // uses the SAME mapping the points did. An earlier version re-derived centroid positions by
      // interpolating between the extreme points; it happened to agree, but only because the
      // scale is linear, and it would have drifted silently the day anyone changed the scale.
      xScale,
      yScale,
      delaunay: Delaunay.from(px.map((v, i) => [v, py[i]] as [number, number])),
      // The decision regions: which group a new person landing at a spot would be put into. That
      // is nearest-centroid, which is exactly a Voronoi diagram of the centres — so it is computed
      // rather than approximated, from the same centroids the badges use. The caption promises
      // these regions, and for a while the interactive chart quietly did not have them.
      regions: (() => {
        const live = spec.centroids
        if (live.length < 2) return []
        const cells = Delaunay
          .from(live.map((c) => [xScale(c.x), yScale(c.y)] as [number, number]))
          .voronoi([0, 0, width, height])
        return live.map((c, i) => ({ segment: c.segment, path: cells.renderCell(i) }))
      })(),
    }
  }, [spec, pad.left, pad.right, pad.top, pad.bottom])

  const total = spec.points.x.length

  /**
   * The marks are computed once and do NOT depend on which one is under the cursor.
   *
   * The first version dimmed every other mark on hover, which meant each pointer move re-rendered
   * all of them: at the cap of 6,000 that is 6,000 elements rebuilt per mouse move, and even at a
   * few hundred it made the whole plot flicker as the cursor crossed it. Only the ring and the
   * reading line react now, so pointing at a dot costs two small updates instead of the chart.
   */
  const marks = useMemo(() => spec.points.x.map((_, i) => {
    const key = byIndex[spec.points.segment[i]]
    if (!key) return null
    return (
      <path
        key={i}
        className="imap-mark"
        d={markerPath(key.marker, sizes[i])}
        transform={`translate(${xs[i]} ${ys[i]})`}
        fill={`var(--seg-${key.index}, ${key.colour})`}
        opacity={0.72}
      />
    )
  }), [spec, byIndex, sizes, xs, ys])

  function pointerMove(event: React.PointerEvent<SVGSVGElement>) {
    const svg = svgRef.current
    if (!svg) return
    const box = svg.getBoundingClientRect()
    // Client pixels back into viewBox units, or the nearest-point search is done in the wrong
    // coordinate system and picks the wrong dot on any screen where the chart is not 720 wide.
    const vx = ((event.clientX - box.left) / box.width) * width
    const vy = ((event.clientY - box.top) / box.height) * height
    setActive(delaunay.find(vx, vy))
  }

  function keyDown(event: React.KeyboardEvent<SVGSVGElement>) {
    const step = event.key === 'ArrowRight' || event.key === 'ArrowDown' ? 1
      : event.key === 'ArrowLeft' || event.key === 'ArrowUp' ? -1 : 0
    if (step === 0) {
      if (event.key === 'Escape') setActive(null)
      return
    }
    event.preventDefault()
    setActive((was) => {
      const next = was === null ? 0 : (was + step + total) % total
      return next
    })
  }

  const hovered = active === null ? null : {
    segment: byIndex[spec.points.segment[active]],
    people: spec.points.people[active],
    x: xs[active],
    y: ys[active],
  }

  return (
    <div className="imap">
      <svg
        ref={svgRef}
        className="chart"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-labelledby={titleId}
        tabIndex={0}
        onPointerMove={pointerMove}
        onPointerLeave={() => setActive(null)}
        onKeyDown={keyDown}
        onBlur={() => setActive(null)}
      >
        <title id={titleId}>
          {`${title}. ${spec.people.toLocaleString()} respondents at ${total.toLocaleString()} `}
          {'distinct answer patterns. Use the arrow keys to step through them.'}
        </title>

        {/* Behind everything, and faint: the regions exist to expose the pie-slice failure, not
            to identify anybody — that is what the marks are for. */}
        {regions.map((r) => {
          const key = byIndex[r.segment]
          return key && r.path ? (
            <path key={`region-${r.segment}`} d={r.path}
                  fill={`var(--seg-${key.index}, ${key.colour})`}
                  opacity={Math.max(0.05, 0.13 - 0.012 * spec.segments.length)} />
          ) : null
        })}

        {marks}

        {spec.centroids.map((c) => {
          const key = byIndex[c.segment]
          if (!key) return null
          const cx = xScale(c.x)
          const cy = yScale(c.y)
          return (
            <g key={c.segment} transform={`translate(${cx} ${cy})`}>
              <circle r={11} fill="var(--chart-surface, #FBFAF3)"
                      stroke={`var(--seg-${key.index}, ${key.colour})`} strokeWidth={2.6} />
              <text textAnchor="middle" dominantBaseline="central" fontSize={11}
                    fontWeight="700" fill={`var(--seg-${key.index}, ${key.colour})`}>
                {c.segment}
              </text>
            </g>
          )
        })}

        {hovered && (
          <g pointerEvents="none" transform={`translate(${hovered.x} ${hovered.y})`}>
            <circle r={Math.sqrt(sizes[active as number]) / 2 + 5} fill="none"
                    stroke="currentColor" strokeWidth={1.5} opacity={0.85} />
          </g>
        )}

        <text x={width / 2} y={height - 14} textAnchor="middle" fontSize={12} fill="currentColor">
          {`Direction 1 — ${pct(spec.axes.x_share)} of the variation →`}
        </text>
        {/* The vertical axis carries its own share too. Reading whether a separation means
            anything needs to know what the direction it happens along is worth. */}
        <text x={12} y={height / 2} fontSize={12} fill="currentColor" textAnchor="middle"
              transform={`rotate(-90 12 ${height / 2})`}>
          {`← Direction 2 — ${pct(spec.axes.y_share)}`}
        </text>
      </svg>

      {/* A legend, because the static chart has one and losing it made the interactive version
          worse at a glance: without it the only way to learn which colour is which group is to
          hover every one of them. Each entry shows the segment's real marker shape, since shape
          is half of the identity — colour alone cannot separate eight groups. */}
      <ul className="imap-key">
        {spec.segments.map((key) => (
          <li key={key.index}>
            <svg viewBox="-9 -9 18 18" aria-hidden="true" focusable="false">
              <path d={markerPath(key.marker, 90)} fill={`var(--seg-${key.index}, ${key.colour})`} />
            </svg>
            {key.label}
          </li>
        ))}
      </ul>

      <div className="imap-read" role="status" aria-live="polite">
        {hovered ? (
          <>
            <span className="imap-swatch" aria-hidden="true"
                  style={{ background: `var(--seg-${hovered.segment.index}, ${hovered.segment.colour})` }} />
            <strong>{hovered.segment.label}</strong>
            {` — ${hovered.people.toLocaleString()} ${hovered.people === 1 ? 'person' : 'people'} `}
            {'answered exactly like this'}
          </>
        ) : (
          `${spec.people.toLocaleString()} respondents at ${total.toLocaleString()} distinct answer patterns. Point at a mark, or press the arrow keys, to read one.`
        )}
      </div>

      <button type="button" className="imap-toggle" onClick={() => setShowTable((v) => !v)}>
        {showTable ? 'Hide the numbers' : 'Show the numbers'}
      </button>

      {showTable && (
        <div className="imap-table">
          <table>
            <caption>
              {`Every mark on the map. ${spec.people.toLocaleString()} respondents at `}
              {`${total.toLocaleString()} distinct answer patterns.`}
            </caption>
            <thead>
              <tr><th scope="col">Group</th><th scope="col">People</th>
                  <th scope="col">Direction 1</th><th scope="col">Direction 2</th></tr>
            </thead>
            <tbody>
              {spec.points.x.map((x, i) => (
                <tr key={i}>
                  <td>{byIndex[spec.points.segment[i]]?.label ?? spec.points.segment[i]}</td>
                  <td>{spec.points.people[i].toLocaleString()}</td>
                  <td>{x.toFixed(2)}</td>
                  <td>{spec.points.y[i].toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function extent(values: number[]): [number, number] {
  let lo = Infinity
  let hi = -Infinity
  for (const v of values) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  // A single distinct value has no extent to scale; widen it so the mark lands mid-axis rather
  // than dividing by zero.
  return lo === hi ? [lo - 0.5, hi + 0.5] : [lo, hi]
}

function pct(share: number): string {
  return `${Math.round(share * 100)}%`
}
