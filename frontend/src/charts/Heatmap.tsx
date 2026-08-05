import { useId, useMemo, useState } from 'react'

import type { HeatmapSpec } from './spec'
import { segColour, useDarkMode } from './theme'

/**
 * The full grid, drawn in the browser so a cell can be interrogated.
 *
 * The static version prints every value already — the grid is small enough that labelling every
 * cell is right rather than chaotic. What it cannot show is the second number behind each cell:
 * how far that group sits from the question's own average, which is what the COLOUR encodes.
 * Hovering, or moving through with the arrow keys, reports both.
 *
 * Built as a real HTML table rather than as SVG rects. A grid of numbers is a table, and saying so
 * in markup means a screen reader announces "Group 1, Price matters most, 4.7" by walking it, with
 * no extra work and no parallel "accessible version" to keep in sync. Colour is applied to the
 * cells; it is never the only carrier of the value.
 */
export function Heatmap({ spec, title }: { spec: HeatmapSpec; title: string }) {
  const [active, setActive] = useState<{ row: number; col: number } | null>(null)
  const captionId = useId()
  const dark = useDarkMode()

  const { colourFor } = useMemo(() => ({
    // Diverging, anchored on the largest deviation in either direction so the two arms are
    // comparable. Deliberately NOT the identity palette: on this chart colour means "above or
    // below average", and reusing a segment hue would make orange mean a group here and a
    // direction there.
    colourFor(deviation: number): string {
      const limit = spec.limit || 1
      const t = Math.max(-1, Math.min(1, deviation / limit))
      const strength = Math.abs(t) * 0.85
      return t >= 0
        ? `color-mix(in oklab, var(--heat-high, #e34948) ${strength * 100}%, transparent)`
        : `color-mix(in oklab, var(--heat-low, #2a78d6) ${strength * 100}%, transparent)`
    },
  }), [spec.limit])

  const reading = active
    ? `${spec.segments[active.row]?.label ?? ''} — ${spec.items[active.col]}: `
      + `${spec.kind_of_value} ${show(spec.values[active.row][active.col])}, `
      + describe(spec.deviation[active.row][active.col])
    : `${spec.segments.length} groups against ${spec.items.length} questions. `
      + 'Point at a cell, or move through with the arrow keys, to read it.'

  return (
    <div className="iheat">
      <div className="iheat-scroll">
        <table aria-describedby={captionId}>
          <caption id={captionId} className="visually-hidden">
            {`${title}. Every group's ${spec.kind_of_value} on every question, with how far it sits `}
            {'from that question\'s average.'}
          </caption>
          <thead>
            <tr>
              <th scope="col">Question</th>
              {spec.segments.map((s) => (
                <th key={s.index} scope="col">
                  <span className="iheat-chip" aria-hidden="true"
                        style={{ background: segColour(s, dark) }} />
                  {s.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {spec.items.map((item, col) => (
              <tr key={item}>
                <th scope="row">{item}</th>
                {spec.segments.map((s, row) => (
                  <td
                    key={s.index}
                    tabIndex={0}
                    style={{ background: colourFor(spec.deviation[row]?.[col] ?? 0) }}
                    className={active && active.row === row && active.col === col
                      ? 'iheat-on' : undefined}
                    onPointerOver={() => setActive({ row, col })}
                    onFocus={() => setActive({ row, col })}
                    onPointerLeave={() => setActive(null)}
                    onBlur={() => setActive(null)}
                  >
                    {show(spec.values[row]?.[col])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="imap-read" role="status" aria-live="polite">{reading}</div>
    </div>
  )
}

/** One decimal, matching the static chart exactly.
 *
 * The spec carries full precision because the readout below uses it, but the CELL has to read the
 * same in both renderings — printing 2.973 where the printed report says 3.0 is precisely the kind
 * of quiet disagreement the shared spec exists to prevent.
 */
function show(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return '—'
  return value.toFixed(1)
}

/** The sentence a colour is standing in for. */
function describe(deviation: number): string {
  const size = Math.abs(deviation)
  if (size < 0.05) return 'right on the average for this question'
  const direction = deviation > 0 ? 'above' : 'below'
  return `${size.toFixed(2)} ${direction} the average for this question`
}
