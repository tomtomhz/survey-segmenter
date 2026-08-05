import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { scaleLinear } from 'd3-scale'

import { SegmentMap } from './SegmentMap'
import { usableSpec, SUPPORTED_SPEC_VERSION } from './spec'
import { markerPath } from './marks'
import type { SegmentMapSpec } from './spec'

function spec(overrides: Partial<SegmentMapSpec> = {}): SegmentMapSpec {
  return {
    version: SUPPORTED_SPEC_VERSION,
    kind: 'segment_map',
    points: {
      x: [0, 1, 2, 3],
      y: [0, 1, 0.5, 2],
      segment: [0, 1, 0, 1],
      people: [1, 40, 7, 3],
    },
    centroids: [
      { segment: 0, x: 1, y: 0.25 },
      { segment: 1, x: 2, y: 1.5 },
    ],
    segments: [
      { index: 0, label: 'Loyal Fans', colour: '#2a78d6', colour_dark: '#3987e5', marker: 'o' },
      { index: 1, label: 'Price Hunters', colour: '#eb6834', colour_dark: '#d95926', marker: 's' },
    ],
    axes: { x_share: 0.68, y_share: 0.21, kept: 0.89 },
    people: 51,
    busiest_spot: 40,
    faithful: 1,
    ...overrides,
  }
}

describe('the spec gate', () => {
  it('refuses anything it cannot be sure it understands', () => {
    // A saved project from a different build must fall back to the static drawing rather than be
    // read as though its numbers meant what this build thinks they mean.
    expect(usableSpec(undefined)).toBeNull()
    expect(usableSpec({})).toBeNull()
    expect(usableSpec(spec({ version: SUPPORTED_SPEC_VERSION + 1 }))).toBeNull()
    expect(usableSpec({ ...spec(), kind: 'something_else' })).toBeNull()
    expect(usableSpec(spec())).not.toBeNull()
  })

  it('refuses parallel arrays of different lengths', () => {
    // The arrays are parallel, so a short one would silently drop respondents off the chart —
    // the single thing this chart exists not to do.
    const short = spec()
    short.points.people = [1, 2]
    expect(usableSpec(short)).toBeNull()
  })
})

describe('marker shapes', () => {
  it('gives every segment its own shape', () => {
    // Identity is colour AND shape: on a scatter every pair of colours is on screen at once, and
    // the worst pair measures CVD ΔE 3.2, which colour alone cannot carry.
    const shapes = ['o', 's', '^', 'D', 'v', 'P', 'X', '*'].map((m) => markerPath(m, 100))
    expect(new Set(shapes).size).toBe(shapes.length)
    for (const path of shapes) expect(path.startsWith('M')).toBe(true)
  })

  it('scales by area, so a stack of many reads bigger than a single respondent', () => {
    const small = markerPath('s', 16)
    const large = markerPath('s', 400)
    const extent = (d: string) => Math.abs(Number(d.match(/M(-?[\d.]+)/)![1]))
    expect(extent(large)).toBeGreaterThan(extent(small) * 2)
  })
})

describe('the interactive segment map', () => {
  it('draws every mark and names the study without anyone touching it', () => {
    render(<SegmentMap spec={spec()} title="The segment map" />)
    // One mark per distinct answer pattern. Targeted by class rather than by element type: the
    // chart also draws decision regions and centroid badges as paths, and counting all of them
    // would make this assertion about the chart's internals instead of about the data.
    expect(document.querySelectorAll('svg.chart .imap-mark').length).toBe(4)
    // The reading line states the whole study before any interaction, so the chart is informative
    // to somebody who never hovers.
    expect(screen.getByRole('status')).toHaveTextContent('51 respondents')
    expect(screen.getByRole('status')).toHaveTextContent('4 distinct answer patterns')
  })

  it('is reachable by keyboard, not only by pointer', async () => {
    // A hover-only chart is one a keyboard user cannot read at all.
    const user = userEvent.setup()
    render(<SegmentMap spec={spec()} title="The segment map" />)
    const plot = document.querySelector('svg.chart') as SVGSVGElement
    plot.focus()
    await user.keyboard('{ArrowRight}')
    const reading = screen.getByRole('status')
    expect(reading.textContent).toMatch(/Loyal Fans|Price Hunters/)
    expect(reading.textContent).toMatch(/person|people/)
  })

  it('reports how many people share the mark, not just which group', async () => {
    const user = userEvent.setup()
    render(<SegmentMap spec={spec()} title="The segment map" />)
    const plot = document.querySelector('svg.chart') as SVGSVGElement
    plot.focus()
    // Step to the second mark, which 40 people share — the number the static chart can only
    // encode as area.
    await user.keyboard('{ArrowRight}{ArrowRight}')
    expect(screen.getByRole('status')).toHaveTextContent('40 people')
  })

  it('offers the same numbers as a table', async () => {
    // Colour and position are not available to every reader, and a chart with no other way in is
    // a chart some people cannot read.
    const user = userEvent.setup()
    render(<SegmentMap spec={spec()} title="The segment map" />)
    await user.click(screen.getByRole('button', { name: /show the numbers/i }))
    const table = screen.getByRole('table')
    expect(table).toHaveTextContent('Loyal Fans')
    // One row per mark, plus the header row.
    expect(table.querySelectorAll('tbody tr').length).toBe(4)
  })

  it('uses the theme variables so it follows light and dark like the static chart', () => {
    render(<SegmentMap spec={spec()} title="The segment map" />)
    const mark = document.querySelector('svg.chart .imap-mark') as SVGPathElement
    expect(mark.getAttribute('fill')).toContain('var(--seg-0')
    // The light hex stays as the fallback, so the chart is coloured even with no stylesheet.
    expect(mark.getAttribute('fill')).toContain('#2a78d6')
  })

  it('survives a single distinct answer pattern without dividing by zero', () => {
    const flat = spec({
      points: { x: [1, 1], y: [2, 2], segment: [0, 0], people: [5, 5] },
      centroids: [{ segment: 0, x: 1, y: 2 }],
      busiest_spot: 5,
      people: 10,
    })
    render(<SegmentMap spec={flat} title="The segment map" />)
    for (const mark of document.querySelectorAll('svg.chart .imap-mark')) {
      expect(mark.getAttribute('transform')).not.toMatch(/NaN/)
    }
  })
})

describe('parity with the static chart', () => {
  it('places centroids with the same scale the points used', () => {
    // An earlier version re-derived centroid positions by interpolating between the extreme
    // points. It agreed with the real scale only because the scale is linear, and would have
    // drifted silently the day anybody changed it.
    const map = spec()
    render(<SegmentMap spec={map} title="t" />)
    const badge = document.querySelectorAll('svg.chart g')[0] as SVGGElement
    const got = badge.getAttribute('transform')!.match(/translate\(([-\d.]+) ([-\d.]+)\)/)!
    // Rebuild the scale the component says it uses and compare.
    const x = scaleLinear().domain([0, 3]).range([34, 720 - 18]).nice()
    const y = scaleLinear().domain([0, 2]).range([400 - 52, 18]).nice()
    expect(Number(got[1])).toBeCloseTo(x(1), 0)
    expect(Number(got[2])).toBeCloseTo(y(0.25), 0)
  })

  it('does not crash when a point names a segment with no key', () => {
    // Defensive: the spec is produced by this project's own Python, but a component that throws
    // takes the whole report card down with it, and a missing key is a survivable condition.
    const map = spec()
    map.points.segment = [0, 1, 9, 1]      // 9 has no entry in segments
    expect(() => render(<SegmentMap spec={map} title="t" />)).not.toThrow()
  })

  it('shows which colour is which group without needing to hover', () => {
    // The static chart has a legend. The first interactive version did not, which made it worse
    // at a glance than the picture it replaced: the only way to learn which colour meant which
    // group was to hover every one of them.
    render(<SegmentMap spec={spec()} title="t" />)
    // The static chart has a legend. Without one, a glance cannot tell the groups apart.
    const key = document.querySelector('.imap-key')
    expect(key).not.toBeNull()
    expect(key!.textContent).toContain('Loyal Fans')
    expect(key!.textContent).toContain('Price Hunters')
    // Shape is half the identity, so the legend has to show it, not just a colour chip.
    expect(key!.querySelectorAll('svg path').length).toBe(2)
  })

  it('labels both axes, as the static chart does', () => {
    // Judging whether a separation means anything needs to know what the direction it happens
    // along is worth, so both shares belong on the chart, not only the horizontal one.
    const { container } = render(<SegmentMap spec={spec()} title="t" />)
    const text = container.textContent ?? ''
    expect(text).toMatch(/Direction 1/)
    expect(text).toMatch(/Direction 2/)
  })
})

describe('the decision regions', () => {
  it('draws the regions the caption promises', () => {
    // The shared caption tells the reader that the shaded regions show where a new person would
    // land. For a while the interactive chart quietly had none, so the caption described
    // something that was not on screen.
    render(<SegmentMap spec={spec()} title="t" />)
    const regions = document.querySelectorAll('svg.chart path:not(.imap-mark)')
    expect(regions.length).toBeGreaterThanOrEqual(2)
  })

  it('draws none when there is only one centre to divide between', () => {
    // One group has no boundary with anything, and a single region covering the whole plot would
    // imply a decision that was never made.
    const alone = spec({ centroids: [{ segment: 0, x: 1, y: 1 }] })
    render(<SegmentMap spec={alone} title="t" />)
    // Only the centroid badge remains as a non-mark path-bearing element.
    const shaded = [...document.querySelectorAll('svg.chart path:not(.imap-mark)')]
      .filter((p) => (p.getAttribute('d') ?? '').length > 40)
    expect(shaded.length).toBe(0)
  })
})
