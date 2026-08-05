import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { Heatmap } from './Heatmap'
import { usableHeatmap, SUPPORTED_SPEC_VERSION } from './spec'
import type { HeatmapSpec } from './spec'

function grid(over: Partial<HeatmapSpec> = {}): HeatmapSpec {
  return {
    version: SUPPORTED_SPEC_VERSION,
    kind: 'heatmap',
    items: ['Price matters most', 'I buy on impulse'],
    segments: [
      { index: 0, label: 'Loyal Fans', colour: '#2a78d6', colour_dark: '#3987e5', marker: 'o' },
      { index: 1, label: 'Price Hunters', colour: '#eb6834', colour_dark: '#d95926', marker: 's' },
    ],
    values: [[3.1, 4.7], [4.7, 1.3]],
    deviation: [[-0.8, 1.7], [0.8, -1.7]],
    limit: 1.7,
    kind_of_value: 'average',
    ...over,
  }
}

describe('the grid gate', () => {
  it('refuses a spec whose rows do not line up with its columns', () => {
    // A short row would render as a blank cell rather than as the number it is.
    expect(usableHeatmap(grid({ values: [[1], [2, 3]] }))).toBeNull()
    expect(usableHeatmap(grid({ deviation: [[1, 2]] }))).toBeNull()
    expect(usableHeatmap(grid({ version: SUPPORTED_SPEC_VERSION + 1 }))).toBeNull()
    expect(usableHeatmap(grid())).not.toBeNull()
  })

  it('does not accept a segment map', () => {
    expect(usableHeatmap({ version: SUPPORTED_SPEC_VERSION, kind: 'segment_map' })).toBeNull()
  })
})

describe('the interactive grid', () => {
  it('is a real table, so it can be read by walking it', () => {
    // A grid of numbers IS a table. Saying so in markup means a screen reader announces group,
    // question and value with no parallel "accessible version" to keep in sync.
    render(<Heatmap spec={grid()} title="Full grid" />)
    const table = screen.getByRole('table')
    expect(table.querySelectorAll('tbody tr').length).toBe(2)
    expect(screen.getByRole('columnheader', { name: /Loyal Fans/ })).toBeInTheDocument()
    expect(screen.getByRole('rowheader', { name: 'Price matters most' })).toBeInTheDocument()
  })

  it('shows the value in every cell, never colour alone', () => {
    render(<Heatmap spec={grid()} title="Full grid" />)
    for (const value of ['3.1', '4.7', '1.3']) {
      expect(screen.getAllByText(value).length).toBeGreaterThan(0)
    }
  })

  it('prints cells to the same precision as the static chart', () => {
    // The spec carries full precision for the readout, but a cell reading 2.973 where the printed
    // report says 3.0 is the quiet disagreement the shared spec exists to prevent.
    render(<Heatmap spec={grid({ values: [[2.973, 4.7], [4.7, 1.3]] })} title="Full grid" />)
    expect(screen.getByText('3.0')).toBeInTheDocument()
    expect(screen.queryByText('2.973')).toBeNull()
  })

  it('reports the second number colour is standing in for', async () => {
    // The static chart prints the value. What it cannot show is how far the cell sits from that
    // question's own average, which is exactly what the colour encodes.
    const user = userEvent.setup()
    render(<Heatmap spec={grid()} title="Full grid" />)
    // 1.3 appears once, so this targets a known cell; 4.7 appears twice and an earlier version of
    // this test asserted against whichever came first in the DOM, which was not the one it meant.
    await user.hover(screen.getByText('1.3'))
    const reading = screen.getByRole('status')
    expect(reading.textContent).toMatch(/below the average for this question/)
    expect(reading.textContent).toMatch(/1\.70/)
    expect(reading.textContent).toMatch(/Price Hunters/)
  })

  it('reaches every cell by keyboard', async () => {
    const user = userEvent.setup()
    render(<Heatmap spec={grid()} title="Full grid" />)
    await user.tab()
    expect(screen.getByRole('status').textContent).toMatch(/Loyal Fans|Price Hunters/)
  })

  it('says so plainly when a cell sits on the average', async () => {
    // Through user-event rather than a bare .focus(): React only sees the event the browser would
    // actually deliver, and a direct DOM call left the readout untouched.
    const user = userEvent.setup()
    render(<Heatmap spec={grid({ deviation: [[0, 0], [0, 0]] })} title="Full grid" />)
    await user.hover(screen.getByText('3.1'))
    expect(screen.getByRole('status').textContent).toMatch(/right on the average/)
  })
})
