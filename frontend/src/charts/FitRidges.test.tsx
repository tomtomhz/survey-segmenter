import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { FitRidges } from './FitRidges'
import { KChoice } from './KChoice'
import { usableFit, usableKChoice, SUPPORTED_SPEC_VERSION } from './spec'
import type { FitSpec, KChoiceSpec } from './spec'

function fit(over: Partial<FitSpec> = {}): FitSpec {
  return {
    version: SUPPORTED_SPEC_VERSION,
    kind: 'fit',
    edges: [0, 0.25, 0.5, 0.75, 1],
    segments: [
      { index: 0, label: 'Loyal Fans', colour: '#2a78d6', colour_dark: '#3987e5', marker: 'o' },
      { index: 1, label: 'Price Hunters', colour: '#eb6834', colour_dark: '#d95926', marker: 's' },
    ],
    rows: [
      { segment: 0, counts: [2, 8, 30, 10], median: 0.62, people: 50 },
      { segment: 1, counts: [14, 9, 5, 2], median: 0.3, people: 30 },
    ],
    overall_median: 0.5,
    sampled: 0,
    ...over,
  }
}

function kchoice(over: Partial<KChoiceSpec> = {}): KChoiceSpec {
  return {
    version: SUPPORTED_SPEC_VERSION,
    kind: 'k_choice',
    ks: [2, 3, 4],
    chosen: 3,
    cutoff: 0.8,
    series: [
      { key: 'prediction_strength', label: 'Prediction strength', lead: true, values: [0.9, 0.95, 0.6] },
      { key: 'stability_ARI', label: 'Reproducibility (ARI)', lead: false, values: [0.8, 0.85, null] },
    ],
    ...over,
  }
}

describe('the fit ridges', () => {
  it('refuses bands that do not line up with their edges', () => {
    expect(usableFit(fit({ rows: [{ segment: 0, counts: [1, 2], median: 0.5, people: 3 }] })))
      .toBeNull()
    expect(usableFit(fit())).not.toBeNull()
  })

  it('reports how many people sit in a band, which is the number you can act on', async () => {
    // The static chart shows the shapes. The actionable number — how many people in THIS group
    // fit poorly enough to leave off a list — is what pointing at a band answers.
    const user = userEvent.setup()
    render(<FitRidges spec={fit()} title="Who belongs" />)
    const bands = document.querySelectorAll('rect')
    await user.hover(bands[2])
    const reading = screen.getByRole('status')
    expect(reading.textContent).toMatch(/Loyal Fans/)
    expect(reading.textContent).toMatch(/30 people fit/)
  })

  it('names each group beside its own distribution', () => {
    render(<FitRidges spec={fit()} title="Who belongs" />)
    expect(screen.getByText('Loyal Fans')).toBeInTheDocument()
    expect(screen.getByText('Price Hunters')).toBeInTheDocument()
  })

  it('every band is reachable by keyboard and describes itself', () => {
    render(<FitRidges spec={fit()} title="Who belongs" />)
    const bands = [...document.querySelectorAll('rect[tabindex="0"]')]
    // Two groups, four bands each.
    expect(bands.length).toBe(8)
    expect(bands[0].getAttribute('aria-label')).toMatch(/Loyal Fans, 2 people between/)
  })
})

describe('the choice of k', () => {
  it('refuses a series that does not cover every candidate', () => {
    expect(usableKChoice(kchoice({
      series: [{ key: 'a', label: 'A', lead: true, values: [0.5] }],
    }))).toBeNull()
    expect(usableKChoice(kchoice())).not.toBeNull()
  })

  it('reports every measure at once for a candidate', async () => {
    // The point of the chart is not "what is prediction strength" but "what does the evidence
    // look like at four groups rather than three" — which means reading all of them together.
    const user = userEvent.setup()
    render(<KChoice spec={kchoice()} title="How many groups" />)
    await user.hover(screen.getByRole('button', { name: '3 groups' }))
    const reading = screen.getByRole('status')
    expect(reading.textContent).toMatch(/3 groups \(chosen\)/)
    expect(reading.textContent).toMatch(/Prediction strength 0\.95/)
    expect(reading.textContent).toMatch(/Reproducibility \(ARI\) 0\.85/)
  })

  it('shows a missing measure as missing rather than as zero', async () => {
    const user = userEvent.setup()
    render(<KChoice spec={kchoice()} title="How many groups" />)
    await user.hover(screen.getByRole('button', { name: '4 groups' }))
    expect(screen.getByRole('status').textContent).toMatch(/Reproducibility \(ARI\) —/)
  })

  it('marks which criterion decides', () => {
    render(<KChoice spec={kchoice()} title="How many groups" />)
    expect(screen.getByText(/Prediction strength — decides/)).toBeInTheDocument()
  })
})
