import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ScoreSummary } from './DownloadBar'
import { analysis } from '../test/fixtures'
import type { ScoreResult } from '../api/types'

const scored = (over: Partial<ScoreResult> = {}): ScoreResult => ({
  ok: true,
  n: 50,
  breakdown: { '0': 20, '1': 15, '2': 15 },
  mean_confidence: 0.58,
  confidence_floor: 0.33,
  file: 'scored_new_people.csv',
  ...over,
})

describe('what the app says after typing new people', () => {
  it('reports the confidence beside the floor it is measured from', () => {
    // The bare number is not comparable between studies: this scale starts at 1/k, so 0.58 with
    // three groups and 0.58 with two groups are different results, and nothing on screen said so.
    render(<ScoreSummary result={analysis()} scored={scored()} />)
    expect(screen.getByText(/0\.58/)).toBeInTheDocument()
    expect(screen.getByText(/0\.33 means no better than guessing/)).toBeInTheDocument()
  })

  it('shows the two-group floor when there are two groups', () => {
    render(<ScoreSummary result={analysis()}
                         scored={scored({ breakdown: { '0': 25, '1': 25 }, confidence_floor: 0.5 })} />)
    expect(screen.getByText(/0\.5 means no better than guessing/)).toBeInTheDocument()
  })

  it('still renders against an older server that sends no floor', () => {
    const { confidence_floor: _omitted, ...withoutFloor } = scored()
    render(<ScoreSummary result={analysis()} scored={withoutFloor as ScoreResult} />)
    expect(screen.getByText(/50 people scored/)).toBeInTheDocument()
    expect(screen.queryByText(/no better than guessing/)).not.toBeInTheDocument()
  })
})
