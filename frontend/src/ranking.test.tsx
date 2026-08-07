import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { RankingCard } from './components/RankingCard'
import type { RankedItem } from './api/types'

/**
 * A best-worst study is fielded to answer one question: which of these things do people want?
 *
 * The tool computed that ranking, handed the utilities to the segmenter, and then reported only
 * the groups — the answer itself reached the reader as prose inside a `<details>` panel that is
 * collapsed by default and sits below the charts. These tests hold the card that fixes it, and
 * in particular hold the part that is easy to quietly drop: saying when the order is not real.
 */
/** The data rows, typed: `getAllByRole` hands back plain HTMLElements, which have no `.cells`. */
const bodyRows = () =>
  (screen.getAllByRole('row') as HTMLTableRowElement[]).slice(1)

const row = (over: Partial<RankedItem> & { item: string; rank: number }): RankedItem => ({
  score: 0,
  low: null,
  high: null,
  clear_of_next: true,
  ...over,
})

describe('the overall preference ranking', () => {
  it('shows nothing at all for a survey that was never a best-worst exercise', () => {
    // An empty card headed "What matters most" above an ordinary segmentation would be a
    // question the study cannot answer, presented as though it had.
    const { container: none } = render(<RankingCard ranking={undefined} />)
    expect(none).toBeEmptyDOMElement()
    const { container: nulled } = render(<RankingCard ranking={null} />)
    expect(nulled).toBeEmptyDOMElement()
    const { container: empty } = render(<RankingCard ranking={[]} />)
    expect(empty).toBeEmptyDOMElement()
  })

  it('lists the items in order with their scores', () => {
    render(
      <RankingCard
        ranking={[
          row({ rank: 1, item: 'Price', score: 2.31, low: 2.1, high: 2.5 }),
          row({ rank: 2, item: 'Speed', score: 1.5, low: 1.3, high: 1.7 }),
          row({ rank: 3, item: 'Brand', score: -1.4, low: -1.6, high: -1.2, clear_of_next: null }),
        ]}
      />,
    )
    const rows = bodyRows()
    expect(rows.map((r) => r.cells[1].textContent)).toEqual(['Price', 'Speed', 'Brand'])
    // Signed and to two places: the scale is centred on zero, so the sign is the first thing read,
    // and a column that drops trailing zeros arrives ragged.
    expect(rows[0].cells[2].textContent).toBe('+2.31')
    expect(rows[1].cells[2].textContent).toBe('+1.50')
    expect(rows[2].cells[2].textContent).toBe('-1.40')
  })

  it('says on the row when two items are too close to separate', () => {
    render(
      <RankingCard
        ranking={[
          row({ rank: 1, item: 'Price', score: 0.6, low: 0.2, high: 1.0, clear_of_next: false }),
          row({ rank: 2, item: 'Speed', score: 0.55, low: 0.15, high: 0.95 }),
          row({ rank: 3, item: 'Brand', score: -1.1, low: -1.5, high: -0.7, clear_of_next: null }),
        ]}
      />,
    )
    // Scoped to the badge itself. The containing cell matches the same text, so a loose query
    // would also pass if the phrase appeared only in prose and never on the row it describes.
    expect(document.querySelectorAll('.tie')).toHaveLength(1)
    expect(document.querySelector('.tie')?.textContent).toBe('tied with next')
    expect(screen.getByText(/1 pair too close to separate/)).toBeTruthy()
    // The warning must be a warning, not a neutral aside the eye skips.
    expect(document.querySelector('.note.warn')).toBeTruthy()
  })

  it('does not hedge a ranking the data fully supports', () => {
    // A caveat that fires on clean data teaches the reader to ignore the caveat.
    render(
      <RankingCard
        ranking={[
          row({ rank: 1, item: 'Price', score: 2.5, low: 2.3, high: 2.7 }),
          row({ rank: 2, item: 'Brand', score: -2.5, low: -2.7, high: -2.3, clear_of_next: null }),
        ]}
      />,
    )
    expect(screen.queryByText(/too close to separate/)).toBeNull()
    expect(screen.queryByText(/tied with next/)).toBeNull()
    expect(screen.getByText(/clearly ahead of the one below it/)).toBeTruthy()
    expect(document.querySelector('.note.warn')).toBeNull()
  })

  it('survives an estimate that carries no intervals', () => {
    // Older saved projects were scored before the credible interval was kept. Reopening one must
    // show the ranking rather than crash on a null, and must not imply a certainty it lacks.
    render(
      <RankingCard
        ranking={[
          row({ rank: 1, item: 'Price', score: 1.2, clear_of_next: null }),
          row({ rank: 2, item: 'Brand', score: -1.2, clear_of_next: null }),
        ]}
      />,
    )
    expect(bodyRows().map((r) => r.cells[1].textContent)).toEqual(['Price', 'Brand'])
    expect(screen.queryByText(/How sure/)).toBeNull()
    expect(screen.queryByText(/clearly ahead of the one below it/)).toBeNull()
  })
})
