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

  it('says on the row when a position is not settled', () => {
    render(
      <RankingCard
        ranking={[
          row({ rank: 1, item: 'Price', score: 0.6, low: 0.2, high: 1.0, prob_ahead: 0.58,
                clear_of_next: false }),
          row({ rank: 2, item: 'Speed', score: 0.55, low: 0.15, high: 0.95, prob_ahead: 1 }),
          row({ rank: 3, item: 'Brand', score: -1.1, low: -1.5, high: -0.7, prob_ahead: null,
                clear_of_next: null }),
        ]}
      />,
    )
    // Scoped to the badge itself. The containing cell matches the same text, so a loose query
    // would also pass if the phrase appeared only in prose and never on the row it describes.
    expect(document.querySelectorAll('.tie')).toHaveLength(1)
    expect(document.querySelector('.tie')?.textContent).toBe('58% sure of this order')
    expect(screen.getByText(/1 position is not settled/)).toBeTruthy()
    // The warning must be a warning, not a neutral aside the eye skips.
    expect(document.querySelector('.note.warn')).toBeTruthy()
  })

  it('distinguishes a coin flip from a finding that is merely short of certain', () => {
    // The reason the binary was replaced. Both pairs fall under 95% and the old card gave them
    // identical words — "too close to call" for a 58% pair and for a 93% one. A reader deciding
    // what to build needs those told apart: one is unknown, the other is probably true.
    render(
      <RankingCard
        ranking={[
          row({ rank: 1, item: 'Price', score: 0.6, low: 0.2, high: 1.0, prob_ahead: 0.58,
                clear_of_next: false }),
          row({ rank: 2, item: 'Speed', score: 0.5, low: 0.1, high: 0.9, prob_ahead: 0.93,
                clear_of_next: false }),
          row({ rank: 3, item: 'Brand', score: -1.1, low: -1.5, high: -0.7, prob_ahead: null,
                clear_of_next: null }),
        ]}
      />,
    )
    const badges = [...document.querySelectorAll('.tie')].map((n) => n.textContent)
    expect(badges).toEqual(['58% sure of this order', '93% sure of this order'])
    expect(screen.getByText(/2 positions are not settled/)).toBeTruthy()
  })

  it('does not announce a winner when nothing was separated', () => {
    // Sorting noise still produces a first row. Presenting it as "comes out strongest" above a
    // table that says every position is unsettled is the card contradicting itself.
    render(
      <RankingCard
        ranking={[
          row({ rank: 1, item: 'Price', score: 0.06, low: -0.3, high: 0.4, prob_ahead: 0.55,
                clear_of_next: false }),
          row({ rank: 2, item: 'Speed', score: 0.01, low: -0.35, high: 0.36, prob_ahead: 0.52,
                clear_of_next: false }),
          row({ rank: 3, item: 'Brand', score: -0.07, low: -0.42, high: 0.3, prob_ahead: null,
                clear_of_next: null }),
        ]}
      />,
    )
    expect(screen.getByText(/did not separate these items/)).toBeTruthy()
    expect(screen.getByText(/do not read this as a ranking/)).toBeTruthy()
    expect(screen.queryByText(/comes out strongest/)).toBeNull()
  })

  it('calls the top a pair when the leader is not clear of second place', () => {
    render(
      <RankingCard
        ranking={[
          row({ rank: 1, item: 'Price', score: 0.6, low: 0.3, high: 0.9, prob_ahead: 0.6,
                clear_of_next: false }),
          row({ rank: 2, item: 'Speed', score: 0.55, low: 0.25, high: 0.85, prob_ahead: 1 }),
          row({ rank: 3, item: 'Brand', score: -1.2, low: -1.5, high: -0.9, prob_ahead: null,
                clear_of_next: null }),
        ]}
      />,
    )
    expect(screen.getByText(/treat the top as a pair rather than a winner/)).toBeTruthy()
    expect(screen.queryByText(/did not separate these items/)).toBeNull()
    expect(screen.queryByText(/comes out strongest/)).toBeNull()
  })

  it('does not hedge a ranking the data fully supports', () => {
    // A caveat that fires on clean data teaches the reader to ignore the caveat.
    render(
      <RankingCard
        ranking={[
          row({ rank: 1, item: 'Price', score: 2.5, low: 2.3, high: 2.7, prob_ahead: 1 }),
          row({ rank: 2, item: 'Brand', score: -2.5, low: -2.7, high: -2.3, prob_ahead: null,
                clear_of_next: null }),
        ]}
      />,
    )
    expect(screen.queryByText(/not settled/)).toBeNull()
    expect(document.querySelector('.tie')).toBeNull()
    expect(screen.getByText(/beats the one below it with at least 95% certainty/)).toBeTruthy()
    expect(document.querySelector('.note.warn')).toBeNull()
  })

  it('survives rows that omit the interval keys entirely', () => {
    // The type says `low: number | null`, so TypeScript is satisfied — but JSON can leave the key
    // out, and `!== null` is true for `undefined`. That combination reached `row.low.toFixed(2)`
    // and threw "Cannot read properties of undefined". A saved project written by any build that
    // shapes the payload differently is enough to hit it.
    const rows = [
      { rank: 1, item: 'Price', score: 1.2, clear_of_next: null },
      { rank: 2, item: 'Brand', score: -1.2, clear_of_next: null },
    ] as unknown as RankedItem[]
    expect(() => render(<RankingCard ranking={rows} />)).not.toThrow()
    expect(bodyRows().map((r) => r.cells[1].textContent)).toEqual(['Price', 'Brand'])
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
